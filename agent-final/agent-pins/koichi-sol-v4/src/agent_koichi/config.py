"""Explicit workspace-local configuration; credentials are never metadata."""

import os
import re
import sys
from dataclasses import dataclass, field, fields
from importlib.metadata import version as dependency_version
from pathlib import Path

from dotenv import dotenv_values

WORKSPACE = Path(__file__).resolve().parents[2]
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    model: str
    workspace: Path = WORKSPACE
    strategy: str = "hard"
    fast_seconds: float = 12.0
    deep_seconds: float = 35.0
    extraction_seconds: float = 15.0  # Foreground wait ceiling, not request lifetime.
    max_tokens: int = 4096
    max_chats: int = 3
    baseline_version: str | None = None
    model_key: str = field(default="", repr=False)
    extraction_budget_policy: str = "policy_reserve"
    planner_enabled: bool = False
    provider_config: dict | None = field(default=None, repr=False)
    output_limit_policy: str | None = None
    use_provider_default_effort: bool = False

    def __post_init__(self) -> None:
        if type(self.use_provider_default_effort) is not bool:
            raise ConfigurationError("use_provider_default_effort must be a boolean.")
        if (
            self.use_provider_default_effort
            and (self.provider_config or {}).get("effort") is not None
        ):
            raise ConfigurationError(
                "Provider-default effort conflicts with explicit effort."
            )
        if self.provider_config is not None:
            if not isinstance(self.provider_config, dict):
                raise ConfigurationError(
                    "provider_config must be an explicit dictionary."
                )
            if self.provider_config.get("model") != self.model:
                raise ConfigurationError(
                    "Declared model must match provider_config.model."
                )
            route = self.provider_config.get("route")
            if route not in ("chatgpt", "openrouter"):
                raise ConfigurationError("Select an explicit provider route.")
            if self.provider_config.get("focal", True) and (
                route != "chatgpt" or self.model != "gpt-6.1-sol"
            ):
                raise ConfigurationError(
                    "Focal inference requires ChatGPT GPT-6.1 Sol."
                )
            if route == "chatgpt" and self.output_limit_policy != "omit_unsupported":
                raise ConfigurationError(
                    "ChatGPT requires output_limit_policy=omit_unsupported."
                )
            if self.output_limit_policy not in (None, "omit_unsupported", "enforce"):
                raise ConfigurationError("Unknown output-limit policy.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigurationError(
                "Set a nonempty model via AGENT_MODEL or an explicit model override."
            )
        if self.extraction_budget_policy not in ("fraction", "policy_reserve"):
            raise ConfigurationError(
                "Extraction budget policy must be fraction or policy_reserve."
            )
        if self.strategy not in ("hard", "energy", "csp", "assertion", "llm"):
            raise ConfigurationError(
                "Strategy must be hard, energy, csp, assertion, or llm."
            )
        if type(self.planner_enabled) is not bool:
            raise ConfigurationError("planner_enabled must be a boolean.")
        if self.planner_enabled and self.strategy == "llm":
            raise ConfigurationError("Planner requires a mathematical belief strategy.")

    @classmethod
    def from_env(cls, **overrides) -> Config:
        if (
            "use_provider_default_effort" in overrides
            and type(overrides["use_provider_default_effort"]) is not bool
        ):
            raise ConfigurationError("use_provider_default_effort must be a boolean.")
        if set(overrides) - {entry.name for entry in fields(cls)}:
            raise ConfigurationError("Configuration contains unsupported fields.")
        workspace = Path(overrides.get("workspace", WORKSPACE))
        # Explicit local lookup, without importing platform credentials into the environment.
        environment = {
            **dotenv_values(workspace / ".env", interpolate=False),
            **os.environ,
        }
        values = {
            "workspace": workspace,
            "model": environment.get("AGENT_MODEL"),
            "strategy": environment.get("AGENT_STRATEGY") or "hard",
            "model_key": environment.get("OPENROUTER_API_KEY") or "",
        }
        if overrides.get("planner_enabled") is None:
            planner = environment.get("AGENT_PLANNER", "false")
            if not isinstance(planner, str) or planner.lower() not in (
                "true",
                "false",
                "1",
                "0",
            ):
                raise ConfigurationError("AGENT_PLANNER must be true, false, 1, or 0.")
            values["planner_enabled"] = planner.lower() in ("true", "1")
        values.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        config = cls(**values)
        if (
            any(
                type(value) not in (int, float)
                for value in (
                    config.fast_seconds,
                    config.deep_seconds,
                    config.extraction_seconds,
                )
            )
            or not 0 < config.fast_seconds <= config.deep_seconds <= 60
            or not 0 < config.extraction_seconds <= 60
        ):
            raise ConfigurationError(
                "Decision time limits must be between 0 and 60 seconds."
            )
        if type(config.max_tokens) is not int or not 1 <= config.max_tokens <= 16384:
            raise ConfigurationError("Invalid per-request model token limit.")
        if type(config.max_chats) is not int or not 0 <= config.max_chats <= 5:
            raise ConfigurationError("Chat limit must be between 0 and 5.")
        if config.baseline_version is not None and not re.fullmatch(
            r"[0-9a-f]{40}", config.baseline_version
        ):
            raise ConfigurationError("Baseline version must be a full Git commit SHA.")
        return config

    def require_model(self) -> None:
        if self.provider_config is None:
            raise ConfigurationError("An explicit provider_config is required.")

    def metadata(self) -> dict:
        effort = (self.provider_config or {}).get("effort")
        if self.use_provider_default_effort:
            effort = "provider_default"
        return {
            "provider": (self.provider_config or {}).get("route"),
            "endpoint": (
                "https://api.openai.com/v1/responses"
                if (self.provider_config or {}).get("route") == "chatgpt"
                else OPENROUTER_ENDPOINT
            ),
            "model": self.model,
            "strategy": self.strategy,
            "planner_enabled": self.planner_enabled,
            "planner_model": "shallow-uniform-v1" if self.planner_enabled else None,
            "use_provider_default_effort": self.use_provider_default_effort,
            "fast_effort": "low" if effort is None else effort,
            "deep_effort": "high" if effort is None else effort,
            "effort_policy": "provider default for all stages"
            if self.use_provider_default_effort
            else "explicit provider effort overrides all stages; otherwise original low/high",
            "fast_seconds": self.fast_seconds,
            "deep_seconds": self.deep_seconds,
            "extraction_seconds": self.extraction_seconds,
            "extraction_seconds_scope": "foreground_wait_ceiling",
            "extraction_request_deadline": "original_decision_minus_guard",
            "extraction_budget_policy": self.extraction_budget_policy,
            "extraction_execution": "extraction_first_all_pending",
            "extraction_batch_limit": None,
            "extraction_batch_scope": "all_pending",
            "model_concurrency": "unrestricted_local",
            "requested_max_tokens": self.max_tokens,
            "enforced_max_tokens": (
                self.max_tokens
                if (self.provider_config or {}).get("route") == "openrouter"
                else None
            ),
            "output_limit_policy": self.output_limit_policy,
            "max_chats_per_day": self.max_chats,
            "baseline_version": self.baseline_version,
            "automatic_provider_fallback": False,
            "prompts": "src/agent_koichi/prompts.py",
            "policy_response_format": "json_schema",
            "policy_response_format_strict": True,
            "policy_schema": "src/agent_koichi/schemas.py",
            "extraction_response_format": "JSON requested in prompt; locally validated",
            "random_seed": None,
            "sampling_parameters": {
                "temperature": "provider default",
                "stream": (self.provider_config or {}).get("route") == "chatgpt",
                "response_format": (
                    "policy: strict json_schema; "
                    "extraction: JSON requested in prompt; locally validated"
                ),
            },
        }

    def runtime_metadata(self) -> dict:
        return {
            **self.metadata(),
            "python": sys.version.split()[0],
            "dependencies": {
                name: dependency_version(name)
                for name in (
                    "agent-koichi",
                    "final-providers",
                    "werewolf-runtime",
                    "httpx",
                    "python-dotenv",
                )
            },
        }
