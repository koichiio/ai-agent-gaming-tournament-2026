"""Workspace-local credentials, running versions, and generic adapter configuration."""

import json
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import dotenv_values


@dataclass(frozen=True)
class AgentSpec:
    name: str
    workspace: Path
    agent: str
    command: tuple[str, ...]
    strategy: str
    config: dict
    experiment: dict


class Lineup(Protocol):
    agents: dict[str, AgentSpec]

    def route(self, assignment: dict) -> AgentSpec | None: ...

    def details(
        self, assignments: list[dict], versions: dict[str, str]
    ) -> tuple[str, list]: ...


def official_key(workspace: Path) -> str:
    """Read only this workspace's literal .env, with an explicit environment override.

    No upward discovery, interpolation, or mutation of the process environment.
    Model credentials are the agent's responsibility, not this loader's.
    """
    values = dotenv_values(workspace / ".env", interpolate=False)
    key = os.environ.get("ALTRUAGENT_OFFICIAL_AGENT_KEY") or values.get(
        "ALTRUAGENT_OFFICIAL_AGENT_KEY"
    )
    if not key:
        raise ValueError(
            "Configure the Official Agent Key in the runtime workspace's .env."
        )
    return key


def running_version(workspace: Path, *, require_clean: bool = True) -> str:
    """Verify workspace code and its directly declared editable neutral runtime.

    Only tracked pyproject configuration is used; dependency discovery is not
    recursive. Preflight (require_clean=False) still reads just the workspace SHA.
    """
    sources = [workspace]
    pyproject = workspace / "pyproject.toml"
    if require_clean and pyproject.is_file():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", "pyproject.toml"],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode == 0:
            source = (
                tomllib.loads(pyproject.read_text(encoding="utf-8"))
                .get("tool", {})
                .get("uv", {})
                .get("sources", {})
                .get("werewolf-runtime", {})
            )
            if (
                isinstance(source, dict)
                and source.get("editable") is True
                and "path" in source
            ):
                if not isinstance(source["path"], str) or not source["path"]:
                    raise ValueError("Editable runtime requires a nonempty local path.")
                dependency = (workspace / source["path"]).resolve(strict=True)
                if not dependency.is_dir():
                    raise ValueError("Editable runtime source must be a directory.")
                if dependency != workspace.resolve():
                    sources.append(dependency)
    version = None
    for source_workspace in sources:
        if require_clean:
            status = subprocess.run(
                [
                    "git",
                    "--no-optional-locks",
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    ".",
                    ":(exclude)data",
                ],
                cwd=source_workspace,
                check=True,
                capture_output=True,
                text=True,
            )
            if status.stdout.strip():
                raise ValueError(
                    "Commit runtime/agent code, prompts, dependencies, and routing configuration before playing."
                )
        current = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if version is not None and current != version:
            raise ValueError(
                "Editable runtime source must share the workspace's Git HEAD."
            )
        version = current
    return version


def validate_experiments(lineup: Lineup, versions: dict[str, str]) -> None:
    for spec in lineup.agents.values():
        for filename in ("experiments.jsonl", "results.jsonl"):
            path = spec.workspace / "data" / filename
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    row.get("experiment_id") == spec.experiment["experiment_id"]
                    and row.get("version") != versions[spec.name]
                ):
                    raise ValueError(
                        "Use a new experiment ID for a different tested version."
                    )
