"""Provider-neutral completion facade; policy concurrency and envelopes stay local."""

import asyncio
import json
import time
from copy import deepcopy
from weakref import WeakSet

from final_providers.errors import ProviderError
from final_providers.factory import build_client
from werewolf_runtime.audit import Redactor

from .config import Config

RETRY_BASE_SECONDS = 1.0


class ModelError(RuntimeError):
    pass


class ChatModel:
    def __init__(self, config: Config, audit, *, smoke_call_limit=None) -> None:
        config.require_model()
        self.config, self.audit = config, audit
        self.calls = 0
        self._smoke_call_limit = smoke_call_limit
        self.redactor = Redactor([])
        self._redactors = WeakSet([self.redactor])
        self._secrets = set()
        self._register_sink(audit)
        self._client = build_client(
            config.provider_config,
            config.workspace,
            register_secret=self._register_secret,
        )

    def _register_sink(self, sink):
        redactor = getattr(sink, "redactor", None)
        if redactor is not None:
            for secret in self._secrets:
                redactor.add_secret(secret)
            self._redactors.add(redactor)

    def _register_secret(self, secret):
        self._secrets.add(secret)
        for redactor in self._redactors:
            redactor.add_secret(secret)

    async def close(self) -> None:
        await self._client.close()

    async def complete(
        self, messages, effort, seconds, *, audit=None, response_format=None
    ) -> dict:
        sink = self.audit if audit is None else audit
        self._register_sink(sink)
        if effort not in ("low", "high"):
            raise ModelError("Unsupported reasoning effort.")
        started = time.monotonic()
        deadline = asyncio.get_running_loop().time() + seconds
        call_id = None
        route = self.config.provider_config["route"]
        configured_effort = self.config.provider_config.get("effort")
        effective_effort = effort if configured_effort is None else configured_effort
        if self.config.use_provider_default_effort:
            effective_effort = None
        logged_effort = (
            "provider_default" if effective_effort is None else effective_effort
        )
        fmt = (
            deepcopy(response_format)
            if response_format is not None
            else {"type": "json_object"}
        )
        payload = {
            "model": self.config.model,
            "messages": self.redactor.clean(messages),
            "response_format": fmt,
            "effort": logged_effort,
            "requested_effort": effective_effort,
            "policy_effort": effort,
            "provider": route,
            "requested_max_tokens": self.config.max_tokens,
            "enforced_max_tokens": self.config.max_tokens
            if route == "openrouter"
            else None,
        }
        options = (
            {"max_tokens": self.config.max_tokens} if route == "openrouter" else {}
        )
        try:
            async with asyncio.timeout_at(deadline):
                delay = RETRY_BASE_SECONDS
                while True:
                    if (
                        self._smoke_call_limit is not None
                        and self.calls >= self._smoke_call_limit
                    ):
                        raise ModelError("Synthetic smoke model call limit reached.")
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError
                    self.calls += 1
                    call_id = self.calls
                    sink.record(
                        "model_request",
                        {
                            "call_id": call_id,
                            "payload": payload,
                            "deadline_seconds": remaining,
                            "effort": logged_effort,
                            "policy_effort": effort,
                        },
                    )
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError
                    try:
                        completion = await self._client.complete(
                            payload["messages"],
                            seconds=deadline - asyncio.get_running_loop().time(),
                            effort=effective_effort,
                            response_format=fmt,
                            **options,
                        )
                        break
                    except ProviderError as error:
                        if error.usage is not None or error.cost is not None:
                            sink.record(
                                "model_usage",
                                {
                                    "call_id": call_id,
                                    "usage": {
                                        **(error.usage or {}),
                                        "cost": None
                                        if route == "chatgpt"
                                        else error.cost,
                                    },
                                },
                            )
                        sink.record(
                            "provider_error",
                            {
                                "call_id": call_id,
                                "error": error.code,
                                "status": error.status,
                                "retryable": error.retryable,
                                "retry_after_seconds": error.retry_after_seconds,
                            },
                        )
                        remaining = deadline - asyncio.get_running_loop().time()
                        wait = max(delay, error.retry_after_seconds or 0.0)
                        if (
                            not error.retryable
                            or wait >= remaining
                            or (
                                self._smoke_call_limit is not None
                                and self.calls >= self._smoke_call_limit
                            )
                        ):
                            raise ModelError(error.code) from None
                        sink.record(
                            "model_retry",
                            {
                                "call_id": call_id,
                                "delay_seconds": wait,
                                "effort": logged_effort,
                                "policy_effort": effort,
                            },
                        )
                        await asyncio.sleep(wait)
                        delay = min(8.0, delay * 2)
                usage = {**completion.usage, "cost": completion.cost}
                sink.record("model_usage", {"call_id": call_id, "usage": usage})
                parsed = completion.parsed
                if parsed is None:
                    parsed = json.loads(completion.text)
                if not isinstance(parsed, dict):
                    raise ModelError("Model output is not a JSON object.")
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError
                result = {
                    "parsed": self.redactor.clean(parsed),
                    "usage": usage,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "effort": logged_effort,
                    "requested_effort": effective_effort,
                    "policy_effort": effort,
                    "call_id": call_id,
                    "served_model": completion.served_model,
                    "provider": route,
                    "served_provider": completion.metadata.get("served_provider"),
                    "service_tier": completion.service_tier,
                    "cost": completion.cost,
                    "metadata": completion.metadata,
                }
                sink.record("model_response", self.redactor.clean(result))
                return result
        except asyncio.CancelledError:
            sink.record("model_cancelled", {"call_id": call_id})
            raise
        except (TimeoutError, ValueError, TypeError, ModelError) as error:
            reason = (
                str(error) if isinstance(error, ModelError) else type(error).__name__
            )
            sink.record(
                "model_error",
                {
                    "call_id": call_id,
                    "error": reason,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                },
            )
            raise ModelError(
                "deadline_exceeded" if isinstance(error, TimeoutError) else reason
            ) from None
