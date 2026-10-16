"""Explicit, conversation-free HTTP connectors. Constructors never contact providers.

Capability records are trusted local diagnostic output, not provider defaults. The
catalog diagnostic proves model access only: it does not infer effort or Fast/tier
entitlement from a model name. Persist its returned record in the owning CLI.

OpenRouter accepts the fixed reasoning.effort wire enum without a ChatGPT catalog.
Model-specific support is enforced by provider.require_parameters, not inferred
locally. Service-tier overrides still require separately verified metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from jsonschema import validators
from referencing import Registry
from referencing.exceptions import NoSuchResource

from .errors import ProviderError, normalize_usage as _usage

SOL_MODEL = "gpt-6.1-sol"
RESPONSES_URL = "https://api.openai.com/v1/responses"
MODELS_URL = "https://api.openai.com/v1/models"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Exact codes documented in Sign in with ChatGPT / Errors and recovery.
_PROVIDER_CODES = frozenset(
    {
        "subscription_sharing_user_not_eligible",
        "subscription_sharing_usage_limit_exceeded",
        "subscription_sharing_usage_unavailable",
        "subscription_sharing_unsupported_capability",
        "subscription_sharing_route_not_supported",
        "subscription_sharing_invalid_user",
        "subscription_sharing_user_unavailable",
        "chatpass_v2_scope_not_authorized",
        "chatpass_v2_invalid_authorization_context",
    }
)
_USAGE_LIMIT = "subscription_sharing_usage_limit_exceeded"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,199}\Z")
_MAX_BODY = 8 * 1024 * 1024
# OpenRouter reasoning.effort request enum; not a model/account entitlement list.
# https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})


class TokenStore(Protocol):
    async def access_token(
        self, account: str, deadline: float, client: httpx.AsyncClient
    ) -> str: ...


@dataclass(frozen=True)
class ModelCapabilities:
    """Account/source-bound cached verification, with separately evidenced settings.

    Times are Unix seconds. Empty allowlists permit no explicit ChatGPT effort or
    service-tier overrides. A trusted operator diagnostic may add values with its
    verification evidence; the catalog endpoint alone does not establish these
    capabilities. The factory/cache loader must bind registration_fingerprint to
    the selected account's validated registration identity and invalidate stale
    records on re-registration, even when account label and source are unchanged.
    """

    route: Literal["chatgpt", "openrouter"]
    model: str
    credential_source: str = field(repr=False)
    account: str | None = field(default=None, repr=False)
    verified_at: float = 0
    expires_at: float = 0
    efforts: tuple[str, ...] = ()
    service_tiers: tuple[str, ...] = ()
    effort_evidence: str | None = field(default=None, repr=False)
    service_tier_evidence: str | None = field(default=None, repr=False)
    registration_fingerprint: str | None = field(default=None, repr=False)
    fast_service_tiers: tuple[str, ...] = ()
    fast_evidence: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class RouteConfig:
    """credential_source is the selected store directory (or explicit key-source label).

    ChatGPT usage blocks live inside that directory, unless usage_limit_path is
    supplied. All seats for a credential must use the same path. Only an explicit
    operator action should remove a block after checking usage; no reset is guessed.
    """

    route: Literal["chatgpt", "openrouter"]
    model: str = SOL_MODEL
    auth_mode: Literal["oauth", "api_key"] | None = None
    credential_source: str | Path | None = field(default=None, repr=False)
    account: str | None = field(default=None, repr=False)
    effort: str | None = None
    service_tier: str | None = None
    focal: bool = False
    capabilities: ModelCapabilities | None = field(default=None, repr=False)
    usage_limit_path: Path | None = field(default=None, repr=False)
    require_fast: bool = False
    fast_stop_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.require_fast) is not bool:
            raise ProviderError("invalid_setting")
        if self.route not in ("chatgpt", "openrouter"):
            raise ProviderError("invalid_route")
        expected = "oauth" if self.route == "chatgpt" else "api_key"
        if self.auth_mode is None:
            object.__setattr__(self, "auth_mode", expected)
        if self.auth_mode != expected:
            raise ProviderError("invalid_auth_mode")
        if not isinstance(self.model, str) or not _SAFE_ID.fullmatch(self.model):
            raise ProviderError("invalid_model")
        if self.focal and (self.route != "chatgpt" or self.model != SOL_MODEL):
            raise ProviderError("focal_route_required")
        if not isinstance(self.credential_source, (str, Path)) or not str(
            self.credential_source
        ):
            raise ProviderError("credential_source_required")
        if any(
            value is not None and not isinstance(value, str)
            for value in (self.effort, self.service_tier)
        ):
            raise ProviderError("invalid_setting")
        if self.route == "chatgpt":
            if not isinstance(self.account, str) or not self.account:
                raise ProviderError("account_required")

    @property
    def resolved_usage_limit_path(self) -> Path | None:
        if self.route != "chatgpt":
            return None
        if self.usage_limit_path is not None:
            return self.usage_limit_path
        suffix = hashlib.sha256(self.account.encode()).hexdigest()
        return Path(self.credential_source) / (".usage-blocked-" + suffix)

    @property
    def resolved_fast_stop_path(self) -> Path:
        """Share a stop marker beside the key file or inside the ChatGPT store.

        Low-level callers using an OpenRouter source label rather than a key-file
        path must supply an explicit private fast_stop_path.
        """
        if self.fast_stop_path is not None:
            return self.fast_stop_path
        identity = json.dumps([self.route, str(self.credential_source), self.account])
        suffix = hashlib.sha256(identity.encode()).hexdigest()
        source = Path(self.credential_source)
        directory = source.parent if self.route == "openrouter" else source
        return directory / (".fast-stopped-" + suffix)


@dataclass(frozen=True)
class Completion:
    text: str
    parsed: Any
    usage: dict[str, Any]
    latency_ms: int
    served_model: str | None
    service_tier: str | None
    cost: float | None
    metadata: dict[str, Any]


def _budget(seconds: float | None, deadline: float | None) -> float:
    if (seconds is None) == (deadline is None):
        raise ProviderError("invalid_deadline")
    value = seconds if seconds is not None else deadline
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderError("invalid_deadline")
    if not math.isfinite(value):
        raise ProviderError("invalid_deadline")
    result = (
        asyncio.get_running_loop().time() + seconds if seconds is not None else deadline
    )
    if result <= asyncio.get_running_loop().time():
        raise ProviderError("deadline_exceeded", retryable=True)
    return result


def _check_deadline(deadline: float) -> None:
    # Synchronous JSON/schema work cannot be interrupted by asyncio's timer.
    if asyncio.get_running_loop().time() >= deadline:
        raise ProviderError("deadline_exceeded", retryable=True)


def _capabilities(config: RouteConfig, effort: str | None, tier: str | None) -> None:
    cap = config.capabilities
    if config.require_fast and (
        not isinstance(tier, str)
        or not tier
        or tier.strip().casefold()
        in {"standard", "auto", "default", "flex", "ultrafast"}
    ):
        raise ProviderError("fast_unverified")
    if config.route == "openrouter" and effort is not None:
        if not isinstance(effort, str) or effort not in _REASONING_EFFORTS:
            raise ProviderError("unsupported_effort")
    if config.route == "chatgpt" or tier is not None or config.require_fast:
        now = time.time()
        if (
            cap is None
            or cap.route != config.route
            or cap.model != config.model
            or cap.account != config.account
            or cap.credential_source != str(config.credential_source)
            or not (0 < cap.verified_at <= now < cap.expires_at)
        ):
            raise ProviderError("capabilities_unverified")
        if config.require_fast and (
            not cap.fast_evidence or tier not in cap.fast_service_tiers
        ):
            raise ProviderError("fast_unverified")
        if (
            config.route == "chatgpt"
            and effort is not None
            and (
                not isinstance(effort, str)
                or not cap.effort_evidence
                or effort not in cap.efforts
            )
        ):
            raise ProviderError("unsupported_effort")
        if (
            tier is not None
            and not config.require_fast
            and (
                not isinstance(tier, str)
                or not cap.service_tier_evidence
                or tier not in cap.service_tiers
            )
        ):
            raise ProviderError("unsupported_service_tier")


def _no_external_resource(uri: str):
    raise NoSuchResource(ref=uri)


def _json_loads(text: str) -> Any:
    def invalid_constant(value: str):
        raise ValueError

    def finite_float(value: str):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError
        return number

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=invalid_constant,
        parse_float=finite_float,
        object_pairs_hook=unique_object,
    )


def _format(response_format: Mapping | None) -> tuple[dict | None, Any]:
    if response_format is None:
        return None, None
    if not isinstance(response_format, Mapping):
        raise ProviderError("invalid_response_format")
    fmt = deepcopy(dict(response_format))
    kind = fmt.get("type")
    if kind in ("text", "json_object") and set(fmt) == {"type"}:
        return fmt, None
    if kind != "json_schema" or set(fmt) != {"type", "json_schema"}:
        raise ProviderError("invalid_response_format")
    spec = fmt["json_schema"]
    if (
        not isinstance(spec, dict)
        or set(spec) - {"name", "description", "strict", "schema"}
        or not isinstance(spec.get("name"), str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", spec["name"])
        or not isinstance(spec.get("schema"), (dict, bool))
        or ("strict" in spec and not isinstance(spec["strict"], bool))
        or ("description" in spec and not isinstance(spec["description"], str))
    ):
        raise ProviderError("invalid_response_format")
    schema = spec["schema"]

    # No schema can trigger remote resolution, including during validation.
    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef", "$recursiveRef") and (
                    not isinstance(value, str) or not value.startswith("#")
                ):
                    raise ProviderError("invalid_response_format")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)

    check_refs(schema)
    try:
        cls = (
            validators.validator_for(schema, default=None)
            if isinstance(schema, dict) and "$schema" in schema
            else validators.validator_for(schema)
        )
        if cls is None:
            raise ProviderError("invalid_response_format")
        cls.check_schema(schema)
        validator = cls(schema, registry=Registry(retrieve=_no_external_resource))
    except Exception:
        raise ProviderError("invalid_response_format") from None
    return fmt, validator


def _parsed(text: str, fmt: dict | None, validator) -> Any:
    if fmt is None or fmt["type"] == "text":
        return None
    try:
        value = _json_loads(text)
    except ValueError, RecursionError:
        raise ProviderError("invalid_json") from None
    if fmt["type"] == "json_object" and not isinstance(value, dict):
        raise ProviderError("invalid_json")
    if validator is not None:
        try:
            validator.validate(value)
        except Exception:
            raise ProviderError("schema_validation_failed") from None
    return value


def _messages(messages: Sequence[Mapping], *, chatgpt: bool) -> list[dict]:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ProviderError("invalid_messages")
    result = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ProviderError("invalid_messages")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool", "function"):
            raise ProviderError("unsupported_message")
        if chatgpt:
            if role in ("tool", "function") or set(message) != {"role", "content"}:
                raise ProviderError("unsupported_message")
            content = message["content"]
            if isinstance(content, list):
                # This small connector supports text only; reject rather than lose input.
                if any(
                    not isinstance(part, dict)
                    or set(part) != {"type", "text"}
                    or part["type"] != "text"
                    or not isinstance(part["text"], str)
                    for part in content
                ):
                    raise ProviderError("unsupported_message")
                content = [
                    dict(type="input_text", text=part["text"]) for part in content
                ]
            elif not isinstance(content, str):
                raise ProviderError("unsupported_message")
            result.append(
                {"role": "developer" if role == "system" else role, "content": content}
            )
        else:
            result.append(deepcopy(dict(message)))
    return result


def _safe_id(value: Any, token: str) -> str | None:
    return (
        value
        if isinstance(value, str) and _SAFE_ID.fullmatch(value) and token not in value
        else None
    )


def _retry_after_seconds(value: str | None) -> float | None:
    """Preserve the provider delay without clipping it to an arbitrary retry cap.

    The caller decides whether it fits the remaining decision deadline; this
    connector never sleeps or retries. Raw header text is never returned.
    """
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            delay = max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
        except TypeError, ValueError, OverflowError, IndexError:
            return None
    return delay if math.isfinite(delay) and delay >= 0 else None


async def _close(resource, deadline: float) -> None:
    try:
        async with asyncio.timeout_at(deadline):
            await resource.aclose()
    except TimeoutError:
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError from None
        raise


@asynccontextmanager
async def _client(injected: httpx.AsyncClient | None, deadline: float):
    if injected is not None:
        yield injected
    else:
        client = httpx.AsyncClient(trust_env=False, follow_redirects=False)
        try:
            yield client
        finally:
            await _close(client, deadline)


async def _read_json(response: httpx.Response) -> Any:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > _MAX_BODY:
            raise ProviderError("response_too_large", status=response.status_code)
    try:
        return _json_loads(body.decode("utf-8"))
    except ValueError, UnicodeError, RecursionError:
        raise ProviderError("invalid_response", status=response.status_code) from None


class InferenceClient:
    """No history, fallback, retry, price table, environment-key lookup, or tools.

    The injected client remains caller-owned. A fresh no-retry httpx client is
    otherwise scoped to each call. CredentialStore must register rotating secrets
    using the redaction callback supplied when that store was constructed.
    """

    def __init__(
        self,
        config: RouteConfig,
        *,
        credential_store: TokenStore | None = None,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        register_secret: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._store = credential_store
        self._api_key = api_key
        self._client = client
        self._register_secret = register_secret or (lambda secret: None)
        self._usage_storage_failed = False
        self._fast_stopped = False
        if config.route == "chatgpt":
            if credential_store is None or api_key is not None:
                raise ProviderError("invalid_credentials")
            self._usage_limit_path = config.resolved_usage_limit_path
        else:
            if (
                not isinstance(api_key, str)
                or not api_key.strip()
                or credential_store is not None
            ):
                raise ProviderError("invalid_credentials")
            self._register(api_key)
            self._usage_limit_path = None

    def _register(self, secret: str) -> None:
        try:
            self._register_secret(secret)
        except Exception:
            raise ProviderError("secret_registration_failed") from None

    async def close(self) -> None:
        """No persistent connection is owned; never refresh or send on close."""

    def _check_fast_stop(self) -> None:
        if not self.config.require_fast:
            return
        if self._fast_stopped:
            raise ProviderError("service_tier_downgraded")
        path = self.config.resolved_fast_stop_path
        try:
            os.lstat(path)
        except FileNotFoundError:
            try:
                # Refuse inference if the shared marker cannot be located safely.
                if not path.parent.is_dir():
                    raise ProviderError("fast_marker_unavailable")
            except OSError:
                raise ProviderError("fast_marker_unavailable") from None
            return
        except OSError:
            raise ProviderError("fast_marker_unavailable") from None
        raise ProviderError("service_tier_downgraded")

    def _stop_fast(self) -> None:
        self._fast_stopped = True
        try:
            fd = os.open(
                self.config.resolved_fast_stop_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.close(fd)
        except FileExistsError:
            pass
        except OSError:
            raise ProviderError("fast_marker_unavailable") from None
        raise ProviderError("service_tier_downgraded")

    def _check_usage(self) -> None:
        if self._usage_limit_path is not None:
            try:
                os.lstat(self._usage_limit_path)
            except FileNotFoundError:
                return
            except OSError:
                raise ProviderError("usage_marker_unavailable") from None
            raise ProviderError(_USAGE_LIMIT)

    def _failure(self, body: Any, status: int) -> None:
        raw_usage = body.get("usage") if isinstance(body, dict) else None
        usage, cost = _usage(raw_usage)
        metadata = {
            "usage": usage if isinstance(raw_usage, dict) else None,
            "cost": cost if self.config.route == "openrouter" else None,
        }
        error = body.get("error") if isinstance(body, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        if not isinstance(code, str) or code not in _PROVIDER_CODES:
            code = {
                400: "invalid_request",
                401: "authentication_failed",
                403: "permission_denied",
                408: "provider_timeout",
                429: "rate_limited",
            }.get(status, "provider_unavailable" if status >= 500 else "provider_error")
        if code == _USAGE_LIMIT and self._usage_limit_path is not None:
            try:
                fd = os.open(
                    self._usage_limit_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                os.close(fd)
            except FileExistsError:
                pass
            except OSError:
                # Fail closed in this process even if persistent storage fails.
                self._usage_storage_failed = True
                raise ProviderError(
                    "usage_marker_unavailable", status=status, **metadata
                ) from None
        retryable = code in {
            "rate_limited",
            "provider_timeout",
            "provider_unavailable",
            "subscription_sharing_usage_unavailable",
            "subscription_sharing_user_unavailable",
        }
        raise ProviderError(code, status=status, retryable=retryable, **metadata)

    async def _responses(self, response: httpx.Response) -> dict:
        data: list[str] = []
        completed_messages: dict[int, dict] = {}
        message_ids: dict[str, int] = {}
        event_name = None
        size = 0
        first_line = True
        async for line in response.aiter_lines():
            if first_line:
                line = line.removeprefix("\ufeff")
                first_line = False
            size += len(line)
            if size > _MAX_BODY:
                raise ProviderError("response_too_large", status=response.status_code)
            if line != "":
                if line.startswith("data:"):
                    data.append(line[5:].removeprefix(" "))
                elif line.startswith("event:"):
                    event_name = line[6:].strip()
                continue
            if not data:
                event_name = None
                continue
            try:
                event = _json_loads("\n".join(data))
            except ValueError, RecursionError:
                raise ProviderError(
                    "invalid_stream", status=response.status_code
                ) from None
            data = []
            if not isinstance(event, dict):
                raise ProviderError("invalid_stream", status=response.status_code)
            kind = event.get("type", event_name)
            if event_name is not None and kind != event_name:
                raise ProviderError("invalid_stream", status=response.status_code)
            event_name = None
            if kind in ("response.failed", "error", "response.error"):
                body = event.get("response", event)
                if kind == "error" and "error" not in event:
                    body = {"error": event}
                self._failure(body, response.status_code)
            if kind == "response.incomplete":
                raise ProviderError("incomplete_output", status=response.status_code)
            if kind in ("response.refusal.delta", "response.refusal.done"):
                raise ProviderError("refusal", status=response.status_code)
            if kind == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "message":
                    index = event.get("output_index")
                    item_id = item.get("id")
                    if (
                        type(index) is not int
                        or index < 0
                        or not isinstance(item_id, str)
                        or not item_id
                        or (
                            index in completed_messages
                            and completed_messages[index] != item
                        )
                        or (item_id in message_ids and message_ids[item_id] != index)
                    ):
                        raise ProviderError(
                            "invalid_stream", status=response.status_code
                        )
                    # Validate complete assistant content, including refusals; never
                    # assemble deltas, partial messages, or reasoning items.
                    self._output({"output": [item]}, chatgpt=True)
                    completed_messages[index] = item
                    message_ids[item_id] = index
            if kind == "response.completed":
                body = event.get("response")
                if not isinstance(body, dict) or body.get("status") != "completed":
                    raise ProviderError("invalid_response", status=response.status_code)
                if body.get("error"):
                    self._failure(body, response.status_code)
                if body.get("incomplete_details"):
                    raise ProviderError(
                        "incomplete_output", status=response.status_code
                    )
                # Some ChatGPT streams omit output from the completed envelope.
                if body.get("output") == [] and completed_messages:
                    body = dict(
                        body,
                        output=[
                            completed_messages[index]
                            for index in sorted(completed_messages)
                        ],
                    )
                return body
        raise ProviderError("stream_lost", status=response.status_code, retryable=True)

    async def complete(
        self,
        messages: Sequence[Mapping],
        *,
        seconds: float | None = None,
        deadline: float | None = None,
        effort: str | None = None,
        response_format: Mapping | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        max_tokens: int | None = None,
        service_tier: str | None = None,
        **unsupported: Any,
    ) -> Completion:
        end = _budget(seconds, deadline)
        started = time.monotonic()
        status = None
        retry_after = None
        usage = None
        cost = None
        try:
            async with asyncio.timeout_at(end):
                config = self.config
                effort = config.effort if effort is None else effort
                tier = config.service_tier if service_tier is None else service_tier
                chatgpt = config.route == "chatgpt"
                self._check_fast_stop()
                if unsupported or (
                    chatgpt
                    and any(
                        v is not None
                        for v in (temperature, max_output_tokens, max_tokens)
                    )
                ):
                    raise ProviderError("unsupported_parameter")
                if max_output_tokens is not None and max_tokens is not None:
                    raise ProviderError("conflicting_output_limits")
                limit = max_tokens if max_tokens is not None else max_output_tokens
                if limit is not None and (type(limit) is not int or limit <= 0):
                    raise ProviderError("invalid_output_limit")
                if temperature is not None and (
                    type(temperature) not in (int, float)
                    or not math.isfinite(temperature)
                ):
                    raise ProviderError("invalid_temperature")
                _capabilities(config, effort, tier)
                fmt, validator = _format(response_format)
                history = _messages(messages, chatgpt=chatgpt)
                payload = {"model": config.model, "stream": chatgpt}
                if chatgpt:
                    payload.update(input=history, store=False)
                    if fmt is not None:
                        text_format = {"type": fmt["type"]}
                        if fmt["type"] == "json_schema":
                            text_format.update(fmt["json_schema"])
                        payload["text"] = {"format": text_format}
                else:
                    payload.update(
                        messages=history,
                        provider={"allow_fallbacks": False, "require_parameters": True},
                    )
                    if fmt is not None:
                        payload["response_format"] = fmt
                    if temperature is not None:
                        payload["temperature"] = temperature
                    if limit is not None:
                        payload["max_tokens"] = limit
                if effort is not None:
                    payload["reasoning"] = {"effort": effort}
                    if not chatgpt:
                        payload["reasoning"]["exclude"] = True
                if tier is not None:
                    payload["service_tier"] = tier
                if self._usage_storage_failed:
                    raise ProviderError("usage_marker_unavailable")
                self._check_usage()
                _check_deadline(end)
                async with _client(self._client, end) as client:
                    _check_deadline(end)
                    binding = (
                        {
                            "registration_fingerprint": config.capabilities.registration_fingerprint
                        }
                        if chatgpt
                        and config.capabilities.registration_fingerprint is not None
                        else {}
                    )
                    token = (
                        await self._store.access_token(
                            config.account, end, client, **binding
                        )
                        if chatgpt
                        else self._api_key
                    )
                    if not isinstance(token, str) or not token.strip():
                        raise ProviderError("invalid_credentials")
                    self._register(token)
                    _check_deadline(end)
                    self._check_usage()
                    self._check_fast_stop()
                    # Build a standalone request: no injected client's default auth,
                    # cookies, query parameters, or conversation headers are inherited.
                    request = httpx.Request(
                        "POST",
                        RESPONSES_URL if chatgpt else OPENROUTER_URL,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Accept": "text/event-stream"
                            if chatgpt
                            else "application/json",
                        },
                        json=payload,
                        extensions={
                            "timeout": httpx.Timeout(
                                end - asyncio.get_running_loop().time()
                            ).as_dict()
                        },
                    )
                    response = await client.send(
                        request, stream=True, auth=None, follow_redirects=False
                    )
                    status = response.status_code
                    retry_after = _retry_after_seconds(
                        response.headers.get("retry-after")
                    )
                    try:
                        if status != 200:
                            try:
                                body = await _read_json(response)
                            except ProviderError as error:
                                if error.code != "invalid_response":
                                    raise
                                body = None
                            self._failure(body, status)
                        if chatgpt:
                            # The ChatGPT route can omit this header; success still
                            # requires strict SSE completion and output validation.
                            content_type = response.headers.get("content-type")
                            if (
                                content_type is not None
                                and content_type.split(";")[0].strip().lower()
                                != "text/event-stream"
                            ):
                                raise ProviderError("invalid_stream", status=status)
                            body = await self._responses(response)
                        else:
                            body = await _read_json(response)
                        if not isinstance(body, dict):
                            raise ProviderError("invalid_response", status=status)
                        raw_usage = body.get("usage")
                        normalized_usage, reported_cost = _usage(raw_usage)
                        usage = (
                            normalized_usage if isinstance(raw_usage, dict) else None
                        )
                        cost = reported_cost if not chatgpt else None
                        if body.get("error"):
                            self._failure(body, status)
                        if config.require_fast and body.get("service_tier") != tier:
                            self._stop_fast()
                        text = self._output(body, chatgpt=chatgpt)
                        parsed = _parsed(text, fmt, validator)
                        reasoning = body.get("reasoning")
                        observed_effort = (
                            reasoning.get("effort")
                            if isinstance(reasoning, dict)
                            else None
                        )
                        if (
                            not isinstance(observed_effort, str)
                            or observed_effort not in _REASONING_EFFORTS
                            or token in observed_effort
                        ):
                            observed_effort = None
                        _check_deadline(end)
                        result = Completion(
                            text=text,
                            parsed=parsed,
                            usage=normalized_usage,
                            latency_ms=round((time.monotonic() - started) * 1000),
                            served_model=_safe_id(body.get("model"), token),
                            service_tier=_safe_id(body.get("service_tier"), token),
                            cost=None if chatgpt else cost,
                            metadata={
                                "route": config.route,
                                "auth_mode": config.auth_mode,
                                "requested_model": config.model,
                                "requested_effort": effort,
                                "observed_effort": observed_effort,
                                "requested_service_tier": tier,
                            },
                        )
                        if config.focal and body.get("model") not in (
                            None,
                            SOL_MODEL,
                        ):
                            raise ProviderError("served_model_mismatch", status=status)
                    finally:
                        await _close(response, end)
                _check_deadline(end)
                return result
        except ProviderError as error:
            raise ProviderError(
                error.code,
                status=error.status if error.status is not None else status,
                retryable=error.retryable,
                retry_after_seconds=error.retry_after_seconds
                if error.retry_after_seconds is not None
                else retry_after,
                usage=error.usage if error.usage is not None else usage,
                cost=(error.cost if error.cost is not None else cost)
                if self.config.route == "openrouter"
                else None,
            ) from None
        except TimeoutError:
            raise ProviderError(
                "deadline_exceeded",
                status=status,
                retryable=True,
                retry_after_seconds=retry_after,
                usage=usage,
                cost=cost,
            ) from None
        except httpx.TimeoutException:
            raise ProviderError(
                "deadline_exceeded",
                status=status,
                retryable=True,
                retry_after_seconds=retry_after,
                usage=usage,
                cost=cost,
            ) from None
        except httpx.HTTPError:
            raise ProviderError(
                "stream_lost" if status == 200 else "network_error",
                status=status,
                retryable=True,
                retry_after_seconds=retry_after,
                usage=usage,
                cost=cost,
            ) from None
        except ValueError, TypeError, RecursionError, UnicodeError:
            raise ProviderError(
                "invalid_response",
                status=status,
                retry_after_seconds=retry_after,
                usage=usage,
                cost=cost,
            ) from None
        except Exception:
            raise ProviderError(
                "provider_error",
                status=status,
                retry_after_seconds=retry_after,
                usage=usage,
                cost=cost,
            ) from None

    @staticmethod
    def _output(body: dict, *, chatgpt: bool) -> str:
        if not chatgpt:
            choices = body.get("choices")
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], dict)
            ):
                raise ProviderError("invalid_response")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ProviderError("invalid_response")
            if (
                message.get("refusal")
                or choice.get("finish_reason") == "content_filter"
            ):
                raise ProviderError("refusal")
            if choice.get("finish_reason") != "stop":
                raise ProviderError("incomplete_output")
            text = message.get("content")
            if (
                not isinstance(text, str)
                or message.get("tool_calls")
                or message.get("function_call")
            ):
                raise ProviderError("invalid_response")
            return text
        output = body.get("output")
        if not isinstance(output, list):
            raise ProviderError("invalid_response")
        parts = []
        for item in output:
            if not isinstance(item, dict):
                raise ProviderError("invalid_response")
            if item.get("type") == "reasoning":
                continue
            if item.get("type") != "message" or item.get("role") != "assistant":
                raise ProviderError("invalid_response")
            if item.get("status") != "completed" or not isinstance(
                item.get("content"), list
            ):
                raise ProviderError("incomplete_output")
            for part in item["content"]:
                if not isinstance(part, dict):
                    raise ProviderError("invalid_response")
                if part.get("type") == "refusal":
                    raise ProviderError("refusal")
                if part.get("type") != "output_text" or not isinstance(
                    part.get("text"), str
                ):
                    raise ProviderError("invalid_response")
                parts.append(part["text"])
        if not parts:
            raise ProviderError("missing_output")
        return "".join(parts)


async def verify_chatgpt_catalog(
    config: RouteConfig,
    credential_store: TokenStore,
    *,
    seconds: float | None = None,
    deadline: float | None = None,
    client: httpx.AsyncClient | None = None,
    register_secret: Callable[[str], None] | None = None,
    cache_seconds: float = 3600,
) -> ModelCapabilities:
    """Explicit operator diagnostic, never called by startup or complete().

    Only confirms the exact requested slug; no search, alternative selection,
    inference, or invented capability fields. The caller attaches the validated
    registration fingerprint and persists the record; it is never derived from
    an account label or unvalidated catalog data here.
    """
    if config.route != "chatgpt":
        raise ProviderError("invalid_route")
    if (
        type(cache_seconds) not in (int, float)
        or not math.isfinite(cache_seconds)
        or cache_seconds <= 0
    ):
        raise ProviderError("invalid_cache_duration")
    end = _budget(seconds, deadline)
    try:
        async with asyncio.timeout_at(end), _client(client, end) as http:
            _check_deadline(end)
            token = await credential_store.access_token(config.account, end, http)
            if not isinstance(token, str) or not token.strip():
                raise ProviderError("invalid_credentials")
            if register_secret is not None:
                try:
                    register_secret(token)
                except Exception:
                    raise ProviderError("secret_registration_failed") from None
            _check_deadline(end)
            request = httpx.Request(
                "GET",
                MODELS_URL,
                headers={"Authorization": f"Bearer {token}"},
                extensions={
                    "timeout": httpx.Timeout(
                        end - asyncio.get_running_loop().time()
                    ).as_dict()
                },
            )
            response = await http.send(
                request, stream=True, auth=None, follow_redirects=False
            )
            try:
                if response.status_code != 200:
                    # This diagnostic does not return raw catalog/admission bodies.
                    raise ProviderError(
                        "catalog_verification_failed", status=response.status_code
                    )
                body = await _read_json(response)
                if not isinstance(body, dict) or not isinstance(
                    body.get("models"), list
                ):
                    raise ProviderError("invalid_catalog")
                if not any(
                    isinstance(model, dict) and model.get("slug") == config.model
                    for model in body["models"]
                ):
                    raise ProviderError("model_not_entitled")
            finally:
                await _close(response, end)
            _check_deadline(end)
            now = time.time()
            return ModelCapabilities(
                route="chatgpt",
                model=config.model,
                account=config.account,
                credential_source=str(config.credential_source),
                verified_at=now,
                expires_at=now + cache_seconds,
            )
    except TimeoutError, httpx.TimeoutException:
        raise ProviderError("deadline_exceeded", retryable=True) from None
    except httpx.HTTPError:
        raise ProviderError("network_error", retryable=True) from None
    except ProviderError:
        raise
    except Exception:
        raise ProviderError("catalog_verification_failed") from None
