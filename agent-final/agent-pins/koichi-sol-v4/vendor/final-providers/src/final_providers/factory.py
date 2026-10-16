"""Local-only adapter configuration. No environment or provider fallback discovery."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path

from . import __version__
from .auth import CredentialStore
from .cli import _load_key, _private_text, load_capabilities, registration_fingerprint
from .errors import ProviderError
from .inference import InferenceClient, ModelCapabilities, RouteConfig, _capabilities


def _source(value, workspace: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ProviderError("credential_source_required")
    workspace = Path(workspace).absolute()
    owner = next(
        (p for p in (workspace, *workspace.parents) if p.name == "agent-final"), None
    )
    if owner is None:
        raise ProviderError("final_workspace_required")
    source = Path(value)
    if not source.is_absolute():
        source = owner / source
    source = Path(os.path.abspath(source))
    if not source.is_relative_to(owner / ".private"):
        raise ProviderError("credential_source_outside_private_workspace")
    for part in (source, *source.parents):
        if part.is_symlink():
            raise ProviderError("credential_path_unsafe")
    return source


def build_client(
    provider_config: dict, workspace: Path, register_secret=None
) -> InferenceClient:
    """Resolve one explicitly selected seat route using protected local files only.

    Relative credential paths are relative to agent-final, not process CWD. The
    network-disabled preflight checks cached grants/capabilities without refresh.
    """
    allowed = {
        "route",
        "auth_mode",
        "model",
        "focal",
        "credential_source",
        "account",
        "capabilities_file",
        "effort",
        "service_tier",
        "require_fast",
    }
    if not isinstance(provider_config, dict) or set(provider_config) - allowed:
        raise ProviderError("invalid_provider_configuration")
    values = dict(provider_config)
    if not {"route", "model", "credential_source"}.issubset(values):
        raise ProviderError("explicit_provider_configuration_required")
    if type(values.get("focal", True)) is not bool:
        raise ProviderError("invalid_focal_setting")
    values.setdefault("focal", True)
    if type(values.get("require_fast", True)) is not bool:
        raise ProviderError("invalid_require_fast_setting")
    values.setdefault("require_fast", True)
    source = _source(values["credential_source"], workspace)
    values["credential_source"] = str(source)
    capabilities_file = values.pop("capabilities_file", None)
    store = None
    key = None
    try:
        if values["route"] == "chatgpt":
            store = CredentialStore(source, register_secret=register_secret)
            account = values.get("account")
            if not isinstance(account, str) or not account:
                raise ProviderError("account_required")
            if not store.status(account)["inference_enabled"]:
                raise ProviderError("login_or_plan_permission_required")
            capabilities = load_capabilities(store, account)
        elif values["route"] == "openrouter":
            if values.get("account") is not None:
                raise ProviderError("unexpected_account")
            key = _load_key(source)
            if register_secret is not None:
                register_secret(key)
            capabilities = None
        else:
            raise ProviderError("invalid_route")
        if capabilities_file is not None:
            record = json.loads(_private_text(_source(capabilities_file, workspace)))
            for name in ("efforts", "service_tiers", "fast_service_tiers"):
                if name == "fast_service_tiers":
                    record.setdefault(name, [])
                if not isinstance(record.get(name), list) or not all(
                    isinstance(v, str) for v in record[name]
                ):
                    raise ProviderError("capability_record_invalid")
                record[name] = tuple(record[name])
            capabilities = ModelCapabilities(**record)
            if store is not None:
                fingerprint = registration_fingerprint(store, account)
                if (
                    not fingerprint
                    or capabilities.registration_fingerprint != fingerprint
                ):
                    raise ProviderError("capabilities_unverified")
        config = RouteConfig(**values, capabilities=capabilities)
        _capabilities(config, config.effort, config.service_tier)
        return InferenceClient(
            config, credential_store=store, api_key=key, register_secret=register_secret
        )
    except ProviderError:
        raise
    except Exception:
        raise ProviderError("invalid_provider_configuration") from None


def resolved_metadata(client: InferenceClient) -> dict:
    """Allowlisted public metadata, excluding credential locations and identities."""
    config = client.config
    settings = {
        "provider": config.route,
        "auth_mode": config.auth_mode,
        "model": config.model,
        "focal": config.focal,
        "require_fast": config.require_fast,
        "requested_effort": config.effort,
        "requested_service_tier": config.service_tier,
        "observed_effort": None,
        "observed_service_tier": None,
        "connector_version": __version__,
        "conversation_policy": "complete_seat_authorized_history_per_call",
        "cost_policy": "unknown_subscription_cost"
        if config.route == "chatgpt"
        else "provider_reported",
    }
    dependencies = {}
    for name in ("final-providers", "httpx", "PyJWT", "cryptography", "jsonschema"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    settings["connector_dependencies"] = dependencies
    settings["provider_configuration_digest"] = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return settings
