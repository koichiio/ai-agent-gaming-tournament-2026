"""Explicit operator actions; imports, status, and diagnose never run inference.

Always select a dedicated private store and a stable local account label, e.g.
``final-provider --store /absolute/agent-final/.private/chatgpt --account main status``.
No CWD-relative discovery, environment credentials, fallback, or automatic unblock
is performed. Capability and probe records are private, account-bound JSON files.
Store and key paths must be absolute and contained in agent-final/.private.
Probe success proves request acceptance, not that an unobserved setting was used.
All inference is Fast-only. Probes require protected operator documentation of
the exact route/model/tier mapping; priority is not assumed to mean Fast. Each
selected label has one capability cache, so verifying another route replaces it.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlsplit

import httpx

from .auth import ISSUER, CredentialStore, OAuthFlow
from .errors import ProviderError
from .inference import (
    SOL_MODEL,
    InferenceClient,
    ModelCapabilities,
    RouteConfig,
    verify_chatgpt_catalog,
)

_STATUS_FIELDS = (
    "registered",
    "signed_in",
    "inference_enabled",
    "reauthorization_required",
)
_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "diagnostic",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean", "enum": [True]}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    },
}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default errors echo arbitrary supplied values and paths.
        raise ProviderError("invalid_arguments")


def parser() -> argparse.ArgumentParser:
    result = _Parser(description=__doc__)
    result.add_argument(
        "--store",
        required=True,
        type=Path,
        help="Absolute dedicated directory inside agent-final/.private",
    )
    result.add_argument(
        "--account", required=True, help="Stable selected local label (not printed)"
    )
    result.add_argument(
        "--timeout", type=float, default=60, help="Operation deadline in seconds"
    )
    commands = result.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login", help="Explicit browser-based OAuth login")
    login.add_argument("--app-name", default="UCLA final Werewolf agent")
    login.add_argument("--new-registration", action="store_true")
    login.add_argument(
        "--authorization-file",
        metavar="BASENAME",
        help="Use authorization-NAME.txt inside --store instead of a browser; removed on exit",
    )
    commands.add_parser("status", help="Local boolean metadata only; no network")
    commands.add_parser(
        "logout", help="Bounded remote revocation and local token clearing"
    )
    commands.add_parser(
        "catalog",
        help="Verify exact Sol access and save private capabilities; no inference",
    )
    for name in ("diagnose", "infer", "probe"):
        command = commands.add_parser(
            name,
            help={
                "diagnose": "Local route/config checks only; no network",
                "infer": "Explicit minimal structured inference (may consume usage)",
                "probe": "Explicit setting probe; success alone does not verify silent settings",
            }[name],
        )
        command.add_argument(
            "--route", choices=("chatgpt", "openrouter"), default="chatgpt"
        )
        command.add_argument("--model", default=SOL_MODEL)
        command.add_argument(
            "--config",
            type=Path,
            help="JSON RouteConfig fields; no credentials in this file",
        )
        command.add_argument(
            "--key-file",
            type=Path,
            help='Absolute owner-only JSON file inside agent-final/.private: {"api_key":"..."}',
        )
        if name == "probe":
            command.add_argument("--probe-effort")
            command.add_argument("--probe-service-tier")
            command.add_argument(
                "--fast-evidence-file",
                type=Path,
                help="Protected JSON route/model/service_tier/reference_url/statement documenting Fast",
            )
    clear = commands.add_parser(
        "clear-usage-block", help="Operator-only unblock after checking provider usage"
    )
    clear.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm that provider usage has been checked",
    )
    clear.add_argument("--route", choices=("chatgpt", "openrouter"), default="chatgpt")
    clear.add_argument("--model", default=SOL_MODEL)
    clear.add_argument("--key-file", type=Path)
    clear.set_defaults(config=None)
    return result


def _validate_private_path(path: Path) -> None:
    # Validate lexically before any storage operation. Resolving a path would
    # hide symlinks; descriptor-based store/key access rejects those separately.
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not any(
            parent.name == ".private" and parent.parent.name == "agent-final"
            for parent in path.parents
        )
    ):
        raise ProviderError("private_path_required")


def _record_name(store: CredentialStore, account: str, kind: str) -> str:
    return store._name(account) + "." + kind + ".json"


def _read_record(store: CredentialStore, account: str, kind: str):
    with store._directory() as directory:
        return store._read(directory, _record_name(store, account, kind))


def _write_record(store: CredentialStore, account: str, kind: str, record: dict):
    # Reuse the store's no-symlink, owner-only, fsync + atomic replacement rules.
    with store._directory() as directory:
        store._write(directory, _record_name(store, account, kind), record)


def registration_fingerprint(store: CredentialStore, account: str) -> str | None:
    """Hash the current private registration identity, never its rotating tokens.

    None means the selected account has no complete registration to bind to.
    Auth uses a fixed issuer; its records need not repeat that constant.
    """
    record = store.load_account(account)
    if record is None:
        return None
    identity = {
        "issuer": record.get("issuer", ISSUER),
        **{key: record.get(key) for key in ("client_id", "subject", "host_id")},
    }
    if not all(isinstance(value, str) and value.strip() for value in identity.values()):
        return None
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_capabilities(store: CredentialStore, account: str) -> ModelCapabilities | None:
    """Load the CLI's private cache for a caller constructing a RouteConfig.

    Reject changed/missing ChatGPT registration identities, including old unbound
    records. InferenceClient validates freshness and route/model/source/account.
    """
    record = _read_record(store, account, "capabilities")
    if record is None:
        return None
    if record.get("route") == "chatgpt":
        fingerprint = registration_fingerprint(store, account)
        if fingerprint is None or record.get("registration_fingerprint") != fingerprint:
            return None
    try:
        for key in ("efforts", "service_tiers", "fast_service_tiers"):
            values = record.get(key, [])
            if not isinstance(values, list) or not all(
                isinstance(v, str) for v in values
            ):
                raise ValueError
            record[key] = tuple(values)
        return ModelCapabilities(**record)
    except KeyError, ValueError, TypeError:
        raise ProviderError("capability_record_invalid") from None


def _private_text(path: Path) -> str:
    """Read a bounded owner-only regular file without following any symlinks."""
    path = path.absolute()
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in path.parts[1:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        fd = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=directory,
        )
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            os.close(fd)
            raise ProviderError("private_file_unsafe")
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            value = stream.read(65537)
    finally:
        os.close(directory)
    if len(value) > 65536:
        raise ProviderError("private_file_invalid")
    return value


def _load_key(path: Path) -> str:
    """Load only the protected JSON key format shared with the route factory."""
    _validate_private_path(path)
    try:
        record = json.loads(_private_text(path))
        if (
            not isinstance(record, dict)
            or set(record) != {"api_key"}
            or not isinstance(record["api_key"], str)
            or not record["api_key"].strip()
        ):
            raise ValueError
    except ValueError, UnicodeError, RecursionError:
        raise ProviderError("key_file_invalid") from None
    return record["api_key"].strip()


def _fast_documentation(
    path: Path | None, config: RouteConfig, tier: str | None
) -> dict:
    if path is None or tier is None:
        raise ProviderError("fast_evidence_required")
    _validate_private_path(path)
    try:
        record = json.loads(_private_text(path))
        if not isinstance(record, dict) or set(record) != {
            "route",
            "model",
            "service_tier",
            "reference_url",
            "statement",
        }:
            raise ValueError
        if (record["route"], record["model"], record["service_tier"]) != (
            config.route,
            config.model,
            tier,
        ):
            raise ValueError
        url = urlsplit(record["reference_url"])
        domain = "openai.com" if config.route == "chatgpt" else "openrouter.ai"
        if (
            url.scheme != "https"
            or not url.hostname
            or not (url.hostname == domain or url.hostname.endswith("." + domain))
            or url.username is not None
            or url.password is not None
            or url.port not in (None, 443)
            or not isinstance(record["statement"], str)
            or not re.search(r"\bFast\b", record["statement"], re.IGNORECASE)
        ):
            raise ValueError
    except ValueError, TypeError, AttributeError, UnicodeError, RecursionError:
        raise ProviderError("fast_evidence_invalid") from None
    return record


def _current(config: RouteConfig) -> bool:
    cap = config.capabilities
    return bool(
        cap
        and cap.route == config.route
        and cap.model == config.model
        and cap.account == config.account
        and cap.credential_source == str(config.credential_source)
        and 0 < cap.verified_at <= time.time() < cap.expires_at
    )


def _fast_ready(config: RouteConfig) -> bool:
    cap = config.capabilities
    return bool(
        _current(config)
        and config.service_tier
        and config.service_tier != "ultrafast"
        and config.service_tier in cap.service_tiers
        and cap.service_tier_evidence
        and config.service_tier in cap.fast_service_tiers
        and cap.fast_evidence
    )


def _route(args, store: CredentialStore) -> RouteConfig:
    values = {"route": args.route, "model": args.model}
    if args.config is not None:
        # Configuration is not secret, but arbitrary contents are never echoed.
        with args.config.open(encoding="utf-8") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ProviderError("route_config_invalid")
        try:
            supplied = json.loads(raw)
            allowed = {
                "route",
                "model",
                "auth_mode",
                "effort",
                "service_tier",
                "focal",
                "account",
                "credential_source",
                "require_fast",
            }
            if not isinstance(supplied, dict) or set(supplied) - allowed:
                raise ValueError
            account = (
                args.account if supplied.get("route", args.route) == "chatgpt" else None
            )
            if supplied.get("account", account) != account:
                raise ValueError
            values.update(supplied)
        except ValueError, TypeError:
            raise ProviderError("route_config_invalid") from None
    source = str(store.root)
    if values["route"] == "openrouter":
        if args.key_file is None:
            raise ProviderError("key_file_required")
        source = str(args.key_file.absolute())
    if values.get("credential_source", source) != source:
        raise ProviderError("route_config_invalid")
    values.update(
        credential_source=source,
        account=args.account if values["route"] == "chatgpt" else None,
    )
    if values.get("require_fast", True) is not True:
        raise ProviderError("fast_required")
    values["require_fast"] = True
    try:
        return RouteConfig(
            **values, capabilities=load_capabilities(store, args.account)
        )
    except TypeError:
        raise ProviderError("route_config_invalid") from None


async def _open_browser(url: str, deadline: float) -> None:
    # webbrowser can fall back to text browsers which print URLs. Isolate all
    # browser/child output at the OS descriptor level, not just Python stdout.
    # Send the URL via stdin, never command-line arguments or diagnostics.
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys, webbrowser; sys.exit(0 if webbrowser.open(sys.stdin.read(), new=1) else 1)",
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout_at(
            min(deadline, asyncio.get_running_loop().time() + 10)
        ):
            await process.communicate(url.encode())
        if process.returncode != 0:
            raise ProviderError("browser_unavailable_use_authorization_file")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def _login(args, store: CredentialStore, deadline: float, client) -> dict:
    loop = asyncio.get_running_loop()
    callback = loop.create_future()
    flow = None
    tasks = set()

    async def receive(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        accepted = False
        try:
            async with asyncio.timeout_at(min(deadline, loop.time() + 5)):
                raw = await reader.readuntil(b"\r\n\r\n")
                lines = raw.decode("ascii").split("\r\n")
                method, target, version = lines[0].split(" ")
                parsed = urlsplit(target)
                hosts = [
                    line[5:].strip()
                    for line in lines[1:]
                    if line.lower().startswith("host:")
                ]
                query = parse_qs(
                    parsed.query, keep_blank_values=True, strict_parsing=True
                )
                state = query.get("state", [])
                accepted = bool(
                    flow is not None
                    and method == "GET"
                    and version == "HTTP/1.1"
                    and parsed.path == "/auth/callback"
                    and not parsed.scheme
                    and not parsed.netloc
                    and not parsed.fragment
                    and hosts == [urlsplit(flow.redirect_uri).netloc]
                    and len(state) == 1
                    and state[0].isascii()
                    and secrets.compare_digest(state[0], flow.state)
                    and not callback.done()
                )
                if accepted:
                    callback.set_result(flow.redirect_uri + "?" + parsed.query)
                body = (
                    b"Return to the terminal.\n" if accepted else b"Request rejected.\n"
                )
                status = b"200 OK" if accepted else b"400 Bad Request"
                writer.write(
                    b"HTTP/1.1 " + status + b"\r\nContent-Type: text/plain\r\n"
                    b"Cache-Control: no-store\r\nReferrer-Policy: no-referrer\r\n"
                    b"Content-Security-Policy: default-src 'none'\r\nConnection: close\r\n"
                    + b"Content-Length: "
                    + str(len(body)).encode()
                    + b"\r\n\r\n"
                    + body
                )
                await writer.drain()
        except (
            ValueError,
            UnicodeError,
            OSError,
            TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            pass
        finally:
            writer.close()
            tasks.discard(task)

    # Binding must precede begin(): the selected port forms the redirect URI.
    server = await asyncio.start_server(receive, "127.0.0.1", 0, limit=8192)
    authorization_name = args.authorization_file
    authorization_directory = None
    try:
        port = server.sockets[0].getsockname()[1]
        flow = OAuthFlow.begin(
            store,
            args.account,
            app_name=args.app_name,
            port=port,
            new_registration=args.new_registration,
        )
        if authorization_name is not None:
            if not re.fullmatch(
                r"authorization-[A-Za-z0-9_-]{1,64}\.txt", authorization_name
            ):
                raise ProviderError("authorization_filename_invalid")
            # O_EXCL avoids overwriting credentials or another live flow's URL.
            with store._directory() as directory:
                fd = os.open(
                    authorization_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=directory,
                )
                authorization_directory = os.dup(directory)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.write(flow.authorization_url)
                    stream.flush()
                    os.fsync(stream.fileno())
            print('{"authorization_file_ready":true}', flush=True)
        else:
            await _open_browser(flow.authorization_url, deadline)
        async with asyncio.timeout_at(deadline):
            callback_url = await callback
            metadata = await flow.finish(callback_url, deadline, client)
        return {key: metadata.get(key) is True for key in _STATUS_FIELDS}
    finally:
        server.close()
        server.close_clients()
        pending = tuple(tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await server.wait_closed()
        if authorization_directory is not None:
            try:
                try:
                    os.unlink(authorization_name, dir_fd=authorization_directory)
                except FileNotFoundError:
                    pass
            finally:
                os.close(authorization_directory)


async def _run(args, *, connector_factory=None) -> dict:
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 600:
        raise ProviderError("invalid_timeout")
    _validate_private_path(args.store)
    if getattr(args, "key_file", None) is not None:
        _validate_private_path(args.key_file)
    if getattr(args, "fast_evidence_file", None) is not None:
        _validate_private_path(args.fast_evidence_file)
    if args.command == "probe" and args.probe_service_tier == "ultrafast":
        raise ProviderError("unsupported_service_tier")
    store = CredentialStore(args.store)
    store._name(args.account)
    deadline = asyncio.get_running_loop().time() + args.timeout
    if args.command == "status":
        metadata = store.status(args.account)
        cap = load_capabilities(store, args.account)
        local = RouteConfig(
            "chatgpt",
            credential_source=str(store.root),
            account=args.account,
            service_tier=next(iter(cap.fast_service_tiers), None) if cap else None,
            capabilities=cap,
            require_fast=True,
        )
        metadata["inference_enabled"] = bool(
            metadata.get("inference_enabled")
            and _fast_ready(local)
            and not os.path.lexists(local.resolved_usage_limit_path)
            and not os.path.lexists(local.resolved_fast_stop_path)
        )
        return {key: metadata.get(key) is True for key in _STATUS_FIELDS}
    if args.command == "clear-usage-block":
        if not args.confirm:
            raise ProviderError("confirmation_required")
        config = _route(args, store)
        cleared = []
        for path in (
            config.resolved_usage_limit_path,
            config.resolved_fast_stop_path,
        ):
            if path is None:
                cleared.append(False)
                continue
            with CredentialStore(path.parent)._directory() as directory:
                name = path.name
                try:
                    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    cleared.append(False)
                    continue
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_nlink != 1
                ):
                    raise ProviderError("usage_marker_unsafe")
                os.unlink(name, dir_fd=directory)
                os.fsync(directory)
                cleared.append(True)
        return {
            "usage_block_cleared": cleared[0],
            "service_downgrade_cleared": cleared[1],
            "remote_usage_checked": False,
        }
    config = None
    if args.command in ("diagnose", "infer", "probe"):
        config = _route(args, store)
        if args.command == "diagnose":
            cap = config.capabilities
            valid = _current(config)
            key_ready = False
            if config.route == "openrouter":
                key_ready = bool(_load_key(args.key_file))
            credential_ready = (
                key_ready
                if config.route == "openrouter"
                else store.status(args.account)["inference_enabled"]
            )
            blocked = bool(
                os.path.lexists(config.resolved_fast_stop_path)
                or (
                    config.resolved_usage_limit_path
                    and os.path.lexists(config.resolved_usage_limit_path)
                )
            )
            return {
                "route": config.route,
                "config_valid": True,
                "credential_ready": credential_ready,
                "capabilities_current": valid,
                "fast_ready": bool(
                    credential_ready and _fast_ready(config) and not blocked
                ),
                "service_downgrade_blocked": os.path.lexists(
                    config.resolved_fast_stop_path
                ),
                "explicit_effort_verified": bool(
                    valid and config.effort in cap.efforts and cap.effort_evidence
                ),
                "explicit_service_tier_verified": bool(
                    valid
                    and config.service_tier in cap.service_tiers
                    and cap.service_tier_evidence
                ),
                "usage_blocked": bool(
                    config.resolved_usage_limit_path
                    and os.path.lexists(config.resolved_usage_limit_path)
                ),
                "network_checked": False,
            }
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
        if args.command == "login":
            return await _login(args, store, deadline, client)
        if args.command == "logout":
            try:
                async with asyncio.timeout_at(deadline):
                    result = await store.logout(args.account, deadline, client)
            except TimeoutError:
                # Cancellation inside logout clears tokens once locked. If the
                # lock itself timed out, inspect locally instead of claiming it.
                result = {
                    "logged_in": store.status(args.account)["signed_in"],
                    "revoked": False,
                }
            confirmed = result.get("revoked") is True
            return {
                "local_tokens_cleared": result.get("logged_in") is False,
                "remote_revocation_confirmed": confirmed,
                "remote_revocation": "confirmed" if confirmed else "unconfirmed",
            }
        if args.command == "catalog":
            fingerprint = registration_fingerprint(store, args.account)
            if fingerprint is None:
                raise ProviderError("login_required")
            config = RouteConfig(
                "chatgpt",
                model=SOL_MODEL,
                credential_source=str(store.root),
                account=args.account,
            )
            cap = await verify_chatgpt_catalog(
                config, store, deadline=deadline, client=client
            )
            if registration_fingerprint(store, args.account) != fingerprint:
                raise ProviderError("credential_changed")
            # Catalog verification alone can never authorize setting overrides.
            cap = replace(
                cap,
                efforts=(),
                service_tiers=(),
                effort_evidence=None,
                service_tier_evidence=None,
                fast_service_tiers=(),
                fast_evidence=None,
                registration_fingerprint=fingerprint,
            )
            _write_record(store, args.account, "capabilities", asdict(cap))
            return {
                "catalog_verified": True,
                "model": SOL_MODEL,
                "explicit_settings_verified": False,
            }
        if args.command == "probe":
            documentation = _fast_documentation(
                args.fast_evidence_file, config, args.probe_service_tier
            )
            if config.effort is not None or config.service_tier is not None:
                raise ProviderError("probe_requires_unset_config_settings")
            for value in (args.probe_effort, args.probe_service_tier):
                if value is not None and not re.fullmatch(
                    r"[a-z][a-z0-9_-]{0,63}", value
                ):
                    raise ProviderError("invalid_probe_setting")
            cap = config.capabilities
            now = time.time()
            if not _current(config):
                if config.route == "chatgpt":
                    raise ProviderError("capabilities_unverified")
                cap = ModelCapabilities(
                    route=config.route,
                    model=config.model,
                    account=config.account,
                    credential_source=str(config.credential_source),
                    verified_at=now,
                    expires_at=now + 3600,
                )
            # Temporary request authorization only. Never persist this provisional
            # allowlist as verification: a provider may silently ignore a setting.
            temporary = replace(
                cap,
                efforts=(args.probe_effort,) if args.probe_effort else (),
                service_tiers=(args.probe_service_tier,)
                if args.probe_service_tier
                else (),
                effort_evidence="explicit operator probe request, not verified",
                service_tier_evidence="explicit operator probe request, not verified",
                fast_service_tiers=(args.probe_service_tier,),
                fast_evidence=json.dumps(documentation, separators=(",", ":")),
            )
            config = replace(
                config,
                capabilities=temporary,
                effort=args.probe_effort,
                service_tier=args.probe_service_tier,
                require_fast=True,
            )
        key = _load_key(args.key_file) if config.route == "openrouter" else None
        if not _fast_ready(config):
            raise ProviderError("fast_capabilities_unverified")
        connector = (connector_factory or InferenceClient)(
            config,
            credential_store=store if key is None else None,
            api_key=key,
            client=client,
        )
        completion = await connector.complete(
            [{"role": "user", "content": 'Return only {"ok":true}.'}],
            deadline=deadline,
            response_format=_FORMAT,
        )
        if completion.parsed != {"ok": True}:
            raise ProviderError("diagnostic_output_invalid")
        if (
            completion.served_model != config.model
            or completion.service_tier != config.service_tier
        ):
            raise ProviderError("fast_service_downgrade")
        if args.command == "probe":
            observed_effort = completion.metadata.get("observed_effort")
            effort_verified = bool(
                args.probe_effort
                and observed_effort == args.probe_effort
                and completion.served_model == config.model
            )
            tier_verified = bool(
                args.probe_service_tier
                and completion.service_tier == args.probe_service_tier
                and completion.served_model == config.model
            )
            evidence = {
                "route": config.route,
                "model": config.model,
                "observed_model": completion.served_model,
                "account": config.account,
                "credential_source": str(config.credential_source),
                "registration_fingerprint": cap.registration_fingerprint,
                "timestamp": time.time(),
                "request_succeeded": True,
                "requested_effort": args.probe_effort,
                "observed_effort": observed_effort,
                "requested_service_tier": args.probe_service_tier,
                "observed_service_tier": completion.service_tier,
                "fast_documentation": documentation,
                "limitation": "Acceptance alone does not establish use. Each setting requires exact observed setting and model matches.",
            }
            _write_record(store, args.account, "probe", evidence)
            updates = {}
            serialized_evidence = json.dumps(evidence, separators=(",", ":"))
            if effort_verified:
                updates.update(
                    efforts=tuple(sorted(set(cap.efforts) | {args.probe_effort})),
                    effort_evidence=serialized_evidence,
                )
            if tier_verified:
                updates.update(
                    service_tiers=tuple(
                        sorted(set(cap.service_tiers) | {args.probe_service_tier})
                    ),
                    service_tier_evidence=serialized_evidence,
                    fast_service_tiers=tuple(
                        sorted(set(cap.fast_service_tiers) | {args.probe_service_tier})
                    ),
                    fast_evidence=serialized_evidence,
                )
            if updates:
                verified = replace(cap, **updates)
                _write_record(store, args.account, "capabilities", asdict(verified))
            return {
                "probe_succeeded": True,
                "effort_verified": effort_verified,
                "service_tier_verified": tier_verified,
            }
        return {"inference_succeeded": True, "structured_output_valid": True}


def main(argv: list[str] | None = None) -> int:
    """Return an exit status; all public output is allowlisted JSON, never reprs."""
    try:
        args = parser().parse_args(argv)
        result = asyncio.run(_run(args), debug=False)
    except ProviderError as error:
        print(
            json.dumps({"error": error.code, "retryable": error.retryable}),
            file=sys.stderr,
        )
        return 1
    except TimeoutError, KeyboardInterrupt:
        print('{"error":"operation_interrupted_or_timed_out"}', file=sys.stderr)
        return 1
    except Exception:
        # Process/browser/filesystem/JSON errors may embed a URL, code, or path.
        print('{"error":"operation_failed"}', file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0
