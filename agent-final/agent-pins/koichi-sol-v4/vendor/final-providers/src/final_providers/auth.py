"""Explicit ChatGPT OAuth and local, process-safe rotating credentials.

No method contacts the network except OAuthFlow.finish, access_token (when a
refresh is needed), and logout. The CLI owns the loopback HTTP listener: bind it
before begin(), and pass the complete callback URL to finish(). Never print that
URL or a credential record. A flow is single-use and is not persisted.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import time
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid

import httpx
import jwt

from .errors import ProviderError

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
JWKS_URL = ISSUER + "/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
INFERENCE_SCOPES = frozenset({"resource.invoke", "chatgpt.tokens.use.direct"})
_TOKENS = ("access_token", "refresh_token", "id_token")
_TERMINAL = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)
_ERROR_CODES = _TERMINAL | frozenset(
    {
        "access_denied",
        "temporarily_unavailable",
        "server_error",
        "invalid_scope",
        "invalid_client",
        "unauthorized_client",
    }
)


def _remaining(deadline: float) -> float:
    if not _finite(deadline):
        raise ProviderError("invalid_deadline")
    remaining = deadline - asyncio.get_running_loop().time()
    if not math.isfinite(remaining) or remaining <= 0:
        raise ProviderError("auth_deadline", retryable=True)
    return remaining


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and len(value) <= 131072


def _fingerprint(record: dict[str, Any] | None) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


class CredentialStore:
    """Private per-label records; callers must not log load_account() results.

    All path components are opened relative to verified directory descriptors,
    without following symlinks. Files must be single-link, owner-only regular
    files. Account names are hashed, not interpreted as paths. Use a dedicated
    root; an existing root is tightened to 0700, never a shared credential store.
    """

    def __init__(
        self, root: Path, register_secret: Callable[[str], None] | None = None
    ) -> None:
        self.root = Path(os.path.abspath(root))
        self._register_secret = register_secret

    def _register(self, value: Any) -> None:
        if _text(value) and self._register_secret is not None:
            try:
                self._register_secret(value)
            except Exception:
                raise ProviderError("secret_registration_failed") from None

    def _register_tokens(self, record: Any) -> None:
        if isinstance(record, dict):
            for source in (record, record.get("pending_rotation")):
                if isinstance(source, dict):
                    for name in (*_TOKENS, "client_secret"):
                        self._register(source.get(name))

    @contextmanager
    def _directory(self):
        fd = None
        try:
            fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            parts = self.root.parts[1:]
            if not parts:
                raise ProviderError("credential_path_unsafe")
            for index, part in enumerate(parts):
                try:
                    child = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=fd,
                    )
                except FileNotFoundError:
                    try:
                        os.mkdir(part, 0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                    child = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=fd,
                    )
                os.close(fd)
                fd = child
                if index == len(parts) - 1:
                    if os.fstat(fd).st_uid != os.getuid():
                        raise ProviderError("credential_path_unsafe")
                    os.fchmod(fd, 0o700)
            yield fd
        except OSError:
            raise ProviderError("credential_storage_error") from None
        finally:
            if fd is not None:
                os.close(fd)

    def _open(self, directory: int, name: str, *, create: bool = False) -> int:
        flags = os.O_RDWR if create else os.O_RDONLY
        fd = os.open(
            name,
            flags
            | os.O_NOFOLLOW
            | os.O_CLOEXEC
            | os.O_NONBLOCK
            | (os.O_CREAT if create else 0),
            0o600,
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
            raise ProviderError("credential_file_unsafe")
        return fd

    def _read(self, directory: int, name: str) -> dict[str, Any] | None:
        try:
            fd = self._open(directory, name)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            try:
                raw = stream.read(1048577)
                if len(raw) > 1048576:
                    raise ValueError
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise ValueError
            except ValueError, UnicodeError, RecursionError:
                raise ProviderError("credential_record_invalid") from None
        self._register_tokens(record)
        return record

    def _write(self, directory: int, name: str, record: dict[str, Any]) -> None:
        self._register_tokens(record)
        # Refuse unsafe destinations rather than replacing a symlink/hardlink.
        try:
            existing = self._open(directory, name)
        except FileNotFoundError:
            pass
        else:
            os.close(existing)
        temporary = ".pending-" + secrets.token_hex(16)
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record, stream, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass

    @staticmethod
    def _name(account: str) -> str:
        if not isinstance(account, str) or not account.strip() or len(account) > 256:
            raise ProviderError("account_label_invalid")
        try:
            encoded = account.encode()
        except UnicodeError:
            raise ProviderError("account_label_invalid") from None
        return "account-" + hashlib.sha256(encoded).hexdigest()

    def host_id(self) -> str:
        with self._directory() as directory:
            fd = self._open(directory, "host.lock", create=True)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ProviderError("credential_busy", retryable=True) from None
                record = self._read(directory, "host.json")
                if record is None:
                    record = {"host_id": uuid.uuid4().urn}
                    self._write(directory, "host.json", record)
                value = record.get("host_id")
                try:
                    if not isinstance(value, str) or uuid.UUID(value).urn != value:
                        raise ValueError
                except ValueError, AttributeError:
                    raise ProviderError("credential_record_invalid") from None
                return value
            finally:
                os.close(fd)

    def load_account(self, account: str) -> dict[str, Any] | None:
        """Load private data locally. Registers stored secrets before returning."""
        name = self._name(account)
        with self._directory() as directory:
            record = self._read(directory, name + ".json")
        if record is not None and record.get("account") != account:
            raise ProviderError("credential_record_invalid")
        return record

    def status(self, account: str) -> dict[str, Any]:
        """Allowlisted local metadata only; does not refresh or probe a catalog."""
        record = self.load_account(account) or {}
        expires = record.get("expires_at")
        valid_expiry = _finite(expires)
        scopes = record.get("scopes", [])
        entitled = isinstance(scopes, list) and all(isinstance(s, str) for s in scopes)
        entitled = entitled and INFERENCE_SCOPES.issubset(scopes)
        signed_in = "pending_rotation" not in record and _text(
            record.get("access_token")
        )
        usable = valid_expiry and (
            expires > time.time() or _text(record.get("refresh_token"))
        )
        return {
            "registered": _text(record.get("client_id")),
            "signed_in": signed_in,
            "inference_enabled": bool(signed_in and entitled and usable),
            "reauthorization_required": not signed_in
            or not entitled
            or not valid_expiry
            or (expires <= time.time() and not _text(record.get("refresh_token"))),
            "expires_at": expires if valid_expiry else None,
        }

    @asynccontextmanager
    async def _locked(self, account: str, deadline: float):
        name = self._name(account)
        _remaining(deadline)
        with self._directory() as directory:
            fd = self._open(directory, name + ".lock", create=True)
            try:
                while True:
                    _remaining(deadline)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        await asyncio.sleep(min(0.025, _remaining(deadline)))
                _remaining(deadline)
                yield directory, name + ".json"
            finally:
                os.close(fd)

    async def access_token(
        self,
        account: str,
        deadline: float,
        client: httpx.AsyncClient,
        *,
        registration_fingerprint: str | None = None,
    ) -> str:
        """Get/refresh under a lock, optionally bound to the CLI registration hash.

        A failed validation after a grant quarantines the received tokens rather
        than leaving an already-consumed refresh token eligible for another use.
        """
        async with self._locked(account, deadline) as (directory, name):
            record = self._read(directory, name)
            if record is None or record.get("account") != account:
                raise ProviderError("login_required")
            if registration_fingerprint is not None:
                identity = {
                    "issuer": record.get("issuer", ISSUER),
                    **{
                        key: record.get(key)
                        for key in ("client_id", "subject", "host_id")
                    },
                }
                if not all(
                    isinstance(value, str) and value.strip()
                    for value in identity.values()
                ):
                    raise ProviderError("registration_mismatch")
                encoded = json.dumps(
                    identity, sort_keys=True, separators=(",", ":")
                ).encode()
                fingerprint = hashlib.sha256(encoded).hexdigest()
                if (
                    not isinstance(registration_fingerprint, str)
                    or not registration_fingerprint.isascii()
                    or not secrets.compare_digest(fingerprint, registration_fingerprint)
                ):
                    raise ProviderError("registration_mismatch")
            if "pending_rotation" in record:
                raise ProviderError("reauthorization_required")
            if not _text(record.get("access_token")) and not _text(
                record.get("refresh_token")
            ):
                raise ProviderError("reauthorization_required")
            scopes = record.get("scopes", [])
            if not isinstance(scopes, list) or not all(
                isinstance(s, str) for s in scopes
            ):
                raise ProviderError("credential_record_invalid")
            if not INFERENCE_SCOPES.issubset(scopes):
                raise ProviderError("missing_inference_scope")
            expires = record.get("expires_at")
            if not _finite(expires):
                raise ProviderError("credential_record_invalid")
            if _text(record.get("access_token")) and expires > time.time() + 30:
                _remaining(deadline)
                return record["access_token"]
            if not _text(record.get("refresh_token")):
                raise ProviderError("reauthorization_required")
            if not _text(record.get("client_id")) or not _text(record.get("subject")):
                raise ProviderError("credential_record_invalid")
            keys = await _identity_keys(self, client, deadline)
            try:
                payload = await _request(
                    self,
                    client,
                    "POST",
                    TOKEN_URL,
                    deadline,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": record["refresh_token"],
                        "client_id": record["client_id"],
                        "resource": RESOURCE,
                    },
                )
            except ProviderError as error:
                if error.code in _TERMINAL:
                    self._clear(directory, name, record)
                    raise ProviderError(
                        "reauthorization_required", error.status
                    ) from None
                if error.code == "auth_response_invalid":
                    self._clear(
                        directory,
                        name,
                        record,
                        pending_rotation={
                            "validation_error": error.code,
                        },
                    )
                    raise ProviderError("reauthorization_required") from None
                raise
            # No suspension or deadline check between a consumed grant and its
            # atomic persistence, including when validation fails closed.
            try:
                renewed = _token_record(payload, record, login=False)
                if "id_token" in payload:
                    _validate_id(
                        payload["id_token"],
                        record["client_id"],
                        None,
                        record["subject"],
                        keys,
                    )
            except ProviderError as error:
                self._clear(
                    directory,
                    name,
                    record,
                    pending_rotation={
                        **{
                            key: payload[key]
                            for key in _TOKENS
                            if _text(payload.get(key))
                        },
                        "validation_error": error.code,
                    },
                )
                raise ProviderError("reauthorization_required") from None
            self._write(directory, name, renewed)
            if not INFERENCE_SCOPES.issubset(renewed["scopes"]):
                raise ProviderError("missing_inference_scope")
            _remaining(deadline)
            return renewed["access_token"]

    def _clear(
        self,
        directory: int,
        name: str,
        record: dict[str, Any],
        *,
        pending_rotation: dict[str, Any] | None = None,
    ) -> None:
        retained = {
            key: value
            for key, value in record.items()
            if key in ("account", "issuer", "client_id", "subject", "host_id")
        }
        if pending_rotation is not None:
            retained["pending_rotation"] = pending_rotation
        self._write(directory, name, retained)

    async def logout(
        self, account: str, deadline: float, client: httpx.AsyncClient
    ) -> dict[str, Any]:
        """Always clear local tokens once locked, even on revoke failure/cancel."""
        async with self._locked(account, deadline) as (directory, name):
            record = self._read(directory, name)
            if record is None:
                return {
                    "logged_in": False,
                    "revoked": False,
                    "revocation_status": "no_tokens",
                }
            if record.get("account") != account:
                raise ProviderError("credential_record_invalid")
            status = "no_tokens"
            revoked = False
            try:
                session = record.get("pending_rotation", record)
                if not isinstance(session, dict):
                    raise ProviderError("credential_record_invalid")
                token = session.get("refresh_token") or session.get("access_token")
                if _text(token):
                    if not _text(record.get("client_id")):
                        raise ProviderError("credential_record_invalid")
                    discovery = await _discovery(self, client, deadline)
                    endpoint = discovery.get("revocation_endpoint")
                    if not _valid_endpoint(endpoint):
                        raise ProviderError("revocation_unavailable")
                    await _request(
                        self,
                        client,
                        "POST",
                        endpoint,
                        deadline,
                        data={
                            "token": token,
                            "client_id": record["client_id"],
                            "token_type_hint": "refresh_token"
                            if session.get("refresh_token")
                            else "access_token",
                        },
                        empty_ok=True,
                    )
                    status, revoked = "revoked", True
            except ProviderError as error:
                status = error.code
            finally:
                self._clear(directory, name, record)
            return {"logged_in": False, "revoked": revoked, "revocation_status": status}


def _valid_endpoint(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme == "https"
            and parsed.netloc == "auth.openai.com"
            and bool(parsed.path)
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


async def _request(
    store: CredentialStore,
    client: httpx.AsyncClient,
    method: str,
    url: str,
    deadline: float,
    *,
    data: dict[str, str] | None = None,
    empty_ok: bool = False,
) -> dict[str, Any]:
    remaining = _remaining(deadline)
    try:
        async with asyncio.timeout_at(deadline):
            # Do not inherit an injected client's auth, cookies, headers, or query.
            request = httpx.Request(
                method,
                url,
                data=data,
                headers={"Accept": "application/json"},
                extensions={"timeout": httpx.Timeout(remaining).as_dict()},
            )
            response = await client.send(request, auth=None, follow_redirects=False)
    except TimeoutError, httpx.TimeoutException:
        raise ProviderError("auth_deadline", retryable=True) from None
    except httpx.HTTPError:
        raise ProviderError("auth_transport", retryable=True) from None
    try:
        payload = response.json()
    except ValueError, UnicodeError, RecursionError:
        payload = None
    store._register_tokens(payload)
    if not 200 <= response.status_code < 300:
        code = payload.get("error") if isinstance(payload, dict) else None
        code = (
            code
            if isinstance(code, str) and code in _ERROR_CODES
            else "auth_http_error"
        )
        raise ProviderError(
            code,
            response.status_code,
            response.status_code in (408, 429) or response.status_code >= 500,
        )
    if empty_ok:
        return {}
    if not isinstance(payload, dict):
        raise ProviderError("auth_response_invalid")
    return payload


async def _discovery(
    store: CredentialStore, client: httpx.AsyncClient, deadline: float
) -> dict[str, Any]:
    payload = await _request(store, client, "GET", DISCOVERY_URL, deadline)
    if (
        payload.get("issuer") != ISSUER
        or payload.get("authorization_endpoint") != AUTHORIZE_URL
        or payload.get("token_endpoint") != TOKEN_URL
        or payload.get("jwks_uri") != JWKS_URL
    ):
        raise ProviderError("discovery_invalid")
    return payload


async def _identity_keys(
    store: CredentialStore,
    client: httpx.AsyncClient,
    deadline: float,
) -> list[jwt.PyJWK]:
    """Fetch and validate signing keys before consuming a one-use grant."""
    await _discovery(store, client, deadline)
    jwks = await _request(store, client, "GET", JWKS_URL, deadline)
    try:
        raw_keys = jwks.get("keys")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise ValueError
        keys = []
        for raw in raw_keys:
            if not isinstance(raw, dict):
                raise ValueError
            if raw.get("use", "sig") != "sig" or raw.get("alg", "RS256") not in (
                "RS256",
                "ES256",
            ):
                continue
            if (
                not _text(raw.get("kid"))
                or "d" in raw
                or (
                    "key_ops" in raw
                    and (
                        not isinstance(raw["key_ops"], list)
                        or "verify" not in raw["key_ops"]
                    )
                )
            ):
                raise ValueError
            key = jwt.PyJWK.from_dict(raw)
            if key.algorithm_name in ("RS256", "ES256"):
                keys.append(key)
        if not keys:
            raise ValueError
    except jwt.PyJWTError, ValueError, TypeError, KeyError, OverflowError:
        raise ProviderError("identity_token_invalid") from None
    _remaining(deadline)
    return keys


def _validate_id(
    token: str,
    client_id: str,
    nonce: str | None,
    subject: str | None,
    keys: list[jwt.PyJWK],
) -> dict[str, Any]:
    """Validate against prefetched keys without suspending after a grant."""
    if not _text(token):
        raise ProviderError("identity_token_invalid")
    try:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        if algorithm not in ("RS256", "ES256") or not _text(header.get("kid")):
            raise ValueError
        matching = [
            key
            for key in keys
            if key.key_id == header["kid"] and key.algorithm_name == algorithm
        ]
        if len(matching) != 1:
            raise ValueError
        key = matching[0].key
        required = ["iss", "aud", "exp", "iat", "sub"]
        if nonce is not None:
            required.append("nonce")
        claims = jwt.decode(
            token,
            key,
            algorithms=[algorithm],
            audience=client_id,
            issuer=ISSUER,
            options={"require": required},
            leeway=0,
        )
        if (
            not _text(claims.get("sub"))
            or type(claims["exp"]) not in (int, float)
            or type(claims["iat"]) not in (int, float)
            or not math.isfinite(claims["exp"])
            or not math.isfinite(claims["iat"])
            or claims["exp"] <= claims["iat"]
            or (nonce is not None and claims.get("nonce") != nonce)
            or (subject is not None and claims["sub"] != subject)
            or ("azp" in claims and claims["azp"] != client_id)
            or (
                isinstance(claims["aud"], list)
                and len(claims["aud"]) > 1
                and claims.get("azp") != client_id
            )
        ):
            raise ValueError
    except jwt.PyJWTError, ValueError, TypeError, KeyError, OverflowError:
        raise ProviderError("identity_token_invalid") from None
    return claims


def _token_record(
    payload: dict[str, Any], previous: dict[str, Any], *, login: bool
) -> dict[str, Any]:
    expires = payload.get("expires_in")
    if (
        not _text(payload.get("access_token"))
        or not isinstance(payload.get("token_type"), str)
        or payload["token_type"].lower() != "bearer"
        or not _finite(expires)
        or not 0 < expires <= 31536000
        or (login and not _text(payload.get("id_token")))
        or any(name in payload and not _text(payload[name]) for name in _TOKENS)
    ):
        raise ProviderError("auth_response_invalid")
    scopes = payload.get("scope")
    if scopes is not None and not isinstance(scopes, str):
        raise ProviderError("auth_response_invalid")
    record = dict(previous)
    record.update({name: payload[name] for name in _TOKENS if name in payload})
    record["expires_at"] = time.time() + expires
    record["scopes"] = (
        sorted(set(scopes.split()))
        if scopes is not None
        else ([] if login else previous["scopes"])
    )
    return record


@dataclass(repr=False)
class OAuthFlow:
    """In-memory, single-use operator login. All fields are private data.

    begin(store, label, app_name=..., port=...) is network-free. finish() takes
    the exact callback URL and a monotonic deadline. Transactions expire after
    ten minutes. Returning sign-in reuses the registration, with an ID-token
    hint only when available. new_registration=True explicitly replaces the
    registration but still verifies the previous subject for an existing label.
    """

    store: CredentialStore
    account: str
    authorization_url: str
    redirect_uri: str
    state: str
    nonce: str
    verifier: str
    client_id: str | None
    subject: str | None
    _snapshot: str
    _expires_at: float = field(default_factory=lambda: time.monotonic() + 600)
    _used: bool = field(default=False)

    @classmethod
    def begin(
        cls,
        store: CredentialStore,
        account: str,
        *,
        app_name: str,
        port: int,
        new_registration: bool = False,
    ) -> OAuthFlow:
        if not isinstance(app_name, str) or not app_name.strip() or len(app_name) > 200:
            raise ProviderError("app_name_invalid")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ProviderError("callback_uri_invalid")
        record = store.load_account(account)
        if record is not None and (
            not _text(record.get("client_id")) or not _text(record.get("subject"))
        ):
            raise ProviderError("credential_record_invalid")
        client_id = record.get("client_id") if record and not new_registration else None
        hint = record.get("id_token") if record else None
        state, nonce, verifier = (secrets.token_urlsafe(48) for _ in range(3))
        for value in (state, nonce, verifier):
            store._register(value)
        redirect_uri = f"http://127.0.0.1:{port}/auth/callback"
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        parameters = {
            "response_type": "code",
            "client_id": client_id or "dynamic_agent_client",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge.rstrip(b"=").decode(),
            "code_challenge_method": "S256",
            "ext_agent_host_id": store.host_id(),
        }
        if client_id:
            if _text(hint):
                parameters["id_token_hint"] = hint
        else:
            parameters["agent_name_hint"] = app_name
        authorization_url = AUTHORIZE_URL + "?" + urlencode(parameters)
        store._register(authorization_url)
        return cls(
            store,
            account,
            authorization_url,
            redirect_uri,
            state,
            nonce,
            verifier,
            client_id,
            record.get("subject") if record else None,
            _fingerprint(record),
        )

    async def finish(
        self, callback_url: str, deadline: float, client: httpx.AsyncClient
    ) -> dict[str, Any]:
        """Validate and persist sign-in, returning only store.status() metadata."""
        if self._used:
            raise ProviderError("login_flow_used")
        self._used = True
        self.store._register(callback_url)
        remaining = self._expires_at - time.monotonic()
        if remaining <= 0:
            raise ProviderError("login_flow_expired")
        _remaining(deadline)
        deadline = min(deadline, asyncio.get_running_loop().time() + remaining)
        try:
            parsed = urlsplit(callback_url)
            target = urlsplit(self.redirect_uri)
            if (parsed.scheme, parsed.netloc, parsed.path) != (
                target.scheme,
                target.netloc,
                target.path,
            ) or parsed.fragment:
                raise ValueError
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
            if any(len(values) != 1 for values in query.values()):
                raise ValueError
            values = {key: value[0] for key, value in query.items()}
        except ValueError, TypeError:
            raise ProviderError("callback_invalid") from None
        for name in ("code", *_TOKENS):
            self.store._register(values.get(name))
        callback_state = values.get("state", "")
        if not callback_state.isascii() or not secrets.compare_digest(
            callback_state, self.state
        ):
            raise ProviderError("oauth_state_invalid")
        if "iss" in values and values["iss"] != ISSUER:
            raise ProviderError("callback_invalid")
        if "error" in values:
            raise ProviderError("oauth_denied")
        issued_id = values.get("client_id")
        if self.client_id:
            if issued_id is not None and issued_id != self.client_id:
                raise ProviderError("oauth_client_invalid")
            issued_id = self.client_id
        if not _text(issued_id) or issued_id == "dynamic_agent_client":
            raise ProviderError("oauth_client_invalid")
        if not _text(values.get("code")):
            raise ProviderError("callback_invalid")
        async with self.store._locked(self.account, deadline) as (directory, name):
            previous = self.store._read(directory, name)
            if _fingerprint(previous) != self._snapshot:
                raise ProviderError("credential_changed", retryable=True)
            host_id = self.store.host_id()
            keys = await _identity_keys(self.store, client, deadline)
            payload = await _request(
                self.store,
                client,
                "POST",
                TOKEN_URL,
                deadline,
                data={
                    "grant_type": "authorization_code",
                    "code": values["code"],
                    "client_id": issued_id,
                    "resource": RESOURCE,
                    "redirect_uri": self.redirect_uri,
                    "code_verifier": self.verifier,
                },
            )
            record = _token_record(payload, {}, login=True)
            claims = _validate_id(
                record["id_token"],
                issued_id,
                self.nonce,
                self.subject,
                keys,
            )
            record.update(
                {
                    "account": self.account,
                    "client_id": issued_id,
                    "subject": claims["sub"],
                    "host_id": host_id,
                }
            )
            self.store._write(directory, name, record)
        _remaining(deadline)
        return self.store.status(self.account)
