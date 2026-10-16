"""Neutral seat-scoped official transport, with no policy or workspace configuration.

The official https://api.altruagent-game.com/skill/mcp documents stateless,
JSON-only raw tools/call requests without an initialize handshake. Connections
are reused; only a definite HTTP 401 permits replaying a gameplay request.
"""

import asyncio
import json
import re
import time
from urllib.parse import quote, urlsplit

import httpx

from .audit import Audit, Redactor

CONTROL_URL = "https://api.altruagent-game.com"
_ERROR_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}")
_TOKEN = re.compile(r"[!-~]+")
_LEASE_STOPS = {
    "seat_busy",
    "invalid_official_agent_key",
    "registration_incomplete",
    "assignment_not_grantable",
    "assignment_not_found",
    "seat_identity_changed",
    "seat_grant_mismatch",
    "invalid_seat_grant",
    "unsafe_seat_server_url",
    "invalid_lease_response",
}


def _error_code(value: object, fallback: str, redactor: Redactor | None = None) -> str:
    if (
        not isinstance(value, str)
        or not _ERROR_CODE.fullmatch(value)
        or (redactor is not None and redactor.clean(value) != value)
    ):
        return fallback
    return value


class PlatformError(RuntimeError):
    """A bounded code and HTTP status, never a response body or network exception."""

    def __init__(self, code: str, status: int = 0) -> None:
        self.code = _error_code(code, "platform_error")
        self.status = status if type(status) is int and 0 <= status <= 599 else 0
        super().__init__(f"Platform error {self.code} (HTTP {self.status}).")


def authorized_message(message: dict, self_id: int) -> dict:
    """Apply the authenticated seven-seat visibility rules before returning text."""
    if (
        not isinstance(message, dict)
        or type(self_id) is not int
        or not 0 <= self_id < 7
    ):
        raise PlatformError("unauthorized_seat_message")
    sender, recipients = message.get("sender"), message.get("recipients")
    visibility = message.get("visibility")
    if (
        type(sender) is not int
        or not 0 <= sender < 7
        or not isinstance(message.get("content"), str)
        or not isinstance(recipients, list)
        or len(recipients) > 1
        or any(type(seat) is not int or not 0 <= seat < 7 for seat in recipients)
        or sender in recipients
        or message.get("type") not in ("chat", "terminate")
        or visibility not in ("broadcast", "direct")
        or (visibility == "broadcast" and recipients)
        or (
            visibility == "direct"
            and (
                len(recipients) != 1
                or (sender != self_id and self_id not in recipients)
            )
        )
    ):
        raise PlatformError("unauthorized_seat_message")
    return {
        key: value
        for key, value in message.items()
        if key
        in (
            "message_id",
            "seq",
            "sender",
            "recipients",
            "visibility",
            "type",
            "content",
            "state_version",
            "sent_at",
        )
    }


class Control:
    def __init__(self, official_key: str, redactor: Redactor) -> None:
        if not isinstance(official_key, str) or not _TOKEN.fullmatch(official_key):
            raise ValueError(
                "A nonempty official agent key without whitespace is required."
            )
        self.official_key, self.redactor = official_key, redactor
        self.redactor.add_secret(official_key)
        self.token = ""
        self._generation = 0
        self._auth_lock = asyncio.Lock()
        self.client = httpx.AsyncClient(
            base_url=CONTROL_URL, timeout=15, follow_redirects=False, trust_env=False
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def authenticate(self, *, expired_generation: int | None = None) -> None:
        async with self._auth_lock:
            # Track refreshes even if authentication reissues the same JWT.
            if self.token and (
                expired_generation is None or expired_generation != self._generation
            ):
                return
            try:
                response = await self.client.post(
                    CONTROL_URL + "/tournament/agent/authenticate",
                    json={"official_agent_key": self.official_key},
                    follow_redirects=False,
                )
            except httpx.HTTPError:
                raise PlatformError("network_error") from None
            data = self.decode(response, self.redactor)
            token = data.get("access_token")
            if not isinstance(token, str) or not _TOKEN.fullmatch(token):
                raise PlatformError("invalid_auth_response")
            self.redactor.add_secret(token)
            self.token = token
            self._generation += 1

    @staticmethod
    def decode(response: httpx.Response, redactor: Redactor | None = None) -> dict:
        try:
            data = response.json()
        except ValueError:
            raise PlatformError("invalid_json", response.status_code) from None
        if not isinstance(data, dict):
            raise PlatformError("invalid_response", response.status_code)
        if response.status_code >= 300 or "error" in data:
            code = _error_code(data.get("error"), "http_error", redactor)
            raise PlatformError(code, response.status_code)
        return data

    async def request(self, method: str, path: str, data: dict | None = None) -> dict:
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or path.startswith("//")
            or any(ord(char) <= 32 or ord(char) >= 127 for char in path)
            or "\\" in path
            or "#" in path
        ):
            raise PlatformError("unsafe_control_path")
        if not self.token:
            await self.authenticate()
        for attempt in range(2):
            generation, token = self._generation, self.token
            try:
                response = await self.client.request(
                    method,
                    CONTROL_URL + path,
                    json=data,
                    headers={"Authorization": f"Bearer {token}"},
                    follow_redirects=False,
                )
            except httpx.HTTPError:
                raise PlatformError("network_error") from None
            if response.status_code == 401 and attempt == 0:
                await self.authenticate(expired_generation=generation)
                continue
            return self.decode(response, self.redactor)
        raise PlatformError("authentication_failed", 401)


class SeatConnection:
    """One seat's transport; callers gate writes on stopped and lease_until."""

    def __init__(
        self, control: Control, assignment: dict, execution_id: str, redactor: Redactor
    ) -> None:
        if (
            not isinstance(assignment, dict)
            or any(
                not isinstance(assignment.get(key), str) or not assignment[key].strip()
                for key in ("seat_id", "match_id", "game_type")
            )
            or type(assignment.get("seat_position")) is not int
            or not 0 <= assignment["seat_position"] < 7
            or (
                "seat_count" in assignment
                and (
                    type(assignment["seat_count"]) is not int
                    or assignment["seat_count"] != 7
                )
            )
        ):
            raise PlatformError("invalid_seat_assignment")
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("A nonempty execution ID is required.")
        self.control, self.assignment = control, dict(assignment)
        self.execution_id, self.redactor = execution_id, redactor
        self.redactor.add_secret(execution_id)
        self.grant: dict | None = None
        self.client = httpx.AsyncClient(
            timeout=30, follow_redirects=False, trust_env=False
        )
        self.url, self.token = "", ""
        self.lease_until = 0.0
        self.counter = 0
        self._generation = 0
        self._grant_lock = asyncio.Lock()
        self._assignment_path = "/tournament/agent/assignments/" + quote(
            assignment["seat_id"], safe=""
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def refresh(self, *, expired_generation: int | None = None) -> None:
        async with self._grant_lock:
            if (
                expired_generation is not None
                and expired_generation != self._generation
            ):
                return
            request_started = time.monotonic()
            grant = await self.control.request(
                "POST",
                self._assignment_path + "/grant",
                {"execution_id": self.execution_id},
            )
            if not isinstance(grant, dict):
                raise PlatformError("invalid_seat_grant")
            server_url = grant.get("gameapi_server_url")
            try:
                if (
                    not isinstance(server_url, str)
                    or not server_url
                    or any(ord(char) <= 32 or ord(char) >= 127 for char in server_url)
                    or any(char in server_url for char in "\\?#")
                ):
                    raise ValueError
                host = urlsplit(server_url)
                if (
                    host.scheme != "https"
                    or not host.hostname
                    or host.username is not None
                    or host.password is not None
                    or host.path not in ("", "/")
                    or (host.port is not None and not 1 <= host.port <= 65535)
                ):
                    raise ValueError
                url = str(httpx.URL(server_url).copy_with(path="/mcp"))
            except ValueError, httpx.InvalidURL:
                raise PlatformError("unsafe_seat_server_url") from None
            if (
                any(
                    grant.get(key) != self.assignment[key]
                    for key in ("seat_id", "match_id", "seat_position", "game_type")
                )
                or type(grant.get("seat_position")) is not int
            ):
                raise PlatformError("seat_grant_mismatch")
            if "seat_count" in grant and (
                type(grant["seat_count"]) is not int or grant["seat_count"] != 7
            ):
                raise PlatformError("seat_grant_mismatch")
            token = grant.get("access_token")
            if (
                not isinstance(token, str)
                or not _TOKEN.fullmatch(token)
                or any(
                    not isinstance(grant.get(key), str) or not grant[key].strip()
                    for key in ("game_session_id", "agent_id")
                )
            ):
                raise PlatformError("invalid_seat_grant")
            if self.grant is not None and any(
                grant[key] != self.grant[key] for key in ("game_session_id", "agent_id")
            ):
                raise PlatformError("seat_identity_changed")
            self.redactor.add_secret(token)
            self.grant, self.url, self.token = dict(grant), url, token
            self.lease_until = request_started + 25
            self._generation += 1

    async def call(self, tool: str, **arguments) -> dict:
        if self.grant is None:
            raise PlatformError("seat_grant_required")
        session_id = self.grant["game_session_id"]
        if "session_id" in arguments and arguments["session_id"] != session_id:
            raise PlatformError("unauthorized_seat_state")
        for attempt in range(2):
            self.counter += 1
            generation = self._generation
            try:
                response = await self.client.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Accept": "application/json, text/event-stream",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": self.counter,
                        "method": "tools/call",
                        "params": {
                            "name": tool,
                            "arguments": {**arguments, "session_id": session_id},
                        },
                    },
                    follow_redirects=False,
                )
            except httpx.HTTPError:
                raise PlatformError("game_network_error") from None
            if response.status_code == 401 and attempt == 0:
                await self.refresh(expired_generation=generation)
                continue
            if response.status_code >= 300:
                raise PlatformError("game_http_error", response.status_code)
            try:
                envelope = response.json()
                if not isinstance(envelope, dict):
                    raise ValueError
                if "error" in envelope:
                    raise PlatformError("mcp_protocol_error")
                result = envelope["result"]
                if not isinstance(result, dict):
                    raise ValueError
            except KeyError, TypeError, ValueError:
                raise PlatformError("invalid_mcp_response") from None
            try:
                data = result.get("structuredContent")
                if data is None:
                    data = json.loads(result["content"][0]["text"])
                if not isinstance(data, dict):
                    raise ValueError
            except KeyError, IndexError, TypeError, ValueError:
                code = (
                    "unknown_game_error"
                    if result.get("isError")
                    else "invalid_mcp_response"
                )
                raise PlatformError(code) from None
            if "error" in data:
                raise PlatformError(
                    _error_code(data["error"], "unknown_game_error", self.redactor)
                )
            if result.get("isError"):
                raise PlatformError("unknown_game_error")
            if (
                "session_id" in data
                or tool
                in (
                    "get_game_state",
                    "wait_for_update",
                    "get_messages",
                    "get_result",
                    "get_legal_actions",
                )
            ) and data.get("session_id") != session_id:
                raise PlatformError("unauthorized_seat_state")
            observed = (
                data
                if tool in ("get_game_state", "wait_for_update")
                or any(
                    key in data
                    for key in ("observation", "your_position", "game_state")
                )
                else None
            )
            nested = data.get("state")
            for state in (observed, nested):
                if state is not None and (
                    not isinstance(state, dict)
                    or state.get("session_id") != session_id
                    or type(state.get("your_position")) is not int
                    or state["your_position"] != self.assignment["seat_position"]
                ):
                    raise PlatformError("unauthorized_seat_state")
            for payload in (data, nested):
                if payload is None:
                    continue
                if (
                    "is_terminal" in payload
                    and type(payload["is_terminal"]) is not bool
                ):
                    raise PlatformError("invalid_terminal_flag")
                for field in ("new_messages", "messages"):
                    if field not in payload:
                        continue
                    messages = payload[field]
                    if messages is None:
                        messages = []
                    if not isinstance(messages, list):
                        raise PlatformError("unauthorized_seat_message")
                    payload[field] = [
                        authorized_message(message, self.assignment["seat_position"])
                        for message in messages
                    ]
                if "message" in payload:
                    payload["message"] = authorized_message(
                        payload["message"], self.assignment["seat_position"]
                    )
            return data
        raise PlatformError("seat_authentication_failed", 401)

    async def renew(self, stopped: asyncio.Event, audit: Audit) -> None:
        backoff = 1
        while not stopped.is_set():
            delay = 10
            try:
                request_started = time.monotonic()
                result = await self.control.request(
                    "POST",
                    self._assignment_path + "/lease/renew",
                    {"execution_id": self.execution_id},
                )
                if (
                    not isinstance(result, dict)
                    or result.get("seat_id") != self.assignment["seat_id"]
                ):
                    raise PlatformError("invalid_lease_response")
                self.lease_until = request_started + 25
                backoff = 1
            except PlatformError as error:
                delay = backoff
                backoff = min(30, backoff * 2)
                audit.record(
                    "lease_error", {"code": error.code, "status": error.status}
                )
                if error.code == "lease_not_held":
                    self.lease_until = 0.0
                    try:
                        await self.refresh()
                        delay, backoff = 10, 1
                    except PlatformError as grant_error:
                        audit.record(
                            "lease_error",
                            {"code": grant_error.code, "status": grant_error.status},
                        )
                        if (
                            grant_error.code in _LEASE_STOPS
                            or grant_error.status == 401
                        ):
                            stopped.set()
                            return
                elif error.code in _LEASE_STOPS or error.status == 401:
                    self.lease_until = 0.0
                    stopped.set()
                    return
            try:
                await asyncio.wait_for(stopped.wait(), delay)
            except TimeoutError:
                pass

    async def wait(self, state: dict, seen: int) -> dict:
        try:
            return await self.call(
                "wait_for_update",
                since_version=state["state_version"],
                since_message_seq=seen,
                since_phase=state.get("phase"),
                since_is_current_actor=state.get("is_current_actor"),
                timeout_seconds=20,
            )
        except PlatformError as error:
            if error.code != "TOO_MANY_WAITS":
                raise
            await asyncio.sleep(5)
            return await self.call("get_game_state")


def state_key(raw: dict) -> tuple:
    """Freshness token for a decision: version, phase, day, actor and seat identity."""
    return (
        raw.get("session_id"),
        raw.get("state_version"),
        raw.get("phase"),
        (raw.get("game_state") or {}).get("day"),
        (raw.get("game_state") or {}).get("phase"),
        raw.get("your_position"),
        raw.get("eliminated"),
        raw.get("is_current_actor"),
    )


def terminal_status(final: dict) -> tuple[str, str | None]:
    if final.get("status") == "completed":
        return "completed", None
    if final.get("status") in ("cancelled", "aborted"):
        return "aborted", "Platform cancelled the match before normal completion."
    return "failed", "Platform ended the match without a normal completion result."
