"""Versioned JSONL stdio for workspace-owned adapters, without agent policy.

Requests contain non-secret context/configuration and seat-scoped observations,
never transport credentials. Adapters load their own model credentials locally
and register them with ``sink.redactor.add_secret`` before emitting any events.
The runtime must redact these events again before storing them.
"""

import asyncio
import json
import math
import sys
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import TextIOBase
from threading import RLock
from typing import Protocol, TextIO

from .audit import Redactor

VERSION = 1
MAX_LINE_BYTES = 16 * 1024 * 1024


class Adapter(Protocol):
    async def start(self, context: dict, config: dict) -> dict: ...

    async def decide(
        self, state: dict, messages: list[dict], kind: str, seconds: float
    ) -> dict: ...

    async def close(self) -> None: ...


class EventSink:
    """An audit-compatible, flushed event writer; stdout is the default stream."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.redactor = Redactor([])
        self._stream = sys.stdout if stream is None else stream
        self._mutex = RLock()

    def _write(self, value: dict) -> None:
        line = json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n"
        if len(line) > MAX_LINE_BYTES:
            raise ValueError("Protocol line exceeds the size limit.")
        with self._mutex:
            self._stream.write(line)
            self._stream.flush()

    def record(self, event: str, data: object) -> None:
        if not isinstance(event, str) or not event.strip():
            raise ValueError("Event name must be a nonempty string.")
        self._write(
            {
                "type": "event",
                "event": self.redactor.clean(event),
                "data": self.redactor.clean(data),
            }
        )


def validate_metadata(metadata: dict) -> dict:
    """Validate adapter declarations without supplying defaults or selecting policy."""
    if not isinstance(metadata, dict):
        raise ValueError("Adapter metadata must be a dictionary.")  # noqa: TRY004
    seconds, chats = metadata.get("decision_seconds"), metadata.get("max_chats")
    if (
        not isinstance(metadata.get("strategy"), str)
        or not metadata["strategy"].strip()
        or not isinstance(metadata.get("configuration"), dict)
        or type(seconds) not in (int, float)
        or not 0 < seconds <= 60
        or not math.isfinite(seconds)
        or type(chats) is not int
        or not 0 <= chats <= 5
    ):
        raise ValueError("Invalid adapter metadata.")
    return metadata


def validate_decision(decision: dict, state: dict, kind: str) -> dict:
    """Validate exactly one legal action, broadcast/direct message, or termination.

    An empty message content means termination. Direct recipients use seven-seat
    positions, excluding self. If supplied, player records and the public alive
    list constrain the target; alive lists may contain positions or player names.
    """
    if not isinstance(decision, dict) or not isinstance(state, dict):
        raise ValueError("Decision and state must be dictionaries.")  # noqa: TRY004
    if kind == "action":
        legal = state.get("legal_actions")
        if (
            set(decision) != {"action_id"}
            or not isinstance(decision["action_id"], str)
            or not isinstance(legal, dict)
            or not isinstance(legal.get("actions"), list)
            or not any(
                isinstance(action, dict)
                and isinstance(action.get("action_id"), str)
                and action["action_id"] == decision["action_id"]
                for action in legal["actions"]
            )
        ):
            raise ValueError("Invalid action decision.")
        return decision
    if kind != "message" or set(decision) != {"content", "recipients"}:
        raise ValueError("Invalid decision kind or fields.")
    content, recipients = decision["content"], decision["recipients"]
    if (
        not isinstance(content, str)
        or len(content) > 1024
        or len(content.split()) > 50
        or not isinstance(recipients, list)
        or len(recipients) > 1
        or (not content.strip() and recipients)
        or any(type(seat) is not int or not 0 <= seat <= 6 for seat in recipients)
    ):
        raise ValueError("Invalid message decision.")
    if recipients:
        target, self_id = recipients[0], state.get("your_position")
        if type(self_id) is not int or not 0 <= self_id <= 6 or target == self_id:
            raise ValueError("Invalid message recipient.")
        player = None
        if "players" in state:
            if not isinstance(state["players"], list):
                raise ValueError("Invalid player records.")
            player = next(
                (
                    entry
                    for entry in state["players"]
                    if isinstance(entry, dict)
                    and type(entry.get("position")) is int
                    and entry["position"] == target
                ),
                None,
            )
            if player is None or ("alive" in player and player["alive"] is not True):
                raise ValueError("Message recipient is not alive.")
        public = state.get("game_state", {})
        if not isinstance(public, dict):
            raise ValueError("Invalid public state.")
        if "alive" in public:
            alive = public["alive"]
            if (
                not isinstance(alive, list)
                or any(
                    not (
                        (type(seat) is int and 0 <= seat <= 6)
                        or (isinstance(seat, str) and seat.strip())
                    )
                    for seat in alive
                )
                or (
                    target not in alive
                    and (player is None or player.get("name") not in alive)
                )
            ):
                raise ValueError("Message recipient is not alive.")
    return decision


class _Discard(TextIOBase):
    def write(self, text: str) -> int:
        return len(text)


def serve(factory: Callable[[EventSink], Adapter]) -> int:
    """Run an adapter until close/EOF; return zero on success, nonzero on failure.

    Close is attempted exactly once, including EOF and failed requests. A close
    request acknowledges with a null result and exits. Malformed requests without
    a trustworthy positive integer ID receive an error with ``id: null``. All
    failures are generic; incidental Python stdout/stderr output is discarded.
    """
    source, sink = getattr(sys.stdin, "buffer", sys.stdin), EventSink()

    async def run() -> int:
        adapter = None
        started = closed = failed = False
        request_id = None
        try:
            adapter = factory(sink)
            while True:
                request_id = None
                line = await asyncio.to_thread(source.readline, MAX_LINE_BYTES + 1)
                if not line:
                    break
                if isinstance(line, str):
                    line = line.encode("utf-8")
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError("Protocol line exceeds the size limit.")
                request = json.loads(line.decode("utf-8"))
                if (
                    isinstance(request, dict)
                    and type(request.get("id")) is int
                    and request["id"] > 0
                ):
                    request_id = request["id"]
                if (
                    not isinstance(request, dict)
                    or set(request) != {"id", "method", "params"}
                    or request_id is None
                    or request["method"] not in ("start", "decide", "close")
                    or not isinstance(request["params"], dict)
                ):
                    raise ValueError("Invalid adapter request.")
                method, params = request["method"], request["params"]
                if sink.redactor.clean(params) != params:
                    raise ValueError("Adapter requests must be non-secret JSON data.")
                if method == "start":
                    if (
                        started
                        or set(params) != {"protocol_version", "context", "config"}
                        or type(params["protocol_version"]) is not int
                        or params["protocol_version"] != VERSION
                        or not isinstance(params["context"], dict)
                        or not isinstance(params["config"], dict)
                    ):
                        raise ValueError("Invalid start request.")
                    result = validate_metadata(
                        await adapter.start(params["context"], params["config"])
                    )
                    result = validate_metadata(sink.redactor.clean(result))
                    started = True
                elif method == "decide":
                    if (
                        not started
                        or set(params) != {"state", "messages", "kind", "seconds"}
                        or not isinstance(params["state"], dict)
                        or not isinstance(params["messages"], list)
                        or any(
                            not isinstance(item, dict) for item in params["messages"]
                        )
                        or params["kind"] not in ("action", "message")
                        or type(params["seconds"]) not in (int, float)
                        or not 0 < params["seconds"] <= 60
                        or not math.isfinite(params["seconds"])
                    ):
                        raise ValueError("Invalid decide request.")
                    # Validate against the supplied state, not adapter mutations.
                    state = deepcopy(params["state"])
                    result = validate_decision(
                        await adapter.decide(
                            params["state"],
                            params["messages"],
                            params["kind"],
                            float(params["seconds"]),
                        ),
                        state,
                        params["kind"],
                    )
                    result = validate_decision(
                        sink.redactor.clean(result), state, params["kind"]
                    )
                else:
                    if not started or params:
                        raise ValueError("Invalid close request.")
                    closed = True
                    await adapter.close()
                    sink._write({"id": request_id, "result": None})
                    return 0
                sink._write({"id": request_id, "result": result})
        except BaseException:  # noqa: BLE001
            failed = True
        finally:
            if adapter is not None and not closed:
                closed = True
                try:
                    await adapter.close()
                except BaseException:  # noqa: BLE001
                    failed = True
        if failed:
            try:
                sink._write({"id": request_id, "error": "adapter_error"})
            except BaseException:  # noqa: BLE001, S110
                pass
            return 1
        return 0

    with _Discard() as discard, redirect_stdout(discard), redirect_stderr(discard):
        try:
            return asyncio.run(run())
        except BaseException:  # noqa: BLE001
            return 1
