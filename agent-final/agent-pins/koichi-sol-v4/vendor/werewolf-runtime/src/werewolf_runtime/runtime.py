"""Caller-routed supervisor, isolated decisions, and terminal-only records."""

import asyncio
import json
import re
import signal
import time
from pathlib import Path
from uuid import uuid4

from .audit import Audit, Redactor, append_experiment
from .config import AgentSpec, Lineup, running_version
from .platform import Control, PlatformError, SeatConnection, state_key, terminal_status
from .protocol import VERSION, validate_decision, validate_metadata
from .recovery import SeatBusyError, seat_lock
from .worker import AdapterError, Worker

_REJECTED = {
    "STALE_STATE",
    "NOT_YOUR_TURN",
    "WRONG_PHASE",
    "INVALID_ACTION",
    "PLAYER_ELIMINATED",
    "GAME_ALREADY_COMPLETE",
    "GAME_NOT_STARTED",
    "MESSAGES_QUOTA_EXCEEDED",
}


def context_for(
    spec: AgentSpec, assignment: dict, version: str, runner_version: str
) -> dict:
    return {
        "workspace": str(spec.workspace),
        "agent": spec.agent,
        "match_id": "playground:" + assignment["match_id"],
        "player_id": assignment["seat_id"],
        "seat_position": assignment["seat_position"],
        "version": version,
        "runner_version": runner_version,
        "experiment_id": spec.experiment["experiment_id"],
    }


def transient(error: PlatformError) -> bool:
    return (
        error.status == 429
        or error.status >= 500
        or error.code
        in ("network_error", "game_network_error", "agent_session_unavailable")
    )


async def _pause(seconds: float, *stops: asyncio.Event) -> None:
    waiters = [asyncio.create_task(stop.wait()) for stop in stops]
    try:
        done, _ = await asyncio.wait(
            waiters, timeout=seconds, return_when=asyncio.FIRST_COMPLETED
        )
        if done:
            raise asyncio.CancelledError
    finally:
        for task in waiters:
            task.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)


async def _retry_read(
    operation, audit: Audit, *stops: asyncio.Event, deadline: float | None = None
):
    """Only safe reads and same-execution grants may use this retry loop."""
    delay = 1
    while not any(stop.is_set() for stop in stops):
        try:
            if deadline is None:
                return await operation()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            result = await asyncio.wait_for(operation(), remaining)
            if time.monotonic() >= deadline:
                raise TimeoutError
            return result
        except PlatformError as error:
            if not transient(error) and error.code != "terminal_result_not_ready":
                raise
            audit.record("read_retry", {"code": error.code, "status": error.status})
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError from None
            await _pause(delay if remaining is None else min(delay, remaining), *stops)
            delay = min(30, delay * 2)
    raise asyncio.CancelledError


async def _watch_health(progress: list, *stops: asyncio.Event) -> None:
    warned = progress[0]
    while not any(stop.is_set() for stop in stops):
        await _pause(max(0, max(progress[0], warned) + 300 - time.monotonic()), *stops)
        now = time.monotonic()
        if now - progress[0] >= 300:
            print(
                "Health warning: no confirmed seat progress for at least 300 seconds; "
                "safe read recovery continues. No match timeout or cancellation was requested.",
                flush=True,
            )
            warned = now


async def play_seat(
    spec: AgentSpec,
    assignment: dict,
    control: Control,
    execution_id: str,
    version: str,
    runner_version: str,
    lineup_type: str,
    lineup_details: list,
    stopped: asyncio.Event,
) -> dict:
    try:
        with seat_lock(
            spec.workspace,
            "playground:" + assignment["match_id"],
            spec.agent,
            assignment["seat_id"],
        ):
            return await _play_seat(
                spec,
                assignment,
                control,
                execution_id,
                version,
                runner_version,
                lineup_type,
                lineup_details,
                stopped,
            )
    except SeatBusyError:
        print(
            "Seat already owned by runtime or recovery; duplicate refused before journal or platform access.",
            flush=True,
        )
        return {"deferred": True, "result": None}


async def _play_seat(
    spec: AgentSpec,
    assignment: dict,
    control: Control,
    execution_id: str,
    version: str,
    runner_version: str,
    lineup_type: str,
    lineup_details: list,
    stopped: asyncio.Event,
) -> dict:
    redactor = control.redactor
    audit = Audit(
        spec.workspace,
        version,
        assignment["seat_id"],
        assignment["match_id"],
        spec.strategy,
        {
            "adapter_command": list(spec.command),
            "adapter_config": spec.config,
            "runner_version": runner_version,
            "protocol_version": VERSION,
            "lineup": lineup_details,
            "assignment_context": assignment.get("context"),
            "experiment": dict(spec.experiment),
        },
        redactor,
        experiment_id=spec.experiment["experiment_id"],
        lineup_type=lineup_type,
        agent_id=spec.agent,
    )
    connection, worker = None, None
    seat_stopped = asyncio.Event()
    lease = None
    final, role, match_gone = None, None, False
    status, reason = "aborted", "Runtime stopped before the match ended."
    metadata = None
    chats, external_seq, discussion_day, seen = 0, -1, None, -1
    local_terminated = False
    last_activity, has_acted = time.monotonic(), False
    progress = [time.monotonic()]
    progress_key, progress_seq = None, -1
    ambiguous_key, terminal_seen = None, False
    health = asyncio.create_task(_watch_health(progress, stopped, seat_stopped))

    async def read(tool, *, deadline=None, **arguments):
        nonlocal progress_key, progress_seq, terminal_seen
        if tool not in (
            "get_game_state",
            "get_messages",
            "get_result",
            "wait_for_update",
        ):
            raise ValueError("Recovery only permits safe reads.")

        async def operation():
            if tool == "wait_for_update":
                result = await connection.wait(arguments["state"], arguments["seen"])
            else:
                result = await connection.call(tool, **arguments)
            if "is_terminal" in result and type(result["is_terminal"]) is not bool:
                raise PlatformError("invalid_terminal_flag")
            if tool == "get_result" and result.get("is_terminal") is not True:
                raise PlatformError("terminal_result_not_ready")
            return result

        result = await _retry_read(
            operation, audit, stopped, seat_stopped, deadline=deadline
        )
        if tool in ("get_game_state", "wait_for_update"):
            if result.get("is_terminal") is True and not terminal_seen:
                audit.record(
                    "termination_verified", {"source": tool, "is_terminal": True}
                )
            terminal_seen |= result.get("is_terminal") is True
            marker = (
                state_key(result),
                result.get("is_terminal"),
                result.get("waiting_for_players"),
            )
            if marker != progress_key:
                progress[0], progress_key = time.monotonic(), marker
        seq = max(
            [
                progress_seq,
                *(
                    m["seq"]
                    for m in (
                        result.get("new_messages") or result.get("messages") or []
                    )
                    if type(m.get("seq")) is int
                ),
            ]
        )
        if seq > progress_seq:
            progress[0], progress_seq = time.monotonic(), seq
        return result

    try:
        audit.record("assignment", assignment)
        connection = SeatConnection(control, assignment, execution_id, redactor)
        worker = Worker(spec, audit.record)
        await _retry_read(connection.refresh, audit, stopped, seat_stopped)
        lease = asyncio.create_task(connection.renew(seat_stopped, audit))
        # Connect before adapter startup: even slow initialization cannot hold the start gate.
        state = await read("get_game_state")
        await worker.launch()
        metadata = validate_metadata(
            await worker.request(
                "start",
                {
                    "protocol_version": VERSION,
                    "context": context_for(spec, assignment, version, runner_version),
                    "config": spec.config,
                },
            )
        )
        if metadata["strategy"] != spec.strategy:
            raise AdapterError("Adapter strategy differs from the fixed lineup.")
        audit.record("adapter_metadata", metadata)
        while not stopped.is_set() and not seat_stopped.is_set():
            try:
                audit.record("state", state)
                if "is_terminal" in state and type(state["is_terminal"]) is not bool:
                    raise PlatformError("invalid_terminal_flag")
                if state.get("is_terminal") is True:
                    if not terminal_seen:
                        audit.record(
                            "termination_verified",
                            {"source": "action_state", "is_terminal": True},
                        )
                    terminal_seen = True
                    status, reason = (
                        "failed",
                        "Platform terminal state observed; final result unavailable.",
                    )
                    final = await read("get_result")
                    status, reason = terminal_status(final)
                    break
                if state.get("waiting_for_players"):
                    state = await read("wait_for_update", state=state, seen=seen)
                    continue
                observation = state.get("observation")
                if isinstance(observation, str):
                    role_match = re.search(
                        r"^Your role: (WOLF|SEER|VILLAGER)$", observation, re.MULTILINE
                    )
                    if role_match:
                        role = role_match[1].lower()
                seat, phase = state["your_position"], state.get("phase")
                day = (state.get("game_state") or {}).get("day")
                if phase == "messaging" and day != discussion_day:
                    chats, external_seq, discussion_day, local_terminated = (
                        0,
                        -1,
                        day,
                        False,
                    )
                messages = state.get("new_messages") or []
                seen = max(
                    [seen, *(m["seq"] for m in messages if type(m.get("seq")) is int)]
                )
                if ambiguous_key is not None:
                    if state_key(state) == ambiguous_key:
                        state = await read("wait_for_update", state=state, seen=seen)
                        continue
                    ambiguous_key = None
                can_chat = (
                    phase == "messaging"
                    and not state.get("eliminated")
                    and not local_terminated
                    and seat not in (state.get("terminated_messaging") or [])
                )
                kind = (
                    "message"
                    if can_chat
                    else "action"
                    if (
                        phase == "moving"
                        and not state.get("eliminated")
                        and state.get("is_current_actor")
                        and (state.get("legal_actions") or {}).get("actions")
                    )
                    else None
                )
                new_external = max(
                    [
                        external_seq,
                        *(
                            m["seq"]
                            for m in messages
                            if m.get("sender") != seat and type(m.get("seq")) is int
                        ),
                    ]
                )
                terminate = can_chat and (
                    chats >= metadata["max_chats"]
                    or (state.get("messaging_seconds_left") or 0) <= 5
                )
                if (
                    kind == "message"
                    and not terminate
                    and chats
                    and new_external <= external_seq
                ):
                    kind = None
                if kind is None:
                    state = await read("wait_for_update", state=state, seen=seen)
                    continue
                key = state_key(state)
                seconds = min(
                    float(metadata["decision_seconds"]),
                    max(0.0, float(state.get("messaging_seconds_left") or 0) - 5)
                    if kind == "message"
                    else max(0.0, 115 - (time.monotonic() - last_activity)),
                )
                if (
                    kind == "action"
                    and not has_acted
                    and assignment.get("match_status") == "in_progress"
                ):
                    seconds = min(seconds, 8)
                decision_deadline = time.monotonic() + seconds
                if terminate:
                    decision = {"content": "", "recipients": []}
                else:
                    try:
                        transcript = await read(
                            "get_messages", deadline=decision_deadline
                        )
                        messages = transcript.get("messages") or []
                        audit.record("messages", messages)
                    except TimeoutError:
                        messages = []
                        seconds = 0.0
                        audit.record("transcript_timeout", {"kind": kind})
                    else:
                        seconds = max(0.0, decision_deadline - time.monotonic())
                    if seconds <= 0:
                        decision = None
                        audit.record("deadline_fallback", {"kind": kind})
                    else:
                        decision = validate_decision(
                            await worker.request(
                                "decide",
                                {
                                    "state": redactor.clean(state),
                                    "messages": redactor.clean(messages),
                                    "kind": kind,
                                    "seconds": seconds,
                                },
                                seconds + 1,
                            ),
                            state,
                            kind,
                        )
                if stopped.is_set() or seat_stopped.is_set():
                    audit.record(
                        "decision_discarded",
                        {"reason": "No confirmed execution lease."},
                    )
                    break
                refreshed = await read("get_game_state")
                if (
                    stopped.is_set()
                    or seat_stopped.is_set()
                    or time.monotonic() >= connection.lease_until
                ):
                    audit.record(
                        "decision_discarded",
                        {"reason": "No confirmed execution lease."},
                    )
                    state = refreshed
                    if not stopped.is_set() and not seat_stopped.is_set():
                        await _pause(1, stopped, seat_stopped)
                    continue
                if (
                    refreshed.get("is_terminal") is True
                    or state_key(refreshed) != key
                    or (
                        kind == "message"
                        and seat in (refreshed.get("terminated_messaging") or [])
                    )
                    or (
                        kind == "action"
                        and not (refreshed.get("legal_actions") or {}).get("actions")
                    )
                ):
                    audit.record(
                        "decision_discarded", {"reason": "Stale state/phase/actor."}
                    )
                    state = refreshed
                    continue
                if decision is None:
                    decision = (
                        {"content": "", "recipients": []}
                        if kind == "message"
                        else {
                            "action_id": refreshed["legal_actions"]["actions"][0][
                                "action_id"
                            ]
                        }
                    )
                decision = validate_decision(decision, refreshed, kind)
                audit.record(
                    "submission",
                    {
                        "kind": kind,
                        "decision": decision,
                        "state_version": refreshed["state_version"],
                    },
                )
                if kind == "action":
                    try:
                        result = await connection.call(
                            "play_action",
                            state_version=refreshed["state_version"],
                            action_id=decision["action_id"],
                        )
                    except PlatformError as error:
                        if transient(error):
                            ambiguous_key = key
                            audit.record(
                                "ambiguous_action",
                                {
                                    "reason": "Unconfirmed write; only reads until state advances."
                                },
                            )
                        raise
                    state = result.get("state") or await read("get_game_state")
                else:
                    content = decision["content"].strip()
                    try:
                        result = await connection.call(
                            "send_message",
                            message_type="chat" if content else "terminate",
                            content=content,
                            recipients=decision["recipients"],
                        )
                    except PlatformError as error:
                        if transient(error):
                            # No idempotency key: never repeat an ambiguous chat write.
                            audit.record(
                                "ambiguous_chat",
                                {
                                    "reason": "Unconfirmed write; chat will not be retried."
                                },
                            )
                            local_terminated = True
                            state = await read("get_game_state")
                            continue
                        raise
                    chats += bool(content)
                    local_terminated = not bool(content)
                    external_seq = max(
                        [
                            external_seq,
                            *(
                                m["seq"]
                                for m in messages
                                if m.get("sender") != seat and type(m.get("seq")) is int
                            ),
                        ]
                    )
                    if (result.get("remaining") or {}).get("this_turn") == 0:
                        chats = metadata["max_chats"]
                    state = await read("get_game_state")
                last_activity, has_acted = time.monotonic(), True
            except PlatformError as error:
                audit.record(
                    "platform_error", {"code": error.code, "status": error.status}
                )
                if error.code in _REJECTED:
                    if error.code == "MESSAGES_QUOTA_EXCEEDED":
                        chats = metadata["max_chats"]
                elif transient(error):
                    await _pause(1, stopped, seat_stopped)
                else:
                    raise
                # Read retries cannot replay an earlier move/message.
                state = await read("get_game_state")
    except asyncio.CancelledError:
        reason = (
            "Platform terminal state observed; final result unavailable when runtime stopped."
            if terminal_seen
            else "Runtime cancelled the seat; private trace retained pending match termination."
        )
    except (PlatformError, AdapterError, ValueError, OSError) as error:
        match_gone = (
            isinstance(error, PlatformError) and error.code == "SESSION_NOT_FOUND"
        )
        status, reason = (
            "failed",
            (
                f"Platform failure: {error.code}."
                if isinstance(error, PlatformError)
                else "Adapter startup/decision failed; see private protocol events."
            ),
        )
        audit.record("worker_error", {"reason": reason})
    finally:
        seat_stopped.set()
        tasks = [health, *([lease] if lease is not None else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            if worker is not None:
                await worker.close()
        finally:
            if connection is not None:
                try:
                    if final is None and connection.grant is not None:
                        try:
                            observed = await connection.call("get_result")
                            if observed.get("is_terminal") is True:
                                final = observed
                                # Preserve a failed/aborted worker attempt.
                        except PlatformError as error:
                            match_gone |= error.code == "SESSION_NOT_FOUND"
                finally:
                    await connection.close()
        own_return = (final or {}).get("your_return")
        outcome = "unknown"
        if status == "completed" and type(own_return) in (int, float):
            outcome = {1: "win", -1: "loss", 0: "draw"}.get(own_return, "unknown")
        finished = audit.finish(
            status,
            outcome=outcome,
            role=role,
            reason=reason,
            final=final,
            match_terminated=match_gone or terminal_seen,
        )
        if finished.get("deferred"):
            print(
                "Finalization deferred; private journal retained. Restart with unchanged configuration "
                "to resume, or use recovery only after match termination is confirmed.",
                flush=True,
            )
    return finished


def summarize(
    lineup: Lineup, played: set[str], versions: dict[str, str], redactor: Redactor
) -> None:
    for spec in lineup.agents.values():
        path = spec.workspace / "data" / "results.jsonl"
        latest = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["match_id"], row["agent"], row["player_id"])
                if key not in latest or row.get("revision", 1) > latest[key].get(
                    "revision", 1
                ):
                    latest[key] = row
        summaries = spec.workspace / "data" / "experiments.jsonl"
        previous = None
        if summaries.exists():
            for line in summaries.read_text().splitlines():
                if not line.strip():
                    continue
                candidate = json.loads(line)
                if candidate["experiment_id"] == spec.experiment["experiment_id"] and (
                    previous is None
                    or candidate.get("revision", 1) > previous.get("revision", 1)
                ):
                    previous = candidate
        experiment = {
            **spec.experiment,
            "baseline_version": spec.experiment["baseline_version"] or None,
        }
        if previous is not None:
            if previous["version"] != versions[spec.name]:
                raise ValueError("Experiment ID belongs to a different tested version.")
            if any(
                redactor.clean(previous.get(key)) != redactor.clean(experiment[key])
                for key in ("baseline_version", "changes", "hypothesis")
            ):
                raise ValueError(
                    "Experiment metadata conflicts with original published details."
                )
        match_ids = (
            {"playground:" + match for match in played}
            | set((previous or {}).get("match_ids", []))
            | {
                attempt["match_id"]
                for attempt in (previous or {}).get("recovery_attempts", [])
            }
        )
        rows = [
            row
            for row in latest.values()
            if row["match_id"] in match_ids
            and row.get("experiment_id") == spec.experiment["experiment_id"]
            and row["version"] == versions[spec.name]
        ]
        if not rows:
            continue
        summary = {
            **(previous or experiment),
            "version": versions[spec.name],
            "match_ids": sorted(
                set((previous or {}).get("match_ids", []))
                | {
                    attempt["match_id"]
                    for attempt in (previous or {}).get("recovery_attempts", [])
                }
                | {row["match_id"] for row in rows}
            ),
            "findings": {
                "games": len({row["match_id"] for row in rows}),
                "evaluated_seats": len(rows),
                "outcomes_by_lineup": {
                    kind: [
                        row["outcome"]
                        for row in rows
                        if row.get("lineup_type", "unknown") == kind
                    ]
                    for kind in ("homogeneous", "mixed", "unknown")
                },
                "incomplete_or_unknown": sum(
                    row["status"] != "completed" or row["outcome"] == "unknown"
                    for row in rows
                ),
            },
            "limitations": "Small match batch; seats in one game are correlated. Unknown outcomes are not losses; no win-rate estimate or improvement claim. Deferred live attempts are excluded.",
            "recommendation": "retest",
        }
        if previous is not None:
            if (
                previous["match_ids"] == summary["match_ids"]
                and previous["findings"] == summary["findings"]
            ):
                continue
            summary.update(
                revision=previous.get("revision", 1) + 1,
                correction_reason="Additional finalized matches or corrected results in this batch.",
            )
        append_experiment(spec.workspace, summary, redactor)


async def supervise(
    lineup: Lineup,
    official_key: str,
    versions: dict[str, str],
    runner_version: str,
    max_matches: int | None = 1,
    match_id: str | None = None,
    *,
    runtime_workspace: Path | None = None,
    keep_alive: bool = False,
) -> None:
    redactor = Redactor([official_key])
    control = Control(official_key, redactor)
    execution_id = str(uuid4())
    redactor.add_secret(execution_id)
    workers, played, missing, retry_at, known_lineups = {}, set(), {}, {}, {}
    terminal_seats, cancelled_seats = set(), set()
    match_seats, summarized_seats = {}, set()
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopped.set)
    backoff = 10
    try:
        while not stopped.is_set():
            try:
                response = await control.request("GET", "/tournament/agent/assignments")
                assignments = response.get("assignments")
                if not isinstance(assignments, list) or any(
                    not isinstance(a, dict) for a in assignments
                ):
                    raise PlatformError("invalid_assignments")
                backoff = 10
            except PlatformError as error:
                if not transient(error):
                    raise
                print(
                    "Assignment service temporarily unavailable; existing seats remain running.",
                    flush=True,
                )
                try:
                    await asyncio.wait_for(stopped.wait(), backoff)
                except TimeoutError:
                    pass
                backoff = min(60, backoff * 2)
                continue
            listed = {a.get("seat_id") for a in assignments}
            for seat_id, task in list(workers.items()):
                if task.done():
                    if not task.cancelled():
                        finished = task.result()
                        if not finished.get("deferred"):
                            terminal_seats.add(seat_id)
                    del workers[seat_id]
                    retry_at[seat_id] = time.monotonic() + 35
                elif seat_id not in listed:
                    missing[seat_id] = missing.get(seat_id, 0) + 1
                    if missing[seat_id] >= 2 and seat_id not in cancelled_seats:
                        task.cancel()
                        cancelled_seats.add(seat_id)
                else:
                    missing[seat_id] = 0
            for assignment in assignments:
                spec = lineup.route(assignment)
                if spec is None or (
                    match_id is not None and assignment.get("match_id") != match_id
                ):
                    continue
                current_match, seat_id = assignment["match_id"], assignment["seat_id"]
                if (
                    current_match not in played
                    and max_matches is not None
                    and len(played) >= max_matches
                ):
                    continue
                if (
                    seat_id in terminal_seats
                    or seat_id in workers
                    or time.monotonic() < retry_at.get(seat_id, 0)
                ):
                    continue
                if (
                    runtime_workspace is not None
                    and running_version(runtime_workspace) != runner_version
                ) or running_version(spec.workspace) != versions[spec.name]:
                    raise ValueError(
                        "Running code changed during the batch; refusing a new seat process."
                    )
                if current_match not in known_lineups:
                    known_lineups[current_match] = lineup.details(
                        [a for a in assignments if a.get("match_id") == current_match],
                        versions,
                    )
                played.add(current_match)
                match_seats.setdefault(current_match, set()).add(seat_id)
                kind, details = known_lineups[current_match]
                workers[seat_id] = asyncio.create_task(
                    play_seat(
                        spec,
                        assignment,
                        control,
                        execution_id,
                        versions[spec.name],
                        runner_version,
                        kind,
                        details,
                        stopped,
                    )
                )
                cancelled_seats.discard(seat_id)
                print(
                    f"Started seat {assignment['seat_position']} with adapter {spec.name}.",
                    flush=True,
                )
            if keep_alive:
                finalized = {
                    match
                    for match, seats in match_seats.items()
                    if seats & (terminal_seats - summarized_seats)
                }
                if finalized:
                    summarize(lineup, finalized, versions, redactor)
                    summarized_seats.update(terminal_seats)
            if (
                not keep_alive
                and played
                and not workers
                and not any(
                    a.get("match_id") in played
                    and a.get("seat_id") not in terminal_seats
                    and a.get("match_status")
                    in ("starting", "in_progress", "pending", "lobby")
                    for a in assignments
                )
            ):
                return
            try:
                await asyncio.wait_for(stopped.wait(), 10)
            except TimeoutError:
                pass
    finally:
        stopped.set()
        for seat_id, task in workers.items():
            if seat_id not in cancelled_seats:
                task.cancel()
        finished = await asyncio.gather(*workers.values(), return_exceptions=True)
        for seat_id, result in zip(workers, finished, strict=True):
            if isinstance(result, dict) and not result.get("deferred"):
                terminal_seats.add(seat_id)
        await control.close()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        if keep_alive:
            finalized = {
                match
                for match, seats in match_seats.items()
                if seats & (terminal_seats - summarized_seats)
            }
            if finalized:
                summarize(lineup, finalized, versions, redactor)
        elif played:
            summarize(lineup, played, versions, redactor)
