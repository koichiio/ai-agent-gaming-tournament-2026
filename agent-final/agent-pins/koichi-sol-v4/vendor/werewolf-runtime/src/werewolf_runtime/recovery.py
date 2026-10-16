"""Explicit, seat-authorized finalization of original orphan attempt journals.

Recovery never starts an adapter or connects via get_game_state. Runtime writers
must hold seat_lock for the whole seat attempt, including finalization.
"""

import fcntl
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4

from .audit import (
    _SHA,
    Audit,
    Redactor,
    _directory,
    _locked,
    _open,
    _publication_rows,
    _PublicationError,
    _published_attempt,
    _read_events,
    _seal_journals,
    _source_identity,
    _source_metadata,
    _validate_finish_fields,
    _validate_recovery_attempts,
    append_experiment,
)
from .platform import Control, PlatformError, SeatConnection, terminal_status


class SeatBusyError(RuntimeError):
    """The exact seat already has a writer or recovery holding its lock."""


def _match_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Recovery requires a nonempty platform match ID.")
    if not value.startswith("playground:"):
        if value.startswith(("local:", "self-play:")):
            raise ValueError("Recovery only supports platform matches.")
        value = "playground:" + value
    if not value.removeprefix("playground:").strip():
        raise ValueError("Recovery requires a nonempty platform match ID.")
    return value


def _workspace(path: Path) -> Path:
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Recovery paths must not contain symlinks.")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Recovery requires an existing workspace directory.")
    return path


@contextmanager
def seat_lock(
    workspace: Path, match_id: str, agent: str, player_id: str
) -> Iterator[None]:
    """Nonblocking process lock shared by play_seat and explicit recovery."""
    workspace = _workspace(workspace)
    match_id = _match_id(match_id)
    if any(
        not isinstance(value, str) or not value.strip() or "\0" in value
        for value in (match_id, agent, player_id)
    ):
        raise ValueError("Recovery seat identity must be nonempty and unambiguous.")
    identity = "\0".join((match_id, agent, player_id))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    _directory(workspace / ".private")
    locks = workspace / ".private" / "recovery-locks"
    _directory(locks)
    with _open(locks / (digest + ".lock"), "a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SeatBusyError("This seat has an active writer or recovery.") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _archive(workspace: Path, log_path: str) -> dict:
    path = Path(log_path) if isinstance(log_path, str) else Path()
    if (
        path.is_absolute()
        or len(path.parts) != 3
        or path.parts[:2] != ("data", "logs")
        or path.suffix != ".json"
    ):
        raise ValueError("Invalid finalized archive path.")
    for parent in (workspace / "data", workspace / "data" / "logs"):
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("Invalid finalized archive directory.")
    with _open(workspace / path, "r") as stream:
        archive = json.load(stream)
    if not isinstance(archive, dict):
        raise ValueError("Invalid finalized archive.")
    try:
        _validate_finish_fields(archive)
    except ValueError, TypeError:
        raise _PublicationError() from None
    return archive


def _original_experiment(
    workspace: Path, metadata: dict, experiment: dict, redactor: Redactor
) -> dict | None:
    """Read under the caller's audit lock and reject conflicting published details."""
    data = workspace / "data"
    if data.is_symlink():
        raise ValueError("Recovery data directory must not be a symlink.")
    previous = None
    for row in _publication_rows(workspace, "experiments"):
        if row["experiment_id"] == metadata["experiment_id"] and (
            previous is None or row.get("revision", 1) >= previous.get("revision", 1)
        ):
            previous = row
    if previous is not None:
        _validate_recovery_attempts(
            previous.get("recovery_attempts", []), complete=True
        )
        if previous.get("version") != metadata["version"]:
            raise ValueError("Original experiment ID belongs to another version.")
        expected = {
            **experiment,
            "baseline_version": experiment["baseline_version"] or None,
        }
        if any(
            redactor.clean(previous.get(key)) != redactor.clean(expected[key])
            for key in ("baseline_version", "changes", "hypothesis")
        ):
            raise ValueError(
                "Supplied experiment differs from original published details."
            )
    return previous


def _summary(
    workspace: Path,
    metadata: dict,
    experiment: dict,
    finished: dict,
    redactor: Redactor,
) -> None:
    """Publish original caller-authored experiment details, retrying append races."""
    for _ in range(5):
        with _locked(workspace):
            previous = _original_experiment(workspace, metadata, experiment, redactor)
        recovered = deepcopy((previous or {}).get("recovery_attempts", []))
        if any(row.get("attempt_id") == metadata["attempt_id"] for row in recovered):
            return
        result = finished.get("result") or {}
        recovered.append(
            {
                "attempt_id": metadata["attempt_id"],
                "match_id": metadata["match_id"],
                "agent": metadata["agent"],
                "player_id": metadata["player_id"],
                "lineup_type": metadata["lineup_type"],
                "log_path": finished["log_path"],
                "status": finished["status"],
                "outcome": finished["outcome"],
                "incomplete_or_unknown": finished["status"] != "completed"
                or finished["outcome"] == "unknown",
                "result_log_path": result.get("log_path"),
            }
        )
        summary = {
            **(previous or experiment),
            "version": metadata["version"],
            "baseline_version": experiment["baseline_version"] or None,
            "match_ids": sorted(
                set((previous or {}).get("match_ids", [])) | {metadata["match_id"]}
            ),
            "findings": (previous or {}).get(
                "findings",
                {
                    "recovery_only": True,
                    "note": "Orphan attempt recovery is not evidence of strategy improvement.",
                },
            ),
            "recovery_attempts": recovered,
            "recovery_findings": {
                "attempts": len(recovered),
                "games": len({row["match_id"] for row in recovered}),
                "incomplete_or_unknown": sum(
                    row["incomplete_or_unknown"] for row in recovered
                ),
                "outcomes_by_lineup": {
                    kind: [
                        row["outcome"]
                        for row in recovered
                        if row.get("lineup_type", "unknown") == kind
                    ]
                    for kind in ("homogeneous", "mixed", "unknown")
                },
            },
            "limitations": (previous or {}).get(
                "limitations",
                "Recovery-only observations; incomplete and unknown outcomes are "
                "not losses. No win-rate or improvement estimate; correlated seats "
                "are not independent games.",
            ),
            "recommendation": "retest",
        }
        if previous is not None:
            summary.update(
                revision=previous.get("revision", 1) + 1,
                correction_reason="Additional authorized orphan attempt recovery.",
            )
        try:
            append_experiment(workspace, summary, redactor)
            return
        except _PublicationError:
            raise
        except RuntimeError:
            # append_experiment rejects stale revisions under its process lock.
            # Re-read before retrying rather than overwriting another seat's summary.
            continue
    raise RuntimeError("Unable to publish recovery experiment summary.")


async def recover_attempt(
    workspace: Path,
    agent: str,
    attempt_id: str,
    match_id: str,
    player_id: str,
    official_key_value: str,
    experiment: dict | None,
    confirmed_ended: bool = False,
    reason: str | None = None,
    *,
    redactor: Redactor | None = None,
) -> dict:
    """Recover exactly one original attempt, never inferring session termination.

    Supply original experiment details (experiment_id, baseline_version, changes,
    hypothesis) if they were not persisted in configuration.experiment. A supplied
    redactor should already register model credentials; platform tokens are added
    automatically. Operator attestation applies only after successful assignment
    retrieval proves the original assignment unavailable, never to active results
    or authentication/network errors. Repeated finalized recovery is local-only.
    Corrupted public ledgers require operator intervention, never automatic repair.
    """
    try:
        if not isinstance(attempt_id, str) or str(UUID(attempt_id)) != attempt_id:
            raise ValueError
    except ValueError, AttributeError:
        raise ValueError("Recovery requires an exact canonical attempt UUID.") from None
    if type(confirmed_ended) is not bool or (
        confirmed_ended and (not isinstance(reason, str) or not reason.strip())
    ):
        raise ValueError(
            "Operator termination confirmation requires a nonempty reason."
        )
    redactor = redactor or Redactor([])
    redactor.add_secret(official_key_value)
    workspace, match_id = _workspace(workspace), _match_id(match_id)
    try:
        with seat_lock(workspace, match_id, agent, player_id):
            live = workspace / ".private" / "live"
            source = live / attempt_id
            if any(path.is_symlink() or not path.is_dir() for path in (live, source)):
                raise ValueError("Original attempt directory is unavailable or unsafe.")
            journal = source / "events.jsonl"
            with _locked(workspace), _open(journal, "r", errors="replace") as stream:
                header_line = stream.readline()
                if "\ufffd" in header_line:
                    raise ValueError(
                        "Original journal metadata contains unreadable text."
                    )
                header = json.loads(header_line)
                metadata = header.get("data") if isinstance(header, dict) else None
                if (
                    not isinstance(metadata, dict)
                    or header.get("event") != "start"
                    or metadata.get("kind") != "match"
                    or any(
                        metadata.get(key) != value
                        for key, value in (
                            ("attempt_id", attempt_id),
                            ("match_id", match_id),
                            ("agent", agent),
                            ("player_id", player_id),
                        )
                    )
                ):
                    raise ValueError(
                        "Original journal does not match the requested attempt/seat."
                    )
                _source_metadata(metadata)
                if metadata["test_mode"] != "playground":
                    raise ValueError(
                        "Original journal lacks valid reproducibility metadata."
                    )
                persisted = metadata["configuration"].get("experiment")
                if persisted is not None and not isinstance(persisted, dict):
                    raise ValueError("Invalid original experiment configuration.")
                experiment = deepcopy(
                    experiment if experiment is not None else persisted
                )
                required = {
                    "experiment_id",
                    "baseline_version",
                    "changes",
                    "hypothesis",
                }
                if (
                    not isinstance(experiment, dict)
                    or not required.issubset(experiment)
                    or experiment["experiment_id"] != metadata.get("experiment_id")
                    or not isinstance(experiment["experiment_id"], str)
                    or not experiment["experiment_id"].strip()
                    or (
                        persisted is not None
                        and any(
                            experiment[key] != persisted.get(key) for key in required
                        )
                    )
                    or (
                        experiment["baseline_version"] not in (None, "")
                        and (
                            not isinstance(experiment["baseline_version"], str)
                            or not _SHA.fullmatch(experiment["baseline_version"])
                        )
                    )
                    or (
                        "version" in experiment
                        and experiment["version"] != metadata["version"]
                    )
                ):
                    raise ValueError(
                        "Supply the original attempt's experiment details and ID."
                    )
                _original_experiment(workspace, metadata, experiment, redactor)
                _publication_rows(workspace, "results")
                # Identity is checked before advancing past the metadata line.
                remaining, integrity = _read_events(stream)
                events = [header, *remaining]
                recorded_identity = _source_identity(metadata, events)

            pending = None
            for entry in events:
                if entry.get("event") == "finish_deferred":
                    data = entry.get("data")
                    if not isinstance(data, dict) or any(
                        data.get(key) != metadata[key]
                        for key in (
                            "attempt_id",
                            "match_id",
                            "agent",
                            "player_id",
                            "version",
                        )
                    ):
                        raise ValueError(
                            "Deferred finish does not match the original attempt."
                        )
                    pending = data.get("pending_finish")
                    if not isinstance(pending, dict) or pending.get("status") not in (
                        "completed",
                        "failed",
                        "aborted",
                    ):
                        raise ValueError("Invalid original deferred finish.")
            status = (pending or {}).get("status", "failed")
            original_reason = (pending or {}).get("reason")
            if pending is None:
                original_reason = (
                    "Recovery observed an orphan journal without a finish event; "
                    "the original process did not record completion (possible hard crash)."
                )
            role = (pending or {}).get("role")
            finished = None
            with _locked(workspace):
                results = _publication_rows(workspace, "results")
                marker = source / "finalized.json"
                published = _published_attempt(workspace, metadata, redactor, results)
                sealed = None
                if marker.exists() or marker.is_symlink():
                    with _open(marker, "r") as stream:
                        sealed = json.load(stream)
                elif published is not None:
                    sealed = {"log_path": published["log_path"]}
                if sealed is not None:
                    archive = _archive(workspace, sealed["log_path"])
                    identity = ("match_id", "agent", "player_id")
                    if any(archive.get(key) != metadata[key] for key in identity) or (
                        archive.get("match_termination_verified") is not True
                    ):
                        raise ValueError(
                            "Finalized archive does not verify this seat's termination."
                        )
                    original = (
                        archive
                        if archive.get("attempt_id") == attempt_id
                        else next(
                            (
                                row["metadata"]
                                for row in archive.get("recovered_attempts", [])
                                if row.get("metadata", {}).get("attempt_id")
                                == attempt_id
                            ),
                            None,
                        )
                    )
                    # Top-level archives contain finish fields in addition to metadata.
                    if not isinstance(original, dict) or any(
                        redactor.clean(original.get(key)) != redactor.clean(value)
                        for key, value in metadata.items()
                    ):
                        raise ValueError(
                            "Finalized archive does not preserve the original metadata."
                        )
                    result = None
                    for row in results:
                        if all(row[key] == metadata[key] for key in identity) and (
                            result is None
                            or row.get("revision", 1) >= result.get("revision", 1)
                        ):
                            result = row
                    if result is None or published is None:
                        raise RuntimeError(
                            "Public audit cannot validate the archived result; operator intervention required."
                        )
                    finished = {
                        "log_path": sealed["log_path"],
                        "result": result,
                        "status": status,
                        "outcome": "unknown",
                    }
                    if archive.get("attempt_id") == attempt_id:
                        finished.update(
                            {
                                key: archive.get(key)
                                for key in (
                                    "timestamp",
                                    "status",
                                    "outcome",
                                    "won",
                                    "role",
                                    "reason",
                                )
                            }
                        )
                    _seal_journals([journal], archive["timestamp"], sealed["log_path"])

            if finished is None:
                assignments = [
                    entry["data"]
                    for entry in events
                    if entry.get("event") == "assignment"
                ]
                if any(not isinstance(row, dict) for row in assignments) or (
                    not assignments and not confirmed_ended
                ):
                    raise ValueError("Original assignment identity was not recorded.")
                assignment = (
                    assignments[0]
                    if assignments
                    else {
                        "seat_id": player_id,
                        "match_id": match_id.removeprefix("playground:"),
                    }
                )
                fields = (
                    "seat_id",
                    "match_id",
                    "seat_position",
                    "game_type",
                    "context",
                    "seat_count",
                    "game_session_id",
                    "session_id",
                    "agent_id",
                )
                # Bind known session identities from original authorized observations.
                sessions = (
                    {recorded_identity["session"]}
                    if "session" in recorded_identity
                    else set()
                )
                platform_agent = recorded_identity.get("platform_agent")
                control = Control(official_key_value, redactor)
                connection = None
                final = None
                source_kind = "operator_confirmation"
                try:
                    await control.authenticate()
                    response = await control.request(
                        "GET", "/tournament/agent/assignments"
                    )
                    current = response.get("assignments")
                    if not isinstance(current, list) or any(
                        not isinstance(row, dict) for row in current
                    ):
                        raise PlatformError("invalid_assignments")
                    matches = [
                        row
                        for row in current
                        if row.get("seat_id") == player_id
                        and row.get("match_id") == assignment["match_id"]
                    ]
                    if not matches:
                        if not confirmed_ended:
                            raise ValueError(
                                "Original assignment unavailable; termination is unverified."
                            )
                        final = {"recovery": {"source": source_kind, "reason": reason}}
                    else:
                        if not assignments:
                            raise ValueError(
                                "Original assignment identity was not recorded; cannot grant a listed seat."
                            )
                        if len(matches) != 1 or any(
                            matches[0].get(key) != assignment[key]
                            for key in fields
                            if key in assignment
                        ):
                            raise ValueError(
                                "Current assignment differs from the original seat configuration."
                            )
                        connection = SeatConnection(
                            control, matches[0], str(uuid4()), redactor
                        )
                        await connection.refresh()
                        grant = connection.grant
                        if (
                            not isinstance(grant, dict)
                            or any(
                                grant.get(key) != assignment[key]
                                for key in (
                                    "seat_id",
                                    "match_id",
                                    "seat_position",
                                    "game_type",
                                    "agent_id",
                                    "game_session_id",
                                )
                                if key in assignment
                            )
                            or (
                                sessions
                                and grant.get("game_session_id") not in sessions
                            )
                            or (
                                platform_agent is not None
                                and grant.get("agent_id") != platform_agent
                            )
                        ):
                            raise ValueError(
                                "Recovery grant differs from the original seat/session identity."
                            )
                        try:
                            final = await connection.call("get_result")
                        except PlatformError as error:
                            if error.code != "SESSION_NOT_FOUND":
                                raise
                            source_kind = "authorized_session_not_found"
                            final = {
                                "recovery": {"source": source_kind, "code": error.code}
                            }
                        else:
                            if (
                                not isinstance(final, dict)
                                or final.get("session_id") != grant["game_session_id"]
                            ):
                                raise ValueError(
                                    "Recovery result does not match the granted session."
                                )
                            if final.get("is_terminal") is not True:
                                raise ValueError(
                                    "Recovery rejected: the platform session is still active."
                                )
                            source_kind = "platform_result"
                            final = {**final, "recovery": {"source": source_kind}}
                finally:
                    try:
                        if connection is not None:
                            await connection.close()
                    finally:
                        await control.close()

                # Rehydrate, never create a new journal or label it with current HEAD.
                audit = Audit.__new__(Audit)
                audit._workspace, audit._journal = workspace, journal
                audit._metadata, audit._kind = metadata, "match"
                audit._redactor, audit._mutex, audit._finished = redactor, RLock(), None
                own_return = final.get("your_return")
                outcome = "unknown"
                if source_kind == "platform_result" and status == "completed":
                    status, original_reason = terminal_status(final)
                if (
                    source_kind == "platform_result"
                    and status == "completed"
                    and type(own_return) in (int, float)
                ):
                    outcome = {1: "win", -1: "loss", 0: "draw"}.get(
                        own_return, "unknown"
                    )
                if source_kind != "platform_result" and status == "completed":
                    status = "failed"
                    original_reason = "Original deferred completion could not be verified during recovery."
                final["recovery"].update(attempt_id=attempt_id, **integrity)
                finished = audit.finish(
                    status,
                    outcome=outcome,
                    role=role,
                    reason=original_reason,
                    final=final,
                    match_terminated=True,
                )
            _summary(workspace, metadata, experiment, finished, redactor)
            return redactor.clean(finished)
    except PlatformError as error:
        code = redactor.clean(error.code)
        raise RuntimeError(
            f"Recovery platform verification failed: {code} (HTTP {error.status})."
        ) from None
    except (
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        UnicodeError,
        RecursionError,
    ):
        raise RuntimeError(
            "Recovery could not read or validate the original audit data."
        ) from None
