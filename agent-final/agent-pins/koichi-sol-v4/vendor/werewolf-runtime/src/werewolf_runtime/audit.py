"""Neutral private attempt journals and terminal-only, sanitized audit archives.

Callers must exclude ``.private`` from game-decision tools and isolate live
journals from other agents; Unix permissions do not isolate the same Unix user.
Worker termination alone never permits publishing match logs. Match attempts
stay private until actual session termination is verified by the caller.
The caller supplies the running commit SHA, configuration, IDs and outcomes;
this module never inspects Git, credentials, or game state to infer them.
"""

import fcntl
import json
import math
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import TextIO
from urllib.parse import unquote
from uuid import UUID, uuid4

_SHA = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
_SECRET_FIELDS = {
    "authorization",
    "proxyauthorization",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "token",
    "bearertoken",
    "authtoken",
    "apitoken",
    "xauthtoken",
    "apikey",
    "apikeys",
    "xapikey",
    "officialagentkey",
    "executionid",
    "executionids",
    "password",
    "passwd",
    "passphrase",
    "clientsecret",
    "secret",
    "secrets",
    "secretkey",
    "privatekey",
    "signingkey",
    "accesskey",
    "accesskeyid",
    "secretaccesskey",
    "auth",
    "credential",
    "credentials",
    "cookie",
    "cookies",
    "setcookie",
    "xamzcredential",
    "xamzsignature",
    "xamzsecuritytoken",
}
_JWT = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]+={0,2}\."
    r"[A-Za-z0-9_-]+={0,2}\.[A-Za-z0-9_-]*={0,2}(?![A-Za-z0-9_-])"
)
_LIVE_KEY = re.compile(r"(?i)(?<![A-Za-z0-9_])eak_live_[0-9a-f]+(?![A-Za-z0-9_])")
_API_KEY = re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_])")
_USERINFO = re.compile(r"(?i)((?:\b[a-z][a-z0-9+.-]*:)?//)[^/\s?#]*@")
_QUERY = re.compile(r"([?&#;])([^?&#;=\s<>\"']+)=(?:\"[^\"]*\"|'[^']*'|[^&#;\s<>\"']*)")
_AUTH = re.compile(r"(?i)\b(?:Bearer|Basic)\s+[^\s,;\"'<>]+")


class Redactor:
    """Produce JSON-safe data without rendering opaque objects or exceptions."""

    def __init__(self, secrets: list[str]) -> None:
        if not isinstance(secrets, list) or any(
            not isinstance(secret, str) for secret in secrets
        ):
            raise ValueError("Secrets must be a list of strings.")
        try:
            known = sorted(
                {secret for secret in secrets if secret}, key=len, reverse=True
            )
            self._secrets = set(known)
            self._secret_mutex = RLock()
            self._known = (
                re.compile("|".join(re.escape(secret) for secret in known))
                if known
                else None
            )
        except Exception:
            raise ValueError("Unable to initialize redaction.") from None

    def add_secret(self, secret: str) -> None:
        """Add a renewed token before storage, without output or discarding old tokens."""
        if not isinstance(secret, str):
            raise ValueError("Secret must be a string.")
        if not secret:
            return
        try:
            with self._secret_mutex:
                known = self._secrets | {secret}
                pattern = re.compile(
                    "|".join(
                        re.escape(value)
                        for value in sorted(known, key=len, reverse=True)
                    )
                )
                self._secrets = known
                self._known = pattern
        except Exception:
            raise ValueError("Unable to update redaction.") from None

    @staticmethod
    def _sensitive(key: str) -> bool:
        name = re.sub(r"[^a-z0-9]", "", key.casefold())
        return name in _SECRET_FIELDS or name.endswith(
            ("apikey", "accesstoken", "refreshtoken", "clientsecret", "privatekey")
        )

    def _text(self, text: str) -> str:
        if self._known is not None:
            text = self._known.sub("[REDACTED]", text)
        text = _USERINFO.sub(r"\1[REDACTED]@", text)
        text = _QUERY.sub(
            lambda match: (
                f"{match[1]}{match[2]}=[REDACTED]"
                if self._sensitive(unquote(match[2]))
                or unquote(match[2]).casefold() in ("key", "sig", "signature")
                else match[0]
            ),
            text,
        )
        text = _JWT.sub("[REDACTED]", text)
        text = _LIVE_KEY.sub("[REDACTED]", text)
        text = _API_KEY.sub("[REDACTED]", text)
        return _AUTH.sub("[REDACTED]", text)

    def clean(self, value: object) -> object:
        """Redact nested dictionaries/lists, preserving noncredential IDs.

        String keys and integer seat keys are supported; JSON stores the latter
        as strings. Cycles, opaque objects and nonfinite numbers get explicit
        placeholders rather than potentially secret reprs.
        """
        active: set[int] = set()

        def visit(item: object) -> object:
            if isinstance(item, str):
                return self._text(item)
            if item is None or isinstance(item, (bool, int)):
                return item
            if isinstance(item, float):
                return item if math.isfinite(item) else "[NON-FINITE]"
            if not isinstance(item, (dict, list, tuple)):
                return "[UNSUPPORTED]"
            identity = id(item)
            if identity in active:
                return "[CYCLE]"
            active.add(identity)
            try:
                if isinstance(item, dict):
                    cleaned = {}
                    for key, child in item.items():
                        if type(key) is int:
                            key = str(key)
                        if not isinstance(key, str):
                            raise ValueError("Audit dictionary keys must be strings.")
                        cleaned[self._text(key)] = (
                            "[REDACTED]" if self._sensitive(key) else visit(child)
                        )
                    return cleaned
                return [visit(child) for child in item]
            finally:
                active.remove(identity)

        try:
            return visit(value)
        except Exception:
            raise ValueError("Unable to redact audit data.") from None


def _directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("Audit directories must not be symlinks.")
    path.mkdir(mode=0o700, exist_ok=True)
    path.chmod(0o700)


def _open(path: Path, mode: str, *, errors: str = "strict") -> TextIO:
    flags = {
        "r": os.O_RDONLY,
        "x": os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        "a": os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        "a+": os.O_RDWR | os.O_CREAT | os.O_APPEND,
    }[mode]
    descriptor = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, mode, encoding="utf-8", errors=errors)
    except Exception:
        os.close(descriptor)
        raise


def _write(stream: TextIO, value: object) -> None:
    stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _read_events(stream: TextIO) -> tuple[list, dict]:
    """Retain readable entries without repairing a crash-damaged journal."""
    events, unreadable, truncated = [], 0, 0
    for line in stream:
        if not line.strip():
            continue
        try:
            if "\ufffd" in line:
                raise ValueError
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError
            events.append(entry)
        except ValueError, RecursionError:
            unreadable += 1
            truncated += not line.endswith("\n")
    integrity = {
        "unreadable_event_lines": unreadable,
        "truncated_event_lines": truncated,
    }
    if unreadable:
        integrity["journal_read_reason"] = (
            "Unreadable journal lines were omitted; readable events were retained "
            "without changing the original journal."
        )
    return events, integrity


class _PublicationError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "Public audit journal cannot be validated; operator intervention required. "
            "Original publication records were not repaired or rewritten."
        )


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]+)?Z",
        value,
    ):
        return False
    try:
        return datetime.fromisoformat(value).tzinfo is UTC
    except ValueError:
        return False


def _validate_finish_fields(row: dict) -> None:
    if (
        not {"timestamp", "status", "outcome", "won", "role"}.issubset(row)
        or not _valid_timestamp(row["timestamp"])
        or row["status"] not in ("completed", "failed", "aborted")
        or row["outcome"] not in ("win", "loss", "draw", "unknown")
        or row["won"] is not {"win": True, "loss": False}.get(row["outcome"])
        or (row["status"] != "completed" and row["outcome"] != "unknown")
        or (
            row["role"] is not None
            and (not isinstance(row["role"], str) or not row["role"].strip())
        )
        or (row.get("reason") is not None and not isinstance(row["reason"], str))
    ):
        raise ValueError("Invalid published finish fields.")


def _validate_recovery_attempts(value: object, *, complete: bool = False) -> None:
    """Validate available evidence without inferring missing historical fields."""
    if not isinstance(value, list):
        raise _PublicationError()
    for row in value:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("attempt_id", "match_id", "outcome")
        ):
            raise _PublicationError()
        status, outcome = row.get("status"), row["outcome"]
        if (
            outcome not in ("win", "loss", "draw", "unknown")
            or (
                (complete or "status" in row or outcome != "unknown")
                and status not in ("completed", "failed", "aborted")
            )
            or (status in ("failed", "aborted") and outcome != "unknown")
            or (
                "won" in row
                and row["won"] is not {"win": True, "loss": False}.get(outcome)
            )
            or (
                "lineup_type" in row
                and row["lineup_type"] not in ("homogeneous", "mixed", "unknown")
            )
            or (
                (complete or "incomplete_or_unknown" in row)
                and (
                    status not in ("completed", "failed", "aborted")
                    or type(row.get("incomplete_or_unknown")) is not bool
                    or row["incomplete_or_unknown"]
                    is not (status != "completed" or outcome == "unknown")
                )
            )
            or ("timestamp" in row and not _valid_timestamp(row["timestamp"]))
        ):
            raise _PublicationError()


def _publication_rows(workspace: Path, kind: str) -> list[dict]:
    """Fail closed on damaged append-only ledgers, without exposing raw fragments."""
    path = workspace / "data" / (kind + ".jsonl")
    rows = []
    try:
        if (workspace / "data").is_symlink() or path.is_symlink():
            raise ValueError
        if not path.exists():
            return rows
        required = (
            {
                "timestamp",
                "match_id",
                "agent",
                "player_id",
                "version",
                "strategy",
                "test_mode",
                "lineup_type",
                "experiment_id",
                "role",
                "status",
                "outcome",
                "won",
                "log_path",
            }
            if kind == "results"
            else {
                "experiment_id",
                "version",
                "baseline_version",
                "changes",
                "hypothesis",
                "match_ids",
                "findings",
                "limitations",
                "recommendation",
            }
        )
        revisions, versions = {}, {}
        with _open(path, "r") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    raise ValueError
                if not line.strip():
                    continue
                row = json.loads(line)
                json.dumps(row, allow_nan=False)
                if not isinstance(row, dict) or not required.issubset(row):
                    raise ValueError
                if any(
                    not isinstance(row.get(key), str) or not row[key].strip()
                    for key in ("experiment_id", "version")
                ) or not _SHA.fullmatch(row["version"]):
                    raise ValueError
                revision = row.get("revision", 1)
                if (
                    type(revision) is not int
                    or revision < 1
                    or (
                        revision > 1
                        and (
                            not isinstance(row.get("correction_reason"), str)
                            or not row["correction_reason"].strip()
                        )
                    )
                ):
                    raise ValueError
                if kind == "results":
                    if (
                        any(
                            not isinstance(row[key], str) or not row[key].strip()
                            for key in (
                                "timestamp",
                                "match_id",
                                "agent",
                                "player_id",
                                "strategy",
                                "log_path",
                            )
                        )
                        or row["test_mode"] not in ("local", "playground", "self-play")
                        or row["lineup_type"] not in ("homogeneous", "mixed", "unknown")
                    ):
                        raise ValueError
                    _validate_finish_fields(row)
                    identity = tuple(
                        row[key] for key in ("match_id", "agent", "player_id")
                    )
                else:
                    baseline = row["baseline_version"]
                    if (
                        (
                            baseline is not None
                            and (
                                not isinstance(baseline, str)
                                or not _SHA.fullmatch(baseline)
                            )
                        )
                        or not isinstance(row["match_ids"], list)
                        or any(
                            not isinstance(value, str) or not value.strip()
                            for value in row["match_ids"]
                        )
                        or row["recommendation"] not in ("adopt", "reject", "retest")
                    ):
                        raise ValueError
                    identity = (row["experiment_id"],)
                    if identity in versions and versions[identity] != row["version"]:
                        raise ValueError
                    versions[identity] = row["version"]
                    if "recovery_attempts" in row:
                        _validate_recovery_attempts(row["recovery_attempts"])
                key = (*identity, revision)
                if key in revisions:
                    raise ValueError
                revisions[key] = True
                rows.append(row)
        return rows
    except OSError, ValueError, TypeError, UnicodeError, RecursionError:
        raise _PublicationError() from None


def _source_metadata(metadata: dict) -> None:
    """Validate original match fields; persisted experiment details are optional."""
    if (
        any(
            not isinstance(metadata.get(key), str) or not metadata[key].strip()
            for key in (
                "attempt_id",
                "match_id",
                "agent",
                "player_id",
                "strategy",
                "experiment_id",
                "version",
            )
        )
        or metadata.get("kind") != "match"
        or not _SHA.fullmatch(metadata["version"])
        or not isinstance(metadata.get("configuration"), dict)
        or metadata.get("lineup_type") not in ("homogeneous", "mixed", "unknown")
        or metadata.get("test_mode") not in ("local", "playground", "self-play")
    ):
        raise ValueError("Invalid original reproducibility metadata.")
    if str(UUID(metadata["attempt_id"])) != metadata["attempt_id"]:
        raise ValueError("Invalid original attempt identity.")
    experiment = metadata["configuration"].get("experiment")
    if experiment is not None:
        if (
            not isinstance(experiment, dict)
            or not {
                "experiment_id",
                "baseline_version",
                "changes",
                "hypothesis",
            }.issubset(experiment)
            or experiment["experiment_id"] != metadata["experiment_id"]
            or experiment.get("version", metadata["version"]) != metadata["version"]
        ):
            raise ValueError("Invalid original experiment configuration.")
        baseline = experiment["baseline_version"]
        if baseline not in (None, "") and (
            not isinstance(baseline, str) or not _SHA.fullmatch(baseline)
        ):
            raise ValueError("Invalid original experiment version.")


def _source_identity(metadata: dict, events: list[dict]) -> dict:
    """Bind all recorded assignment/state/grant/session identities before collection."""
    identity = {}
    assignment = None
    for entry in events:
        event, data = entry.get("event"), entry.get("data")
        if event not in ("assignment", "state", "result", "grant", "finish_deferred"):
            continue
        if not isinstance(data, dict):
            raise ValueError("Invalid recorded seat identity.")
        if event == "assignment":
            if (
                data.get("seat_id") != metadata["player_id"]
                or data.get("match_id")
                != metadata["match_id"].removeprefix("playground:")
                or type(data.get("seat_position")) is not int
                or not 0 <= data["seat_position"] < 7
                or data.get("game_type") != "werewolf"
            ):
                raise ValueError("Conflicting recorded assignment.")
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
            if assignment is not None and any(
                data.get(key) != assignment.get(key) for key in fields
            ):
                raise ValueError("Conflicting recorded assignments.")
            assignment = data
            cfg = metadata["configuration"]
            if "assignment_context" in cfg and cfg["assignment_context"] != data.get(
                "context"
            ):
                raise ValueError("Conflicting recorded assignment context.")
        if event == "finish_deferred":
            if any(
                data.get(key) != metadata[key]
                for key in ("attempt_id", "match_id", "agent", "player_id", "version")
            ):
                raise ValueError("Conflicting deferred identity.")
            pending = data.get("pending_finish")
            if not isinstance(pending, dict) or pending.get("status") not in (
                "completed",
                "failed",
                "aborted",
            ):
                raise ValueError("Invalid deferred finish.")
            data = pending.get("final")
            if data is None:
                data = {}
            if not isinstance(data, dict):
                raise ValueError("Invalid deferred final identity.")
        for key, value in data.items():
            bound = {
                "session_id": "session",
                "game_session_id": "session",
                "seat_position": "position",
                "your_position": "position",
                "agent_id": "platform_agent",
                "seat_id": "seat",
                "player_id": "seat",
                "match_id": "match",
                "context": "context",
                "seat_count": "seat_count",
                "game_type": "game_type",
            }.get(key)
            if bound is None:
                continue
            if bound == "position":
                if type(value) is not int or not 0 <= value < 7:
                    raise ValueError("Invalid recorded seat position.")
            elif bound == "seat_count":
                if type(value) is not int or value != 7:
                    raise ValueError("Invalid recorded seat count.")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError("Invalid recorded session/seat identity.")
            if bound == "match":
                value = value.removeprefix("playground:")
            if bound in identity and identity[bound] != value:
                raise ValueError("Conflicting recorded session/seat identity.")
            identity[bound] = value
        if event == "state" and type(data.get("your_position")) is not int:
            raise ValueError("Unverifiable recorded state seat.")
        if event == "grant" and (
            assignment is None
            or any(
                data.get(key) != assignment.get(key)
                for key in ("seat_id", "match_id", "seat_position", "game_type")
            )
        ):
            raise ValueError("Unverifiable recorded grant seat.")
    if identity.get("seat", metadata["player_id"]) != metadata[
        "player_id"
    ] or identity.get(
        "match", metadata["match_id"].removeprefix("playground:")
    ) != metadata["match_id"].removeprefix("playground:"):
        raise ValueError("Conflicting original seat identity.")
    if assignment is None and identity:
        raise ValueError("Original assignment identity is unavailable.")
    return identity


def _published_attempt(
    workspace: Path, metadata: dict, redactor: Redactor, results: list[dict]
) -> dict | None:
    """Locate the exact archived attempt after a crash before finish markers."""
    identity = ("match_id", "agent", "player_id")
    same = [
        row for row in results if all(row[key] == metadata[key] for key in identity)
    ]
    if not same:
        return None
    result = max(same, key=lambda row: row.get("revision", 1))
    logs = workspace / "data" / "logs"
    if logs.is_symlink() or not logs.is_dir():
        raise _PublicationError()
    path = Path(result["log_path"])
    if (
        path.is_absolute()
        or len(path.parts) != 3
        or path.parts[:2] != ("data", "logs")
        or path.suffix != ".json"
    ):
        raise _PublicationError()
    try:
        _validate_finish_fields(result)
        with _open(workspace / path, "r") as stream:
            original = json.load(stream)
        if (
            not isinstance(original, dict)
            or original.get("match_termination_verified") is not True
            or any(
                original.get(key) != result[key]
                for key in (
                    *identity,
                    "version",
                    "strategy",
                    "experiment_id",
                    "lineup_type",
                    "test_mode",
                    "timestamp",
                    "status",
                    "outcome",
                    "won",
                    "role",
                )
            )
            or original.get("won") is not result["won"]
            or original.get("reason") != result.get("reason")
        ):
            raise ValueError
        _source_metadata(original)
        if original.get("log_path") != path.as_posix():
            raise ValueError
        recovered_archive = None
        for archive_path in [workspace / path, *sorted(logs.glob("*.json"))]:
            if archive_path == workspace / path:
                archive = original
            else:
                with _open(archive_path, "r") as stream:
                    archive = json.load(stream)
            if not isinstance(archive, dict):
                raise ValueError
            if any(archive.get(key) != metadata[key] for key in identity):
                continue
            if (
                archive.get("log_path")
                != archive_path.relative_to(workspace).as_posix()
            ):
                raise ValueError
            _validate_finish_fields(archive)
            source = (
                archive
                if archive.get("attempt_id") == metadata["attempt_id"]
                else next(
                    (
                        row.get("metadata")
                        for row in archive.get("recovered_attempts", [])
                        if row.get("metadata", {}).get("attempt_id")
                        == metadata["attempt_id"]
                    ),
                    None,
                )
            )
            if source is None:
                continue
            if archive.get("match_termination_verified") is not True or any(
                redactor.clean(source.get(key)) != redactor.clean(value)
                for key, value in metadata.items()
            ):
                raise ValueError
            if archive.get("attempt_id") == metadata["attempt_id"]:
                return archive
            if recovered_archive is None:
                recovered_archive = archive
        return recovered_archive
    except OSError, ValueError, TypeError, AttributeError, UnicodeError, RecursionError:
        raise _PublicationError() from None


def _seal_journals(journals: list[Path], timestamp: str, log_path: str) -> None:
    for journal in journals:
        marker = journal.parent / "finalized.json"
        if marker.is_symlink():
            raise ValueError("Unsafe finalized marker.")
        if marker.exists():
            continue
        temporary = journal.parent / f".finalized-{uuid4()}.json"
        with _open(temporary, "x") as stream:
            _write(stream, {"timestamp": timestamp, "log_path": log_path})
        os.link(temporary, marker, follow_symlinks=False)
        temporary.unlink()


@contextmanager
def _locked(workspace: Path) -> Iterator[None]:
    private = workspace / ".private"
    _directory(private)
    with _open(private / "audit.lock", "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class Audit:
    """One seat/attempt journal; all writers share the workspace's flock.

    ``match_id=None`` never creates a result or a synthetic match ID. Supply an
    explicit ``local:<uuid>`` for an unassigned match attempt, and a stable local
    instance ID if no platform player exists. Raw platform match IDs receive
    only a ``playground:`` prefix, never a seat-specific suffix.

    Non-match kinds, including ``online-integration``, archive diagnostics only,
    with unknown outcome and null role. Callers mark synthetic integration
    metadata explicitly. Experiment summaries are caller-authored separately.
    """

    def __init__(
        self,
        workspace: Path,
        version: str,
        player_id: str,
        match_id: str | None,
        strategy: str,
        configuration: dict,
        redactor: Redactor,
        experiment_id: str | None = None,
        kind: str = "match",
        lineup_type: str = "unknown",
        test_mode: str = "playground",
        *,
        agent_id: str,
    ) -> None:
        if not isinstance(version, str) or not _SHA.fullmatch(version):
            raise ValueError("A supplied full running Git SHA is required.")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (player_id, strategy, kind, agent_id)
        ):
            raise ValueError("Player, strategy, kind and agent IDs must be nonempty.")
        if not isinstance(configuration, dict):
            raise ValueError("Configuration must be a dictionary.")
        if lineup_type not in ("homogeneous", "mixed", "unknown"):
            raise ValueError("Invalid lineup type.")
        if kind == "match" and test_mode not in ("local", "playground", "self-play"):
            raise ValueError("Invalid match test mode.")
        if experiment_id is not None and (
            not isinstance(experiment_id, str) or not experiment_id.strip()
        ):
            raise ValueError("Experiment ID must be a nonempty string.")
        if kind == "match" and experiment_id is None:
            raise ValueError("Match attempts require a caller-supplied experiment ID.")
        if match_id is not None:
            if not isinstance(match_id, str) or not match_id.strip():
                raise ValueError("Match ID must be a nonempty string or null.")
            if not match_id.startswith(("playground:", "local:", "self-play:")):
                match_id = "playground:" + match_id
            if not match_id.partition(":")[2].strip():
                raise ValueError("A namespaced match ID requires an identifier.")

        self._redactor = redactor
        self._mutex = RLock()
        self._finished: dict | None = None
        self._kind = kind
        try:
            self._workspace = Path(workspace).resolve(strict=True)
            if not self._workspace.is_dir():
                raise ValueError("Workspace must be an existing directory.")
            attempt_id = str(uuid4())
            started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            self._metadata = redactor.clean(
                {
                    "started_at": started_at,
                    "attempt_id": attempt_id,
                    "kind": kind,
                    "version": version,
                    "agent": agent_id,
                    "player_id": player_id,
                    "match_id": match_id,
                    "strategy": strategy,
                    "configuration": configuration,
                    "test_mode": test_mode if kind == "match" else None,
                    "lineup_type": lineup_type,
                    "experiment_id": experiment_id or f"smoke:{attempt_id}",
                }
            )
            with _locked(self._workspace):
                live = self._workspace / ".private" / "live"
                _directory(live)
                journal = live / attempt_id
                _directory(journal)
                self._journal = journal / "events.jsonl"
                entry = redactor.clean(
                    {"timestamp": started_at, "event": "start", "data": self._metadata}
                )
                with _open(self._journal, "x") as stream:
                    _write(stream, entry)
        except Exception:
            raise RuntimeError("Unable to initialize private audit journal.") from None

    def record(self, event: str, data: object) -> None:
        """Append a redacted event before any storage, serialized with other writers."""
        with self._mutex:
            if self._finished is not None:
                raise RuntimeError("Audit has already finished.")
            if not isinstance(event, str) or not event.strip():
                raise ValueError("Event name must be a nonempty string.")
            try:
                entry = self._redactor.clean(
                    {
                        "timestamp": datetime.now(UTC)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "event": event,
                        "data": data,
                    }
                )
                with _locked(self._workspace), _open(self._journal, "a") as stream:
                    if (self._journal.parent / "finalized.json").exists():
                        raise RuntimeError(
                            "Private journal has already been finalized."
                        )
                    _write(stream, entry)
            except Exception:
                raise RuntimeError("Unable to append private audit event.") from None

    def finish(
        self,
        status: str,
        outcome: str = "unknown",
        role: str | None = None,
        reason: str | None = None,
        final: dict | None = None,
        match_terminated: bool = False,
    ) -> dict:
        """Publish match archives only after verified actual match termination.

        Every match status requires ``final['is_terminal'] is True`` or
        ``match_terminated=True``, meaning the caller verified the session ended
        or is gone. A worker exiting, failing, or being stopped is not enough.
        Otherwise a private ``finish_deferred`` event preserves the supplied
        metadata and pending finish arguments, and no public files are written.
        A deferred call does not seal the journal; call again after termination.

        Automatic terminal-state retrieval is not implemented. Only after
        verified termination, a new finalizer for the same assigned tuple can
        archive earlier unfinished journals, retaining each source's version and
        configuration rather than relabeling them as the current tested version.
        Other journals are inspected only for their redacted start metadata;
        their event bodies are read only after the exact match/agent/seat matches.
        Outcome is caller-supplied, never inferred from ``your_return`` or role;
        supply the assigned role or None. Failures retain the private journal.
        Public finalization is locally idempotent; duplicates return the latest
        existing result without inventing a correction, plus a unique archive.
        """
        with self._mutex:
            if self._finished is not None:
                return deepcopy(self._finished)
            if status not in ("completed", "aborted", "failed"):
                raise ValueError("Invalid audit status.")
            if final is not None and not isinstance(final, dict):
                raise ValueError("Final metadata must be a dictionary or null.")
            if type(match_terminated) is not bool:
                raise ValueError("Match termination flag must be a boolean.")
            termination_verified = match_terminated or (
                final is not None and final.get("is_terminal") is True
            )
            if status != "completed" or self._kind != "match":
                outcome = "unknown"
            if outcome not in ("win", "loss", "draw", "unknown"):
                raise ValueError("Invalid audit outcome.")
            if self._kind != "match":
                role = None
            if isinstance(role, str) and role.strip().casefold() == "unknown":
                role = None
            if role is not None and (not isinstance(role, str) or not role.strip()):
                raise ValueError("Role must be a known nonempty string or null.")
            if reason is not None and not isinstance(reason, str):
                raise ValueError("Reason must be a string or null.")
            try:
                now = datetime.now(UTC)
                attempt = self._redactor.clean(
                    {
                        "timestamp": now.isoformat().replace("+00:00", "Z"),
                        "status": status,
                        "outcome": outcome,
                        "won": {"win": True, "loss": False}.get(outcome),
                        "role": role,
                        "reason": reason,
                    }
                )
                final = self._redactor.clean(final)
                if self._kind == "match" and not termination_verified:
                    self.record(
                        "finish_deferred",
                        {
                            **self._metadata,
                            "pending_finish": {**attempt, "final": final},
                            "match_termination_verified": False,
                            "deferred": True,
                        },
                    )
                    return {
                        "log_path": self._journal.relative_to(
                            self._workspace
                        ).as_posix(),
                        "result": None,
                        "deferred": True,
                    }
                identity = self._metadata["match_id"] or self._metadata["attempt_id"]
                safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", identity)[:120]
                filename = f"{now:%Y%m%dT%H%M%SZ}_{safe_id}_{uuid4()}.json"
                log_path = (Path("data") / "logs" / filename).as_posix()
                result = None
                with _locked(self._workspace):
                    results = _publication_rows(self._workspace, "results")
                    _publication_rows(self._workspace, "experiments")
                    if self._kind == "match" and self._metadata["match_id"] is not None:
                        published = _published_attempt(
                            self._workspace, self._metadata, self._redactor, results
                        )
                        if (
                            published is not None
                            and published.get("attempt_id")
                            == self._metadata["attempt_id"]
                        ):
                            result = max(
                                (
                                    row
                                    for row in results
                                    if all(
                                        row[key] == self._metadata[key]
                                        for key in ("match_id", "agent", "player_id")
                                    )
                                ),
                                key=lambda row: row.get("revision", 1),
                            )
                            _seal_journals(
                                [self._journal],
                                published["timestamp"],
                                published["log_path"],
                            )
                            self._finished = {
                                **{key: published[key] for key in attempt},
                                "log_path": published["log_path"],
                                "result": self._redactor.clean(result),
                            }
                            return deepcopy(self._finished)
                    with _open(self._journal, "r", errors="replace") as stream:
                        header_line = stream.readline()
                        if "\ufffd" in header_line:
                            raise ValueError("Unreadable original journal identity.")
                        header = json.loads(header_line)
                        if (
                            not isinstance(header, dict)
                            or header.get("event") != "start"
                            or self._redactor.clean(header.get("data"))
                            != self._redactor.clean(self._metadata)
                        ):
                            raise ValueError(
                                "Original journal identity does not match."
                            )
                        remaining, integrity = _read_events(stream)
                        events = [header, *remaining]
                    original_identity = (
                        _source_identity(
                            self._metadata,
                            [*events, {"event": "result", "data": final}]
                            if final is not None
                            else events,
                        )
                        if self._kind == "match"
                        and self._metadata["match_id"] is not None
                        else {}
                    )
                    recovered_attempts = []
                    journals = [self._journal]
                    identity_fields = ("match_id", "agent", "player_id")
                    if self._kind == "match" and self._metadata["match_id"] is not None:
                        live = self._journal.parent.parent
                        _directory(live)
                        for source in sorted(live.iterdir()):
                            if (
                                source == self._journal.parent
                                or source.is_symlink()
                                or not source.is_dir()
                                or (source / "finalized.json").exists()
                                or (source / "finalized.json").is_symlink()
                            ):
                                continue
                            source_journal = source / "events.jsonl"
                            if (
                                source_journal.is_symlink()
                                or not source_journal.is_file()
                            ):
                                continue
                            with _open(source_journal, "r", errors="replace") as stream:
                                # Select the exact seat before reading any event body.
                                try:
                                    header_line = stream.readline()
                                    if "\ufffd" in header_line:
                                        continue
                                    header = json.loads(header_line)
                                except json.JSONDecodeError:
                                    continue
                                if not isinstance(header, dict):
                                    continue
                                metadata = header.get("data")
                                if (
                                    header.get("event") != "start"
                                    or not isinstance(metadata, dict)
                                    or metadata.get("kind") != "match"
                                    or metadata.get("attempt_id") != source.name
                                    or any(
                                        metadata.get(key) != self._metadata[key]
                                        for key in identity_fields
                                    )
                                ):
                                    continue
                                try:
                                    _source_metadata(metadata)
                                except ValueError:
                                    continue
                                if (
                                    _published_attempt(
                                        self._workspace,
                                        metadata,
                                        self._redactor,
                                        results,
                                    )
                                    is not None
                                ):
                                    continue
                                remaining, source_integrity = _read_events(stream)
                                source_events = [header, *remaining]
                                try:
                                    source_identity = _source_identity(
                                        metadata, source_events
                                    )
                                    if (
                                        "session" in source_identity
                                        and "session" not in original_identity
                                    ):
                                        continue
                                    if any(
                                        source_identity[key] != original_identity[key]
                                        for key in source_identity.keys()
                                        & original_identity.keys()
                                    ):
                                        continue
                                except ValueError:
                                    continue
                            recovered_attempts.append(
                                {
                                    "source_journal": source_journal.relative_to(
                                        self._workspace
                                    ).as_posix(),
                                    "source_start": header.get("timestamp"),
                                    "metadata": metadata,
                                    "events": source_events,
                                    "version_mismatch": metadata.get("version")
                                    != self._metadata["version"],
                                    "configuration_mismatch": self._redactor.clean(
                                        metadata.get("configuration")
                                    )
                                    != self._redactor.clean(
                                        self._metadata["configuration"]
                                    ),
                                    **source_integrity,
                                }
                            )
                            if len(source_events) <= 1:
                                recovered_attempts[-1]["trace_missing_reason"] = (
                                    "No readable caller events were available in this journal."
                                )
                            journals.append(source_journal)
                    data = self._workspace / "data"
                    _directory(data)
                    _directory(data / "logs")
                    result_path = data / "results.jsonl"
                    existing = None
                    if self._kind == "match" and self._metadata["match_id"] is not None:
                        fields = (
                            "match_id",
                            "agent",
                            "player_id",
                            "version",
                            "strategy",
                            "test_mode",
                            "lineup_type",
                            "experiment_id",
                        )
                        result = {key: self._metadata[key] for key in fields}
                        result.update(attempt)
                        result["log_path"] = log_path
                        for row in results:
                            if any(row[key] != result[key] for key in identity_fields):
                                continue
                            if existing is None or row.get(
                                "revision", 1
                            ) >= existing.get("revision", 1):
                                existing = row
                        if existing is not None:
                            result = self._redactor.clean(existing)
                        else:
                            result = self._redactor.clean(result)
                    archive = {
                        **self._metadata,
                        **attempt,
                        "log_path": log_path,
                        "events": events,
                        "final": final,
                        "recovered_attempts": recovered_attempts,
                        **integrity,
                        "match_termination_verified": termination_verified
                        if self._kind == "match"
                        else None,
                    }
                    if len(events) <= 1:
                        archive["trace_missing_reason"] = (
                            "No caller events were recorded in this journal."
                        )
                    with _open(self._workspace / log_path, "x") as stream:
                        _write(stream, self._redactor.clean(archive))
                    if result is not None and existing is None:
                        with _open(result_path, "a") as stream:
                            _write(stream, result)
                    _seal_journals(journals, attempt["timestamp"], log_path)
                self._finished = {**attempt, "log_path": log_path, "result": result}
                return deepcopy(self._finished)
            except _PublicationError:
                raise
            except Exception:
                raise RuntimeError(
                    "Unable to finish audit; private journal retained."
                ) from None


def validate_public_logs(workspace: Path, redactor: Redactor) -> dict:
    """Check sanitized public records without exposing credentials or raw traces."""
    logs, latest, problems = {}, {}, []
    for path in sorted((workspace / "data" / "logs").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(workspace).as_posix()
        logs[relative] = row
        if redactor.clean(row) != row:
            problems.append(f"Unredacted data in {relative}.")
        if row.get("lineup_type") not in (
            "homogeneous",
            "mixed",
            "unknown",
        ) or not _SHA.fullmatch(row.get("version", "")):
            problems.append(f"Invalid reproducibility fields in {relative}.")
        key = (row.get("kind"), row.get("attempt_id"))
        if key not in latest or row.get("revision", 1) > latest[key].get("revision", 1):
            latest[key] = row
    for row in latest.values():
        try:
            _validate_finish_fields(row)
        except ValueError, TypeError:
            problems.append("An archive has invalid published finish fields.")
        if row.get("kind") == "match":
            if row.get("match_termination_verified") is not True and not (
                row.get("final") or {}
            ).get("is_terminal"):
                problems.append("A published match lacks verified termination.")
        elif row.get("test_mode") is not None or row.get("match_id") is not None:
            problems.append("A non-match diagnostic was classified as a match.")
    summaries_path = workspace / "data" / "experiments.jsonl"
    summaries = (
        [
            json.loads(line)
            for line in summaries_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if summaries_path.exists()
        else []
    )
    experiments = {}
    for summary in summaries:
        try:
            _validate_recovery_attempts(summary.get("recovery_attempts", []))
        except _PublicationError:
            problems.append(
                "An experiment summary has contradictory recovery evidence."
            )
        experiment_id = summary["experiment_id"]
        if experiment_id not in experiments or summary.get("revision", 1) > experiments[
            experiment_id
        ].get("revision", 1):
            experiments[experiment_id] = summary
    results_path = workspace / "data" / "results.jsonl"
    results = (
        [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if results_path.exists()
        else []
    )
    required = {
        "timestamp",
        "match_id",
        "agent",
        "player_id",
        "version",
        "strategy",
        "test_mode",
        "lineup_type",
        "experiment_id",
        "role",
        "status",
        "outcome",
        "won",
        "log_path",
    }
    revisions = {}
    for row in results:
        if not required.issubset(row):
            problems.append("A result is missing required fields.")
            continue
        try:
            _validate_finish_fields(row)
        except ValueError, TypeError:
            problems.append("A result has invalid published finish fields.")
        key = (row["match_id"], row["agent"], row["player_id"])
        revision = row.get("revision", 1)
        if revision <= revisions.get(key, 0) or (
            revision > 1 and not row.get("correction_reason")
        ):
            problems.append("Duplicate result or non-increasing correction revision.")
        revisions[key] = revision
        if row["status"] not in ("completed", "failed", "aborted") or row[
            "test_mode"
        ] not in ("local", "playground", "self-play"):
            problems.append("Invalid result status or test mode.")
        expected_won = {"win": True, "loss": False, "draw": None, "unknown": None}.get(
            row["outcome"]
        )
        if (
            row["outcome"] not in ("win", "loss", "draw", "unknown")
            or row["won"] is not expected_won
            or (row["status"] != "completed" and row["outcome"] != "unknown")
        ):
            problems.append("Invalid or fabricated result outcome.")
        archive = logs.get(row["log_path"])
        if (
            archive is None
            or any(
                archive.get(field) != row[field]
                for field in (
                    "match_id",
                    "agent",
                    "player_id",
                    "version",
                    "strategy",
                    "test_mode",
                    "lineup_type",
                    "experiment_id",
                    "timestamp",
                    "status",
                    "outcome",
                    "won",
                    "role",
                )
            )
            or archive.get("won") is not row["won"]
            or archive.get("reason") != row.get("reason")
        ):
            problems.append(
                "A result does not match its canonical archive metadata/finish fields."
            )
        if row["experiment_id"] not in experiments:
            problems.append("A result has no experiment summary.")
        else:
            summary = experiments[row["experiment_id"]]
            if (
                summary["version"] != row["version"]
                or row["match_id"] not in summary["match_ids"]
            ):
                problems.append(
                    "A result's experiment summary does not cover its version/match."
                )
    if redactor.clean(results) != results or redactor.clean(summaries) != summaries:
        problems.append("Unredacted result or experiment summary data.")
    return {
        "ok": not problems,
        "archives": len(logs),
        "latest_attempts": len(latest),
        "game_result_records": len(results),
        "experiment_summary_records": len(summaries),
        "problems": problems,
    }


def append_experiment(workspace: Path, summary: dict, redactor: Redactor) -> None:
    """Append caller-authored findings only after all referenced matches terminate.

    No changes, hypothesis, recommendation or match outcomes are inferred.
    Corrections require an increasing revision and a correction reason; each
    tested version needs its own experiment ID. Originals are never rewritten.
    """
    required = {
        "experiment_id",
        "version",
        "baseline_version",
        "changes",
        "hypothesis",
        "match_ids",
        "findings",
        "limitations",
        "recommendation",
    }
    if not isinstance(summary, dict) or not required.issubset(summary):
        raise ValueError("Experiment summary is missing required fields.")
    if (
        not isinstance(summary["experiment_id"], str)
        or not summary["experiment_id"].strip()
    ):
        raise ValueError("Experiment ID must be a nonempty string.")
    for key in ("version", "baseline_version"):
        value = summary[key]
        if key == "baseline_version" and value is None:
            continue
        if not isinstance(value, str) or not _SHA.fullmatch(value):
            raise ValueError("Experiment versions must be full supplied Git SHAs.")
    if summary["recommendation"] not in ("adopt", "reject", "retest"):
        raise ValueError("Invalid experiment recommendation.")
    if not isinstance(summary["match_ids"], list) or any(
        not isinstance(match_id, str) or not match_id.strip()
        for match_id in summary["match_ids"]
    ):
        raise ValueError("Experiment match IDs must be a list of nonempty strings.")
    revision = summary.get("revision", 1)
    if type(revision) is not int or revision < 1:
        raise ValueError("Experiment revision must be a positive integer.")
    if revision > 1 and (
        not isinstance(summary.get("correction_reason"), str)
        or not summary["correction_reason"].strip()
    ):
        raise ValueError("Experiment corrections require a reason.")
    try:
        workspace = Path(workspace).resolve(strict=True)
        if not workspace.is_dir():
            raise ValueError("Workspace must be an existing directory.")
        cleaned = redactor.clean(summary)
        if "recovery_attempts" in cleaned:
            _validate_recovery_attempts(cleaned["recovery_attempts"])
        with _locked(workspace):
            _directory(workspace / "data")
            _publication_rows(workspace, "results")
            path = workspace / "data" / "experiments.jsonl"
            latest = None
            for row in _publication_rows(workspace, "experiments"):
                if row["experiment_id"] != cleaned["experiment_id"]:
                    continue
                if latest is None or row.get("revision", 1) >= latest.get(
                    "revision", 1
                ):
                    latest = row
            if latest is None:
                if revision != 1:
                    raise ValueError("An initial experiment must have revision one.")
            elif revision <= latest.get("revision", 1) or cleaned[
                "version"
            ] != latest.get("version"):
                raise ValueError(
                    "Experiment revisions must increase for the same version."
                )
            with _open(path, "a") as stream:
                _write(stream, cleaned)
    except _PublicationError:
        raise
    except Exception:
        raise RuntimeError("Unable to append experiment summary.") from None
