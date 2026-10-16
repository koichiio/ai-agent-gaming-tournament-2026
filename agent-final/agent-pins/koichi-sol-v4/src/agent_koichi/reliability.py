"""Read-only operational evidence from finalized public match archives.

Schema 1: seats contain original-provenance attempts, counters, usage, batches,
jobs and decisions; matches reference seats; aggregate groups count MATCHES by
lineup/timing/version/strategy. Counts are descriptive, never quality scores.
Missing evidence is not success. No prompts, chat, model output or configuration
blobs are returned. This is structural inspection, not ledger/secret attestation.
"""

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .ledger import ClaimLedger, authorized_message
from .replay import _attempts

_EVENTS = """model_request model_response model_error provider_error model_cancelled
model_usage model_retry extraction_batch extraction_job_started extraction_job_finished
extraction extraction_error belief_checkpoint policy_dispatch selected_decision
adapter_decision adapter_deadline deadline_fallback fallback submission
submission_acknowledged decision_discarded error timeout invalid_decision worker_error
evidence_error candidate_error cancellation""".split()
_IDENTITY = "match_id agent player_id attempt_id version strategy experiment_id lineup_type test_mode".split()
_USAGE = "prompt_tokens completion_tokens total_tokens cost".split()
_COHORT = "agent version strategy lineup_type timing_mode test_mode".split()
_LINEUP_FIELDS = (
    "position seat_position agent version strategy provider model evaluated".split()
)


def _count(value):
    return type(value) is int and 0 <= value < 2**64


def _number(value):
    return _count(value) or (
        type(value) is float and math.isfinite(value) and value >= 0
    )


def _time(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.utcoffset() is not None else None
    except TypeError, ValueError:
        return None


def _invalid_constant(_):
    raise ValueError("Nonfinite JSON is not an archive scalar.")


def _decision_key(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 8
        and isinstance(value[0], str)
        and bool(value[0].strip())
        and _count(value[1])
        and value[2] in ("messaging", "moving")
        and _count(value[3])
        and value[3] >= 1
        and value[4] in ("night_wolf", "night_seer", "day_vote")
        and type(value[5]) is int
        and 0 <= value[5] < 7
        and type(value[6]) is bool
        and type(value[7]) is bool
    )


def _seat_identity(original, identities, anchor=None):
    positions = set() if anchor is None else {anchor}
    # Only simulator-local seat IDs encode a position; platform IDs are opaque.
    local = re.fullmatch(r"local-seat-([0-6])", original["player_id"])
    if original["match_id"].startswith("local:") and local:
        positions.add(int(local[1]))
    for identity in [original, *identities]:
        for key in ("seat_id", "player_id"):
            if key in identity and identity[key] != original["player_id"]:
                return None, "seat_identity_mismatch"
        if "match_id" in identity and (
            not isinstance(identity["match_id"], str)
            or identity["match_id"].removeprefix("playground:")
            != original["match_id"].removeprefix("playground:")
        ):
            return None, "match_identity_mismatch"
        for key in ("seat_position", "your_position"):
            if key in identity:
                value = identity[key]
                if type(value) is not int or not 0 <= value < 7:
                    return None, "invalid_seat_identity"
                positions.add(value)
    if len(positions) > 1:
        return None, "seat_identity_mismatch"
    return next(iter(positions), None), None


def _trace(events):
    counts = Counter(dict.fromkeys(_EVENTS, 0))
    calls, jobs, decisions, batches = {}, {}, [], []
    cache, processed, current, policy_ready = {}, set(), None, False
    cache_known, processed_known = False, True
    sources, adapter_sources, acknowledgments = Counter(), Counter(), Counter()
    issues = Counter()

    def project(key, value, valid):
        if value is None or valid(value):
            return value
        issues["invalid_" + key] += 1
        return None

    def decision(index):
        row = {
            "event_index": index,
            "checkpoints": [],
            "dispatches": [],
            "job_ids": [],
            "first_policy_receipt": None,
            "first_policy_event_index": None,
            "selected_source": None,
            "pending_chats": None,
        }
        decisions.append(row)
        return row

    for index, event in enumerate(events):
        name, data = event["event"], event.get("data")
        receipt = project(
            "timestamp", event.get("timestamp"), lambda v: _time(v) is not None
        )
        if name in counts and not isinstance(data, dict):
            issues["malformed_operational_events"] += 1
        data = data if isinstance(data, dict) else {}
        if name in counts:
            counts[name] += 1
        if name == "observation":
            current, policy_ready = decision(index), False
            state = data.get("state", {})
            try:
                merged = dict(cache)
                if (
                    type(state.get("your_position")) is not int
                    or not 0 <= state["your_position"] < 7
                ):
                    raise ValueError
                for message in [
                    *data.get("messages", []),
                    *state.get("new_messages", []),
                ]:
                    message = authorized_message(message, state.get("your_position"))
                    source = ClaimLedger.source_id(message)
                    if source in merged and merged[source] != message:
                        raise ValueError
                    merged[source] = message
                cache = merged
                cache_known = True
                if processed_known:
                    current["pending_chats"] = sum(
                        m["type"] == "chat" and s not in processed
                        for s, m in cache.items()
                    )
            except ValueError, TypeError, AttributeError:
                issues["invalid_observation_cache"] += 1
                cache_known = False
        if name == "evidence_error":
            current, policy_ready = decision(index), False
        if name == "extraction":
            try:
                committed = {
                    ClaimLedger.source_id(m) for m in data["input"]["messages"]
                }
                processed.update(committed)
            except KeyError, ValueError, TypeError:
                issues["unavailable_committed_sources"] += 1
                processed_known = False
                if current is not None:
                    current["pending_chats"] = None
        job_id = data.get("extraction_job_id")
        if job_id is not None and not isinstance(job_id, str):
            issues["invalid_job_id"] += 1
            job_id = None
        if isinstance(job_id, str):
            if (
                current is not None
                and current["selected_source"] is None
                and name
                in (
                    "extraction_job_started",
                    "extraction",
                    "extraction_job_finished",
                )
                and job_id not in current["job_ids"]
            ):
                current["job_ids"].append(job_id)
            job = jobs.setdefault(
                job_id,
                {
                    "job_id": job_id,
                    "started": False,
                    "disposition": "unfinished",
                    "cause": None,
                },
            )
            if name == "extraction_job_started":
                job["started"] = True
            if name == "extraction_job_finished":
                disposition = data.get("disposition")
                job["disposition"] = (
                    disposition
                    if disposition in ("committed", "failed", "discarded")
                    else "unknown"
                )
                job["cause"] = {
                    "Agent closed before extraction commit.": "close",
                    "Unfinished extraction superseded by all currently pending chats.": "superseded",
                    "Extraction no longer matches the authenticated pending head.": "invalidated",
                }.get(data.get("reason"), job["cause"])
            if name in ("extraction_error", "model_error"):
                error = data.get("error", "")
                job["cause"] = (
                    "timeout"
                    if error
                    in (
                        "Extraction exceeded its deadline.",
                        "TimeoutError",
                        "ReadTimeout",
                        "ConnectTimeout",
                        "WriteTimeout",
                        "PoolTimeout",
                    )
                    else "schema"
                    if error
                    in (
                        "Extraction must be a dictionary containing an events list.",
                        "Malformed extraction response.",
                        "Extraction response is not a JSON object.",
                        "Model output is not a JSON object.",
                        "Invalid model response structure.",
                    )
                    else job["cause"] or "unknown"
                )
        if name == "extraction_batch":
            batch = {
                k: project(k, data.get(k), _count)
                for k in ("batch_count", "pending_count", "deferred_count")
            }
            batch["batch_scope"] = project(
                "batch_scope", data.get("batch_scope"), lambda v: v == "all_pending"
            )
            batches.append(batch)
        if name in (
            "policy_pre_dispatch",
            "belief_checkpoint",
            "policy_dispatch",
            "selected_decision",
        ):
            if current is None or current["selected_source"] is not None:
                current = decision(index)
            if cache_known and processed_known:
                current["pending_chats"] = sum(
                    m["type"] == "chat" and s not in processed for s, m in cache.items()
                )
            if name in ("policy_pre_dispatch", "belief_checkpoint"):
                policy_ready = True
            if name in ("belief_checkpoint", "policy_dispatch"):
                if (
                    name == "policy_dispatch"
                    and job_id is not None
                    and job_id not in current["job_ids"]
                ):
                    current["job_ids"].append(job_id)
                current[
                    "checkpoints" if name == "belief_checkpoint" else "dispatches"
                ].append(
                    {
                        "event_index": index,
                        "ledger_revision": project(
                            "ledger_revision", data.get("ledger_revision"), _count
                        ),
                        "job_id": job_id,
                        "key": project("decision_key", data.get("key"), _decision_key),
                        "kind": project(
                            "kind",
                            data.get("kind"),
                            lambda v: v in ("action", "message"),
                        ),
                    }
                )
            if name == "selected_decision":
                source = data.get("source")
                source = (
                    source
                    if source in ("high", "low", "fallback", "planner")
                    else "unknown"
                )
                current["selected_source"] = (
                    "agent_fallback" if source == "fallback" else source
                )
                sources[current["selected_source"]] += 1
        if name.startswith("model_") or name == "provider_error":
            if (
                name == "model_request"
                and job_id is None
                and policy_ready
                and current is not None
                and current["first_policy_event_index"] is None
            ):
                current["first_policy_receipt"] = receipt
                current["first_policy_event_index"] = index
            call_id = data.get("call_id")
            if type(call_id) not in (int, str):
                issues["unkeyed_model_events"] += 1
                continue
            call = calls.setdefault(
                call_id, {"call_id": call_id, "events": {}, "job_id": job_id}
            )
            if name in call["events"]:
                issues["duplicate_model_events"] += 1
            else:
                call["events"][name] = (data, receipt)
            if isinstance(job_id, str):
                if call["job_id"] not in (None, job_id):
                    issues["conflicting_call_job_ids"] += 1
                else:
                    call["job_id"] = job_id
        if name == "adapter_decision":
            answer = data.get("answer")
            source = answer.get("source") if isinstance(answer, dict) else None
            adapter_sources[
                source
                if source in ("high", "low", "fallback", "planner")
                else "unknown"
            ] += 1
        if name == "submission":
            accepted = data.get("accepted")
            disposition = (
                "accepted"
                if accepted is True
                else "rejected"
                if accepted is False
                else "unavailable"
            )
            acknowledgments[disposition] += 1
            if data.get("fallback_selected") is True:
                acknowledgments["fallback_" + disposition] += 1
    usage, usage_fields, usage_calls, intervals = (
        Counter(),
        Counter(),
        0,
        defaultdict(list),
    )
    for call in calls.values():
        entries = call["events"]
        billing = entries.get("model_usage", entries.get("model_response", ({}, None)))[
            0
        ].get("usage")
        if isinstance(billing, dict):
            usage_calls += 1
            for key in _USAGE:
                if _number(billing.get(key)):
                    usage[key] += billing[key]
                    if type(usage[key]) is float and not math.isfinite(usage[key]):
                        raise ValueError("Observed usage sum is not finite.")
                    usage_fields[key] += 1
                elif billing.get(key) is not None:
                    issues["invalid_usage_" + key] += 1
        if call["job_id"] is not None:
            request, response = (
                entries.get("model_request", ({}, None)),
                entries.get("model_response", ({}, None)),
            )
            if request[0].get("extraction_job_id") != call["job_id"]:
                request = ({}, None)
            if response[0].get("extraction_job_id") != call["job_id"]:
                response = ({}, None)
            intervals[call["job_id"]].append(
                {
                    "call_id": call["call_id"],
                    "request_occurred_at": project(
                        "occurred_at",
                        request[0].get("occurred_at"),
                        lambda v: _time(v) is not None,
                    ),
                    "response_occurred_at": project(
                        "occurred_at",
                        response[0].get("occurred_at"),
                        lambda v: _time(v) is not None,
                    ),
                    "request_receipt": request[1],
                    "response_receipt": response[1],
                }
            )
    overlap, revisions = Counter(), Counter()
    for row in decisions:
        checkpoints, dispatches = row["checkpoints"], row["dispatches"]
        equality = "unavailable"
        if len(checkpoints) > 1 or len(dispatches) > 1:
            equality = "ambiguous"
        elif len(checkpoints) == len(dispatches) == 1:
            left, right = checkpoints[0], dispatches[0]
            if all(
                _count(r["ledger_revision"])
                and _decision_key(r["key"])
                and r["kind"] in ("action", "message")
                for r in (left, right)
            ):
                equality = (
                    "equal"
                    if all(
                        left[k] == right[k] for k in ("ledger_revision", "key", "kind")
                    )
                    else "mismatch"
                )
        row["revision_equality"] = equality
        revisions[equality] += 1
        row["extraction_intervals"] = []
        for job_id in row["job_ids"]:
            evidence = intervals.get(job_id, [])
            if not evidence:
                overlap["unavailable"] += 1
            for interval in evidence:
                start, end, policy = map(
                    _time,
                    (
                        interval["request_occurred_at"],
                        interval["response_occurred_at"],
                        row["first_policy_receipt"],
                    ),
                )
                relation = "unavailable"
                if (
                    start is not None
                    and end is not None
                    and policy is not None
                    and start <= end
                ):
                    relation = (
                        "response_before_or_at_policy_receipt"
                        if end <= policy
                        else "recorded_interval_spans_policy_receipt"
                        if start <= policy < end
                        else "request_after_policy_receipt"
                    )
                overlap[relation] += 1
                receipt = _time(interval["response_receipt"])
                row["extraction_intervals"].append(
                    {
                        **interval,
                        "job_id": job_id,
                        "relation": relation,
                        "buffered_response_after_policy_receipt": end
                        <= policy
                        < receipt
                        if all(t is not None for t in (end, policy, receipt))
                        else None,
                    }
                )
        for evidence in checkpoints + dispatches:
            evidence.pop("key")
            evidence.pop("kind")
    return {
        "event_counts": dict(counts),
        "issues": dict(issues),
        "model_call_count": len(calls),
        "model_counts": {
            name: sum(name in c["events"] for c in calls.values())
            for name in (
                "model_request",
                "model_response",
                "model_error",
                "provider_error",
                "model_cancelled",
            )
        },
        "unfinished_model_calls": sum(
            "model_request" in c["events"]
            and not set(c["events"]).intersection(
                ("model_response", "model_error", "provider_error", "model_cancelled")
            )
            for c in calls.values()
        ),
        "usage_calls": usage_calls,
        "usage_field_calls": dict(usage_fields),
        "usage_observed_sum": {k: usage.get(k) for k in _USAGE},
        "selected_sources": dict(sources),
        "adapter_sources": dict(adapter_sources),
        "submission_acceptance": dict(acknowledgments),
        "jobs": list(jobs.values()),
        "extraction_dispositions": dict(
            Counter(j["disposition"] for j in jobs.values())
        ),
        "extraction_causes": dict(
            Counter(j["cause"] for j in jobs.values() if j["cause"])
        ),
        "batches": batches,
        "batch_max": max(
            (b["batch_count"] for b in batches if _number(b["batch_count"])),
            default=None,
        ),
        "pending_max": max(
            (b["pending_count"] for b in batches if _number(b["pending_count"])),
            default=None,
        ),
        "all_pending_batches": sum(
            b["batch_scope"] == "all_pending"
            and type(b["batch_count"]) is int
            and type(b["pending_count"]) is int
            and b["batch_count"] >= 0
            and b["batch_count"] == b["pending_count"]
            and b["deferred_count"] == 0
            for b in batches
        ),
        "decisions": decisions,
        "revision_equality": dict(revisions),
        "overlap_evidence": dict(overlap),
        "last_decision_pending_chats": decisions[-1]["pending_chats"]
        if decisions
        else None,
    }


def summarize_reliability(paths: list[Path], workspace: Path) -> dict:
    """Inspect only supplied, directly owned data/logs/*.json; never open private paths.

    Relative paths are workspace-relative. Reject unsafe paths, nonterminal
    archives, malformed trace envelopes and duplicate seat archives (choose a
    correction revision explicitly). Recovered traces use replay's identity and
    inclusion checks but never its inference/scoring or private journal readers.
    """
    workspace = workspace.absolute()
    if ".." in workspace.parts:
        raise ValueError("Workspace must not contain parent traversal.")
    root = workspace / "data" / "logs"
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Public archive paths must not contain symlinks.")
    seats, seen_paths, seen_seats = [], set(), set()
    for path in sorted(set(paths), key=str):
        path = path if path.is_absolute() else workspace / path
        if (
            ".." in path.parts
            or path.parent != root
            or path.suffix != ".json"
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(
                "Only direct workspace data/logs JSON archives are allowed."
            )
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            log = json.loads(
                path.read_text(encoding="utf-8"), parse_constant=_invalid_constant
            )
            final = log.get("final") or {}
            if (
                log.get("kind") != "match"
                or log.get("status") not in ("completed", "aborted", "failed")
                or not (
                    log.get("match_termination_verified") is True
                    or "match_termination_verified" not in log
                    and final.get("is_terminal") is True
                )
            ):
                raise ValueError
            proof = final.get("controller_termination")
            if proof is not None and (
                proof.get("match_id") != log.get("match_id")
                or proof.get("workers_stopped") is not True
                or not (
                    proof.get("engine_terminal") is True
                    or log["status"] != "completed"
                    and proof.get("engine_aborted") is True
                )
            ):
                raise ValueError
            traces = [log, *log.get("recovered_attempts", [])]
            for trace in traces:
                metadata = trace.get("metadata", trace)
                if (
                    any(
                        not isinstance(metadata.get(k), str) or not metadata[k].strip()
                        for k in _IDENTITY[:5]
                    )
                    or not re.fullmatch(
                        r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", metadata["version"]
                    )
                    or not isinstance(trace.get("events"), list)
                    or any(
                        not isinstance(e, dict)
                        or not isinstance(e.get("event"), str)
                        or e.get("event")
                        in (
                            "start",
                            "assignment",
                            "state",
                            "observation",
                            "grant",
                            "result",
                        )
                        and not isinstance(e.get("data"), dict)
                        for e in trace["events"]
                    )
                ):
                    raise ValueError
                for key in [*_IDENTITY, "status"]:
                    value = metadata.get(key)
                    if value is not None and (
                        not isinstance(value, str) or not value.strip()
                    ):
                        raise ValueError
                for key, allowed in {
                    "lineup_type": ("homogeneous", "mixed", "unknown"),
                    "test_mode": ("local", "playground", "self-play"),
                    "status": ("completed", "aborted", "failed"),
                }.items():
                    if metadata.get(key) is not None and metadata[key] not in allowed:
                        raise ValueError
                for key in ("started_at", "timestamp"):
                    if metadata.get(key) is not None and _time(metadata[key]) is None:
                        raise ValueError
                config = metadata.get("configuration")
                if config is None:
                    config = {}
                if not isinstance(config, dict) or not isinstance(
                    config.get("lineup", []), list
                ):
                    raise ValueError
                if config.get("timing_mode", "unknown") not in (
                    "normal",
                    "accelerated",
                    "unknown",
                ):
                    raise ValueError
                for key in ("window_seconds", "max_seconds"):
                    if config.get(key) is not None and not _number(config[key]):
                        raise ValueError
                for key in ("unreadable_event_lines", "truncated_event_lines"):
                    if trace.get(key) is not None and not _count(trace[key]):
                        raise ValueError
                for member in config.get("lineup", []):
                    if not isinstance(member, dict):
                        raise ValueError
                    for key in _LINEUP_FIELDS:
                        value = member.get(key)
                        if value is None:
                            continue
                        if key in ("position", "seat_position"):
                            valid = type(value) is int and 0 <= value < 7
                        elif key == "evaluated":
                            valid = type(value) is bool
                        else:
                            valid = isinstance(value, str) and bool(value.strip())
                        if not valid:
                            raise ValueError
            identity = tuple(log[k] for k in _IDENTITY[:3])
            if identity in seen_seats:
                raise ValueError
            seen_seats.add(identity)
            bound_attempts = _attempts(log)
            anchor = None
            recovered_positions = []
            for (metadata, _), trace in zip(bound_attempts, traces, strict=True):
                original = trace.get("metadata", trace)
                if metadata["included"]:
                    position, reason = _seat_identity(
                        original, metadata["identities"], anchor
                    )
                    if reason:
                        metadata.update(included=False, skip_reason=reason)
                    elif not metadata["recovered"]:
                        anchor = position
                    elif position is not None:
                        recovered_positions.append((metadata, position))
            if anchor is None and len({p for _, p in recovered_positions}) > 1:
                # Unanchored recoveries cannot elect the first or majority position.
                for metadata, _ in recovered_positions:
                    metadata.update(
                        included=False,
                        skip_reason="conflicting_recovered_seat_positions",
                    )
            attempts = []
            for (metadata, events), trace in zip(bound_attempts, traces, strict=True):
                original = trace.get("metadata", trace)
                config = original.get("configuration") or {}
                attempts.append(
                    {
                        **{k: metadata.get(k) for k in _IDENTITY},
                        "trace_id": metadata["trace_id"],
                        "included": metadata["included"],
                        "skip_reason": metadata.get("skip_reason"),
                        "event_count": len(events),
                        "operational_telemetry_available": any(
                            e["event"] in _EVENTS for e in events
                        ),
                        "trace_missing_reported": trace.get("trace_missing_reason")
                        is not None,
                        "started_at": original.get("started_at"),
                        "timestamp": original.get("timestamp"),
                        "status": original.get("status"),
                        "timing_mode": config.get("timing_mode", "unknown"),
                        "window_seconds": config.get("window_seconds"),
                        "max_seconds": config.get("max_seconds"),
                        "lineup": [
                            {k: member.get(k) for k in _LINEUP_FIELDS}
                            for member in config.get("lineup", [])
                        ],
                        "integrity": {
                            k: trace.get(k)
                            for k in ("unreadable_event_lines", "truncated_event_lines")
                        },
                        "operations": _trace(events) if metadata["included"] else None,
                    }
                )
            seats.append(
                {
                    **{k: log.get(k) for k in _IDENTITY},
                    "status": log["status"],
                    "log_path": path.relative_to(workspace).as_posix(),
                    "attempts": attempts,
                }
            )
        except ValueError, TypeError, KeyError, AttributeError, OSError:
            raise ValueError(
                "Archive failed finalized public structure/identity checks."
            ) from None
    matches, groups = {}, {}
    for index, seat in enumerate(seats):
        match = matches.setdefault(
            seat["match_id"],
            {
                "match_id": seat["match_id"],
                "seat_indices": [],
                "seat_statuses": Counter(),
                "seat_event_totals": Counter(),
            },
        )
        match["seat_indices"].append(index)
        match["seat_statuses"][seat["status"]] += 1
        for attempt in seat["attempts"]:
            if attempt["included"]:
                key = tuple(attempt.get(k) for k in _COHORT)
                group = groups.setdefault(
                    key,
                    {
                        "match_ids": set(),
                        "attempt_count": 0,
                        "seat_event_totals": Counter(),
                    },
                )
                group["match_ids"].add(seat["match_id"])
                group["attempt_count"] += 1
                group["seat_event_totals"].update(attempt["operations"]["event_counts"])
                match["seat_event_totals"].update(attempt["operations"]["event_counts"])
    return {
        "schema_version": 1,
        "seats": seats,
        "matches": list(matches.values()),
        "aggregate": [
            {
                **dict(zip(_COHORT, key)),
                **group,
                "match_ids": sorted(group["match_ids"]),
                "match_count": len(group["match_ids"]),
            }
            for key, group in sorted(groups.items(), key=lambda pair: str(pair[0]))
        ],
        "caveats": [
            "Counts describe recorded events, not quality or absence of failures; missing telemetry stays unavailable.",
            "Seats within a match are correlated. Aggregate units are distinct matches, not independent seats; cohorts are not pooled.",
            "seat_event_totals sum evaluated attempt events, including repeated match-wide controller events in different seat archives; they are not unique incident counts. Opponent wrappers are excluded from seat policy counters.",
            "occurred_at is a buffered agent wall-clock record; receipt timestamps are controller/runtime transport records, not physical admission/completion times. Clock skew and buffering are unmeasured; overlap is recorded-interval evidence only.",
            "Usage is counted once per attempt/call_id, preferring model_usage over repeated model_response usage; observed sums can be incomplete.",
            "Pending chats use authenticated observation cache minus committed extraction inputs at the last recorded decision, not final transcripts. Integrity checks do not certify ledger links or secret redaction.",
        ],
    }
