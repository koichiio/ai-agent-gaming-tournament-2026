"""Offline action paths and descriptive, match-clustered replay summaries.

``action_path_report`` accepts a finalized runtime archive: top-level metadata
and ``events``, plus optional ``recovered_attempts[{metadata, events, ...}]``.
It never pairs events across attempts. Event references are zero-based indexes
in that attempt, not clocks. Current runtime ``submission`` events are intents;
only an explicit ``submission_acknowledged`` event establishes acknowledgment.
Selection latency is copied verbatim, never estimated from audit timestamps.
Attempt inclusion and skipped-trace provenance come from replay's authoritative
``_attempts(log)`` helper, including session identity validation. Standalone
inspection and replay therefore apply identical trace boundaries and exclusions.

``aggregate_reports`` accepts replay dictionaries with ``match_id``, ``agent``,
``player_id``, ``version``, optional ``strategy``/``configuration``, ``status``,
``outcome``, ``lineup_type`` and ``checkpoints``. Each checkpoint needs an
``attempt_id``, ``version`` (or report defaults), ``kind``, and an occurrence ID:
``decision_id`` or ``checkpoint_id``, or an attempt-local ``event_index`` or
``invocation_ordinal``. Canonical checkpoint IDs are
``f"{attempt_id}:{belief_event_index}"``; action paths use that same ID when a
checkpoint exists, otherwise their first relevant event index. IDs are scoped by
kind; supplied explicit IDs must uniquely name an invocation, including across
attempts. State/decision keys are freshness metadata, NEVER occurrence
IDs: legitimate discussion invocations can share the same key. Other provenance,
including frozen ``ledger_revision`` and ``day``/``subphase``, is
preserved. Its ``modes`` maps names (including ``llm_raw``/``llm_conditioned``)
to dictionaries with ``forecast_status`` and ``tasks``. Status is ``valid``,
``invalid``, ``missing``, ``fallback``, ``no_score``, ``fallback_no_score``,
``unavailable``, or ``valid_unscored``. A task maps to scoring
helper fields: ``eligible_targets``, ``labeled_targets``, ``brier``, ``log_loss``,
``infinite_log_loss``. Counts must be supplied even without a forecast, so label
coverage is independent of forecast availability. A null task is inapplicable.

Existing direct task names (``unknown_living_role_metrics``,
``unknown_living_wolf_metrics``, ``wolf_seer_identification``) are also accepted;
those legacy scored modes default to valid. ``mode.as_run`` can wrap the task
metrics; ``mode.recomputed`` is NOT pooled with as-run scores. Expose a rescore as
a separate mode/report with its evaluator version if aggregation is desired.
Checkpoint-level ``label_coverage`` may independently map task names to eligible
and labeled counts, including when modes lack usable scores. These are reported
as ``label/TASK/coverage``, without conditioning on any forecast status.
Optional ``action_paths`` (alias ``action_path_report``) is the output of
``action_path_report`` (or its ``decisions`` list). Without it action rates are
unavailable, not zero. A missing root ``agent`` may be supplied by unambiguous
checkpoint/attempt metadata. Skipped attempt metadata is retained for disclosure.
Forecast status rates use recorded checkpoint opportunities, not all actions.
``action/checkpoint_available_rate`` separately exposes actions without a matched
checkpoint; early fallback returns do not acquire fabricated belief snapshots.

Policy, declared before reading scores: first record of each occurrence ID within
seat/kind/version/strategy/configuration wins for checkpoint metrics.
Only repeated records of the same occurrence deduplicate; new invocations and
new attempts stay separate unless an explicit shared occurrence ID proves a
retry, even if state keys or payloads match. All duplicate
provenance is retained. Boolean action evidence is unioned on repeated occurrence
records. Equal checkpoints within seat, equal observed seats within match, equal
matches within lineup/cohort; each metric uses only children where it is known.
``units`` reports that denominator at every level. No target-weighted pooling,
confidence intervals, causal comparisons, or independent-seat uncertainty is
claimed. Any positively weighted infinite log loss remains explicitly infinite.
Missing identities are rejected for checkpoints/seats; null-match diagnostics
are excluded, never combined into an invented match. Conflicting lineup labels
put the entire match in ``unknown``. Draws and unknown/incomplete outcomes are
excluded from the decisive win-rate denominator; draw rates are separate.
"""

import json
import math
from collections import Counter, defaultdict

_TASKS = (
    "unknown_living_role_metrics",
    "unknown_living_wolf_metrics",
    "wolf_seer_identification",
)
_STATUSES = (
    "valid",
    "invalid",
    "missing",
    "fallback",
    "no_score",
    "fallback_no_score",
    "unavailable",
    "valid_unscored",
)
_SOURCES = {
    "high": "high",
    "low": "low",
    "fallback": "agent_fallback",
    "planner": "planner",
}


def _token(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(row):
    occurrence = row.get("decision_id") or row.get("checkpoint_id")
    if occurrence is None and row.get("attempt_id") is not None:
        if row.get("event_index") is not None:
            occurrence = f"{row['attempt_id']}:{row['event_index']}"
        elif row.get("invocation_ordinal") is not None:
            occurrence = [
                row["attempt_id"],
                "invocation",
                row.get("decision_key", row.get("key")),
                row["invocation_ordinal"],
            ]
    if occurrence is None:
        raise ValueError("A decision needs an occurrence ID, not just a state key.")
    return _token([row.get("kind"), "occurrence", occurrence])


def action_path_report(log: dict) -> dict:
    """Return attempt provenance, deduplicated decision paths, and evidence counts.

    ``valid_action`` is tri-state: true only for a matching validated candidate
    or runtime-validated submission, otherwise unknown. A successful adapter
    return alone does not prove runtime validation. ``acknowledged`` is likewise
    unknown without explicit evidence, even after terminal state progression.
    ``discarded`` is independent of selection source and submission intent.
    """
    # Lazy import keeps replay's reporting call sites cycle-free while sharing
    # its complete attempt inclusion policy, rather than mirroring validators.
    from .replay import _attempts

    decisions, attempts = [], []
    for metadata, events in _attempts(log):
        attempt_id = metadata.get("attempt_id")
        trace_id = metadata["trace_id"]
        identity = attempt_id or trace_id
        attempt = {
            **metadata,
            "origin": "recovered" if metadata["recovered"] else "top",
        }
        attempts.append(attempt)
        if not attempt["included"]:
            continue
        provenance = {
            name: metadata.get(name)
            for name in (
                "match_id",
                "agent",
                "player_id",
                "attempt_id",
                "version",
                "strategy",
                "configuration",
                "lineup_type",
            )
        }
        provenance["trace_id"] = trace_id
        attempt_decisions = []
        rows, pending, last_submission, state = {}, None, None, {}
        candidates = []
        for event_index, event in enumerate(events):
            name, data = event.get("event"), event.get("data")
            if not isinstance(data, dict):
                continue
            if name in ("state", "observation"):
                state = data.get("state", data)
                if name == "observation":
                    # A new agent call is an invocation boundary, even when its
                    # public state version has not advanced during discussion.
                    pending, candidates = None, []
                continue
            if name == "validated_candidate":
                candidates.append(data.get("decision"))
                continue
            if name not in (
                "belief_checkpoint",
                "llm_belief_checkpoint",
                "selected_decision",
                "adapter_decision",
                "deadline_fallback",
                "decision_discarded",
                "submission",
                "submission_acknowledged",
                "ambiguous_action",
                "ambiguous_chat",
            ):
                continue
            answer = data.get("answer", data) if name == "adapter_decision" else data
            decision = answer.get("decision")
            kind = data.get("kind") or answer.get("kind")
            if kind is None and pending is not None:
                kind = pending["kind"]
            if kind is None and isinstance(decision, dict):
                # Historical decisions contain every schema field, including
                # action_id=None on messages (and content=None on actions).
                if decision.get("action_id") is not None:
                    kind = "action"
                elif isinstance(decision.get("content"), str):
                    kind = "message"
            key = answer.get("decision_key", answer.get("key"))
            state_version = answer.get("state_version", state.get("state_version"))
            if (
                key is None
                and name
                not in (
                    "decision_discarded",
                    "submission_acknowledged",
                    "ambiguous_action",
                    "ambiguous_chat",
                )
                and state.get("session_id") is not None
                and state_version == state.get("state_version")
            ):
                key = [
                    state["session_id"],
                    state_version,
                    state.get("phase"),
                    (state.get("game_state") or {}).get("day"),
                    (state.get("game_state") or {}).get("phase"),
                    state.get("your_position"),
                    state.get("eliminated"),
                    state.get("is_current_actor"),
                ]
            if isinstance(key, (list, tuple)) and len(key) == 8:
                state_version = key[1]
            context = {
                "kind": kind,
                "decision_key": key,
                "state_version": state_version,
                "day": answer.get("day", (state.get("game_state") or {}).get("day")),
                "subphase": answer.get(
                    "subphase", (state.get("game_state") or {}).get("phase")
                ),
            }
            if isinstance(key, (list, tuple)) and len(key) == 8:
                context.update(day=key[3], subphase=key[4])
            if name in (
                "submission_acknowledged",
                "ambiguous_action",
                "ambiguous_chat",
            ):
                row = last_submission
                if row is None:
                    continue
                acknowledgment_id = answer.get("decision_id") or answer.get(
                    "checkpoint_id"
                )
                if acknowledgment_id is not None and acknowledgment_id not in (
                    row["decision_id"],
                    row["checkpoint_id"],
                ):
                    continue
                if key is not None and _token(key) != _token(row["decision_key"]):
                    continue
                if (
                    data.get("state_version") is not None
                    and data["state_version"] != row["state_version"]
                ):
                    continue
            else:
                explicit_id = answer.get("decision_id") or answer.get("checkpoint_id")
                row = (
                    rows.get(_token([kind, explicit_id]))
                    if explicit_id is not None
                    else None
                )
                if row is None and pending is not None:
                    # Refresh-state events can intervene before discard/submission.
                    compatible = (
                        kind is None
                        or pending["kind"] is None
                        or kind == pending["kind"]
                    ) and (
                        key is None
                        or pending["decision_key"] is None
                        or _token(key) == _token(pending["decision_key"])
                    )
                    boundary = (
                        name == "belief_checkpoint"
                        or name == "deadline_fallback"
                        or name == "selected_decision"
                        and pending["selected"]
                        or name == "adapter_decision"
                        and pending["adapter_returned"]
                    )
                    if (
                        not boundary
                        and compatible
                        and (
                            name == "decision_discarded"
                            or pending["state_version"] is None
                            or state_version in (None, pending["state_version"])
                        )
                    ):
                        row = pending
                if row is None:
                    occurrence_id = explicit_id or f"{identity}:{event_index}"
                    row = {
                        **provenance,
                        **context,
                        "decision_id": occurrence_id,
                        "checkpoint_id": None,
                        "event_index": event_index,
                        "selected": False,
                        "selected_source": None,
                        "adapter_returned": False,
                        "runtime_fallback": False,
                        "discussion_termination": False,
                        "discarded": False,
                        "submission_intent": False,
                        "acknowledged": None,
                        "ambiguous_write": False,
                        "valid_action": None,
                        "latency_ms": None,
                        "events": [],
                    }
                    decisions.append(row)
                    attempt_decisions.append(row)
                for field, value in context.items():
                    if row.get(field) is None:
                        row[field] = value
                rows[_token([row["kind"], row["decision_id"]])] = row
                if explicit_id is not None:
                    rows[_token([row["kind"], explicit_id])] = row
                pending = row
            row["events"].append({"event": name, "event_index": event_index})
            if name in ("selected_decision", "adapter_decision"):
                row["selected"] = True
                row["selected_source"] = _SOURCES.get(
                    answer.get("source"), answer.get("source") or "unknown"
                )
                row["latency_ms"] = answer.get("latency_ms", row["latency_ms"])
                row["ledger_revision"] = answer.get(
                    "ledger_revision", row.get("ledger_revision")
                )
                row["decision"] = decision
                row["adapter_returned"] |= name == "adapter_decision"
                if decision in candidates:
                    row["valid_action"] = True
            elif name in ("belief_checkpoint", "llm_belief_checkpoint"):
                if row["checkpoint_id"] is None:
                    row["checkpoint_id"] = (
                        answer.get("checkpoint_id") or f"{identity}:{event_index}"
                    )
                    # The checkpoint occurrence is the shared replay/action ID.
                    if explicit_id is None:
                        row["decision_id"] = row["checkpoint_id"]
                row["ledger_revision"] = data.get("ledger_revision")
            elif name == "deadline_fallback":
                row["runtime_fallback"] = True
            elif name == "decision_discarded":
                row["discarded"] = True
                row["discard_reason"] = data.get("reason")
                pending = None
                candidates = []
            elif name == "submission":
                row["submission_intent"] = True
                row["valid_action"] = True
                row["submitted_decision"] = decision
                row["discussion_termination"] = (
                    kind == "message"
                    and isinstance(decision, dict)
                    and isinstance(decision.get("content"), str)
                    and not decision["content"].strip()
                )
                last_submission, pending, candidates = row, None, []
            elif name == "submission_acknowledged":
                row["acknowledged"] = True
            else:
                row["ambiguous_write"] = True
        for row in attempt_decisions:
            row["path"] = (
                "selected_but_discarded"
                if row["selected"] and row["discarded"]
                else "runtime_fallback"
                if row["runtime_fallback"]
                else row["selected_source"]
                if row["selected"]
                else "discussion_termination"
                if row["discussion_termination"]
                else "unknown"
            )
    counts = Counter(row["path"] for row in decisions)
    return {
        "attempts": attempts,
        "decisions": decisions,
        "path_counts": dict(counts),
        "decision_count": len(decisions),
        "selection_count": sum(row["selected"] for row in decisions),
        "submission_intent_count": sum(row["submission_intent"] for row in decisions),
        "acknowledgment_count": sum(row["acknowledged"] is True for row in decisions),
        "timing_boundary": "Recorded agent selection latency only; excludes cleanup, adapter transport, refresh, and platform submission. Missing timing stays null.",
    }


def _mean(values):
    values = [value for value in values if value is not None]
    infinite = any(value == math.inf for value in values)
    return {
        "value": math.fsum(values) / len(values) if values and not infinite else None,
        "units": len(values),
        "infinite_log_loss": infinite,
    }


def _combine(rows):
    keys = sorted({key for row in rows for key in row})
    return {
        key: _mean(
            [
                math.inf if row[key]["infinite_log_loss"] else row[key]["value"]
                for row in rows
                if key in row
            ]
        )
        for key in keys
    }


def _cohort(row, default):
    configuration = row.get("configuration", default.get("configuration"))
    strategy = row.get("strategy", default.get("strategy"))
    if strategy is None and isinstance(configuration, dict):
        strategy = configuration.get("strategy")
    return _token(
        {
            "version": row.get("version", default.get("version")),
            "strategy": strategy,
            "configuration": configuration,
        }
    )


def _tasks(mode):
    scored = mode.get("as_run", mode)
    if not isinstance(scored, dict):
        return {}
    return scored.get(
        "tasks", {name: scored[name] for name in _TASKS if name in scored}
    )


def aggregate_reports(reports: list[dict]) -> dict:
    """Aggregate replay reports without treating seats/checkpoints as matches.

    Output ``lineups`` always has homogeneous/mixed/unknown buckets, each with
    distinct ``match_count`` and version/configuration ``cohorts``. Cohorts have
    ``matches`` containing seat units and match metrics, and equal-match
    ``metrics``. Metric paths are ``forecast/MODE/availability`` and status rates,
    ``task/MODE/TASK/{label_coverage,brier,log_loss}``, ``action/*_rate``, and
    ``outcome/{decisive_win_rate,draw_rate}``. Every value has ``units`` and an
    ``infinite_log_loss`` flag; units are checkpoints, seats, or matches at the
    corresponding level. Missing metrics are never imputed as perfect scores.
    """
    seats, labels, excluded, report_provenance = {}, defaultdict(set), [], []
    for report_index, report in enumerate(reports):
        match_id = report.get("match_id")
        if match_id is None:
            excluded.append(
                {"report_index": report_index, "reason": "missing_match_id"}
            )
            continue
        agent = report.get("agent")
        if agent is None:
            agents = {
                row["agent"]
                for row in [*report.get("checkpoints", []), *report.get("attempts", [])]
                if row.get("agent") is not None
            }
            agent = next(iter(agents)) if len(agents) == 1 else None
        if agent is None or report.get("player_id") is None:
            raise ValueError("Match reports require agent and player_id.")
        report_provenance.append(
            {
                "report_index": report_index,
                **{
                    name: report.get(name)
                    for name in (
                        "match_id",
                        "agent",
                        "player_id",
                        "version",
                        "configuration",
                        "strategy",
                        "attempt_id",
                        "attempts",
                        "evaluator_version",
                        "log_path",
                    )
                },
            }
        )
        lineup = report.get("lineup_type", "unknown")
        labels[match_id].add(
            lineup if lineup in ("homogeneous", "mixed") else "unknown"
        )
        seat_id = (match_id, agent, report["player_id"])
        seat = seats.setdefault(seat_id, {"reports": [], "cohorts": {}})
        seat["reports"].append(report)
        for checkpoint in report.get("checkpoints", []):
            if any(
                checkpoint.get(name) is not None and checkpoint[name] != value
                for name, value in (
                    ("match_id", match_id),
                    ("agent", agent),
                    ("player_id", report["player_id"]),
                )
            ):
                raise ValueError("Checkpoint identity differs from its report seat.")
            cohort_key = _cohort(checkpoint, report)
            group = seat["cohorts"].setdefault(
                cohort_key, {"checkpoints": {}, "actions": {}}
            )
            identity = _identity({"attempt_id": report.get("attempt_id"), **checkpoint})
            entry = group["checkpoints"].setdefault(
                identity, {"checkpoint": checkpoint, "provenance": []}
            )
            entry["provenance"].append(
                {
                    **{
                        name: checkpoint.get(name, report.get(name))
                        for name in (
                            "attempt_id",
                            "version",
                            "checkpoint_id",
                            "decision_id",
                            "event_index",
                            "invocation_ordinal",
                            "decision_key",
                            "key",
                            "kind",
                            "day",
                            "state_version",
                            "subphase",
                            "ledger_revision",
                        )
                    },
                    "report_index": report_index,
                }
            )
        paths = report.get("action_paths", report.get("action_path_report", {}))
        paths = paths.get("decisions", []) if isinstance(paths, dict) else paths
        for action in paths:
            if any(
                action.get(name) is not None and action[name] != value
                for name, value in (
                    ("match_id", match_id),
                    ("agent", agent),
                    ("player_id", report["player_id"]),
                )
            ):
                raise ValueError("Action identity differs from its report seat.")
            group = seat["cohorts"].setdefault(
                _cohort(action, report), {"checkpoints": {}, "actions": {}}
            )
            identity = _identity({"attempt_id": report.get("attempt_id"), **action})
            group["actions"].setdefault(identity, []).append(action)

    lineups = {
        name: {"match_count": 0, "cohorts": []}
        for name in ("homogeneous", "mixed", "unknown")
    }
    match_lineups = {
        match: next(iter(values)) if len(values) == 1 else "unknown"
        for match, values in labels.items()
    }
    for lineup in match_lineups.values():
        lineups[lineup]["match_count"] += 1
    grouped = defaultdict(lambda: defaultdict(list))
    # Use the union of modes within a cohort: an omitted mode is missing, not a
    # successful-only denominator. Inapplicable tasks must be explicitly null.
    modes = defaultdict(set)
    for seat in seats.values():
        for cohort, group in seat["cohorts"].items():
            for entry in group["checkpoints"].values():
                modes[cohort].update(entry["checkpoint"].get("modes", {}))
    for (match_id, agent, player_id), seat in seats.items():
        highest_revision = max(report.get("revision", 1) for report in seat["reports"])
        outcomes = set()
        for report in seat["reports"]:
            if report.get("revision", 1) != highest_revision:
                continue
            status = report.get("status")
            status = (
                status if status in ("completed", "failed", "aborted") else "unknown"
            )
            outcome = report.get("outcome")
            outcome = (
                outcome
                if status == "completed" and outcome in ("win", "loss", "draw")
                else "unknown"
            )
            outcomes.add((status, outcome))
        if len(outcomes) != 1:
            raise ValueError(
                "Conflicting seat outcomes require explicit report revisions."
            )
        status, outcome = outcomes.pop()
        for cohort, group in seat["cohorts"].items() or [
            (_cohort({}, seat["reports"][0]), {"checkpoints": {}, "actions": {}})
        ]:
            samples = defaultdict(list)
            forecast_counts = defaultdict(Counter)
            target_counts = defaultdict(Counter)
            for entry in group["checkpoints"].values():
                checkpoint = entry["checkpoint"]
                for task_name, task in checkpoint.get("label_coverage", {}).items():
                    if task is None:
                        continue
                    eligible, labeled = (
                        task.get("eligible_targets"),
                        task.get("labeled_targets"),
                    )
                    if (
                        type(eligible) is not int
                        or type(labeled) is not int
                        or not 0 <= labeled <= eligible
                    ):
                        raise ValueError(
                            "Label counts require 0 <= labeled_targets <= eligible_targets."
                        )
                    prefix = f"label/{task_name}"
                    target_counts[prefix].update(
                        eligible_targets=eligible, labeled_targets=labeled
                    )
                    samples[f"{prefix}/coverage"].append(
                        labeled / eligible if eligible else None
                    )
                for mode_name in sorted(modes[cohort]):
                    mode = checkpoint.get("modes", {}).get(mode_name) or {}
                    tasks = _tasks(mode)
                    forecast_status = mode.get("forecast_status", mode.get("status"))
                    if forecast_status is None and isinstance(mode.get("as_run"), dict):
                        forecast_status = mode["as_run"].get("status")
                    if forecast_status is None:
                        forecast_status = "valid" if tasks else "missing"
                    if forecast_status not in _STATUSES:
                        raise ValueError("Unknown forecast_status.")
                    forecast_counts[mode_name][forecast_status] += 1
                    samples[f"forecast/{mode_name}/availability"].append(
                        forecast_status in ("valid", "valid_unscored")
                    )
                    for name in _STATUSES:
                        samples[f"forecast/{mode_name}/{name}_rate"].append(
                            forecast_status == name
                        )
                    for task_name, task in tasks.items():
                        if task is None:
                            continue
                        prefix = f"task/{mode_name}/{task_name}"
                        eligible, labeled = (
                            task.get("eligible_targets"),
                            task.get("labeled_targets"),
                        )
                        if eligible is not None or labeled is not None:
                            if (
                                type(eligible) is not int
                                or type(labeled) is not int
                                or not 0 <= labeled <= eligible
                            ):
                                raise ValueError(
                                    "Task counts require 0 <= labeled_targets <= eligible_targets."
                                )
                            target_counts[prefix].update(
                                eligible_targets=eligible, labeled_targets=labeled
                            )
                            samples[f"{prefix}/label_coverage"].append(
                                labeled / eligible if eligible else None
                            )
                        for score in ("brier", "log_loss"):
                            value = task.get(score)
                            infinite = (
                                score == "log_loss"
                                and task.get("infinite_log_loss") is True
                            )
                            if value is not None and (
                                type(value) not in (int, float)
                                or not math.isfinite(value)
                                or value < 0
                            ):
                                raise ValueError(
                                    "Scores must be finite and nonnegative; use infinite_log_loss for infinity."
                                )
                            if forecast_status != "valid" or labeled == 0:
                                value, infinite = None, False
                            samples[f"{prefix}/{score}"].append(
                                math.inf if infinite else value
                            )
            for identity, actions in group["actions"].items():
                samples["action/checkpoint_available_rate"].append(
                    identity in group["checkpoints"]
                )
                for field in (
                    "selected",
                    "adapter_returned",
                    "submission_intent",
                    "discarded",
                    "runtime_fallback",
                    "discussion_termination",
                ):
                    samples[f"action/{field}_rate"].append(
                        any(action.get(field) is True for action in actions)
                    )
                for field in ("valid_action", "acknowledged"):
                    samples[f"action/{field}_rate"].append(
                        any(action.get(field) is True for action in actions)
                    )
                    samples[f"action/{field}_known_rate"].append(
                        any(type(action.get(field)) is bool for action in actions)
                    )
                for source in ("high", "low", "agent_fallback", "planner"):
                    samples[f"action/selected_{source}_rate"].append(
                        any(
                            action.get("selected_source") == source
                            for action in actions
                        )
                    )
            samples["outcome/decisive_win_rate"].append(
                outcome == "win" if outcome in ("win", "loss") else None
            )
            samples["outcome/draw_rate"].append(
                outcome == "draw"
                if status == "completed" and outcome != "unknown"
                else None
            )
            unit = {
                "agent": agent,
                "player_id": player_id,
                "status": status,
                "outcome": outcome,
                "checkpoint_count": len(group["checkpoints"]),
                "duplicate_checkpoint_count": sum(
                    len(entry["provenance"]) - 1
                    for entry in group["checkpoints"].values()
                ),
                "checkpoints": list(group["checkpoints"].values()),
                "action_count": len(group["actions"]),
                "selection_count": sum(
                    any(action.get("selected") is True for action in actions)
                    for actions in group["actions"].values()
                ),
                "submission_intent_count": sum(
                    any(action.get("submission_intent") is True for action in actions)
                    for actions in group["actions"].values()
                ),
                "acknowledgment_count": sum(
                    any(action.get("acknowledged") is True for action in actions)
                    for actions in group["actions"].values()
                ),
                "action_provenance": list(group["actions"].values()),
                "forecast_counts": {
                    name: dict(counts) for name, counts in forecast_counts.items()
                },
                "target_counts": {
                    name: dict(counts) for name, counts in target_counts.items()
                },
                "metrics": {name: _mean(values) for name, values in samples.items()},
            }
            grouped[(match_lineups[match_id], cohort)][match_id].append(unit)
    for (lineup, cohort), matches in sorted(grouped.items()):
        units = [
            {
                "match_id": match_id,
                "seat_count": len(units),
                "seats": units,
                "statuses": dict(Counter(unit["status"] for unit in units)),
                "outcomes": dict(Counter(unit["outcome"] for unit in units)),
                "metrics": _combine([unit["metrics"] for unit in units]),
            }
            for match_id, units in sorted(matches.items())
        ]
        lineups[lineup]["cohorts"].append(
            {
                **json.loads(cohort),
                "match_count": len(units),
                "matches": units,
                "metrics": _combine([unit["metrics"] for unit in units]),
            }
        )
    return {
        "policy": {
            "weighting": "equal checkpoints within seat; equal available seats within match; equal available matches within lineup/cohort, per metric",
            "deduplication": "first checkpoint per occurrence ID/kind/cohort; repeated occurrence action evidence union; IDs include attempt identity unless an explicit shared ID proves a retry; state keys never deduplicate invocations; duplicate provenance retained",
            "win_rate_denominator": "completed wins + losses only; draws, unknown outcomes, and incomplete attempts excluded",
            "uncertainty": "descriptive match units only; no independent-seat confidence intervals",
            "action_rates": "observed decision opportunities; valid/acknowledged rates count confirmed evidence only, with known-evidence rates alongside",
        },
        "match_count": len(labels),
        "lineups": lineups,
        "excluded_reports": excluded,
        "report_provenance": report_provenance,
    }
