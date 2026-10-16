"""API-free replay of finalized, workspace-owned seat archives.

Schema 2 exposes ``attempts`` and invocation-specific ``checkpoints``. Each checkpoint
retains its original version/configuration, decision identity and frozen snapshot.
Its ``modes`` include hard/energy/csp and, for LLM decisions, llm_raw/llm_conditioned.
Each mode has ``as_run``, ``recomputed`` and ``discrepancy_status``; legacy metric
keys alias as_run, never recomputation. Status separates valid, invalid, missing,
fallback_no_score and valid_unscored (a forecast without a usable snapshot).
Label coverage is independent of forecast availability. Unpaired decisions and
skipped traces/checkpoints are disclosed under attempts, not silently scored.
Checkpoint-linked action-path event references bound forecast pairing. Historical
adapter-only evidence is labeled ``adapter_return``; it never fabricates a
selected_decision event or selection timestamp.
"""

import hashlib
import json
import math
import subprocess
from pathlib import Path

from .belief import ROLES, BeliefEngine, Facts
from .evaluation import binary_metrics, role_metrics


def _load_finalized(path: Path, workspace: Path) -> tuple[Path, dict]:
    path = path.resolve(strict=True)
    if not path.is_relative_to((workspace / "data" / "logs").resolve(strict=True)):
        raise ValueError(
            "Replay accepts only finalized logs in this workspace's data/logs."
        )
    # Python's JSON reader otherwise accepts NaN/Infinity and can leak them back
    # through action-path evidence. Keep invalid values explicit, never normalized.
    log = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: f"[INVALID_NONFINITE:{value}]",
    )
    if log.get("status") not in ("completed", "failed", "aborted"):
        raise ValueError("Only finalized attempts can be reviewed.")
    return path, log


def inspect_log(path: Path, workspace: Path) -> dict:
    """A focused post-attempt report without credentials or hidden reasoning."""
    path, log = _load_finalized(path, workspace)
    details = []
    for event in log.get("events", []):
        name, data = event["event"], event["data"]
        if name == "model_request":
            details.append(
                {
                    "event": name,
                    "call_id": data["call_id"],
                    "effort": data.get(
                        "effort", data["payload"].get("reasoning_effort")
                    ),
                    "max_tokens": data["payload"]["max_tokens"],
                    "deadline_seconds": data.get("deadline_seconds"),
                    **{
                        field: data[field]
                        for field in ("extraction_job_id", "occurred_at")
                        if field in data
                    },
                }
            )
        elif name == "model_response":
            details.append(
                {
                    "event": name,
                    "call_id": data["call_id"],
                    "effort": data["effort"],
                    "latency_ms": data["latency_ms"],
                    "usage": data["usage"],
                    "parsed": data["parsed"],
                    **{
                        field: data[field]
                        for field in ("extraction_job_id", "occurred_at")
                        if field in data
                    },
                }
            )
        elif name == "belief_checkpoint":
            details.append(
                {
                    "event": name,
                    **{
                        field: data.get(field)
                        for field in ("key", "kind", "ledger_revision")
                    },
                }
            )
        elif name in (
            "policy_dispatch",
            "provider_error",
            "model_error",
            "model_cancelled",
            "extraction_error",
            "extraction",
            "candidate_error",
            "integration_error",
            "integration_checks",
        ):
            details.append({"event": name, "data": data})
    report = {
        "log_path": path.relative_to(workspace).as_posix(),
        "version": log["version"],
        "kind": log["kind"],
        "status": log["status"],
        "configuration": log["configuration"],
        "details": details,
    }
    from .reporting import action_path_report

    report["action_path_report"] = action_path_report(log)
    return report


def _attempts(log: dict) -> list[tuple[dict, list]]:
    """Keep trace boundaries and original metadata, including rejected traces."""
    attempts = []
    seen = set()
    for index, trace in enumerate([log, *log.get("recovered_attempts", [])]):
        metadata = trace if index == 0 else trace.get("metadata", {})
        events = trace.get("events", [])
        report = {
            key: metadata.get(key)
            for key in (
                "attempt_id",
                "match_id",
                "agent",
                "player_id",
                "version",
                "configuration",
                "strategy",
                "experiment_id",
                "lineup_type",
                "test_mode",
                "kind",
                "status",
                "outcome",
                "won",
            )
        }
        report.update(
            trace_id="top_level" if index == 0 else f"recovered_attempts:{index - 1}",
            recovered=index != 0,
            version_mismatch=trace.get("version_mismatch", False),
            configuration_mismatch=trace.get("configuration_mismatch", False),
            computed_version_mismatch=metadata.get("version") != log.get("version"),
            computed_configuration_mismatch=metadata.get("configuration")
            != log.get("configuration"),
            included=True,
            skipped_checkpoints=[],
        )
        for key in (
            "source_journal",
            "source_start",
            "trace_missing_reason",
            "integrity",
        ):
            if key in trace:
                report[key] = trace[key]
        # Preserve writer integrity flags without importing its private reader.
        if index:
            report["trace_metadata"] = {
                key: value
                for key, value in trace.items()
                if key not in ("events", "metadata")
            }
        report["identities"] = []
        for event in events:
            data = event.get("data", {})
            if event.get("event") == "observation":
                data = data.get("state", {})
            if event.get("event") not in (
                "assignment",
                "state",
                "observation",
                "grant",
                "result",
            ):
                continue
            identity = {
                key: data[key]
                for key in (
                    "session_id",
                    "game_session_id",
                    "match_id",
                    "seat_id",
                    "player_id",
                    "your_position",
                    "seat_position",
                    "agent_id",
                )
                if key in data
            }
            if identity and identity not in report["identities"]:
                report["identities"].append(identity)
        identity = metadata.get("attempt_id")
        reason = None
        if identity is not None and identity in seen:
            reason = "duplicate_attempt"
        elif index and any(
            metadata.get(key) != log.get(key)
            for key in ("match_id", "agent", "player_id")
        ):
            reason = "archive_identity_mismatch"
        elif any(
            event.get("event") == "start"
            and any(
                event.get("data", {}).get(key) != metadata.get(key)
                for key in (
                    "attempt_id",
                    "match_id",
                    "agent",
                    "player_id",
                    "version",
                    "configuration",
                )
                if key in event.get("data", {})
            )
            for event in events
        ):
            reason = "start_metadata_mismatch"
        if identity is not None:
            seen.add(identity)
        if reason:
            report.update(included=False, skip_reason=reason)
        report["event_count"] = len(events)
        attempts.append((report, events))
    reference_sessions = {
        identity[field]
        for identity in attempts[0][0]["identities"]
        for field in ("session_id", "game_session_id")
        if field in identity
    }
    for report, _ in attempts:
        sessions = {
            identity[field]
            for identity in report["identities"]
            for field in ("session_id", "game_session_id")
            if field in identity
        }
        if report["included"] and (
            len(sessions) > 1
            or (reference_sessions and sessions and sessions != reference_sessions)
        ):
            report.update(included=False, skip_reason="session_identity_mismatch")
    return attempts


def _labels(log: dict, attempts: list) -> tuple[dict, dict]:
    role_labels, faction_labels = {}, {}
    terminal = next(
        (
            event["data"]
            for attempt, events in attempts
            if attempt["included"]
            for event in reversed(events)
            if event.get("event") == "state"
            and event["data"].get("is_terminal") is True
        ),
        None,
    )
    if terminal is not None:
        names = {
            player["name"]: player["position"] for player in terminal.get("players", [])
        }
        public = terminal.get("game_state") or {}
        for death in public.get("dead", []):
            if death.get("player") in names and death.get("role") in (
                "wolf",
                "seer",
                "villager",
            ):
                role_labels[names[death["player"]]] = death["role"]
                faction_labels[names[death["player"]]] = death["role"] == "wolf"
        resigned = {
            death["player"]
            for death in public.get("dead", [])
            if death.get("cause") == "resigned"
        }
        winner = public.get("winner")
        for name, score in ((log.get("final") or {}).get("returns") or {}).items():
            if (
                name in names
                and name not in resigned
                and winner in ("wolf", "villager")
                and score in (-1, 1)
            ):
                faction_labels[names[name]] = (winner == "wolf") == (score == 1)
    return role_labels, faction_labels


def _eligibility(hard: dict, alive: list[int]) -> dict:
    # Membership in legal worlds, never positive/zero floating-point marginals.
    supported = {
        seat: {
            "wolf"
            if seat in world["wolves"]
            else "seer"
            if seat == world["seer"]
            else "villager"
            for world in hard["worlds"]
        }
        for seat in alive
    }
    return {
        "role": [seat for seat, roles in supported.items() if len(roles) > 1],
        "wolf": [
            seat
            for seat, roles in supported.items()
            if "wolf" in roles and roles != {"wolf"}
        ],
        "seer": [
            seat
            for seat, roles in supported.items()
            if "seer" in roles and roles != {"seer"}
        ],
        "supported_roles": {
            str(seat): sorted(roles) for seat, roles in supported.items()
        },
    }


def _validate_rows(rows: dict) -> None:
    if not isinstance(rows, dict) or set(rows) != {str(seat) for seat in range(7)}:
        raise ValueError("Forecast requires all seven seats.")
    for row in rows.values():
        if not isinstance(row, dict) or set(row) != set(ROLES):
            raise ValueError("Forecast requires all three role probabilities.")
        if any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in row.values()
        ) or not math.isclose(math.fsum(row.values()), 1, abs_tol=1e-6):
            raise ValueError(
                "Invalid recorded probabilities; no normalization was applied."
            )


def _scores(
    rows: dict | None, facts: Facts | None, kind: str, eligible: dict, labels: tuple
) -> dict:
    if facts is None:
        return {
            name: None
            for name in (
                "unknown_living_role_metrics",
                "unknown_living_wolf_metrics",
                "wolf_seer_identification",
                "wolf_faction_inference",
            )
        }
    roles, factions = labels
    action, wolf = kind == "action", facts.role == "wolf"

    def measure(targets, known, role=None):
        if rows is not None:
            return (
                binary_metrics(rows, known, targets, role)
                if role
                else role_metrics(rows, known, targets)
            )
        count = sum(seat in known for seat in targets)
        return {
            "eligible_targets": len(targets),
            "labeled_targets": count,
            "label_coverage": count / len(targets) if targets else None,
            "brier": None,
            "log_loss": None,
            "infinite_log_loss": False,
        }

    return {
        "unknown_living_role_metrics": measure(eligible["role"], roles)
        if action and not wolf
        else None,
        "unknown_living_wolf_metrics": measure(eligible["wolf"], factions, "wolf")
        if action and not wolf
        else None,
        "wolf_seer_identification": measure(
            eligible["seer"],
            {seat: role == "seer" for seat, role in roles.items()},
            "seer",
        )
        if action and wolf
        else None,
        "wolf_faction_inference": "N/A: faction is privately known" if wolf else None,
    }


def _forecast(
    recorded, recomputed, facts, kind, eligible, labels, evaluator, missing="missing"
) -> dict:
    def evaluate(distribution, source):
        if distribution is None:
            return {
                "status": missing if source == "recorded" else "unavailable",
                **_scores(None, facts, kind, eligible, labels),
            }
        rows = distribution.get("marginals") if isinstance(distribution, dict) else None
        try:
            _validate_rows(rows)
        except ValueError as error:
            return {
                "status": "invalid",
                "reason": str(error),
                **_scores(None, facts, kind, eligible, labels),
            }
        result = {
            "status": "valid" if facts is not None else "valid_unscored",
            "prediction_source": source,
            "marginals": rows,
            "feasible_worlds": distribution.get("feasible_worlds"),
            "entropy_bits": distribution.get("entropy"),
            "world_mass": math.fsum(w["probability"] for w in distribution["worlds"])
            if isinstance(distribution.get("worlds"), list)
            and distribution["worlds"]
            and all(
                isinstance(w, dict)
                and type(w.get("probability")) in (int, float)
                and math.isfinite(w["probability"])
                and 0 <= w["probability"] <= 1
                for w in distribution["worlds"]
            )
            else None,
            "wolf_mass": math.fsum(row["wolf"] for row in rows.values()),
            "seer_mass": math.fsum(row["seer"] for row in rows.values()),
            **_scores(rows, facts, kind, eligible, labels),
        }
        if source == "recomputed":
            result["evaluator_version"] = evaluator
        return result

    as_run = evaluate(recorded, "recorded")
    rescore = evaluate(recomputed, "recomputed")
    error = None
    if as_run["status"] == rescore["status"] == "valid":
        error = max(
            abs(as_run["marginals"][seat][role] - rescore["marginals"][seat][role])
            for seat in as_run["marginals"]
            for role in ROLES
        )
    return {
        # Existing metric/statistic keys now consistently mean recorded as-run.
        **{key: value for key, value in as_run.items() if key != "marginals"},
        "as_run": as_run,
        "recomputed": rescore,
        "maximum_recorded_marginal_error": error,
        "discrepancy_status": "unavailable"
        if error is None
        else "match"
        if error <= 1e-9
        else "different",
    }


def _key(data: dict, observation: dict | None = None):
    explicit = data.get("key", data.get("decision_key"))
    if explicit is not None and observation is None:
        return list(explicit)
    if observation is None:
        return None
    from werewolf_runtime.platform import state_key

    # A later snapshot must not lend its identity to an earlier checkpoint.
    public = observation.get("game_state") or {}
    if any(
        key in data and data[key] != value
        for key, value in (
            ("state_version", observation.get("state_version")),
            ("day", public.get("day")),
            ("subphase", public.get("phase")),
        )
    ):
        return None
    if data.get("facts", {}).get(
        "self_id", observation.get("your_position")
    ) != observation.get("your_position"):
        return None
    observed = list(state_key(observation))
    return observed if explicit is None or list(explicit) == observed else None


def _llm_recordings(llm: dict, answer: dict) -> tuple:
    raw = llm.get("role_scores", llm.get("raw_scores"))
    if raw is None:
        raw = answer.get("decision", {}).get("role_scores")
    conditioned = llm.get(
        "conditioned_distribution",
        llm.get("distribution", llm if "marginals" in llm else None),
    )
    source = llm.get("selected_source", llm.get("source", answer.get("source")))
    missing = "fallback_no_score" if source == "fallback" and raw is None else "missing"
    return raw, conditioned, missing


def _compatible(data: dict, selected: dict, key) -> bool:
    selected_key = _key(selected)
    if key is not None and selected_key is not None and key != selected_key:
        return False
    if (
        selected_key is not None
        and len(selected_key) == 8
        and any(
            value is not None and value != selected_key[position]
            for position, value in (
                (1, data.get("state_version")),
                (3, data.get("day")),
                (4, data.get("subphase")),
                (5, data.get("facts", {}).get("self_id")),
            )
        )
    ):
        return False
    return all(
        data.get(field) is None
        or selected.get(field) is None
        or data[field] == selected[field]
        for field in (
            "kind",
            "ledger_revision",
            "state_version",
            "day",
            "subphase",
            "facts",
            "alive",
        )
    )


def _evaluator_version() -> dict:
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("replay.py", "belief.py", "evaluation.py", "agent.py", "reporting.py"):
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        commit = None
    return {"git_commit": commit, "source_sha256": digest.hexdigest()}


def replay(path: Path, workspace: Path) -> dict:
    """Score recorded forecasts, with current-engine rescoring explicitly separate.

    Checkpoints are weighted nowhere here. Reporting must first aggregate within
    matches; selected forecasts are not evidence of platform submission.
    """
    path, log = _load_finalized(path, workspace)
    from .reporting import action_path_report

    attempts = _attempts(log)
    action_paths = action_path_report(log)
    checkpoint_actions = {}
    for action in action_paths["decisions"]:
        for reference in action["events"]:
            if reference["event"] in ("belief_checkpoint", "llm_belief_checkpoint"):
                checkpoint_actions.setdefault(
                    (action["trace_id"], reference["event_index"]), []
                ).append(action)
    labels = _labels(log, attempts)
    evaluator = _evaluator_version()
    checkpoints, seen_explicit_ids = [], {}
    for attempt, events in attempts:
        if not attempt["included"]:
            continue
        observation = None
        observation_index, active_checkpoint = None, None
        invocation_counts = {}
        used_llm, used_selected, used_adapter = set(), set(), set()
        # Self-contained LLM checkpoints are also replayable without a shadow checkpoint.
        for index, event in enumerate(events):
            name, data = event.get("event"), event.get("data", {})
            if name == "observation":
                observation = data.get("state")
                observation_index, active_checkpoint = index, None
            if (name == "selected_decision" and index in used_selected) or (
                name == "adapter_decision" and index in used_adapter
            ):
                active_checkpoint = None
            standalone = (
                name == "llm_belief_checkpoint"
                and index not in used_llm
                and "facts" in data
                and active_checkpoint is None
            )
            if name != "belief_checkpoint" and not standalone:
                continue
            active_checkpoint = index
            end = next(
                (
                    j
                    for j in range(index + 1, len(events))
                    if events[j].get("event")
                    in (
                        "belief_checkpoint",
                        "observation",
                        "adapter_decision",
                        "submission",
                        "decision_discarded",
                    )
                ),
                len(events),
            )
            # Adapter return, submission, and discard close this invocation even
            # when the next call fails before emitting an observation/checkpoint.
            if end < len(events) and events[end].get("event") in (
                "adapter_decision",
                "submission",
                "decision_discarded",
            ):
                end += 1
            block = list(enumerate(events[index + 1 : end], index + 1))
            linked = checkpoint_actions.get((attempt["trace_id"], index), [])
            occurrence_indices = (
                {reference["event_index"] for reference in linked[0]["events"]}
                if len(linked) == 1
                else set()
            )
            key = _key(data) or _key(data, observation)
            selections = [
                (i, e["data"])
                for i, e in block
                if e.get("event") == "selected_decision"
                and i in occurrence_indices
                and i not in used_selected
                and _compatible(data, e["data"], key)
            ]
            selected_index, selected = (
                selections[0] if len(selections) == 1 else (None, {})
            )
            adapters = [
                (
                    i,
                    {
                        **e["data"]["answer"],
                        "kind": e["data"].get("kind", e["data"]["answer"].get("kind")),
                    },
                )
                for i, e in block
                if e.get("event") == "adapter_decision"
                and i in occurrence_indices
                and i not in used_adapter
                and _compatible(data, e["data"]["answer"], key)
            ]
            adapter_index, adapter = adapters[0] if len(adapters) == 1 else (None, {})
            # A receipt is useful historical evidence, not a fabricated selection.
            answer = (
                selected
                if selected_index is not None
                else adapter
                if not selections
                else {}
            )
            answer_evidence = (
                "selected_decision"
                if selected_index is not None
                else "adapter_return"
                if answer
                else None
            )
            llms = [
                (i, e["data"])
                for i, e in block
                if e.get("event") == "llm_belief_checkpoint"
                and i in occurrence_indices
                and _compatible(data, e["data"], key)
                and (
                    "facts" in e["data"]
                    or _key(e["data"]) is not None
                    or answer
                    or not any(
                        other.get("event") == "selected_decision"
                        and not _compatible(data, other["data"], key)
                        for _, other in block
                    )
                )
            ]
            if standalone:
                llms = [(index, data)]
            llm_index, llm = llms[0] if len(llms) == 1 else (None, {})
            if selected_index is not None:
                used_selected.add(selected_index)
            if adapter_index is not None:
                used_adapter.add(adapter_index)
            if llm_index is not None:
                used_llm.add(llm_index)
            kind = data.get("kind", data.get("decision_kind", answer.get("kind")))
            key = key if key is not None else _key(llm) or _key(answer)
            checkpoint_id = (
                data.get("checkpoint_id")
                or f"{attempt['attempt_id'] or attempt['trace_id']}:{index}"
            )
            explicit_id = data.get("decision_id") or data.get("checkpoint_id")
            duplicate_key = json.dumps(
                [attempt["match_id"], attempt["player_id"], kind, explicit_id],
                sort_keys=True,
            )
            # A freshness key identifies a game snapshot, not a policy invocation.
            # Identical keys, snapshots, or timestamps alone prove no retry.
            if explicit_id is not None and duplicate_key in seen_explicit_ids:
                attempt["skipped_checkpoints"].append(
                    {
                        "event_index": index,
                        "reason": "duplicate_explicit_decision_id",
                        "duplicate_of": seen_explicit_ids[duplicate_key],
                        "decision_key": key,
                        "decision_id": explicit_id,
                    }
                )
                continue
            if explicit_id is not None:
                seen_explicit_ids[duplicate_key] = checkpoint_id
            invocation_key = json.dumps([kind, key], sort_keys=True)
            ordinal = invocation_counts.get(invocation_key, 0)
            invocation_counts[invocation_key] = ordinal + 1
            row = {
                field: attempt[field]
                for field in (
                    "match_id",
                    "agent",
                    "player_id",
                    "attempt_id",
                    "trace_id",
                    "version",
                    "configuration",
                    "strategy",
                    "experiment_id",
                    "lineup_type",
                    "test_mode",
                    "version_mismatch",
                    "configuration_mismatch",
                )
            }
            row.update(
                checkpoint_id=checkpoint_id,
                decision_id=explicit_id if explicit_id is not None else checkpoint_id,
                invocation_ordinal=ordinal,
                observation_event_index=observation_index,
                event_index=index,
                timestamp=event.get("timestamp"),
                kind=kind,
                decision_key=key,
                day=data.get("day"),
                state_version=data.get("state_version"),
                subphase=data.get("subphase"),
                ledger_revision=data.get(
                    "ledger_revision",
                    llm.get("ledger_revision", answer.get("ledger_revision")),
                ),
                selected_source=llm.get(
                    "selected_source", llm.get("source", answer.get("source"))
                ),
                selected_decision_event_index=selected_index,
                adapter_decision_event_index=adapter_index,
                decision_evidence=answer_evidence,
                selection_recorded=bool(selections),
                adapter_return_recorded=bool(adapters),
                source_evidence="llm_checkpoint"
                if llm.get("selected_source", llm.get("source")) is not None
                else answer_evidence
                if answer.get("source") is not None
                else None,
                raw_scores_evidence="llm_checkpoint"
                if llm.get("role_scores", llm.get("raw_scores")) is not None
                else answer_evidence
                if answer.get("decision", {}).get("role_scores") is not None
                else None,
                llm_checkpoint_event_index=llm_index,
                pairing_status="unambiguous"
                if answer
                else "ambiguous"
                if len(selections) > 1 or len(adapters) > 1
                else "missing",
                facts=data.get("facts"),
                alive=data.get("alive"),
                modes={},
            )
            for field in ("day", "state_version", "subphase"):
                if row[field] is None:
                    row[field] = llm.get(field)
            if key is not None and len(key) == 8:
                for field, position in (
                    ("state_version", 1),
                    ("day", 3),
                    ("subphase", 4),
                ):
                    if row[field] is None:
                        row[field] = key[position]
            try:
                facts_data = dict(data["facts"])
                for field in ("investigations", "revealed"):
                    facts_data[field] = {
                        int(seat): role
                        for seat, role in facts_data.get(field, {}).items()
                    }
                facts = Facts(**facts_data)
                if (
                    key is not None
                    and len(key) == 8
                    and (
                        key[5] != facts.self_id
                        or any(
                            data.get(field) is not None and data[field] != key[position]
                            for field, position in (
                                ("state_version", 1),
                                ("day", 3),
                                ("subphase", 4),
                            )
                        )
                    )
                ):
                    raise ValueError(
                        "Checkpoint identity contradicts its frozen snapshot."
                    )
                engine = BeliefEngine(facts)
                hard = engine.distribution("hard")
                alive = data.get("alive")
                if (
                    alive is None
                    and observation is not None
                    and _key(data, observation) is not None
                ):
                    names = {
                        p["name"]: p["position"] for p in observation.get("players", [])
                    }
                    alive = [
                        names[name]
                        for name in (observation.get("game_state") or {}).get(
                            "alive", []
                        )
                    ]
                if alive is None:
                    raise ValueError("Missing matching alive snapshot.")
                if (
                    not isinstance(alive, (list, tuple))
                    or len(set(alive)) != len(alive)
                    or any(type(s) is not int or not 0 <= s < 7 for s in alive)
                ):
                    raise ValueError("Invalid alive snapshot.")
                row.update(
                    facts=facts_data,
                    alive=list(alive),
                    self_id=facts.self_id,
                    eligibility=_eligibility(hard, alive),
                    scoring_status="available",
                )
            except (KeyError, TypeError, ValueError) as error:
                row.update(
                    scoring_status="invalid_or_missing_snapshot",
                    snapshot_error=str(error),
                )
                for mode, recorded in data.get("distributions", {}).items():
                    row["modes"][mode] = _forecast(
                        recorded, None, None, kind, {}, labels, evaluator
                    )
                if attempt["strategy"] == "llm" or llm:
                    raw, conditioned, missing = _llm_recordings(llm, answer)
                    for mode, recorded in (
                        ("llm_raw", {"marginals": raw} if raw is not None else None),
                        ("llm_conditioned", conditioned),
                    ):
                        row["modes"][mode] = _forecast(
                            recorded, None, None, kind, {}, labels, evaluator, missing
                        )
                    row["llm_availability"] = {
                        mode: row["modes"][mode]["status"]
                        for mode in ("llm_raw", "llm_conditioned")
                    }
                checkpoints.append(row)
                continue
            eligible = row["eligibility"]
            row["label_coverage"] = {
                task: {
                    field: metrics[field]
                    for field in (
                        "eligible_targets",
                        "labeled_targets",
                        "label_coverage",
                    )
                }
                for task, metrics in (
                    (
                        "role",
                        role_metrics(hard["marginals"], labels[0], eligible["role"]),
                    ),
                    (
                        "wolf",
                        binary_metrics(
                            hard["marginals"], labels[1], eligible["wolf"], "wolf"
                        ),
                    ),
                    (
                        "seer",
                        binary_metrics(
                            hard["marginals"],
                            {s: r == "seer" for s, r in labels[0].items()},
                            eligible["seer"],
                            "seer",
                        ),
                    ),
                )
            }
            if name == "belief_checkpoint":
                modes = ("hard", "energy", "csp") + (
                    ("assertion",)
                    if "assertion" in data.get("distributions", {})
                    else ()
                )
                for mode in modes:
                    row["modes"][mode] = _forecast(
                        data.get("distributions", {}).get(mode),
                        engine.distribution(mode, data.get("events", [])),
                        facts,
                        kind,
                        eligible,
                        labels,
                        evaluator,
                    )
            if (
                attempt["strategy"] == "llm"
                or llm
                or answer.get("decision", {}).get("role_scores") is not None
            ):
                raw, conditioned, missing = _llm_recordings(llm, answer)
                row["modes"]["llm_raw"] = _forecast(
                    {"marginals": raw} if raw is not None else None,
                    None,
                    facts,
                    kind,
                    eligible,
                    labels,
                    evaluator,
                    missing,
                )
                raw_error = None
                if row["modes"]["llm_raw"]["status"] == "valid":
                    for seat, supported in _eligibility(hard, range(7))[
                        "supported_roles"
                    ].items():
                        if len(supported) == 1 and not math.isclose(
                            raw[seat][supported[0]], 1, abs_tol=1e-6
                        ):
                            raw_error = "Raw forecast contradicts authenticated hard-world support."
                if raw_error is not None:
                    invalid = {
                        "status": "invalid",
                        "reason": raw_error,
                        **_scores(None, facts, kind, eligible, labels),
                    }
                    row["modes"]["llm_raw"].update(invalid, as_run=invalid)
                projected, projection_error = None, None
                if row["modes"]["llm_raw"]["status"] == "valid":
                    from .agent import project_llm_scores
                    from .model import ModelError

                    try:
                        projected = project_llm_scores(raw, hard)
                    except (ValueError, ModelError) as error:
                        projection_error = str(error)
                row["modes"]["llm_conditioned"] = _forecast(
                    conditioned,
                    projected,
                    facts,
                    kind,
                    eligible,
                    labels,
                    evaluator,
                    missing,
                )
                if projection_error is not None:
                    row["modes"]["llm_conditioned"]["recomputed"].update(
                        status="invalid",
                        reason=projection_error,
                        evaluator_version=evaluator,
                    )
                row["llm_availability"] = {
                    mode: row["modes"][mode]["status"]
                    for mode in ("llm_raw", "llm_conditioned")
                }
                row["candidate_error_count"] = sum(
                    e.get("event") == "candidate_error" for _, e in block
                )
                row["candidate_failures"] = [
                    {
                        "event_index": i,
                        "effort": e["data"].get("effort"),
                        "error": e["data"].get("error"),
                    }
                    for i, e in block
                    if e.get("event") == "candidate_error"
                ]
            checkpoints.append(row)
        attempt["unpaired_llm_checkpoint_indices"] = [
            i
            for i, e in enumerate(events)
            if e.get("event") == "llm_belief_checkpoint" and i not in used_llm
        ]
        attempt["unpaired_llm_forecasts"] = []
        for i in attempt["unpaired_llm_checkpoint_indices"]:
            data = events[i]["data"]
            raw, conditioned, missing = _llm_recordings(data, {})
            attempt["unpaired_llm_forecasts"].append(
                {
                    "event_index": i,
                    "decision_key": _key(data),
                    "scoring_status": "missing_or_ambiguous_invocation_snapshot",
                    "counted_for_scoring": False,
                    "modes": {
                        mode: _forecast(
                            recorded,
                            None,
                            None,
                            data.get("kind"),
                            {},
                            labels,
                            evaluator,
                            missing,
                        )
                        for mode, recorded in (
                            (
                                "llm_raw",
                                {"marginals": raw} if raw is not None else None,
                            ),
                            ("llm_conditioned", conditioned),
                        )
                    },
                }
            )
        attempt["unpaired_selected_decision_indices"] = [
            i
            for i, e in enumerate(events)
            if e.get("event") == "selected_decision" and i not in used_selected
        ]
        attempt["unpaired_adapter_decision_indices"] = [
            i
            for i, e in enumerate(events)
            if e.get("event") == "adapter_decision" and i not in used_adapter
        ]
        attempt["unpaired_decisions"] = []
        for i in attempt["unpaired_selected_decision_indices"]:
            data = events[i]["data"]
            raw = data.get("decision", {}).get("role_scores")
            status = (
                "fallback_no_score"
                if data.get("source") == "fallback" and raw is None
                else "missing"
            )
            if raw is not None:
                try:
                    _validate_rows(raw)
                    status = "valid_unscored"
                except ValueError:
                    status = "invalid"
            attempt["unpaired_decisions"].append(
                {
                    "event_index": i,
                    "decision_key": _key(data),
                    "kind": data.get("kind"),
                    "selected_source": data.get("source"),
                    "ledger_revision": data.get("ledger_revision"),
                    "scoring_status": "missing_or_ambiguous_checkpoint",
                    "counted_for_scoring": False,
                    "llm_raw_status": status
                    if attempt["strategy"] == "llm"
                    else "not_applicable",
                }
            )
            if attempt["strategy"] == "llm":
                attempt["unpaired_decisions"][-1]["llm_availability"] = {
                    "llm_raw": status,
                    "llm_conditioned": "fallback_no_score"
                    if data.get("source") == "fallback" and raw is None
                    else "missing",
                }
        attempt["checkpoint_count"] = sum(
            row["trace_id"] == attempt["trace_id"] for row in checkpoints
        )
    return {
        "schema_version": 2,
        "log_path": path.relative_to(workspace).as_posix(),
        "version": log["version"],
        "agent": log.get("agent"),
        "attempt_id": log.get("attempt_id"),
        "strategy": log.get("strategy"),
        "configuration": log.get("configuration"),
        "revision": log.get("revision", 1),
        "kind": log["kind"],
        "match_id": log.get("match_id"),
        "player_id": log.get("player_id"),
        "status": log["status"],
        "outcome": log.get("outcome"),
        "won": log.get("won"),
        "lineup_type": log.get("lineup_type", "unknown"),
        "evaluator_version": evaluator,
        "attempts": [attempt for attempt, _ in attempts],
        "checkpoints": checkpoints,
        "action_path_report": action_paths,
        "score_source": "validated_recorded_predictions",
        "checkpoint_weighting": "none; checkpoint-level diagnostics, not independent match samples",
        "deduplication_policy": "only duplicate attempt IDs or explicit recorded decision/checkpoint IDs establish duplicates; freshness keys never establish invocation identity; checkpoint_id is attempt_id:event_index and invocation_ordinal is zero-based within attempt/kind/key",
        "ground_truth_scoring": "Only post-game authorized reveals/faction returns, on living targets unresolved for each task in hard-world support; unknown special-role labels stay unknown.",
        "counterfactual_wins": None,
    }
