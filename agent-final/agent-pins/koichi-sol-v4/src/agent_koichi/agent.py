"""Facts -> cached extraction -> exact beliefs -> deadline-bounded LLM policy."""

import asyncio
import json
import math
import time
from copy import deepcopy

from werewolf_runtime.platform import state_key

from .belief import BeliefEngine, EvidenceError, SeatView
from .config import Config, ConfigurationError
from .extraction import ExtractionJob
from .ledger import ClaimLedger, authorized_message
from .model import ChatModel, ModelError
from .planner import plan
from .prompts import POLICY
from .schemas import policy_response_format


def validate_decision(output: dict, view: SeatView, kind: str, strategy: str) -> dict:
    if not isinstance(output, dict) or set(output) != {
        "action_id",
        "content",
        "recipients",
        "reason",
        "role_scores",
    }:
        raise ModelError("Decision has missing or unsupported fields.")
    content, recipients, reason = (
        output["content"],
        output["recipients"],
        output["reason"],
    )
    if not isinstance(content, str) or not isinstance(reason, str) or len(reason) > 600:
        raise ModelError("Invalid decision text.")
    if not isinstance(recipients, list):
        raise ModelError("Invalid recipients.")
    if kind == "action":
        if output["action_id"] not in view.action_ids or content or recipients:
            raise ModelError("Action is not in the current legal action list.")
    elif kind == "message":
        if (
            output["action_id"] is not None
            or len(content) > 1024
            or len(content.split()) > 50
        ):
            raise ModelError("Invalid message or message length.")
        if len(recipients) > 1 or any(
            type(seat) is not int
            or seat not in view.alive
            or seat == view.facts.self_id
            for seat in recipients
        ):
            raise ModelError("Invalid private recipient.")
        if not content.strip() and recipients:
            raise ModelError("Terminate messages cannot be private.")
    else:
        raise ModelError("Unknown decision kind.")
    scores = output["role_scores"]
    if strategy != "llm":
        if scores is not None:
            raise ModelError("The policy cannot replace mathematical beliefs.")
    else:
        project_llm_scores(scores, BeliefEngine(view.facts).distribution("hard"))
    return dict(output)


def project_llm_scores(scores: dict, hard: dict) -> dict:
    """Condition independent role-score products on authenticated legal worlds.

    This does not preserve the input marginals. The conditioned distribution is
    diagnostic/validation output, not a second action-selection stage.
    """
    if not isinstance(scores, dict) or set(scores) != {str(i) for i in range(7)}:
        raise ModelError("LLM baseline requires scores for all seven seats.")
    for seat, row in scores.items():
        if not isinstance(row, dict) or set(row) != {"wolf", "seer", "villager"}:
            raise ModelError("Invalid baseline role score row.")
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in row.values()
        ):
            raise ModelError("Invalid baseline role probability.")
        if not math.isclose(math.fsum(row.values()), 1.0, abs_tol=1e-6):
            raise ModelError("Baseline role score row does not sum to one.")
        if any(
            probability == 1.0 and not math.isclose(row[role], 1.0, abs_tol=1e-6)
            for role, probability in hard["marginals"][seat].items()
        ):
            raise ModelError("Baseline score contradicts authenticated facts.")
    logs = []
    for world in hard["worlds"]:
        logs.append(
            math.fsum(
                math.log(
                    max(
                        1e-6,
                        scores[str(seat)][
                            "wolf"
                            if seat in world["wolves"]
                            else "seer"
                            if seat == world["seer"]
                            else "villager"
                        ],
                    )
                )
                for seat in range(7)
            )
        )
    peak = max(logs)
    weights = [math.exp(value - peak) for value in logs]
    total = math.fsum(weights)
    worlds = [
        {**world, "probability": weight / total}
        for world, weight in zip(hard["worlds"], weights)
    ]
    marginals = {
        str(seat): {
            role: math.fsum(
                world["probability"]
                for world in worlds
                if (
                    "wolf"
                    if seat in world["wolves"]
                    else "seer"
                    if seat == world["seer"]
                    else "villager"
                )
                == role
            )
            for role in ("wolf", "seer", "villager")
        }
        for seat in range(7)
    }
    return {
        "mode": "llm",
        "worlds": worlds,
        "marginals": marginals,
        "feasible_worlds": len(worlds),
        "warnings": [
            "Structurally constrained LLM-only baseline; score floor 1e-6, not calibrated."
        ],
    }


def fallback(raw: dict, kind: str, view: SeatView | None = None) -> dict:
    decision = {
        "action_id": None,
        "content": "",
        "recipients": [],
        "reason": "Deterministic legal deadline fallback.",
        "role_scores": None,
    }
    if kind == "message":
        return decision
    entries = (raw.get("legal_actions") or {}).get("actions") or []
    legal = [
        entry["action_id"]
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("action_id") in tuple(str(i) for i in range(8))
    ]
    if not legal:
        raise EvidenceError("No legal action is available for fallback.")
    pick = legal[0]
    if view is not None:
        beliefs = BeliefEngine(view.facts).distribution("hard")["marginals"]
        targets = [target for target in legal if target != "7"]
        if targets:
            if view.subphase == "night_seer":
                fresh = [
                    target
                    for target in targets
                    if int(target) not in view.facts.investigations
                ]

                def information(target):
                    p = beliefs[target]["wolf"]
                    return (
                        -p * math.log2(p) - (1 - p) * math.log2(1 - p)
                        if 0 < p < 1
                        else 0
                    )

                pick = max(fresh or targets, key=information)
            else:
                if view.subphase == "day_vote" and view.facts.role == "wolf":
                    nonallies = [
                        target
                        for target in targets
                        if int(target)
                        not in (view.facts.self_id, view.facts.fellow_wolf)
                    ]
                    targets = nonallies or targets
                role = "seer" if view.facts.role == "wolf" else "wolf"
                pick = max(targets, key=lambda target: beliefs[target][role])
    decision["action_id"] = pick
    return decision


class WerewolfAgent:
    def __init__(self, config: Config, model: ChatModel, audit) -> None:
        self.config, self.model, self.audit = config, model, audit
        self.ledger = ClaimLedger()
        self.messages: dict[str, dict] = {}
        self.previous_facts = None
        self._identity = None
        self._ledger_revision = 0
        self._extraction: ExtractionJob | None = None
        self._decision_task = None
        self._decision_done = None
        self._closed = False
        self._closing = None

    @property
    def ledger_revision(self):
        return self._ledger_revision

    def _pending(self):
        pending = [
            message
            for source, message in self.messages.items()
            if source not in self.ledger.processed and message.get("type") == "chat"
        ]
        if all(
            type(message.get("seq")) is int and message["seq"] >= 0
            for message in pending
        ):
            pending.sort(key=lambda message: message["seq"])
        return pending

    def _harvest_extraction(self):
        job = self._extraction
        if job is None:
            return
        if (
            job.identity != self._identity
            or job.ledger_revision != self._ledger_revision
            or job.batch != self._pending()[: len(job.batch)]
        ):
            job.cancel("Extraction no longer matches the authenticated pending head.")
        elif not job.task.done() and job.batch != self._pending():
            job.cancel(
                "Unfinished extraction superseded by all currently pending chats."
            )
        # Only a decision owner writes to the live audit stream or ledger.
        job.audit.flush(self.audit)
        if not job.task.done():
            return
        disposition = "discarded"
        if job.discard_reason is None and not job.task.cancelled():
            result = job.task.result()
            if result["error"] is not None:
                self.audit.record(
                    "extraction_error",
                    {"error": result["error"], "extraction_job_id": job.job_id},
                )
                disposition = "failed"
            else:
                try:
                    response = result["response"]
                    events = self.ledger.add_extraction(
                        job.batch,
                        response["parsed"],
                        job.identity[1],
                        job.extraction_input["day"],
                    )
                    self._ledger_revision += 1
                    self.audit.record(
                        "extraction",
                        deepcopy(
                            {
                                "input": job.extraction_input,
                                "cached_output": response["parsed"],
                                "accepted": events,
                                "rejected": self.ledger.rejected,
                                "extraction_job_id": job.job_id,
                                "ledger_revision": self._ledger_revision,
                            }
                        ),
                    )
                    disposition = "committed"
                except (EvidenceError, KeyError, TypeError) as error:
                    self.audit.record(
                        "extraction_error",
                        {
                            "error": str(error)
                            if isinstance(error, EvidenceError)
                            else "Malformed extraction response.",
                            "extraction_job_id": job.job_id,
                        },
                    )
                    disposition = "failed"
        self.audit.record(
            "extraction_job_finished",
            {
                "extraction_job_id": job.job_id,
                "disposition": disposition,
                "reason": job.discard_reason,
            },
        )
        self._extraction = None

    async def close(self):
        if self._closing is None:
            self._closed = True

            async def finish():
                decision = self._decision_task
                decision_done = self._decision_done
                if decision is not None and not decision.done():
                    decision.cancel()
                    await decision_done.wait()
                job = self._extraction
                if job is not None:
                    job.cancel("Agent closed before extraction commit.")
                    await asyncio.gather(job.task, return_exceptions=True)
                    job.audit.flush(self.audit)
                    self.audit.record(
                        "extraction_job_finished",
                        {
                            "extraction_job_id": job.job_id,
                            "disposition": "discarded",
                            "reason": job.discard_reason,
                        },
                    )
                    self._extraction = None

            self._closing = asyncio.create_task(finish())
        await asyncio.shield(self._closing)

    async def settle_extraction(self):
        """Diagnostic-only settling: no new job or extension to its deadline."""
        if self._closed or self._decision_task is not None:
            raise ConfigurationError(
                "Cannot settle extraction during a decision or after close."
            )
        self._decision_task = asyncio.current_task()
        self._decision_done = asyncio.Event()
        try:
            job = self._extraction
            if job is not None and not job.task.done():
                remaining = max(0.0, job.deadline - time.monotonic())
                await asyncio.wait({job.task}, timeout=remaining)
                if not job.task.done():
                    job.cancel("Diagnostic reached the original extraction deadline.")
            self._harvest_extraction()
        except BaseException:
            if self._extraction is not None:
                self._extraction.cancel("Diagnostic settling interrupted.")
            raise
        finally:
            self._decision_task = None
            self._decision_done.set()

    async def choose(
        self, raw: dict, messages: list[dict], kind: str, seconds: float
    ) -> dict:
        if self._closed or self._decision_task is not None:
            raise ConfigurationError(
                "Agent is closed or already evaluating a decision."
            )
        self._decision_task = asyncio.current_task()
        self._decision_done = asyncio.Event()
        try:
            return await self._choose(raw, messages, kind, seconds)
        except BaseException:
            if self._extraction is not None:
                self._extraction.cancel(
                    "Decision interrupted before extraction commit."
                )
            raise
        finally:
            self._decision_task = None
            self._decision_done.set()

    async def _choose(
        self, raw: dict, messages: list[dict], kind: str, seconds: float
    ) -> dict:
        started = time.monotonic()
        deadline = started + seconds
        key = state_key(raw)
        checkpoint = None

        def finish(decision, source):
            if (
                checkpoint is not None
                and self.config.strategy == "llm"
                and decision.get("role_scores") is not None
            ):
                self.audit.record(
                    "llm_belief_checkpoint",
                    {
                        **project_llm_scores(
                            decision["role_scores"], checkpoint["distributions"]["hard"]
                        ),
                        **{
                            field: checkpoint[field]
                            for field in (
                                "key",
                                "kind",
                                "alive",
                                "day",
                                "state_version",
                                "subphase",
                                "facts",
                                "ledger_revision",
                            )
                        },
                        "role_scores": deepcopy(decision["role_scores"]),
                        "source": source,
                    },
                )
            # Selection time excludes candidate cleanup and all adapter/transport work.
            answer = {
                "decision": decision,
                "source": source,
                "key": key,
                "kind": kind,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }
            if checkpoint is not None:
                answer["ledger_revision"] = checkpoint["ledger_revision"]
            self.audit.record("selected_decision", answer)
            return answer

        fallback_view = None
        try:
            view = SeatView.from_state(raw)
            facts = view.facts
            identity = (raw.get("session_id"), facts.self_id)
            if (
                not isinstance(identity[0], str)
                or not identity[0]
                or (self._identity is not None and self._identity != identity)
            ):
                raise EvidenceError("Authenticated session/seat identity changed.")
            previous = self.previous_facts
            if previous is not None and (
                previous.self_id != facts.self_id
                or previous.role != facts.role
                or previous.fellow_wolf != facts.fellow_wolf
                or any(
                    facts.investigations.get(seat) != result
                    for seat, result in previous.investigations.items()
                )
                or any(
                    facts.revealed.get(seat) != role
                    for seat, role in previous.revealed.items()
                )
            ):
                raise EvidenceError("Authenticated match facts changed or disappeared.")
            # Message errors must not discard independently trusted role knowledge.
            fallback_view = view
            messages = [
                deepcopy(authorized_message(message, facts.self_id))
                for message in messages
            ]
            visible_new = [
                deepcopy(authorized_message(message, facts.self_id))
                for message in raw.get("new_messages") or []
            ]
            merged = dict(self.messages)
            for message in [*messages, *visible_new]:
                source_id = self.ledger.source_id(message)
                if source_id in merged and merged[source_id] != message:
                    raise EvidenceError("An authenticated message changed.")
                merged[source_id] = message
            self.audit.record(
                "observation",
                {
                    "state": {**raw, "new_messages": visible_new},
                    "messages": messages,
                    "kind": kind,
                },
            )
            self.previous_facts = facts
            self._identity = identity
            self.messages = merged
        except EvidenceError as error:
            if self._extraction is not None:
                self._extraction.cancel("Invalid authenticated observation.")
            self.audit.record("evidence_error", {"error": str(error)})
            return finish(fallback(raw, kind, fallback_view), "fallback")

        reserve = 0.5
        if self.config.extraction_budget_policy == "policy_reserve":
            reserve += self.config.fast_seconds
        foreground_started = time.monotonic()
        foreground_deadline = min(
            foreground_started + self.config.extraction_seconds, deadline - reserve
        )
        if self.config.extraction_budget_policy == "fraction":
            foreground_deadline = min(
                foreground_deadline, foreground_started + seconds * 0.3
            )

        self._harvest_extraction()
        launched = False

        def launch_extraction():
            nonlocal launched
            pending = self._pending()
            if (
                self.config.strategy != "llm"
                and self._extraction is None
                and pending
                and not launched
                and deadline - time.monotonic() > 0.5
            ):
                # A cancelled job retains ownership until drained. Never run an
                # appended tail beside an immutable, unfinished prefix request.
                batch = pending
                # Foreground waiting can end while this immutable request runs.
                request_deadline = deadline - 0.5
                extraction_seconds = max(
                    0.0, min(request_deadline, foreground_deadline) - time.monotonic()
                )
                extraction_input = {
                    "day": view.day,
                    "messages": [
                        {**message, "source_id": self.ledger.source_id(message)}
                        for message in batch
                    ],
                    "previous_claims": deepcopy(self.ledger.events),
                }
                job = self._extraction = ExtractionJob(
                    model=self.model,
                    config=self.config,
                    batch=batch,
                    extraction_input=extraction_input,
                    identity=identity,
                    ledger_revision=self._ledger_revision,
                    deadline=request_deadline,
                    clock=time.monotonic,
                )
                launched = True
                self.audit.record(
                    "extraction_batch",
                    {
                        "pending_count": len(pending),
                        "batch_count": len(batch),
                        "deferred_count": 0,
                        "batch_limit": None,
                        "batch_scope": "all_pending",
                        "budget_seconds": extraction_seconds,
                    },
                )
                self.audit.record(
                    "extraction_job_started",
                    {
                        "extraction_job_id": job.job_id,
                        "source_ids": [
                            self.ledger.source_id(message) for message in batch
                        ],
                        "ledger_revision": self._ledger_revision,
                        "day": view.day,
                        "origin_state_key": state_key(raw),
                        "request_budget_seconds": max(
                            0.0, request_deadline - time.monotonic()
                        ),
                        "foreground_wait_seconds": max(
                            0.0,
                            min(request_deadline, foreground_deadline)
                            - time.monotonic(),
                        ),
                    },
                )

        while self.config.strategy != "llm":
            launch_extraction()
            job = self._extraction
            if job is None:
                break
            foreground_seconds = max(
                0.0, min(job.deadline, foreground_deadline) - time.monotonic()
            )
            if not job.task.done() and foreground_seconds > 0:
                await asyncio.wait({job.task}, timeout=foreground_seconds)
            self._harvest_extraction()
            if self._extraction is not None:
                break

        prompts = None
        response_format = None
        dispatch = None
        admissions = asyncio.Queue()

        async def admit_policy(effort):
            nonlocal checkpoint, prompts, response_format, dispatch
            if prompts is not None:
                return
            # Shared first-candidate admission, not task creation, is the cut.
            # No await occurs between the last completion check and model entry.
            while True:
                self._harvest_extraction()
                if self._extraction is None and not launched and self._pending():
                    launch_extraction()
                    if self._extraction is not None:
                        # A superseded prefix may have drained during admission.
                        # Start its union request before either policy, even with
                        # no foreground allowance left.
                        await asyncio.sleep(0)
                        self._harvest_extraction()
                ledger_snapshot = deepcopy(self.ledger.events)
                ledger_revision = self._ledger_revision
                engine = BeliefEngine(facts)
                shadows = {
                    mode: engine.distribution(mode, ledger_snapshot)
                    for mode in (
                        ("hard", "energy", "csp", "assertion")
                        if self.config.strategy == "assertion"
                        else ("hard", "energy", "csp")
                    )
                }
                snapshot = {
                    "key": key,
                    "alive": view.alive,
                    "day": view.day,
                    "state_version": view.state_version,
                    "subphase": view.subphase,
                    "stalled_days": view.stalled_days,
                    "facts": {
                        "self_id": facts.self_id,
                        "role": facts.role,
                        "fellow_wolf": facts.fellow_wolf,
                        "investigations": facts.investigations,
                        "revealed": facts.revealed,
                    },
                    "events": ledger_snapshot,
                    "ledger_revision": ledger_revision,
                    "distributions": shadows,
                    "kind": kind,
                }
                selected = (
                    None
                    if self.config.strategy == "llm"
                    else shadows[self.config.strategy]
                )
                policy_input = {
                    "kind": kind,
                    "self_id": facts.self_id,
                    "own_role": facts.role,
                    "fellow_wolf": facts.fellow_wolf,
                    "investigations": facts.investigations,
                    "revealed": facts.revealed,
                    "day": view.day,
                    "subphase": view.subphase,
                    "alive": view.alive,
                    "legal_actions": view.action_ids,
                    "players": [
                        {"position": player["position"], "name": player["name"]}
                        for player in raw["players"]
                    ],
                    "public_record": {
                        key: raw["game_state"].get(key)
                        for key in (
                            "day",
                            "phase",
                            "alive",
                            "dead",
                            "vote_history",
                            "winner",
                        )
                    },
                    "messages": list(self.messages.values()),
                    "claim_ledger": ledger_snapshot if selected is not None else [],
                    "belief_input": {
                        key: selected[key]
                        for key in ("mode", "feasible_worlds", "marginals", "warnings")
                    }
                    if selected
                    else None,
                    "seconds_left": max(0, deadline - time.monotonic()),
                }
                prepared = [
                    {"role": "system", "content": POLICY},
                    {
                        "role": "user",
                        "content": json.dumps(policy_input, ensure_ascii=True),
                    },
                ]
                prepared_format = policy_response_format(
                    view, kind, self.config.strategy
                )
                self.audit.record(
                    "policy_pre_dispatch",
                    {"key": key, "kind": kind, "ledger_revision": ledger_revision},
                )
                job = self._extraction
                if job is not None and job.task.done():
                    # Prompt/schema preparation or audit can expose a completed result.
                    # Harvest and rebuild both policies, without launching a retry.
                    continue
                checkpoint, prompts, response_format = (
                    snapshot,
                    prepared,
                    prepared_format,
                )
                dispatch = {
                    "key": key,
                    "kind": kind,
                    "identity": list(identity),
                    "effort": effort,
                    "ledger_revision": ledger_revision,
                    "extraction_pending_count": len(self._pending()),
                    "extraction_job_id": job.job_id if job is not None else None,
                    "extraction_job_state": (
                        "cancelling" if job.discard_reason is not None else "running"
                    )
                    if job is not None
                    else "none",
                    "extraction_source_ids": [
                        self.ledger.source_id(message) for message in job.batch
                    ]
                    if job is not None
                    else [],
                    "extraction_deadline": job.deadline if job is not None else None,
                }
                return

        if self.config.planner_enabled and kind == "action":
            await admit_policy(None)
            if deadline - time.monotonic() <= 0.5:
                self.audit.record("belief_checkpoint", checkpoint)
                return finish(fallback(raw, kind, view), "fallback")
            try:
                recommendation = plan(
                    view, checkpoint["distributions"][self.config.strategy]
                )
                decision = validate_decision(
                    {
                        "action_id": recommendation["action_id"],
                        "content": "",
                        "recipients": [],
                        "reason": "Shallow one-resolution planner; uncalibrated heuristic value.",
                        "role_scores": None,
                    },
                    view,
                    kind,
                    self.config.strategy,
                )
                self.audit.record("belief_checkpoint", checkpoint)
                self.audit.record(
                    "planner_recommendation",
                    {
                        "key": key,
                        "ledger_revision": checkpoint["ledger_revision"],
                        "belief_mode": self.config.strategy,
                        **recommendation,
                    },
                )
                if time.monotonic() >= deadline:
                    return finish(fallback(raw, kind, view), "fallback")
                return finish(decision, "planner")
            except (EvidenceError, ModelError) as error:
                self.audit.record("planner_error", {"error": str(error), "key": key})
                # Invalid/ambiguous planning context keeps the existing policy path.

        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            # Preserve diagnostic checkpoints for deadline-only decisions too.
            await admit_policy(None)
            self.audit.record("belief_checkpoint", checkpoint)
            return finish(fallback(raw, kind, view), "fallback")

        async def candidate(effort: str, limit: float):
            request = asyncio.get_running_loop().create_future()
            admissions.put_nowait((effort, limit, request))
            return await (await request)

        async def policy_request(effort: str, limit: float):
            response = await self.model.complete(
                prompts, effort, limit, response_format=response_format
            )
            decision = validate_decision(
                response["parsed"], view, kind, self.config.strategy
            )
            if time.monotonic() >= deadline:
                raise ModelError(
                    "Policy response exceeded the original decision deadline."
                )
            self.audit.record(
                "validated_candidate", {"effort": effort, "decision": decision}
            )
            return decision

        fast = asyncio.create_task(
            candidate("low", min(self.config.fast_seconds, remaining))
        )
        deep = asyncio.create_task(
            candidate("high", min(self.config.deep_seconds, remaining))
        )
        tasks = {fast, deep}
        requests = []
        fast_result, chosen, source = None, None, "fallback"
        try:
            for index in range(2):
                effort, limit, admission = await admissions.get()
                # Only the decision owner harvests or initializes the shared
                # snapshot. Eager model entry leaves no owner/worker handoff gap.
                await admit_policy(effort)
                remaining = deadline - time.monotonic()
                limit = min(limit, max(0.0, remaining))
                if remaining <= 0.5 or limit <= 0:
                    admission.set_exception(
                        ModelError("Original policy deadline exhausted before request.")
                    )
                else:
                    request = asyncio.create_task(
                        policy_request(effort, limit), eager_start=True
                    )
                    requests.append(request)
                    admission.set_result(request)
                    if len(requests) == 1:
                        self.audit.record("policy_dispatch", dispatch)
                if index == 0:
                    self.audit.record("belief_checkpoint", checkpoint)
            while tasks:
                done, tasks = await asyncio.wait(
                    tasks,
                    timeout=max(0, deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    break
                for task in done:
                    try:
                        result = task.result()
                    except (ModelError, ConfigurationError) as error:
                        self.audit.record(
                            "candidate_error",
                            {
                                "effort": "high" if task is deep else "low",
                                "error": str(error),
                            },
                        )
                        continue
                    if task is deep:
                        chosen, source = result, "high"
                    else:
                        fast_result = result
                if chosen is not None or (deep.done() and fast_result is not None):
                    break
            if chosen is None and fast_result is not None:
                chosen, source = fast_result, "low"
            if chosen is None:
                chosen = fallback(raw, kind, view)
            return finish(chosen, source)
        finally:
            for task in (fast, deep, *requests):
                if not task.done():
                    task.cancel()
            await asyncio.gather(fast, deep, *requests, return_exceptions=True)
