"""Exact seven-seat role worlds; dialogue never changes authenticated facts."""

import math
import re
from dataclasses import dataclass, field
from itertools import combinations

from .assertions import Assertion, score_world

ROLES = ("wolf", "seer", "villager")
MIN_CLAIM_CONFIDENCE = 0.9  # Parsing-confidence gate, not a credibility estimate.
_WORLDS = tuple(
    (wolves, seer)
    for wolves in combinations(range(7), 2)
    for seer in range(7)
    if seer not in wolves
)


class EvidenceError(ValueError):
    """Malformed evidence or mutually inconsistent authenticated facts."""


def _is_seat(value: object) -> bool:
    return type(value) is int and 0 <= value < 7


def _role(world: tuple[tuple[int, int], int], seat: int) -> str:
    wolves, seer = world
    return "wolf" if seat in wolves else "seer" if seat == seer else "villager"


@dataclass
class Facts:
    self_id: int
    role: str
    fellow_wolf: int | None = None
    investigations: dict[int, str] = field(default_factory=dict)
    revealed: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if (
            not _is_seat(self.self_id)
            or not isinstance(self.role, str)
            or self.role not in ROLES
        ):
            raise EvidenceError("Invalid authenticated self seat or role.")
        if self.role == "wolf":
            if not _is_seat(self.fellow_wolf) or self.fellow_wolf == self.self_id:
                raise EvidenceError("A wolf requires exactly one distinct fellow wolf.")
        elif self.fellow_wolf is not None:
            raise EvidenceError("Only a wolf can have authenticated wolf partners.")
        if not isinstance(self.investigations, dict) or not isinstance(
            self.revealed, dict
        ):
            raise EvidenceError("Authenticated role evidence must be dictionaries.")
        if self.investigations and self.role != "seer":
            raise EvidenceError("Only the seer receives private investigations.")
        for seat, result in self.investigations.items():
            if (
                not _is_seat(seat)
                or seat == self.self_id
                or result not in ("wolf", "villager")
            ):
                raise EvidenceError("Invalid authenticated investigation.")
        for seat, role in self.revealed.items():
            if not _is_seat(seat) or role not in ROLES:
                raise EvidenceError("Invalid authenticated death-role reveal.")
        if not any(self._allows(world) for world in _WORLDS):
            raise EvidenceError("Authenticated facts admit no legal role world.")

    def _allows(self, world: tuple[tuple[int, int], int]) -> bool:
        wolves, _ = world
        return (
            _role(world, self.self_id) == self.role
            and (self.fellow_wolf is None or self.fellow_wolf in wolves)
            and all(
                (seat in wolves) == (result == "wolf")
                for seat, result in self.investigations.items()
            )
            and all(_role(world, seat) == role for seat, role in self.revealed.items())
        )


@dataclass
class SeatView:
    facts: Facts
    alive: tuple[int, ...]
    day: int
    subphase: str
    phase: str
    state_version: int
    action_ids: tuple[str, ...]
    stalled_days: int | None = 0

    @classmethod
    def from_state(cls, raw: dict) -> SeatView:
        """Parse an MCP Werewolf state, not a REST state or a message transcript.

        Null terminal day/subphase use the observation day and ``terminal``.
        Pre-start states without an observation must be handled by the caller.
        """
        if not isinstance(raw, dict) or not _is_seat(raw.get("your_position")):
            raise EvidenceError("State requires an exact self position from 0 to 6.")
        self_id = raw["your_position"]
        players = raw.get("players")
        if not isinstance(players, list) or len(players) != 7:
            raise EvidenceError("State requires the complete seven-seat player map.")
        names: dict[str, int] = {}
        positions: set[int] = set()
        for player in players:
            if (
                not isinstance(player, dict)
                or not _is_seat(player.get("position"))
                or not isinstance(player.get("name"), str)
                or not player["name"]
                or not isinstance(player.get("agent_id"), str)
                or not player["agent_id"]
            ):
                raise EvidenceError("Malformed authenticated player-map entry.")
            if player["position"] in positions or player["name"] in names:
                raise EvidenceError("Player-map seats and full names must be unique.")
            positions.add(player["position"])
            names[player["name"]] = player["position"]

        def seat_for(name: object) -> int:
            if not isinstance(name, str) or name not in names:
                raise EvidenceError(
                    "Public record references an unmapped display name."
                )
            return names[name]

        observation = raw.get("observation")
        if not isinstance(observation, str):
            raise EvidenceError("Missing authenticated private observation.")
        lines = observation.splitlines()
        if len(lines) < 3 or lines[0] != "=== Werewolf ===":
            raise EvidenceError("Private observation requires the Werewolf header.")
        role_line = re.fullmatch(r"Your role: (WOLF|SEER|VILLAGER)", lines[1])
        boundary = next(
            (
                index
                for index in range(2, len(lines))
                if lines[index].startswith("Day ")
            ),
            None,
        )
        if role_line is None or boundary is None:
            raise EvidenceError("Missing anchored private role or public Day boundary.")
        day_line = re.fullmatch(
            r"Day ([1-9][0-9]*) \| phase ([^|]+) \| alive ([0-7])/7", lines[boundary]
        )
        if day_line is None:
            raise EvidenceError("Malformed observation Day boundary.")
        try:
            observed_day = int(day_line[1])
        except ValueError:
            raise EvidenceError("Malformed observation day number.") from None
        role = role_line[1].lower()
        fellow_wolf = None
        investigations: dict[int, str] = {}
        knowledge_seen = False
        for line in lines[2:boundary]:
            if line in (
                "",
                "You are DEAD. You may watch, but not act.",
                "You RESIGNED and are out of the game. You may watch, but not act.",
            ):
                continue
            if knowledge_seen:
                raise EvidenceError("Repeated authenticated private-knowledge section.")
            if role == "wolf":
                match = re.fullmatch(r"Your fellow wolves: Player([0-6])", line)
                if match is None:
                    raise EvidenceError("Malformed private wolf-partner declaration.")
                fellow_wolf = int(match[1])
            elif role == "seer":
                if line != "You have not investigated anyone yet.":
                    prefix = "Your investigations: "
                    if not line.startswith(prefix):
                        raise EvidenceError(
                            "Malformed private investigation declaration."
                        )
                    for check in line[len(prefix) :].split(", "):
                        match = re.fullmatch(r"Player([0-6]) is (wolf|villager)", check)
                        if match is None or (
                            int(match[1]) in investigations
                            and investigations[int(match[1])] != match[2]
                        ):
                            raise EvidenceError(
                                "Malformed or contradictory private investigation."
                            )
                        investigations[int(match[1])] = match[2]
            elif line != "You do not know anyone else's role.":
                raise EvidenceError("Unexpected private information for a villager.")
            knowledge_seen = True
        if role in ("wolf", "seer") and not knowledge_seen:
            raise EvidenceError(
                "Missing authenticated role-specific private knowledge."
            )

        public = raw.get("game_state")
        if not isinstance(public, dict) or not {
            "day",
            "phase",
            "alive",
            "dead",
            "vote_history",
        }.issubset(public):
            raise EvidenceError("Missing structured Werewolf public record.")
        terminal = raw.get("is_terminal", False)
        if type(terminal) is not bool:
            raise EvidenceError("Terminal flag must be a boolean.")
        day, subphase = public.get("day"), public.get("phase")
        if day is None and subphase is None and terminal:
            day, subphase = observed_day, "terminal"
        elif (
            type(day) is not int
            or day < 1
            or subphase not in ("night_wolf", "night_seer", "day_vote")
            or day != observed_day
            or subphase != day_line[2]
        ):
            raise EvidenceError("Structured day/phase contradicts the observation.")
        if "phase" not in raw:
            raise EvidenceError("Missing platform phase.")
        phase = raw["phase"]
        if terminal and phase is None:
            phase = "terminal"
        if phase not in ("messaging", "moving", "terminal") or (
            phase == "terminal" and not terminal
        ):
            raise EvidenceError("Invalid platform phase.")
        if phase == "messaging" and subphase != "day_vote":
            raise EvidenceError("Werewolf discussion must occur during day_vote.")
        version = raw.get("state_version")
        if type(version) is not int or version < 0:
            raise EvidenceError("State version must be a nonnegative integer.")
        alive_names, deaths = public.get("alive"), public.get("dead")
        if not isinstance(alive_names, list) or not isinstance(deaths, list):
            raise EvidenceError("Alive and dead records must be lists.")
        alive = tuple(seat_for(name) for name in alive_names)
        if len(set(alive)) != len(alive) or len(alive) != int(day_line[3]):
            raise EvidenceError("Duplicate living seat or contradictory alive count.")
        revealed: dict[int, str] = {}
        death_by_seat: dict[int, dict] = {}
        for death in deaths:
            if (
                not isinstance(death, dict)
                or type(death.get("day")) is not int
                or not 1 <= death["day"] <= day
                or death.get("role") not in ROLES
                or death.get("cause") not in ("night_kill", "vote", "resigned")
            ):
                raise EvidenceError("Malformed authenticated death record.")
            seat = seat_for(death.get("player"))
            if seat in revealed or seat in alive:
                raise EvidenceError("Repeated death or living/dead contradiction.")
            if death["cause"] == "night_kill" and death["role"] == "wolf":
                raise EvidenceError("A night kill cannot reveal a wolf victim.")
            revealed[seat] = death["role"]
            death_by_seat[seat] = death
        if set(alive) | set(revealed) != positions:
            raise EvidenceError("Alive and dead records must cover all original seats.")

        history = public.get("vote_history")
        if not isinstance(history, list):
            raise EvidenceError("Vote history must be a structured list.")
        vote_days: set[int] = set()
        for vote in history:
            if (
                not isinstance(vote, dict)
                or type(vote.get("day")) is not int
                or not 1 <= vote["day"] <= day
                or vote["day"] in vote_days
                or not isinstance(vote.get("votes"), dict)
                or "eliminated" not in vote
            ):
                raise EvidenceError("Malformed authenticated vote-history entry.")
            vote_days.add(vote["day"])
            for voter, target in vote["votes"].items():
                voter_seat = seat_for(voter)
                if target != "abstain" and seat_for(target) == voter_seat:
                    raise EvidenceError("Public vote history contains a self vote.")
            if vote["eliminated"] is not None:
                eliminated = seat_for(vote["eliminated"])
                death = death_by_seat.get(eliminated)
                if (
                    death is None
                    or death["cause"] != "vote"
                    or death["day"] != vote["day"]
                ):
                    raise EvidenceError("Vote elimination contradicts death records.")

        # Public resignations omit their phase: a day resignation resets the
        # stall streak, a night resignation does not. Never guess for planning.
        resigned_days = {
            death["day"] for death in deaths if death["cause"] == "resigned"
        }
        streaks = {0}
        ordered_votes = sorted(history, key=lambda vote: vote["day"])
        for vote in ordered_votes:
            if vote["eliminated"] is not None:
                streaks = {0}
            else:
                streaks = {streak + 1 for streak in streaks}
                if vote["day"] in resigned_days:
                    streaks.add(0)
        complete_history = vote_days == set(range(1, day))
        stalled_days = (
            next(iter(streaks))
            if complete_history
            and len(streaks) == 1
            and not (subphase == "day_vote" and day in resigned_days)
            else None
        )
        facts = Facts(self_id, role, fellow_wolf, investigations, revealed)
        actions = raw.get("legal_actions")
        action_ids: list[str] = []
        if actions is not None:
            if not isinstance(actions, dict) or not isinstance(
                actions.get("actions"), list
            ):
                raise EvidenceError(
                    "Legal actions require the MCP action-list envelope."
                )
            if "state_version" in actions and (
                type(actions["state_version"]) is not int
                or actions["state_version"] != version
            ):
                raise EvidenceError(
                    "Legal-action envelope has a different state version."
                )
            for action in actions["actions"]:
                if not isinstance(action, dict) or action.get("action_id") not in tuple(
                    str(seat) for seat in range(8)
                ):
                    raise EvidenceError("Malformed Werewolf action ID.")
                action_id = action["action_id"]
                target = int(action_id)
                if (
                    action_id in action_ids
                    or terminal
                    or self_id not in alive
                    or (target == 7 and subphase != "day_vote")
                    or (target != 7 and (target not in alive or target == self_id))
                    or (
                        subphase == "night_wolf"
                        and (role != "wolf" or target == fellow_wolf)
                    )
                    or (subphase == "night_seer" and role != "seer")
                ):
                    raise EvidenceError(
                        "Legal action contradicts authenticated mechanics."
                    )
                if "input" in action:
                    action_input = action["input"]
                    if (
                        not isinstance(action_input, dict)
                        or action_input.get("action_id") != action_id
                        or type(action_input.get("state_version")) is not int
                        or action_input["state_version"] != version
                    ):
                        raise EvidenceError(
                            "Action input contradicts its ID or state version."
                        )
                action_ids.append(action_id)
        return cls(
            facts,
            tuple(sorted(alive)),
            day,
            subphase,
            phase,
            version,
            tuple(action_ids),
            stalled_days,
        )


class BeliefEngine:
    """Recompute from 105 original worlds, never from an earlier posterior.

    Own-role conditioning alone leaves villager=60, seer=15, wolf=5 worlds.
    Energy rewards consistent self-role claims by 0.20 log units and checks by
    0.25, averaging each claim kind per speaker (at most 0.45 per speaker).
    Checks are satisfied only by the world's seer reporting a correct result.
    CSP uses a per-speaker geometric mean of factors 1 or 2 for the experimental
    conditional ``speaker is wolf OR claim holds``; each speaker's factor <= 2.
    Assertion separates content truth from speaker legitimacy, marginalizing a
    bounded faithful/uninformative account model (see plan/assertion-scorer.md).
    None of these models assumes accusations are truth or provides calibration.
    """

    def __init__(self, facts: Facts) -> None:
        if not isinstance(facts, Facts):
            raise EvidenceError("BeliefEngine requires authenticated Facts.")
        facts._validate()
        self.facts = facts

    def distribution(self, mode: str, events: list[dict] | None = None) -> dict:
        """Return normalized world weights and Shannon entropy in bits."""
        if mode not in ("hard", "energy", "csp", "assertion"):
            raise EvidenceError(
                "Unknown belief mode; use hard, energy, csp, or assertion."
            )
        if events is not None and not isinstance(events, list):
            raise EvidenceError("Soft evidence must be an event list.")
        self.facts._validate()
        worlds = [world for world in _WORLDS if self.facts._allows(world)]
        warnings: list[str] = []
        claims: dict[int, list[dict]] = {}
        seen: set[tuple] = set()
        malformed = False
        if mode != "hard":
            warnings.append(
                "Experimental model distribution, not a calibrated posterior."
            )
            if mode == "csp":
                warnings.append(
                    "Experimental truthfulness conditional: wolves automatically "
                    "satisfy claims; non-wolves are weakly rewarded for consistent "
                    "role/check claims. This assumption can favor bluffers; each speaker "
                    "factor is <= 2."
                )
            if mode == "assertion":
                warnings.append(
                    "Experimental assertion hypotheses: content truth is distinct "
                    "from seer legitimacy; role-conditioned credibility is assumed, "
                    "not learned or derived from parsing confidence/informativeness. "
                    "Each speaker contributes at most +/-0.60 log units."
                )
            # Occurrences retain their own parsing confidence and observation day.
            # The canonical event controls retraction; validation precedes dedup.
            occurrences = []
            for assertion in events or []:
                occurrences.append(assertion)
                if (
                    not isinstance(assertion, dict)
                    or assertion.get("kind") not in ("role_claim", "check_claim")
                    or assertion.get("active") is not True
                    or "retracted_by" in assertion
                ):
                    continue
                repetitions = assertion.get("repetitions", [])
                if not isinstance(repetitions, list):
                    malformed = True
                    continue
                for repeat in repetitions:
                    if (
                        not isinstance(repeat, dict)
                        or any(
                            repeat.get(field) != assertion.get(field)
                            for field in (
                                "speaker",
                                "kind",
                                "target",
                                "result",
                                "check_day",
                                "uncertain",
                                "negated",
                                "quoted",
                            )
                        )
                        or (
                            assertion["kind"] == "role_claim"
                            and repeat.get("role") != assertion.get("role")
                        )
                    ):
                        malformed = True
                        continue
                    occurrences.append(repeat)
            for event in occurrences:
                if not isinstance(event, dict):
                    malformed = True
                    continue
                if event.get("kind") not in ("role_claim", "check_claim"):
                    continue
                required = {
                    "speaker",
                    "target",
                    "role",
                    "result",
                    "check_day",
                    "confidence",
                    "source_id",
                    "quote",
                    "uncertain",
                    "negated",
                    "quoted",
                    "active",
                }
                if (
                    not required.issubset(event)
                    or not _is_seat(event.get("speaker"))
                    or any(
                        type(event.get(flag)) is not bool
                        for flag in ("uncertain", "negated", "quoted", "active")
                    )
                    or type(event.get("confidence")) not in (int, float)
                    or not 0 <= event["confidence"] <= 1
                    or not math.isfinite(event["confidence"])
                    or not isinstance(event.get("source_id"), str)
                    or not event["source_id"]
                    or not isinstance(event.get("quote"), str)
                    or not event["quote"]
                    or (event["target"] is not None and not _is_seat(event["target"]))
                ):
                    malformed = True
                    continue
                if (
                    event["speaker"] == self.facts.self_id
                    or not event["active"]
                    or "retracted_by" in event
                    or event["uncertain"]
                    or event["negated"]
                    or event["quoted"]
                    or event["confidence"] < MIN_CLAIM_CONFIDENCE
                ):
                    continue
                speaker, target = event["speaker"], event["target"]
                if event["kind"] == "role_claim":
                    valid = (
                        target in (None, speaker)
                        and event["role"] in ROLES
                        and event["result"] is None
                        and event["check_day"] is None
                    )
                    target = speaker
                else:
                    check_day, observed_day = event["check_day"], event.get("day")
                    valid = (
                        _is_seat(target)
                        and target != speaker
                        and event["role"] in (None, "seer")
                        and event["result"] in ("wolf", "villager")
                        and (
                            check_day is None
                            or (type(check_day) is int and check_day >= 1)
                        )
                        and (
                            observed_day is None
                            or (
                                type(observed_day) is int
                                and observed_day >= 1
                                and (check_day is None or check_day <= observed_day)
                            )
                        )
                    )
                if not valid:
                    malformed = True
                    continue
                if "source_text" in event and (
                    not isinstance(event["source_text"], str)
                    or event["quote"] not in event["source_text"]
                ):
                    malformed = True
                    continue
                key = (
                    speaker,
                    event["kind"],
                    target,
                    event["role"] if event["kind"] == "role_claim" else None,
                    event["result"],
                    event["check_day"],
                )
                if key not in seen:
                    seen.add(key)
                    claims.setdefault(speaker, []).append(event)
            if malformed:
                warnings.append(
                    "Ignored malformed, unsupported, or implausibly timed soft claims."
                )

        assertions = (
            {
                speaker: [Assertion.from_event(statement) for statement in statements]
                for speaker, statements in claims.items()
            }
            if mode == "assertion"
            else {}
        )
        scores: list[float] = []
        for world in worlds:
            if mode == "assertion":
                scores.append(score_world(world, assertions))
                continue
            wolves, seer = world
            score = 0.0
            for speaker, statements in claims.items():
                role_truths, check_truths = [], []
                for statement in statements:
                    if statement["kind"] == "role_claim":
                        role_truths.append(_role(world, speaker) == statement["role"])
                    else:
                        check_truths.append(
                            speaker == seer
                            and (statement["target"] in wolves)
                            == (statement["result"] == "wolf")
                        )
                if mode == "energy":
                    for weight, truths in ((0.20, role_truths), (0.25, check_truths)):
                        if truths:
                            score += weight * math.fsum(truths) / len(truths)
                elif mode == "csp":
                    truths = role_truths + check_truths
                    consistency = (
                        1.0 if speaker in wolves else math.fsum(truths) / len(truths)
                    )
                    score += math.log(2.0) * consistency
            scores.append(score)

        # Max-shifted log-sum-exp normalization keeps every feasible world positive.
        peak = max(scores)
        weights = [math.exp(score - peak) for score in scores]
        normalizer = math.fsum(weights)
        probabilities = [weight / normalizer for weight in weights]
        marginals: dict[str, dict[str, float]] = {}
        for seat in range(7):
            assignments = [_role(world, seat) for world in worlds]
            marginals[str(seat)] = {
                role: (
                    1.0
                    if all(assignment == role for assignment in assignments)
                    else math.fsum(
                        probability
                        for assignment, probability in zip(assignments, probabilities)
                        if assignment == role
                    )
                )
                for role in ROLES
            }
        return {
            "mode": mode,
            "feasible_worlds": len(worlds),
            "worlds": [
                {"wolves": list(world[0]), "seer": world[1], "probability": probability}
                for world, probability in zip(worlds, probabilities)
            ],
            "marginals": marginals,
            "entropy": max(0.0, -math.fsum(p * math.log2(p) for p in probabilities)),
            "warnings": warnings,
        }
