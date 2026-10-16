"""Pure one-resolution planning over a caller-supplied joint role distribution."""

import math
from collections import defaultdict

from .belief import EvidenceError, Facts, SeatView

MODEL = "shallow-uniform-v1"
INFORMATION_WEIGHT = 0.15
NORMALIZATION_TOLERANCE = 1e-9


def _seat(value: object) -> bool:
    return type(value) is int and 0 <= value < 7


def _inputs(view: SeatView, belief: dict) -> list[tuple[tuple[int, int], int, float]]:
    if not isinstance(view, SeatView) or not isinstance(view.facts, Facts):
        raise EvidenceError("Planner requires an authenticated SeatView and Facts.")
    view.facts._validate()
    if (
        not isinstance(view.alive, tuple)
        or not all(_seat(seat) for seat in view.alive)
        or len(set(view.alive)) != len(view.alive)
        or set(view.alive) != set(range(7)) - set(view.facts.revealed)
        or view.facts.self_id not in view.alive
        or view.phase != "moving"
        or view.subphase not in ("day_vote", "night_wolf", "night_seer")
        or type(view.day) is not int
        or view.day < 1
        or type(view.stalled_days) is not int
        or not 0 <= view.stalled_days < 3
    ):
        raise EvidenceError("Planner requires a living moving seat and valid history.")
    if (view.subphase == "night_wolf" and view.facts.role != "wolf") or (
        view.subphase == "night_seer" and view.facts.role != "seer"
    ):
        raise EvidenceError("Night phase does not belong to the acting role.")
    if (
        not isinstance(view.action_ids, tuple)
        or not 1 <= len(view.action_ids) <= 7
        or any(type(action) is not str for action in view.action_ids)
        or len(set(view.action_ids)) != len(view.action_ids)
    ):
        raise EvidenceError("Planner requires a nonempty unique legal action tuple.")
    legal = {str(seat) for seat in view.alive if seat != view.facts.self_id}
    if view.subphase == "day_vote":
        legal.add("7")
    elif view.subphase == "night_wolf":
        legal.discard(str(view.facts.fellow_wolf))
    if not set(view.action_ids) <= legal:
        raise EvidenceError("Supplied actions contradict authenticated mechanics.")

    entries = belief.get("worlds") if isinstance(belief, dict) else None
    if not isinstance(entries, list) or not 1 <= len(entries) <= 105:
        raise EvidenceError("Planner requires 1..105 supplied joint role worlds.")
    worlds, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise EvidenceError("Malformed joint role world.")
        wolves, seer, probability = (
            entry.get("wolves"),
            entry.get("seer"),
            entry.get("probability"),
        )
        if (
            not isinstance(wolves, (list, tuple))
            or len(wolves) != 2
            or not all(_seat(seat) for seat in wolves)
            or wolves[0] == wolves[1]
            or not _seat(seer)
            or seer in wolves
            or type(probability) not in (int, float)
            or not 0 <= probability <= 1
            or not math.isfinite(probability)
        ):
            raise EvidenceError("Invalid joint role assignment or finite weight.")
        pair = tuple(sorted(wolves))
        world = pair, seer
        if world in seen or not view.facts._allows(world):
            raise EvidenceError("Duplicate world or world contradicting facts.")
        seen.add(world)
        worlds.append((pair, seer, float(probability)))
    total = math.fsum(world[2] for world in worlds)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=NORMALIZATION_TOLERANCE):
        raise EvidenceError("Joint world weights must sum to one.")
    # Only floating-point normalization residue is repaired, never malformed mass.
    worlds = sorted((wolves, seer, p / total) for wolves, seer, p in worlds if p > 0)
    if any(
        _winner(view.alive, wolves, view.stalled_days) is not None
        for wolves, _, _ in worlds
    ):
        raise EvidenceError("Supplied action state is already terminal.")
    return worlds


def _winner(
    alive: tuple[int, ...], wolves: tuple[int, int], stalled: int
) -> str | None:
    count = sum(seat in wolves for seat in alive)
    if count == 0:
        return "villager"
    if 2 * count >= len(alive) or stalled >= 3:
        return "wolf"
    return None


def _value(
    alive: tuple[int, ...], wolves: tuple[int, int], seer: int, stalled: int
) -> tuple[float, str | None]:
    """Village-oriented team utility, not a calibrated continuation probability."""
    winner = _winner(alive, wolves, stalled)
    if winner is not None:
        return (1.0 if winner == "villager" else -1.0), winner
    count = sum(seat in wolves for seat in alive)
    return (
        0.5 * (len(alive) - 2 * count) / len(alive)
        - 0.25 * count
        + 0.1 * (seer in alive)
        - 0.05 * stalled,
        None,
    )


def _plurality(counts: tuple[int, ...]) -> int | None:
    """Seven target bins followed by an ignored abstention bin."""
    peak = max(counts[:7])
    leaders = [seat for seat in range(7) if counts[seat] == peak]
    return leaders[0] if peak > 0 and len(leaders) == 1 else None


def _day_outcomes(view: SeatView) -> tuple[dict[str, dict], dict]:
    # Integer path counts integrate hidden ballots once, independent of role worlds.
    counts = {(0,) * 8: 1}
    transitions, peak_states = 0, 1
    denominator = 1
    for voter in sorted(set(view.alive) - {view.facts.self_id}):
        choices = [seat for seat in sorted(view.alive) if seat != voter] + [7]
        following = defaultdict(int)
        for tally, multiplicity in counts.items():
            for target in choices:
                changed = list(tally)
                changed[target] += 1
                following[tuple(changed)] += multiplicity
                transitions += 1
        counts = following
        denominator *= len(choices)
        peak_states = max(peak_states, len(counts))
    outcomes = {}
    for action in sorted(view.action_ids, key=int):
        eliminated = defaultdict(int)
        for tally, multiplicity in counts.items():
            changed = list(tally)
            changed[int(action)] += 1
            eliminated[_plurality(tuple(changed))] += multiplicity
        outcomes[action] = {
            target: mass / denominator for target, mass in eliminated.items()
        }
    return outcomes, {
        "ballot_distributions_built": 1,
        "ballot_peak_states": peak_states,
        "ballot_transitions": transitions,
        "ballot_tallies_resolved": len(counts) * len(view.action_ids),
        "hidden_ballot_profiles": denominator,
    }


def _night_outcomes(
    alive: tuple[int, ...], wolves: tuple[int, int], day: int, target: int | None
) -> dict[int | None, float]:
    """Uniform independent wolf targets; known own choice replaces one draw."""
    if day == 1:
        return {None: 1.0}
    living_wolves = sum(seat in wolves for seat in alive)
    candidates = sorted(seat for seat in alive if seat not in wolves)
    if target is not None:
        if living_wolves == 1:
            return {target: 1.0}
        outcomes = defaultdict(float)
        for ally_target in candidates:
            outcomes[min(target, ally_target)] += 1 / len(candidates)
        return dict(outcomes)
    # P(min of k uniform draws = j), integrating all hidden wolf choices exactly.
    size = len(candidates)
    return {
        seat: ((size - index) ** living_wolves - (size - index - 1) ** living_wolves)
        / size**living_wolves
        for index, seat in enumerate(candidates)
    }


def plan(view: SeatView, belief: dict) -> dict:
    """Recommend one supplied action after averaging worlds, without I/O or mutation.

    Only ``belief['worlds']`` is consumed. Each entry has two ``wolves`` seats,
    one ``seer`` seat, and a finite normalized ``probability``. Other scorer
    metadata and marginals are deliberately ignored. Invalid inputs raise
    EvidenceError; the caller owns admission, fallback, and action submission.
    """
    worlds = _inputs(view, belief)
    actions = sorted(view.action_ids, key=int)
    alive = tuple(sorted(view.alive))
    day_outcomes, work = (
        _day_outcomes(view)
        if view.subphase == "day_vote"
        else (
            {},
            {
                "ballot_distributions_built": 0,
                "ballot_peak_states": 0,
                "ballot_transitions": 0,
                "ballot_tallies_resolved": 0,
                "hidden_ballot_profiles": 0,
            },
        )
    )
    work.update(
        world_count=len(worlds), action_count=len(actions), outcome_evaluations=0
    )
    values, diagnostics = {}, {}
    for action in actions:
        contributions = []
        terminal = {"wolf": 0.0, "villager": 0.0}
        deaths = defaultdict(float)
        # The public dawn observation is (victim, revealed role). Information is
        # conditional on that observation, not credited again for a public reveal.
        information_groups = defaultdict(lambda: [0.0, 0.0])
        surviving_seer = 0.0
        for wolves, seer, probability in worlds:
            outcomes = (
                day_outcomes[action]
                if view.subphase == "day_vote"
                else _night_outcomes(
                    alive,
                    wolves,
                    view.day,
                    int(action) if view.subphase == "night_wolf" else None,
                )
            )
            for victim, response_mass in outcomes.items():
                work["outcome_evaluations"] += 1
                mass = probability * response_mass
                remaining = tuple(seat for seat in alive if seat != victim)
                stalled = view.stalled_days
                if view.subphase == "day_vote":
                    stalled = stalled + 1 if victim is None else 0
                value, winner = _value(remaining, wolves, seer, stalled)
                contributions.append(mass * value)
                deaths["none" if victim is None else str(victim)] += mass
                if winner is not None:
                    terminal[winner] += mass
                if seer in remaining:
                    surviving_seer += mass
                if (
                    view.subphase == "night_seer"
                    and seer in remaining
                    and winner is None
                ):
                    revealed_role = (
                        None
                        if victim is None
                        else "seer"
                        if victim == seer
                        else "wolf"
                        if victim in wolves
                        else "villager"
                    )
                    information_groups[(victim, revealed_role)][
                        int(int(action) in wolves)
                    ] += mass
        information_bits = 0.0
        for bins in information_groups.values():
            group_mass = math.fsum(bins)
            information_bits += math.fsum(
                -mass * math.log2(mass / group_mass) for mass in bins if mass > 0
            )
        sign = -1 if view.facts.role == "wolf" else 1
        cutoff_value = sign * math.fsum(contributions)
        values[action] = cutoff_value + INFORMATION_WEIGHT * information_bits
        diagnostics[action] = {
            "cutoff_value": cutoff_value,
            "usable_information_bits": information_bits,
            "information_utility": INFORMATION_WEIGHT * information_bits,
            "terminal_mass": terminal,
            "death_distribution": dict(sorted(deaths.items())),
            "seer_survival_mass": surviving_seer,
        }
    return {
        "action_id": max(actions, key=lambda action: values[action]),
        "action_values": values,
        "assumptions": [
            "One resolution: current day tally or current night through dawn only.",
            "All other day ballots independently uniform over living nonself seats "
            "and abstention; role-blind, no private facts or belief shared with them.",
            "Wolf night choices independently uniform over living nonwolves; "
            "wolves know only their team, not the seer or private checks.",
            "Hidden votes/kill choices are integrated, not observed or coordinated.",
            "No future resignations, timeouts, discussion, or subsequent actions.",
            "Terminal team utility is +/-1; cutoff and information utility are "
            "uncalibrated heuristics, not win probabilities.",
            "Seer check precedes dawn death; information utility requires a living "
            "seer and nonterminal dawn, conditional on the public death reveal.",
            "Maximum is taken after world aggregation; numeric action ID breaks ties.",
        ],
        "diagnostics": {
            "model": MODEL,
            "horizon_resolutions": 1,
            "response_integration": "exact_count_distribution",
            "sampling": False,
            "normalization_tolerance": NORMALIZATION_TOLERANCE,
            "cutoff": {
                "village_margin_weight": 0.5,
                "living_wolf_penalty": 0.25,
                "living_seer_bonus": 0.1,
                "stalled_day_penalty": 0.05,
                "usable_information_bit_weight": INFORMATION_WEIGHT,
            },
            "work": work,
            "actions": diagnostics,
        },
    }
