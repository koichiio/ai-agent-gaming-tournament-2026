import ast
import inspect
import json
import math
from collections import Counter
from copy import deepcopy
from itertools import combinations, product

import pytest

from agent_koichi import planner
from agent_koichi.belief import EvidenceError, Facts, SeatView
from agent_koichi.planner import (
    INFORMATION_WEIGHT,
    _day_outcomes,
    _night_outcomes,
    _plurality,
    _value,
    plan,
)


def role_at(seat, wolves=(1, 2), seer=3):
    return "wolf" if seat in wolves else "seer" if seat == seer else "villager"


def view_for(
    *,
    self_id=0,
    wolves=(1, 2),
    seer=3,
    alive=tuple(range(7)),
    day=2,
    subphase="day_vote",
    stalled=0,
    actions=None,
    investigations=None,
):
    role = role_at(self_id, wolves, seer)
    facts = Facts(
        self_id,
        role,
        next(seat for seat in wolves if seat != self_id) if role == "wolf" else None,
        investigations or {},
        {seat: role_at(seat, wolves, seer) for seat in range(7) if seat not in alive},
    )
    if actions is None:
        actions = tuple(
            str(seat)
            for seat in sorted(alive)
            if seat != self_id and (subphase != "night_wolf" or seat not in wolves)
        ) + (("7",) if subphase == "day_vote" else ())
    return SeatView(facts, alive, day, subphase, "moving", 10, actions, stalled)


def joint(*entries):
    return {
        "worlds": [
            {"wolves": list(wolves), "seer": seer, "probability": probability}
            for wolves, seer, probability in entries
        ]
    }


def uniform(view):
    worlds = [
        (wolves, seer)
        for wolves in combinations(range(7), 2)
        for seer in range(7)
        if seer not in wolves and view.facts._allows((wolves, seer))
    ]
    return joint(*((wolves, seer, 1 / len(worlds)) for wolves, seer in worlds))


def action_diagnostics(result, action):
    return result["diagnostics"]["actions"][action]


@pytest.mark.parametrize(
    "self_id,subphase", [(0, "day_vote"), (1, "night_wolf"), (3, "night_seer")]
)
def test_every_phase_scores_only_supplied_legal_actions(self_id, subphase):
    view = view_for(self_id=self_id, subphase=subphase)
    view.action_ids = view.action_ids[-2:]
    result = plan(view, uniform(view))
    assert result["action_id"] in view.action_ids
    assert set(result["action_values"]) == set(view.action_ids)
    assert all(math.isfinite(value) for value in result["action_values"].values())
    assert result["assumptions"]
    assert result["diagnostics"]["horizon_resolutions"] == 1
    assert result["diagnostics"]["sampling"] is False
    for diagnostic in result["diagnostics"]["actions"].values():
        assert sum(diagnostic["death_distribution"].values()) == pytest.approx(1)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "counts,expected",
    [
        ((0, 0, 0, 0, 0, 0, 0, 7), None),
        ((2, 2, 0, 0, 0, 0, 0, 3), None),
        ((2, 1, 1, 0, 0, 0, 0, 3), 0),
        ((0, 0, 0, 0, 0, 0, 1, 6), 6),
    ],
)
def test_unique_plurality_ties_and_abstention(counts, expected):
    assert _plurality(counts) == expected


@pytest.mark.parametrize("size", [3, 4, 5])
def test_hidden_ballot_dp_matches_small_exhaustive_reference(size):
    view = view_for(wolves=(1, 6), seer=2, alive=tuple(range(size)))
    actual, work = _day_outcomes(view)
    voters = [seat for seat in view.alive if seat != view.facts.self_id]
    profiles = list(
        product(
            *[[seat for seat in view.alive if seat != voter] + [7] for voter in voters]
        )
    )
    assert work["hidden_ballot_profiles"] == size ** (size - 1)
    assert work["ballot_distributions_built"] == 1
    for action in view.action_ids:
        expected = Counter()
        for profile in profiles:
            votes = Counter((*profile, int(action)))
            votes.pop(7, None)
            peak = max(votes.values(), default=0)
            leaders = [seat for seat, count in votes.items() if count == peak]
            expected[leaders[0] if peak > 0 and len(leaders) == 1 else None] += 1
        assert actual[action] == pytest.approx(
            {seat: count / len(profiles) for seat, count in expected.items()}
        )


def test_hidden_votes_can_eliminate_someone_other_than_our_target_including_us():
    view = view_for(wolves=(1, 6), seer=2, alive=(0, 1, 2))
    result = plan(view, joint(((1, 6), 2, 1)))
    diagnostic = action_diagnostics(result, "1")
    deaths = diagnostic["death_distribution"]
    assert 0 < deaths["1"] < 1
    assert deaths["0"] > 0 and deaths["none"] > 0
    # With three seats and our vote for 1, only one other seat can vote for 2;
    # that cannot beat our existing vote. A vote for us can still win outright.
    assert deaths.get("2", 0) == 0
    assert diagnostic["terminal_mass"]["villager"] == pytest.approx(deaths["1"])
    assert diagnostic["terminal_mass"]["wolf"] == pytest.approx(deaths["0"])


def test_stall_loss_is_only_no_elimination_branch_and_elimination_resets_counter():
    view = view_for(stalled=2)
    belief = joint(((1, 2), 3, 1))
    stalled = plan(view, belief)
    view.stalled_days = 0
    fresh = plan(view, belief)
    for action in view.action_ids:
        diagnostic = action_diagnostics(stalled, action)
        assert diagnostic["terminal_mass"]["wolf"] == pytest.approx(
            diagnostic["death_distribution"]["none"]
        )
        assert action_diagnostics(fresh, action)["terminal_mass"]["wolf"] == 0
        # Fresh no-elimination cutoff uses d=1; d=3 instead ends with utility -1.
        fresh_none_value = _value(view.alive, (1, 2), 3, 1)[0]
        assert stalled["action_values"][action] == pytest.approx(
            fresh["action_values"][action]
            + diagnostic["death_distribution"]["none"] * (-1 - fresh_none_value)
        )


@pytest.mark.parametrize(
    "alive,wolves,stalled,winner,value",
    [
        ((0, 3, 4), (1, 2), 0, "villager", 1),
        ((0, 1, 2, 3), (1, 2), 0, "wolf", -1),
        ((0, 1, 2), (1, 2), 0, "wolf", -1),
        (tuple(range(7)), (1, 2), 3, "wolf", -1),
    ],
)
def test_terminal_team_value(alive, wolves, stalled, winner, value):
    assert _value(alive, wolves, 3, stalled) == (value, winner)


@pytest.mark.parametrize("wolves", [(1, 2), (1, 6)])
def test_night_hidden_choices_match_exhaustive_team_only_policy(wolves):
    alive = (0, 1, 2, 3, 4)
    candidates = [seat for seat in alive if seat not in wolves]
    living_wolves = len(set(alive) & set(wolves))
    expected = Counter(
        min(choices) for choices in product(candidates, repeat=living_wolves)
    )
    assert _night_outcomes(alive, wolves, 2, None) == pytest.approx(
        {
            seat: count / len(candidates) ** living_wolves
            for seat, count in expected.items()
        }
    )


def test_wolf_disagreement_is_minimum_not_guaranteed_requested_target():
    view = view_for(self_id=1, subphase="night_wolf")
    result = plan(view, joint(((1, 2), 3, 1)))
    assert action_diagnostics(result, "6")["death_distribution"] == pytest.approx(
        {"0": 0.2, "3": 0.2, "4": 0.2, "5": 0.2, "6": 0.2}
    )
    assert action_diagnostics(result, "3")["death_distribution"] == pytest.approx(
        {"0": 0.2, "3": 0.8}
    )
    assert action_diagnostics(result, "0")["death_distribution"] == {"0": 1}


def test_lone_wolf_night_choice_is_decisive_and_seer_removal_has_team_value():
    view = view_for(
        self_id=1, wolves=(1, 6), alive=(0, 1, 2, 3, 4, 5), subphase="night_wolf"
    )
    result = plan(view, joint(((1, 6), 3, 1)))
    for action in view.action_ids:
        assert action_diagnostics(result, action)["death_distribution"] == {action: 1}
    assert result["action_id"] == "3"
    assert result["action_values"]["3"] > result["action_values"]["0"]


@pytest.mark.parametrize("self_id,subphase", [(1, "night_wolf"), (3, "night_seer")])
def test_peaceful_first_night_has_no_death_or_parity(self_id, subphase):
    view = view_for(self_id=self_id, subphase=subphase, day=1)
    result = plan(view, uniform(view))
    for diagnostic in result["diagnostics"]["actions"].values():
        assert diagnostic["death_distribution"] == {"none": pytest.approx(1)}
        assert diagnostic["terminal_mass"] == {"wolf": 0, "villager": 0}
    if subphase == "night_wolf":
        assert len(set(result["action_values"].values())) == 1
        assert result["action_id"] == "0"
    else:
        p = 2 / 6
        expected = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        for diagnostic in result["diagnostics"]["actions"].values():
            assert diagnostic["usable_information_bits"] == pytest.approx(expected)


def test_dawn_parity_ends_game_and_makes_check_information_unusable():
    view = view_for(
        self_id=0, wolves=(3, 4), seer=0, alive=(0, 1, 2, 3, 4), subphase="night_seer"
    )
    result = plan(view, uniform(view))
    for action, diagnostic in result["diagnostics"]["actions"].items():
        assert diagnostic["terminal_mass"]["wolf"] == pytest.approx(1)
        assert diagnostic["usable_information_bits"] == 0
        assert result["action_values"][action] == pytest.approx(-1)


def test_check_is_useful_only_if_seer_survives_and_result_is_not_public():
    view = view_for(self_id=0, seer=0, subphase="night_seer")
    belief = joint(((1, 2), 0, 0.5), ((1, 3), 0, 0.5))
    result = plan(view, belief)
    # Wolf minimum targets 0 with probability 1-(4/5)^2=9/25. On all
    # other dawns seat 2 is either alive or its public death disambiguates it.
    diagnostic = action_diagnostics(result, "2")
    assert diagnostic["seer_survival_mass"] == pytest.approx(16 / 25)
    # Only deaths at 4, 5, or 6 leave the two result hypotheses ambiguous.
    assert diagnostic["usable_information_bits"] == pytest.approx(9 / 25)
    assert action_diagnostics(result, "1")["usable_information_bits"] == 0
    assert result["action_values"]["2"] == pytest.approx(
        diagnostic["cutoff_value"]
        + INFORMATION_WEIGHT * diagnostic["usable_information_bits"]
    )


def test_night_information_discount_includes_death_and_public_revelation():
    # The hypotheses swap wolf/nonwolf seats; the living seer is minimum target
    # in both. The first night has one full usable bit, not so on later nights.
    view = view_for(self_id=0, wolves=(1, 6), seer=0, subphase="night_seer", day=1)
    belief = joint(((1, 6), 0, 0.5), ((2, 6), 0, 0.5))
    peaceful = plan(view, belief)
    view.day = 2
    dangerous = plan(view, belief)
    assert action_diagnostics(peaceful, "1")["usable_information_bits"] == 1
    # Both living wolves choose 0 with 9/25 probability: those checks add zero.
    assert action_diagnostics(dangerous, "1")[
        "usable_information_bits"
    ] == pytest.approx(9 / 25)
    assert action_diagnostics(dangerous, "1")["seer_survival_mass"] == pytest.approx(
        16 / 25
    )


def test_doomed_seer_check_is_legal_but_has_no_communication_value(monkeypatch):
    view = view_for(self_id=0, seer=0, subphase="night_seer")
    # Isolate the dawn death branch of the same model, where no terminal parity
    # occurs and the legal check must not acquire posthumous information utility.
    monkeypatch.setattr(planner, "_night_outcomes", lambda *args: {0: 1.0})
    result = plan(view, uniform(view))
    assert result["action_id"] in view.action_ids
    for diagnostic in result["diagnostics"]["actions"].values():
        assert diagnostic["death_distribution"] == {"0": pytest.approx(1)}
        assert diagnostic["terminal_mass"] == {"wolf": 0, "villager": 0}
        assert diagnostic["seer_survival_mass"] == 0
        assert diagnostic["usable_information_bits"] == 0


def test_authenticated_previous_check_has_no_information_bonus():
    view = view_for(
        self_id=0, seer=0, subphase="night_seer", day=1, investigations={1: "wolf"}
    )
    result = plan(view, uniform(view))
    assert action_diagnostics(result, "1")["usable_information_bits"] == 0
    assert any(
        diagnostic["usable_information_bits"] > 0
        for diagnostic in result["diagnostics"]["actions"].values()
    )


def test_day_opponent_ballots_ignore_worlds_and_our_private_checks():
    first = view_for(self_id=0, seer=0, investigations={1: "wolf"})
    second = view_for(self_id=0, seer=0, investigations={1: "villager"})
    a = plan(first, uniform(first))
    b = plan(second, uniform(second))
    for action in first.action_ids:
        assert action_diagnostics(a, action)["death_distribution"] == pytest.approx(
            action_diagnostics(b, action)["death_distribution"]
        )


def test_night_wolf_opponent_policy_knows_team_but_not_true_seer():
    view = view_for(self_id=1, subphase="night_wolf")
    a = plan(view, joint(((1, 2), 3, 1)))
    b = plan(view, joint(((1, 2), 6, 1)))
    for action in view.action_ids:
        assert action_diagnostics(a, action)["death_distribution"] == pytest.approx(
            action_diagnostics(b, action)["death_distribution"]
        )
    assert a["action_values"] != b["action_values"]


def test_aggregate_world_values_then_maximize_not_per_world_policy():
    view = view_for(
        self_id=1, wolves=(1, 6), alive=(0, 1, 2, 3, 4, 5), subphase="night_wolf"
    )
    first = plan(view, joint(((1, 6), 2, 1)))
    second = plan(view, joint(((1, 6), 3, 1)))
    mixed = plan(view, joint(((1, 6), 2, 0.25), ((1, 6), 3, 0.75)))
    assert first["action_id"] == "2"
    assert second["action_id"] == mixed["action_id"] == "3"
    for action in view.action_ids:
        assert mixed["action_values"][action] == pytest.approx(
            0.25 * first["action_values"][action]
            + 0.75 * second["action_values"][action]
        )
    assert max(mixed["action_values"].values()) < (
        0.25 * max(first["action_values"].values())
        + 0.75 * max(second["action_values"].values())
    )


def test_metadata_is_ignored_and_caller_inputs_are_unchanged():
    view = view_for()
    belief = uniform(view)
    original_view, original_belief = deepcopy(vars(view)), deepcopy(belief)
    result = plan(view, belief)
    assert vars(view) == original_view
    assert belief == original_belief
    belief.update(
        mode="any-independent-scorer", marginals={"bad": "ignored"}, entropy=-100
    )
    assert plan(view, belief) == result


def test_stable_under_repetition_and_input_order_and_does_not_touch_global_rng():
    import random

    view = view_for()
    belief = uniform(view)
    state = random.getstate()
    result = plan(view, belief)
    assert plan(view, belief) == result
    view.action_ids = tuple(reversed(view.action_ids))
    view.alive = tuple(reversed(view.alive))
    belief["worlds"].reverse()
    for world in belief["worlds"]:
        world["wolves"].reverse()
    assert plan(view, belief) == result
    assert random.getstate() == state


def test_worst_table_work_is_bounded_and_not_multiplied_by_worlds_or_actions():
    view = view_for()
    result = plan(view, uniform(view))
    work = result["diagnostics"]["work"]
    assert work["world_count"] == 60
    assert work["action_count"] == 7
    assert work["ballot_distributions_built"] == 1
    assert work["hidden_ballot_profiles"] == 7**6
    assert work["ballot_peak_states"] <= math.comb(13, 7)
    assert work["ballot_tallies_resolved"] <= 7 * math.comb(13, 7)
    assert work["ballot_transitions"] < 20000
    assert work["outcome_evaluations"] <= 60 * 7 * 8
    single = plan(view, joint(((1, 2), 3, 1)))["diagnostics"]["work"]
    for field in (
        "ballot_distributions_built",
        "ballot_tallies_resolved",
        "ballot_transitions",
    ):
        assert work[field] == single[field]


def test_module_has_no_scorer_model_or_io_dependencies():
    tree = ast.parse(inspect.getsource(planner))
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports == {"math", "collections", "belief"}
    assert "BeliefEngine" not in inspect.getsource(planner)


@pytest.mark.parametrize(
    "belief",
    [
        None,
        {},
        {"worlds": []},
        {"worlds": [None]},
        {"worlds": [{}]},
        joint(((1, 2), 3, float("nan"))),
        joint(((1, 2), 3, float("inf"))),
        joint(((1, 2), 3, -0.1)),
        joint(((1, 2), 3, 0)),
        joint(((1, 2), 3, 0.9)),
        joint(((1, 2), 3, 1.1)),
        joint(((1, 2), 3, True)),
        joint(((1, 2), 3, 10**400)),
        joint(((1, 1), 3, 1)),
        joint(((True, 2), 3, 1)),
        joint(((1, 7), 3, 1)),
        joint(((1, 2), True, 1)),
        joint(((1, 2), 1, 1)),
        joint(((1, 2), 3, 0.5), ((2, 1), 3, 0.5)),
        joint(((0, 2), 3, 1)),
        joint(((1, 2), 0, 1)),
        {"worlds": [{"wolves": [1, 2, 4], "seer": 3, "probability": 1}]},
        {"worlds": [{}] * 106},
    ],
)
def test_malformed_or_infeasible_joint_beliefs_fail_closed(belief):
    with pytest.raises(EvidenceError):
        plan(view_for(), belief)


@pytest.mark.parametrize("constraint", ["partner", "investigation", "reveal"])
def test_all_authenticated_fact_kinds_constrain_supplied_worlds(constraint):
    if constraint == "partner":
        view = view_for(self_id=1)
        belief = joint(((1, 4), 3, 1))
    elif constraint == "investigation":
        view = view_for(self_id=0, seer=0, investigations={1: "wolf"})
        belief = joint(((2, 3), 0, 1))
    else:
        view = view_for(alive=(0, 1, 2, 4, 5, 6))
        belief = joint(((1, 2), 4, 1))
    with pytest.raises(EvidenceError):
        plan(view, belief)


def test_normalization_tolerance_zero_and_tiny_finite_weights():
    view = view_for(self_id=0, seer=0, subphase="night_seer", day=1)
    almost = joint(((1, 2), 0, 1 + 0.5e-9))
    # Individual weights must still be valid probabilities, even within sum tolerance.
    with pytest.raises(EvidenceError):
        plan(view, almost)
    belief = joint(((1, 2), 0, 1 - 0.5e-9), ((1, 3), 0, 0), ((1, 4), 0, 5e-324))
    result = plan(view, belief)
    assert result["diagnostics"]["work"]["world_count"] == 2
    json.dumps(result, allow_nan=False)
    assert all(math.isfinite(value) for value in result["action_values"].values())


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "messaging"),
        ("phase", "terminal"),
        ("subphase", "terminal"),
        ("subphase", "night_seer"),
        ("subphase", "night_wolf"),
        ("day", True),
        ("day", 0),
        ("stalled_days", True),
        ("stalled_days", -1),
        ("stalled_days", 3),
        ("action_ids", ()),
        ("action_ids", ("1", "1")),
        ("action_ids", ("0",)),
        ("action_ids", ("01",)),
        ("action_ids", ("8",)),
        ("action_ids", (True,)),
        ("alive", (0, 1, 2, 3, 4, 5, 5)),
        ("alive", (1, 2, 3, 4, 5, 6)),
        ("alive", (0, 1, 2, 3, 4, 5)),
    ],
)
def test_invalid_or_nonactionable_view_fails_closed(field, value):
    view = view_for()
    belief = uniform(view)
    setattr(view, field, value)
    with pytest.raises(EvidenceError):
        plan(view, belief)


@pytest.mark.parametrize("action", ["7", "2"])
def test_night_actions_cannot_abstain_or_target_fellow_wolf(action):
    view = view_for(self_id=1, subphase="night_wolf", actions=(action,))
    with pytest.raises(EvidenceError):
        plan(view, joint(((1, 2), 3, 1)))


def test_terminal_parity_view_cannot_plan_even_with_supplied_actions():
    view = view_for(alive=(0, 1, 2, 3))
    with pytest.raises(EvidenceError):
        plan(view, joint(((1, 2), 3, 1)))
