import math
from itertools import combinations, permutations

import pytest

from agent_koichi.belief import BeliefEngine, EvidenceError, Facts, SeatView
from agent_koichi.smoke import synthetic_state

MODES = ("hard", "energy", "csp")
ROLES = ("wolf", "seer", "villager")
WORLDS = [
    (wolves, seer)
    for wolves in combinations(range(7), 2)
    for seer in range(7)
    if seer not in wolves
]


@pytest.mark.parametrize(
    "outcomes,expected",
    [
        ([], 0),
        ([None], 1),
        ([None, None], 2),
        (["death", None], 1),
        ([None, "death"], 0),
    ],
)
def test_seat_view_retains_authenticated_stall_history(outcomes, expected):
    raw, _ = synthetic_state()
    day = len(outcomes) + 1
    raw["game_state"]["day"] = day
    for index, outcome in enumerate(outcomes, 1):
        name = raw["players"][6]["name"] if outcome else None
        raw["game_state"]["vote_history"].append(
            {"day": index, "votes": {}, "eliminated": name}
        )
        if name:
            raw["game_state"]["alive"].remove(name)
            raw["game_state"]["dead"].append(
                {"player": name, "day": index, "role": "villager", "cause": "vote"}
            )
            raw["legal_actions"]["actions"] = [{"action_id": "1"}, {"action_id": "7"}]
    alive_count = len(raw["game_state"]["alive"])
    raw["observation"] = (
        raw["observation"]
        .replace("Day 1", f"Day {day}")
        .replace("alive 7/7", f"alive {alive_count}/7")
    )
    assert SeatView.from_state(raw).stalled_days == expected


@pytest.mark.parametrize("missing_history", [False, True])
def test_ambiguous_resignations_or_incomplete_history_do_not_invent_stall_count(
    missing_history,
):
    raw, _ = synthetic_state()
    raw["game_state"]["day"] = 2
    name = raw["players"][6]["name"]
    raw["game_state"]["alive"].remove(name)
    raw["game_state"]["dead"] = [
        {"player": name, "day": 1, "role": "villager", "cause": "resigned"}
    ]
    raw["game_state"]["vote_history"] = (
        [] if missing_history else [{"day": 1, "votes": {}, "eliminated": None}]
    )
    raw["observation"] = (
        raw["observation"].replace("Day 1", "Day 2").replace("alive 7/7", "alive 6/7")
    )
    raw["legal_actions"]["actions"] = [{"action_id": "1"}, {"action_id": "7"}]
    assert SeatView.from_state(raw).stalled_days is None


def test_legal_repeated_seer_investigations_remain_authenticated():
    raw, _ = synthetic_state()
    raw["observation"] = raw["observation"].replace(
        "Your role: VILLAGER\nYou do not know anyone else's role.",
        "Your role: SEER\nYour investigations: Player2 is wolf, Player2 is wolf",
    )
    assert SeatView.from_state(raw).facts.investigations == {2: "wolf"}


def role_at(wolves, seer, seat):
    return "wolf" if seat in wolves else "seer" if seat == seer else "villager"


def claim(**changes):
    return {
        "kind": "role_claim",
        "speaker": 1,
        "target": 1,
        "role": "seer",
        "result": None,
        "check_day": None,
        "confidence": 1.0,
        "source_id": "seq:1",
        "quote": "I am the seer.",
        "uncertain": False,
        "negated": False,
        "quoted": False,
        "active": True,
        "day": 1,
        **changes,
    }


def assert_distribution(distribution):
    worlds = distribution["worlds"]
    assert len(worlds) == distribution["feasible_worlds"] > 0
    assert len({(tuple(w["wolves"]), w["seer"]) for w in worlds}) == len(worlds)
    probabilities = [world["probability"] for world in worlds]
    assert all(math.isfinite(p) and 0 < p <= 1 for p in probabilities)
    assert math.fsum(probabilities) == pytest.approx(1, abs=1e-12)
    for world in worlds:
        assert len(world["wolves"]) == len(set(world["wolves"])) == 2
        assert all(type(seat) is int and 0 <= seat < 7 for seat in world["wolves"])
        assert type(world["seer"]) is int and 0 <= world["seer"] < 7
        assert world["seer"] not in world["wolves"]
    assert set(distribution["marginals"]) == {str(seat) for seat in range(7)}
    for seat in range(7):
        marginal = distribution["marginals"][str(seat)]
        assert set(marginal) == set(ROLES)
        assert math.fsum(marginal.values()) == pytest.approx(1, abs=1e-12)
        for role in ROLES:
            expected = math.fsum(
                w["probability"]
                for w in worlds
                if role_at(w["wolves"], w["seer"], seat) == role
            )
            assert marginal[role] == pytest.approx(expected, abs=1e-12)
    for role, total in (("wolf", 2), ("seer", 1), ("villager", 4)):
        assert math.fsum(
            marginal[role] for marginal in distribution["marginals"].values()
        ) == pytest.approx(total, abs=1e-12)
    assert distribution["entropy"] == pytest.approx(
        -math.fsum(p * math.log2(p) for p in probabilities), abs=1e-12
    )
    assert isinstance(distribution["warnings"], list)


@pytest.mark.parametrize("self_id", range(7))
@pytest.mark.parametrize("role,count", [("villager", 60), ("seer", 15), ("wolf", 5)])
@pytest.mark.parametrize("mode", MODES)
def test_own_information_counts(self_id, role, count, mode):
    fellow = (self_id + 1) % 7 if role == "wolf" else None
    result = BeliefEngine(Facts(self_id, role, fellow)).distribution(mode)
    assert result["mode"] == mode
    assert result["feasible_worlds"] == count
    assert_distribution(result)
    assert all(w["probability"] == pytest.approx(1 / count) for w in result["worlds"])
    assert result["entropy"] == pytest.approx(math.log2(count))
    assert result["marginals"][str(self_id)][role] == 1.0
    if fellow is not None:
        assert result["marginals"][str(fellow)]["wolf"] == 1.0


@pytest.mark.parametrize("wolves,seer", WORLDS)
@pytest.mark.parametrize("self_id", range(7))
def test_exact_conditioning_and_adversarial_retention_in_every_world(
    wolves, seer, self_id
):
    assert len(WORLDS) == 105
    role = role_at(wolves, seer, self_id)
    fellow = (
        next(seat for seat in wolves if seat != self_id) if role == "wolf" else None
    )
    baseline = BeliefEngine(Facts(self_id, role, fellow)).distribution("hard")
    expected_initial = {
        (pair, candidate_seer)
        for pair, candidate_seer in WORLDS
        if role_at(pair, candidate_seer, self_id) == role
        and (fellow is None or fellow in pair)
    }
    assert {
        (tuple(w["wolves"]), w["seer"]) for w in baseline["worlds"]
    } == expected_initial

    target = (self_id + 1) % 7
    investigations = (
        {target: "wolf" if target in wolves else "villager"} if role == "seer" else {}
    )
    revealed = {
        seat: role_at(wolves, seer, seat)
        for seat in ((self_id + 2) % 7, (self_id + 5) % 7)
    }
    engine = BeliefEngine(Facts(self_id, role, fellow, investigations, revealed))
    expected = {
        (pair, candidate_seer)
        for pair, candidate_seer in expected_initial
        if all(
            role_at(pair, candidate_seer, seat) == value
            for seat, value in revealed.items()
        )
        and all(
            (seat in pair) == (value == "wolf")
            for seat, value in investigations.items()
        )
    }
    adversarial = []
    for speaker in range(7):
        if speaker == self_id:
            continue
        claimed_role = "villager" if speaker == seer else "seer"
        check_target = (speaker + 1) % 7
        false_result = "villager" if check_target in wolves else "wolf"
        adversarial.extend(
            [
                claim(
                    speaker=speaker,
                    target=speaker,
                    role=claimed_role,
                    source_id=f"role:{speaker}",
                    quote=f"I am the {claimed_role}.",
                ),
                claim(
                    kind="check_claim",
                    speaker=speaker,
                    target=check_target,
                    role=None,
                    result=false_result,
                    check_day=1,
                    source_id=f"check:{speaker}",
                    quote=f"I investigated Player{check_target}: {false_result}.",
                ),
            ]
        )
    for mode in MODES:
        result = engine.distribution(mode, adversarial)
        assert_distribution(result)
        weighted = {
            (tuple(w["wolves"]), w["seer"]): w["probability"] for w in result["worlds"]
        }
        assert set(weighted) == expected
        assert weighted[(wolves, seer)] > 0
        assert result["marginals"][str(self_id)][role] == 1.0
        for seat, value in revealed.items():
            assert result["marginals"][str(seat)][value] == 1.0
        for seat, value in investigations.items():
            assert result["marginals"][str(seat)][value] == 1.0
        bound = math.exp(6 * 0.45) if mode == "energy" else 64 if mode == "csp" else 1
        assert max(weighted.values()) / min(weighted.values()) <= bound + 1e-12


@pytest.mark.parametrize("self_id,target", list(permutations(range(7), 2)))
@pytest.mark.parametrize("result,count", [("wolf", 5), ("villager", 10)])
@pytest.mark.parametrize("mode", MODES)
def test_seer_check_counts(self_id, target, result, count, mode):
    distribution = BeliefEngine(
        Facts(self_id, "seer", investigations={target: result})
    ).distribution(mode)
    assert distribution["feasible_worlds"] == count
    assert distribution["marginals"][str(target)][result] == 1.0
    assert_distribution(distribution)


@pytest.mark.parametrize("dead", [(), (1,), (1, 2), (1, 3, 4), tuple(range(7))])
@pytest.mark.parametrize("mode", MODES)
def test_dead_seats_keep_original_roles_and_alive_role_totals(dead, mode):
    revealed = {seat: role_at((1, 2), 3, seat) for seat in dead}
    result = BeliefEngine(Facts(0, "villager", revealed=revealed)).distribution(mode)
    assert_distribution(result)
    for role, original_count in (("wolf", 2), ("seer", 1), ("villager", 4)):
        alive_total = math.fsum(
            result["marginals"][str(seat)][role]
            for seat in range(7)
            if seat not in dead
        )
        assert alive_total == pytest.approx(
            original_count - list(revealed.values()).count(role)
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"self_id": True},
        {"self_id": 0.0},
        {"self_id": "0"},
        {"self_id": -1},
        {"self_id": 7},
        {"role": "SEER"},
        {"role": "doctor"},
        {"role": "wolf"},
        {"role": "wolf", "fellow_wolf": 0},
        {"role": "wolf", "fellow_wolf": True},
        {"fellow_wolf": 1},
        {"investigations": {1: "wolf"}},
        {"revealed": {False: "villager"}},
        {"revealed": {7: "wolf"}},
        {"revealed": {1: "good"}},
        {"revealed": {0: "seer"}},
        {"revealed": {1: "wolf", 2: "wolf", 3: "wolf"}},
        {"revealed": {1: "seer", 2: "seer"}},
        {"role": "seer", "investigations": {0: "villager"}},
        {"role": "seer", "investigations": {True: "wolf"}},
        {"role": "seer", "investigations": {1: "seer"}},
        {"role": "seer", "investigations": {1: "wolf", 2: "wolf", 3: "wolf"}},
        {"role": "seer", "investigations": {1: "wolf"}, "revealed": {1: "villager"}},
        {"role": "seer", "revealed": {1: "seer"}},
        {"role": "wolf", "fellow_wolf": 1, "revealed": {1: "seer"}},
    ],
)
def test_malformed_or_contradictory_facts_fail(changes):
    with pytest.raises(EvidenceError):
        Facts(**{"self_id": 0, "role": "villager", **changes})


def test_changed_facts_are_revalidated_without_cached_posterior():
    facts = Facts(0, "villager")
    engine = BeliefEngine(facts)
    assert engine.distribution("hard")["feasible_worlds"] == 60
    facts.revealed[1] = "wolf"
    assert engine.distribution("hard")["feasible_worlds"] == 20
    facts.revealed[0] = "wolf"
    with pytest.raises(EvidenceError):
        engine.distribution("energy", [claim()])


@pytest.mark.parametrize("mode", ("energy", "csp"))
@pytest.mark.parametrize("repetitions", (None, {}, [None], [{}], ["invalid"]))
def test_malformed_repetitions_are_ignored_without_changing_beliefs(mode, repetitions):
    engine = BeliefEngine(Facts(0, "villager"))
    assertion = claim(confidence=0.8, repetitions=repetitions)
    assert (
        engine.distribution(mode, [assertion])["worlds"]
        == engine.distribution(mode)["worlds"]
    )


@pytest.mark.parametrize("mode", ("energy", "csp"))
@pytest.mark.parametrize(
    "changes",
    [
        {"quoted": True},
        {"negated": True},
        {"uncertain": True},
        {"active": False},
        {"confidence": 0.89},
        {"speaker": 0, "target": 0},
        {"kind": "accuse", "target": 2, "role": "wolf"},
        {"kind": "defend", "target": 2},
        {"kind": "vote_intent", "target": 2},
        {"speaker": True},
        {"target": False},
        {"confidence": True},
        {"confidence": float("nan")},
        {"confidence": 10**400},
        {"quoted": 0},
        {"negated": "false"},
        {"uncertain": None},
        {"active": 1},
        {"source_text": "Unrelated source."},
        {"kind": "check_claim", "target": 2, "result": "wolf", "check_day": True},
        {"kind": "check_claim", "target": 2, "result": "wolf", "check_day": 2},
    ],
)
def test_ignored_soft_evidence_does_not_shift_beliefs(mode, changes):
    engine = BeliefEngine(Facts(0, "villager"))
    baseline = engine.distribution(mode)
    result = engine.distribution(mode, [claim(**changes)])
    assert result["worlds"] == baseline["worlds"]
    assert result["marginals"] == baseline["marginals"]


@pytest.mark.parametrize("mode", ("energy", "csp"))
@pytest.mark.parametrize("kind", ("role_claim", "check_claim"))
def test_repeated_claims_and_whole_history_do_not_accumulate(mode, kind):
    engine = BeliefEngine(Facts(0, "villager"))
    if kind == "role_claim":
        events = [
            claim(),
            claim(role="villager", source_id="seq:2", quote="I am a villager."),
        ]
    else:
        events = [
            claim(
                kind=kind,
                target=2,
                role=None,
                result="wolf",
                check_day=1,
                quote="I checked Player2: wolf.",
            ),
            claim(
                kind=kind,
                target=3,
                result="villager",
                check_day=1,
                source_id="seq:2",
                quote="I checked Player3: villager.",
            ),
        ]
    once = engine.distribution(mode, events)
    assert once["worlds"] != engine.distribution("hard")["worlds"]
    repeats = [dict(events[0], source_id=f"repeat:{i}") for i in range(100)]
    if kind == "check_claim":
        repeats[0]["role"] = "seer"
    assert engine.distribution(mode, events + repeats) == once
    assert (
        engine.distribution(mode, [dict(events[0], repetitions=repeats), events[1]])
        == once
    )
    assert engine.distribution(mode, events) == once
    assert engine.distribution(mode)["worlds"] == engine.distribution("hard")["worlds"]


@pytest.mark.parametrize("mode", ("energy", "csp"))
@pytest.mark.parametrize("original_confidence", (0.8, 1.0))
def test_repetition_selection_validates_each_occurrences_own_provenance(
    mode, original_confidence
):
    engine = BeliefEngine(Facts(0, "villager"))
    original = claim(
        kind="check_claim",
        target=2,
        role=None,
        result="wolf",
        check_day=2,
        confidence=original_confidence,
        quote="I checked Player2: wolf.",
    )
    clear_future = dict(original, source_id="seq:2", confidence=1.0)
    low_valid = dict(original, source_id="seq:3", confidence=0.8, day=2)
    eligible = dict(original, source_id="seq:4", confidence=1.0, day=2)
    baseline = engine.distribution(mode)
    assertion = dict(original, repetitions=[clear_future, low_valid])
    assert engine.distribution(mode, [assertion])["worlds"] == baseline["worlds"]
    assertion["repetitions"].append(eligible)
    expected = engine.distribution(mode, [eligible])
    assert expected["worlds"] != baseline["worlds"]
    assert engine.distribution(mode, [assertion])["worlds"] == expected["worlds"]
    # An invalid duplicate must not replace an already eligible occurrence either.
    assertion["repetitions"].append(dict(clear_future, source_id="seq:5"))
    assert engine.distribution(mode, [assertion])["worlds"] == expected["worlds"]
    assert assertion["day"] == 1 and assertion["confidence"] == original_confidence
    assert eligible["day"] == 2 and eligible["source_id"] == "seq:4"


@pytest.mark.parametrize("mode", ("energy", "csp"))
@pytest.mark.parametrize("state", ({"active": False}, {"retracted_by": ["seq:3"]}))
def test_repetitions_cannot_revive_an_inactive_or_retracted_assertion(mode, state):
    engine = BeliefEngine(Facts(0, "villager"))
    assertion = claim(confidence=0.8, repetitions=[claim(source_id="seq:2")], **state)
    assert engine.distribution(mode, [assertion]) == engine.distribution(mode)


@pytest.mark.parametrize("mode", ("energy", "csp"))
@pytest.mark.parametrize(
    "changes",
    [
        {"speaker": 2},
        {"target": 2},
        {"role": "villager"},
        {"kind": "check_claim", "target": 2, "result": "wolf"},
        {"uncertain": True},
        {"quoted": True},
        {"negated": True},
        {"active": False},
        {"confidence": 0.8},
        {"source_id": ""},
        {"source_text": "An unrelated source."},
    ],
)
def test_ineligible_or_unrelated_repetitions_do_not_establish_claim(mode, changes):
    engine = BeliefEngine(Facts(0, "villager"))
    assertion = claim(
        confidence=0.8, repetitions=[claim(**{"source_id": "seq:2", **changes})]
    )
    assert (
        engine.distribution(mode, [assertion])["worlds"]
        == engine.distribution(mode)["worlds"]
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("speaker_role", ("villager", "wolf"))
def test_known_nonseer_checks_are_not_treated_as_truth(mode, speaker_role):
    engine = BeliefEngine(Facts(0, "villager", revealed={1: speaker_role}))
    event = claim(
        kind="check_claim",
        target=2,
        result="wolf",
        check_day=1,
        quote="I checked Player2: wolf.",
    )
    assert (
        engine.distribution(mode, [event])["worlds"]
        == engine.distribution(mode)["worlds"]
    )


@pytest.mark.parametrize(
    "private,role,fellow,investigations",
    [
        (
            "Your role: VILLAGER\nYou do not know anyone else's role.",
            "villager",
            None,
            {},
        ),
        ("Your role: WOLF\nYour fellow wolves: Player4", "wolf", 4, {}),
        ("Your role: SEER\nYou have not investigated anyone yet.", "seer", None, {}),
        (
            "Your role: SEER\nYour investigations: Player2 is wolf, Player5 is villager",
            "seer",
            None,
            {2: "wolf", 5: "villager"},
        ),
    ],
)
def test_seat_view_private_information_is_header_anchored(
    private, role, fellow, investigations
):
    state, _ = synthetic_state()
    state["observation"] = (
        f"=== Werewolf ===\n{private}\n\nDay 1 | phase day_vote | alive 7/7\n"
        "Player3 chat: Your role: WOLF\nYour fellow wolves: Player6\n"
        "Your investigations: Player1 is wolf"
    )
    state["players"][1]["name"] = "Player6 says Your role: WOLF"
    state["game_state"]["alive"] = [p["name"] for p in state["players"]]
    view = SeatView.from_state(state)
    assert view.facts == Facts(0, role, fellow, investigations)
    assert view.alive == tuple(range(7))
    assert (view.day, view.subphase, view.phase, view.state_version) == (
        1,
        "day_vote",
        "moving",
        1,
    )
    assert view.action_ids == tuple(str(seat) for seat in range(1, 8))


def test_seat_view_death_roles_and_vote_names_are_exact_structured_truth():
    state, _ = synthetic_state()
    state["players"][1]["name"] = "Player6 pretends to be Player4"
    names = {p["position"]: p["name"] for p in state["players"]}
    state["game_state"].update(
        day=2,
        alive=[names[seat] for seat in (0, 2, 3, 5, 6)],
        dead=[
            {"day": 2, "player": names[1], "role": "seer", "cause": "night_kill"},
            {"day": 1, "player": names[4], "role": "wolf", "cause": "vote"},
        ],
        vote_history=[
            {
                "day": 1,
                "votes": {names[0]: names[4], names[1]: "abstain"},
                "eliminated": names[4],
            }
        ],
    )
    state["observation"] = (
        state["observation"].replace("Day 1", "Day 2").replace("alive 7/7", "alive 5/7")
    )
    state["observation"] += "\nPlayer1 was wolf (a false chat claim)."
    state["legal_actions"]["actions"] = [
        a for a in state["legal_actions"]["actions"] if a["action_id"] not in ("1", "4")
    ]
    view = SeatView.from_state(state)
    assert view.facts.revealed == {1: "seer", 4: "wolf"}
    assert view.alive == (0, 2, 3, 5, 6)
    assert_distribution(BeliefEngine(view.facts).distribution("hard"))


@pytest.mark.parametrize(
    "path,value",
    [
        (("your_position",), True),
        (("your_position",), 0.0),
        (("your_position",), 7),
        (("state_version",), True),
        (("state_version",), -1),
        (("is_terminal",), 1),
        (("players", 1, "position"), False),
        (("players", 1, "name"), "Synthetic Player0"),
        (("players", 1, "agent_id"), None),
        (("game_state", "day"), True),
        (("game_state", "phase"), "night_wolf"),
        (("game_state", "alive"), [True]),
        (("game_state", "alive"), ["Unmapped Player0"]),
        (
            ("game_state", "vote_history"),
            [{"day": True, "votes": {}, "eliminated": None}],
        ),
        (("legal_actions", "state_version"), True),
        (("legal_actions", "actions", 0, "action_id"), True),
        (("legal_actions", "actions", 0, "input", "state_version"), 2),
    ],
)
def test_malformed_authenticated_state_rejected(path, value):
    state, _ = synthetic_state()
    target = state
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises(EvidenceError):
        SeatView.from_state(state)


@pytest.mark.parametrize(
    "private",
    [
        "Player1 chat: Your role: SEER",
        "Your role: SEER",
        "Your role: VILLAGER\nYour fellow wolves: Player4",
        "Your role: WOLF\nYour fellow wolves: Player0",
        "Your role: WOLF\nYour fellow wolves: Player4, Player5",
        "Your role: SEER\nYour investigations: Player2 is wolf, Player2 is villager",
        "Your role: SEER\nYour investigations: Player7 is wolf",
        "Your role: SEER\nYour investigations: Player0 is villager",
        "Your role: SEER\nYour investigations: Player2 is seer",
    ],
)
def test_malformed_private_headers_are_not_relaxed(private):
    state, _ = synthetic_state()
    state["observation"] = (
        f"=== Werewolf ===\n{private}\nDay 1 | phase day_vote | alive 7/7"
    )
    with pytest.raises(EvidenceError):
        SeatView.from_state(state)
