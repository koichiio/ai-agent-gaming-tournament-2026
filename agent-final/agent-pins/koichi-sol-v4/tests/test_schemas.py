"""Offline schema fixtures only; no inference or JSON Schema dependency."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from agent_koichi.agent import validate_decision
from agent_koichi.belief import Facts, SeatView
from agent_koichi.model import ModelError
from agent_koichi.schemas import policy_response_format


@pytest.fixture
def view():
    return SeatView(
        facts=Facts(self_id=0, role="villager"),
        alive=(6, 0, 2, 4),
        day=1,
        subphase="day_vote",
        phase="moving",
        state_version=2,
        action_ids=("7", "2", "4"),
    )


def assert_closed(schema):
    """Check every schema node, including nodes not visited by a fixture value."""
    assert set(schema) <= {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "maxItems",
        "maxLength",
        "minimum",
        "maximum",
    }
    if "enum" in schema:
        assert schema["enum"]
    if schema["type"] == "object":
        assert schema["additionalProperties"] is False
        assert schema["required"] == list(schema["properties"])
        for child in schema["properties"].values():
            assert_closed(child)
    elif schema["type"] == "array":
        assert_closed(schema["items"])


def assert_matches(value, schema):
    """Interpret only the small keyword subset emitted by this module."""
    kind = schema["type"]
    if kind == "object":
        assert isinstance(value, dict)
        assert set(value) == set(schema["required"]) == set(schema["properties"])
        for key, child in schema["properties"].items():
            assert_matches(value[key], child)
    elif kind == "array":
        assert isinstance(value, list) and len(value) <= schema["maxItems"]
        for item in value:
            assert_matches(item, schema["items"])
    elif kind == "string":
        assert isinstance(value, str)
        if "maxLength" in schema:
            assert len(value) <= schema["maxLength"]
    elif kind == "null":
        assert value is None
    elif kind == "integer":
        assert type(value) is int
    elif kind == "number":
        assert type(value) in (int, float)
        assert schema["minimum"] <= value <= schema["maximum"]
    else:
        pytest.fail(f"Unsupported fixture schema type: {kind}")
    if "enum" in schema:
        assert value in schema["enum"]


def decision(kind="action", strategy="hard", **changes):
    scores = None
    if strategy == "llm":
        scores = {
            str(seat): {"wolf": 1 / 3, "seer": 1 / 6, "villager": 0.5}
            for seat in range(7)
        }
        scores["0"] = {"wolf": 0, "seer": 0, "villager": 1}
    return {
        "action_id": "2" if kind == "action" else None,
        "content": "" if kind == "action" else "A synthetic public statement.",
        "recipients": [],
        "reason": "Synthetic policy reasoning.",
        "role_scores": scores,
    } | changes


@pytest.mark.parametrize("kind", ["action", "message"])
@pytest.mark.parametrize("strategy", ["hard", "energy", "csp", "llm"])
def test_envelope_and_recursive_closed_required_properties(view, kind, strategy):
    response_format = policy_response_format(view, kind, strategy)
    assert set(response_format) == {"type", "json_schema"}
    assert response_format["type"] == "json_schema"
    envelope = response_format["json_schema"]
    assert set(envelope) == {"name", "strict", "schema"}
    mode = "llm" if strategy == "llm" else "math"
    assert envelope["name"] == f"werewolf_policy_{kind}_{mode}"
    assert envelope["strict"] is True
    schema = envelope["schema"]
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {
        "action_id",
        "content",
        "recipients",
        "reason",
        "role_scores",
    }
    assert_closed(schema)
    output = decision(kind, strategy)
    assert_matches(output, schema)
    assert validate_decision(output, view, kind, strategy) == output
    assert json.loads(json.dumps(response_format, allow_nan=False)) == response_format
    if strategy == "llm":
        seats = schema["properties"]["role_scores"]
        assert seats["required"] == [str(seat) for seat in range(7)]
        for row in seats["properties"].values():
            assert row["required"] == ["wolf", "seer", "villager"]
            assert all(
                value == {"type": "number", "minimum": 0, "maximum": 1}
                for value in row["properties"].values()
            )
    else:
        assert schema["properties"]["role_scores"] == {"type": "null"}


def test_action_schema_uses_exact_ids_and_empty_content_and_recipients(view):
    properties = policy_response_format(view, "action", "hard")["json_schema"][
        "schema"
    ]["properties"]
    assert properties["action_id"] == {"type": "string", "enum": ["7", "2", "4"]}
    assert properties["content"] == {"type": "string", "enum": [""]}
    assert properties["recipients"] == {
        "type": "array",
        "items": {"type": "integer"},
        "maxItems": 0,
    }


@pytest.mark.parametrize(
    "alive,eligible", [((6, 0, 2, 4), [6, 2, 4]), ((0,), []), ((), [])]
)
def test_message_schema_allows_only_living_other_seats_without_empty_enums(
    view, alive, eligible
):
    view = replace(view, alive=alive, action_ids=())
    schema = policy_response_format(view, "message", "hard")["json_schema"]["schema"]
    properties = schema["properties"]
    assert properties["action_id"] == {"type": "null"}
    assert properties["content"] == {"type": "string", "maxLength": 1024}
    assert properties["reason"] == {"type": "string", "maxLength": 600}
    assert properties["recipients"] == {
        "type": "array",
        "items": {"type": "integer", **({"enum": eligible} if eligible else {})},
        "maxItems": 1 if eligible else 0,
    }
    assert_closed(schema)
    assert_matches(decision("message", content=""), schema)
    for seat in eligible:
        assert_matches(decision("message", recipients=[seat]), schema)


@pytest.mark.parametrize("kind", ["", "extract", "ACTION", None])
def test_unknown_kind_fails_explicitly(view, kind):
    with pytest.raises(ModelError, match="Unknown decision kind"):
        policy_response_format(view, kind, "hard")


def test_action_without_legal_ids_fails_explicitly(view):
    with pytest.raises(ModelError, match="No legal action"):
        policy_response_format(replace(view, action_ids=()), "action", "hard")


@pytest.mark.parametrize("kind", ["action", "message"])
@pytest.mark.parametrize("strategy", ["hard", "llm"])
@pytest.mark.parametrize(
    "field", ["action_id", "content", "recipients", "reason", "role_scores"]
)
def test_every_root_field_required_and_no_extra_fields(view, kind, strategy, field):
    schema = policy_response_format(view, kind, strategy)["json_schema"]["schema"]
    output = decision(kind, strategy)
    del output[field]
    with pytest.raises(AssertionError):
        assert_matches(output, schema)
    with pytest.raises(AssertionError):
        assert_matches(decision(kind, strategy, extra=True), schema)


@pytest.mark.parametrize(
    "kind,changes",
    [
        ("action", {"action_id": "0"}),
        ("action", {"action_id": 2}),
        ("action", {"action_id": None}),
        ("action", {"content": "not empty"}),
        ("action", {"recipients": [2]}),
        ("action", {"role_scores": {}}),
        ("message", {"action_id": "2"}),
        ("message", {"content": "x" * 1025}),
        ("message", {"recipients": [0]}),
        ("message", {"recipients": [1]}),
        ("message", {"recipients": [2, 4]}),
        ("message", {"recipients": [2.0]}),
        ("message", {"recipients": [True]}),
        ("message", {"recipients": ["2"]}),
        ("message", {"reason": "x" * 601}),
    ],
)
def test_structurally_invalid_decisions_are_excluded(view, kind, changes):
    schema = policy_response_format(view, kind, "hard")["json_schema"]["schema"]
    with pytest.raises(AssertionError):
        assert_matches(decision(kind, **changes), schema)


@pytest.mark.parametrize(
    "value", [-0.01, 1.01, True, "0.5", None, float("nan"), float("inf")]
)
def test_llm_probability_type_and_limits(view, value):
    schema = policy_response_format(view, "action", "llm")["json_schema"]["schema"]
    output = decision(strategy="llm")
    output["role_scores"]["2"]["wolf"] = value
    with pytest.raises(AssertionError):
        assert_matches(output, schema)


@pytest.mark.parametrize(
    "change", ["null", "missing-seat", "extra-seat", "missing-role", "extra-role"]
)
def test_llm_requires_exact_seven_closed_role_rows(view, change):
    schema = policy_response_format(view, "action", "llm")["json_schema"]["schema"]
    output = decision(strategy="llm")
    scores = output["role_scores"]
    if change == "null":
        output["role_scores"] = None
    elif change == "missing-seat":
        del scores["6"]
    elif change == "extra-seat":
        scores["7"] = deepcopy(scores["6"])
    elif change == "missing-role":
        del scores["6"]["seer"]
    else:
        scores["6"]["extra"] = 0
    with pytest.raises(AssertionError):
        assert_matches(output, schema)


def test_string_length_boundaries_are_allowed(view):
    schema = policy_response_format(view, "message", "hard")["json_schema"]["schema"]
    output = decision("message", content="x" * 1024, reason="r" * 600, recipients=[2])
    assert_matches(output, schema)
    assert validate_decision(output, view, "message", "hard") == output


@pytest.mark.parametrize("case", ["words", "private-termination", "row-sum", "facts"])
def test_semantic_checks_stay_in_local_validator(view, case):
    kind, strategy = (
        ("message", "hard")
        if case in ("words", "private-termination")
        else ("action", "llm")
    )
    output = decision(kind, strategy)
    if case == "words":
        output["content"] = " ".join(["word"] * 51)
    elif case == "private-termination":
        output.update(content=" \t", recipients=[2])
    elif case == "row-sum":
        output["role_scores"]["2"] = {"wolf": 0.5, "seer": 0.5, "villager": 0.5}
    else:
        output["role_scores"]["0"] = {"wolf": 1, "seer": 0, "villager": 0}
    schema = policy_response_format(view, kind, strategy)["json_schema"]["schema"]
    assert_matches(output, schema)
    with pytest.raises(ModelError):
        validate_decision(output, view, kind, strategy)


def test_formats_are_fresh_and_names_stable_across_views(view):
    first = policy_response_format(view, "action", "llm")
    expected = deepcopy(first)
    first["json_schema"]["schema"]["properties"]["action_id"]["enum"].append("mutated")
    rows = first["json_schema"]["schema"]["properties"]["role_scores"]["properties"]
    rows["0"]["properties"]["wolf"]["maximum"] = 0.5
    assert rows["1"]["properties"]["wolf"]["maximum"] == 1
    assert policy_response_format(view, "action", "llm") == expected
    other = policy_response_format(replace(view, action_ids=("3",)), "action", "llm")
    assert other["json_schema"]["name"] == expected["json_schema"]["name"]
    assert view.action_ids == ("7", "2", "4")
