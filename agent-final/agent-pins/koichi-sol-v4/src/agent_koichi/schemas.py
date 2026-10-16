"""Strict policy output shapes; semantic checks remain in validate_decision."""

from .belief import ROLES, SeatView
from .model import ModelError


def policy_response_format(view: SeatView, kind: str, strategy: str) -> dict:
    """Build a fresh OpenRouter response_format for this admitted decision."""
    if kind not in ("action", "message"):
        raise ModelError("Unknown decision kind.")
    if kind == "action" and not view.action_ids:
        raise ModelError("No legal action is available for the policy schema.")

    recipients = {"type": "array", "items": {"type": "integer"}, "maxItems": 0}
    if kind == "action":
        action_id = {"type": "string", "enum": list(view.action_ids)}
        content = {"type": "string", "enum": [""]}
    else:
        action_id = {"type": "null"}
        content = {"type": "string", "maxLength": 1024}
        eligible = [seat for seat in view.alive if seat != view.facts.self_id]
        if eligible:
            recipients["items"]["enum"] = eligible
            recipients["maxItems"] = 1

    role_scores = {"type": "null"}
    if strategy == "llm":
        role_scores = {
            "type": "object",
            "properties": {
                str(seat): {
                    "type": "object",
                    "properties": {
                        role: {"type": "number", "minimum": 0, "maximum": 1}
                        for role in ROLES
                    },
                    "required": list(ROLES),
                    "additionalProperties": False,
                }
                for seat in range(7)
            },
            "required": [str(seat) for seat in range(7)],
            "additionalProperties": False,
        }

    properties = {
        "action_id": action_id,
        "content": content,
        "recipients": recipients,
        "reason": {"type": "string", "maxLength": 600},
        "role_scores": role_scores,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"werewolf_policy_{kind}_{'llm' if strategy == 'llm' else 'math'}",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }
