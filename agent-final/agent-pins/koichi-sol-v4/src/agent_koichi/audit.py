"""Koichi-specific corrections for finalized model-integration diagnostics."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from werewolf_runtime import audit as audit_core


def correct_diagnostic_mode(path: Path, workspace: Path) -> dict:
    """Append diagnostic classification/input repairs, preserving original evidence."""
    workspace = workspace.resolve(strict=True)
    path = path.resolve(strict=True)
    if not path.is_relative_to((workspace / "data" / "logs").resolve(strict=True)):
        raise ValueError("Corrections accept only this workspace's finalized logs.")
    redactor = audit_core.Redactor([])
    with audit_core._locked(workspace), audit_core._open(path, "r") as stream:
        original = json.load(stream)
        if original.get("kind") != "online-integration" or original.get(
            "status"
        ) not in ("completed", "failed", "aborted"):
            raise ValueError(
                "This correction is only for finalized model-integration diagnostics."
            )
        repaired = deepcopy(original)
        reasons = []
        if original.get("test_mode") is not None:
            reasons.append(
                "Online model integration used synthetic inputs and was not a Playground match; test_mode is not applicable."
            )
        actual_input = None
        for event in repaired.get("events", []):
            if event.get("event") == "model_request":
                messages = event["data"].get("payload", {}).get("messages", [])
                try:
                    candidate = json.loads(messages[-1]["content"])
                except IndexError, KeyError, TypeError, ValueError:
                    continue
                if isinstance(candidate, dict) and set(candidate) == {
                    "day",
                    "messages",
                    "previous_claims",
                }:
                    actual_input = candidate
            elif (
                event.get("event") == "extraction"
                and actual_input is not None
                and event["data"].get("input") != actual_input
            ):
                event["data"]["input"] = deepcopy(actual_input)
                reasons.append(
                    "Extraction input was reconstructed from its immutable model_request; a mutable previous_claims reference had incorrectly included subsequently accepted claims."
                )
                actual_input = None
        if not reasons:
            raise ValueError(
                "No diagnostic classification or input correction is needed."
            )
        reason = " ".join(dict.fromkeys(reasons))
        revision = original.get("revision", 1) + 1
        timestamp = datetime.fromisoformat(original["timestamp"])
        name = f"{timestamp:%Y%m%dT%H%M%SZ}_{original['attempt_id']}_revision{revision}_{uuid4()}.json"
        relative = (Path("data") / "logs" / name).as_posix()
        corrected = {
            **repaired,
            "test_mode": None,
            "revision": revision,
            "correction_reason": reason,
            "source_log_path": path.relative_to(workspace).as_posix(),
            "log_path": relative,
            "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        with audit_core._open(workspace / relative, "x") as output:
            audit_core._write(output, redactor.clean(corrected))
        with audit_core._open(
            workspace / "data" / "experiments.jsonl", "r"
        ) as summaries:
            rows = [json.loads(line) for line in summaries if line.strip()]
        matches = [
            row for row in rows if row.get("experiment_id") == original["experiment_id"]
        ]
        latest = max(matches, key=lambda row: row.get("revision", 1))
        summary = {
            **latest,
            "revision": latest.get("revision", 1) + 1,
            "correction_reason": reason,
            "findings": {**latest["findings"], "log_path": relative, "test_mode": None},
        }
    audit_core.append_experiment(workspace, summary, redactor)
    return {
        "corrected_log_path": relative,
        "original_preserved": True,
        "tested_version": original["version"],
        "experiment_revision": summary["revision"],
    }
