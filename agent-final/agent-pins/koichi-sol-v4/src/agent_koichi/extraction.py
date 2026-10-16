"""Owner-harvested extraction work with bounded, in-memory audit events."""

import asyncio
import json
import math
from collections import deque
from copy import deepcopy
from datetime import UTC, datetime
from uuid import uuid4

from werewolf_runtime.audit import Redactor
from werewolf_runtime.protocol import MAX_LINE_BYTES

from .config import ConfigurationError
from .model import ModelError
from .prompts import EXTRACTION

_USAGE_FIELDS = {
    "input_tokens": (),
    "output_tokens": (),
    "input_tokens_details": ("cached_tokens",),
    "output_tokens_details": ("reasoning_tokens",),
    "prompt_tokens": (),
    "completion_tokens": (),
    "total_tokens": (),
    "cost": (),
    "prompt_tokens_details": (
        "cached_tokens",
        "cache_write_tokens",
        "audio_tokens",
        "video_tokens",
    ),
    "completion_tokens_details": (
        "reasoning_tokens",
        "image_tokens",
        "audio_tokens",
        "accepted_prediction_tokens",
        "rejected_prediction_tokens",
    ),
}


class ExtractionAudit:
    """Buffer sanitized protocol envelopes; only the owner calls flush."""

    def __init__(self, job_id, model_key):
        self.redactor = Redactor([model_key])
        self._job_id = job_id
        self._records = deque()
        self._bytes = 0

    def record(self, event, data):
        if event == "model_usage":

            def scalar(value):
                return (
                    value is None
                    or (type(value) is int and value.bit_length() <= 64)
                    or (type(value) is float and math.isfinite(value))
                )

            usage = {}
            raw = data.get("usage")
            for key, details in _USAGE_FIELDS.items():
                if not isinstance(raw, dict) or key not in raw:
                    continue
                value = raw[key]
                if details and isinstance(value, dict):
                    usage[key] = {
                        name: value[name]
                        for name in details
                        if name in value and scalar(value[name])
                    }
                elif (not details or value is None) and scalar(value):
                    usage[key] = value
            data = {**data, "usage": usage}
        envelope = self.redactor.clean(
            {
                "type": "event",
                "event": event,
                "data": {
                    **data,
                    "extraction_job_id": self._job_id,
                    "occurred_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                },
            }
        )
        size = len(
            (json.dumps(envelope, ensure_ascii=True, allow_nan=False) + "\n").encode(
                "utf-8"
            )
        )
        # Requests cannot consume the space needed to retain returned billing.
        reserve = min(64 * 1024, MAX_LINE_BYTES // 4)
        limit = MAX_LINE_BYTES
        if event not in ("model_usage", "model_error", "model_cancelled"):
            limit -= reserve
        if self._bytes + size > limit:
            raise ModelError("Extraction audit exceeds the size limit.")
        self._records.append((envelope, size))
        self._bytes += size

    def flush(self, sink):
        while self._records:
            envelope, size = self._records[0]
            sink.record(envelope["event"], envelope["data"])
            self._records.popleft()
            self._bytes -= size


class ExtractionJob:
    """One persistent task, without ledger access, successor jobs, or output."""

    def __init__(
        self,
        *,
        model,
        config,
        batch,
        extraction_input,
        identity,
        ledger_revision,
        deadline,
        clock,
    ):
        self.batch = deepcopy(batch)
        self.extraction_input = deepcopy(extraction_input)
        self.identity = identity
        self.ledger_revision = ledger_revision
        self.deadline = deadline
        self.job_id = str(uuid4())
        self.discard_reason = None
        self.audit = ExtractionAudit(self.job_id, config.model_key)
        prompts = [
            {"role": "system", "content": EXTRACTION},
            {
                "role": "user",
                "content": json.dumps(self.extraction_input, ensure_ascii=True),
            },
        ]
        self.task = asyncio.create_task(self._run(model, prompts, clock))
        self.task.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )

    async def _run(self, model, prompts, clock):
        try:
            remaining = self.deadline - clock()
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                response = await model.complete(
                    prompts, "low", remaining, audit=self.audit
                )
                completed_at = clock()
                if not completed_at < self.deadline:
                    raise TimeoutError
                if not isinstance(response, dict):
                    raise ModelError("Extraction response is not a JSON object.")
            return {"response": response, "error": None, "completed_at": completed_at}
        except TimeoutError:
            error = "Extraction exceeded its deadline."
        except (ModelError, ConfigurationError) as failure:
            error = self.audit.redactor.clean(str(failure))
        except Exception as failure:
            error = self.audit.redactor.clean(type(failure).__name__)
        return {"response": None, "error": error, "completed_at": clock()}

    def cancel(self, reason):
        if self.discard_reason is not None:
            return
        self.discard_reason = self.audit.redactor.clean(reason)
        if not self.task.done():
            self.task.cancel()
