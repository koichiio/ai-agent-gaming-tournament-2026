"""Public provider failures contain machine codes, never provider payloads."""

import math
import re


def _nonnegative_float(value):
    if type(value) not in (int, float):
        return None
    try:
        value = float(value)
    except ValueError, OverflowError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def normalize_usage(raw) -> tuple[dict, float | None]:
    """Keep only numeric token counters and an explicitly reported finite cost."""
    if not isinstance(raw, dict):
        return {}, None
    usage = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        value = raw.get(key)
        if type(value) is int and value >= 0:
            usage[key] = value
    for key in (
        "input_tokens_details",
        "output_tokens_details",
        "prompt_tokens_details",
        "completion_tokens_details",
    ):
        details = raw.get(key)
        if isinstance(details, dict):
            usage[key] = {
                k: v
                for k, v in details.items()
                if k
                in (
                    "cached_tokens",
                    "reasoning_tokens",
                    "audio_tokens",
                    "accepted_prediction_tokens",
                    "rejected_prediction_tokens",
                )
                and type(v) is int
                and v >= 0
            }
    return usage, _nonnegative_float(raw.get("cost"))


class ProviderError(Exception):
    def __init__(
        self,
        code: str,
        status: int | None = None,
        retryable: bool = False,
        *,
        retry_after_seconds: float | None = None,
        usage: dict | None = None,
        cost: float | None = None,
    ) -> None:
        self.code = (
            code
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", code)
            else "provider_error"
        )
        self.status = status if type(status) is int and 100 <= status <= 599 else None
        self.retryable = bool(retryable)
        self.retry_after_seconds = _nonnegative_float(retry_after_seconds)
        self.usage = normalize_usage(usage)[0] if isinstance(usage, dict) else None
        self.cost = _nonnegative_float(cost)
        super().__init__(self.code)
