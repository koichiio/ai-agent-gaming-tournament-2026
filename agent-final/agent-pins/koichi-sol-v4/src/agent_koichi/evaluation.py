"""Proper probability scoring with explicit unknown labels and label coverage."""

import math


def role_metrics(marginals: dict, labels: dict[int, str], eligible: list[int]) -> dict:
    evaluated = [
        seat for seat in eligible if labels.get(seat) in ("wolf", "seer", "villager")
    ]
    if len(set(eligible)) != len(eligible):
        raise ValueError("Eligible seats must not repeat.")
    for seat in evaluated:
        row = marginals.get(str(seat))
        if not isinstance(row, dict) or set(row) != {"wolf", "seer", "villager"}:
            raise ValueError("A labeled target is missing its probability row.")
        if any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in row.values()
        ) or not math.isclose(math.fsum(row.values()), 1, abs_tol=1e-6):
            raise ValueError("Invalid role probabilities cannot be scored.")
    count = len(evaluated)
    zero = sum(marginals[str(seat)][labels[seat]] == 0 for seat in evaluated)
    brier = (
        math.fsum(
            (p - (role == labels[seat])) ** 2
            for seat in evaluated
            for role, p in marginals[str(seat)].items()
        )
        / count
        if count
        else None
    )
    log_loss = (
        -math.fsum(math.log(marginals[str(seat)][labels[seat]]) for seat in evaluated)
        / count
        if count and not zero
        else None
    )
    return {
        "eligible_targets": len(eligible),
        "labeled_targets": count,
        "label_coverage": count / len(eligible) if eligible else None,
        "brier": brier,
        "log_loss": log_loss,
        "infinite_log_loss": bool(zero),
        "zero_probability_true_roles": zero,
    }


def binary_metrics(
    marginals: dict, labels: dict[int, bool], eligible: list[int], role: str
) -> dict:
    if role not in ("wolf", "seer") or len(set(eligible)) != len(eligible):
        raise ValueError("Invalid binary scoring role or repeated eligible seat.")
    evaluated = [seat for seat in eligible if type(labels.get(seat)) is bool]
    probabilities = {seat: marginals[str(seat)][role] for seat in evaluated}
    if any(
        type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
        for p in probabilities.values()
    ):
        raise ValueError("Invalid wolf probabilities cannot be scored.")
    count = len(evaluated)
    zero = sum((p if labels[seat] else 1 - p) == 0 for seat, p in probabilities.items())
    return {
        "eligible_targets": len(eligible),
        "labeled_targets": count,
        "label_coverage": count / len(eligible) if eligible else None,
        "brier": math.fsum((p - labels[seat]) ** 2 for seat, p in probabilities.items())
        / count
        if count
        else None,
        "log_loss": -math.fsum(
            math.log(p if labels[seat] else 1 - p) for seat, p in probabilities.items()
        )
        / count
        if count and not zero
        else None,
        "infinite_log_loss": bool(zero),
    }
