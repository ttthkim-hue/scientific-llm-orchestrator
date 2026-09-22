from __future__ import annotations

import math
import random
from typing import Sequence


def exact_mcnemar_p(controller_worse: int, controller_better: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    if controller_worse < 0 or controller_better < 0:
        raise ValueError("discordant counts must be non-negative")
    n = controller_worse + controller_better
    if n == 0:
        return 1.0
    k = min(controller_worse, controller_better)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_savings_interval(
    controller_costs: Sequence[float],
    baseline_costs: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 20260922,
) -> tuple[float, float]:
    """Percentile CI for paired total-cost savings: 1-sum(C)/sum(B)."""
    if len(controller_costs) != len(baseline_costs) or not controller_costs:
        raise ValueError("paired costs must be non-empty and aligned")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if any(float(value) < 0 for value in controller_costs):
        raise ValueError("controller costs must be non-negative")
    if any(float(value) <= 0 for value in baseline_costs):
        raise ValueError("baseline costs must be positive")

    rng = random.Random(seed)
    n = len(controller_costs)
    estimates = []
    for _ in range(resamples):
        c_total = 0.0
        b_total = 0.0
        for _ in range(n):
            index = rng.randrange(n)
            c_total += float(controller_costs[index])
            b_total += float(baseline_costs[index])
        estimates.append(1.0 - c_total / b_total)
    estimates.sort()

    alpha = 1.0 - confidence
    low_index = max(0, min(resamples - 1, int(math.floor((alpha / 2.0) * (resamples - 1)))))
    high_index = max(
        0,
        min(
            resamples - 1,
            int(math.ceil((1.0 - alpha / 2.0) * (resamples - 1))),
        ),
    )
    return estimates[low_index], estimates[high_index]
