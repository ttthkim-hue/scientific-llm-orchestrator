from __future__ import annotations

from typing import Sequence

from .pvoca import Action


def relative_disagreement(stop_value: float | None, think_value: float | None) -> float:
    if stop_value is None or think_value is None:
        return float("inf")
    scale = max(abs(float(stop_value)), abs(float(think_value)), 1e-12)
    return abs(float(stop_value) - float(think_value)) / scale


def agreement_verify_action(trace: dict, *, threshold: float) -> Action:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    stages = {row["action"]: row for row in trace["stages"]}
    disagreement = relative_disagreement(
        stages["STOP"].get("predicted_value"),
        stages["THINK"].get("predicted_value"),
    )
    return Action.VERIFY if disagreement >= threshold else Action.THINK


def select_agreement_verify_threshold(
    rows: Sequence[dict],
    *,
    candidates: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
) -> float:
    """Tune a simple observable disagreement baseline on training data only.

    Objective: maximize correctness, then minimize cumulative staged tokens.
    This deliberately gives the heuristic a strong train-side advantage while
    keeping the held-out fold untouched.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    if not candidates:
        raise ValueError("candidates must be non-empty")

    best: tuple[int, int, float] | None = None
    best_threshold = 0.0
    for threshold in candidates:
        correct = 0
        tokens = 0
        for row in rows:
            stages = {stage["action"]: stage for stage in row["stages"]}
            action = agreement_verify_action(row, threshold=float(threshold))
            stop_tokens = int(stages["STOP"]["incremental_tokens"])
            think_tokens = int(stages["THINK"]["incremental_tokens"])
            verify_tokens = int(stages["VERIFY"]["incremental_tokens"])
            if action == Action.THINK:
                correct += int(bool(stages["THINK"]["correct"]))
                tokens += stop_tokens + think_tokens
            else:
                correct += int(bool(stages["VERIFY"]["correct"]))
                tokens += stop_tokens + think_tokens + verify_tokens
        key = (correct, -tokens, float(threshold))
        if best is None or key > best:
            best = key
            best_threshold = float(threshold)
    return best_threshold
