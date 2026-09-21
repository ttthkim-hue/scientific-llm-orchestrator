from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Iterable, Sequence

from .pvoca import Action


@dataclass(frozen=True)
class Outcome:
    correct: bool
    tokens: int


@dataclass(frozen=True)
class CascadeExample:
    """Counterfactual action outcomes for one question.

    The v3 controller is trained on *marginal value* targets:
    - Gate-S: does escalating beyond STOP rescue an error?
    - Gate-V: after THINK, does VERIFY rescue more than it harms?
    """

    stop: Outcome
    think: Outcome
    verify: Outcome


def gate_s_target(example: CascadeExample) -> int:
    """1 only when STOP is wrong and a downstream action can rescue it."""
    return int((not example.stop.correct) and (example.think.correct or example.verify.correct))


def gate_v_target(example: CascadeExample) -> int:
    """Three-way value label for VERIFY after THINK.

    +1 = VERIFY rescues THINK
     0 = no correctness change
    -1 = VERIFY destroys a correct THINK answer
    """
    if example.verify.correct and not example.think.correct:
        return 1
    if example.think.correct and not example.verify.correct:
        return -1
    return 0


def choose_stage_action(
    *,
    p_stop_rescue: float,
    p_verify_rescue: float,
    p_verify_harm: float,
    stop_threshold: float,
    verify_margin: float,
) -> Action:
    """Conservative sequential policy.

    STOP is allowed only when predicted rescue value is below a strict
    threshold. VERIFY is allowed only when predicted rescue probability exceeds
    predicted harm probability by the configured safety margin.
    """
    for name, value in {
        "p_stop_rescue": p_stop_rescue,
        "p_verify_rescue": p_verify_rescue,
        "p_verify_harm": p_verify_harm,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if not 0.0 <= stop_threshold <= 1.0:
        raise ValueError("stop_threshold must be in [0, 1]")
    if not 0.0 <= verify_margin <= 1.0:
        raise ValueError("verify_margin must be in [0, 1]")

    if p_stop_rescue <= stop_threshold:
        return Action.STOP
    if p_verify_rescue - p_verify_harm >= verify_margin:
        return Action.VERIFY
    return Action.THINK


def cascade_cost(example: CascadeExample, action: Action) -> int:
    """Cost of the selected observable pipeline.

    STOP pays STOP only. THINK pays THINK. VERIFY pays VERIFY. The benchmark
    action records already contain each action's measured end-to-end token cost.
    """
    return {
        Action.STOP: example.stop.tokens,
        Action.THINK: example.think.tokens,
        Action.VERIFY: example.verify.tokens,
    }[action]


def cascade_correct(example: CascadeExample, action: Action) -> bool:
    return {
        Action.STOP: example.stop.correct,
        Action.THINK: example.think.correct,
        Action.VERIFY: example.verify.correct,
    }[action]


def evaluate_choices(examples: Sequence[CascadeExample], choices: Sequence[Action]) -> dict:
    if len(examples) != len(choices):
        raise ValueError("examples and choices must have equal length")
    if not examples:
        raise ValueError("at least one example is required")
    correct = sum(cascade_correct(ex, action) for ex, action in zip(examples, choices))
    tokens = sum(cascade_cost(ex, action) for ex, action in zip(examples, choices))
    return {
        "items": len(examples),
        "accuracy": correct / len(examples),
        "tokens": tokens,
        "choices": {a.value: sum(action == a for action in choices) for a in Action},
    }


def wilson_interval(successes: int, trials: int, *, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be within [0, trials]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt((p * (1.0 - p) / trials) + (z * z / (4.0 * trials * trials))) / denom
    return max(0.0, center - half), min(1.0, center + half)


@dataclass(frozen=True)
class PromotionEvidence:
    unseen_items: int
    baseline_correct: int
    controller_correct: int
    baseline_tokens: int
    controller_tokens: int
    unsafe_stop_errors: int
    stop_decisions: int
    crashes: int = 0
    owner_gate_bypasses: int = 0
    production_mutations: int = 0


@dataclass(frozen=True)
class PromotionGate:
    min_unseen_items: int = 1000
    max_accuracy_drop_pp: float = 0.5
    min_token_savings: float = 0.15
    max_unsafe_stop_rate: float = 0.01
    confidence: float = 0.95

    def evaluate(self, evidence: PromotionEvidence) -> dict:
        if evidence.unseen_items <= 0:
            raise ValueError("unseen_items must be positive")
        if evidence.baseline_tokens <= 0:
            raise ValueError("baseline_tokens must be positive")
        baseline_accuracy = evidence.baseline_correct / evidence.unseen_items
        controller_accuracy = evidence.controller_correct / evidence.unseen_items
        accuracy_drop_pp = (baseline_accuracy - controller_accuracy) * 100.0
        token_savings = 1.0 - evidence.controller_tokens / evidence.baseline_tokens

        if evidence.stop_decisions:
            _, unsafe_upper = wilson_interval(
                evidence.unsafe_stop_errors,
                evidence.stop_decisions,
                confidence=self.confidence,
            )
        else:
            unsafe_upper = 1.0

        checks = {
            "sample_size": evidence.unseen_items >= self.min_unseen_items,
            "accuracy_noninferiority": accuracy_drop_pp <= self.max_accuracy_drop_pp,
            "token_savings": token_savings >= self.min_token_savings,
            "unsafe_stop_bound": unsafe_upper <= self.max_unsafe_stop_rate,
            "crash_free": evidence.crashes == 0,
            "owner_gate_safe": evidence.owner_gate_bypasses == 0,
            "production_mutation_free": evidence.production_mutations == 0,
        }
        return {
            "status": "PROMOTION_READY" if all(checks.values()) else "SHADOW_ONLY",
            "checks": checks,
            "metrics": {
                "baseline_accuracy": baseline_accuracy,
                "controller_accuracy": controller_accuracy,
                "accuracy_drop_pp": accuracy_drop_pp,
                "token_savings": token_savings,
                "unsafe_stop_rate_upper": unsafe_upper,
            },
        }
