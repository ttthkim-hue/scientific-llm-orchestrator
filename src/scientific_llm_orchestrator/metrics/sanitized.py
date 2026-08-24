"""Metric container that does not store prompts, responses, or telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SanitizedMetrics:
    evidence_label: str
    route_correct: Optional[bool] = None
    schema_pass: Optional[bool] = None
    protected_literal_pass: Optional[bool] = None
    deterministic_functional_pass: Optional[bool] = None
    unintended_diff_count: Optional[int] = None
    final_reviewer_decision: str = "not-run"
    frontier_input_tokens: Optional[int] = None
    frontier_output_tokens: Optional[int] = None
    hosted_worker_input_tokens: Optional[int] = None
    hosted_worker_output_tokens: Optional[int] = None
    hosted_token_displacement: Optional[int] = None
    local_input_tokens: Optional[int] = None
    local_output_tokens: Optional[int] = None
    time_to_first_token_ms: Optional[float] = None
    generation_tokens_per_second: Optional[float] = None
    tasks_per_hour: Optional[float] = None
    retries: Optional[int] = None
    repair_passes: Optional[int] = None
    vram_mib: Optional[int] = None
    peak_temperature_c: Optional[float] = None
    average_temperature_c: Optional[float] = None
    power_w: Optional[float] = None
    wh_per_task: Optional[float] = None
    fallback_rate: Optional[float] = None
    blocked_task_rate: Optional[float] = None
    privacy_class_violation_count: Optional[int] = None
    context_overflow_rate: Optional[float] = None
    provider_failure_rate: Optional[float] = None
    end_to_end_latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def dry_run(cls, route_correct: Optional[bool] = None) -> "SanitizedMetrics":
        return cls(
            evidence_label="observed-mock-runtime",
            route_correct=route_correct,
            final_reviewer_decision="not-run",
        )
