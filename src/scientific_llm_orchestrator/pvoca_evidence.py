from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def aggregate_controller_runs(
    runs: Sequence[dict],
    *,
    matched_baselines: int,
    shadow: dict | None = None,
) -> dict:
    """Build conservative paper/deployment evidence from multiple v3 runs.

    Repeated evaluation of the same benchmark on multiple models does not
    multiply the unique-item count. Performance/safety metrics use the worst
    observed run so one strong model cannot hide a weak one.
    """
    if not runs:
        raise ValueError("at least one controller run is required")
    if matched_baselines < 0:
        raise ValueError("matched_baselines must be non-negative")

    models: set[str] = set()
    task_families: set[str] = set()
    benchmark_items: dict[str, int] = {}
    heldout_groups: set[str] = set()
    heterogeneous_by_benchmark: dict[str, list[int]] = defaultdict(list)

    headrooms: list[float] = []
    calibration: list[float] = []
    accuracy_drop_upper: list[float] = []
    token_savings: list[float] = []
    latency_savings: list[float] = []
    unsafe_stop_upper: list[float] = []
    leakage_detected = False

    for run in runs:
        if run.get("schema") != "pvoca.cascade.controller.v3":
            raise ValueError("expected pvoca.cascade.controller.v3 run")
        model = str(run.get("model_label") or "unknown")
        family = str(run.get("task_family") or "unknown")
        benchmark = str(run.get("benchmark") or "unknown")
        if "unknown" in {model, family, benchmark}:
            raise ValueError("model_label, task_family, and benchmark must be explicit")

        models.add(model)
        task_families.add(family)
        benchmark_items[benchmark] = max(
            benchmark_items.get(benchmark, 0),
            int(run.get("items") or 0),
        )

        for fold in run.get("fold_reports") or []:
            for group in fold.get("held_out_groups") or []:
                heldout_groups.add(f"{benchmark}:{group}")

        heterogeneous_by_benchmark[benchmark].append(
            int(run.get("heterogeneous_action_items") or 0)
        )
        headrooms.append(float((run.get("oracle") or {}).get("headroom_pp_vs_best_fixed", -999.0)))
        calibration.append(float((run.get("calibration") or {}).get("max_ece", 1.0)))

        metrics = run.get("offline_decision_metrics") or {}
        accuracy_drop_upper.append(float(metrics.get("accuracy_drop_upper_pp", 999.0)))
        token_savings.append(float(metrics.get("token_savings", -999.0)))
        latency_savings.append(float(metrics.get("latency_savings", -999.0)))
        unsafe_stop_upper.append(float(metrics.get("unsafe_stop_rate_upper", 1.0)))
        leakage_detected = leakage_detected or bool(run.get("leakage_detected", False))

    paired_transition_items = sum(
        min(values) for values in heterogeneous_by_benchmark.values() if values
    )

    shadow = dict(shadow or {})
    return {
        "schema": "pvoca.research-decision.evidence.v1",
        "unseen_items": sum(benchmark_items.values()),
        "distinct_models": len(models),
        "task_families": len(task_families),
        "heldout_groups": len(heldout_groups),
        "matched_baselines": matched_baselines,
        "reproducible_runs": len(runs),
        "paired_transition_items": paired_transition_items,
        "oracle_headroom_pp": min(headrooms),
        "calibration_ece": max(calibration),
        "leakage_detected": leakage_detected,
        "accuracy_drop_upper_pp": max(accuracy_drop_upper),
        "token_savings": min(token_savings),
        "latency_savings": min(latency_savings),
        "unsafe_stop_rate_upper": max(unsafe_stop_upper),
        "shadow_items": int(shadow.get("shadow_items", 0)),
        "shadow_accuracy_drop_upper_pp": float(
            shadow.get("shadow_accuracy_drop_upper_pp", 999.0)
        ),
        "shadow_token_savings": float(shadow.get("shadow_token_savings", 0.0)),
        "shadow_latency_savings": float(shadow.get("shadow_latency_savings", 0.0)),
        "shadow_unsafe_stop_rate_upper": float(
            shadow.get("shadow_unsafe_stop_rate_upper", 1.0)
        ),
        "shadow_crashes": int(shadow.get("shadow_crashes", 0)),
        "owner_gate_bypasses": int(shadow.get("owner_gate_bypasses", 0)),
        "production_mutations": int(shadow.get("production_mutations", 0)),
        "aggregation": {
            "unique_item_rule": "max items per benchmark; repeated models do not multiply sample size",
            "performance_rule": "worst run across accuracy, savings, calibration, and unsafe-stop metrics",
            "headroom_rule": "minimum oracle headroom across runs",
            "transition_rule": "minimum heterogeneous count per benchmark, summed across benchmarks",
        },
    }
