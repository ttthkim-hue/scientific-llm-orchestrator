from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.pvoca import Action  # noqa: E402
from scientific_llm_orchestrator.pvoca_cascade import (  # noqa: E402
    CascadeExample,
    Outcome,
    PromotionEvidence,
    PromotionGate,
    choose_stage_action,
    evaluate_choices,
)
from scientific_llm_orchestrator.pvoca_gate import (  # noqa: E402
    BinaryLogit,
    expected_calibration_error,
    gate_s_features,
    gate_v_features,
    select_safe_stop_threshold,
    select_verify_margin,
)
from scientific_llm_orchestrator.pvoca_split import stable_group_folds  # noqa: E402


def load_traces(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != "pvoca.cascade.trace.v3":
            raise ValueError("expected pvoca.cascade.trace.v3 rows")
        if row.get("cost_semantics") != "incremental-stage-cost":
            raise ValueError("trace does not use incremental-stage-cost")
        rows.append(row)
    if not rows:
        raise ValueError("trace file is empty")
    return rows


def to_example(trace: dict) -> CascadeExample:
    stages = {row["action"]: row for row in trace["stages"]}
    return CascadeExample(
        stop=Outcome(
            bool(stages["STOP"]["correct"]),
            int(stages["STOP"]["incremental_tokens"]),
            float(stages["STOP"].get("incremental_latency_ms") or 0.0),
        ),
        think=Outcome(
            bool(stages["THINK"]["correct"]),
            int(stages["THINK"]["incremental_tokens"]),
            float(stages["THINK"].get("incremental_latency_ms") or 0.0),
        ),
        verify=Outcome(
            bool(stages["VERIFY"]["correct"]),
            int(stages["VERIFY"]["incremental_tokens"]),
            float(stages["VERIFY"].get("incremental_latency_ms") or 0.0),
        ),
    )


def action_correct(trace: dict, action: Action) -> bool:
    stages = {row["action"]: row for row in trace["stages"]}
    return bool(stages[action.value]["correct"])


def fit_binary(rows: list[dict], feature_fn, label_key: str, *, seed: int) -> BinaryLogit:
    features = [feature_fn(row) for row in rows]
    labels = [int(row["labels"][label_key]) for row in rows]
    model = BinaryLogit.create(len(features[0]))
    model.fit(features, labels, seed=seed)
    return model


def baseline_from_train(rows: list[dict]) -> Action:
    candidates = (Action.THINK, Action.VERIFY)
    return max(
        candidates,
        key=lambda action: (
            sum(action_correct(row, action) for row in rows) / len(rows),
            -sum(
                int(next(stage for stage in row["stages"] if stage["action"] == action.value)["incremental_tokens"])
                for row in rows
            ),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train/evaluate conservative P-VoCA v3 gates with category-held-out folds."
    )
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--min-stop-decisions", type=int, default=100)
    parser.add_argument("--max-unsafe-stop-rate", type=float, default=0.01)
    parser.add_argument("--promotion-min-items", type=int, default=1000)
    parser.add_argument("--model-label", default="unknown")
    parser.add_argument("--task-family", default="unknown")
    parser.add_argument("--benchmark", default="unknown")
    args = parser.parse_args()

    rows = load_traces(args.traces)
    groups = [str(row["item"].get("domain") or "unknown") for row in rows]
    assignment = stable_group_folds(groups, folds=args.folds, seed=args.seed)

    fold_reports = []
    all_controller_choices: list[Action] = []
    all_baseline_choices: list[Action] = []
    all_examples: list[CascadeExample] = []
    unsafe_stop_errors = 0
    stop_decisions = 0
    test_stop_probs: list[float] = []
    test_stop_labels: list[int] = []
    test_verify_rescue_probs: list[float] = []
    test_verify_rescue_labels: list[int] = []
    test_verify_harm_probs: list[float] = []
    test_verify_harm_labels: list[int] = []

    for fold in range(args.folds):
        train = [row for row in rows if assignment[str(row["item"].get("domain") or "unknown")] != fold]
        test = [row for row in rows if assignment[str(row["item"].get("domain") or "unknown")] == fold]
        if not train or not test:
            raise ValueError(f"empty train/test partition for fold={fold}")

        gate_s = fit_binary(train, gate_s_features, "gate_s_rescue", seed=args.seed + fold * 17)
        gate_v_rescue = fit_binary(train, gate_v_features, "gate_v_rescue", seed=args.seed + fold * 17 + 1)
        gate_v_harm = fit_binary(train, gate_v_features, "gate_v_harm", seed=args.seed + fold * 17 + 2)

        train_stop_probs = [gate_s.predict_proba(gate_s_features(row)) for row in train]
        train_stop_labels = [int(row["labels"]["gate_s_rescue"]) for row in train]
        stop_threshold = select_safe_stop_threshold(
            train_stop_probs,
            train_stop_labels,
            max_unsafe_stop_rate=args.max_unsafe_stop_rate,
            min_stop_decisions=args.min_stop_decisions,
        )

        train_v_rescue = [gate_v_rescue.predict_proba(gate_v_features(row)) for row in train]
        train_v_harm = [gate_v_harm.predict_proba(gate_v_features(row)) for row in train]
        verify_margin = select_verify_margin(
            train_v_rescue,
            train_v_harm,
            [int(row["labels"]["gate_v_rescue"]) for row in train],
            [int(row["labels"]["gate_v_harm"]) for row in train],
            [
                int(next(stage for stage in row["stages"] if stage["action"] == "VERIFY")["incremental_tokens"])
                for row in train
            ],
        )

        choices = []
        for row in test:
            p_stop_rescue = gate_s.predict_proba(gate_s_features(row))
            p_verify_rescue = gate_v_rescue.predict_proba(gate_v_features(row))
            p_verify_harm = gate_v_harm.predict_proba(gate_v_features(row))
            test_stop_probs.append(p_stop_rescue)
            test_stop_labels.append(int(row["labels"]["gate_s_rescue"]))
            test_verify_rescue_probs.append(p_verify_rescue)
            test_verify_rescue_labels.append(int(row["labels"]["gate_v_rescue"]))
            test_verify_harm_probs.append(p_verify_harm)
            test_verify_harm_labels.append(int(row["labels"]["gate_v_harm"]))

            action = choose_stage_action(
                p_stop_rescue=p_stop_rescue,
                p_verify_rescue=p_verify_rescue,
                p_verify_harm=p_verify_harm,
                stop_threshold=stop_threshold,
                verify_margin=verify_margin,
            )
            choices.append(action)
            if action == Action.STOP:
                stop_decisions += 1
                unsafe_stop_errors += int(row["labels"]["gate_s_rescue"])

        examples = [to_example(row) for row in test]
        baseline_action = baseline_from_train(train)
        baseline_choices = [baseline_action] * len(test)
        controller_metrics = evaluate_choices(examples, choices)
        baseline_metrics = evaluate_choices(examples, baseline_choices)
        think_metrics = evaluate_choices(examples, [Action.THINK] * len(test))
        verify_metrics = evaluate_choices(examples, [Action.VERIFY] * len(test))

        fold_reports.append(
            {
                "fold": fold,
                "train_items": len(train),
                "test_items": len(test),
                "held_out_groups": sorted(
                    group for group, assigned in assignment.items() if assigned == fold
                ),
                "stop_threshold": stop_threshold,
                "verify_margin": verify_margin,
                "baseline_action": baseline_action.value,
                "controller": controller_metrics,
                "selected_baseline": baseline_metrics,
                "always_think": think_metrics,
                "always_verify": verify_metrics,
            }
        )

        all_examples.extend(examples)
        all_controller_choices.extend(choices)
        all_baseline_choices.extend(baseline_choices)

    controller = evaluate_choices(all_examples, all_controller_choices)
    baseline = evaluate_choices(all_examples, all_baseline_choices)
    always_stop = evaluate_choices(all_examples, [Action.STOP] * len(all_examples))
    always_think = evaluate_choices(all_examples, [Action.THINK] * len(all_examples))
    always_verify = evaluate_choices(all_examples, [Action.VERIFY] * len(all_examples))

    fixed = {
        Action.STOP: always_stop,
        Action.THINK: always_think,
        Action.VERIFY: always_verify,
    }
    best_fixed_action = max(
        fixed,
        key=lambda action: (
            fixed[action]["accuracy"],
            -fixed[action]["tokens"],
            -fixed[action]["latency_ms"],
        ),
    )
    best_fixed_accuracy = fixed[best_fixed_action]["accuracy"]

    oracle_correct = 0
    oracle_tokens = 0
    oracle_latency_ms = 0.0
    correctness_patterns: dict[str, int] = {}
    heterogeneous_action_items = 0
    for example in all_examples:
        outcomes = {
            Action.STOP: (example.stop.correct, example.stop.incremental_tokens, example.stop.incremental_latency_ms),
            Action.THINK: (
                example.think.correct,
                example.stop.incremental_tokens + example.think.incremental_tokens,
                example.stop.incremental_latency_ms + example.think.incremental_latency_ms,
            ),
            Action.VERIFY: (
                example.verify.correct,
                example.stop.incremental_tokens + example.think.incremental_tokens + example.verify.incremental_tokens,
                example.stop.incremental_latency_ms + example.think.incremental_latency_ms + example.verify.incremental_latency_ms,
            ),
        }
        bits = "".join("1" if outcomes[action][0] else "0" for action in (Action.STOP, Action.THINK, Action.VERIFY))
        correctness_patterns[bits] = correctness_patterns.get(bits, 0) + 1
        heterogeneous_action_items += int(bits not in {"000", "111"})
        correct_actions = [action for action in outcomes if outcomes[action][0]]
        if correct_actions:
            oracle_correct += 1
            chosen = min(correct_actions, key=lambda action: (outcomes[action][1], outcomes[action][2]))
        else:
            chosen = min(outcomes, key=lambda action: (outcomes[action][1], outcomes[action][2]))
        oracle_tokens += int(outcomes[chosen][1])
        oracle_latency_ms += float(outcomes[chosen][2])

    oracle_accuracy = oracle_correct / len(all_examples)
    oracle_headroom_pp = (oracle_accuracy - best_fixed_accuracy) * 100.0

    worse = better = 0
    for example, controller_action, baseline_action in zip(
        all_examples, all_controller_choices, all_baseline_choices
    ):
        controller_ok = {
            Action.STOP: example.stop.correct,
            Action.THINK: example.think.correct,
            Action.VERIFY: example.verify.correct,
        }[controller_action]
        baseline_ok = {
            Action.STOP: example.stop.correct,
            Action.THINK: example.think.correct,
            Action.VERIFY: example.verify.correct,
        }[baseline_action]
        worse += int(baseline_ok and not controller_ok)
        better += int(controller_ok and not baseline_ok)

    calibration = {
        "gate_s_ece": expected_calibration_error(test_stop_probs, test_stop_labels),
        "gate_v_rescue_ece": expected_calibration_error(
            test_verify_rescue_probs, test_verify_rescue_labels
        ),
        "gate_v_harm_ece": expected_calibration_error(
            test_verify_harm_probs, test_verify_harm_labels
        ),
    }
    calibration["max_ece"] = max(calibration.values())

    latency_savings_vs_selected_baseline = (
        1.0 - controller["latency_ms"] / baseline["latency_ms"]
        if baseline["latency_ms"] > 0
        else 0.0
    )

    promotion = PromotionGate(
        min_unseen_items=args.promotion_min_items,
        max_accuracy_drop_pp=0.5,
        min_token_savings=0.15,
        max_unsafe_stop_rate=args.max_unsafe_stop_rate,
    ).evaluate(
        PromotionEvidence(
            unseen_items=len(all_examples),
            baseline_correct=round(baseline["accuracy"] * len(all_examples)),
            controller_correct=round(controller["accuracy"] * len(all_examples)),
            baseline_tokens=int(baseline["tokens"]),
            controller_tokens=int(controller["tokens"]),
            unsafe_stop_errors=unsafe_stop_errors,
            stop_decisions=stop_decisions,
            controller_worse_items=worse,
            controller_better_items=better,
            crashes=0,
            owner_gate_bypasses=0,
            production_mutations=0,
        )
    )

    out = {
        "schema": "pvoca.cascade.controller.v3",
        "status": "PASS",
        "items": len(rows),
        "folds": args.folds,
        "split": "primary_category_group_holdout",
        "feature_boundary": "question + outputs observable up to current stage only",
        "uses_hidden_reasoning": False,
        "uses_reference_answer_as_feature": False,
        "model_label": args.model_label,
        "task_family": args.task_family,
        "benchmark": args.benchmark,
        "heldout_group_count": len(set(groups)),
        "controller": controller,
        "selected_baseline": baseline,
        "always_stop": always_stop,
        "always_think": always_think,
        "always_verify": always_verify,
        "best_fixed_action": best_fixed_action.value,
        "best_fixed_accuracy": best_fixed_accuracy,
        "oracle": {
            "accuracy": oracle_accuracy,
            "tokens": oracle_tokens,
            "latency_ms": oracle_latency_ms,
            "headroom_pp_vs_best_fixed": oracle_headroom_pp,
        },
        "correctness_patterns": dict(sorted(correctness_patterns.items())),
        "heterogeneous_action_items": heterogeneous_action_items,
        "unsafe_stop_errors": unsafe_stop_errors,
        "stop_decisions": stop_decisions,
        "paired_controller_worse_items": worse,
        "paired_controller_better_items": better,
        "calibration": calibration,
        "latency_savings_vs_selected_baseline": latency_savings_vs_selected_baseline,
        "offline_decision_metrics": {
            "accuracy_drop_upper_pp": promotion["metrics"]["accuracy_drop_upper_pp"],
            "token_savings": promotion["metrics"]["token_savings"],
            "latency_savings": latency_savings_vs_selected_baseline,
            "unsafe_stop_rate_upper": promotion["metrics"]["unsafe_stop_rate_upper"],
            "calibration_ece": calibration["max_ece"],
        },
        "promotion": promotion,
        "fold_reports": fold_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
