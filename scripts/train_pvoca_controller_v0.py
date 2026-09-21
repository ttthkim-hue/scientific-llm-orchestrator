from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.pvoca import Action  # noqa: E402
from scientific_llm_orchestrator.pvoca_controller import (  # noqa: E402
    ControllerExample,
    cross_validate,
    evaluate_policy,
    majority_action,
)


def load_examples(path: Path) -> list[ControllerExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pvoca.oracle.v0":
        raise ValueError("expected pvoca.oracle.v0 result file")
    examples = []
    for record in payload["records"]:
        oracle = record.get("oracle_action")
        if oracle is None:
            continue
        outcomes = {}
        for row in record["results"]:
            action = Action(row["action"])
            outcomes[action] = (bool(row["correct"]), int(row["total_tokens"]))
        if set(outcomes) != {Action.STOP, Action.THINK, Action.VERIFY}:
            continue
        examples.append(
            ControllerExample(
                question=str(record["item"]["question"]),
                oracle_action=Action(oracle),
                outcomes=outcomes,
            )
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the P-VoCA v0 question-only controller.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", type=Path, default=Path("pvoca-controller-v0-summary.json"))
    args = parser.parse_args()

    examples = load_examples(args.results)
    if len(examples) < max(12, args.folds):
        raise SystemExit(f"not enough solved oracle examples: {len(examples)}")

    metrics = cross_validate(examples, folds=args.folds, seed=args.seed)
    fixed = majority_action(examples)
    fixed_accuracy, fixed_tokens = evaluate_policy(examples, [fixed] * len(examples))
    metrics["majority_fixed_action"] = fixed.value
    metrics["majority_fixed_accuracy"] = fixed_accuracy
    metrics["majority_fixed_tokens"] = fixed_tokens
    metrics["schema"] = "pvoca.controller.v0.summary"
    metrics["features"] = "question-only-pre-action"
    metrics["uses_hidden_reasoning"] = False
    metrics["uses_benchmark_difficulty_as_feature"] = False
    metrics["uses_benchmark_domain_as_feature"] = False

    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
