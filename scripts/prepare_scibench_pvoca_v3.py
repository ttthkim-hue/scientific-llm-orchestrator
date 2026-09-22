from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.pvoca_scibench import normalize_scibench_rows  # noqa: E402
from scientific_llm_orchestrator.pvoca_split import (  # noqa: E402
    stable_group_folds,
    summarize_group_folds,
    validate_group_holdout,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a solution-free, deduplicated SciBench pool for P-VoCA v3 "
            "and assign whole textbook sources to held-out folds."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", default="xw27/scibench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: optional package 'datasets' is required.", file=sys.stderr)
        return 2

    dataset = load_dataset(args.dataset, split=args.split)
    normalized, skipped = normalize_scibench_rows(dataset)
    if not normalized:
        raise SystemExit("no usable SciBench rows")

    groups = [row["domain"] for row in normalized]
    assignment = stable_group_folds(groups, folds=args.folds, seed=args.seed)
    validate_group_holdout(groups, assignment, folds=args.folds)
    for row in normalized:
        row["holdout_fold"] = assignment[row["domain"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    by_source = Counter(row["domain"] for row in normalized)
    fold_summary = summarize_group_folds(groups, assignment)
    manifest = {
        "schema": "pvoca.scibench_pool.v3",
        "dataset": args.dataset,
        "split": args.split,
        "raw_rows": len(dataset),
        "eligible_after_dedup": len(normalized),
        "skipped": skipped,
        "source_counts": dict(sorted(by_source.items())),
        "folds": args.folds,
        "seed": args.seed,
        "group_key": "source",
        "group_holdout": True,
        "contains_reference_solution": False,
        "contains_reference_answer_for_scoring_only": True,
        "task_family": "college_science_numeric",
        "benchmark": "SciBench",
        "contamination_risk": (
            "Public textbook problems and answers may overlap model pretraining; "
            "use as a secondary generalization axis, not sole evidence."
        ),
        "fold_summary": [
            {"fold": row.fold, "groups": list(row.groups), "item_count": row.item_count}
            for row in fold_summary
        ],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
