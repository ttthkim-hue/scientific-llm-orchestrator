from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.pvoca_split import (  # noqa: E402
    stable_group_folds,
    summarize_group_folds,
    validate_group_holdout,
)

SCIENTIFIC = re.compile(
    r"([+-]?\s*(?:\d+(?:\.\d*)?|\.\d+))\s*(?:\\times|×|x|\*)\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?",
    flags=re.IGNORECASE,
)
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def has_image(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return True


def parse_single_numeric_reference(value) -> float | None:
    text = str(value or "").replace("−", "-").replace("–", "-")

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1).replace(' ', '')}e{match.group(2).replace(' ', '')}"

    text = SCIENTIFIC.sub(repl, text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    values = NUMBER.findall(text)
    if len(values) != 1:
        return None
    try:
        value = float(values[0])
    except ValueError:
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the P-VoCA v3 MatSciBench pool and category-held-out folds. "
            "Reference solutions are intentionally excluded."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", default="JunkaiZ/MatSciBench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--target-traces", type=int, default=1000)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: optional package 'datasets' is required.", file=sys.stderr)
        return 2

    dataset = load_dataset(args.dataset, split=args.split)
    rows = []
    skipped = Counter()
    for row in dataset:
        if str(row.get("type", "")).upper() != "NUM":
            skipped["non_num"] += 1
            continue
        if has_image(row.get("image")):
            skipped["image"] += 1
            continue
        if str(row.get("number_of_answers", "")).lower() != "single":
            skipped["multiple"] += 1
            continue
        answer = parse_single_numeric_reference(row.get("answer"))
        if answer is None:
            skipped["non_scalar_reference"] += 1
            continue
        domain = str(row.get("primary_category") or "unknown")
        difficulty = str(row.get("difficulty_level") or "unknown").lower()
        rows.append(
            {
                "id": str(row["qid"]),
                "domain": domain,
                "difficulty": difficulty,
                "question": str(row["question"]),
                "answer": answer,
                "modality": "text",
                "unit": str(row.get("unit") or ""),
                "source": str(row.get("source") or ""),
            }
        )

    rows.sort(key=lambda row: row["id"])
    groups = [row["domain"] for row in rows]
    assignment = stable_group_folds(groups, folds=args.folds, seed=args.seed)
    validate_group_holdout(groups, assignment, folds=args.folds)

    for row in rows:
        row["holdout_fold"] = assignment[row["domain"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    by_difficulty = Counter(row["difficulty"] for row in rows)
    by_domain = Counter(row["domain"] for row in rows)
    fold_summary = summarize_group_folds(groups, assignment)
    manifest = {
        "schema": "pvoca.matscibench_pool.v3",
        "dataset": args.dataset,
        "split": args.split,
        "eligible": len(rows),
        "target_traces": args.target_traces,
        "target_met": len(rows) >= args.target_traces,
        "folds": args.folds,
        "seed": args.seed,
        "group_key": "primary_category",
        "group_holdout": True,
        "contains_reference_solution": False,
        "contains_reference_answer_for_scoring_only": True,
        "difficulty_counts": dict(sorted(by_difficulty.items())),
        "domain_counts": dict(sorted(by_domain.items())),
        "skipped": dict(sorted(skipped.items())),
        "fold_summary": [
            {"fold": row.fold, "groups": list(row.groups), "item_count": row.item_count}
            for row in fold_summary
        ],
        "next_action": (
            "run staged traces directly"
            if len(rows) >= args.target_traces
            else "add a second public benchmark before training; do not duplicate MatSciBench items"
        ),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
