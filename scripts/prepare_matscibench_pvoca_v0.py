from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

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
        return float(values[0])
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the clean text-only, single-numeric MatSciBench pool for P-VoCA Oracle v0."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="JunkaiZ/MatSciBench")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: optional package 'datasets' is required for this data-preparation script.", file=sys.stderr)
        return 2

    dataset = load_dataset(args.dataset, split=args.split)
    rows = []
    skipped = {"non_num": 0, "image": 0, "multiple": 0, "non_scalar_reference": 0}
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
        rows.append(
            {
                "id": str(row["qid"]),
                "domain": str(row.get("primary_category") or "unknown"),
                "difficulty": str(row["difficulty_level"]).lower(),
                "question": str(row["question"]),
                "answer": answer,
                "modality": "text",
                "unit": str(row.get("unit") or ""),
            }
        )

    rows.sort(key=lambda x: x["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "schema": "pvoca.matscibench_pool.v0",
                "dataset": args.dataset,
                "split": args.split,
                "eligible": len(rows),
                "skipped": skipped,
                "contains_reference_solution": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
