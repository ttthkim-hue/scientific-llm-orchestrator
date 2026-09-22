from __future__ import annotations

import math
import re
from typing import Iterable


NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_scibench_answer(value) -> float | None:
    text = str(value or "").strip().replace("−", "-").replace("–", "-")
    if not text:
        return None
    # answer_number is intended to be numeric. Fail closed if extra text or
    # multiple numeric candidates remain.
    matches = NUMBER.findall(text.replace(",", ""))
    if len(matches) != 1:
        return None
    try:
        result = float(matches[0])
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def normalize_scibench_rows(rows: Iterable[dict]) -> tuple[list[dict], dict]:
    """Create a solution-free, deduplicated numeric SciBench pool.

    Hugging Face snapshots may concatenate original and solution-bearing JSON
    files. Deduplication is therefore based on the stable scientific content,
    not on row index. Worked solutions are never copied into the output.
    """
    dedup: dict[tuple[str, str, str, str], dict] = {}
    skipped = {
        "missing_question": 0,
        "non_numeric_answer": 0,
        "duplicate": 0,
    }

    for raw in rows:
        question = str(raw.get("problem_text") or "").strip()
        if not question:
            skipped["missing_question"] += 1
            continue
        answer_raw = str(raw.get("answer_number") or "").strip()
        answer = parse_scibench_answer(answer_raw)
        if answer is None:
            skipped["non_numeric_answer"] += 1
            continue

        source = str(raw.get("source") or "unknown").strip() or "unknown"
        problem_id = str(raw.get("problemid") or "").strip()
        key = (source, problem_id, question, answer_raw)
        if key in dedup:
            skipped["duplicate"] += 1
            continue

        dedup[key] = {
            "id": f"scibench:{source}:{problem_id or len(dedup)}",
            "domain": source,
            "difficulty": "unknown",
            "question": question,
            "answer": answer,
            "modality": "text",
            "unit": str(raw.get("unit") or "").strip(),
            "task_family": "college_science_numeric",
            "benchmark": "SciBench",
        }

    normalized = sorted(dedup.values(), key=lambda row: row["id"])
    return normalized, skipped
