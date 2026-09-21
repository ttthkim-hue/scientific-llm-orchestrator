from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.pvoca import (  # noqa: E402
    Item,
    OpenAICompatibleLocalProvider,
    choose_oracle,
    records_to_jsonable,
    run_item,
    balanced_pilot_sample,
    summarize,
)


def load_jsonl(path: Path) -> list[Item]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(
            Item(
                item_id=str(row["id"]),
                domain=str(row["domain"]),
                difficulty=str(row["difficulty"]).lower(),
                question=str(row["question"]),
                answer=float(row["answer"]),
                modality=str(row.get("modality", "text")),
            )
        )
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the P-VoCA Oracle v0 STOP/THINK/VERIFY pilot.")
    parser.add_argument("--dataset", required=True, type=Path, help="Normalized public benchmark JSONL.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000", help="Local OpenAI-compatible base URL.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=Path, default=Path("pvoca-oracle-v0-results.json"))
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--per-difficulty", type=int, default=36)
    args = parser.parse_args()

    sampled = balanced_pilot_sample(load_jsonl(args.dataset), per_difficulty=args.per_difficulty, seed=args.seed)
    provider = OpenAICompatibleLocalProvider(endpoint=args.endpoint, model=args.model)
    records = []
    for index, item in enumerate(sampled, start=1):
        results = run_item(provider, item)
        records.append({"item": item, "results": results, "oracle_action": choose_oracle(results)})
        print(f"[{index}/{len(sampled)}] {item.item_id} oracle={records[-1]['oracle_action']}")

    payload = {
        "schema": "pvoca.oracle.v0",
        "model": args.model,
        "seed": args.seed,
        "per_difficulty": args.per_difficulty,
        "summary": summarize(records),
        "records": records_to_jsonable(records),
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
