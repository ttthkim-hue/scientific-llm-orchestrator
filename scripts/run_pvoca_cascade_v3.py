from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.pvoca import Item, OllamaLocalProvider  # noqa: E402
from scientific_llm_orchestrator.pvoca_trace import run_cascade_trace, trace_to_jsonable  # noqa: E402


def load_items(path: Path) -> list[Item]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(
            Item(
                item_id=str(row["id"]),
                domain=str(row.get("domain") or "unknown"),
                difficulty=str(row.get("difficulty") or "unknown"),
                question=str(row["question"]),
                answer=float(row["answer"]),
                modality=str(row.get("modality", "text")),
            )
        )
    return items


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids.add(str(row["item"]["item_id"]))
    return ids


def summarize(path: Path) -> dict:
    actions = ("STOP", "THINK", "VERIFY")
    correct = {a: 0 for a in actions}
    incremental = {a: 0 for a in actions}
    patterns: dict[str, int] = {}
    gate_s = gate_v_rescue = gate_v_harm = 0
    count = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stages = {stage["action"]: stage for stage in row["stages"]}
        if set(stages) != set(actions):
            continue
        count += 1
        bits = ""
        for action in actions:
            ok = bool(stages[action]["correct"])
            correct[action] += int(ok)
            incremental[action] += int(stages[action]["incremental_tokens"])
            bits += "1" if ok else "0"
        patterns[bits] = patterns.get(bits, 0) + 1
        labels = row["labels"]
        gate_s += int(labels["gate_s_rescue"])
        gate_v_rescue += int(labels["gate_v_rescue"])
        gate_v_harm += int(labels["gate_v_harm"])

    if not count:
        return {"schema": "pvoca.cascade.v3.summary", "status": "EMPTY", "items": 0}

    stop_total = incremental["STOP"]
    think_total = incremental["STOP"] + incremental["THINK"]
    verify_total = think_total + incremental["VERIFY"]
    return {
        "schema": "pvoca.cascade.v3.summary",
        "status": "PASS",
        "items": count,
        "stage_accuracy": {a: correct[a] / count for a in actions},
        "incremental_tokens": incremental,
        "cumulative_policy_tokens": {
            "STOP": stop_total,
            "THINK": think_total,
            "VERIFY": verify_total,
        },
        "correctness_patterns": dict(sorted(patterns.items())),
        "gate_s_rescue_items": gate_s,
        "gate_v_rescue_items": gate_v_rescue,
        "gate_v_harm_items": gate_v_harm,
        "cost_semantics": "incremental-stage-cost",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect resumable P-VoCA v3 staged traces on local Ollama.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=Path("pvoca-cascade-v3-summary.json"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining items.")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    provider = OllamaLocalProvider(
        endpoint=args.endpoint,
        model=args.model,
        max_tokens=args.max_tokens,
    )
    items = load_items(args.dataset)
    done = completed_ids(args.output)
    remaining = [item for item in items if item.item_id not in done]
    if args.limit > 0:
        remaining = remaining[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for item in remaining:
            trace = run_cascade_trace(provider, item)
            handle.write(json.dumps(trace_to_jsonable(trace), ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            processed += 1
            print(json.dumps({"item_id": item.item_id, "processed_this_run": processed}, ensure_ascii=False))

    summary = summarize(args.output)
    summary["model"] = args.model
    summary["processed_this_run"] = processed
    summary["resume_skipped"] = len(done)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
