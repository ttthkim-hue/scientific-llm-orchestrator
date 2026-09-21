from __future__ import annotations

import argparse
import json
from pathlib import Path

ACTIONS = ("STOP", "THINK", "VERIFY")


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pvoca.oracle.v0":
        raise ValueError("expected pvoca.oracle.v0 result file")
    return list(payload.get("records") or [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit P-VoCA v3 cascade labels from v0 counterfactual results.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("pvoca-cascade-v3-audit.json"))
    args = parser.parse_args()

    records = load_records(args.results)
    patterns: dict[str, int] = {}
    gate_s_positive = 0
    gate_v_rescue = 0
    gate_v_harm = 0
    gate_v_neutral = 0

    for record in records:
        rows = {row["action"]: row for row in record["results"]}
        if set(rows) != set(ACTIONS):
            continue
        stop = bool(rows["STOP"]["correct"])
        think = bool(rows["THINK"]["correct"])
        verify = bool(rows["VERIFY"]["correct"])
        pattern = f"{int(stop)}{int(think)}{int(verify)}"
        patterns[pattern] = patterns.get(pattern, 0) + 1

        gate_s_positive += int((not stop) and (think or verify))
        if verify and not think:
            gate_v_rescue += 1
        elif think and not verify:
            gate_v_harm += 1
        else:
            gate_v_neutral += 1

    usable = sum(patterns.values())
    out = {
        "schema": "pvoca.cascade.v3.audit",
        "status": "PASS",
        "items": usable,
        "correctness_patterns": dict(sorted(patterns.items())),
        "gate_s": {
            "positive_rescue_items": gate_s_positive,
            "negative_or_no_gain_items": usable - gate_s_positive,
        },
        "gate_v": {
            "rescue_items": gate_v_rescue,
            "harm_items": gate_v_harm,
            "neutral_items": gate_v_neutral,
        },
        "cost_boundary": (
            "v0 actions were measured as independent counterfactual executions; "
            "their action totals must not be treated as v3 sequential cascade cost. "
            "A v3 experiment must record incremental STOP, THINK, and VERIFY stage costs."
        ),
    }
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
