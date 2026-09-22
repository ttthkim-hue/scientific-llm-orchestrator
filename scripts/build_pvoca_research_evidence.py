from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.pvoca_evidence import aggregate_controller_runs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple isolated P-VoCA v3 run summaries conservatively."
    )
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--matched-baselines", type=int, required=True)
    parser.add_argument("--shadow", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.run]
    shadow = (
        json.loads(args.shadow.read_text(encoding="utf-8"))
        if args.shadow is not None
        else None
    )
    evidence = aggregate_controller_runs(
        runs,
        matched_baselines=args.matched_baselines,
        shadow=shadow,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
