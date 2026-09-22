from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.pvoca_decision import (  # noqa: E402
    ResearchDecisionEvidence,
    ResearchDecisionGate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assess isolated P-VoCA evidence. This command never deploys, "
            "promotes, mutates production, or bypasses an owner gate."
        )
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    if payload.get("schema") != "pvoca.research-decision.evidence.v1":
        raise ValueError("expected pvoca.research-decision.evidence.v1")

    fields = {
        name: payload[name]
        for name in ResearchDecisionEvidence.__dataclass_fields__
        if name in payload
    }
    evidence = ResearchDecisionEvidence(**fields)
    result = ResearchDecisionGate().evaluate(evidence)
    result["schema"] = "pvoca.research-decision.v1"
    result["execution_effect"] = "READ_ONLY_DECISION"
    result["production_changed"] = False
    result["owner_gate_bypassed"] = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
