#!/usr/bin/env python3
"""Build a deterministic sanitized review bundle from public fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from scientific_llm_orchestrator.contracts import ReviewBundle, WorkOrder
    from scientific_llm_orchestrator.metrics import SanitizedMetrics
    from scientific_llm_orchestrator.providers import DryRunAdapter
    from scientific_llm_orchestrator.routing import Availability, route_work_order

    results = []
    for fixture_path in sorted((root / "benchmarks" / "fixtures" / "synthetic-public").glob("*.json")):
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        order = WorkOrder(
            work_order_id=fixture["fixture_id"],
            task_class=fixture["task_class"],
            operation=fixture["operation"],
            objective="Create a bounded proposal receipt for a public synthetic fixture.",
            architecture="FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL",
            allowed_paths=["benchmarks/fixtures/" + fixture["fixture_id"] + ".json"],
            protected_literals=fixture["protected_literals"],
            validation_ids=["schema", "path_scope", "protected_literals", "fixture_expected"],
        )
        decision = route_work_order(order, Availability(frontier=True, hosted_worker=True, local_worker=True))
        results.append(DryRunAdapter().execute(order, decision).to_dict())
    bundle = ReviewBundle(
        bundle_id="synthetic-public-dry-run",
        results=results,
        metrics=SanitizedMetrics.dry_run().to_dict(),
        evidence_label="observed-mock-runtime",
    )
    output = args.output or root / "review-bundle.json"
    output = output if output.is_absolute() else root / output
    output.write_text(bundle.to_json(), encoding="utf-8")
    print(f"review_bundle: PASS results={len(results)} evidence=observed-mock-runtime output={output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
