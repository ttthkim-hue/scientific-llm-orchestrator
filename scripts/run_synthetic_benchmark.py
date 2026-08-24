#!/usr/bin/env python3
"""Run the credential-free public fixture benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict


ARCHITECTURES = (
    "FRONTIER_ONLY",
    "FRONTIER_HOSTED_WORKER",
    "FRONTIER_LOCAL_WORKER",
    "FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL",
)


def _load_package(root: Path):
    sys.path.insert(0, str(root / "src"))
    from scientific_llm_orchestrator.contracts import WorkOrder
    from scientific_llm_orchestrator.routing import Availability, route_work_order
    from scientific_llm_orchestrator.validators import check_path_scope, check_protected_literals, check_validation_allowlist

    return WorkOrder, Availability, route_work_order, check_path_scope, check_protected_literals, check_validation_allowlist


def compute_expected(fixture: Dict[str, Any]) -> Dict[str, Any]:
    task_class = fixture["task_class"]
    data = fixture["input"]
    if task_class == "coding":
        if fixture["operation"] == "implement_function":
            return {"function_name": data["function_name"], "normalized": " ".join(data["value"].split())}
        denominator = data["denominator"]
        return {"function_name": data["function_name"], "value": None if denominator == 0 else data["numerator"] / denominator}
    if task_class == "data_scientific_mechanics":
        if fixture["operation"] == "csv_schema_unit_audit":
            headers = [item.strip().lower().replace(" ", "_") for item in data["headers"]]
            return {
                "header_count": len(headers),
                "row_count": len(data["rows"]),
                "units_complete": all(item in data["units"] for item in headers if item != "sample_id"),
                "normalized_headers": headers,
            }
        levels = Counter(event["level"] for event in data["events"])
        return {"event_count": len(data["events"]), "levels": dict(sorted(levels.items())), "root_cause_claim": None}
    if task_class == "visual_figure_qa":
        if fixture["operation"] == "multi_panel_label_inventory":
            return {
                "panel_count": len(data["panels"]),
                "panel_labels_sorted": sorted(data["panels"]),
                "axis_label_count": len(data["axis_labels"]),
                "legend_present": bool(data["legend"]),
            }
        return {
            "axis_present": bool(data["axis"]),
            "legend_count": len(data["legend_entries"]),
            "scale_bar_present": bool(data["scale_bar"]),
        }
    if task_class == "parametric_cad":
        width, depth, height, wall = (data[key] for key in ("width_mm", "depth_mm", "height_mm", "wall_mm"))
        inner = (width - 2 * wall) * (depth - 2 * wall) * (height - 2 * wall)
        return {
            "bounding_box_mm": [width, depth, height],
            "outer_volume_mm3": width * depth * height,
            "inner_volume_mm3": inner,
            "wall_thickness_mm": wall,
            "clearance_mm": data["clearance_mm"],
        }
    raise ValueError("unsupported fixture task class")


def load_fixtures(root: Path):
    fixture_dir = root / "benchmarks" / "fixtures" / "synthetic-public"
    expected_path = root / "benchmarks" / "expected" / "synthetic-public.expected.json"
    expected_index = json.loads(expected_path.read_text(encoding="utf-8"))
    fixtures = []
    for path in sorted(fixture_dir.glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        if not fixture.get("public_safe"):
            raise ValueError(f"fixture is not public-safe: {path.name}")
        if fixture["fixture_id"] not in expected_index:
            raise ValueError(f"missing expected result: {fixture['fixture_id']}")
        fixtures.append(fixture)
    if not fixtures:
        raise ValueError("no synthetic fixtures found")
    return fixtures, expected_index


def run(root: Path) -> Dict[str, Any]:
    WorkOrder, Availability, route_work_order, check_path_scope, check_protected_literals, check_validation_allowlist = _load_package(root)
    fixtures, expected_index = load_fixtures(root)
    route_expected = {
        "FRONTIER_ONLY": "frontier",
        "FRONTIER_HOSTED_WORKER": "hosted_worker",
        "FRONTIER_LOCAL_WORKER": "local_worker",
        "FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL": "local_worker",
    }
    availability = {
        "FRONTIER_ONLY": Availability(frontier=True),
        "FRONTIER_HOSTED_WORKER": Availability(frontier=True, hosted_worker=True),
        "FRONTIER_LOCAL_WORKER": Availability(frontier=True, local_worker=True),
        "FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL": Availability(frontier=True, hosted_worker=True, local_worker=True),
    }
    failures = []
    cases = 0
    by_class = Counter()
    for architecture in ARCHITECTURES:
        for fixture in fixtures:
            cases += 1
            by_class[fixture["task_class"]] += 1
            computed = compute_expected(fixture)
            checks = {
                "expected_file": computed == expected_index[fixture["fixture_id"]],
                "fixture_expected": computed == fixture["expected"],
                "path_scope": check_path_scope(["benchmarks/fixtures/" + fixture["fixture_id"] + ".json"]).passed,
                "protected_literals": check_protected_literals(fixture["protected_literals"], " ".join(fixture["protected_literals"])).passed,
                "validation_allowlist": check_validation_allowlist(["schema", "path_scope", "protected_literals", "fixture_expected"]).passed,
                "route": route_work_order(
                    WorkOrder(
                        work_order_id=fixture["fixture_id"],
                        task_class=fixture["task_class"],
                        operation=fixture["operation"],
                        objective="Evaluate one public synthetic fixture deterministically.",
                        architecture=architecture,
                        allowed_paths=["benchmarks/fixtures/" + fixture["fixture_id"] + ".json"],
                        protected_literals=fixture["protected_literals"],
                        validation_ids=["schema", "path_scope", "protected_literals", "fixture_expected"],
                    ),
                    availability[architecture],
                ).route == route_expected[architecture],
            }
            failures.extend(f"{architecture}/{fixture['fixture_id']}:{name}" for name, passed in checks.items() if not passed)
    return {
        "status": "PASS" if not failures else "FAIL",
        "evidence_label": "observed-deterministic",
        "actual_model_evidence": "not-run",
        "architecture_count": len(ARCHITECTURES),
        "fixture_count": len(fixtures),
        "case_count": cases,
        "task_classes": dict(sorted(by_class.items())),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        result = run(args.root.resolve())
    except Exception as exc:  # deterministic CLI boundary
        print(f"synthetic_benchmark: ERROR {type(exc).__name__}: {exc}")
        return 1
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "synthetic_benchmark: {status} cases={case_count} architectures={architecture_count} "
            "evidence={evidence_label} actual_model={actual_model_evidence}".format(**result)
        )
        if result["failures"]:
            print("failures: " + ", ".join(result["failures"]))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
