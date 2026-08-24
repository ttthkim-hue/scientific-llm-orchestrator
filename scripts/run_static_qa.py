#!/usr/bin/env python3
"""Run offline compile, schema, link, contract, and manifest checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Iterable, List

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "THIRD_PARTY_NOTICES.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "publication-manifest.json",
)
REQUIRED_DIRS = ("src", "schemas", "configs", "benchmarks", "examples", "scripts", "tests", "docs", ".github/workflows")
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
SCHEMA_FILES = {
    "WorkOrder": "schemas/work-order.schema.json",
    "Result": "schemas/result.schema.json",
    "ReviewBundle": "schemas/review-bundle.schema.json",
}


def _all_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
            yield path


def _basic_schema_check(schema: dict, name: str) -> List[str]:
    errors = []
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append(f"{name}: invalid schema declaration")
    if schema.get("type") != "object" or not isinstance(schema.get("required"), list):
        errors.append(f"{name}: schema must define object and required fields")
    if schema.get("additionalProperties") is not False:
        errors.append(f"{name}: additionalProperties must be false")
    return errors


def _schema_contract_check(root: Path) -> List[str]:
    """Validate representative contracts with the project's schema subset."""

    from scientific_llm_orchestrator.contracts import ReviewBundle, WorkOrder
    from scientific_llm_orchestrator.metrics import SanitizedMetrics
    from scientific_llm_orchestrator.providers import DryRunAdapter
    from scientific_llm_orchestrator.routing import Availability, route_work_order
    from scientific_llm_orchestrator.validators import validate

    schemas = {
        name: json.loads((root / relative).read_text(encoding="utf-8"))
        for name, relative in SCHEMA_FILES.items()
    }
    order = WorkOrder(
        work_order_id="static-schema-qa",
        task_class="coding",
        operation="schema_subset_smoke",
        objective="Exercise representative public contracts offline.",
        architecture="FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL",
        allowed_paths=["tests/schema_subset_smoke.py"],
        protected_literals=["schema_subset"],
        validation_ids=["schema", "path_scope"],
    )
    decision = route_work_order(order, Availability(frontier=True, hosted_worker=True, local_worker=True))
    result = DryRunAdapter().execute(order, decision)
    bundle = ReviewBundle(
        bundle_id="static-schema-qa-bundle",
        results=[result.to_dict()],
        metrics=SanitizedMetrics.dry_run().to_dict(),
        evidence_label="observed-mock-runtime",
    )
    instances = {
        "WorkOrder": order.to_dict(),
        "Result": result.to_dict(),
        "ReviewBundle": bundle.to_dict(),
    }
    errors: List[str] = []
    for name, instance in instances.items():
        schema = schemas[name]
        errors.extend(f"{name} schema: {error}" for error in validate(instance, schema))

        with_extra = dict(instance)
        with_extra["unexpected_public_qa_property"] = True
        if not validate(with_extra, schema):
            errors.append(f"{name} schema negative test accepted an extra property")

        with_apply = dict(instance)
        with_apply["automatic_apply"] = True
        if not validate(with_apply, schema):
            errors.append(f"{name} schema negative test accepted automatic_apply=true")
    return errors


def run(root: Path) -> List[str]:
    errors: List[str] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append("missing required file: " + relative)
    for relative in REQUIRED_DIRS:
        if not (root / relative).is_dir():
            errors.append("missing required directory: " + relative)
    json_files = list(root.rglob("*.json"))
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid JSON {path.relative_to(root).as_posix()}: {exc}")
    for path in sorted((root / "schemas").glob("*.json")):
        try:
            errors.extend(_basic_schema_check(json.loads(path.read_text(encoding="utf-8")), path.name))
        except Exception:
            pass
    for path in _all_files(root):
        if path.suffix == ".py":
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except Exception as exc:
                errors.append(f"compile failure {path.relative_to(root).as_posix()}: {exc}")
    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            target = target.split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            link_path = (path.parent / target).resolve()
            try:
                link_path.relative_to(root.resolve())
            except ValueError:
                errors.append(f"link escapes root {path.relative_to(root).as_posix()}: {target}")
                continue
            if not link_path.exists():
                errors.append(f"broken link {path.relative_to(root).as_posix()}: {target}")
    sys.path.insert(0, str(root / "src"))
    try:
        from scientific_llm_orchestrator.contracts import WorkOrder
        from scientific_llm_orchestrator.routing import Availability, route_work_order
        order = WorkOrder(
            work_order_id="static-qa",
            task_class="coding",
            operation="static_contract_smoke",
            objective="Exercise the public contract offline.",
            architecture="FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL",
            allowed_paths=["tests/static_contract_smoke.py"],
            protected_literals=["static-qa"],
            validation_ids=["schema", "path_scope"],
        )
        decision = route_work_order(order, Availability(frontier=True, hosted_worker=True, local_worker=True))
        if decision.route != "local_worker" or not decision.allowed:
            errors.append("contract smoke route mismatch")
    except Exception as exc:
        errors.append(f"contract smoke failure: {type(exc).__name__}: {exc}")
    try:
        errors.extend(_schema_contract_check(root))
    except Exception as exc:
        errors.append(f"schema subset contract failure: {type(exc).__name__}: {exc}")
    try:
        from scan_publication_boundary import build_manifest
        expected, _ = build_manifest(root)
        actual = json.loads((root / "publication-manifest.json").read_text(encoding="utf-8"))
        if actual != expected:
            errors.append("publication manifest is stale or not deterministic")
    except Exception as exc:
        errors.append(f"manifest check failure: {type(exc).__name__}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    errors = run(args.root.resolve())
    if errors:
        print(f"static_qa: FAIL errors={len(errors)}")
        for error in errors:
            print("error: " + error)
        return 1
    print("static_qa: PASS compile=json-schema-subset=links=contracts=manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
