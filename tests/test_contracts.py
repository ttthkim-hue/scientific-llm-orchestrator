import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.contracts import (  # noqa: E402
    ARCHITECTURES,
    EVIDENCE_LABELS,
    Result,
    ReviewBundle,
    WorkOrder,
)
from scientific_llm_orchestrator.metrics import SanitizedMetrics  # noqa: E402
from scientific_llm_orchestrator.providers import DryRunAdapter  # noqa: E402
from scientific_llm_orchestrator.routing import Availability, route_work_order  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_work_order_round_trip_and_no_apply(self):
        order = WorkOrder(
            work_order_id="contract-test",
            task_class="coding",
            operation="unit_test_generation",
            objective="Create a deterministic test proposal.",
            architecture="FRONTIER_ONLY",
            allowed_paths=["examples/coding/example.py"],
            protected_literals=["example_function"],
            validation_ids=["schema", "path_scope"],
        )
        restored = WorkOrder.from_dict(json.loads(order.to_json()))
        self.assertEqual(order, restored)
        self.assertFalse(order.automatic_apply)
        self.assertTrue(all(order.authority_flags.to_dict().values()))

    def test_result_and_bundle_evidence_are_allowlisted(self):
        result = Result(
            work_order_id="contract-test",
            route="frontier",
            status="dry-run",
            evidence_label="observed-mock-runtime",
            provider="dry-run-no-provider",
            dry_run=True,
        )
        bundle = ReviewBundle(
            bundle_id="bundle-test",
            results=[result.to_dict()],
            metrics={"evidence_label": "observed-mock-runtime"},
        )
        self.assertIn(result.evidence_label, EVIDENCE_LABELS)
        self.assertIn(bundle.evidence_label, EVIDENCE_LABELS)
        self.assertFalse(bundle.automatic_apply)

    def test_architecture_names_are_exact(self):
        self.assertEqual(
            ARCHITECTURES,
            (
                "FRONTIER_ONLY",
                "FRONTIER_HOSTED_WORKER",
                "FRONTIER_LOCAL_WORKER",
                "FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL",
            ),
        )

    def test_dry_run_does_not_claim_unexecuted_checks(self):
        order = WorkOrder(
            work_order_id="dry-run-contract",
            task_class="coding",
            operation="receipt_only",
            objective="Create a route receipt without executing a worker.",
            architecture="FRONTIER_LOCAL_WORKER",
            allowed_paths=["tests/test_contracts.py"],
            protected_literals=["protected"],
            validation_ids=["schema", "path_scope"],
        )
        decision = route_work_order(order, Availability(frontier=True, local_worker=True))
        result = DryRunAdapter().execute(order, decision)
        metrics = SanitizedMetrics.dry_run()
        self.assertIsNone(result.protected_literals_preserved)
        self.assertEqual(result.validation_results, {})
        self.assertIsNone(metrics.protected_literal_pass)
        self.assertIsNone(metrics.deterministic_functional_pass)
        self.assertIsNone(metrics.fallback_rate)
        self.assertIsNone(metrics.privacy_class_violation_count)
        self.assertIsNone(metrics.hosted_token_displacement)
        self.assertIsNone(metrics.vram_mib)
