import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.contracts import ReviewBundle, WorkOrder  # noqa: E402
from scientific_llm_orchestrator.metrics import SanitizedMetrics  # noqa: E402
from scientific_llm_orchestrator.providers import DryRunAdapter  # noqa: E402
from scientific_llm_orchestrator.routing import Availability, route_work_order  # noqa: E402
from scientific_llm_orchestrator.validators import validate  # noqa: E402


class SchemaSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = {
            "WorkOrder": json.loads((ROOT / "schemas/work-order.schema.json").read_text(encoding="utf-8")),
            "Result": json.loads((ROOT / "schemas/result.schema.json").read_text(encoding="utf-8")),
            "ReviewBundle": json.loads((ROOT / "schemas/review-bundle.schema.json").read_text(encoding="utf-8")),
        }
        cls.order = WorkOrder(
            work_order_id="schema-test",
            task_class="coding",
            operation="schema_subset_test",
            objective="Validate representative public contracts.",
            architecture="FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL",
            allowed_paths=["tests/test_schema_subset.py"],
            protected_literals=["schema-test"],
            validation_ids=["schema", "path_scope"],
        )
        decision = route_work_order(cls.order, Availability(frontier=True, hosted_worker=True, local_worker=True))
        cls.result = DryRunAdapter().execute(cls.order, decision)
        cls.bundle = ReviewBundle(
            bundle_id="schema-test-bundle",
            results=[cls.result.to_dict()],
            metrics=SanitizedMetrics.dry_run().to_dict(),
            evidence_label="observed-mock-runtime",
        )

    def test_representative_contracts_validate(self):
        instances = {
            "WorkOrder": self.order.to_dict(),
            "Result": self.result.to_dict(),
            "ReviewBundle": self.bundle.to_dict(),
        }
        for name, instance in instances.items():
            with self.subTest(name=name):
                self.assertEqual(validate(instance, self.schemas[name]), [])

    def test_extra_properties_are_rejected(self):
        instance = self.order.to_dict()
        instance["unexpected"] = True
        self.assertTrue(validate(instance, self.schemas["WorkOrder"]))

    def test_automatic_apply_true_is_rejected(self):
        instance = self.bundle.to_dict()
        instance["automatic_apply"] = True
        self.assertTrue(validate(instance, self.schemas["ReviewBundle"]))
