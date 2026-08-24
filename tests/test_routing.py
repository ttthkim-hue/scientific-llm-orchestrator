from pathlib import Path
import sys
import unittest
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.contracts import WorkOrder  # noqa: E402
from scientific_llm_orchestrator.routing import Availability, route_work_order  # noqa: E402


def make_order(architecture):
    return WorkOrder(
        work_order_id="route-test",
        task_class="coding",
        operation="bounded_change",
        objective="Produce a proposal receipt.",
        architecture=architecture,
        allowed_paths=["tests/example.py"],
        protected_literals=[],
        validation_ids=["schema"],
    )


class RoutingTests(unittest.TestCase):
    def test_each_baseline_selects_its_worker(self):
        cases = {
            "FRONTIER_ONLY": (Availability(frontier=True), "frontier"),
            "FRONTIER_HOSTED_WORKER": (Availability(frontier=True, hosted_worker=True), "hosted_worker"),
            "FRONTIER_LOCAL_WORKER": (Availability(frontier=True, local_worker=True), "local_worker"),
            "FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL": (Availability(frontier=True, hosted_worker=True, local_worker=True), "local_worker"),
        }
        for architecture, (availability, expected) in cases.items():
            with self.subTest(architecture=architecture):
                decision = route_work_order(make_order(architecture), availability)
                self.assertEqual(decision.route, expected)
                self.assertTrue(decision.allowed)

    def test_worker_failure_falls_back_or_blocks(self):
        decision = route_work_order(
            make_order("FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL"),
            Availability(frontier=True, hosted_worker=True, local_worker=False),
        )
        self.assertEqual(decision.route, "hosted_worker")
        blocked = route_work_order(
            make_order("FRONTIER_ONLY"),
            Availability(frontier=False),
        )
        self.assertEqual(blocked.route, "blocked")
        self.assertFalse(blocked.allowed)

    def test_invalid_paths_fail_closed_before_worker_selection(self):
        for path in ("../outside.py", "C" + ":/outside.py", "/outside.py"):
            with self.subTest(path=path):
                order = replace(make_order("FRONTIER_LOCAL_WORKER"), allowed_paths=[path])
                decision = route_work_order(
                    order,
                    Availability(frontier=True, hosted_worker=True, local_worker=True),
                )
                self.assertEqual(decision.route, "blocked")
                self.assertFalse(decision.allowed)
                self.assertEqual(decision.reason, "path_scope_failed")

    def test_invalid_validation_id_fails_closed_before_worker_selection(self):
        order = replace(make_order("FRONTIER_LOCAL_WORKER"), validation_ids=["execute_shell"])
        decision = route_work_order(
            order,
            Availability(frontier=True, hosted_worker=True, local_worker=True),
        )
        self.assertEqual(decision.route, "blocked")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "validation_allowlist_failed")
