import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.routing import (  # noqa: E402
    CaptainAvailability,
    CaptainRequest,
    route_captain_task,
)


class CaptainPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.available = CaptainAvailability(
            frontier=True,
            local_mechanical_worker=True,
            private_implementation_worker=True,
            routine_hosted_worker=True,
            complex_hosted_worker=True,
            hosted_route_active=True,
            hosted_usage_headroom=True,
        )

    def test_exact_tool_and_authority_take_their_fixed_routes(self):
        exact = route_captain_task(
            CaptainRequest("public-synthetic", "complex", exact_tool_available=True),
            self.available,
        )
        authority = route_captain_task(
            CaptainRequest("public-synthetic", "authority"), self.available
        )
        self.assertEqual(exact.role, "deterministic_tool")
        self.assertEqual(authority.role, "frontier")
        self.assertFalse(exact.automatic_apply)
        self.assertTrue(exact.requires_frontier_acceptance)

    def test_mechanical_private_routine_and_complex_roles_are_distinct(self):
        cases = {
            ("public-synthetic", "mechanical"): "local_mechanical_worker",
            ("local-private", "complex"): "private_implementation_worker",
            ("public-synthetic", "routine"): "routine_hosted_worker",
            ("public-synthetic", "complex"): "complex_hosted_worker",
        }
        for (privacy, shape), expected in cases.items():
            with self.subTest(privacy=privacy, shape=shape):
                decision = route_captain_task(
                    CaptainRequest(privacy, shape), self.available
                )
                self.assertEqual(decision.role, expected)
                self.assertTrue(decision.allowed)

    def test_hosted_preflight_failure_stays_private_without_provider_retry(self):
        unavailable = CaptainAvailability(
            frontier=True,
            private_implementation_worker=True,
            complex_hosted_worker=True,
            hosted_route_active=True,
            hosted_usage_headroom=False,
        )
        decision = route_captain_task(
            CaptainRequest("public-synthetic", "complex"), unavailable
        )
        self.assertEqual(decision.role, "private_implementation_worker")
        self.assertEqual(decision.reason, "hosted_preflight_unavailable_private_route")

    def test_missing_evaluator_and_unknown_inputs_fail_closed(self):
        no_evaluator = route_captain_task(
            CaptainRequest(
                "public-synthetic", "complex", deterministic_evaluator_available=False
            ),
            self.available,
        )
        unknown = route_captain_task(
            CaptainRequest("unknown", "complex"), self.available
        )
        self.assertEqual(no_evaluator.role, "frontier")
        self.assertEqual(unknown.role, "blocked")
        self.assertFalse(unknown.allowed)

    def test_public_policy_is_provider_neutral_and_provisional(self):
        policy_path = ROOT / "configs" / "captain-portfolio.example.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        text = policy_path.read_text(encoding="utf-8").lower()
        self.assertFalse(policy["provider_bindings_included"])
        self.assertFalse(policy["automatic_apply"])
        self.assertFalse(policy["promotion_gate"]["quality_equivalence_claimed"])
        self.assertEqual(
            policy["promotion_gate"]["minimum_independent_qa_receipts"], 20
        )
        self.assertEqual(policy["promotion_gate"]["minimum_stable_task_families"], 3)
        for forbidden in (
            '"model_id"',
            '"provider_account"',
            '"credential_value"',
            '"price_usd"',
        ):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
