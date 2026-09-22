import unittest

from scientific_llm_orchestrator.pvoca import Action
from scientific_llm_orchestrator.pvoca_baselines import (
    agreement_verify_action,
    relative_disagreement,
    select_agreement_verify_threshold,
)


def row(stop_value, think_value, think_ok, verify_ok):
    return {
        "stages": [
            {"action": "STOP", "predicted_value": stop_value, "correct": False, "incremental_tokens": 10},
            {"action": "THINK", "predicted_value": think_value, "correct": think_ok, "incremental_tokens": 20},
            {"action": "VERIFY", "predicted_value": think_value, "correct": verify_ok, "incremental_tokens": 30},
        ]
    }


class PvocaBaselineTests(unittest.TestCase):
    def test_relative_disagreement(self):
        self.assertAlmostEqual(relative_disagreement(100.0, 110.0), 10.0 / 110.0)
        self.assertEqual(relative_disagreement(None, 1.0), float("inf"))

    def test_agreement_router_uses_verify_on_large_disagreement(self):
        trace = row(1.0, 2.0, False, True)
        self.assertEqual(agreement_verify_action(trace, threshold=0.2), Action.VERIFY)
        self.assertEqual(agreement_verify_action(trace, threshold=0.8), Action.THINK)

    def test_threshold_is_selected_on_correctness_then_cost(self):
        rows = [
            row(10.0, 10.1, True, True),
            row(1.0, 2.0, False, True),
            row(5.0, 5.1, True, False),
        ]
        threshold = select_agreement_verify_threshold(rows)
        choices = [agreement_verify_action(item, threshold=threshold) for item in rows]
        self.assertEqual(choices[1], Action.VERIFY)
        self.assertEqual(choices[2], Action.THINK)


if __name__ == "__main__":
    unittest.main()
