import unittest

from scientific_llm_orchestrator.pvoca_stats import (
    exact_mcnemar_p,
    paired_bootstrap_savings_interval,
)


class PvocaStatsTests(unittest.TestCase):
    def test_mcnemar_no_disagreement(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)

    def test_mcnemar_detects_strong_asymmetry(self):
        self.assertLess(exact_mcnemar_p(20, 2), 0.01)
        self.assertEqual(exact_mcnemar_p(20, 2), exact_mcnemar_p(2, 20))

    def test_bootstrap_savings_is_paired_and_deterministic(self):
        controller = [8.0] * 50
        baseline = [10.0] * 50
        first = paired_bootstrap_savings_interval(
            controller,
            baseline,
            resamples=200,
            seed=7,
        )
        second = paired_bootstrap_savings_interval(
            controller,
            baseline,
            resamples=200,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 0.2)
        self.assertAlmostEqual(first[1], 0.2)

    def test_bootstrap_rejects_zero_baseline_cost(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_savings_interval([1.0], [0.0], resamples=100)


if __name__ == "__main__":
    unittest.main()
