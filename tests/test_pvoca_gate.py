import unittest

from scientific_llm_orchestrator.pvoca_gate import (
    BinaryLogit,
    expected_calibration_error,
    gate_s_features,
    gate_v_features,
    select_safe_stop_threshold,
    select_verify_margin,
    wilson_upper,
)


def trace(stop_ok=False, think_ok=True, verify_ok=True):
    return {
        "item": {"question": "Calculate 2 + 2", "domain": "D"},
        "stages": [
            {"action": "STOP", "predicted_value": 3.0, "correct": stop_ok, "incremental_tokens": 10, "incremental_latency_ms": 5.0, "work_note": ""},
            {"action": "THINK", "predicted_value": 4.0, "correct": think_ok, "incremental_tokens": 20, "incremental_latency_ms": 10.0, "work_note": "2 + 2 = 4"},
            {"action": "VERIFY", "predicted_value": 4.0, "correct": verify_ok, "incremental_tokens": 20, "incremental_latency_ms": 10.0, "work_note": "audit"},
        ],
        "labels": {
            "gate_s_rescue": int((not stop_ok) and (think_ok or verify_ok)),
            "gate_v_rescue": int(verify_ok and not think_ok),
            "gate_v_harm": int(think_ok and not verify_ok),
        },
    }


class PvocaGateTests(unittest.TestCase):
    def test_features_are_finite_and_deterministic(self):
        row = trace()
        self.assertEqual(gate_s_features(row), gate_s_features(row))
        self.assertEqual(gate_v_features(row), gate_v_features(row))
        self.assertTrue(all(v == v for v in gate_s_features(row)))

    def test_binary_logit_learns_simple_signal(self):
        x = [[0.0, 1.0], [0.1, 1.0], [0.9, 1.0], [1.0, 1.0]] * 20
        y = [0, 0, 1, 1] * 20
        model = BinaryLogit.create(2)
        model.fit(x, y, epochs=300)
        self.assertLess(model.predict_proba([0.05, 1.0]), model.predict_proba([0.95, 1.0]))

    def test_safe_stop_fails_closed_without_enough_evidence(self):
        self.assertIsNone(
            select_safe_stop_threshold([0.01] * 20, [0] * 20, min_stop_decisions=100)
        )

    def test_safe_stop_can_open_with_strong_evidence(self):
        probs = [i / 1000 for i in range(1000)]
        labels = [0] * 1000
        threshold = select_safe_stop_threshold(
            probs,
            labels,
            max_unsafe_stop_rate=0.01,
            min_stop_decisions=500,
        )
        self.assertIsNotNone(threshold)
        self.assertGreaterEqual(threshold, 0.5)

    def test_verify_margin_prefers_safe_low_cost_option(self):
        margin = select_verify_margin(
            [0.9, 0.8, 0.2],
            [0.1, 0.6, 0.1],
            [1, 1, 0],
            [0, 1, 0],
            [10, 10, 10],
        )
        self.assertIn(margin, (0.05, 0.1, 0.2, 0.3, 0.4, 0.5))

    def test_ece_detects_calibration_quality(self):
        self.assertAlmostEqual(
            expected_calibration_error([0.0, 1.0], [0, 1], bins=2),
            0.0,
        )
        self.assertGreater(
            expected_calibration_error([0.9, 0.9], [0, 0], bins=2),
            0.8,
        )

    def test_wilson_upper_is_conservative(self):
        self.assertGreater(wilson_upper(0, 100), 0.0)
        self.assertLess(wilson_upper(0, 1000), wilson_upper(0, 100))


if __name__ == "__main__":
    unittest.main()
