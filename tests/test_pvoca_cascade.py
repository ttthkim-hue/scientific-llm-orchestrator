import unittest

from scientific_llm_orchestrator.pvoca import Action
from scientific_llm_orchestrator.pvoca_cascade import (
    CascadeExample,
    Outcome,
    PromotionEvidence,
    PromotionGate,
    choose_stage_action,
    evaluate_choices,
    gate_s_target,
    gate_v_target,
)


class PvocaCascadeTests(unittest.TestCase):
    def test_gate_targets_capture_marginal_value(self):
        rescued = CascadeExample(Outcome(False, 10), Outcome(True, 40), Outcome(False, 60))
        verify_only = CascadeExample(Outcome(False, 10), Outcome(False, 40), Outcome(True, 60))
        harmful_verify = CascadeExample(Outcome(False, 10), Outcome(True, 40), Outcome(False, 60))
        neutral = CascadeExample(Outcome(True, 10), Outcome(True, 40), Outcome(True, 60))
        self.assertEqual(gate_s_target(rescued), 1)
        self.assertEqual(gate_s_target(verify_only), 1)
        self.assertEqual(gate_s_target(neutral), 0)
        self.assertEqual(gate_v_target(verify_only), 1)
        self.assertEqual(gate_v_target(harmful_verify), -1)
        self.assertEqual(gate_v_target(neutral), 0)

    def test_conservative_stage_policy(self):
        self.assertEqual(
            choose_stage_action(
                p_stop_rescue=0.01,
                p_verify_rescue=0.9,
                p_verify_harm=0.0,
                stop_threshold=0.02,
                verify_margin=0.2,
            ),
            Action.STOP,
        )
        self.assertEqual(
            choose_stage_action(
                p_stop_rescue=0.5,
                p_verify_rescue=0.8,
                p_verify_harm=0.2,
                stop_threshold=0.02,
                verify_margin=0.3,
            ),
            Action.VERIFY,
        )
        self.assertEqual(
            choose_stage_action(
                p_stop_rescue=0.5,
                p_verify_rescue=0.4,
                p_verify_harm=0.3,
                stop_threshold=0.02,
                verify_margin=0.2,
            ),
            Action.THINK,
        )

    def test_policy_evaluation(self):
        rows = [
            CascadeExample(Outcome(True, 10), Outcome(True, 40), Outcome(True, 60)),
            CascadeExample(Outcome(False, 10), Outcome(True, 40), Outcome(True, 60)),
        ]
        metrics = evaluate_choices(rows, [Action.STOP, Action.THINK])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["tokens"], 50)

    def test_promotion_gate_stays_shadow_on_small_sample(self):
        result = PromotionGate().evaluate(
            PromotionEvidence(
                unseen_items=108,
                baseline_correct=46,
                controller_correct=46,
                baseline_tokens=100000,
                controller_tokens=80000,
                unsafe_stop_errors=0,
                stop_decisions=20,
            )
        )
        self.assertEqual(result["status"], "SHADOW_ONLY")
        self.assertFalse(result["checks"]["sample_size"])

    def test_promotion_gate_can_pass_strong_evidence(self):
        result = PromotionGate(max_unsafe_stop_rate=0.01).evaluate(
            PromotionEvidence(
                unseen_items=5000,
                baseline_correct=4000,
                controller_correct=3995,
                baseline_tokens=5_000_000,
                controller_tokens=3_500_000,
                unsafe_stop_errors=0,
                stop_decisions=1000,
            )
        )
        self.assertEqual(result["status"], "PROMOTION_READY")
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
