import unittest

from scientific_llm_orchestrator.pvoca_decision import (
    ResearchDecisionEvidence,
    ResearchDecisionGate,
)


def complete_evidence(**overrides):
    data = dict(
        unseen_items=2000,
        distinct_models=2,
        task_families=2,
        heldout_groups=10,
        matched_baselines=5,
        reproducible_runs=3,
        paired_transition_items=100,
        oracle_headroom_pp=5.0,
        calibration_ece=0.05,
        leakage_detected=False,
        accuracy_drop_upper_pp=0.4,
        token_savings=0.18,
        latency_savings=0.12,
        unsafe_stop_rate_upper=0.008,
    )
    data.update(overrides)
    return ResearchDecisionEvidence(**data)


class ResearchDecisionGateTests(unittest.TestCase):
    def test_incomplete_study_continues_research(self):
        result = ResearchDecisionGate().evaluate(
            complete_evidence(unseen_items=108, distinct_models=1, task_families=1)
        )
        self.assertEqual(result["status"], "RESEARCH_CONTINUE")
        self.assertFalse(result["paper_evidence_ready"])

    def test_complete_negative_result_can_be_paper_ready(self):
        result = ResearchDecisionGate().evaluate(
            complete_evidence(
                accuracy_drop_upper_pp=3.0,
                token_savings=0.01,
                latency_savings=0.01,
                unsafe_stop_rate_upper=0.05,
            )
        )
        self.assertEqual(result["status"], "PAPER_READY_NEGATIVE_OR_NULL")
        self.assertTrue(result["paper_evidence_ready"])
        self.assertFalse(result["global_shadow_candidate"])

    def test_methodologically_complete_null_study_is_still_paper_ready(self):
        result = ResearchDecisionGate().evaluate(
            complete_evidence(
                oracle_headroom_pp=0.0,
                paired_transition_items=0,
                accuracy_drop_upper_pp=4.0,
                token_savings=0.0,
                latency_savings=0.0,
                unsafe_stop_rate_upper=0.10,
            )
        )
        self.assertEqual(result["status"], "PAPER_READY_NEGATIVE_OR_NULL")
        self.assertTrue(result["paper_evidence_ready"])
        self.assertFalse(result["mechanism_signal"])
        self.assertFalse(result["global_shadow_candidate"])

    def test_positive_paper_without_shadow_candidate(self):
        result = ResearchDecisionGate().evaluate(
            complete_evidence(
                accuracy_drop_upper_pp=0.8,
                token_savings=0.12,
                latency_savings=0.05,
                unsafe_stop_rate_upper=0.02,
            )
        )
        self.assertEqual(result["status"], "PAPER_READY_POSITIVE")
        self.assertFalse(result["global_shadow_candidate"])

    def test_strong_offline_result_enters_shadow_only(self):
        result = ResearchDecisionGate().evaluate(complete_evidence())
        self.assertEqual(result["status"], "GLOBAL_SHADOW_CANDIDATE")
        self.assertTrue(result["global_shadow_candidate"])
        self.assertFalse(result["global_deploy_candidate"])

    def test_global_deploy_requires_large_clean_shadow(self):
        result = ResearchDecisionGate().evaluate(
            complete_evidence(
                shadow_items=15000,
                shadow_accuracy_drop_upper_pp=0.1,
                shadow_token_savings=0.25,
                shadow_latency_savings=0.20,
                shadow_unsafe_stop_rate_upper=0.003,
                shadow_crashes=0,
                owner_gate_bypasses=0,
                production_mutations=0,
            )
        )
        self.assertEqual(result["status"], "GLOBAL_DEPLOY_CANDIDATE")
        self.assertTrue(result["global_deploy_candidate"])

    def test_any_shadow_safety_incident_blocks_deploy(self):
        result = ResearchDecisionGate().evaluate(
            complete_evidence(
                shadow_items=15000,
                shadow_accuracy_drop_upper_pp=0.1,
                shadow_token_savings=0.25,
                shadow_latency_savings=0.20,
                shadow_unsafe_stop_rate_upper=0.003,
                shadow_crashes=1,
            )
        )
        self.assertEqual(result["status"], "GLOBAL_SHADOW_CANDIDATE")
        self.assertFalse(result["global_deploy_candidate"])


if __name__ == "__main__":
    unittest.main()
