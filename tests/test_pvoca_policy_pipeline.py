import unittest

from scientific_llm_orchestrator.pvoca_decision import (
    ResearchDecisionEvidence,
    ResearchDecisionGate,
)
from scientific_llm_orchestrator.pvoca_evidence import aggregate_controller_runs


def controller_run(model, benchmark, family, *, items, token=0.20, latency=0.15, drop=0.2):
    return {
        "schema": "pvoca.cascade.controller.v3",
        "model_label": model,
        "benchmark": benchmark,
        "task_family": family,
        "items": items,
        "matched_baselines": [
            "always_stop",
            "always_think",
            "always_verify",
            "question_only_router",
        ],
        "fold_reports": [
            {"held_out_groups": ["g1", "g2", "g3"]},
            {"held_out_groups": ["g4", "g5"]},
        ],
        "heterogeneous_action_items": 80,
        "oracle": {"headroom_pp_vs_best_fixed": 4.0},
        "calibration": {"max_ece": 0.05},
        "offline_decision_metrics": {
            "accuracy_drop_upper_pp": drop,
            "token_savings": token,
            "latency_savings": latency,
            "unsafe_stop_rate_upper": 0.008,
        },
    }


class PvocaPolicyPipelineTests(unittest.TestCase):
    def test_multi_run_offline_evidence_reaches_shadow_candidate(self):
        runs = [
            controller_run("qwen3.8:27b", "MatSciBench", "materials_numeric", items=700),
            controller_run("qwen3.5:4b", "MatSciBench", "materials_numeric", items=700),
            controller_run("qwen3.8:27b", "SciBench", "college_science_numeric", items=500),
            controller_run("qwen3.5:4b", "SciBench", "college_science_numeric", items=500),
        ]
        evidence_dict = aggregate_controller_runs(runs)
        evidence = ResearchDecisionEvidence(
            **{
                name: evidence_dict[name]
                for name in ResearchDecisionEvidence.__dataclass_fields__
                if name in evidence_dict
            }
        )
        decision = ResearchDecisionGate().evaluate(evidence)
        self.assertEqual(decision["status"], "GLOBAL_SHADOW_CANDIDATE")
        self.assertFalse(decision["global_deploy_candidate"])

    def test_shadow_candidate_does_not_imply_deployment(self):
        runs = [
            controller_run("m1", "B1", "F1", items=600),
            controller_run("m2", "B2", "F2", items=600),
        ]
        evidence_dict = aggregate_controller_runs(
            runs,
            shadow={
                "shadow_items": 9999,
                "shadow_accuracy_drop_upper_pp": 0.1,
                "shadow_token_savings": 0.25,
                "shadow_latency_savings": 0.20,
                "shadow_unsafe_stop_rate_upper": 0.003,
                "shadow_crashes": 0,
                "owner_gate_bypasses": 0,
                "production_mutations": 0,
            },
        )
        evidence = ResearchDecisionEvidence(
            **{
                name: evidence_dict[name]
                for name in ResearchDecisionEvidence.__dataclass_fields__
                if name in evidence_dict
            }
        )
        decision = ResearchDecisionGate().evaluate(evidence)
        self.assertEqual(decision["status"], "GLOBAL_SHADOW_CANDIDATE")
        self.assertFalse(decision["deploy_checks"]["shadow_sample_size"])

    def test_complete_null_result_stays_publishable_but_isolated(self):
        evidence = ResearchDecisionEvidence(
            unseen_items=1200,
            distinct_models=2,
            task_families=2,
            heldout_groups=10,
            matched_baselines=4,
            reproducible_runs=4,
            paired_transition_items=0,
            oracle_headroom_pp=0.0,
            calibration_ece=0.05,
            leakage_detected=False,
            accuracy_drop_upper_pp=3.0,
            token_savings=0.0,
            latency_savings=0.0,
            unsafe_stop_rate_upper=0.10,
        )
        decision = ResearchDecisionGate().evaluate(evidence)
        self.assertEqual(decision["status"], "PAPER_READY_NEGATIVE_OR_NULL")
        self.assertFalse(decision["global_shadow_candidate"])


if __name__ == "__main__":
    unittest.main()
