import unittest

from scientific_llm_orchestrator.pvoca_evidence import aggregate_controller_runs


def run(model, benchmark, family, *, items=600, headroom=5.0, acc_drop=0.2, token=0.2, latency=0.15, ece=0.05):
    return {
        "schema": "pvoca.cascade.controller.v3",
        "model_label": model,
        "benchmark": benchmark,
        "task_family": family,
        "items": items,
        "fold_reports": [
            {"held_out_groups": ["A", "B"]},
            {"held_out_groups": ["C"]},
        ],
        "heterogeneous_action_items": 80,
        "oracle": {"headroom_pp_vs_best_fixed": headroom},
        "calibration": {"max_ece": ece},
        "offline_decision_metrics": {
            "accuracy_drop_upper_pp": acc_drop,
            "token_savings": token,
            "latency_savings": latency,
            "unsafe_stop_rate_upper": 0.008,
        },
    }


class PvocaEvidenceTests(unittest.TestCase):
    def test_repeated_models_do_not_double_unique_items(self):
        rows = [
            run("m1", "MatSciBench", "materials_numeric", items=700),
            run("m2", "MatSciBench", "materials_numeric", items=700),
            run("m1", "SciBench", "college_science_numeric", items=500),
            run("m2", "SciBench", "college_science_numeric", items=500),
        ]
        evidence = aggregate_controller_runs(rows, matched_baselines=5)
        self.assertEqual(evidence["unseen_items"], 1200)
        self.assertEqual(evidence["distinct_models"], 2)
        self.assertEqual(evidence["task_families"], 2)
        self.assertEqual(evidence["reproducible_runs"], 4)

    def test_worst_run_controls_deployment_metrics(self):
        rows = [
            run("m1", "B1", "F1", headroom=6.0, acc_drop=0.1, token=0.25, latency=0.2, ece=0.03),
            run("m2", "B2", "F2", headroom=2.5, acc_drop=0.7, token=0.12, latency=0.08, ece=0.11),
        ]
        evidence = aggregate_controller_runs(rows, matched_baselines=5)
        self.assertEqual(evidence["oracle_headroom_pp"], 2.5)
        self.assertEqual(evidence["accuracy_drop_upper_pp"], 0.7)
        self.assertEqual(evidence["token_savings"], 0.12)
        self.assertEqual(evidence["latency_savings"], 0.08)
        self.assertEqual(evidence["calibration_ece"], 0.11)

    def test_requires_explicit_run_identity(self):
        bad = run("m1", "B1", "F1")
        bad["model_label"] = "unknown"
        with self.assertRaises(ValueError):
            aggregate_controller_runs([bad], matched_baselines=4)


if __name__ == "__main__":
    unittest.main()
