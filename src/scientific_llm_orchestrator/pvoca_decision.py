from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchDecisionEvidence:
    # Scientific-evidence completeness
    unseen_items: int
    distinct_models: int
    task_families: int
    heldout_groups: int
    matched_baselines: int
    reproducible_runs: int
    paired_transition_items: int
    oracle_headroom_pp: float
    calibration_ece: float
    leakage_detected: bool = False

    # Offline controller performance
    accuracy_drop_upper_pp: float = 999.0
    token_savings: float = 0.0
    latency_savings: float = 0.0
    unsafe_stop_rate_upper: float = 1.0

    # Global shadow evidence; zero means not attempted.
    shadow_items: int = 0
    shadow_accuracy_drop_upper_pp: float = 999.0
    shadow_token_savings: float = 0.0
    shadow_latency_savings: float = 0.0
    shadow_unsafe_stop_rate_upper: float = 1.0
    shadow_crashes: int = 0
    owner_gate_bypasses: int = 0
    production_mutations: int = 0


@dataclass(frozen=True)
class ResearchDecisionGate:
    """Two-purpose gate: paper evidence vs. production absorption.

    A scientifically complete negative/null result may be PAPER_EVIDENCE_READY
    even when it is not useful enough for deployment. Global deployment is
    intentionally much stricter and requires a separate large shadow stage.
    """

    # Paper evidence: completeness/reproducibility, not guaranteed positive result.
    paper_min_items: int = 1000
    paper_min_models: int = 2
    paper_min_task_families: int = 2
    paper_min_heldout_groups: int = 5
    paper_min_baselines: int = 4
    paper_min_reproducible_runs: int = 2
    paper_min_transition_items: int = 30
    paper_max_calibration_ece: float = 0.10
    paper_min_oracle_headroom_pp: float = 2.0

    # Strong offline result before even entering global shadow.
    shadow_candidate_max_accuracy_drop_pp: float = 0.5
    shadow_candidate_min_token_savings: float = 0.15
    shadow_candidate_min_latency_savings: float = 0.10
    shadow_candidate_max_unsafe_stop_rate: float = 0.01

    # Global deploy requires much stronger live/shadow evidence.
    deploy_min_shadow_items: int = 10_000
    deploy_max_accuracy_drop_pp: float = 0.25
    deploy_min_token_savings: float = 0.20
    deploy_min_latency_savings: float = 0.15
    deploy_max_unsafe_stop_rate: float = 0.005

    def evaluate(self, evidence: ResearchDecisionEvidence) -> dict:
        if evidence.unseen_items < 0 or evidence.shadow_items < 0:
            raise ValueError("item counts must be non-negative")
        if evidence.distinct_models < 0 or evidence.task_families < 0:
            raise ValueError("model/task counts must be non-negative")
        for name in (
            "calibration_ece",
            "token_savings",
            "latency_savings",
            "unsafe_stop_rate_upper",
            "shadow_token_savings",
            "shadow_latency_savings",
            "shadow_unsafe_stop_rate_upper",
        ):
            value = float(getattr(evidence, name))
            if name.endswith("_rate_upper") or name == "calibration_ece":
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1]")

        paper_checks = {
            "sample_size": evidence.unseen_items >= self.paper_min_items,
            "multi_model": evidence.distinct_models >= self.paper_min_models,
            "multi_task": evidence.task_families >= self.paper_min_task_families,
            "group_holdout": evidence.heldout_groups >= self.paper_min_heldout_groups,
            "matched_baselines": evidence.matched_baselines >= self.paper_min_baselines,
            "reproducibility": evidence.reproducible_runs >= self.paper_min_reproducible_runs,
            "paired_transitions": evidence.paired_transition_items >= self.paper_min_transition_items,
            "oracle_headroom": evidence.oracle_headroom_pp >= self.paper_min_oracle_headroom_pp,
            "calibration": evidence.calibration_ece <= self.paper_max_calibration_ece,
            "no_leakage": not evidence.leakage_detected,
        }
        paper_ready = all(paper_checks.values())

        offline_checks = {
            "paper_evidence_ready": paper_ready,
            "accuracy_noninferiority": (
                evidence.accuracy_drop_upper_pp <= self.shadow_candidate_max_accuracy_drop_pp
            ),
            "token_savings": evidence.token_savings >= self.shadow_candidate_min_token_savings,
            "latency_savings": evidence.latency_savings >= self.shadow_candidate_min_latency_savings,
            "unsafe_stop_bound": (
                evidence.unsafe_stop_rate_upper <= self.shadow_candidate_max_unsafe_stop_rate
            ),
        }
        global_shadow_candidate = all(offline_checks.values())

        deploy_checks = {
            "offline_shadow_candidate": global_shadow_candidate,
            "shadow_sample_size": evidence.shadow_items >= self.deploy_min_shadow_items,
            "shadow_accuracy_noninferiority": (
                evidence.shadow_accuracy_drop_upper_pp <= self.deploy_max_accuracy_drop_pp
            ),
            "shadow_token_savings": evidence.shadow_token_savings >= self.deploy_min_token_savings,
            "shadow_latency_savings": evidence.shadow_latency_savings >= self.deploy_min_latency_savings,
            "shadow_unsafe_stop_bound": (
                evidence.shadow_unsafe_stop_rate_upper <= self.deploy_max_unsafe_stop_rate
            ),
            "shadow_crash_free": evidence.shadow_crashes == 0,
            "owner_gate_safe": evidence.owner_gate_bypasses == 0,
            "production_mutation_free": evidence.production_mutations == 0,
        }
        global_deploy_candidate = all(deploy_checks.values())

        # Negative/null scientific results are still publishable evidence when
        # the experimental package is complete. Positive-result labeling is
        # separated so deployment value is never inferred from paper readiness.
        positive_offline_result = (
            paper_ready
            and evidence.accuracy_drop_upper_pp <= 1.0
            and (
                evidence.token_savings >= 0.10
                or evidence.latency_savings >= 0.10
                or evidence.accuracy_drop_upper_pp < 0.0
            )
        )

        if global_deploy_candidate:
            status = "GLOBAL_DEPLOY_CANDIDATE"
        elif global_shadow_candidate:
            status = "GLOBAL_SHADOW_CANDIDATE"
        elif paper_ready and positive_offline_result:
            status = "PAPER_READY_POSITIVE"
        elif paper_ready:
            status = "PAPER_READY_NEGATIVE_OR_NULL"
        else:
            status = "RESEARCH_CONTINUE"

        return {
            "status": status,
            "paper_evidence_ready": paper_ready,
            "positive_offline_result": positive_offline_result,
            "global_shadow_candidate": global_shadow_candidate,
            "global_deploy_candidate": global_deploy_candidate,
            "paper_checks": paper_checks,
            "offline_shadow_checks": offline_checks,
            "deploy_checks": deploy_checks,
            "metrics": {
                "unseen_items": evidence.unseen_items,
                "oracle_headroom_pp": evidence.oracle_headroom_pp,
                "calibration_ece": evidence.calibration_ece,
                "accuracy_drop_upper_pp": evidence.accuracy_drop_upper_pp,
                "token_savings": evidence.token_savings,
                "latency_savings": evidence.latency_savings,
                "unsafe_stop_rate_upper": evidence.unsafe_stop_rate_upper,
                "shadow_items": evidence.shadow_items,
                "shadow_accuracy_drop_upper_pp": evidence.shadow_accuracy_drop_upper_pp,
                "shadow_token_savings": evidence.shadow_token_savings,
                "shadow_latency_savings": evidence.shadow_latency_savings,
                "shadow_unsafe_stop_rate_upper": evidence.shadow_unsafe_stop_rate_upper,
            },
        }
