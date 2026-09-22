# P-VoCA v3 Paper Skeleton

Working title options are intentionally conservative until preregistered results are available.

## Working title options

1. **P-VoCA: Marginal-Value Routing of Cognitive Actions for Cost-Aware Language Model Inference**
2. **When to Stop, Think, or Verify: Staged Cognitive-Action Routing for Language Models**
3. **Beyond Adaptive Thinking: Harm-Aware Routing Across Distinct Cognitive Operations**

Do not select the strongest title until cross-model/cross-task results are known.

## Abstract slots

### Motivation
Static inference applies the same workflow to heterogeneous queries, while adaptive-compute methods mainly decide how much reasoning to allocate.

### Gap
Distinct cognitive operations such as direct answering, additional reasoning, and independent verification can have different benefits and risks. In particular, verification may rescue an error or damage an already-correct answer.

### Method
P-VoCA models the marginal value of the next observable cognitive action in a staged STOP -> THINK -> VERIFY controller. It uses only the query and outputs observable up to the current stage, not hidden chain-of-thought.

### Evaluation
Fill only after preregistered runs:
- benchmarks: [MatSciBench, SciBench]
- models: [Qwen scales, Gemma extension if completed]
- held-out groups: [N]
- unique items: [N]
- matched baselines: [N]

### Result
Do not fill from the 108-item pilot.
Primary slots:
- paired accuracy difference / non-inferiority bound,
- token saving with 95% CI,
- latency saving with 95% CI,
- Oracle headroom,
- VERIFY rescue/harm rates,
- calibration.

### Conclusion
Positive path: staged action-value routing preserves quality while reducing unnecessary cognition across held-out groups/models.
Negative/null path: heterogeneous action value exists/does not generalize, and lightweight routing has identifiable limits.

## 1. Introduction

### Problem
Inference workflows differ in cost and capability:
- direct answer,
- extra reasoning,
- independent verification,
- later extension: memory retrieval.

### Key distinction
P-VoCA is not primarily a reasoning-token allocator.
It asks whether a different *kind* of cognitive operation has sufficient marginal value.

### Contributions — placeholders
1. Formalize staged marginal value for heterogeneous cognitive actions.
2. Introduce harm-aware VERIFY routing with separate rescue/harm modeling.
3. Provide a hidden-CoT-free staged evaluation protocol with incremental cost accounting.
4. Evaluate group-held-out and cross-model generalization.
5. Release reproducibility receipts and preregistered decision gates.

Only retain contribution claims supported by final results.

## 2. Related Work

### Adaptive reasoning
- RPO
- SAT

### Test-time routing/alignment
- TARo

### Retrieval + reasoning routing
- X-Router

### Selective/adaptive verification
- Adaptive Generate-Rank-Verify and related verifier-routing work

### Personalized inference routing
Reserved for the later RECALL/user-utility extension.

End section with exact novelty boundary:
P-VoCA targets operation-specific marginal value and verification harm, not adaptive compute alone.

## 3. Problem Formulation

State:
- query q,
- current observable state s_t,
- candidate next action a_t,
- action cost C(a_t),
- latency T(a_t),
- expected quality delta ΔQ(a_t | s_t),
- action harm probability.

Utility template:
U(a | s) = E[ΔQ(a | s)] - λ C(a) - μ T(a) - ρ Risk(a | s)

Current action ladder:
STOP -> THINK -> VERIFY.

Later extension:
RECALL as a separate operation.

## 4. P-VoCA v3

### 4.1 Observable staged trace
STOP:
- fast numeric answer.

THINK:
- observable compact calculation scaffold,
- corrected numeric answer.

VERIFY:
- independent audit note,
- corrected numeric answer.

No hidden chain-of-thought required.

### 4.2 Gate-S
Predict whether downstream cognition can rescue a STOP error.
Fail closed: no statistically safe threshold => STOP shortcut disabled.

### 4.3 Gate-V
Separate models:
- P(VERIFY rescues THINK),
- P(VERIFY harms THINK).

Invoke VERIFY only when rescue-harm margin exceeds a train-only threshold.

### 4.4 Cost accounting
Incremental sequential cost:
- STOP,
- STOP + THINK increment,
- STOP + THINK + VERIFY increment.

### 4.5 Decision safety
Offline paper gate,
global shadow gate,
global deploy-candidate gate.
No benchmark result automatically authorizes deployment.

## 5. Experimental Setup

### 5.1 Benchmarks
MatSciBench:
- numeric/text/single-scalar subset,
- primary_category group holdout.

SciBench:
- solution-free,
- deduplicated,
- numeric subset,
- source/textbook group holdout.

### 5.2 Models
Minimum:
- qwen3.8:27b,
- qwen3.5:4b.

Strong extension:
- gemma3:27b.

Exact digest/quantization/GPU receipts required.

### 5.3 Baselines
1. Always STOP.
2. Always THINK.
3. Always VERIFY.
4. Question-only learned router.
5. STOP/THINK numerical-disagreement VERIFY heuristic.
6. Add a faithful adaptive-reasoning baseline if implementation/reproducibility permits.

### 5.4 Metrics
Quality:
- accuracy,
- paired correctness transitions,
- exact McNemar p.

Efficiency:
- total tokens,
- total latency,
- paired bootstrap 95% CI for savings.

Mechanism:
- correctness pattern counts,
- Oracle headroom,
- VERIFY rescue rate,
- VERIFY harm rate.

Reliability:
- ECE,
- unsafe STOP Wilson upper bound.

### 5.5 Statistical protocol
Use preregistered thresholds in PVOCA_PREREGISTRATION_V3.md.

## 6. Results

### Table 1 — Main results
Rows:
model × benchmark.
Columns:
fixed best accuracy,
question router,
agreement router,
P-VoCA accuracy,
token saving,
latency saving,
ECE,
Oracle headroom.

### Figure 1 — Quality vs cost Pareto
x: tokens or latency.
y: accuracy.
Points:
STOP / THINK / VERIFY / question router / agreement router / P-VoCA / Oracle.

### Figure 2 — Cognitive-action transition map
Stacked correctness patterns:
000, 001, 010, 011, 100, 101, 110, 111.

### Figure 3 — VERIFY rescue vs harm
Per benchmark/model:
- rescue,
- harm,
- neutral.

### Figure 4 — Calibration
Gate-S, Gate-V rescue, Gate-V harm reliability curves/ECE.

### Table 2 — Held-out group generalization
Per fold:
accuracy, savings, selected action shares.

### Table 3 — Cross-family transfer
Qwen vs Gemma result consistency.

## 7. Ablations

1. Remove STOP observable features.
2. Remove THINK work-note features.
3. Remove rescue/harm split; predict a single VERIFY value.
4. Replace learned Gate-V with disagreement heuristic.
5. Flat 3-way classifier vs staged cascade.
6. Cost-neutral λ/threshold sensitivity.
7. Optional: calibration method sensitivity.

## 8. Failure Analysis

Mandatory categories:
- all actions wrong,
- STOP-only,
- THINK-only,
- VERIFY-only,
- VERIFY rescue,
- VERIFY harm,
- numerical parsing failures,
- unit/sign/order-of-magnitude failures.

Do not cherry-pick only successful examples.

## 9. Discussion

Questions:
- Is cognitive-action heterogeneity stable across model families?
- Does a stronger base model reduce or increase Oracle headroom?
- Is VERIFY value predictable before paying full verification cost?
- When does cost saving come mainly from recognizing hopeless items?
- How much of the effect is benchmark-specific?

## 10. Limitations

Must include:
- public benchmark contamination risk,
- scientific-numeric focus,
- limited action repertoire,
- local-model dependence,
- no personalization in v3,
- no production safety claim from offline results,
- explicit observable work notes are not equivalent to hidden internal reasoning.

## 11. Future Work

Only after v3:
- RECALL,
- user-conditioned utility,
- long-horizon preference adaptation,
- non-numeric/general QA,
- real shadow deployment if gates pass.

## Appendix

- complete prompt/action templates,
- exact model receipts,
- dataset hashes,
- category/source folds,
- preregistration,
- baseline threshold-selection rules,
- all per-fold results,
- all action-transition counts.
