# P-VoCA v3 Paper Positioning and Evidence Gate

Status: isolated research branch only. No production or V4 integration.

## One-sentence problem

A frozen language model should not apply the same cognitive workflow to every query. P-VoCA asks whether the next cognitive action is worth its expected quality gain, latency, token cost, and risk of destroying an already-correct answer.

## Current v3 action ladder

1. STOP — use the fast answer.
2. THINK — pay for an additional reasoning stage.
3. VERIFY — pay for an independent audit and correction stage.
4. RECALL — reserved for the personalization/memory phase; not part of the current MatSciBench pilot.

The controller is sequential rather than a flat 3-way classifier. Gate-S asks whether escalating beyond STOP can rescue the item. Gate-V asks whether VERIFY is more likely to rescue THINK than to harm it.

## What the 108-item pilot has established

Observed on a Qwen3.8 27B MatSciBench numeric pilot:

- STOP accuracy: 12.96%.
- THINK accuracy: 40.74%.
- VERIFY accuracy: 42.59%.
- Corrected full Oracle accuracy: 50.00%.
- Accuracy headroom over best fixed action: +7.41 percentage points.
- Correctness patterns include THINK-only, VERIFY-only, STOP-only, and all-correct cases.
- 54/108 items were unsolved by every tested action.

Interpretation: heterogeneous action value exists, but the base model and benchmark slice impose a large ceiling.

## Important negative result

The v0-v2 learned controllers did not beat the best fixed policy on the observed pilot. The best v2 operating point remained below always-THINK/always-VERIFY accuracy.

This is a useful result, not a failure to hide: action heterogeneity is present, but naive question-only or lightweight draft-feature routing does not recover it reliably from 108 samples.

## Closest prior-art risk

### Adaptive reasoning / overthinking
Recent work already shows that extra reasoning can have diminishing returns or destroy correct answers, and learns whether/how long to reason.

Therefore P-VoCA must not claim novelty from "reason only when needed."

### Multi-axis routing
X-Router profiles Direct/RAG/CoT/RAG+CoT and learns a compact cost-quality router.

Therefore P-VoCA must not claim novelty from "choose among several inference pipelines under cost."

### Token/step-level routing
TARo and SAT adapt reasoning at finer granularity.

Therefore P-VoCA should not compete primarily on token-level control.

## Defensible differentiation to test

P-VoCA should be evaluated as **marginal value routing across heterogeneous cognitive actions**, not merely adaptive compute:

- action-specific rescue value,
- action-specific harm value,
- sequential fail-closed decisions,
- explicit verification as a distinct cognitive operation,
- later retrieval/RECALL as a distinct operation,
- later user-conditioned utility rather than one global cost weight,
- no requirement for hidden chain-of-thought.

The strongest scientific claim would be:
"Different cognitive operations have query- and user-dependent marginal value, and a calibrated controller can select them under a quality-preserving budget."

This claim is not yet proven.

## Required evidence before a serious paper submission

### Dataset scale
- At least 1,000 unseen staged traces.
- No duplicate inflation.
- Category-held-out evaluation for MatSciBench.
- Add a second benchmark if MatSciBench eligible numeric items are insufficient.

### Generalization
- At least two frozen backbone models.
- At least two task families.
- Report in-domain and group/domain-held-out results separately.

### Baselines
- always STOP,
- always THINK,
- always VERIFY,
- simple difficulty classifier,
- question-only router,
- draft-confidence router,
- adaptive-reasoning baseline where reproducible,
- multi-axis routing baseline or a faithful simplified analogue.

### Core metrics
- accuracy,
- token count,
- latency,
- accuracy at matched cost,
- cost at matched accuracy,
- paired error transitions,
- rescue rate,
- harm rate,
- calibration,
- Pareto frontier.

### Safety/non-inferiority gate
A production-oriented result should remain shadow-only unless:
- unseen n >= 1,000,
- paired 95% upper bound on accuracy degradation <= 0.5 percentage points,
- token savings >= 15%,
- unsafe STOP 95% upper bound <= 1%,
- zero owner-gate bypasses,
- zero production mutations,
- zero crashes in the evaluation window.

## Paper-readiness interpretation

### Current 108-item pilot
Evidence level: feasibility / mechanism-discovery pilot.

Suitable contribution:
- internal technical report,
- workshop-style pilot,
- ablation/motivation section of a later paper.

Not sufficient by itself for a strong full paper because:
- n=108 is small,
- one model,
- one narrow benchmark slice,
- half the items are unsolved by all actions,
- learned controller does not yet beat the fixed baseline.

### Strong v3 result
A credible full paper becomes plausible if the staged controller:
- preserves or improves accuracy versus the best fixed workflow,
- saves >=15-20% tokens/latency,
- generalizes to held-out categories,
- reproduces on >=2 models and >=2 task families,
- shows statistically meaningful VERIFY rescue/harm prediction.

### Strong v4 research result
A substantially stronger paper would add RECALL and user-conditioned utility, showing that the same query can rationally trigger different cognitive actions for different users/budgets while preserving factual performance.

That stage is the clearest separation from ordinary adaptive-compute routing.

## Publication claim discipline

Do not write:
- "P-VoCA outperforms adaptive reasoning" before a matched baseline exists.
- "Personalized metacognition" before user-conditioned experiments exist.
- "safer" without paired risk measurements.
- "human-like" as a scientific conclusion.

Prefer:
- "staged marginal-value controller,"
- "heterogeneous cognitive-action routing,"
- "quality-preserving cost control,"
- "user-conditioned extension" only after the personalization phase.
