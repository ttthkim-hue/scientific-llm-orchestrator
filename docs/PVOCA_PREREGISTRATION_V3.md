# P-VoCA v3 Experimental Preregistration

Date frozen: 2026-09-22

This document is frozen before the large v3 staged-trace campaign. It defines the primary hypotheses, evaluation boundaries, and interpretation rules so that thresholds are not selected after observing held-out results.

## Research question

Can a lightweight staged controller estimate the marginal value of heterogeneous cognitive actions (STOP, THINK, VERIFY) well enough to preserve answer quality while reducing inference cost on unseen scientific problem groups?

## Primary hypotheses

### H1 — Cognitive-action heterogeneity exists
Across each benchmark/model run, there are non-trivial correctness transitions among STOP, THINK, and VERIFY rather than one action dominating every solvable item.

Primary mechanism summaries:
- STOP-only count,
- THINK-only count,
- VERIFY-only count,
- THINK-to-VERIFY rescue count,
- THINK-to-VERIFY harm count,
- all-correct count,
- all-wrong count,
- Oracle headroom over the best fixed action.

A mechanism signal is considered practically meaningful for the multi-run package when:
- the conservative aggregate Oracle headroom is at least 2.0 percentage points, and
- at least 30 heterogeneous transition items remain after conservative aggregation.

### H2 — Staged marginal-value routing adds value beyond simple routing
The v3 controller is compared against all of:
1. Always STOP,
2. Always THINK,
3. Always VERIFY,
4. question-only learned router,
5. STOP-vs-THINK numerical-disagreement VERIFY heuristic.

No baseline may use held-out labels to select its test-time threshold.

### H3 — Positive paper result
A positive result requires all methodological completeness checks plus:
- conservative multi-run paired accuracy degradation upper bound <= 1.0 percentage point,
- and at least one of:
  - conservative token saving >= 10%, or
  - conservative latency saving >= 10%, or
  - conservative accuracy improvement (negative degradation),
- mechanism signal present,
- maximum held-out calibration ECE <= 0.10.

Failure of H3 does not invalidate a methodologically complete negative/null paper.

## Strong deployment-relevant result

This is intentionally stricter than H3.

Offline GLOBAL_SHADOW_CANDIDATE requires:
- paired 95% upper bound on accuracy degradation <= 0.5 percentage points,
- token saving >= 15%,
- latency saving >= 10%,
- unsafe STOP upper bound <= 1%,
- mechanism signal present.

Actual GLOBAL_DEPLOY_CANDIDATE additionally requires a separate >=10,000-decision shadow campaign with:
- paired 95% upper bound on accuracy degradation <= 0.25 percentage points,
- token saving >= 20%,
- latency saving >= 15%,
- unsafe STOP upper bound <= 0.5%,
- zero crashes,
- zero owner-gate bypasses,
- zero production mutations.

Deployment itself remains an explicit owner decision and is not authorized by benchmark results.

## Minimum experiment matrix

Primary minimum:
- MatSciBench × qwen3.8:27b
- MatSciBench × qwen3.5:4b
- SciBench × qwen3.8:27b
- SciBench × qwen3.5:4b

Strong-paper extension:
- MatSciBench × gemma3:27b
- SciBench × gemma3:27b

The Qwen pair tests scale transfer. Gemma 3 adds a separate model family.

## Split policy

### MatSciBench
- text-only,
- NUM,
- single answer,
- parseable single scalar reference,
- group-held-out by primary_category.

### SciBench
- solution field excluded,
- deduplicated by stable scientific content,
- numeric answer only,
- group-held-out by source/textbook.

Group assignments are deterministic and fixed before controller training.

## Feature boundary

Allowed:
- question text,
- STOP observable output/statistics,
- THINK observable work note/output/statistics,
- model-independent structural features,
- relative agreement/disagreement available at the current stage.

Forbidden as controller features:
- reference answer,
- solution text,
- benchmark difficulty label,
- benchmark domain/category label,
- hidden chain-of-thought,
- future-stage output not yet executed.

## Cost accounting

All v3 costs are sequential incremental costs.

- STOP cost = STOP stage.
- THINK cost = STOP + incremental THINK stage.
- VERIFY cost = STOP + incremental THINK + incremental VERIFY stage.

Independent v0 action totals must not be reused as v3 sequential costs.

Primary efficiency measures:
- total tokens,
- total latency,
- token saving against the selected best fixed workflow,
- latency saving against the selected best fixed workflow,
- matched-cost and matched-accuracy summaries.

## Statistical policy

- Paired item-level correctness transitions are the basis for non-inferiority.
- Paired accuracy differences also report a two-sided exact McNemar p-value.
- Token and latency savings report item-paired percentile-bootstrap 95% confidence intervals.
- 95% uncertainty bounds are reported.
- Unsafe STOP risk uses a Wilson upper bound.
- Gate calibration reports held-out ECE.
- Multi-run aggregation is conservative:
  - worst accuracy degradation,
  - lowest token saving,
  - lowest latency saving,
  - worst ECE,
  - worst unsafe-STOP bound,
  - lowest Oracle headroom.
- Repeating the same benchmark under multiple models does not multiply the unique-item count.

## Stopping/falsification rules

Do not promote the mechanism as a positive result if:
- conservative Oracle headroom < 2 percentage points,
- heterogeneous transitions are too rare,
- learned v3 cannot improve on simple question-only/disagreement routing,
- cross-family replication removes the effect,
- calibration remains poor,
- apparent cost gains require unacceptable accuracy loss.

Do not change the primary thresholds above after looking at held-out results. Any exploratory post-hoc analysis must be labeled exploratory.

## Reporting policy

Report both favorable and unfavorable action transitions.

In particular, VERIFY must be reported as:
- rescue,
- harm,
- neutral.

Do not report only examples where verification helps.

The 108-item v0 pilot is explicitly labeled exploratory/pilot evidence and is not pooled into the preregistered v3 primary estimates.
