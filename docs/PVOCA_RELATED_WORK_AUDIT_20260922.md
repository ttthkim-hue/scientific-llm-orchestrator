# P-VoCA Related-Work Audit — 2026-09-22

This note fixes the novelty boundary for the isolated P-VoCA research lane. It is not a claim that novelty has already been established.

## Closest 2026 work

### RPO — ACL 2026 Long, 2026.acl-long.816
Root-token Policy Optimization trains a language model to decide whether to enter reasoning mode. It directly addresses overthinking and adaptive think/no-think decisions across model families and scales.

Implication for P-VoCA:
- "reason only when needed" is not novel.
- Binary STOP/THINK routing is not enough.

### SAT — ACL 2026 Long, 2026.acl-long.2009
Stepwise Adaptive Thinking uses a process reward model and multiple thinking modes to prune/expand reasoning at the step level.

Implication for P-VoCA:
- fine-grained adaptive compute is already strong prior art.
- P-VoCA should not compete primarily on token/step budget control.

### TARo — Findings ACL 2026, 2026.findings-acl.50
TARo learns token-level routing for test-time alignment and demonstrates cross-domain and cross-scale transfer.

Implication for P-VoCA:
- frozen-model inference-time routing is not novel by itself.
- cross-domain and cross-model evidence is expected for a strong paper.

### X-Router — Findings ACL 2026, 2026.findings-acl.994
X-Router independently routes retrieval necessity and reasoning necessity among Direct, RAG, CoT, and RAG+CoT using cost-quality supervision and lightweight online probes.

Implication for P-VoCA:
- multi-axis inference routing under token/latency cost is already established.
- adding RECALL to THINK without a different scientific formulation would be too close.

### Adaptive Generate-Rank-Verify — arXiv:2605.17609
This work studies inference-time search when verification is costly and adaptively decides how much expensive verification to purchase.

Implication for P-VoCA:
- "verification is expensive, use it selectively" is not sufficient novelty.
- P-VoCA must model the *marginal value and harm* of verification relative to an already-produced answer, not just verifier scheduling.

## Defensible P-VoCA scientific hypothesis

The target hypothesis is:

> Distinct cognitive operations have different query-dependent marginal values and risks. A staged controller can estimate whether the next operation is worth invoking while preserving answer quality under cost constraints.

The current v3 operations are:
- STOP: keep the fast answer.
- THINK: add an observable structured recomputation stage.
- VERIFY: add an independent audit/correction stage.
- RECALL: reserved for the later memory/personalization phase.

## Required differentiation

A strong P-VoCA result should demonstrate all of the following:

1. **Operation heterogeneity**
   - STOP-only, THINK-only, VERIFY-only, rescue, harm, and neutral patterns occur at meaningful rates.

2. **Harm-aware VERIFY routing**
   - predict VERIFY rescue and VERIFY harm separately.
   - show advantage over an agreement/disagreement heuristic.

3. **Sequential marginal value**
   - decisions use only information observable up to the current stage.
   - staged incremental tokens and latency are measured directly.

4. **Quality-preserving cost control**
   - paired non-inferiority, matched-cost, and matched-accuracy reporting.
   - no favorable averaging that hides a weak model or benchmark.

5. **Cross-domain and cross-model generalization**
   - group-held-out benchmark splits.
   - at least two model scales and, for a stronger paper, at least two model families.

6. **No hidden-chain-of-thought dependency**
   - controller features are limited to the query and explicitly observable stage outputs/statistics.

7. **Personalization only as a later extension**
   - user-conditioned utility/RECALL must be tested separately from the base metacognitive controller.

## Claims to avoid

Do not claim novelty from:
- adaptive reasoning alone,
- routing among multiple inference pipelines alone,
- selective verification alone,
- token/latency savings alone,
- personalization without user-conditioned experiments.

Do not claim production safety from offline benchmark performance.

## Strong-paper falsification conditions

The central mechanism should be considered weak if, after >=1,000 unique held-out items:
- one fixed action dominates almost all solvable items,
- Oracle headroom collapses below 2 percentage points across runs,
- VERIFY rescue/harm transitions are too rare to model,
- the learned controller cannot beat simple question-only or disagreement baselines,
- cross-family replication eliminates the effect.

These outcomes can still support a rigorous negative/null paper if the experimental package is complete.
