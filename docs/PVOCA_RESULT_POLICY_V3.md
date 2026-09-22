# P-VoCA v3 Result Policy

Status: isolated research only. No V4.3.10, V3 LKG, Jev, or production mutation is authorized by this document.

## Decision rule

P-VoCA has two independent success paths.

### Path A — Publication path

A methodologically complete experiment may be publishable even when the controller is not useful enough for deployment.

Minimum evidence target:
- at least 1,000 unique held-out items across benchmarks,
- at least 2 models,
- at least 2 task families,
- at least 5 held-out groups,
- at least 4 matched baselines,
- repeated runs,
- calibration reported,
- no train/test leakage.

Possible outcomes:
- PAPER_READY_POSITIVE
- PAPER_READY_NEGATIVE_OR_NULL

A negative/null result remains scientifically usable if the protocol is complete and the limitations are reported honestly.

### Path B — Global absorption path

Offline success is not enough for production.

First gate: GLOBAL_SHADOW_CANDIDATE
- paper evidence complete,
- heterogeneous cognitive-action signal present,
- paired 95% upper bound on accuracy degradation <= 0.5 percentage points,
- token saving >= 15%,
- latency saving >= 10%,
- unsafe STOP upper bound <= 1%.

Second gate: GLOBAL_DEPLOY_CANDIDATE
- at least 10,000 shadow decisions,
- paired 95% upper bound on accuracy degradation <= 0.25 percentage points,
- token saving >= 20%,
- latency saving >= 15%,
- unsafe STOP upper bound <= 0.5%,
- zero crashes,
- zero owner-gate bypasses,
- zero production mutations.

GLOBAL_DEPLOY_CANDIDATE is only a candidate state. Actual deployment remains an explicit owner decision.

## Interpretation

Materials-engineering analogy:

- PAPER_READY means the material system has been characterized well enough to publish the mechanism, even if it is not industrially useful.
- GLOBAL_SHADOW_CANDIDATE means the formulation looks strong enough for a pilot line.
- GLOBAL_DEPLOY_CANDIDATE means the pilot line stayed stable under a much larger validation campaign and is ready for an owner-controlled scale-up decision.

## Current state

The 108-item v0 pilot is motivation/mechanism evidence only. It is not v3 staged evidence and cannot satisfy either publication-completeness or deployment gates by itself.

The next scientific milestone is the isolated v3 matrix:
- MatSciBench × qwen3.8:27b
- MatSciBench × qwen3.5:4b
- SciBench × qwen3.8:27b
- SciBench × qwen3.5:4b

For a stronger submission, add at least one second model family after the minimum matrix.
