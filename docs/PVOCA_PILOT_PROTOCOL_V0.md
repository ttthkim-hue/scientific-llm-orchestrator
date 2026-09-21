# P-VoCA Oracle v0 Pilot Protocol

Status: pre-registered pilot protocol for the independent Adaptive Metacognition research lane.

## Scope

The frozen language model is not trained. For each public MatSciBench item, the pilot compares three observable cognitive actions and records quality, token use, latency, and calls.

No Web-P8 V3/V4 control-plane policy is modified by this experiment.

## Public benchmark slice

- Source: MatSciBench public test split.
- Eligible items: text-only, NUM type, single-answer, scalar numeric reference.
- Pilot size: 108.
- Difficulty balance: 36 easy, 36 medium, 36 hard.
- Within each difficulty, primary categories are sampled round-robin to maximize category coverage.
- Deterministic seed: 20260921.
- Correctness: numeric answer within 5% relative error; zero-reference answers use a small absolute tolerance.

The reference solution/explanation is never provided to the model.

## Actions

### STOP

One model call. Solve the problem and return one numeric value.

### THINK

Two model calls.

1. Produce a compact calculation scaffold: relevant formula, converted given values, and rough result.
2. Recompute independently, correct any scaffold error, and return one numeric value.

This deliberately uses an observable work note instead of depending on hidden chain-of-thought.

### VERIFY

Three accounted calls, including the STOP call.

1. Generate the STOP answer.
2. Independently audit the proposed answer for governing formula, substitution, unit conversion, sign, and interpretation errors.
3. Recompute using the audit and return the corrected numeric value.

## Oracle

For each item, among correct actions choose the least expensive action by:

1. total input + output tokens,
2. latency,
3. number of calls,
4. deterministic tie break: STOP, VERIFY, THINK.

If no action is correct, the item is marked unsolved.

## Primary pilot outputs

- action accuracy for STOP / THINK / VERIFY,
- Oracle accuracy,
- Oracle action counts,
- input/output/total tokens,
- end-to-end latency,
- token savings relative to always-THINK,
- per-item result records for downstream controller training.

## Controller boundary

The first controller baseline may use only information observable before the selected cognitive action, initially question-only features.

It must not use:
- benchmark difficulty labels,
- benchmark domain labels,
- reference solutions,
- reference answers,
- hidden model reasoning traces.

The purpose of the pilot is to test whether action heterogeneity exists before adding personalization and RECALL.
