# Synthetic benchmark

The benchmark is a small public fixture suite for contract, route, protected-
literal, data-mechanics, visual-label, and parametric-CAD checks. It does not
call a model and does not measure scientific validity.

Run it from the repository root:

```text
python scripts/run_synthetic_benchmark.py --root .
```

The four reported architectures are exactly `FRONTIER_ONLY`,
`FRONTIER_HOSTED_WORKER`, `FRONTIER_LOCAL_WORKER`, and
`FRONTIER_LOCAL_THEN_HOSTED_RESIDUAL`. The result is
`observed-deterministic`; actual-model evidence is `not-run`.
