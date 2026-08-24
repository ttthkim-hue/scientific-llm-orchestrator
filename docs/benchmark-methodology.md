# Benchmark methodology

The public fixture suite is balanced across coding, data/scientific mechanics,
visual/figure QA, and parametric CAD. Each fixture contains only synthetic
inputs, protected literals, and an expected deterministic result.

Every fixture is evaluated under all four exact architectures. The evaluator
checks route selection, contract construction, path scope, protected literal
preservation, allowlisted validation IDs, and fixture-specific arithmetic or
inventory. It does not execute proposed code, call a model, inspect private
data, or infer a scientific root cause.

The benchmark output is labelled `observed-deterministic`. Any future provider
run must keep quality, latency, tokens, local resources, fallback, privacy,
and reproducibility as separate fields. Actual-model results remain
`not-run` until measured on identical public fixtures.
