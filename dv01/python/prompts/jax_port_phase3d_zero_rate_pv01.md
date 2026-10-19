# Phase 3d — Zero-Rate PV01

## Objective
Calculate zero-rate PV01 by differentiating the JAX swap PV with respect to the
global curve parameter vector.

## Implementation

Create `pv01_jax/risk.py` with:

```python
zero_rate_pv01(problem, swap: SwapArrays, z) -> jax.Array
```

Return the gradient of:

```python
pv_swap(
    swap,
    curves_from(problem, z),
    problem.discount_curve,
    problem.forward_curve,
)
```

multiplied by `1e-4`, producing GBP per bp per curve node in `quote_labels` order.
Use `jax.grad`; do not port the hand-coded reverse pass or parameter-sensitivity
implementation.

Also add:

```python
as_labelled(problem, vector) -> dict[str, float]
```

Return values keyed by `problem.quote_labels` in their existing order.

## Tests

Add tests to `tests/jax_port/test_phase3_risk.py` using `unittest`. For each
`SETTINGS_SUFFIX`, compare `zero_rate_pv01(problem, swap_arrays, z0)` against the
hand-coded zero-rate sensitivities:

```python
provider.parameter_sensitivity(
    DiscountingSwapTradePricer.present_value_sensitivity(swap, provider)
)
```

Multiply the hand-coded parameter sensitivities by `1e-4`, sum or place them by curve
and node label, and compare every node on both curves with the JAX result to
`1e-6` GBP/bp. Use the calibrated provider and `z0` assembled in `quote_labels` order.

## Copilot Instructions

- Use only JAX differentiation for the JAX PV01 implementation.
- Keep the result aligned with `problem.quote_labels` across both curves.
- Use `as_labelled` for readable node-by-node comparison if appropriate.
- The existing hand-coded sensitivity path is the test oracle only.
- Do not import `pv01_pricer` or `pv01_market` from `pv01_jax`.

Do not modify any `pv01_*` packages other than the `pv01_jax` files explicitly in
scope. Do not modify existing tests; add or extend only
`tests/jax_port/test_phase3_risk.py`. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the results.
