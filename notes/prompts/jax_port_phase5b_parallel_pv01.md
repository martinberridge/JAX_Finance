# Phase 5b — Parallel PV01 and Hand-Coded Bump Validation

## Objective
Calculate parallel market-quote PV01 with one forward-mode JAX pass and validate
both the parallel result and individual buckets against existing reference
calculations.

## Implementation

In `pv01_jax/risk.py`, add:

```python
parallel_pv01(problem, swap, q) -> scalar
```

Implement it using the directional derivative of `pv_from_quotes` along a
vector of ones:

```python
jax.jvp(
    lambda q: pv_from_quotes(problem, swap, q),
    (q,),
    (jnp.ones_like(q),),
)[1] * 1e-4
```

Use `pv_from_quotes` and `market_quote_pv01` from Phase 5a.

## Tests

Add tests to `tests/jax_port/test_phase5_risk.py` using `unittest`, for each
`SETTINGS_SUFFIX`:

1. Assert `parallel_pv01(problem, swap_arrays, q0)` equals
   `market_quote_pv01(problem, swap_arrays, q0).sum()` to `1e-8`.
2. Add a hand-coded bump comparison gated with:

   ```python
   @unittest.skipUnless(os.environ.get("RUN_SLOW"), "slow bump-and-recalibrate test")
   ```

   For each quote label, compare the JAX market-quote PV01 bucket with
   `tests/util/bump.py`'s `bump_pv01(config, market_data, swap, VALUATION_DATE)`
   result to `0.01 GBP/bp`. Cache bump results per settings suffix with
   `lru_cache`; the natural-cubic-spline configuration is slow.

## Copilot Instructions

- Keep the benchmark and bump comparison as validation only.
- Use `bump_pv01` only as a test oracle; do not import production pricer code
  into `pv01_jax`.
- Cache per-suffix hand-coded bump calculations to avoid duplicate work.
- Keep tests in `unittest` style.

Do not modify any `pv01_*` packages other than `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only
`tests/jax_port/test_phase5_risk.py`. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the
results. The hand-coded bump test is opt-in with `RUN_SLOW=1`.
