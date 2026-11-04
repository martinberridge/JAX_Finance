# Phase 5a — Market-Quote PV01

## Objective
Calculate swap PV directly from market quotes through implicit calibration, and
differentiate it to produce market-quote PV01.

## Implementation

In `pv01_jax/risk.py`, add:

```python
pv_from_quotes(problem, swap: SwapArrays, q) -> scalar
market_quote_pv01(problem, swap: SwapArrays, q) -> array
```

Define `pv_from_quotes` as:

```python
pv_swap(
    swap,
    curves_from(problem, calibrate(problem, q)),
    problem.discount_curve,
    problem.forward_curve,
)
```

Define `market_quote_pv01` as:

```python
jax.grad(pv_from_quotes, argnums=2)(problem, swap, q) * 1e-4
```

Use `calibrate` from `pv01_jax/calibrate.py`, and use `as_labelled` from the
existing risk API when labeled output is helpful.

Before comparing to the golden MQS files, inspect `_parse_mqs` in
`tests/test_calibration_interpolation_example.py` and `export_mqs` in
`pv01_examples/export_utils.py` to confirm how sensitivities are scaled and
which units the CSV values represent.

## Tests

Add tests to `tests/jax_port/test_phase5_risk.py` using `unittest`, for each
`SETTINGS_SUFFIX`:

1. Compare `market_quote_pv01(problem, swap_arrays, q0)` with
   `tests/golden/<CONFIG>-<suffix>-mqs.csv` per label to `0.01 GBP/bp`.
   Also compare `pv_from_quotes(problem, swap_arrays, q0)` with the golden PV
   to `0.01 GBP`.
2. For each quote column `j`, compare the market-quote PV01 bucket with:

   ```python
   (
       pv_from_quotes(problem, swap_arrays, q0 + 1e-4 * e_j)
       - pv_from_quotes(problem, swap_arrays, q0 - 1e-4 * e_j)
   ) / 2
   ```

   to `0.01 GBP/bp`.
3. Verify `jax.jit(market_quote_pv01)` traces once across calls with `q0` and
   `q0 + 1e-4`, counting traces inside the jitted function. Warm up the compiled
   function, time 20 synchronized calls, and print their median duration.

## Copilot Instructions

- Keep production calibration/pricing imports out of `pv01_jax`; use them only
  as test oracles.
- Preserve the quote-label ordering in the returned PV01 values.
- Synchronize JIT warm-up and timed calls with `block_until_ready()`.
- Report timing for diagnosis only; do not add performance thresholds.
- Keep all new tests in `unittest` style.

Do not modify any `pv01_*` packages other than `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only
`tests/jax_port/test_phase5_risk.py`. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the
results.
