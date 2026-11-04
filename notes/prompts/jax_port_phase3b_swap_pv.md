# Phase 3b — JAX Swap PV

## Objective
Implement a pure-JAX swap present-value function and validate it against the existing
production pricer.

## Implementation

Create `pv01_jax/pricers.py` with:

```python
pv_swap(
    swap: SwapArrays,
    curves: dict[str, Curve],
    discount: str,
    forward: str,
) -> jax.Array
```

Implement:

```text
sign * (
    sum over float periods of notional * forward_rate * tau * discount_factor
    - sum over fixed periods of notional * fixed_rate * tau * discount_factor
)
```

Use `curves[forward].forward(t_fix_start, t_fix_end, tau_fix)` for the floating rates,
and `curves[discount].df(t_pay)` for discount factors. Use vectorized `jax.numpy`
operations; do not loop over periods.

Before implementing, read:

- `pv01_pricer/swap_pricer.py`
- `pv01_pricer/instrument_pricers.py`
- `pv01_pricer/rates_provider.py`

Confirm that this formula matches the current production code. If it differs, follow
production behavior and report the difference.

## Tests

Create `tests/jax_port/test_phase3_pricers.py` using `unittest`. For each
`SETTINGS_SUFFIX`:

1. Calibrate the existing provider and resolve the example swap as in
   `tests/test_reference_correctness.py`.
2. Build `problem`, `z0` from the calibrated curve `y_values` in
   `quote_labels` order, `swap_arrays` with `resolve_swap`, and `curves` with
   `curves_from`.
3. Assert that:

```python
pv_swap(
    swap_arrays,
    curves,
    problem.discount_curve,
    problem.forward_curve,
)
```

matches `DiscountingSwapTradePricer.present_value(swap, provider)["GBP"]` to
`1e-6` GBP.
4. Parse every Python file under `pv01_jax` with `ast` and assert that no `Import` or
   `ImportFrom` imports `pv01_pricer` or `pv01_market`.

## Copilot Instructions

- Use only `jax.numpy` in the pricer implementation.
- Do not import from `pv01_pricer` or `pv01_market`.
- Do not loop over periods.
- Match the current production formula exactly; report any discrepancy found before
  implementing.
- Keep the test in `unittest` style and use the hand-coded pricer only as the test
  oracle.

Do not modify any `pv01_*` packages other than the `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only
`tests/jax_port/test_phase3_pricers.py`. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the results.
