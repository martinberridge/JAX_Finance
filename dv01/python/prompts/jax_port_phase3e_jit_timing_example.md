# Phase 3e — JIT, Timing, and Zero-Rate PV01 Example

## Objective
Verify that `zero_rate_pv01` compiles once across changing curve parameters, report
JIT execution timings, and provide a runnable node-by-node comparison example.

## JIT and Timing Test

In `tests/jax_port/test_phase3_risk.py`, add a `unittest` test that:

1. Wraps `zero_rate_pv01` in `jax.jit`.
2. Counts traces by appending to a Python list inside the jitted function.
3. Calls the jitted function three times with different `z` values for the same
   settings suffix, synchronizing each result with `block_until_ready()`.
4. Asserts that only one trace occurred.
5. For each settings suffix, warms up the jitted function, then measures 100 calls,
   synchronizing each result with `block_until_ready()`. Print the median wall time for
   each suffix. Do not assert on timing values.

Report the median timings.

## Example Script

Create:

```text
pv01_examples/jax/__init__.py
pv01_examples/jax/zero_rate_pv01_example.py
```

Keep `__init__.py` empty. The example script should, for each `SETTINGS_SUFFIX`:

1. Calibrate with the hand-coded `RatesCurveCalibrator`.
2. Build the calibration problem and resolved swap arrays.
3. Calculate the hand-coded zero-rate PV01 using
   `provider.parameter_sensitivity(DiscountingSwapTradePricer.present_value_sensitivity(...))`.
4. Calculate the JAX zero-rate PV01 using `zero_rate_pv01`.
5. Print one table per settings suffix with columns:
   - node label
   - hand-coded zero-rate PV01
   - JAX zero-rate PV01
   - difference

Report the largest absolute difference for each settings suffix.

## Copilot Instructions

- Keep the benchmark diagnostic only; add no performance assertions.
- Synchronize warm-up and timed results with `block_until_ready()`.
- Preserve quote/node label order in the comparison output.
- Do not import `pv01_pricer` or `pv01_market` from `pv01_jax`; production code is
  used only by the example as a reference.

Do not modify any `pv01_*` packages other than the `pv01_jax` files explicitly in
scope. Do not modify existing tests; add or extend only
`tests/jax_port/test_phase3_risk.py`. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the results.
