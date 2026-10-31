# Phase 6c — Frozen JAX Goldens

## Objective
Freeze the JAX market-quote PV01, PV and cross-gamma diagonal per settings
suffix, and regression-test the JAX port against them.

## Implementation

Create:

```text
pv01_examples/jax/write_jax_goldens.py
tests/jax_port/golden_jax/<suffix>.csv
```

The script, for each suffix in `SETTINGS_SUFFIXES`:

1. Builds `problem` and `swap_arrays` as in earlier phases, with
   `q0 = problem.q0`.
2. Computes `market_quote_pv01`, `pv_from_quotes` and the diagonal of
   `cross_gamma` from `pv01_jax/risk.py`.
3. Writes `tests/jax_port/golden_jax/<suffix>.csv` with columns
   `label, mq_pv01, gamma_diag`, one row per quote in `problem.quote_labels`
   order, plus a final row holding the PV.

Choose a clear label for the PV row (for example `PV`) and write floats with
enough digits to round-trip at `1e-6`. Run the script once and commit the
generated files.

## Tests

Create `tests/jax_port/test_phase6_regression.py` using `unittest`. For each
suffix in `SETTINGS_SUFFIXES`:

1. Recompute the same values and compare with the golden file to absolute
   `1e-6` (GBP/bp for PV01, GBP/bp² for gamma, GBP for PV), per label.
2. Assert the set of labels in the file equals `problem.quote_labels`.
3. A missing golden file must fail the test with a message naming the file and
   the script that writes it.

## Copilot Instructions

- Do NOT create a golden file on the fly when it is missing. This differs from
  the Phase 0 golden tests in `tests/test_calibration_interpolation_example.py`.
- Share the value-computing code between the script and the test, so both
  compute exactly the same thing; keep it in `pv01_examples/jax/`, not in
  `pv01_jax`.
- Preserve quote-label ordering in the CSV.
- Keep tests in `unittest` style.

Do not modify any `pv01_*` packages other than adding
`pv01_examples/jax/write_jax_goldens.py`. Do not modify existing tests; add only
`tests/jax_port/test_phase6_regression.py` and the golden files. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the
results.

## Validation

Run only the test module created by this prompt, with `RUN_SLOW` unset:

```bash
env -u RUN_SLOW python -m unittest -v tests.jax_port.test_phase6_regression

Do not run full unittest discovery or unrelated reference-correctness tests.
Report whether the new regression test passed.

