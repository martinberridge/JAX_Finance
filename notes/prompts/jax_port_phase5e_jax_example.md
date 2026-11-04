# Phase 5e — JAX Calibration and Risk Example

## Objective
Create a runnable JAX counterpart to the existing calibration/interpolation
example that writes compatible PV and market-quote sensitivity CSVs and reports
JAX-versus-hand-coded risk and timing comparisons.

## Implementation

Create:

```text
pv01_examples/jax/calibration_interpolation_example_jax.py
```

Mirror `pv01_examples/multicurve2/calibration_interpolation_example.py`. For
each `SETTINGS_SUFFIX`:

1. Load the calibration configuration and market quotes.
2. Build the calibration problem and resolved swap arrays with `pv01_jax`.
3. Calibrate with `pv01_jax.calibrate`.
4. Compute PV with `pv_from_quotes` and market-quote PV01 with
   `market_quote_pv01`.
5. Write `<...>-pv.csv` and `<...>-mqs.csv` in the same format as
   `export_pv` and `export_mqs`. Reuse these helpers after inspecting
   `pv01_examples/export_utils.py`.
6. Write results to `target/example-output-jax/`, keeping existing example
   outputs and golden files untouched.
7. Print, per settings suffix:
   - the largest absolute difference from the golden MQS;
   - the elapsed JAX time for one `market_quote_pv01` call after a synchronized
     warm-up;
   - the elapsed hand-coded time for the equivalent `bump_pv01` calculation.

Skip hand-coded timing for the NCS setting unless `RUN_SLOW` is set.

## Tests

Add a test that runs the new script as a subprocess, following the pattern in
`tests/test_calibration_interpolation_example.py`. Verify the output CSV files
are written to the JAX-specific output directory and compare their contents
with the goldens to `0.01`.

## Copilot Instructions

- Reuse existing export helpers and follow their CSV layout and unit scaling.
- Preserve curve-node ordering and label association.
- Keep JAX implementation independent of `pv01_pricer` and `pv01_market`;
  production pricing and bump logic may be used only as comparison references
  in the example.
- Keep the JAX output folder separate from `target/example-output/`.
- Synchronize JIT warm-up before reporting execution time.
- Do not add timing thresholds to tests.

Do not modify any `pv01_*` packages other than `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only the new Phase 5 example test.
Run `python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report
the results.

## Validation

Run test discovery without setting `RUN_SLOW`, but exclude only the NCS cases
from `tests.test_reference_correctness`. Keep the Linear and DoubleQuadratic
reference-correctness cases and all other discovered tests. Since unittest
discovery has no built-in per-test exclusion option, discover the suite and
filter its test IDs:

```python
import sys
import unittest


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


discovered = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
selected = [
    test
    for test in flatten(discovered)
    if not (
        test.id().startswith("tests.test_reference_correctness.")
        and "__ncs_settings" in test.id()
    )
]
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(selected))
sys.exit(not result.wasSuccessful())
```

Report the test results and note that the NCS reference-correctness cases were
excluded.
