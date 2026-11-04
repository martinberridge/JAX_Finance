# Phase 6e — Strata Comparison (Optional)

## Objective
Compare Strata's Java market-quote sensitivities with the JAX output per bucket,
and explain each difference by a specific simplification in the Python port.

## Implementation

Create `pv01_examples/jax/strata_comparison.py`, runnable as:

```bash
python -m pv01_examples.jax.strata_comparison <path-to-strata-output>
```

1. Read Strata's market-quote sensitivity CSVs for
   `CalibrationInterpolationExample` from the directory given as an argument.
   I will generate them by running the Java example.
2. Read the JAX output from `target/example-output-jax/` (written by the Phase
   5e example). Reuse `_parse_mqs` from
   `tests/test_calibration_interpolation_example.py`, or the same parsing logic,
   after inspecting the Strata file layout.
3. Write `target/strata_comparison.md` with one table per settings suffix:
   label, Strata, JAX, difference.
4. Below the tables, list the simplifications in the Python port that could
   explain the differences:
   - schedule generation;
   - OIS compounding period by period rather than daily;
   - payment-date adjustment;
   - index fixing dates for the forward.

   Check each against the `pv01_product` and `pv01_basics` code, cite the
   relevant file and function, and say which buckets each one most likely
   affects.

## Tests

None. Do not assert a tolerance; the Python port's simplified conventions mean
exact agreement is not expected.

## Copilot Instructions

- Fail with a clear message if the Strata directory or an expected file is
  missing, naming the file.
- Check that Strata and JAX use the same units and sign convention before
  differencing; say what you found.
- Match buckets by label, not by row position; report labels present in one
  source but not the other.
- Base each explanation on code you have read, not on assumption.

Do not modify any `pv01_*` packages other than adding
`pv01_examples/jax/strata_comparison.py`. Do not modify existing tests. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the
results.

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
