# Phase 5c — Cross-Gamma

## Objective
Calculate the market-quote cross-gamma matrix for swap PV and validate its
symmetry and finite-difference columns.

## Implementation

In `pv01_jax/risk.py`, add:

```python
cross_gamma(problem, swap, q) -> jax.Array
```

Calculate the Hessian of `pv_from_quotes` with respect to its quote argument
and scale the result to GBP per bp squared:

```python
jax.hessian(pv_from_quotes, argnums=2)(problem, swap, q) * 1e-8
```

The result shape is `(n_quotes, n_quotes)`.

## Tests

Add tests to `tests/jax_port/test_phase5_risk.py` using `unittest`, for each
`SETTINGS_SUFFIX`:

1. Assert the cross-gamma matrix is symmetric to `1e-6` relative to its
   largest absolute entry.
2. For each quote column `j`, compare that column with the central difference
   of `market_quote_pv01`:

   ```python
   (
       market_quote_pv01(problem, swap_arrays, q0 + h * e_j)
       - market_quote_pv01(problem, swap_arrays, q0 - h * e_j)
   ) / (2 * h) * 1e-4
   ```

   Use `h = 1e-5` in rate units and tolerance `1e-6` relative to the largest
   matrix entry.
3. JIT the cross-gamma calculation, synchronize the warm-up, time one
   subsequent call, and print its elapsed time. Do not assert on timing.

## Copilot Instructions

- Derive the Hessian from `pv_from_quotes`; do not implement a separate
  hand-coded second-derivative path.
- Scale rate-unit derivatives by `1e-8` to report GBP per bp squared.
- Synchronize JIT compilation separately from the measured execution.
- Keep tests in `unittest` style.

Do not modify any `pv01_*` packages other than `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only
`tests/jax_port/test_phase5_risk.py`.

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
