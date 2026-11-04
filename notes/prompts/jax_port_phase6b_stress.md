# Phase 6b — Stress and Failure Tests

## Objective
Calibrate and price under market scenarios the goldens never see, and confirm
the convergence guard raises readable errors when calibration fails.

## Implementation

No production code changes. Build each scenario as a quote vector derived from
`q0 = problem.q0`, using `problem.quote_labels` and `problem.curves` (each
`CurveSpec` has `labels` and `x_nodes`) to find each quote's curve and tenor:

| Scenario | Quote vector |
| --- | --- |
| Parallel up | `q0 + 0.02` |
| Parallel down (negative rates) | `q0 - 0.005` |
| Steepener | per curve, `-50bp` at the shortest tenor rising linearly in tenor to `+50bp` at the longest |
| Single bump | `+25bp` on `GBP-IRS6M-5Y` only |

Inspect `pv01_jax/arrays.py` and `pv01_jax/resolve.py` to confirm how quote
labels map to curves and node times before building the steepener.

## Tests

Create `tests/jax_port/test_phase6_stress.py` using `unittest`. For each suffix
in `SETTINGS_SUFFIXES`, and each scenario:

1. `calibrate_with_info(problem, q)` converges: maximum residual `< 1e-14` and
   iterations `<= 10`.
2. `market_quote_pv01(problem, swap_arrays, q)` equals a JAX central bump of
   `pv_from_quotes` per quote to `0.01 GBP/bp`:

   ```python
   (
       pv_from_quotes(problem, swap_arrays, q + 1e-4 * e_j)
       - pv_from_quotes(problem, swap_arrays, q - 1e-4 * e_j)
   ) / 2
   ```

Failure cases, for each settings suffix:

3. A `NaN` in one quote: `eqx.filter_jit(calibrate_checked)(problem, q)` raises
   with `"calibration did not converge"` in the message.
4. `max_iter=1` with the parallel `+200bp` scenario:
   `eqx.filter_jit(calibrate_checked)(problem, q, max_iter=1)` raises with the
   same message.

## Copilot Instructions

- Derive every scenario from `q0`; do not hard-code quote values.
- Use `eqx.filter_jit`, not `jax.jit`, around `calibrate_checked` so the error
  message survives compilation.
- Match on the message text, and catch the exception type that `eqx.error_if`
  actually raises; check it rather than guessing.
- Vectorise the bump loop with `jax.vmap` where it is simple to do so, as in
  `tests/jax_port/test_phase5_risk.py`.
- Keep tests in `unittest` style.

Do not modify any `pv01_*` packages. Do not modify existing tests; add only
`tests/jax_port/test_phase6_stress.py`. Run
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
