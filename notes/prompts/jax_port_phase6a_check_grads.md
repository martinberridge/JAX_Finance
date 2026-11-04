# Phase 6a — Generic Gradient Checks

## Objective
Check JAX's forward- and reverse-mode derivatives against finite differences, to
second order, on every differentiable function in the JAX port.

## Implementation

No production code changes. Use `jax.test_util.check_grads`:

```python
from jax.test_util import check_grads

check_grads(f, args, order=2, modes=["fwd", "rev"], atol=1e-4, rtol=1e-4)
```

`check_grads` compares JAX derivatives with finite differences in random
directions. Wrap each function as a closure of the single argument being
differentiated, with everything else closed over:

1. Each function in `INTERPOLATORS` from `pv01_jax/interp.py`, as
   `z -> interp(x, z, t)`, with the node grid `x` and a fixed evaluation grid
   `t` closed over. Take `x` and `z` from a curve in the calibrated problem, and
   choose `t` strictly inside the node range, avoiding the nodes themselves.
2. `z -> pv_swap(swap_arrays, curves_from(problem, z), problem.discount_curve,
   problem.forward_curve)`.
3. `z -> residuals(problem, z, q0)`.
4. `q -> residuals(problem, z0, q)`.
5. `q -> calibrate(problem, q)`.
6. `q -> pv_from_quotes(problem, swap_arrays, q)`.

Build `problem`, `swap_arrays`, `q0` and `z0` as in earlier phases: reuse the
cached helpers pattern in `tests/jax_port/test_phase5_risk.py` (`_problem`,
`_swap_arrays`), take `q0 = problem.q0`, and take `z0 = calibrate(problem, q0)`.

## Tests

Create `tests/jax_port/test_phase6_check_grads.py` using `unittest`. For each
suffix in `SETTINGS_SUFFIXES`, run one `subTest` per function in the list
above. Test 1 passes when every check passes at order 2 in both modes for every
function and every settings suffix.

## Copilot Instructions

- Derivatives of PV are large numbers, so rely on `rtol` rather than `atol`.
- If a check fails, report the function, the mode and the order that failed.
- Do not loosen tolerances beyond `1e-3` without saying why in a code comment
  and in your report.
- Do not change production code to make a check pass; report the failure
  instead.
- Keep tests in `unittest` style.

Do not modify any `pv01_*` packages. Do not modify existing tests; add only
`tests/jax_port/test_phase6_check_grads.py`. Run
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
