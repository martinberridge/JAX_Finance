# Phase 6f — Documentation and Final Check

## Objective
Document how the JAX port is organised and how to run it, and confirm the whole
test suite passes with no expected failures left.

## Implementation

Write `docs/pv01_jax.md` covering:

1. **Module map.** One paragraph each for `arrays`, `resolve`, `interp`,
   `curves`, `pricers`, `calibrate` and `risk` in `pv01_jax/`, based on the
   code as it stands.
2. **Hand-coded reverse pass to JAX.** A table mapping:

   | Hand-coded | JAX |
   | --- | --- |
   | `present_value_sensitivity` and `PointSensitivities` | `jax.grad` of `pv_swap` |
   | `parameter_sensitivity` | AD through the interpolator |
   | `JacobianCalibrationMatrix` and `MarketQuoteSensitivityCalculator` | `custom_root` inside `calibrate` |

   Locate each hand-coded class or function in the `pv01_*` packages and name
   its module.
3. **Running it.** How to run the tests with and without `RUN_SLOW=1`, the JAX
   example script (`pv01_examples/jax/calibration_interpolation_example_jax.py`),
   the golden writer (`pv01_examples/jax/write_jax_goldens.py`) and the
   benchmark (`python -m pv01_jax.benchmark`).
4. **Known differences from Strata.** Summarise `target/strata_comparison.md`
   list the known simplifications (schedule
   generation, OIS compounding by period, payment-date adjustment, forward index
   fixing dates) and say the comparison has not been run.

## Tests

Search `tests/` for `expectedFailure` and any other expected-failure markers and report if any.


## Copilot Instructions

- Describe the code as it is; check every function and module name you cite.
- Keep the document concise; it is also source material for the blog series.
- Do not remove expected-failure markers or change tests to make the suite
  pass. Report any that remain, with the file, the test and the reason.

Do not modify any `pv01_*` packages. Do not modify existing tests; add only
`docs/pv01_jax.md`. 