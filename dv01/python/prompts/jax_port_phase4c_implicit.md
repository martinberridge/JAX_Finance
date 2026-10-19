# Phase 4c — Implicit Derivative Checks

## Objective
Verify that the quote derivative of the JAX calibration solution matches the
hand-coded calibration Jacobian, finite-difference recalibration, and is
independent of the Newton iteration strategy.

## Implementation

Use `calibrate` from `pv01_jax/calibrate.py`, `residuals` from
`pv01_jax/pricers.py`, and the existing `pv_swap` / `curves_from` APIs. Do not
replace the implicit derivative with differentiation through solver iterations.

Before implementing the hand-coded Jacobian comparison, read:

- `pv01_market/jacobian.py`
- `pv01_pricer/rates_curve_calibrator.py`

Confirm the matrix’s row and column ordering and that it represents `dz/dq`,
not its inverse. Assemble the reference in the same global order as
`problem.quote_labels`, with the discount curve first.

If needed to test different solver iteration strategies, factor the custom-root
construction through a private helper that accepts a solver function. Preserve
the public `calibrate(problem, q)` API.

## Tests

Create `tests/jax_port/test_phase4_implicit.py` using `unittest`. For each
`SETTINGS_SUFFIX`:

1. **Hand-coded calibration Jacobian:** assert
   `jax.jacfwd(calibrate, argnums=1)(problem, q0)` matches the hand-coded
   calibration Jacobian. Stack each calibrated curve’s
   `curve.jacobian.jacobian_matrix`, preserving discount-curve-first parameter
   rows and quote columns, and compare to absolute tolerance `1e-8`.
2. **Finite-difference recalibration:** for each quote column `j`, compare the
   Jacobian column with:

   ```python
   (calibrate(problem, q0 + h * e_j) -
    calibrate(problem, q0 - h * e_j)) / (2 * h)
   ```

   using `h = 1e-6` and absolute tolerance `1e-6`.
3. **Solver-iteration independence:** differentiate the swap PV with respect
   to quotes through `calibrate`. Compare the gradient when Newton stops at
   tolerance with the gradient when the custom-root solve function performs
   exactly 20 Newton updates in a `jax.lax.fori_loop`. Assert agreement to
   absolute tolerance `1e-10`.

## Copilot Instructions

- Use the hand-coded Jacobian and recalibration only as test oracles.
- Preserve the quote and curve-node order throughout each comparison.
- Ensure the derivative is supplied by `custom_root`, not by backpropagating
  through Newton iterations.
- Do not import `pv01_pricer` or `pv01_market` into `pv01_jax`.
- Keep tests in `unittest` style and do not modify tests from earlier phases.

Do not modify any `pv01_*` packages other than `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only
`tests/jax_port/test_phase4_implicit.py`. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the
results.
