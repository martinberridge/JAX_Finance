# Phase 4b — Calibrate the Real Curves

## Objective
Use the Phase 4a Newton solver to calibrate the real multi-curve problem, and
make the calibration solution differentiable with respect to market quotes using
`jax.lax.custom_root`.

## Implementation

In `pv01_jax/calibrate.py`, add:

```python
calibrate(problem, q) -> z
calibrate_with_info(problem, q)
    -> (z, iterations, max_abs_residual)
```

Use `residuals` from `pv01_jax/pricers.py`. Define the residual function over
parameters as `lambda z: residuals(problem, z, q)`. The calibration root should
be implemented with:

```python
jax.lax.custom_root(
    lambda z: residuals(problem, z, q),
    jax.lax.stop_gradient(q),
    newton_solve,
    tangent_solve,
)
```

Use the tangent solver from Phase 4a:

```python
lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(y), y)
```

`calibrate_with_info` should run `newton_solve_with_info` on the same residual
function and return its diagnostics; it does not need `custom_root`.

Start the solve from `jax.lax.stop_gradient(q)`, as the existing hand-coded
calibrator starts its curve parameters from the quotes. The quote vector and
global parameter vector are in `problem.quote_labels` order (discount curve
first, then forward curve for the current problem).

## Tests

Create `tests/jax_port/test_phase4_calibrate.py` using `unittest`. For each
`SETTINGS_SUFFIX`:

1. Build the calibration problem and quote vector `q0`, and calibrate with the
   existing `RatesCurveCalibrator` as the reference.
2. Assert `calibrate(problem, q0)` equals the hand-coded calibrated `y_values`
   concatenated in `problem.curves` / `problem.quote_labels` order to absolute
   tolerance `1e-12`.
3. Assert `calibrate_with_info(problem, q0)` reports maximum residual below
   `1e-14` and no more than 10 Newton iterations.
4. Wrap `calibrate` in `jax.jit`, count traces by appending to a Python list
   inside the jitted function, then call it with `q0`, `q0 + 1e-4`, and
   `q0 - 1e-4`. Synchronize each result with `block_until_ready()` and assert
   exactly one trace occurred.

## Copilot Instructions

- Keep `RatesCurveCalibrator` as a test oracle only; do not import it into
  `pv01_jax`.
- Implement calibration using `jax.lax.custom_root`; do not differentiate
  through the Newton iterations.
- Use the same residual function for `calibrate` and `calibrate_with_info`.
- Keep parameter and quote ordering aligned with `problem.quote_labels`.
- Use JAX operations on traced values; do not convert them to NumPy.

Do not modify any `pv01_*` packages other than `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only
`tests/jax_port/test_phase4_calibrate.py`. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the
results.
