# Phase 4d — Calibration Convergence Guard

## Objective
Add an explicit, JIT-compatible convergence check to real-curve calibration so
non-convergence and NaN residuals are reported as errors.

## Implementation

In `pv01_jax/calibrate.py`, add:

```python
calibrate_checked(problem, q, tol=1e-12, max_iter=30) -> z
```

Run the same calibration solve as `calibrate`, using the supplied tolerance
and iteration limit. Recompute the residual at the returned solution and guard
it with Equinox:

```python
max_res = jnp.max(jnp.abs(residuals(problem, z, q)))
z = eqx.error_if(
    z,
    ~(max_res <= tol),
    "calibration did not converge",
)
```

Use the negated `<=` condition as shown: it also fails for NaN residuals,
whereas `max_res > tol` evaluates false. Keep the default `calibrate` behavior
and its public API unchanged. If necessary, share an internal custom-root
helper that accepts solver tolerance and iteration limit so
`calibrate_checked` can pass its requested options.

Document in `calibrate_checked`’s docstring that callers should invoke it with
`eqx.filter_jit`; plain `jax.jit` does not preserve the useful error message.

## Tests

Add the following checks to `tests/jax_port/test_phase4_calibrate.py` using
`unittest`:

1. With `q = q0 + 0.02` and `max_iter=1`, assert
   `eqx.filter_jit(calibrate_checked)(problem, q, max_iter=1)` raises an error
   whose message contains `"calibration did not converge"`.
2. Set one quote in `q0` to NaN and assert the same error is raised when
   `calibrate_checked` is called through `eqx.filter_jit`.
3. With default options and `q0`, assert `calibrate_checked(problem, q0)`
   returns the same solution as `calibrate(problem, q0)`.

## Copilot Instructions

- Use the same calibration residual and custom-root path as `calibrate`.
- Ensure both non-convergence and NaN residuals trigger the error.
- Use `eqx.filter_jit` in error-path tests and check the exception message.
- Do not introduce a success-shaped fallback or silently return an unconverged
  solution.
- Keep tests in `unittest` style.

Do not modify any `pv01_*` packages other than `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only the Phase 4 tests in scope.
Run `python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report
the results.
