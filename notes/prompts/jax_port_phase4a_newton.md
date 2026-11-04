# Phase 4a — Generic JAX Newton Solver

## Objective
Implement a generic Newton root solver using JAX control flow, and verify its
solution and implicit derivative on a small vector-valued problem.

## Implementation

Create `pv01_jax/calibrate.py` with:

```python
newton_solve(f, z0, tol=1e-14, max_iter=30) -> z
newton_solve_with_info(f, z0, tol=1e-14, max_iter=30)
    -> (z, iterations, max_abs_residual)
```

Use `jax.lax.while_loop` with a `(z, k)` carry. Continue while
`k < max_iter` and `max(abs(f(z))) > tol`. Each Newton step is:

```python
z - jnp.linalg.solve(jax.jacfwd(f)(z), f(z))
```

The solver must use JAX operations throughout and must not rely on Python loops
or convert traced values to NumPy. `newton_solve` returns only the solution;
`newton_solve_with_info` also returns the iteration count and final maximum
absolute residual.

## Tests

Create `tests/jax_port/test_phase4_newton.py` using `unittest`.

1. For `a = jnp.array([0.5, 2.0, 9.0])`, solve
   `f(z) = z**2 - a` from `z0 = a` and assert the result equals `sqrt(a)` to
   absolute tolerance `1e-14`.
2. Wrap the same root in `jax.lax.custom_root`:

   ```python
   jax.lax.custom_root(
       lambda z: z**2 - a,
       jax.lax.stop_gradient(a),
       newton_solve,
       tangent_solve,
   )
   ```

   where:

   ```python
   tangent_solve = lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(y), y)
   ```

   Differentiate with `jax.jacfwd` with respect to `a` and assert the result
   equals `diag(1 / (2 * sqrt(a)))` to absolute tolerance `1e-12`. The vector
   `a` is captured by the residual closure.
3. Assert `newton_solve_with_info` reports at most 10 iterations and a final
   maximum absolute residual below `1e-14`.

## Copilot Instructions

- Use `jax.lax.while_loop` for the Newton iteration.
- Compute the Newton step with `jax.jacfwd` and `jnp.linalg.solve`.
- Do not differentiate through Newton iterations; the custom-root test must
  verify the implicit derivative.
- Keep tests in `unittest` style.

Do not modify any `pv01_*` packages other than `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only
`tests/jax_port/test_phase4_newton.py`. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the
results.
