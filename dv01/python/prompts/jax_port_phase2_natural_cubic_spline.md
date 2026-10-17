
# Phase 2c — Natural Cubic Spline Interpolator

## Objective
Extend `pv01_jax/interp.py` by adding a pure‑JAX implementation of:

```
natural_cubic_spline(x, z, t)
```

This must match **pv01_market.interpolators.NaturalCubicSplineInterpolator** exactly.

Requirements:

- Use only `jax.numpy` (`jnp`)
- No Python branching on traced values
- No NumPy
- Requires `len(x) >= 3`

Return an array with the same shape as `t`.

---

## Part 1 — Implement `natural_cubic_spline`

### 1. Compute second derivatives `y2`
Compute `y2` **once per call** by solving the `n × n` linear system with **natural end conditions**.

Let:

- `h = diff(x)`
- `s = diff(z) / h`

Construct the system:

- Row `0`: identity  
- Row `n-1`: identity  
- Interior row `i` (1 ≤ i ≤ n‑2):

  ```
  [ h[i-1]/6 , (h[i-1] + h[i])/3 , h[i]/6 ]
  RHS = s[i] - s[i-1]
  ```

Solve using:

```
y2 = jnp.linalg.solve(A, rhs)
```

(`n` is small, ~10.)

### 2. Interpolation
- Clip `t` into `[x[0], x[-1]]`
- Compute interval index:
  ```
  i = _interval_index(x, t)
  ```

Let:

```
a = (x[i+1] - t) / h[i]
b = 1 - a
```

Return:

```
a*z[i] + b*z[i+1]
+ (a*(a*a - 1)*y2[i] + b*(b*b - 1)*y2[i+1]) * (h[i]**2) / 6
```

### 3. Sensitivities
Do **not** port:

- `_second_deriv_sensitivities`
- `parameter_sensitivity`

Automatic differentiation through the `solve` replaces them.  
Test (2) proves correctness.

### 4. Registration
Add:

```python
INTERPOLATORS["NaturalCubicSpline"] = natural_cubic_spline
```

Keys must match the names used in the curve settings CSVs.

---

## Part 2 — Acceptance Criteria

The existing **Phase 2a tests must pass** for all curves whose settings specify `"NaturalCubicSpline"` — **with no changes to the tests**.

This includes:

- Value correctness to `1e-13`
- Jacobian correctness to `1e-12`
- Finite gradients everywhere
- Automatic inclusion via the test loop over `INTERPOLATORS`

---

## Copilot Instructions

- Implement `natural_cubic_spline` using **only** `jax.numpy`.
- Use `jnp.linalg.solve` for the second‑derivative system.
- Do **not** use NumPy anywhere.
- Do **not** use Python `if/else` on traced values.
- `_interval_index` must match Phase 2a.
- Interpolation formula must match the production spline exactly.
- Do **not** port manual sensitivity code; rely on AD.
- Register the function as `INTERPOLATORS["NaturalCubicSpline"]`.
- Ensure all Phase 2a tests pass for `"NaturalCubicSpline"` curves without modification.
