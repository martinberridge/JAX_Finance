# Phase 2b — Double Quadratic Interpolator

## Objective
Extend `pv01_jax/interp.py` by adding a pure‑JAX implementation of:

```
double_quadratic(x, z, t)
```

This must match **pv01_market.interpolators.DoubleQuadraticInterpolator** exactly.

Requirements:

- Use only `jax.numpy` (`jnp`)
- No Python branching on traced values
- No NumPy
- Requires `len(x) >= 3`

Return an array with the same shape as `t`.

---

## Part 1 — Implement `double_quadratic`

### 1. Preprocessing
- Clip query times:
  ```
  t = jnp.clip(t, x[0], x[-1])
  ```
- Compute interval index:
  ```
  i = _interval_index(x, t)
  ```

### 2. Quadratic construction
For each interval, use the quadratic through nodes:

```
k-1, k, k+1
```

centred at `x[k]`, with coefficients `b` and `c` exactly as in:

```
DoubleQuadraticInterpolator._quadratic_coeffs
```

### 3. Boundary cases
Implement the three cases:

- **i == 0**  
  Use quadratic at `k = 1` only.

- **i == n-2**  
  Use quadratic at `k = n-2` only.

- **otherwise**  
  Weighted blend:
  ```
  w = (x[i+1] - t) / (x[i+1] - x[i])
  w * Q_i(t) + (1 - w) * Q_{i+1}(t)
  ```

### 4. JAX branching
Select between the three cases using `jnp.where`.

Before selection, **clamp every k into `[1, n-2]`** so all branches are finite and safe.

### 5. Registration
Add:

```python
INTERPOLATORS["DoubleQuadratic"] = double_quadratic
```

Keys must match the names used in the curve settings CSVs.

---

## Part 2 — Acceptance Criteria

The existing **Phase 2a tests must pass** for all curves whose settings specify `"DoubleQuadratic"` — **with no changes to the tests**.

This means:

- Value correctness to `1e-13`
- Jacobian correctness to `1e-12`
- Finite gradients everywhere
- Test loop over `INTERPOLATORS` must automatically pick up `"DoubleQuadratic"`

---

## Copilot Instructions

- Implement `double_quadratic` using **only** `jax.numpy`.
- Do **not** use Python `if/else` on traced values.
- Do **not** use NumPy anywhere.
- `_interval_index` must be reused exactly as in Phase 2a.
- Quadratic coefficients must match `DoubleQuadraticInterpolator._quadratic_coeffs`.
- Boundary cases must be handled via `jnp.where`, with clamped `k` values.
- Weighted blending must match the production formula exactly.
- Register the function as `INTERPOLATORS["DoubleQuadratic"]`.
- The existing Phase 2a tests must pass for all `"DoubleQuadratic"` curves without modification.
