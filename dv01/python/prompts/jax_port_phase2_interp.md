# Phase 2 — JAX Interpolators

## Objective
Create `pv01_jax/interp.py` containing **pure JAX** interpolation functions of the form:

```
f(x, z, t)
```

where:

- `x`: node times (1‑D)
- `z`: node values (1‑D, same length as `x`)
- `t`: query times (any shape)

Return an array **with the same shape as `t`**.

All functions must be **pure JAX**, using only `jax.numpy` (`jnp`).  
No Python branching on traced values.  
No NumPy inside the functions.

---

## Part 1 — Implement Interpolators

### 1. `_interval_index(x, t)`
A JAX version of:

```
clip(searchsorted(x, t, side="right") - 1, 0, len(x) - 2)
```

This must match the behavior of:

```
pv01_market.interpolators.lower_bound_index
```

### 2. `linear(x, z, t)`
Implement linear interpolation with **flat extrapolation**:

1. First clip:
   ```
   t = jnp.clip(t, x[0], x[-1])
   ```
   This reproduces the flat extrapolation used by all curves in the config.

2. Then perform standard linear interpolation using JAX operations only.

### 3. `INTERPOLATORS`
Define:

```python
INTERPOLATORS = {
    "Linear": linear,
}
```

Keys must match the names used in the curve settings CSVs.

---

## Part 2 — Tests

Create:

```
tests/jax_port/test_phase2_interp.py
```

Use `unittest`.

### Test Setup

For each `SETTINGS_SUFFIX`:

- Calibrate discount and LIBOR curves whose interpolator is implemented.
- Cache providers using `lru_cache` exactly as in:
  ```
  tests/test_reference_correctness.py
  ```
- Extract:
  - `x` = curve node times
  - `z` = curve.y_values
- Build a `t` grid consisting of:
  - `linspace(-1, 35, 401)`
  - every node exactly
  - every node ± 1e‑9

### Assertions

#### (1) Value correctness
For every `ti`:

```
interp(x, z, t)[i] == curve.y_value(ti)
```

Accuracy tolerance:

```
<= 1e-13
```

#### (2) Jacobian correctness
Use:

```
jax.jacfwd(interp, argnums=1)(x, z, t)
```

Assert equality to:

```
stacked curve.y_value_parameter_sensitivity(ti)
```

Accuracy tolerance:

```
<= 1e-12
```

#### (3) Gradient sanity
Assert that:

- `jax.grad(interp(...).sum(), argnums=z)` is finite everywhere
- `jax.grad(interp(...).sum(), argnums=t)` is finite everywhere

### Test Structure Requirement
Write tests so they **loop over `INTERPOLATORS`**, ensuring that:

- Phase 2b and Phase 2c only need to add new functions
- No test rewrites are required when new interpolators are added

---

## Copilot Instructions

- Use **only** `jax.numpy` inside interpolation functions.
- Do **not** use Python `if/else` on traced values.
- Do **not** use NumPy anywhere inside `interp.py`.
- `_interval_index` must exactly match the semantics of `lower_bound_index`.
- `linear` must implement flat extrapolation via clipping before interpolation.
- `INTERPOLATORS` must be keyed by names from the curve settings CSVs.
- Tests must:
  - validate values to `1e-13`
  - validate Jacobians to `1e-12`
  - validate gradients are finite
  - iterate over all entries in `INTERPOLATORS`
- Follow the existing pricer formulas exactly; do not redesign interpolation logic.
