# Phase 2d — JAX Curve Module

## Objective
Create `pv01_jax/curves.py` containing an Equinox module:

```
Curve(spec: CurveSpec, z: jax.Array)
```

(`CurveSpec` is defined in `pv01_jax/arrays.py`.)

Implement the curve evaluation methods:

- `zero(t)`  
  ```
  INTERPOLATORS[spec.interpolator](spec.x_nodes, z, t)
  ```

- `df(t)`  
  ```
  exp(-t * zero(t))
  ```

- `forward(s, e, tau)` — IBOR forward  
  ```
  (df(s)/df(e) - 1) / tau
  ```

- `compounded(s, e)` — overnight compounding  
  ```
  df(s)/df(e) - 1
  ```

Also implement:

```
curves_from(problem, z) -> dict[str, Curve]
```

Slice `z` using each `CurveSpec`’s static `param_slice`.

---

## Part 1 — Confirm Formula Consistency

Before implementing:

- Read:
  - `pv01_pricer/rates_provider.py`
  - `pv01_pricer/instrument_pricers.py`

- Confirm that the formulas above match the **current Phase‑0‑corrected production code**.

If any formula differs:

1. Match the production code exactly.  
2. State explicitly what differed.

Copilot must follow the production formulas, not reinterpret them.

---

## Part 2 — Implement `Curve` and `curves_from`

### 1. `Curve` module
Implement an Equinox module with fields:

- `spec: CurveSpec`
- `z: jax.Array`

Methods:

- `zero(t)` — interpolated zero rate  
- `df(t)` — discount factor  
- `forward(s, e, tau)` — IBOR forward  
- `compounded(s, e)` — overnight compounding

All methods must use **pure JAX** (`jax.numpy`), no NumPy.

### 2. `curves_from(problem, z)`
For each curve in `problem.curve_specs`:

- Slice `z` using `spec.param_slice`
- Construct a `Curve(spec, z_slice)`
- Return a dict mapping curve names → Curve objects

---

## Part 3 — Tests

Create:

```
tests/jax_port/test_phase2_curves.py
```

Use `unittest`.

For each `SETTINGS_SUFFIX`:

Let:

- `z0` = concatenated calibrated `y_values` in **quote_labels order**
- `swap_arrays` = Phase‑1 `SwapArrays` from `resolve_swap`
- `curves = curves_from(problem, z0)`
- `provider` = existing calibrated provider

### (4a) Discount factors

For every payment period:

```
curves["DSCON"].df(t_pay) == provider.discount_factor(payment_date)
```

Accuracy:

```
<= 1e-14
```

### (4b) LIBOR forwards

For every float period:

```
curves["<LIBOR>"].forward(t_fix_start, t_fix_end, tau_fix)
==
provider.forward_rate(start, end)
```

Accuracy:

```
<= 1e-14
```

### (4c) Overnight compounding

For OIS calibration trades:

```
curves["DSCON"].compounded(s, e)
==
the compounding term used in _pv_float_leg_overnight
```

Accuracy:

```
<= 1e-14
```

### (6) Parameter consistency and gradient isolation

For each curve:

- `curve.z` must equal its calibrated `y_values`
- The gradient:

```
jax.grad(curves["<LIBOR name>"].df(t).sum(), argnums=z)
```

must be **zero on the discount slice**.

This ensures curve parameter isolation.

---

## Copilot Instructions

- Implement `Curve` and `curves_from` using **only** JAX (`jax.numpy` and Equinox).
- Do **not** use NumPy anywhere.
- Follow production formulas exactly; if they differ from the prompt, match production and state the difference.
- `zero(t)` must call the correct interpolator from `INTERPOLATORS`.
- `df(t)`, `forward`, and `compounded` must match the pricer formulas exactly.
- `curves_from` must slice `z` using each `CurveSpec.param_slice`.
- Tests must:
  - validate discount factors to `1e-14`
  - validate LIBOR forwards to `1e-14`
  - validate OIS compounding to `1e-14`
  - validate parameter isolation via gradients
- Ensure all Phase‑2 tests pass without modification.