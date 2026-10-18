# Phase 3a — CalibrationProblem Curve Name Fields

## Objective
Extend `pv01_jax/arrays.py` and `pv01_jax/resolve.py` so that every `CalibrationProblem` explicitly records:

- the name of the discount curve  
- the name of the forward (LIBOR) curve  

These fields must match the production calibrator’s logic exactly.

---

## Part 1 — Add Static Fields to `CalibrationProblem`

In `pv01_jax/arrays.py`, add two static Equinox fields:

```python
discount_curve: str = eqx.field(static=True)
forward_curve: str = eqx.field(static=True)
```

These fields must be **static**, because curve names do not participate in JAX tracing or differentiation.

---

## Part 2 — Populate Fields in `resolve_calibration_problem`

Modify `resolve_calibration_problem` in `pv01_jax/resolve.py` so that it fills the two fields exactly the same way the production calibrator (`RatesCurveCalibrator.calibrate`) builds its provider.

### Required Logic

1. **discount_curve**  
   - Select the `curve_name` of the group entry whose `curve_type` is `"Discount"`.

2. **forward_curve**  
   - Select the `curve_name` of the group entry whose `reference` is `"GBP-LIBOR-6M"`.

### Mandatory Step

Before implementing, **read the current production calibrator**:

- `pv01_pricer/rates_provider.py`
- `pv01_pricer/instrument_pricers.py`

Confirm that the logic above matches the Phase‑0‑corrected calibrator.

If anything differs:

1. Match the production code exactly.  
2. State explicitly what differed.

Copilot must follow production behaviour, not reinterpret it.

---

## Part 3 — Tests

Create:

```
tests/jax_port/test_phase1_resolve_calibration_problem.py
```

Use `unittest`.

For each `SETTINGS_SUFFIX`:

Let:

- `problem = resolve_calibration_problem(...)`
- `provider = calibrated provider from Phase 0`

### Assertions

#### (1) Discount curve name matches provider

```
problem.discount_curve == provider.discount_curve.name
```

#### (2) Forward curve name matches provider

```
problem.forward_curve == provider.libor_curve.name
```

#### (3) Both names appear in `problem.curves`

```
problem.discount_curve in problem.curves
problem.forward_curve in problem.curves
```

These checks ensure:

- correct extraction of curve names  
- consistency with production calibrator  
- correct inclusion in the JAX calibration problem structure  

---

## Copilot Instructions

- Add static fields `discount_curve` and `forward_curve` to `CalibrationProblem`.
- Populate them in `resolve_calibration_problem` using the same logic as the production calibrator.
- Read the calibrator code first; if any behaviour differs from the prompt, match production and state the difference.
- Ensure:
  - discount curve name matches `provider.discount_curve.name`
  - forward curve name matches `provider.libor_curve.name`
  - both names appear in `problem.curves`
- All Phase‑1 and Phase‑3 tests must pass without modification.