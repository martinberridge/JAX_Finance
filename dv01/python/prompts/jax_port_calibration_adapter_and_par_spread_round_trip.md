
	
# Phase 1 — Build Calibration Problem

## Goal
Implement `build_calibration_problem` in `pv01_jax/resolve.py` and validate it with unit tests so that Copilot can reliably port the calibration logic to JAX.

---

## Part 1 — Implement `build_calibration_problem`

### Function
```python
build_calibration_problem(group_defn, market_data, valuation_date) -> CalibrationProblem
```

See `CalibrationProblem` in `pv01_jax/arrays.py`.

### Requirements

- **Reuse existing resolution logic**
  - Use `pv01_product.resolved_trades.resolve_calibration_trade` for dates and schedules.

- **Group trades into calibration blocks**
  - Group trades into one `CalibBlock` per kind:
    - `FIX`
    - `OIS`
    - `IRS`
  - Each block must have **flat per-period arrays**:
    - `FloatLegArrays`
    - `LegArrays`
  - Each trade must have an `int trade_id` identifying its periods.
  - For `FIX` instruments:
    - `fixed_leg = None`.

- **Do not store market or fixed rates in arrays**
  - Do **not** store:
    - `market_rate`
    - `fixed_rate`
  - Instead, store:
    - `quote_idx` into `q0` for each trade.

- **Curve ordering and quotes**
  - Order curves exactly as in:
    - `RatesCurveCalibrator.calibrate` (DSCON first).
  - Order nodes exactly as in:
    - the nodes CSV.
  - `q0` must contain **only curve-node quotes**, in that order:
    - one entry per curve node
    - no extra quotes

- **CurveSpec configuration**
  - For each `CurveSpec`, set:
    - `param_slice` (index range into the global parameter vector)
    - `labels` (node labels)
    - `interpolator` (from the curve settings)
    - `x_nodes` (node times)

---

## Part 2 — Tests for `build_calibration_problem`

Create:

```text
tests/jax_port/test_phase1_build_calibration_problem.py
```

Use `unittest`.

For each `SETTINGS_SUFFIX`:

### (a) Quote mapping correctness

For every trade:

```python
q0[quote_idx] == trade.market_rate
```

Assert that:

- the `quote_idx` stored in the arrays points into `q0`
- the value at `q0[quote_idx]` equals `trade.market_rate`

### (b) Par spread consistency

For every trade:

- Compute a **NumPy par spread** from the arrays:
  - read `q` from `q0`
  - read curves from the **existing calibrated provider**
- Mirror the current `par_spread` formulas exactly, including:
  - Phase 0 sign convention
- Assert:

```python
abs(par_spread_from_arrays - instrument_pricers.par_spread(trade, provider)) <= 1e-12
```

### (c) Period count per trade

For every trade:

- Use `trade_id` in the arrays to recover per-trade period counts.
- Assert:

```python
period_count_from_arrays == len(trade.periods)
```

---

## Copilot Instructions

- Mirror the existing Python implementation logic exactly; do **not** redesign financial formulas.
- Use `resolve_calibration_trade` for all date/schedule resolution.
- Ensure:
  - `CalibBlock` grouping by kind (`FIX`, `OIS`, `IRS`)
  - flat per-period arrays with `trade_id` mapping
  - no market or fixed rates stored in arrays; only `quote_idx` into `q0`
  - curve ordering and node ordering match `RatesCurveCalibrator.calibrate` and the nodes CSV
- Tests must:
  - validate quote mapping (`q0[quote_idx] == trade.market_rate`)
  - validate par spread to `1e-12` against `instrument_pricers.par_spread`
  - validate per-trade period counts via `trade_id`.
