# Jax port phase 1 — Resolve Swap + Reference PV Validation

## Goal
Implement `resolve_swap` in `pv01_jax/resolve.py` and validate it using a NumPy reference pricer in `tests/jax_port/test_phase1_resolve_swap.py`.

---

## Part 1 — Implement `resolve_swap`

### Function
resolve_swap(swap: ResolvedSwapTrade, valuation_date) -> SwapArrays

### Requirements
- **Times**  
  Use:  
  `pv01_basics.day_count.act_365f_between(valuation_date, d)`
- **Accrual fractions**  
  `tau = act_365f_year_fraction(start, end)`
- **Fixing accrual periods**  
  `t_fix_start`, `t_fix_end`, `tau_fix` must match the accrual period used in Phase 0 step 8 forward.
- **trade_id**  
  All zeros.
- **sign**  
  `+1.0` for BUY  
  `-1.0` for SELL
- **fixed_rate**  
  `swap.fixed_rate`

### Structural expectations
- Arrays must reflect the correct number of periods.  
- `t_pay` must be strictly increasing.

---

## Part 2 — Write Phase 1 Tests

Create:

tests/jax_port/test_phase1_resolve_swap.py

Use `unittest`.

---

## Reference Pricer: `pv_from_arrays(arrays, provider)`

### Purpose
A NumPy-based reference implementation that mirrors the existing Python pricer exactly.

### Discount factors
provider.discount_curve.y_value(t)

### Forward rates
provider.libor_curve.y_value(t)

### Formula source of truth
Mirror the formulas in:

- `pv01_pricer/swap_pricer.py`
- `pv01_pricer/rates_provider.py`

This is a direct mirror, not a redesign.

---

## Part 3 — Calibration Setup

For each `SETTINGS_SUFFIX`:

1. Calibrate once  
   Use `lru_cache` as in `tests/test_reference_correctness.py`.
2. Build the swap exactly as in:  
   `calibration_interpolation_example.main()`

---

## Part 4 — Assertions

### (a) PV match
| pv_from_arrays(arrays, provider)

DiscountingSwapTradePricer.present_value(swap, provider)["GBP"] |
< 1e-6

### (b) Structural checks
- Leg array lengths equal the number of periods  
- `t_pay` is strictly increasing

---

## Copilot Instructions

When using this prompt:

- Follow all formulas exactly as implemented in the reference pricers.  
- Do not simplify, optimize, or reinterpret the financial logic.  
- Ensure array shapes, period counts, and time ordering match the production pricer.  
- Use NumPy for the reference pricer; use JAX only in the main implementation.  
- Maintain strict numerical consistency (tolerances ≤ 1e-6).
