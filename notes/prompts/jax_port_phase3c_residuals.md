# Phase 3c — JAX Calibration Residuals

## Objective
Implement calibration residuals as a pure-JAX function of the calibration problem,
curve parameters, and quote vector. Validate parity and derivatives against the
existing calibration pricer.

## Implementation

In `pv01_jax/pricers.py`, add:

```python
residuals(problem: CalibrationProblem, z, q) -> jax.Array
```

Return an array of length `len(q)`, ordered by `problem.quote_labels`.

Build curves with `curves_from(problem, z)`, then select:

```python
D = curves[problem.discount_curve]
L = curves[problem.forward_curve]
```

Loop over the static `problem.blocks` tuple only. Do not loop over periods.
For each block, compute its per-trade residuals using the flat period arrays and
scatter them into the global vector with the block's quote indices:

```python
R = jnp.zeros(len(q))
R = R.at[block.quote_idx].set(block_R)
```

Use the actual field names in `CalibBlock` and `FloatLegArrays` from
`pv01_jax/arrays.py`; the required behavior is:

- **FIX:** `L.forward(t_fix_start, t_fix_end, tau_fix) - q[quote_idx]`.
- **OIS:** float PV divided by fixed annuity, then subtract the quote:
  - float PV per period: `notional * D.compounded(start, end) * D.df(t_pay)`;
  - fixed annuity per period: `notional * tau * D.df(t_pay)`.
- **IRS:** same annuity as OIS, with float PV per period:
  `notional * L.forward(t_fix_start, t_fix_end, tau_fix) * tau * D.df(t_pay)`.

Aggregate period values by trade using `jax.ops.segment_sum` (or its current JAX
equivalent), with the number of trades from the block's per-trade quote indices.
The residual convention is **par minus quote**.

Before implementing, read the current `par_spread` implementation and its production
pricing dependencies. Match current production behavior and report any difference
from the formulas above.

## Tests

Create `tests/jax_port/test_phase3_residuals.py` using `unittest`. For each
`SETTINGS_SUFFIX`:

1. **Residual parity away from the solution:** shift calibrated parameters by
   `+0.001` (10 bp), rebuild the hand-coded provider with `with_y_values`, and compare
   `residuals(problem, z, q0)` with `par_spread` for every resolved calibration trade
   using an absolute tolerance of `1e-14`.
2. **Quote bump parity:** bump every node quote by `+0.0005` (5 bp), update market
   data with `tests/util/bump.py`'s `bumped_market_data`, resolve trades from that
   market data, and compare their `par_spread` values with
   `residuals(problem, z0, q_bumped)` to `1e-14`.
3. **At-solution residuals:** assert `max(abs(residuals(problem, z0, q0))) < 1e-12`.
4. **Quote Jacobian:** assert
   `jax.jacfwd(residuals, argnums=2)(problem, z0, q0)` equals `-I` to `1e-14`.
5. **Curve Jacobian:** compare
   `jax.jacfwd(residuals, argnums=1)(problem, z0, q0)` against the central finite
   difference Jacobian of the hand-coded residuals with bump size `1e-6`, following
   `tests/test_reference_correctness.py`. Use tolerance `1e-8` and report
   `cond(J)`.

Ensure trade comparisons are ordered by quote index so they align with
`problem.quote_labels`.

## Copilot Instructions

- Use JAX operations for the residual function; do not convert traced values to NumPy.
- Only use a Python loop over the static calibration blocks, never over periods.
- Keep quotes as a traced `q` input; residuals must read each quote through its
  `quote_idx`.
- Scatter block residuals into global quote order.
- Do not import `pv01_pricer` or `pv01_market` in `pv01_jax`; those packages are test
  oracles only.

Do not modify any `pv01_*` packages other than the `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only
`tests/jax_port/test_phase3_residuals.py`. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the results.
