# Calibration interpolation example — conversation notes

Notes from a walkthrough of `pv01_examples/multicurve2/calibration_interpolation_example.py` and the pricer/calibrator it calls.

---

## Dataflow of `main`

`main` loads one GBP quote snapshot, builds **three** curve-group configs that share the same nodes but differ only in interpolator, prices one test swap on each calibrated curve set, then writes PV and 1bp market-quote PV01 CSVs.

Valuation date is **2016-08-01**. Config name: `GBP-DSCONOIS-L6MIRS-FRTB` (OIS discount + 6M LIBOR IRS, FRTB node set). Interpolator suffixes: linear, double-quadratic (`-dq`), natural cubic spline (`-ncs`).

### 1. Inputs (paths + raw CSVs)

- `...-group.csv` — which curves exist (Discount vs Forward, names, indexes)
- `...-nodes.csv` — calibration instruments (quote IDs, node types, labels) — **shared**
- `...-{linear,dq,ncs}-settings.csv` — interpolator / day-count / extrapolators — **varies**
- `MARKET-QUOTES-GBP-20160801.csv` — observed market values for 2016-08-01

`load_market_data` filters rows to that date and returns `MarketData(valuation_date, quotes: QuoteId → float)`. That object is reused for all three interpolators.

### 2. Three curve-group definitions

Each `load_calibration` merges group + settings + nodes into `{ "GBP-DSCONOIS-L6MIRS-FRTB": RatesCurveGroupDefinition }`. Same nodes and quotes; only `CurveSettings.interpolator` changes.

### 3. One product, built once

Resolved **8Y GBP fixed vs 6M LIBOR**, 6M forward start, buy, notional 10m, coupon 2.5%. Period schedules are frozen before the interpolator loop.

Calibrator / pricer / MQS calculator are also created once (`RatesCurveCalibrator.standard()`, `DiscountingSwapTradePricer.DEFAULT`, `MarketQuoteSensitivityCalculator.DEFAULT`).

### 4. Per interpolator: calibrate → price → bucket → export

**Calibrate** (`config` + `market_quotes` → `ImmutableRatesProvider`):

1. Each curve node becomes a calibration trade (`OIS` / `FIX` / `IRS`) whose target is the matching quote.
2. Newton solve: nodal **zero rates** so par spreads of those trades are ~0.
3. Objective uses the interpolator from that settings file when building `InterpolatedNodalCurve`s.
4. After convergence, inverse Jacobian `d(zeros)/d(quotes)` is stored on each curve.

**Price:** `present_value` discounts the fixed and floating legs on that provider → `{"GBP": pv}`.

**Sensitivities (three hops):**

```
swap cashflows
  → PointSensitivities     (dPV / d zero at payment times, dPV / d Ibor at period dates)
  → CurrencyParameterSensitivities  (project points onto curve node zeros via interpolator basis)
  → market-quote sensitivities      (zeros @ inverse Jacobian → dPV / d quotes)
```

`export_mqs` multiplies by `BP1 = 1e-4` so the CSV is **bucketed PV01 per 1bp quote bump**.

**Outputs** under `python/target/example-output/`:

- `GBP-DSCONOIS-L6MIRS-FRTB-{linear,dq,ncs}-settings-pv.csv`
- `GBP-DSCONOIS-L6MIRS-FRTB-{linear,dq,ncs}-settings-mqs.csv`

The experiment: **same quotes, same nodes, same swap; interpolator is the only change**.

---

## `RatesCurveCalibrator.calibrate` (high level)

Turns a curve-group definition plus a quote snapshot into a fitted `ImmutableRatesProvider`: interpolated discount and LIBOR curves whose nodal zeros reprice the calibration instruments, plus the inverse Jacobian needed later for market-quote PV01.

### What it is solving

Each curve node is a market instrument (OIS, FIX, or IRS). The unknowns are the **nodal y-values** (zero rates) of all curves, concatenated into one vector \(z\). The residual \(R(z)\) is the **par spread** of each instrument given those curves: implied minus quoted for a FIX node, \(PV / PV01\) for OIS/IRS.

Calibration is Newton’s method for \(R(z) = 0\).

### Setup

1. **Curve order** — discount (`DSCON`) first, then forward (LIBOR). That order is the global parameter layout.
2. **Trades** — each node becomes a resolved calibration trade whose target rate is the matching quote. Node dates and labels come from those trades.
3. **Initial \(z\)** — the quote values themselves (a rough start, not already zeros).
4. **`build_provider(z)`** — slice \(z\) per curve, build `InterpolatedNodalCurve`s with that curve’s interpolator, wrap them as discount + LIBOR in `ImmutableRatesProvider`.

### Newton loop

Each iteration:

1. Price all calibration trades with the current provider → residual vector \(R\).
2. Approximate \(J = \partial R / \partial z\) by central finite differences (bump each zero by \(10^{-6}\)).
3. If \(\max|R| < 10^{-14}\), stop.
4. Else Newton step: \(z \leftarrow z - J^{-1} R\).

If it still misses after 50 iterations, it raises.

### After convergence

Rebuild the provider from the fitted zeros, then invert the last Jacobian:

\[
J^{-1} \approx \frac{\partial z}{\partial q}
\]

That matrix is stored **per curve** as `JacobianCalibrationMatrix` (rows for that curve’s nodes, columns for all quotes). Market-quote sensitivity later does \(\partial PV/\partial q = (\partial PV/\partial z)\, J^{-1}\).

Quotes and interpolator never change inside the solver; only the nodal zeros do. The interpolator only affects how zeros between nodes are used when pricing \(R(z)\).

---

## `present_value_sensitivity` and `parameter_sensitivity`

Two-stage chain rule. The pricer first asks “how does PV move if the curve zero at this *date* moves?”, then the provider asks “how does that date’s zero move if the *nodal* zeros move?”

```
PV  →  present_value_sensitivity  →  PointSensitivities
    →  parameter_sensitivity      →  dPV / d(nodal zeros)
    →  MarketQuoteSensitivity     →  dPV / d(quotes)
```

### `present_value_sensitivity` — risk at cashflow times

On `DiscountingSwapTradePricer`. It does **not** know about curve nodes or interpolators. It walks every period of the resolved swap and records analytic \(\partial PV / \partial z(t)\) at the times that appear in the formula.

PV of a period is (schematically) \(\pm N \times r \times \tau \times DF(t_{pay})\), with \(r\) the fixed coupon or the LIBOR forward.

Two channels:

1. **Discounting (every period).** \(DF(t) = e^{-t\,z(t)}\) on the **OIS** curve, so \(\partial DF / \partial z(t) = -t\,DF\). The cash amount being discounted is the period forecast. That produces a `PointSensitivity("ZeroRate", t_pay, …)`.
2. **Forward (float periods only).** The LIBOR simple forward \(F\) depends on **LIBOR** zeros at period start and end. The pricer computes \(\partial PV / \partial F = \text{sign} \times N \times \tau \times DF\), then `ibor_point_sensitivities` chains that through \(\partial F / \partial z_L(t_s)\) and \(\partial F / \partial z_L(t_e)\). Those become `PointSensitivity("IborRate", …)`.

Buy/sell only flips the sign. Output is a bag of point risks: each item is “this much PV per unit zero at this year-fraction, on this curve type.”

### Meaning of “zero” in that phrase

**“Zero”** means the **continuously compounded zero rate** \(z(t)\) on that curve at that maturity — the same \(z\) used in \(DF(t)=e^{-t\,z(t)}\) — not a market quote, not a discount factor, and not a 1bp bump.

A `PointSensitivity` is

\[
\frac{\partial\,\mathrm{PV}}{\partial z(t)}
\]

at one year-fraction \(t\) (ACT/365F from the valuation date).

- `"ZeroRate"`: \(z(t)\) on the **OIS discount** curve at the payment time.
- `"IborRate"`: \(z_L(t)\) on the **LIBOR** curve (the zero inside the pseudo-DF \(P_L(t)=e^{-t\,z_L(t)}\)) at the period start or end.

**“Per unit zero”** is a bump of \(+1.0\) in that rate (100%, not 1bp). If `value` is \(V\):

- \(+1\) in \(z(t)\) → PV moves by \(\approx V\)
- \(+1\)bp (\(+10^{-4}\)) in that same \(z(t)\) → PV moves by \(\approx V \times 10^{-4}\)

It is a **local** derivative: only the zero *at that \(t\)* on that curve. Neighbouring dates, interpolator nodes, and quotes are still ignored at this stage.

That is why the discount term is `forecast * (-df * yf)`: \(\partial DF/\partial z = -t\,DF\).

### `parameter_sensitivity` — project points onto nodes

On `ImmutableRatesProvider`. It maps each point onto the **discrete nodal zeros** of the right curve.

- `"ZeroRate"` → discount curve
- `"IborRate"` → LIBOR curve

For a point at year-fraction \(t\), the interpolator returns a weight vector \(w = \partial z(t) / \partial z_{\text{nodes}}\). The node-level risk for that point is \(w \times\) (the point’s value).

| Stage | Question | Depends on interpolator? |
|---|---|---|
| `present_value_sensitivity` | If \(z(t)\) at a payment/start/end date moves, how does swap PV move? | No (only on DF / forward formulas) |
| `parameter_sensitivity` | If a **node** zero moves, how does \(z(t)\) move? | Yes |

---

## Where interpolators play a role in `parameter_sensitivity`

They show up in **one** place: turning a zero at an off-node time \(t\) into weights on the **nodal** zeros.

For each point risk the method asks the curve `y_value_parameter_sensitivity(t)`, which is the bound interpolator’s `parameter_sensitivity(t)`:

\[
w_i = \frac{\partial z(t)}{\partial z_i}
\qquad
z(t)=\text{interpolator}(z_1,\ldots,z_n;\,t)
\]

Node risk is then \(w \times\) (the point’s \(\partial\mathrm{PV}/\partial z(t)\)). Which interpolator was stored on the curve (linear / DQS / NCS from the settings CSV) decides \(w\).

Outside the first/last node, all three **flat-extrapolate**: all weight on the first or last node.

### How the stencils differ

- **Linear** — local, two nodes. If \(t\) is between nodes \(i\) and \(i+1\), \(w_i=(t_{i+1}-t)/(t_{i+1}-t_i)\), \(w_{i+1}=1-w_i\). Everyone else is 0.
- **Double quadratic** — local, typically **four** nodes: two overlapping quadratics blended in the interval. Weight leaks to \(i-1\) and \(i+1\) as well, still compact.
- **Natural cubic spline** — **global**. Second derivatives depend on **all** nodes, so \(w\) is generally nonzero everywhere.

What interpolators do **not** do here:

- They do not create the point risks.
- They do not convert to quotes (`mqc.sensitivity` + \(J^{-1}\)).
- They are not re-run as “interpolate then finite-difference”; this is the analytic \(\partial z(t)/\partial z_{\text{nodes}}\).

---

## `MarketQuoteSensitivityCalculator.sensitivity`

Last chain-rule hop: **nodal-zero risk → market-quote risk**, using the inverse Jacobian stored at calibration time.

```
ps = dPV / d(nodal zeros)
J⁻¹ = d(zeros) / d(quotes)     # from RatesCurveCalibrator
mqs = dPV / d(quotes) = (dPV/dz) @ J⁻¹
```

It does not reprice the swap or bump quotes. It is a linear algebra conversion of `parameter_sensitivity` output.

### Inputs

- `ps`: for each curve, a vector of \(\partial PV / \partial z_i\) at that curve’s nodes.
- `provider`: calibrated curves, each carrying `jacobian` = that curve’s **rows** of \(J^{-1}\). Shape `(n_nodes_on_this_curve, n_all_quotes)`.

If a curve has no Jacobian, it raises.

### Per curve-parameter block

1. Look up that curve’s \(J^{-1}\).
2. Multiply: `market_quote_sens = ps.sensitivity @ jacobian` → \(\partial PV / \partial q\) for **every** calibration quote, from this curve’s zeros only.
3. `split_values` chops that full quote vector back into discount vs LIBOR buckets, using the same `curve_order` as calibration.

The split matters because calibration is **simultaneous**. A bump to an OIS quote generally moves **both** OIS and LIBOR zeros (and the other way around). Discount-curve zero risk can contribute to LIBOR quote buckets, and vice versa.

Those pieces are **added** into an accumulator per curve name.

### Output

A new `CurrencyParameterSensitivities` in provider curve order: one block per curve, same node **labels** as the calibration instruments, values = \(\partial PV / \partial q\) in absolute terms (rate bump of 1.0, not 1bp).

In the example, `export_mqs(..., BP1, ...)` then multiplies by \(10^{-4}\) so the CSV is **bucketed PV01 per 1bp quote bump**.

| Method | Risk factor | What it uses |
|---|---|---|
| `present_value_sensitivity` | zeros at cashflow dates | DF / forward formulas |
| `parameter_sensitivity` | nodal zeros | interpolator weights |
| `mqc.sensitivity` | market quotes | calibration \(J^{-1}\) |

Changing interpolator changes both the fitted zeros **and** \(J^{-1}\).

---

## Continuous compounding for discount factors

Yes. Discount factors are **continuously compounded zeros** with ACT/365F time.

```python
# ImmutableRatesProvider.discount_factor
z = c.y_value(yf)
return np.exp(-yf * z)
```

So \(DF(t) = e^{-t\,z(t)}\). The same convention is used for LIBOR **pseudo-DFs** when building the simple forward \(F = (P_L(s)/P_L(e)-1)/\tau\).

That is also why zero-rate sensitivity is \(\partial DF/\partial z = -t\,DF\).

`CurveSettings.value_type` is read from the settings CSV but **not used** when converting y-values to DFs. The code always treats curve nodes as continuously compounded zeros.

---

## Converting to quarterly / semi-annual / annual compounding

**Reporting** another compounding convention is a transform of the same discount factor. **Storing** nodes in that convention would also change `discount_factor` and the zero-rate Jacobian.

With this library’s time \(t\) (ACT/365F) and current continuous zero \(z_c\):

\[
DF(t)=e^{-t\,z_c}
\]

Periodic compounding \(m\) times a year (annual \(m=1\), semi \(m=2\), quarterly \(m=4\)):

\[
DF(t)=\Bigl(1+\frac{z_m}{m}\Bigr)^{-m t}
\]

Conversions that leave \(DF\) unchanged:

\[
z_m = m\bigl(e^{z_c/m}-1\bigr)
\qquad
z_c = m\ln\bigl(1+z_m/m\bigr)
\]

Examples: annual \(z_1=e^{z_c}-1\); semi \(z_2=2(e^{z_c/2}-1)\); quarterly \(z_4=4(e^{z_c/4}-1)\).

### 1. Convert for display only (keep this code as-is)

Keep interpolating and discounting with continuous zeros. After you have \(z_c(t)\) or \(DF(t)\):

```python
z_m = m * (np.exp(z_c / m) - 1.0)
# or equivalently
z_m = m * (df ** (-1.0 / (m * t)) - 1.0)
```

PV, calibration, and PV01 stay the same; only quoted zeros change.

### 2. Store curve nodes as periodic zeros

Then `discount_factor` cannot stay `exp(-yf * z)`:

```python
return (1.0 + z / m) ** (-m * yf)
```

Zero-rate sensitivity would change. Today it uses \(\partial DF/\partial z_c = -t\,DF\). Periodically compounded:

\[
\frac{\partial DF}{\partial z_m}
= -t\,\frac{DF}{1+z_m/m}
\]

`ibor_point_sensitivities` would need the same chain rule on LIBOR pseudo-DFs. Calibration still solves par-spread = 0; the Newton Jacobian is numeric, so it would pick up the new DF formula automatically. Analytic PV01 would not, unless those derivatives are updated.

`CurveSettings.value_type` is already loaded from CSV and ignored — that is the natural switch.

Do **not** mix conventions: if nodes are continuous, interpolate in \(z_c\) (or in \(DF\) / log-DF), then convert the interpolated value. Interpolating \(z_m\) and then converting is a different curve.

Simple compounding \(DF=1/(1+z t)\) is a fourth convention (money-market), not the same as annual compounding.
