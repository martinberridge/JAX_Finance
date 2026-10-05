# Strata Python — CalibrationInterpolationExample

Native Python port of `CalibrationInterpolationExample.java`.

## Run

From the Strata repo root (or `python/` directory):

```bash
cd python
pip install -r requirements.txt
pip install -e .
python -m pv01_examples.multicurve2.calibration_interpolation_example
```

Or run without installing by ensuring your cwd is `python/` (pytest adds it to `PYTHONPATH` automatically).

## Inputs

Reads CSV files from `examples/src/main/resources/example-calibration/`.

## Outputs

Writes to `python/target/example-output/`:

- `GBP-DSCONOIS-L6MIRS-FRTB-{linear,dq,ncs}-settings-pv.csv`
- `GBP-DSCONOIS-L6MIRS-FRTB-{linear,dq,ncs}-settings-mqs.csv`

## Phase 0 corrections

The reference-correctness work fixed three pricing and calibration bugs:

1. Incorrect fixed-leg discount-factor sign.
2. Incorrect OIS compounding.
3. Incorrect calibration residual sign.

The table compares the previous checked-in golden baselines with the regenerated outputs
after these corrections. Swap PV is in GBP; total market-quote PV01 is in GBP/bp.

| Interpolator | Swap PV before | Swap PV after | Total market-quote PV01 before | Total market-quote PV01 after |
| --- | ---: | ---: | ---: | ---: |
| Linear | -1,346,861.73 | -1,374,209.14 | -7,382.13 | 8,555.74 |
| Double quadratic | -1,345,394.24 | -1,370,678.63 | -6,627.87 | 8,554.41 |
| Natural cubic spline | -1,333,083.90 | -1,364,721.26 | -4,110.39 | 8,550.36 |

## LIBOR forward from pseudo-discount curve

`forward_rate` used to return the LIBOR curve's zero rate at the period end, `z_L(t_e)`. That made
each IRS par rate behave like an average of zero rates, so mapping through the inverse calibration
Jacobian produced large IRS buckets that alternated in sign. The forward is now the simple period
forward on a pseudo-discount curve built from the LIBOR zero rates:

```
P_L(t) = exp(-t * z_L(t))
F      = (P_L(s) / P_L(e) - 1) / tau,   tau = act/365F(s, e)
```

`F` depends on `z_L` at both `t_s` and `t_e`, so the swap's hand-coded adjoint now emits two LIBOR
point sensitivities per floating period, `dPV/dF * dF/dz_L(t_s)` and `dPV/dF * dF/dz_L(t_e)`. Both
are mapped through the LIBOR curve interpolator. The calibration also changes, because FIX and IRS
par spreads use the same `forward_rate`.

Market-quote PV01 per node, in GBP/bp ("before" is the Phase 0 output):

| Node | Linear before | Linear after | DQ before | DQ after | NCS before | NCS after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GBP-OIS-3M | 0.00 | 0.00 | 0.00 | 0.00 | -4.83 | -4.99 |
| GBP-OIS-6M | -0.43 | -0.37 | -2.66 | -2.66 | 13.93 | 14.70 |
| GBP-OIS-1Y | 14.97 | 15.00 | 16.12 | 16.01 | -12.85 | -14.65 |
| GBP-OIS-2Y | 34.20 | 34.56 | 21.33 | 22.55 | 110.53 | 114.77 |
| GBP-OIS-3Y | 87.89 | 87.00 | 30.47 | 27.89 | -91.94 | -97.13 |
| GBP-OIS-5Y | 325.29 | 319.66 | 411.62 | 408.83 | 467.42 | 468.79 |
| GBP-OIS-10Y | 194.88 | 199.53 | 198.40 | 205.75 | 198.88 | 205.48 |
| GBP-OIS-15Y | 0.02 | 0.02 | -21.69 | -26.75 | -39.19 | -43.28 |
| GBP-OIS-20Y | 0.00 | 0.00 | 0.03 | 0.43 | 8.08 | 9.25 |
| GBP-OIS-30Y | 0.00 | 0.00 | 0.00 | 0.00 | -0.53 | -0.55 |
| GBP-FIX-L6M | -440.93 | -503.77 | -28.29 | -503.69 | 1,270.06 | -307.32 |
| GBP-IRS6M-1Y | -371.59 | -0.09 | -1,745.65 | 0.02 | -4,943.59 | -585.26 |
| GBP-IRS6M-2Y | 1,977.13 | -0.21 | 5,524.74 | 0.73 | 9,642.43 | 1,749.00 |
| GBP-IRS6M-3Y | -4,449.90 | -1.25 | -8,556.59 | -931.60 | -11,039.35 | -3,422.83 |
| GBP-IRS6M-5Y | 6,117.77 | 2,496.15 | 7,632.64 | 3,185.26 | 7,645.29 | 4,539.03 |
| GBP-IRS6M-10Y | 5,066.44 | 5,908.13 | 5,308.90 | 6,780.28 | 5,818.05 | 6,750.39 |
| GBP-IRS6M-15Y | 0.00 | 0.00 | -211.96 | -628.63 | -561.30 | -1,024.94 |
| GBP-IRS6M-20Y | 0.00 | 0.00 | -22.39 | -1.53 | 71.50 | 217.61 |
| GBP-IRS6M-30Y | 0.00 | 0.00 | -0.59 | -0.00 | -2.21 | -14.80 |
| **Total** | **8,555.74** | **8,554.36** | **8,554.41** | **8,552.87** | **8,550.36** | **8,553.27** |

| Interpolator | Swap PV before | Swap PV after |
| --- | ---: | ---: |
| Linear | -1,374,209.14 | -1,367,900.79 |
| Double quadratic | -1,370,678.63 | -1,366,361.40 |
| Natural cubic spline | -1,364,721.26 | -1,368,040.96 |

For linear and double-quadratic interpolation, the IRS buckets no longer alternate. The 6M-forward
8Y swap's LIBOR risk now sits in the 5Y and 10Y buckets, which bracket its 8.5Y end. The
short-end risk sits in FIX-L6M, which pins the forward at the swap start. The 1Y and 2Y IRS buckets
are about zero. Double quadratic keeps some local overshoot at 3Y and 15Y, from its quadratic
blending across neighbouring nodes. The natural cubic spline still alternates across 1Y to 3Y,
because its curve is non-local, but the amplitude is about a third of what it was. The total PV01
is essentially unchanged, about 8,550 to 8,555 GBP/bp for every interpolator.

## Tests

```bash
cd python
pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -v
```

Run unittest discovery from the `python/` directory so imports resolve correctly.
