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

## Tests

```bash
cd python
pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -v
```

Run unittest discovery from the `python/` directory so imports resolve correctly.
