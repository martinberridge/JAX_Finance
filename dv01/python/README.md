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

## Tests

```bash
cd python
pip install -r requirements.txt
pip install -e .
pytest
```

Run `pytest` from the `python/` directory so imports resolve correctly.
