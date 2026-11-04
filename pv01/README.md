# pv01

Swap PV and market-quote PV01 by automatic differentiation through curve calibration. The
kernel is `pv01_jax`. A NumPy port of Strata's `CalibrationInterpolationExample`, with
hand-coded sensitivities, is kept as the reference it's checked against.

The formulas and benchmark results are in the [root README](../README.md), and
[`docs/pv01_jax.md`](../docs/pv01_jax.md) describes each `pv01_jax` module and the known
differences from Strata.

## Packages

| Layer | Package | Role |
|---|---|---|
| **JAX kernel** | `pv01_jax` | Arrays, interpolators, curves, pricers, Newton calibration with implicit derivatives, risk. `import pv01_jax` loads only JAX, Equinox and NumPy. |
| **Inputs** | `pv01_basics` | Calendars, day counts, schedules, tenors |
| | `pv01_loader` | Reads the CSV curve configuration and quotes |
| | `pv01_product` | Trade definitions and resolved trades |
| **NumPy reference** | `pv01_market` | Interpolated curves and the calibration Jacobian |
| | `pv01_pricer` | Calibrator, hand-coded pricer sensitivities, legacy `bump_pv01` |
| **Scripts** | `pv01_examples` | Example runs, shared constants, CSV export and read helpers |
| | `benchmarks` | Timing scripts, with results in `benchmarks/results/`. Not installed. |

The input and reference layers are frozen. Only `pv01_jax.resolve` and `pv01_jax.workflow`
connect the kernel to the inputs, which is why `pv01_jax` doesn't re-export them.

`data/` holds the GBP curve configuration and quotes. They aren't packaged, so the install
must be editable. Generated output goes to `target/`, which git ignores.

## Running

From this directory, after the [Quickstart](../README.md#quickstart):

```bash
python -m pv01_examples.multicurve2.calibration_interpolation_example  # NumPy reference
python pv01_examples/jax/calibration_interpolation_example_jax.py      # JAX, compared with the goldens
python -m benchmarks.real_config    # add --slow for the legacy NCS bump
python -m benchmarks.scaling
```

## Tests

```bash
python -m unittest discover -s tests -t .
RUN_SLOW=1 python -m unittest discover -s tests -t .  # adds the slow legacy bump comparison
```

`tests/` checks the NumPy reference, and `tests/jax_port/` checks `pv01_jax`. The phase
column maps each module to its prompts in [`notes/prompts/`](../notes/prompts/).

| Module | What it checks | Phase |
|---|---|---|
| `test_reference_correctness` | NumPy sensitivities against bump-and-recalibrate and finite differences | |
| `test_calibration_interpolation_example` | NumPy example output against the goldens | |
| `test_csv_loaders`, `test_interpolators` | Loading and NumPy interpolation | |
| `jax_port/test_scaffold`, `test_arrays` | Package imports, pytree containers | 1 |
| `jax_port/test_resolve_swap`, `test_build_calibration_problem` | Trades and config to arrays | 1 |
| `jax_port/test_jit_ordering` | Quote order and tracing under `jit` | 1 |
| `jax_port/test_curves`, `test_interp` | Curves and interpolators against the NumPy reference | 2 |
| `jax_port/test_pricers`, `test_residuals`, `test_resolve_calibration_problem` | Swap PV and calibration residuals | 3 |
| `jax_port/test_zero_rate_risk` | Zero-rate PV01 against the hand-coded sensitivities | 3 |
| `jax_port/test_newton`, `test_calibrate` | Newton solver and convergence guard | 4 |
| `jax_port/test_implicit_derivatives` | dz/dq against finite differences and the hand-coded Jacobian | 4 |
| `jax_port/test_market_quote_risk`, `test_portfolio` | Market-quote PV01, parallel PV01, cross-gamma, portfolios | 5 |
| `jax_port/test_jax_example` | JAX example output against the goldens | 5 |
| `jax_port/test_check_grads`, `test_stress` | `check_grads`, stressed curves | 6 |
| `jax_port/test_frozen_goldens` | JAX risk against `jax_port/golden_jax/` | 6 |
| `jax_port/test_bump`, `test_synthetic`, `test_workflow` | JAX bumps, structural checks on synthetic curves, `price_and_risk` | Refactor |

The goldens in `tests/golden/` and `tests/jax_port/golden_jax/` are output from this code,
not from Strata. They show that a change preserved the numbers; correctness comes from the
hand-coded sensitivities and bump-and-recalibrate comparisons. Don't regenerate them to make
a test pass. [`notes/reference-corrections.md`](../notes/reference-corrections.md) records
the fixes that led to the current values.
