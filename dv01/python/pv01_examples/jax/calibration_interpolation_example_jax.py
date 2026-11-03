"""Run JAX calibration, market-quote PV01, and reference comparisons."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from pv01_examples.common import (
    BP1,
    SWAP_COUPON,
    SWAP_NOTIONAL,
    SWAP_PERIOD_TO_START_MONTHS,
    SWAP_TENOR_YEARS,
    VALUATION_DATE,
)
from pv01_examples.export_utils import export_mqs, export_pv, read_mqs
from pv01_examples.paths import (
    CONFIG_STR,
    PATH_CONFIG,
    PATH_QUOTES,
    PYTHON_DIR,
    SETTINGS_SUFFIXES,
)
from pv01_jax.calibrate import calibrate
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_jax.risk import market_quote_pv01, pv_from_quotes
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_product.conventions import BuySell
from pv01_pricer.bump import bump_pv01
from pv01_product.resolved_trades import create_test_swap

PATH_JAX_RESULTS = PYTHON_DIR / "target" / "example-output-jax"
GOLDEN_DIR = PYTHON_DIR / "tests" / "golden"


def _sensitivity_export(problem, pv01):
    sensitivities = []
    for curve in problem.curves:
        start, end = curve.param_slice
        sensitivities.append(
            SimpleNamespace(
                market_data_name=curve.name,
                currency="GBP",
                labels=list(curve.labels),
                sensitivity=np.asarray(pv01[start:end]) / BP1,
            )
        )
    return SimpleNamespace(sensitivities=sensitivities)


def _largest_golden_difference(problem, pv01, suffix: str) -> float:
    golden = read_mqs(
        GOLDEN_DIR / f"{CONFIG_STR}{suffix}-mqs.csv"
    )
    values = {
        label: float(value)
        for label, value in zip(problem.quote_labels, np.asarray(pv01))
    }
    missing = set(golden).difference(values)
    if missing:
        raise ValueError(f"JAX sensitivities are missing golden labels: {sorted(missing)}")
    return max(
        (abs(values[label] - expected) for label, expected in golden.items()),
        default=0.0,
    )


def main() -> None:
    config_dir = PATH_CONFIG / CONFIG_STR
    market_data = load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )
    swap = create_test_swap(
        VALUATION_DATE,
        SWAP_PERIOD_TO_START_MONTHS,
        SWAP_TENOR_YEARS,
        BuySell.BUY,
        SWAP_NOTIONAL,
        SWAP_COUPON,
    )
    swap_arrays = resolve_swap(swap, VALUATION_DATE)

    # each interpolation config 
    for suffix in SETTINGS_SUFFIXES:
        config = load_calibration(
            config_dir / f"{CONFIG_STR}-group.csv",
            config_dir / f"{CONFIG_STR}{suffix}.csv",
            config_dir / f"{CONFIG_STR}-nodes.csv",
        )[CONFIG_STR]
        problem = resolve_calibration_problem(
            config,
            market_data,
            VALUATION_DATE,
        )
        quotes = jnp.asarray(problem.q0)

        # smoke test for calibration
        calibrated_parameters = calibrate(problem, quotes)
        calibrated_parameters.block_until_ready() # make python wait until JAX has finished computing and array
        
        pv = float(pv_from_quotes(problem, swap_arrays, quotes))
        pv01 = market_quote_pv01(problem, swap_arrays, quotes)

        output_stem = PATH_JAX_RESULTS / f"{CONFIG_STR}{suffix}"
        export_pv({"GBP": pv}, f"{output_stem}-pv.csv")
        export_mqs(
            _sensitivity_export(problem, pv01),
            BP1,
            f"{output_stem}-mqs.csv",
        )

        golden_difference = _largest_golden_difference(problem, pv01, suffix)
        # time function compiled to XLA 
        compiled_pv01 = jax.jit(
            lambda q: market_quote_pv01(problem, swap_arrays, q)
        )
        #timing
        compiled_pv01(quotes).block_until_ready() # warm-up - tigger JIT compile and wait until queue is empty
        start = perf_counter()
        compiled_pv01(quotes).block_until_ready() # wait for the compute device to finishe
        jax_elapsed = perf_counter() - start

        print(f"\n{suffix}")
        print(f"Largest absolute golden MQS difference: {golden_difference:.8e} GBP/bp")
        print(f"JAX market-quote PV01 time: {jax_elapsed:.6f}s")
        if suffix == "-ncs-settings" and not os.environ.get("RUN_SLOW"):
            print("Hand-coded bump timing skipped (set RUN_SLOW=1 to include NCS).")
        else:
            start = perf_counter()
            bump_pv01(config, market_data, swap, VALUATION_DATE)
            hand_coded_elapsed = perf_counter() - start
            print(f"Hand-coded bump-and-recalibrate time: {hand_coded_elapsed:.6f}s")

    print(f"\nJAX outputs written to: {PATH_JAX_RESULTS}")


if __name__ == "__main__":
    main()
