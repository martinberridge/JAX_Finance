"""Compare JAX zero-rate PV01 with the hand-coded pricer sensitivities."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp

from pv01_examples.multicurve2.calibration_interpolation_example import (
    SWAP_COUPON,
    SWAP_NOTIONAL,
    SWAP_PERIOD_TO_START_MONTHS,
    SWAP_TENOR_YEARS,
    VALUATION_DATE,
)
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_jax.risk import as_labelled, zero_rate_pv01
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_pricer.swap_pricer import DiscountingSwapTradePricer
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import create_test_swap


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
    calibrator = RatesCurveCalibrator.standard()

    for suffix in SETTINGS_SUFFIXES:
        config = load_calibration(
            config_dir / f"{CONFIG_STR}-group.csv",
            config_dir / f"{CONFIG_STR}{suffix}.csv",
            config_dir / f"{CONFIG_STR}-nodes.csv",
        )[CONFIG_STR]
        provider = calibrator.calibrate(config, market_data, VALUATION_DATE)
        problem = resolve_calibration_problem(config, market_data, VALUATION_DATE)
        z0 = jnp.asarray(
            [
                float(value)
                for spec in problem.curves
                for value in provider.find_curve(spec.name).y_values
            ]
        )
        swap_arrays = resolve_swap(swap, VALUATION_DATE)
        jax_values = as_labelled(
            problem,
            zero_rate_pv01(problem, swap_arrays, z0),
        )

        parameter_sensitivities = provider.parameter_sensitivity(
            DiscountingSwapTradePricer.present_value_sensitivity(swap, provider)
        )
        hand_coded_values = {label: 0.0 for label in problem.quote_labels}
        for sensitivity in parameter_sensitivities.sensitivities:
            for label, value in zip(sensitivity.labels, sensitivity.sensitivity):
                hand_coded_values[label] += float(value) * 1e-4

        differences = {
            label: jax_values[label] - hand_coded_values[label]
            for label in problem.quote_labels
        }
        largest_difference = max(
            (abs(difference) for difference in differences.values()),
            default=0.0,
        )

        print(f"\n{suffix}")
        print("| Node label | Hand-coded zero-rate PV01 | JAX zero-rate PV01 | Difference |")
        print("|---|---:|---:|---:|")
        for label in problem.quote_labels:
            print(
                f"| {label} | {hand_coded_values[label]:.10f} "
                f"| {jax_values[label]:.10f} | {differences[label]:.3e} |"
            )
        print(f"Largest absolute difference: {largest_difference:.6e} GBP/bp")


if __name__ == "__main__":
    main()
