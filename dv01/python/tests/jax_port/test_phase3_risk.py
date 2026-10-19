from __future__ import annotations

import unittest
from functools import lru_cache
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

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
from pv01_pricer.rates_provider import ImmutableRatesProvider
from pv01_pricer.swap_pricer import DiscountingSwapTradePricer
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import create_test_swap


@lru_cache(maxsize=None)
def _market_data():
    return load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )


@lru_cache(maxsize=None)
def _config(settings_suffix: str):
    config_dir = PATH_CONFIG / CONFIG_STR
    return load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}{settings_suffix}.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]


@lru_cache(maxsize=None)
def _provider(settings_suffix: str) -> ImmutableRatesProvider:
    return RatesCurveCalibrator.standard().calibrate(
        _config(settings_suffix),
        _market_data(),
        VALUATION_DATE,
    )


@lru_cache(maxsize=None)
def _swap():
    return create_test_swap(
        VALUATION_DATE,
        SWAP_PERIOD_TO_START_MONTHS,
        SWAP_TENOR_YEARS,
        BuySell.BUY,
        SWAP_NOTIONAL,
        SWAP_COUPON,
    )


class TestPhase3ZeroRatePV01(unittest.TestCase):
    def test_zero_rate_pv01_matches_hand_coded_parameter_sensitivities(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                provider = _provider(suffix)
                problem = resolve_calibration_problem(
                    _config(suffix),
                    _market_data(),
                    VALUATION_DATE,
                )
                z0 = np.concatenate(
                    [
                        provider.find_curve(spec.name).y_values
                        for spec in problem.curves
                    ]
                )
                swap_arrays = resolve_swap(_swap(), VALUATION_DATE)
                actual = as_labelled(
                    problem,
                    zero_rate_pv01(problem, swap_arrays, jnp.asarray(z0)),
                )

                parameter_sensitivities = provider.parameter_sensitivity(
                    DiscountingSwapTradePricer.present_value_sensitivity(
                        _swap(),
                        provider,
                    )
                )
                expected = {label: 0.0 for label in problem.quote_labels}
                for sensitivity in parameter_sensitivities.sensitivities:
                    for label, value in zip(
                        sensitivity.labels,
                        sensitivity.sensitivity,
                    ):
                        expected[label] += float(value) * 1e-4

                self.assertEqual(list(actual), list(problem.quote_labels))
                self.assertEqual(set(actual), set(expected))
                for label in problem.quote_labels:
                    self.assertAlmostEqual(
                        actual[label],
                        expected[label],
                        delta=1e-6,
                        msg=f"{suffix}, {label}",
                    )

    def test_jitted_zero_rate_pv01_traces_once_for_changing_parameters(self):
        suffix = SETTINGS_SUFFIXES[0]
        provider = _provider(suffix)
        problem = resolve_calibration_problem(
            _config(suffix),
            _market_data(),
            VALUATION_DATE,
        )
        z0 = np.concatenate(
            [provider.find_curve(spec.name).y_values for spec in problem.curves]
        )
        swap_arrays = resolve_swap(_swap(), VALUATION_DATE)
        trace_count = []

        @jax.jit
        def compiled(parameters):
            trace_count.append(None)
            return zero_rate_pv01(problem, swap_arrays, parameters)

        for offset in (0.0, 1e-4, -1e-4):
            compiled(jnp.asarray(z0 + offset)).block_until_ready()

        self.assertEqual(len(trace_count), 1)

    def test_jitted_zero_rate_pv01_median_timing(self):
        timings = []
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                provider = _provider(suffix)
                problem = resolve_calibration_problem(
                    _config(suffix),
                    _market_data(),
                    VALUATION_DATE,
                )
                z0 = np.concatenate(
                    [
                        provider.find_curve(spec.name).y_values
                        for spec in problem.curves
                    ]
                )
                swap_arrays = resolve_swap(_swap(), VALUATION_DATE)
                compiled = jax.jit(
                    lambda parameters: zero_rate_pv01(
                        problem,
                        swap_arrays,
                        parameters,
                    )
                )
                parameters = jnp.asarray(z0)
                compiled(parameters).block_until_ready()

                run_times = []
                for _ in range(100):
                    start_time = perf_counter()
                    result = compiled(parameters)
                    result.block_until_ready()
                    run_times.append(perf_counter() - start_time)
                timings.append((suffix, median(run_times)))

        print("\nMedian jitted zero-rate PV01 time (100 runs):")
        print("| Settings suffix | Median wall-time (ms) |")
        print("|---|---:|")
        for suffix, median_seconds in timings:
            print(f"| {suffix} | {median_seconds * 1000:.6f} |")


if __name__ == "__main__":
    unittest.main()
