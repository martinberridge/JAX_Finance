from __future__ import annotations

import unittest
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from pv01_examples.multicurve2.calibration_interpolation_example import VALUATION_DATE
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_jax.calibrate import calibrate, calibrate_with_info
from pv01_jax.resolve import resolve_calibration_problem
from pv01_loader.models import MarketData, RatesCurveGroupDefinition
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_pricer.rates_provider import ImmutableRatesProvider


@lru_cache(maxsize=None)
def _market_data() -> MarketData:
    return load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )


@lru_cache(maxsize=None)
def _config(settings_suffix: str) -> RatesCurveGroupDefinition:
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


def _inputs(settings_suffix: str):
    config = _config(settings_suffix)
    market_data = _market_data()
    provider = _provider(settings_suffix)
    problem = resolve_calibration_problem(config, market_data, VALUATION_DATE)
    q0 = jnp.asarray(problem.q0)
    z_expected = np.concatenate(
        [
            provider.find_curve(spec.name).y_values
            for spec in problem.curves
        ]
    )
    return problem, q0, z_expected


class TestPhase4Calibration(unittest.TestCase):
    def test_calibrate_matches_hand_coded_calibrator(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem, q0, z_expected = _inputs(suffix)
                actual = calibrate(problem, q0)
                np.testing.assert_allclose(
                    np.asarray(actual),
                    z_expected,
                    atol=1e-12,
                    rtol=0.0,
                )

    def test_calibrate_with_info_reports_convergence(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem, q0, _ = _inputs(suffix)
                _, iterations, max_residual = calibrate_with_info(problem, q0)
                self.assertLessEqual(int(iterations), 10)
                self.assertLess(float(max_residual), 1e-14)

    def test_jitted_calibrate_traces_once_for_changing_quotes(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem, q0, _ = _inputs(suffix)
                trace_count = []

                @jax.jit
                def compiled(quotes):
                    trace_count.append(None)
                    return calibrate(problem, quotes)

                for shift in (0.0, 1e-4, -1e-4):
                    compiled(q0 + shift).block_until_ready()

                self.assertEqual(len(trace_count), 1)


if __name__ == "__main__":
    unittest.main()
