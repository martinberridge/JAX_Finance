import unittest
from functools import lru_cache

from pv01_examples.multicurve2.calibration_interpolation_example import VALUATION_DATE
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_jax.resolve import resolve_calibration_problem
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator


@lru_cache(maxsize=None)
def _market_data():
    return load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )


@lru_cache(maxsize=None)
def _config(settings_suffix):
    config_dir = PATH_CONFIG / CONFIG_STR
    return load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}{settings_suffix}.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]


@lru_cache(maxsize=None)
def _provider(settings_suffix):
    return RatesCurveCalibrator.standard().calibrate(
        _config(settings_suffix),
        _market_data(),
        VALUATION_DATE,
    )


class TestResolveCalibrationProblemCurveNames(unittest.TestCase):
    def test_curve_names_match_calibrated_provider(self):
        for settings_suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=settings_suffix):
                problem = resolve_calibration_problem(
                    _config(settings_suffix),
                    _market_data(),
                    VALUATION_DATE,
                )
                provider = _provider(settings_suffix)
                problem_curve_names = {curve.name for curve in problem.curves}

                self.assertEqual(
                    problem.discount_curve,
                    provider.discount_curve.name,
                )
                self.assertEqual(
                    problem.forward_curve,
                    provider.libor_curve.name,
                )
                self.assertIn(problem.discount_curve, problem_curve_names)
                self.assertIn(problem.forward_curve, problem_curve_names)


if __name__ == "__main__":
    unittest.main()
