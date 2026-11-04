import ast
import unittest
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

from pv01_examples.multicurve2.calibration_interpolation_example import (
    SWAP_COUPON,
    SWAP_NOTIONAL,
    SWAP_PERIOD_TO_START_MONTHS,
    SWAP_TENOR_YEARS,
    VALUATION_DATE,
)
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_jax.curves import curves_from
from pv01_jax.pricers import pv_swap
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
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


def _z0(problem, provider):
    return np.concatenate(
        [
            provider.find_curve(spec.name).y_values
            for spec in problem.curves
        ]
    )


class TestSwapPricer(unittest.TestCase):
    def test_swap_pv_matches_production_pricer(self):
        swap = _swap()
        swap_arrays = resolve_swap(swap, VALUATION_DATE)

        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                provider = _provider(suffix)
                problem = resolve_calibration_problem(
                    _config(suffix),
                    _market_data(),
                    VALUATION_DATE,
                )
                curves = curves_from(problem, _z0(problem, provider))
                actual = pv_swap(
                    swap_arrays,
                    curves,
                    problem.discount_curve,
                    problem.forward_curve,
                )
                expected = DiscountingSwapTradePricer.present_value(
                    swap,
                    provider,
                )["GBP"]
                self.assertLessEqual(abs(float(actual) - expected), 1e-6)

    def test_pv01_jax_has_no_production_pricer_or_market_imports(self):
        package_dir = Path(__file__).resolve().parents[2] / "pv01_jax"
        forbidden_roots = {"pv01_pricer", "pv01_market"}

        for source_path in package_dir.rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots = {
                        alias.name.split(".", 1)[0] for alias in node.names
                    }
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots = {node.module.split(".", 1)[0]}
                else:
                    continue
                self.assertFalse(
                    forbidden_roots.intersection(imported_roots),
                    f"{source_path} imports a forbidden production package",
                )


if __name__ == "__main__":
    unittest.main()
