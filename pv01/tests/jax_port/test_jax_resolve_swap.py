import unittest
from functools import lru_cache

import numpy as np

from pv01_examples.multicurve2.calibration_interpolation_example import (
    SWAP_COUPON,
    SWAP_NOTIONAL,
    SWAP_PERIOD_TO_START_MONTHS,
    SWAP_TENOR_YEARS,
    VALUATION_DATE,
)
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_jax.resolve import pv_from_arrays, resolve_swap
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_pricer.swap_pricer import DiscountingSwapTradePricer
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import create_test_swap


@lru_cache(maxsize=None)
def _market_data():
    return load_market_data(VALUATION_DATE, PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv")


@lru_cache(maxsize=None)
def _config():
    config_dir = PATH_CONFIG / CONFIG_STR
    return load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}-linear-settings.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]


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


@lru_cache(maxsize=None)
def _provider():
    calibrator = RatesCurveCalibrator.standard()
    return calibrator.calibrate(_config(), _market_data(), VALUATION_DATE)


class TestPhase1ResolveSwap(unittest.TestCase):
    def test_resolve_swap_matches_reference_pv(self):
        swap = _swap()
        arrays = resolve_swap(swap, VALUATION_DATE)

        expected = DiscountingSwapTradePricer.present_value(swap, _provider())["GBP"]
        actual = pv_from_arrays(arrays, _provider())

        self.assertEqual(len(arrays.fixed_leg.start), len(swap.fixed_leg))
        self.assertEqual(len(arrays.float_leg.start), len(swap.float_leg))
        self.assertTrue(np.all(np.diff(arrays.t_pay) > 0.0))
        self.assertTrue(np.all(arrays.trade_id == 0))
        self.assertAlmostEqual(actual, expected, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
