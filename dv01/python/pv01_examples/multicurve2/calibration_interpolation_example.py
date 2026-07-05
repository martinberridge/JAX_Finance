"""
Calibrates one set of curves with several interpolators, computes bucketed PV01 and exports CSV.

Python port of CalibrationInterpolationExample.java.
"""

import sys, os

print("EXEC:", sys.executable)
print("CWD:", os.getcwd())
print("PATH:", sys.path)

from datetime import date

from pv01_examples.export_utils import export_mqs, export_pv
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, PATH_RESULTS, SETTINGS_SUFFIXES
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.market_quote_sensitivity import MarketQuoteSensitivityCalculator
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_pricer.swap_pricer import DiscountingSwapTradePricer
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import create_test_swap

VALUATION_DATE = date(2016, 8, 1)
SWAP_TENOR_YEARS = 8
SWAP_PERIOD_TO_START_MONTHS = 6
SWAP_COUPON = 0.025
SWAP_NOTIONAL = 10_000_000.0
BP1 = 1.0e-4

SUFFIX_CSV = ".csv"
GROUPS_SUFFIX = "-group"
NODES_SUFFIX = "-nodes"


def main() -> None:
    config_dir = PATH_CONFIG / CONFIG_STR
    group_path = config_dir / f"{CONFIG_STR}{GROUPS_SUFFIX}{SUFFIX_CSV}"
    nodes_path = config_dir / f"{CONFIG_STR}{NODES_SUFFIX}{SUFFIX_CSV}"
    quotes_path = PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv"

    market_quotes = load_market_data(VALUATION_DATE, quotes_path)

    configs = []
    for suffix in SETTINGS_SUFFIXES:
        settings_path = config_dir / f"{CONFIG_STR}{suffix}{SUFFIX_CSV}"
        configs.append(load_calibration(group_path, settings_path, nodes_path))

    swap = create_test_swap(
        VALUATION_DATE,
        SWAP_PERIOD_TO_START_MONTHS,
        SWAP_TENOR_YEARS,
        BuySell.BUY,
        SWAP_NOTIONAL,
        SWAP_COUPON,
    )

    calibrator = RatesCurveCalibrator.standard()
    pricer = DiscountingSwapTradePricer.DEFAULT
    mqc = MarketQuoteSensitivityCalculator.DEFAULT

    for i, suffix in enumerate(SETTINGS_SUFFIXES):
        config = configs[i][CONFIG_STR]
        provider = calibrator.calibrate(config, market_quotes, VALUATION_DATE)

        pv = pricer.present_value(swap, provider)
        pts = pricer.present_value_sensitivity(swap, provider)
        ps = provider.parameter_sensitivity(pts)
        mqs = mqc.sensitivity(ps, provider)

        export_mqs(mqs, BP1, str(PATH_RESULTS / f"{CONFIG_STR}{suffix}-mqs{SUFFIX_CSV}"))
        export_pv(pv, str(PATH_RESULTS / f"{CONFIG_STR}{suffix}-pv{SUFFIX_CSV}"))

    print(f"Calibration and export finished: {CONFIG_STR}")


if __name__ == "__main__":
    main()
 

 