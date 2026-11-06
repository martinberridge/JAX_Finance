# Derived in part from OpenGamma Strata (https://github.com/OpenGamma/Strata):
#   CalibrationInterpolationExample
# Copyright (C) 2017 - present by OpenGamma Inc. and the OpenGamma group of companies
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.txt.
# Changed by Martin Berridge, 2026: example constants translated to Python; added
#   load_example_inputs.

"""Test-swap and valuation constants shared by the examples and benchmarks."""

from datetime import date

from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import create_test_swap

VALUATION_DATE = date(2016, 8, 1)
SWAP_TENOR_YEARS = 8
SWAP_PERIOD_TO_START_MONTHS = 6
SWAP_COUPON = 0.025
SWAP_NOTIONAL = 10_000_000.0
BP1 = 1.0e-4


def load_example_inputs(settings_suffix: str):
    """Return ``(config, market_data, swap)`` for the GBP example config.

    ``settings_suffix`` is one of ``paths.SETTINGS_SUFFIXES`` and picks the
    curve interpolator. The swap is the 8y test swap from the constants above.
    """
    config_dir = PATH_CONFIG / CONFIG_STR
    config = load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}{settings_suffix}.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]
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
    return config, market_data, swap


__all__ = [
    "VALUATION_DATE",
    "SWAP_TENOR_YEARS",
    "SWAP_PERIOD_TO_START_MONTHS",
    "SWAP_COUPON",
    "SWAP_NOTIONAL",
    "BP1",
    "load_example_inputs",
]
