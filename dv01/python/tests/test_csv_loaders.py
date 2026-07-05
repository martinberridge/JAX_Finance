"""Tests for CSV loaders."""

from datetime import date
from pathlib import Path

from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES
from pv01_loader.quotes_csv_loader import load, load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration

VALUATION_DATE = date(2016, 8, 1)


def test_quotes_loader_loads_gbp_quotes():
    quotes_path = PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv"
    quotes = load(VALUATION_DATE, quotes_path)
    assert len(quotes) == 29
    assert any(q.ticker == "GBP-OIS-3M" for q in quotes)


def test_calibration_loader_merges_group_settings_nodes():
    config_dir = PATH_CONFIG / CONFIG_STR
    result = load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}-linear-settings.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )
    assert CONFIG_STR in result
    group = result[CONFIG_STR]
    assert len(group.entries) == 3
    assert "GBP-DSCON-OIS" in group.curves
    assert "GBP-LIBOR6M-IRS" in group.curves
    assert len(group.curves["GBP-DSCON-OIS"].nodes) == 10
    assert len(group.curves["GBP-LIBOR6M-IRS"].nodes) == 9


def test_market_data_wrapper():
    quotes_path = PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv"
    md = load_market_data(VALUATION_DATE, quotes_path)
    assert md.valuation_date == VALUATION_DATE
    assert len(md.quotes) == 29
