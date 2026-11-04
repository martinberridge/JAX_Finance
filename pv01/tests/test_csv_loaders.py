"""Tests for CSV loaders."""

import unittest
from datetime import date

from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES
from pv01_loader.quotes_csv_loader import load, load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration

VALUATION_DATE = date(2016, 8, 1)


class TestCsvLoaders(unittest.TestCase):
    def test_quotes_loader_loads_gbp_quotes(self):
        quotes_path = PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv"
        quotes = load(VALUATION_DATE, quotes_path)
        self.assertEqual(len(quotes), 29)
        self.assertTrue(any(q.ticker == "GBP-OIS-3M" for q in quotes))

    def test_calibration_loader_merges_group_settings_nodes(self):
        config_dir = PATH_CONFIG / CONFIG_STR
        result = load_calibration(
            config_dir / f"{CONFIG_STR}-group.csv",
            config_dir / f"{CONFIG_STR}-linear-settings.csv",
            config_dir / f"{CONFIG_STR}-nodes.csv",
        )
        self.assertIn(CONFIG_STR, result)
        group = result[CONFIG_STR]
        self.assertEqual(len(group.entries), 3)
        self.assertIn("GBP-DSCON-OIS", group.curves)
        self.assertIn("GBP-LIBOR6M-IRS", group.curves)
        self.assertEqual(len(group.curves["GBP-DSCON-OIS"].nodes), 10)
        self.assertEqual(len(group.curves["GBP-LIBOR6M-IRS"].nodes), 9)

    def test_market_data_wrapper(self):
        quotes_path = PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv"
        md = load_market_data(VALUATION_DATE, quotes_path)
        self.assertEqual(md.valuation_date, VALUATION_DATE)
        self.assertEqual(len(md.quotes), 29)
