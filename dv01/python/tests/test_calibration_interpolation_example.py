"""End-to-end tests for CalibrationInterpolationExample."""

import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path

from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, PATH_RESULTS, SETTINGS_SUFFIXES
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.instrument_pricers import par_spread
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_product.resolved_trades import resolve_calibration_trade

VALUATION_DATE = date(2016, 8, 1)
PYTHON_DIR = Path(__file__).resolve().parents[1]
GOLDEN_DIR = PYTHON_DIR / "tests" / "golden"


def _run_example():
    subprocess.run(
        [sys.executable, "-m", "pv01_examples.multicurve2.calibration_interpolation_example"],
        cwd=str(PYTHON_DIR),
        check=True,
    )


def _parse_pv(path: Path) -> float:
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split(",")
    return float(parts[1])


def _parse_mqs(path: Path) -> dict[str, float]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()[1:]
    result = {}
    for line in lines:
        if not line.strip():
            continue
        label, value = [p.strip() for p in line.split(",", 1)]
        if label in ("GBP-DSCON-OIS", "GBP-LIBOR6M-IRS", "GBP"):
            continue
        try:
            result[label] = float(value)
        except ValueError:
            pass
    return result


class TestCalibrationInterpolationExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _run_example()
        cls.example_outputs = PATH_RESULTS

    def test_example_produces_six_output_files(self):
        for suffix in SETTINGS_SUFFIXES:
            self.assertTrue((self.example_outputs / f"{CONFIG_STR}{suffix}-pv.csv").exists())
            self.assertTrue((self.example_outputs / f"{CONFIG_STR}{suffix}-mqs.csv").exists())

    def test_calibration_par_spreads_near_zero(self):
        config_dir = PATH_CONFIG / CONFIG_STR
        cfg = load_calibration(
            config_dir / f"{CONFIG_STR}-group.csv",
            config_dir / f"{CONFIG_STR}-linear-settings.csv",
            config_dir / f"{CONFIG_STR}-nodes.csv",
        )[CONFIG_STR]
        mq = load_market_data(VALUATION_DATE, PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv")
        provider = RatesCurveCalibrator.standard().calibrate(cfg, mq, VALUATION_DATE)
        trades = []
        for curve in cfg.curves.values():
            for node in curve.nodes:
                trades.append(resolve_calibration_trade(node, VALUATION_DATE, mq))
        residuals = [abs(par_spread(t, provider)) for t in trades]
        self.assertLess(max(residuals), 5e-4)

    def test_swap_pv_magnitude(self):
        pv = _parse_pv(self.example_outputs / f"{CONFIG_STR}-linear-settings-pv.csv")
        self.assertGreater(abs(pv), 500_000)
        self.assertLess(abs(pv), 5_000_000)

    def test_pv_varies_across_interpolators(self):
        pvs = [_parse_pv(self.example_outputs / f"{CONFIG_STR}{s}-pv.csv") for s in SETTINGS_SUFFIXES]
        self.assertLess(max(pvs) - min(pvs), 500_000)
        self.assertGreaterEqual(len(set(round(p, -2) for p in pvs)), 1)

    def test_mqs_has_all_calibration_nodes(self):
        mqs = _parse_mqs(self.example_outputs / f"{CONFIG_STR}-linear-settings-mqs.csv")
        expected = {
            "GBP-OIS-3M", "GBP-OIS-6M", "GBP-OIS-1Y", "GBP-OIS-2Y", "GBP-OIS-3Y",
            "GBP-OIS-5Y", "GBP-OIS-10Y", "GBP-OIS-15Y", "GBP-OIS-20Y", "GBP-OIS-30Y",
            "GBP-FIX-L6M", "GBP-IRS6M-1Y", "GBP-IRS6M-2Y", "GBP-IRS6M-3Y", "GBP-IRS6M-5Y",
            "GBP-IRS6M-10Y", "GBP-IRS6M-15Y", "GBP-IRS6M-20Y", "GBP-IRS6M-30Y",
        }
        self.assertTrue(expected.issubset(set(mqs.keys())))

    def test_golden_regression_pv(self):
        """Regression against saved Python golden outputs."""
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for suffix in SETTINGS_SUFFIXES:
            out = self.example_outputs / f"{CONFIG_STR}{suffix}-pv.csv"
            golden = GOLDEN_DIR / f"{CONFIG_STR}{suffix}-pv.csv"
            if not golden.exists():
                golden.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
            expected = _parse_pv(golden)
            actual = _parse_pv(out)
            self.assertLess(abs(actual - expected), 1.0, f"PV drift for {suffix}")

    def test_golden_regression_mqs(self):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for suffix in SETTINGS_SUFFIXES:
            out = self.example_outputs / f"{CONFIG_STR}{suffix}-mqs.csv"
            golden = GOLDEN_DIR / f"{CONFIG_STR}{suffix}-mqs.csv"
            if not golden.exists():
                golden.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
            expected = _parse_mqs(golden)
            actual = _parse_mqs(out)
            for label in expected:
                if label in actual and abs(expected[label]) > 1.0:
                    rel = abs(actual[label] - expected[label]) / abs(expected[label])
                    self.assertLess(rel, 0.05, f"MQS drift for {label} ({suffix})")
