import unittest
from functools import lru_cache

import numpy as np

from pv01_basics.day_count import act_365f_between, act_365f_year_fraction
from pv01_examples.multicurve2.calibration_interpolation_example import VALUATION_DATE
from pv01_examples.paths import (
    CONFIG_STR,
    PATH_CONFIG,
    PATH_QUOTES,
    SETTINGS_SUFFIXES,
)
from pv01_jax.resolve import (
    build_calibration_problem,
    par_spread_from_calibration_block,
)
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.instrument_pricers import par_spread
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_product.resolved_trades import resolve_calibration_trade


@lru_cache(maxsize=None)
def _market_data():
    return load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )


@lru_cache(maxsize=None)
def _config(suffix):
    config_dir = PATH_CONFIG / CONFIG_STR
    return load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}{suffix}.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]


@lru_cache(maxsize=None)
def _provider(suffix):
    return RatesCurveCalibrator.standard().calibrate(
        _config(suffix), _market_data(), VALUATION_DATE
    )


class TestBuildCalibrationProblem(unittest.TestCase):
    def test_quote_mapping_par_spreads_and_period_counts(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                self._assert_problem_matches_reference(suffix)

    def _assert_problem_matches_reference(self, suffix):
        config = _config(suffix)
        market_data = _market_data()
        problem = build_calibration_problem(config, market_data, VALUATION_DATE)
        provider = _provider(suffix)

        curve_names = sorted(config.curves)
        curve_names.sort(key=lambda name: (0 if "DSCON" in name else 1, name))
        resolved_by_kind = {"FIX": [], "OIS": [], "IRS": []}
        expected_q0 = []
        expected_curve_start = 0

        self.assertEqual(len(problem.curves), len(curve_names))
        for curve_spec, curve_name in zip(problem.curves, curve_names):
            definition = config.curves[curve_name]
            for node in definition.nodes:
                trade = resolve_calibration_trade(node, VALUATION_DATE, market_data)
                quote_idx = len(expected_q0)
                quote = market_data.get_quote(node.quote_id)
                expected_q0.append(quote)
                resolved_by_kind[trade.trade_type.value].append((trade, quote_idx))

            curve_end = expected_curve_start + len(definition.nodes)
            self.assertEqual(curve_spec.param_slice, (expected_curve_start, curve_end))
            self.assertEqual(curve_spec.labels, tuple(node.label for node in definition.nodes))
            self.assertEqual(curve_spec.interpolator, definition.settings.interpolator)
            expected_x_nodes = [
                act_365f_between(
                    VALUATION_DATE,
                    resolve_calibration_trade(node, VALUATION_DATE, market_data).node_date,
                )
                for node in definition.nodes
            ]
            np.testing.assert_allclose(
                np.asarray(curve_spec.x_nodes), np.asarray(expected_x_nodes), rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                np.asarray(problem.q0)[expected_curve_start:curve_end],
                np.asarray([market_data.get_quote(node.quote_id) for node in definition.nodes]),
            )
            expected_curve_start = curve_end

        np.testing.assert_allclose(np.asarray(problem.q0), np.asarray(expected_q0))
        self.assertEqual(
            len(np.asarray(problem.q0)),
            sum(len(config.curves[name].nodes) for name in curve_names),
        )
        blocks = {block.trade_type: block for block in problem.blocks}
        self.assertEqual(set(blocks), {kind for kind, trades in resolved_by_kind.items() if trades})

        for kind, trades in resolved_by_kind.items():
            if not trades:
                continue
            block = blocks[kind]
            self.assertEqual(len(np.asarray(block.quote_idx)), len(trades))
            trade_ids = np.asarray(block.trade_id)
            period_counts = np.bincount(trade_ids, minlength=len(trades))
            np.testing.assert_array_equal(np.asarray(block.float_leg.rate), 0.0)

            if kind == "FIX":
                self.assertIsNone(block.fixed_leg)
            else:
                np.testing.assert_array_equal(np.asarray(block.fixed_leg.rate), 0.0)

            for trade_id, (trade, quote_idx) in enumerate(trades):
                self.assertEqual(int(np.asarray(block.quote_idx)[trade_id]), quote_idx)
                self.assertAlmostEqual(
                    float(np.asarray(problem.q0)[quote_idx]), trade.market_rate, delta=0.0
                )
                self.assertEqual(int(period_counts[trade_id]), len(trade.periods))

                actual_spread = par_spread_from_calibration_block(
                    block, trade_id, np.asarray(problem.q0), provider
                )
                expected_spread = par_spread(trade, provider)
                self.assertLessEqual(
                    abs(actual_spread - expected_spread),
                    1e-12,
                    f"{suffix} {kind} {trade.label}: "
                    f"{actual_spread} != {expected_spread}",
                )

                if kind != "FIX":
                    self.assertEqual(
                        len(np.asarray(block.tau_fix)[trade_ids == trade_id]),
                        len(trade.periods),
                    )
                    expected_tau = np.asarray(
                        [
                            act_365f_year_fraction(period.start, period.end)
                            for period in trade.periods
                        ]
                    )
                    np.testing.assert_allclose(
                        np.asarray(block.tau_fix)[trade_ids == trade_id],
                        expected_tau,
                        rtol=0.0,
                        atol=0.0,
                    )


if __name__ == "__main__":
    unittest.main()
