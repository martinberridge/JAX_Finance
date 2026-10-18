import unittest
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from pv01_basics.day_count import act_365f_between
from pv01_examples.multicurve2.calibration_interpolation_example import (
    SWAP_COUPON,
    SWAP_NOTIONAL,
    SWAP_PERIOD_TO_START_MONTHS,
    SWAP_TENOR_YEARS,
    VALUATION_DATE,
)
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_jax.curves import curves_from
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import (
    create_test_swap,
    resolve_calibration_trade,
)


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


def _jax_parameters(problem, provider):
    return jnp.concatenate(
        tuple(
            jnp.asarray(provider.find_curve(spec.name).y_values, dtype=jnp.float64)
            for spec in problem.curves
        )
    )


class TestPhase2Curves(unittest.TestCase):
    def test_discount_factors_for_swap_payment_periods(self):
        swap_arrays = resolve_swap(_swap(), VALUATION_DATE)
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                provider = _provider(suffix)
                problem = resolve_calibration_problem(
                    _config(suffix), _market_data(), VALUATION_DATE
                )
                z0 = _jax_parameters(problem, provider)
                curves = curves_from(problem, z0)
                discount_curve = curves[provider.discount_curve.name]

                for period, t_pay in zip(_swap().float_leg, swap_arrays.t_pay):
                    actual = float(discount_curve.df(t_pay))
                    expected = provider.discount_factor(period.payment)
                    self.assertLessEqual(abs(actual - expected), 1e-14)

    def test_libor_forwards_for_swap_periods(self):
        swap = _swap()
        swap_arrays = resolve_swap(swap, VALUATION_DATE)
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                provider = _provider(suffix)
                problem = resolve_calibration_problem(
                    _config(suffix), _market_data(), VALUATION_DATE
                )
                z0 = _jax_parameters(problem, provider)
                curves = curves_from(problem, z0)
                libor_curve = curves[provider.libor_curve.name]

                for i, period in enumerate(swap.float_leg):
                    actual = float(
                        libor_curve.forward(
                            swap_arrays.t_fix_start[i],
                            swap_arrays.t_fix_end[i],
                            swap_arrays.tau_fix[i],
                        )
                    )
                    expected = provider.forward_rate(period.start, period.end)
                    self.assertLessEqual(abs(actual - expected), 1e-14)

    def test_ois_compounding_matches_production_formula(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                config = _config(suffix)
                market_data = _market_data()
                provider = _provider(suffix)
                problem = resolve_calibration_problem(
                    config, market_data, VALUATION_DATE
                )
                curves = curves_from(problem, _jax_parameters(problem, provider))
                discount_curve = curves[provider.discount_curve.name]

                ois_nodes = (
                    node
                    for curve in config.curves.values()
                    for node in curve.nodes
                    if node.node_type == "OIS"
                )
                for node in ois_nodes:
                    trade = resolve_calibration_trade(node, VALUATION_DATE, market_data)
                    for period in trade.periods:
                        start = act_365f_between(VALUATION_DATE, period.start)
                        end = act_365f_between(VALUATION_DATE, period.end)
                        actual = float(discount_curve.compounded(start, end))
                        expected = (
                            provider.discount_factor(period.start)
                            / provider.discount_factor(period.end)
                            - 1.0
                        )
                        self.assertLessEqual(abs(actual - expected), 1e-14)

    def test_curve_parameters_match_calibrated_values_and_are_isolated(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                provider = _provider(suffix)
                problem = resolve_calibration_problem(
                    _config(suffix), _market_data(), VALUATION_DATE
                )
                z0 = _jax_parameters(problem, provider)
                curves = curves_from(problem, z0)

                for spec in problem.curves:
                    start, end = spec.param_slice
                    np.testing.assert_array_equal(
                        np.asarray(curves[spec.name].z),
                        provider.find_curve(spec.name).y_values,
                    )
                    self.assertEqual(
                        tuple(spec.param_slice),
                        (start, end),
                    )

                discount_slice = next(
                    spec.param_slice
                    for spec in problem.curves
                    if spec.name == provider.discount_curve.name
                )
                evaluation_times = jnp.asarray(
                    [1.0, 2.5, 5.0], dtype=jnp.float64
                )

                def libor_discount_sum(global_parameters):
                    libor_curve = curves_from(problem, global_parameters)[
                        provider.libor_curve.name
                    ]
                    return libor_curve.df(evaluation_times).sum()

                gradient = np.asarray(jax.grad(libor_discount_sum)(z0))
                discount_start, discount_end = discount_slice
                np.testing.assert_array_equal(
                    gradient[discount_start:discount_end],
                    np.zeros(discount_end - discount_start),
                )
                self.assertTrue(np.all(np.isfinite(gradient)))


if __name__ == "__main__":
    unittest.main()
