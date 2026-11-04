import unittest
from functools import lru_cache
from statistics import median
from time import perf_counter

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
from pv01_jax.arrays import CurveSpec
from pv01_jax.curves import Curve, curves_from
from pv01_jax.interp import INTERPOLATORS
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


class TestCurves(unittest.TestCase):
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

    def test_jit_trace_stability_and_interpolator_retrace(self):
        provider = _provider(SETTINGS_SUFFIXES[0])
        problem = resolve_calibration_problem(
            _config(SETTINGS_SUFFIXES[0]), _market_data(), VALUATION_DATE
        )
        params = _jax_parameters(problem, provider)
        spec = problem.curves[0]
        start, end = spec.param_slice
        z = params[start:end]
        curve = Curve(spec=spec, z=z)
        query_times = jnp.linspace(-1.0, 35.0, 1000, dtype=jnp.float64)
        trace_count = []

        @jax.jit
        def evaluate_df(curve_arg, t):
            trace_count.append(None)
            return curve_arg.df(t)

        for offset in (0.0, 1e-4, -1e-4):
            varied_curve = Curve(spec=spec, z=z + offset)
            evaluate_df(varied_curve, query_times).block_until_ready()
        self.assertEqual(len(trace_count), 1)

        alternate_name = next(
            name for name in INTERPOLATORS if name != spec.interpolator
        )
        alternate_spec = CurveSpec(
            name=spec.name,
            param_slice=spec.param_slice,
            labels=spec.labels,
            interpolator=alternate_name,
            x_nodes=spec.x_nodes,
        )
        alternate_curve = Curve(spec=alternate_spec, z=z)
        evaluate_df(alternate_curve, query_times).block_until_ready()
        self.assertEqual(len(trace_count), 2)

    def test_interpolator_df_median_timing(self):
        provider = _provider(SETTINGS_SUFFIXES[0])
        problem = resolve_calibration_problem(
            _config(SETTINGS_SUFFIXES[0]), _market_data(), VALUATION_DATE
        )
        params = _jax_parameters(problem, provider)
        base_spec = problem.curves[0]
        start, end = base_spec.param_slice
        z = params[start:end]
        query_times = jnp.linspace(-1.0, 35.0, 1000, dtype=jnp.float64)
        timings = []

        for interpolator_name in INTERPOLATORS:
            spec = CurveSpec(
                name=base_spec.name,
                param_slice=base_spec.param_slice,
                labels=base_spec.labels,
                interpolator=interpolator_name,
                x_nodes=base_spec.x_nodes,
            )
            curve = Curve(spec=spec, z=z)
            evaluate_df = jax.jit(lambda curve_arg, t: curve_arg.df(t))
            evaluate_df(curve, query_times).block_until_ready()

            run_times = []
            for _ in range(100):
                start_time = perf_counter()
                result = evaluate_df(curve, query_times)
                result.block_until_ready()
                run_times.append(perf_counter() - start_time)
            timings.append((interpolator_name, median(run_times)))

        print("\nMedian jitted df time over 1000 queries (100 runs):")
        print("| Interpolator | Median wall-time (ms) |")
        print("|---|---:|")
        for interpolator_name, median_seconds in timings:
            print(f"| {interpolator_name} | {median_seconds * 1000:.6f} |")


if __name__ == "__main__":
    unittest.main()
