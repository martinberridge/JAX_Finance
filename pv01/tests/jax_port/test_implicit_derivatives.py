from __future__ import annotations

import unittest
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from pv01_examples.multicurve2.calibration_interpolation_example import (
    SWAP_COUPON,
    SWAP_NOTIONAL,
    SWAP_PERIOD_TO_START_MONTHS,
    SWAP_TENOR_YEARS,
    VALUATION_DATE,
)
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_jax.calibrate import _calibrate_with_solver, calibrate, newton_solve
from pv01_jax.curves import curves_from
from pv01_jax.pricers import pv_swap
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_pricer.rates_provider import ImmutableRatesProvider
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import create_test_swap

_FD_STEP = 1e-6


@lru_cache(maxsize=None)
def _market_data():
    return load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )


@lru_cache(maxsize=None)
def _config(settings_suffix: str):
    config_dir = PATH_CONFIG / CONFIG_STR
    return load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}{settings_suffix}.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]


@lru_cache(maxsize=None)
def _provider(settings_suffix: str) -> ImmutableRatesProvider:
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


def _inputs(settings_suffix: str):
    provider = _provider(settings_suffix)
    problem = resolve_calibration_problem(
        _config(settings_suffix),
        _market_data(),
        VALUATION_DATE,
    )
    q0 = jnp.asarray(problem.q0)
    return provider, problem, q0


def _hand_coded_calibration_jacobian(
    provider: ImmutableRatesProvider,
    problem,
) -> np.ndarray:
    return np.vstack(
        [
            provider.find_curve(spec.name).jacobian.jacobian_matrix
            for spec in problem.curves
        ]
    )


def _fixed_iteration_newton(f, z0):
    def step(_, z):
        def update(current):
            return current - jnp.linalg.solve(jax.jacfwd(f)(current), f(current))

        return jax.lax.cond(
            jnp.max(jnp.abs(f(z))) > 1e-14,
            update,
            lambda current: current,
            z,
        )

    return jax.lax.fori_loop(0, 20, step, z0)


def _swap_pv_from_quotes(problem, swap_arrays, quotes, solve):
    z = _calibrate_with_solver(problem, quotes, solve)
    curves = curves_from(problem, z)
    return pv_swap(
        swap_arrays,
        curves,
        problem.discount_curve,
        problem.forward_curve,
    )


class TestImplicitDerivatives(unittest.TestCase):
    def test_quote_jacobian_matches_hand_coded_calibration_jacobian(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                provider, problem, q0 = _inputs(suffix)
                actual = np.asarray(jax.jacfwd(calibrate, argnums=1)(problem, q0))
                expected = _hand_coded_calibration_jacobian(provider, problem)
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=1e-8,
                    rtol=0.0,
                )

    def test_quote_jacobian_matches_finite_difference_recalibration(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                _, problem, q0 = _inputs(suffix)
                jacobian = np.asarray(
                    jax.jacfwd(calibrate, argnums=1)(problem, q0)
                )
                quote_count = q0.shape[0]
                bumps = jnp.eye(quote_count, dtype=q0.dtype) * _FD_STEP
                up = jax.vmap(lambda q: calibrate(problem, q))(q0 + bumps)
                down = jax.vmap(lambda q: calibrate(problem, q))(q0 - bumps)
                finite_difference = np.asarray((up - down).T / (2.0 * _FD_STEP))
                np.testing.assert_allclose(
                    jacobian,
                    finite_difference,
                    atol=1e-6,
                    rtol=0.0,
                )

    def test_swap_quote_gradient_is_independent_of_newton_iterations(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                _, problem, q0 = _inputs(suffix)
                swap_arrays = resolve_swap(_swap(), VALUATION_DATE)
                tolerance_gradient = jax.grad(
                    lambda q: _swap_pv_from_quotes(
                        problem,
                        swap_arrays,
                        q,
                        newton_solve,
                    )
                )(q0)
                fixed_iteration_gradient = jax.grad(
                    lambda q: _swap_pv_from_quotes(
                        problem,
                        swap_arrays,
                        q,
                        _fixed_iteration_newton,
                    )
                )(q0)
                np.testing.assert_allclose(
                    np.asarray(tolerance_gradient),
                    np.asarray(fixed_iteration_gradient),
                    atol=1e-10,
                    rtol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
