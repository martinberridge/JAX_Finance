from __future__ import annotations

import unittest
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from pv01_examples.multicurve2.calibration_interpolation_example import VALUATION_DATE
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_jax.pricers import residuals
from pv01_jax.resolve import resolve_calibration_problem
from pv01_loader.models import MarketData, RatesCurveGroupDefinition
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.instrument_pricers import par_spread
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_pricer.rates_provider import ImmutableRatesProvider
from pv01_product.resolved_trades import ResolvedTrade, resolve_calibration_trade
from pv01_pricer.bump import bumped_market_data

_ATOL = 1e-14
_FD_STEP = 1e-6


@lru_cache(maxsize=None)
def _market_data() -> MarketData:
    return load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )


@lru_cache(maxsize=None)
def _config(settings_suffix: str) -> RatesCurveGroupDefinition:
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


def _nodes_by_label(config: RatesCurveGroupDefinition):
    return {
        node.label: node
        for curve_definition in config.curves.values()
        for node in curve_definition.nodes
    }


def _trades_in_quote_order(
    problem,
    config: RatesCurveGroupDefinition,
    market_data: MarketData,
) -> list[ResolvedTrade]:
    nodes = _nodes_by_label(config)
    return [
        resolve_calibration_trade(nodes[label], VALUATION_DATE, market_data)
        for label in problem.quote_labels
    ]


def _quotes_in_quote_order(
    problem,
    config: RatesCurveGroupDefinition,
    market_data: MarketData,
) -> np.ndarray:
    nodes = _nodes_by_label(config)
    return np.asarray(
        [market_data.get_quote(nodes[label].quote_id) for label in problem.quote_labels],
        dtype=np.float64,
    )


def _parameters_in_curve_order(problem, provider: ImmutableRatesProvider) -> np.ndarray:
    return np.concatenate(
        [provider.find_curve(spec.name).y_values for spec in problem.curves]
    )


def _provider_with_parameters(
    problem,
    provider: ImmutableRatesProvider,
    parameters: np.ndarray,
) -> ImmutableRatesProvider:
    curves = {
        spec.name: provider.find_curve(spec.name).with_y_values(
            parameters[spec.param_slice[0] : spec.param_slice[1]]
        )
        for spec in problem.curves
    }
    return ImmutableRatesProvider(
        valuation_date=provider.valuation_date,
        discount_curve=curves[problem.discount_curve],
        libor_curve=curves[problem.forward_curve],
        curve_order=provider.curve_order,
    )


def _oracle_residuals(
    trades: list[ResolvedTrade],
    provider: ImmutableRatesProvider,
) -> np.ndarray:
    return np.asarray([par_spread(trade, provider) for trade in trades])


class TestCalibrationResiduals(unittest.TestCase):
    def _inputs(self, settings_suffix: str):
        config = _config(settings_suffix)
        market_data = _market_data()
        provider = _provider(settings_suffix)
        problem = resolve_calibration_problem(config, market_data, VALUATION_DATE)
        z0 = _parameters_in_curve_order(problem, provider)
        q0 = _quotes_in_quote_order(problem, config, market_data)
        np.testing.assert_array_equal(q0, np.asarray(problem.q0))
        trades = _trades_in_quote_order(problem, config, market_data)
        return config, market_data, provider, problem, z0, q0, trades

    def test_residual_parity_away_from_solution(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                _, _, provider, problem, z0, q0, trades = self._inputs(suffix)
                z = z0 + 0.001
                shifted_provider = _provider_with_parameters(problem, provider, z)
                actual = residuals(problem, jnp.asarray(z), jnp.asarray(q0))
                expected = _oracle_residuals(trades, shifted_provider)
                np.testing.assert_allclose(
                    np.asarray(actual),
                    expected,
                    atol=_ATOL,
                    rtol=0.0,
                )

    def test_quote_bump_parity(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                config, market_data, _, problem, z0, _, _ = self._inputs(suffix)
                bumps = {
                    node.quote_id: 0.0005
                    for curve_definition in config.curves.values()
                    for node in curve_definition.nodes
                }
                bumped_data = bumped_market_data(market_data, bumps)
                q_bumped = _quotes_in_quote_order(problem, config, bumped_data)
                bumped_trades = _trades_in_quote_order(
                    problem,
                    config,
                    bumped_data,
                )
                actual = residuals(problem, jnp.asarray(z0), jnp.asarray(q_bumped))
                expected = _oracle_residuals(bumped_trades, _provider(suffix))
                np.testing.assert_allclose(
                    np.asarray(actual),
                    expected,
                    atol=_ATOL,
                    rtol=0.0,
                )

    def test_at_solution_residuals(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                _, _, _, problem, z0, q0, _ = self._inputs(suffix)
                actual = residuals(problem, jnp.asarray(z0), jnp.asarray(q0))
                self.assertLess(float(jnp.max(jnp.abs(actual))), 1e-12)

    def test_quote_jacobian_is_negative_identity(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                _, _, _, problem, z0, q0, _ = self._inputs(suffix)
                actual = jax.jacfwd(residuals, argnums=2)(
                    problem,
                    jnp.asarray(z0),
                    jnp.asarray(q0),
                )
                np.testing.assert_allclose(
                    np.asarray(actual),
                    -np.eye(len(q0)),
                    atol=_ATOL,
                    rtol=0.0,
                )

    def test_curve_jacobian_matches_central_finite_difference(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                _, _, provider, problem, z0, q0, trades = self._inputs(suffix)
                actual = np.asarray(
                    jax.jacfwd(residuals, argnums=1)(
                        problem,
                        jnp.asarray(z0),
                        jnp.asarray(q0),
                    )
                )
                finite_difference = np.empty_like(actual)
                for column in range(len(z0)):
                    up = z0.copy()
                    down = z0.copy()
                    up[column] += _FD_STEP
                    down[column] -= _FD_STEP
                    provider_up = _provider_with_parameters(problem, provider, up)
                    provider_down = _provider_with_parameters(problem, provider, down)
                    finite_difference[:, column] = (
                        _oracle_residuals(trades, provider_up)
                        - _oracle_residuals(trades, provider_down)
                    ) / (2.0 * _FD_STEP)

                condition = np.linalg.cond(actual)
                print(f"{suffix}: cond(J) = {condition:.6g}")
                np.testing.assert_allclose(
                    actual,
                    finite_difference,
                    atol=1e-8,
                    rtol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
