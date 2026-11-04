from __future__ import annotations

import unittest
from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from pv01_examples.multicurve2.calibration_interpolation_example import (
    SWAP_NOTIONAL,
    SWAP_PERIOD_TO_START_MONTHS,
    SWAP_TENOR_YEARS,
    VALUATION_DATE,
)
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_jax.risk import (
    batch_market_quote_pv01,
    market_quote_pv01,
    portfolio_market_quote_pv01,
    stack_trades,
)
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import create_test_swap

SETTINGS_SUFFIX = "-linear-settings"


@lru_cache(maxsize=None)
def _market_data():
    return load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )


@lru_cache(maxsize=None)
def _problem():
    config_dir = PATH_CONFIG / CONFIG_STR
    config = load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}{SETTINGS_SUFFIX}.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]
    return resolve_calibration_problem(config, _market_data(), VALUATION_DATE)


def _swap_arrays(coupon: float, tenor_years: int = SWAP_TENOR_YEARS):
    swap = create_test_swap(
        VALUATION_DATE,
        SWAP_PERIOD_TO_START_MONTHS,
        tenor_years,
        BuySell.BUY,
        SWAP_NOTIONAL,
        coupon,
    )
    return resolve_swap(swap, VALUATION_DATE)


@lru_cache(maxsize=1)
def _portfolio_swaps():
    return tuple(
        _swap_arrays(float(coupon))
        for coupon in jnp.linspace(0.01, 0.04, 10)
    )


class TestPortfolioRisk(unittest.TestCase):
    def test_batched_market_quote_pv01_matches_per_trade_loop(self):
        problem = _problem()
        q = jnp.asarray(problem.q0)
        swaps = _portfolio_swaps()
        batched = batch_market_quote_pv01(problem, stack_trades(list(swaps)), q)
        expected = jnp.stack(
            [market_quote_pv01(problem, swap, q) for swap in swaps]
        )
        np.testing.assert_allclose(
            np.asarray(batched),
            np.asarray(expected),
            atol=1e-10,
            rtol=0.0,
        )

    def test_portfolio_pv01_matches_sum_of_batched_trade_pv01(self):
        problem = _problem()
        q = jnp.asarray(problem.q0)
        swaps = list(_portfolio_swaps())
        batched = batch_market_quote_pv01(problem, stack_trades(swaps), q)
        portfolio = portfolio_market_quote_pv01(problem, swaps, q)
        np.testing.assert_allclose(
            np.asarray(portfolio),
            np.asarray(batched.sum(axis=0)),
            atol=1e-8,
            rtol=0.0,
        )

    def test_stack_trades_rejects_different_tenor_shapes(self):
        swaps = [
            _swap_arrays(0.02, SWAP_TENOR_YEARS),
            _swap_arrays(0.02, SWAP_TENOR_YEARS - 1),
        ]
        with self.assertRaisesRegex(ValueError, "identical pytree structures and leaf shapes"):
            stack_trades(swaps)


if __name__ == "__main__":
    unittest.main()
