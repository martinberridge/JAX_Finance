from __future__ import annotations

import os
import unittest
from functools import lru_cache
from pathlib import Path
from statistics import median
from time import perf_counter

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
from pv01_examples.export_utils import read_mqs, read_pv
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_jax.risk import (
    cross_gamma,
    market_quote_pv01,
    parallel_pv01,
    pv_from_quotes,
)
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import create_test_swap
from tests.util.bump import bump_pv01

_PROJECT_DIR = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _PROJECT_DIR / "tests" / "golden"


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
def _problem(settings_suffix: str):
    return resolve_calibration_problem(
        _config(settings_suffix),
        _market_data(),
        VALUATION_DATE,
    )


@lru_cache(maxsize=None)
def _swap_arrays():
    swap = create_test_swap(
        VALUATION_DATE,
        SWAP_PERIOD_TO_START_MONTHS,
        SWAP_TENOR_YEARS,
        BuySell.BUY,
        SWAP_NOTIONAL,
        SWAP_COUPON,
    )
    return resolve_swap(swap, VALUATION_DATE)


@lru_cache(maxsize=None)
def _hand_coded_bump_pv01(settings_suffix: str):
    return bump_pv01(
        _config(settings_suffix),
        _market_data(),
        _resolved_swap(),
        VALUATION_DATE,
    )


@lru_cache(maxsize=None)
def _resolved_swap():
    return create_test_swap(
        VALUATION_DATE,
        SWAP_PERIOD_TO_START_MONTHS,
        SWAP_TENOR_YEARS,
        BuySell.BUY,
        SWAP_NOTIONAL,
        SWAP_COUPON,
    )


class TestPhase5MarketQuoteRisk(unittest.TestCase):
    def test_cross_gamma_is_symmetric(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                q0 = jnp.asarray(problem.q0)
                gamma = np.asarray(cross_gamma(problem, _swap_arrays(), q0))
                scale = np.max(np.abs(gamma))
                self.assertGreater(scale, 0.0)
                self.assertLessEqual(
                    np.max(np.abs(gamma - gamma.T)),
                    1e-6 * scale,
                    f"Cross-gamma asymmetry for {suffix}",
                )

    def test_cross_gamma_columns_match_finite_differences(self):
        h = 1e-5
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                swap_arrays = _swap_arrays()
                q0 = jnp.asarray(problem.q0)
                gamma = np.asarray(cross_gamma(problem, swap_arrays, q0))
                bumps = jnp.eye(q0.shape[0], dtype=q0.dtype) * h
                pv01_up = jax.vmap(
                    lambda bumped_q: market_quote_pv01(
                        problem,
                        swap_arrays,
                        bumped_q,
                    )
                )(q0 + bumps)
                pv01_down = jax.vmap(
                    lambda bumped_q: market_quote_pv01(
                        problem,
                        swap_arrays,
                        bumped_q,
                    )
                )(q0 - bumps)
                finite_difference = np.asarray(
                    ((pv01_up - pv01_down) / (2.0 * h) * 1e-4).T
                )
                scale = np.max(np.abs(gamma))
                self.assertGreater(scale, 0.0)
                self.assertLessEqual(
                    np.max(np.abs(gamma - finite_difference)),
                    1e-6 * scale,
                    f"Cross-gamma finite-difference mismatch for {suffix}",
                )

    def test_jitted_cross_gamma_reports_execution_time(self):
        timings = []
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                swap_arrays = _swap_arrays()
                q0 = jnp.asarray(problem.q0)
                compiled = jax.jit(
                    lambda quotes: cross_gamma(problem, swap_arrays, quotes)
                )
                compiled(q0).block_until_ready()
                start = perf_counter()
                compiled(q0).block_until_ready()
                timings.append((suffix, perf_counter() - start))

        print("\nJitted cross-gamma execution time:")
        print("| Settings suffix | Wall-time (ms) |")
        print("|---|---:|")
        for suffix, duration in timings:
            print(f"| {suffix} | {duration * 1000:.6f} |")

    def test_parallel_pv01_matches_sum_of_quote_buckets(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                swap_arrays = _swap_arrays()
                q0 = jnp.asarray(problem.q0)
                parallel = parallel_pv01(problem, swap_arrays, q0)
                buckets = market_quote_pv01(problem, swap_arrays, q0).sum()
                self.assertAlmostEqual(
                    float(parallel),
                    float(buckets),
                    delta=1e-8,
                    msg=f"Parallel PV01 for {suffix}",
                )

    def test_market_quote_pv01_matches_goldens(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                swap_arrays = _swap_arrays()
                q0 = jnp.asarray(problem.q0)
                actual_pv01 = np.asarray(
                    market_quote_pv01(problem, swap_arrays, q0)
                )
                golden_mqs_path = (
                    _GOLDEN_DIR / f"{CONFIG_STR}{suffix}-mqs.csv"
                )
                expected_mqs = read_mqs(golden_mqs_path)
                self.assertEqual(
                    tuple(expected_mqs),
                    problem.quote_labels,
                )
                for label, value in zip(problem.quote_labels, actual_pv01):
                    self.assertAlmostEqual(
                        float(value),
                        expected_mqs[label],
                        delta=0.01,
                        msg=f"{suffix}, {label}",
                    )

                actual_pv = float(pv_from_quotes(problem, swap_arrays, q0))
                expected_pv = read_pv(
                    _GOLDEN_DIR / f"{CONFIG_STR}{suffix}-pv.csv"
                )
                self.assertAlmostEqual(
                    actual_pv,
                    expected_pv,
                    delta=0.01,
                    msg=f"PV for {suffix}",
                )

    def test_market_quote_pv01_matches_quote_finite_differences(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                swap_arrays = _swap_arrays()
                q0 = jnp.asarray(problem.q0)
                actual = np.asarray(
                    market_quote_pv01(problem, swap_arrays, q0)
                )
                quote_count = q0.shape[0]
                bumps = jnp.eye(quote_count, dtype=q0.dtype) * 1e-4
                up = jax.vmap(
                    lambda bumped_q: pv_from_quotes(
                        problem,
                        swap_arrays,
                        bumped_q,
                    )
                )(q0 + bumps)
                down = jax.vmap(
                    lambda bumped_q: pv_from_quotes(
                        problem,
                        swap_arrays,
                        bumped_q,
                    )
                )(q0 - bumps)
                finite_difference = np.asarray((up - down) / 2.0)
                np.testing.assert_allclose(
                    actual,
                    finite_difference,
                    atol=0.01,
                    rtol=0.0,
                )

    def test_jitted_market_quote_pv01_traces_once_and_reports_median_time(self):
        timings = []
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                swap_arrays = _swap_arrays()
                q0 = jnp.asarray(problem.q0)
                trace_count = []

                @jax.jit
                def compiled(quotes):
                    trace_count.append(None)
                    return market_quote_pv01(problem, swap_arrays, quotes)

                compiled(q0).block_until_ready()
                compiled(q0 + 1e-4).block_until_ready()
                self.assertEqual(len(trace_count), 1)

                durations = []
                for _ in range(20):
                    start = perf_counter()
                    compiled(q0).block_until_ready()
                    durations.append(perf_counter() - start)
                timings.append((suffix, median(durations)))

        print("\nMedian jitted market-quote PV01 time (20 calls):")
        print("| Settings suffix | Median wall-time (ms) |")
        print("|---|---:|")
        for suffix, duration in timings:
            print(f"| {suffix} | {duration * 1000:.6f} |")

    @unittest.skipUnless(os.environ.get("RUN_SLOW"), "slow bump-and-recalibrate test")
    def test_market_quote_pv01_matches_hand_coded_bumps(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                q0 = jnp.asarray(problem.q0)
                actual = np.asarray(
                    market_quote_pv01(problem, _swap_arrays(), q0)
                )
                expected = _hand_coded_bump_pv01(suffix)
                self.assertEqual(set(expected), set(problem.quote_labels))
                for label, value in zip(problem.quote_labels, actual):
                    self.assertAlmostEqual(
                        float(value),
                        expected[label],
                        delta=0.01,
                        msg=f"{suffix}, {label}",
                    )


if __name__ == "__main__":
    unittest.main()
