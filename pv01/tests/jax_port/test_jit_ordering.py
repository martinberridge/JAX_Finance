import unittest
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pv01_examples.export_utils import read_mqs
from pv01_examples.paths import (
    CONFIG_STR,
    PATH_CONFIG,
    PATH_QUOTES,
    SETTINGS_SUFFIXES,
)
from pv01_jax.resolve import resolve_calibration_problem
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration


VALUATION_DATE = date(2016, 8, 1)
PROJECT_DIR = Path(__file__).resolve().parents[2]
GOLDEN_DIR = PROJECT_DIR / "tests" / "golden"


def _market_data():
    return load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )


def _problem(settings_suffix):
    config_dir = PATH_CONFIG / CONFIG_STR
    config = load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}{settings_suffix}.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]
    return resolve_calibration_problem(config, _market_data(), VALUATION_DATE)


class TestJitOrdering(unittest.TestCase):
    def test_jit_cache_uses_static_interpolator_settings(self):
        problems = [_problem(SETTINGS_SUFFIXES[0]), _problem(SETTINGS_SUFFIXES[1])]
        trace_count = []

        @jax.jit
        def sum_quotes(problem):
            trace_count.append(None)
            return jnp.sum(problem.q0)

        first_result = sum_quotes(problems[0])
        second_result = sum_quotes(problems[0])
        self.assertEqual(float(first_result), float(second_result))
        self.assertEqual(len(trace_count), 1)

        other_result = sum_quotes(problems[1])
        self.assertEqual(float(first_result), float(other_result))
        self.assertEqual(len(trace_count), 2)

    def test_quote_gradient_is_nonzero_at_node_quote_indices_only(self):
        problem = _problem(SETTINGS_SUFFIXES[0])

        def signed_quote_sum(q0):
            return sum(
                jnp.sum(block.sign * q0[block.quote_idx])
                for block in problem.blocks
            )

        gradient = np.asarray(jax.grad(signed_quote_sum)(problem.q0))
        expected = np.zeros(len(problem.quote_labels), dtype=np.float64)
        for block in problem.blocks:
            np.add.at(
                expected,
                np.asarray(block.quote_idx, dtype=np.int64),
                np.asarray(block.sign, dtype=np.float64),
            )
        np.testing.assert_array_equal(gradient, expected)
        self.assertTrue(np.all(gradient != 0.0))

    def test_quote_labels_match_golden_mqs_order_for_each_setting(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                golden = GOLDEN_DIR / f"{CONFIG_STR}{suffix}-mqs.csv"
                self.assertTrue(golden.is_file(), f"Missing golden output: {golden}")
                self.assertEqual(
                    problem.quote_labels,
                    tuple(read_mqs(golden).keys()),
                )

    def test_curve_parameter_slices_tile_quote_labels(self):
        problem = _problem(SETTINGS_SUFFIXES[0])
        expected_start = 0
        ordered_labels = []

        for curve in problem.curves:
            start, end = curve.param_slice
            self.assertEqual(start, expected_start)
            self.assertGreaterEqual(end, start)
            ordered_labels.extend(curve.labels)
            expected_start = end

        self.assertEqual(expected_start, len(problem.q0))
        self.assertEqual(tuple(ordered_labels), problem.quote_labels)
        self.assertEqual(len(problem.quote_labels), len(problem.q0))


if __name__ == "__main__":
    unittest.main()
