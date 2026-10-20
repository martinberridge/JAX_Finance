from __future__ import annotations

import unittest
from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pv01_examples.paths import SETTINGS_SUFFIXES
from pv01_jax.calibrate import calibrate_checked, calibrate_with_info
from pv01_jax.risk import market_quote_pv01, pv_from_quotes
from tests.jax_port.test_phase5_risk import _problem, _swap_arrays

STRESS_BUMP = 1e-4


@lru_cache(maxsize=None)
def _scenarios(settings_suffix: str):
    problem = _problem(settings_suffix)
    q0 = problem.q0
    scenarios = {
        "parallel up": q0 + 0.02,
        "parallel down": q0 - 0.005,
    }

    steepener = q0
    for curve in problem.curves:
        start, end = curve.param_slice
        tenors = curve.x_nodes
        tenor_range = tenors[-1] - tenors[0]
        weights = (tenors - tenors[0]) / tenor_range
        curve_bumps = -0.005 + 0.01 * weights
        steepener = steepener.at[start:end].add(curve_bumps)
    scenarios["steepener"] = steepener

    bump_index = problem.quote_labels.index("GBP-IRS6M-5Y")
    scenarios["single bump"] = q0.at[bump_index].add(0.0025)
    return tuple(scenarios.items())


class TestPhase6Stress(unittest.TestCase):
    def test_stress_scenarios_converge_and_pv01_matches_central_bumps(self):
        for suffix in SETTINGS_SUFFIXES:
            problem = _problem(suffix)
            swap_arrays = _swap_arrays()
            bumps = jnp.eye(problem.q0.shape[0], dtype=problem.q0.dtype) * STRESS_BUMP

            for scenario_name, quotes in _scenarios(suffix):
                with self.subTest(
                    settings_suffix=suffix,
                    scenario=scenario_name,
                    check="convergence",
                ):
                    _, iterations, max_residual = calibrate_with_info(problem, quotes)
                    self.assertLess(float(max_residual), 1e-14)
                    self.assertLessEqual(int(iterations), 10)

                with self.subTest(
                    settings_suffix=suffix,
                    scenario=scenario_name,
                    check="PV01 finite difference",
                ):
                    actual = market_quote_pv01(problem, swap_arrays, quotes)
                    up = jax.vmap(
                        lambda bumped_q: pv_from_quotes(
                            problem,
                            swap_arrays,
                            bumped_q,
                        )
                    )(quotes + bumps)
                    down = jax.vmap(
                        lambda bumped_q: pv_from_quotes(
                            problem,
                            swap_arrays,
                            bumped_q,
                        )
                    )(quotes - bumps)
                    finite_difference = (up - down) / 2.0
                    np.testing.assert_allclose(
                        np.asarray(actual),
                        np.asarray(finite_difference),
                        atol=0.01,
                        rtol=0.0,
                    )

    def test_calibrate_checked_rejects_nan_quote_for_each_setting(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                quotes = problem.q0.at[0].set(jnp.nan)
                with self.assertRaisesRegex(
                    eqx.EquinoxRuntimeError,
                    "calibration did not converge",
                ):
                    eqx.filter_jit(calibrate_checked)(problem, quotes)

    def test_calibrate_checked_rejects_one_iteration_for_each_setting(self):
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem = _problem(suffix)
                quotes = problem.q0 + 0.02
                with self.assertRaisesRegex(
                    eqx.EquinoxRuntimeError,
                    "calibration did not converge",
                ):
                    eqx.filter_jit(calibrate_checked)(
                        problem,
                        quotes,
                        max_iter=1,
                    )


if __name__ == "__main__":
    unittest.main()
