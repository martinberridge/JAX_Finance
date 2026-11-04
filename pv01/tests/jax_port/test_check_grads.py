from __future__ import annotations

import unittest

from jax.test_util import check_grads 
# see comments in phase 6a prompts
from pv01_examples.paths import SETTINGS_SUFFIXES
from pv01_jax.calibrate import calibrate
from pv01_jax.curves import curves_from
from pv01_jax.interp import INTERPOLATORS
from pv01_jax.pricers import pv_swap, residuals
from pv01_jax.risk import pv_from_quotes
from tests.util.example_inputs import example_problem as _problem, example_swap_arrays as _swap_arrays


class TestGradientChecks(unittest.TestCase):
    def _check_function(self, name, function, argument):
        try:
            check_grads(
                function,
                (argument,),
                order=2,
                modes=["fwd", "rev"],
                atol=1e-4,
                rtol=1e-4,
            )
        except Exception as error:
            self.fail(
                f"Gradient check failed for {name}, modes=fwd/rev, order=2: "
                f"{type(error).__name__}: {error}"
            )

    def test_second_order_gradient_checks_for_each_setting(self):
        for suffix in SETTINGS_SUFFIXES:
            problem = _problem(suffix)
            swap_arrays = _swap_arrays()
            q0 = problem.q0
            z0 = calibrate(problem, q0)

            interpolation_curve = problem.curves[0]
            x = interpolation_curve.x_nodes
            z_curve = z0[
                interpolation_curve.param_slice[0] : interpolation_curve.param_slice[1]
            ]
            t = (x[:-1] + x[1:]) / 2.0

            functions = [
                (
                    f"interpolator {name}",
                    lambda z, interp=interpolator: interp(x, z, t),
                    z_curve,
                )
                for name, interpolator in INTERPOLATORS.items()
            ]
            functions.extend(
                [
                    (
                        "pv_swap",
                        lambda z: pv_swap(
                            swap_arrays,
                            curves_from(problem, z),
                            problem.discount_curve,
                            problem.forward_curve,
                        ),
                        z0,
                    ),
                    (
                        "residuals with respect to parameters",
                        lambda z: residuals(problem, z, q0),
                        z0,
                    ),
                    (
                        "residuals with respect to quotes",
                        lambda q: residuals(problem, z0, q),
                        q0,
                    ),
                    (
                        "calibrate",
                        lambda q: calibrate(problem, q),
                        q0,
                    ),
                    (
                        "pv_from_quotes",
                        lambda q: pv_from_quotes(problem, swap_arrays, q),
                        q0,
                    ),
                ]
            )

            for name, function, argument in functions:
                with self.subTest(settings_suffix=suffix, function=name):
                    self._check_function(name, function, argument)


if __name__ == "__main__":
    unittest.main()
