import unittest

import jax
import jax.numpy as jnp
import numpy as np

from pv01_jax.calibrate import calibrate, calibrate_with_info
from pv01_jax.curves import curves_from
from pv01_jax.pricers import pv_swap
from pv01_jax.risk import bump_market_quote_pv01, market_quote_pv01, pv_from_quotes, zero_rate_pv01
from pv01_jax.synthetic import synthetic_problem

INTERPOLATORS = ("Linear", "DoubleQuadratic", "NaturalCubicSpline")


def _calibrated(n_quotes: int, interpolator: str):
    problem, swap = synthetic_problem(n_quotes, interpolator)
    q = jnp.asarray(problem.q0)
    return problem, swap, q, calibrate(problem, q)


class TestSynthetic(unittest.TestCase):
    """Structural checks on the synthetic problem; none need a reference implementation."""

    def test_calibrates_for_all_sizes_and_interpolators(self):
        for n_quotes in (10, 40, 160):
            for interpolator in INTERPOLATORS:
                with self.subTest(n_quotes=n_quotes, interpolator=interpolator):
                    problem, _ = synthetic_problem(n_quotes, interpolator)
                    _, _, max_residual = jax.jit(
                        lambda q: calibrate_with_info(problem, q)
                    )(jnp.asarray(problem.q0))
                    self.assertLess(float(max_residual), 1e-12)

    def test_zero_rate_buckets_sum_to_parallel_zero_rate_pv01(self):
        """The buckets are a gradient in z, so their sum is the jvp along a ones tangent.

        This is the parallel *zero-rate* shift. ``parallel_pv01`` shifts the
        quotes instead and differs (14,553 vs 14,691 GBP/bp at N = 20).
        """
        for interpolator in INTERPOLATORS:
            with self.subTest(interpolator=interpolator):
                problem, swap, _, z = _calibrated(20, interpolator)

                def pv(z):
                    return pv_swap(swap, curves_from(problem, z), problem.discount_curve, problem.forward_curve)

                parallel = jax.jvp(pv, (z,), (jnp.ones_like(z),))[1] * 1e-4
                buckets = zero_rate_pv01(problem, swap, z)
                np.testing.assert_allclose(float(jnp.sum(buckets)), float(parallel), rtol=1e-12)

    def test_linear_buckets_beyond_last_payment_are_zero(self):
        """With linear interpolation, a rate at time t depends only on the two nodes bracketing t.

        The swap's last payment is at 20y, so on each curve every node after
        the first one at or beyond 20y has exactly zero zero-rate PV01, and
        the nodes up to that one don't.
        """
        for n_quotes in (10, 40, 160):
            with self.subTest(n_quotes=n_quotes):
                problem, swap, _, z = _calibrated(n_quotes, "Linear")
                buckets = np.asarray(zero_rate_pv01(problem, swap, z))
                last_payment = float(jnp.max(swap.t_pay))
                for curve in problem.curves:
                    start, end = curve.param_slice
                    x = np.asarray(curve.x_nodes)
                    bracket = int(np.argmax(x >= last_payment))
                    self.assertTrue(np.all(buckets[start + bracket + 1:end] == 0.0), curve.name)
                    self.assertNotEqual(buckets[start + bracket], 0.0, curve.name)

    def test_grad_forward_mode_and_bump_agree(self):
        """Market-quote PV01 three ways: reverse mode, forward mode over the identity, and bumping.

        The two AD modes compute the same derivative and agree to rounding.
        The 1bp central bump differs by its truncation error, which scales
        with the bucket: up to 1.03e-3 GBP/bp on a 12,500 GBP/bp bucket
        (8e-8 relative) with natural cubic splines. Halving the step cuts
        that error by 4, as an h^2 truncation error should, which shows the
        bump and AD differ only by truncation.
        """
        for interpolator in INTERPOLATORS:
            with self.subTest(interpolator=interpolator):
                problem, swap, q, _ = _calibrated(20, interpolator)
                reverse = np.asarray(market_quote_pv01(problem, swap, q))
                forward = jax.vmap(
                    lambda e: jax.jvp(lambda q: pv_from_quotes(problem, swap, q), (q,), (e,))[1]
                )(jnp.eye(q.shape[0])) * 1e-4
                np.testing.assert_allclose(np.asarray(forward), reverse, rtol=1e-10, atol=1e-9)

                errors = []
                for h in (1e-4, 5e-5):
                    bumped = np.asarray(jax.jit(lambda q: bump_market_quote_pv01(problem, swap, q, h=h))(q))
                    np.testing.assert_allclose(bumped, reverse, rtol=1e-6, atol=1e-3)
                    errors.append(np.max(np.abs(bumped - reverse)))
                self.assertAlmostEqual(errors[0] / errors[1], 4.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
