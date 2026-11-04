from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from pv01_jax.calibrate import newton_solve, newton_solve_with_info


class TestPhase4NewtonSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.a = jnp.asarray([0.5, 2.0, 9.0], dtype=jnp.float64)

    def test_newton_solve_finds_vector_square_roots(self):
        a = self.a
        solution = newton_solve(lambda z: z**2 - a, a)
        np.testing.assert_allclose(
            np.asarray(solution),
            np.asarray(jnp.sqrt(a)),
            atol=1e-14,
            rtol=0.0,
        )

    def test_custom_root_uses_implicit_derivative(self):
        def solve_for_a(a):
            return jax.lax.custom_root(
                lambda z: z**2 - a,
                jax.lax.stop_gradient(a),
                newton_solve,
                lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(y), y),
            )

        derivative = jax.jacfwd(solve_for_a)(self.a)
        expected = jnp.diag(1.0 / (2.0 * jnp.sqrt(self.a)))
        np.testing.assert_allclose(
            np.asarray(derivative),
            np.asarray(expected),
            atol=1e-12,
            rtol=0.0,
        )

    def test_newton_solve_with_info_reports_convergence(self):
        a = self.a
        solution, iterations, max_residual = newton_solve_with_info(
            lambda z: z**2 - a,
            a,
        )
        np.testing.assert_allclose(
            np.asarray(solution),
            np.asarray(jnp.sqrt(a)),
            atol=1e-14,
            rtol=0.0,
        )
        self.assertLessEqual(int(iterations), 10)
        self.assertLess(float(max_residual), 1e-14)


if __name__ == "__main__":
    unittest.main()
