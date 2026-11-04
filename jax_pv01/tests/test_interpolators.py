"""Interpolator tests against Strata test vectors."""

import unittest

import numpy as np

from pv01_market.interpolators import (
    DoubleQuadraticInterpolator,
    LinearInterpolator,
    NaturalCubicSplineInterpolator,
)

TOL = 1e-12

X_DATA = np.array([0.0, 0.4, 1.0, 1.8, 2.8, 5.0])
Y_DATA = np.array([3.0, 4.0, 3.1, 2.0, 7.0, 2.0])
X_TEST = np.array([0.2, 1.1, 2.3])
Y_TEST_LINEAR = np.array([3.5, 3.1 - (1.1 / 8), 4.5])


class TestInterpolators(unittest.TestCase):
    def test_linear_interpolation_nodes(self):
        interp = LinearInterpolator(X_DATA, Y_DATA)
        for i, x in enumerate(X_DATA):
            self.assertLess(abs(interp.interpolate(x) - Y_DATA[i]), TOL)

    def test_linear_interpolation_between(self):
        interp = LinearInterpolator(X_DATA, Y_DATA)
        for x, expected in zip(X_TEST, Y_TEST_LINEAR):
            self.assertLess(abs(interp.interpolate(x) - expected), TOL)

    def test_linear_flat_extrapolation(self):
        interp = LinearInterpolator(X_DATA, Y_DATA)
        self.assertLess(abs(interp.interpolate(-1.0) - Y_DATA[0]), TOL)
        self.assertLess(abs(interp.interpolate(10.0) - Y_DATA[-1]), TOL)

    def test_double_quadratic_at_nodes(self):
        interp = DoubleQuadraticInterpolator(X_DATA, Y_DATA)
        for i, x in enumerate(X_DATA):
            self.assertLess(abs(interp.interpolate(x) - Y_DATA[i]), 1e-10)

    def test_natural_cubic_at_nodes(self):
        interp = NaturalCubicSplineInterpolator(X_DATA, Y_DATA)
        for i, x in enumerate(X_DATA):
            self.assertLess(abs(interp.interpolate(x) - Y_DATA[i]), 1e-10)

    def test_linear_parameter_sensitivity_endpoints(self):
        interp = LinearInterpolator(X_DATA, Y_DATA)
        sens_start = interp.parameter_sensitivity(0.0)
        self.assertLess(abs(sens_start[0] - 1.0), TOL)
        self.assertLess(abs(sens_start[1]), TOL)
        sens_end = interp.parameter_sensitivity(5.0)
        self.assertLess(abs(sens_end[-1] - 1.0), TOL)
