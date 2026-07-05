"""Curve interpolators ported from Strata."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


def lower_bound_index(x_value: float, x_values: np.ndarray) -> int:
    """Find index i such that x_values[i] <= x_value < x_values[i+1]."""
    n = len(x_values)
    if x_value <= x_values[0]:
        return 0
    if x_value >= x_values[n - 1]:
        return n - 2 if n > 1 else 0
    return int(np.searchsorted(x_values, x_value, side="right") - 1)


class BoundInterpolator(ABC):
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray):
        self.x_values = np.asarray(x_values, dtype=float)
        self.y_values = np.asarray(y_values, dtype=float)
        if len(self.x_values) != len(self.y_values):
            raise ValueError("x and y must have same length")
        if len(self.x_values) < 2:
            raise ValueError("At least two nodes required")

    def interpolate(self, x: float) -> float:
        if x < self.x_values[0]:
            return self.y_values[0]  # Flat extrapolation
        if x > self.x_values[-1]:
            return self.y_values[-1]
        if x == self.x_values[-1]:
            return self.y_values[-1]
        return self._interpolate(x)

    def first_derivative(self, x: float) -> float:
        if x < self.x_values[0]:
            return 0.0
        if x > self.x_values[-1]:
            return 0.0
        return self._first_derivative(x)

    def parameter_sensitivity(self, x: float) -> np.ndarray:
        if x < self.x_values[0]:
            sens = np.zeros(len(self.y_values))
            sens[0] = 1.0
            return sens
        if x > self.x_values[-1]:
            sens = np.zeros(len(self.y_values))
            sens[-1] = 1.0
            return sens
        return self._parameter_sensitivity(x)

    @abstractmethod
    def _interpolate(self, x: float) -> float:
        ...

    @abstractmethod
    def _first_derivative(self, x: float) -> float:
        ...

    @abstractmethod
    def _parameter_sensitivity(self, x: float) -> np.ndarray:
        ...


class LinearInterpolator(BoundInterpolator):
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray):
        super().__init__(x_values, y_values)
        self.interval_count = len(x_values) - 1
        self.gradients = np.zeros(self.interval_count)
        for i in range(self.interval_count):
            dx = x_values[i + 1] - x_values[i]
            self.gradients[i] = (y_values[i + 1] - y_values[i]) / dx

    def _interpolate(self, x: float) -> float:
        li = lower_bound_index(x, self.x_values)
        x1 = self.x_values[li]
        y1 = self.y_values[li]
        return y1 + (x - x1) * self.gradients[li]

    def _first_derivative(self, x: float) -> float:
        li = lower_bound_index(x, self.x_values)
        if li == self.interval_count:
            li -= 1
        return self.gradients[li]

    def _parameter_sensitivity(self, x: float) -> np.ndarray:
        result = np.zeros(len(self.y_values))
        li = lower_bound_index(x, self.x_values)
        if li == self.interval_count:
            result[self.interval_count] = 1.0
        else:
            x1 = self.x_values[li]
            x2 = self.x_values[li + 1]
            dx = x2 - x1
            a = (x2 - x) / dx
            result[li] = a
            result[li + 1] = 1.0 - a
        return result


class DoubleQuadraticInterpolator(BoundInterpolator):
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray):
        super().__init__(x_values, y_values)
        self.interval_count = len(x_values) - 1

    @staticmethod
    def _quadratic_coeffs(x, y, index: int) -> Tuple[float, float, float]:
        a = y[index]
        dx1 = x[index] - x[index - 1]
        dx2 = x[index + 1] - x[index]
        dy1 = y[index] - y[index - 1]
        dy2 = y[index + 1] - y[index]
        b = (dx1 * dy2 / dx2 + dx2 * dy1 / dx1) / (dx1 + dx2)
        c = (dy2 / dx2 - dy1 / dx1) / (dx1 + dx2)
        return a, b, c

    def _eval_quad(self, x_val: float, index: int, offset_x: float) -> float:
        a, b, c = self._quadratic_coeffs(self.x_values, self.y_values, index)
        dx = x_val - offset_x
        return a + b * dx + c * dx * dx

    def _deriv_quad(self, x_val: float, index: int, offset_x: float) -> float:
        a, b, c = self._quadratic_coeffs(self.x_values, self.y_values, index)
        dx = x_val - offset_x
        return b + 2.0 * c * dx

    def _interpolate(self, x: float) -> float:
        li = lower_bound_index(x, self.x_values)
        hi = li + 1
        if li == 0:
            return self._eval_quad(x, 1, self.x_values[1])
        if hi == self.interval_count:
            return self._eval_quad(x, self.interval_count - 1, self.x_values[self.interval_count - 1])
        q1 = self._eval_quad(x, li, self.x_values[li])
        q2 = self._eval_quad(x, hi, self.x_values[hi])
        w = (self.x_values[hi] - x) / (self.x_values[hi] - self.x_values[li])
        return w * q1 + (1.0 - w) * q2

    def _first_derivative(self, x: float) -> float:
        li = lower_bound_index(x, self.x_values)
        hi = li + 1
        if li == 0 or self.interval_count == 1:
            return self._deriv_quad(x, 1, self.x_values[1])
        if hi >= self.interval_count:
            return self._deriv_quad(x, self.interval_count - 1, self.x_values[self.interval_count - 1])
        q1 = self._eval_quad(x, li, self.x_values[li])
        q2 = self._eval_quad(x, hi, self.x_values[hi])
        dq1 = self._deriv_quad(x, li, self.x_values[li])
        dq2 = self._deriv_quad(x, hi, self.x_values[hi])
        w = (self.x_values[hi] - x) / (self.x_values[hi] - self.x_values[li])
        return (w * dq1 + (1.0 - w) * dq2 +
                (q2 - q1) / (self.x_values[hi] - self.x_values[li]))

    @staticmethod
    def _quadratic_sensitivities(x_values, x: float, i: int) -> np.ndarray:
        result = np.zeros(3)
        delta_x = x - x_values[i]
        h1 = x_values[i] - x_values[i - 1]
        h2 = x_values[i + 1] - x_values[i]
        result[0] = delta_x * (delta_x - h2) / h1 / (h1 + h2)
        result[1] = 1.0 + delta_x * (h2 - h1 - delta_x) / h1 / h2
        result[2] = delta_x * (h1 + delta_x) / (h1 + h2) / h2
        return result

    def _parameter_sensitivity(self, x: float) -> np.ndarray:
        n = len(self.x_values)
        result = np.zeros(n)
        li = lower_bound_index(x, self.x_values)
        hi = li + 1
        if li == 0:
            temp = self._quadratic_sensitivities(self.x_values, x, 1)
            result[0:3] = temp
            return result
        if hi == self.interval_count:
            temp = self._quadratic_sensitivities(self.x_values, x, n - 2)
            result[n - 3:n] = temp
            return result
        if li == self.interval_count:
            result[n - 1] = 1.0
            return result
        temp1 = self._quadratic_sensitivities(self.x_values, x, li)
        temp2 = self._quadratic_sensitivities(self.x_values, x, hi)
        w = (self.x_values[hi] - x) / (self.x_values[hi] - self.x_values[li])
        result[li - 1] = w * temp1[0]
        result[li] = w * temp1[1] + (1.0 - w) * temp2[0]
        result[hi] = w * temp1[2] + (1.0 - w) * temp2[1]
        result[hi + 1] = (1.0 - w) * temp2[2]
        return result


class NaturalCubicSplineInterpolator(BoundInterpolator):
    EPS = 1e-12

    @staticmethod
    def _inverse_tridiagonal(delta_x: np.ndarray) -> np.ndarray:
        n = len(delta_x) + 1
        a = np.zeros(n)
        b = np.zeros(n - 1)
        c = np.zeros(n - 1)
        for i in range(1, n - 1):
            a[i] = (delta_x[i - 1] + delta_x[i]) / 3.0
            b[i] = delta_x[i] / 6.0
            c[i - 1] = delta_x[i - 1] / 6.0
        a[0] = 1.0
        b[0] = 0.0
        a[n - 1] = 1.0
        c[n - 2] = 0.0
        # Thomas algorithm for inverse via solving identity columns
        inv = np.zeros((n, n))
        for col in range(n):
            rhs = np.zeros(n)
            rhs[col] = 1.0
            inv[:, col] = _solve_tridiagonal(a, b, c, rhs)
        return inv

    def _second_derivatives(self) -> np.ndarray:
        x = self.x_values
        y = self.y_values
        n = len(x)
        delta_x = x[1:] - x[:-1]
        delta_y_over = (y[1:] - y[:-1]) / delta_x
        inv = self._inverse_tridiagonal(delta_x)
        rhs = np.zeros(n)
        for i in range(1, n - 1):
            rhs[i] = delta_y_over[i] - delta_y_over[i - 1]
        return inv @ rhs

    def _second_deriv_sensitivities(self) -> np.ndarray:
        x = self.x_values
        n = len(x)
        delta_x = x[1:] - x[:-1]
        one_over = 1.0 / delta_x
        inv = self._inverse_tridiagonal(delta_x)
        rhs = np.zeros((n, n))
        for i in range(1, n - 1):
            rhs[i, i - 1] = one_over[i - 1]
            rhs[i, i] = -one_over[i] - one_over[i - 1]
            rhs[i, i + 1] = one_over[i]
        return inv @ rhs

    def _interpolate(self, x: float) -> float:
        x_vals = self.x_values
        y_vals = self.y_values
        low = lower_bound_index(x, x_vals)
        high = low + 1
        n = len(x_vals) - 1
        if low == n:
            return y_vals[n]
        delta = x_vals[high] - x_vals[low]
        if abs(delta) < self.EPS:
            raise ValueError("x data points were not distinct")
        a = (x_vals[high] - x) / delta
        b = (x - x_vals[low]) / delta
        y2 = self._second_derivatives()
        return (a * y_vals[low] + b * y_vals[high] +
                (a * (a * a - 1) * y2[low] + b * (b * b - 1) * y2[high]) * delta * delta / 6.0)

    def _first_derivative(self, x: float) -> float:
        x_vals = self.x_values
        y_vals = self.y_values
        low = lower_bound_index(x, x_vals)
        high = low + 1
        n = len(x_vals) - 1
        if low == n:
            low = n - 1
            high = n
        delta = x_vals[high] - x_vals[low]
        a = (x_vals[high] - x) / delta
        b = (x - x_vals[low]) / delta
        y2 = self._second_derivatives()
        return ((y_vals[high] - y_vals[low]) / delta +
                ((-3.0 * a * a + 1.0) * y2[low] + (3.0 * b * b - 1.0) * y2[high]) * delta / 6.0)

    def _parameter_sensitivity(self, x: float) -> np.ndarray:
        x_vals = self.x_values
        n = len(x_vals)
        result = np.zeros(n)
        low = lower_bound_index(x, x_vals)
        if low == n - 1:
            result[n - 1] = 1.0
            return result
        high = low + 1
        delta = x_vals[high] - x_vals[low]
        a = (x_vals[high] - x) / delta
        b = (x - x_vals[low]) / delta
        c = a * (a * a - 1) * delta * delta / 6.0
        d = b * (b * b - 1) * delta * delta / 6.0
        y2_sens = self._second_deriv_sensitivities()
        for i in range(n):
            result[i] = c * y2_sens[low, i] + d * y2_sens[high, i]
        result[low] += a
        result[high] += b
        return result


def _solve_tridiagonal(a, b, c, d):
    """Solve tridiagonal system."""
    n = len(d)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    cp[0] = b[0] / a[0]
    dp[0] = d[0] / a[0]
    for i in range(1, n - 1):
        denom = a[i] - c[i - 1] * cp[i - 1]
        cp[i] = b[i] / denom
        dp[i] = (d[i] - c[i - 1] * dp[i - 1]) / denom
    dp[n - 1] = (d[n - 1] - c[n - 2] * dp[n - 2]) / (a[n - 1] - c[n - 2] * cp[n - 2])
    x = np.zeros(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def create_interpolator(name: str, x_values: np.ndarray, y_values: np.ndarray) -> BoundInterpolator:
    if name == "Linear":
        return LinearInterpolator(x_values, y_values)
    if name == "DoubleQuadratic":
        return DoubleQuadraticInterpolator(x_values, y_values)
    if name == "NaturalCubicSpline":
        return NaturalCubicSplineInterpolator(x_values, y_values)
    raise ValueError(f"Unknown interpolator: {name}")
