from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp


def _interval_index(x, t):
    """Return the clipped lower interval index using right-sided node search."""
    return jnp.clip(jnp.searchsorted(x, t, side="right") - 1, 0, x.shape[0] - 2)


def linear(x, z, t):
    """Linearly interpolate node values with flat extrapolation."""
    t = jnp.clip(t, x[0], x[-1])
    index = _interval_index(x, t)
    x_left = x[index]
    x_right = x[index + 1]
    z_left = z[index]
    z_right = z[index + 1]
    weight = (t - x_left) / (x_right - x_left)
    return z_left + weight * (z_right - z_left)


def _quadratic(x, z, t, k):
    """Evaluate the quadratic through nodes k-1, k, and k+1."""
    dx1 = x[k] - x[k - 1]
    dx2 = x[k + 1] - x[k]
    dy1 = z[k] - z[k - 1]
    dy2 = z[k + 1] - z[k]
    b = (dx1 * dy2 / dx2 + dx2 * dy1 / dx1) / (dx1 + dx2)
    c = (dy2 / dx2 - dy1 / dx1) / (dx1 + dx2)
    delta = t - x[k]
    return z[k] + b * delta + c * delta * delta


def double_quadratic(x, z, t):
    """Double-quadratic interpolation matching the production curve interpolator."""
    t = jnp.clip(t, x[0], x[-1])
    interval = _interval_index(x, t)
    last_node_index = x.shape[0] - 1

    left_k = jnp.clip(interval, 1, last_node_index - 1)
    right_k = jnp.clip(interval + 1, 1, last_node_index - 1)
    left_quadratic = _quadratic(x, z, t, left_k)
    right_quadratic = _quadratic(x, z, t, right_k)

    left_boundary_value = _quadratic(x, z, t, jnp.asarray(1))
    right_boundary_value = _quadratic(x, z, t, jnp.asarray(last_node_index - 1))

    x_left = x[interval]
    x_right = x[interval + 1]
    weight = (x_right - t) / (x_right - x_left)
    blended_value = weight * left_quadratic + (1.0 - weight) * right_quadratic

    return jnp.where(
        interval == 0,
        left_boundary_value,
        jnp.where(interval == last_node_index - 1, right_boundary_value, blended_value),
    )


def natural_cubic_spline(x, z, t):
    """Natural cubic spline interpolation with flat extrapolation."""
    h = jnp.diff(x)
    slopes = jnp.diff(z) / h
    node_count = x.shape[0]

    diagonal = jnp.ones(node_count, dtype=x.dtype)
    diagonal = diagonal.at[1:-1].set((h[:-1] + h[1:]) / 3.0)
    lower_diagonal = jnp.concatenate((h[:-1] / 6.0, jnp.zeros((1,), dtype=x.dtype)))
    upper_diagonal = jnp.concatenate((jnp.zeros((1,), dtype=x.dtype), h[1:] / 6.0))
    system = (
        jnp.diag(diagonal)
        + jnp.diag(lower_diagonal, k=-1)
        + jnp.diag(upper_diagonal, k=1)
    )
    rhs = jnp.zeros(node_count, dtype=z.dtype)
    rhs = rhs.at[1:-1].set(slopes[1:] - slopes[:-1])
    second_derivatives = jnp.linalg.solve(system, rhs)

    t = jnp.clip(t, x[0], x[-1])
    interval = _interval_index(x, t)
    interval_width = h[interval]
    a = (x[interval + 1] - t) / interval_width
    b = 1.0 - a
    return (
        a * z[interval]
        + b * z[interval + 1]
        + (
            a * (a * a - 1.0) * second_derivatives[interval]
            + b * (b * b - 1.0) * second_derivatives[interval + 1]
        )
        * interval_width**2
        / 6.0
    )


INTERPOLATORS: dict[str, Callable] = {
    "Linear": linear,
    "DoubleQuadratic": double_quadratic,
    "NaturalCubicSpline": natural_cubic_spline,
}


__all__ = [
    "_interval_index",
    "linear",
    "double_quadratic",
    "natural_cubic_spline",
    "INTERPOLATORS",
]
