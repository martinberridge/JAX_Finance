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


INTERPOLATORS: dict[str, Callable] = {
    "Linear": linear,
}


__all__ = ["_interval_index", "linear", "INTERPOLATORS"]
