from __future__ import annotations

import jax
import jax.numpy as jnp

from pv01_jax.arrays import CalibrationProblem, SwapArrays
from pv01_jax.curves import curves_from
from pv01_jax.pricers import pv_swap


def zero_rate_pv01(
    problem: CalibrationProblem,
    swap: SwapArrays,
    z: jax.Array,
) -> jax.Array:
    """Return swap PV sensitivities to zero rates in GBP per bp per curve node."""

    def present_value(parameters: jax.Array) -> jax.Array:
        curves = curves_from(problem, parameters)
        return pv_swap(
            swap,
            curves,
            problem.discount_curve,
            problem.forward_curve,
        )

    return jax.grad(present_value)(z) * 1e-4


def as_labelled(
    problem: CalibrationProblem,
    vector: jax.Array,
) -> dict[str, float]:
    """Map a curve-node vector to values keyed by quote label."""
    return {
        label: float(value)
        for label, value in zip(problem.quote_labels, jnp.asarray(vector))
    }


__all__ = ["zero_rate_pv01", "as_labelled"]
