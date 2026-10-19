from __future__ import annotations

import jax
import jax.numpy as jnp

from pv01_jax.arrays import CalibrationProblem, SwapArrays
from pv01_jax.calibrate import calibrate
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


def pv_from_quotes(
    problem: CalibrationProblem,
    swap: SwapArrays,
    q: jax.Array,
) -> jax.Array:
    """Calibrate from quotes and return the resulting swap present value."""
    curves = curves_from(problem, calibrate(problem, q))
    return pv_swap(
        swap,
        curves,
        problem.discount_curve,
        problem.forward_curve,
    )


def market_quote_pv01(
    problem: CalibrationProblem,
    swap: SwapArrays,
    q: jax.Array,
) -> jax.Array:
    """Return swap PV sensitivities to market quotes in GBP per bp."""
    return jax.grad(pv_from_quotes, argnums=2)(problem, swap, q) * 1e-4


def parallel_pv01(
    problem: CalibrationProblem,
    swap: SwapArrays,
    q: jax.Array,
) -> jax.Array:
    """Return the parallel market-quote PV01 in GBP per bp."""
    return jax.jvp(
        lambda quotes: pv_from_quotes(problem, swap, quotes),
        (q,),
        (jnp.ones_like(q),),
    )[1] * 1e-4


__all__ = [
    "zero_rate_pv01",
    "as_labelled",
    "pv_from_quotes",
    "market_quote_pv01",
    "parallel_pv01",
]
