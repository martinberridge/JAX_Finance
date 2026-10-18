from __future__ import annotations

import jax
import jax.numpy as jnp

from pv01_jax.arrays import SwapArrays
from pv01_jax.curves import Curve


def pv_swap(
    swap: SwapArrays,
    curves: dict[str, Curve],
    discount: str,
    forward: str,
) -> jax.Array:
    """Calculate the present value of a resolved fixed-versus-floating swap."""
    discount_curve = curves[discount]
    forward_curve = curves[forward]
    float_leg = swap.float_leg
    fixed_leg = swap.fixed_leg

    discount_factors = discount_curve.df(swap.t_pay)
    forwards = forward_curve.forward(
        swap.t_fix_start,
        swap.t_fix_end,
        swap.tau_fix,
    )
    pv_float = jnp.sum(
        float_leg.notional * forwards * swap.tau * discount_factors
    )
    pv_fixed = jnp.sum(
        fixed_leg.notional
        * swap.fixed_rate
        * swap.tau_fix
        * discount_factors
    )
    return swap.sign[0] * (pv_float - pv_fixed)


__all__ = ["pv_swap"]
