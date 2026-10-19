from __future__ import annotations

import jax
import jax.numpy as jnp

from pv01_jax.arrays import CalibrationProblem, SwapArrays
from pv01_jax.curves import Curve, curves_from


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


def residuals(
    problem: CalibrationProblem,
    z: jax.Array,
    q: jax.Array,
) -> jax.Array:
    """Return par-minus-quote calibration residuals in quote-label order."""
    curves = curves_from(problem, z)
    discount_curve = curves[problem.discount_curve]
    forward_curve = curves[problem.forward_curve]
    result = jnp.zeros_like(q)

    for block in problem.blocks:
        num_trades = block.quote_idx.shape[0]
        discount_factors = discount_curve.df(block.t_pay)
        if block.trade_type == "FIX":
            forwards = forward_curve.forward(
                block.float_leg.start,
                block.float_leg.end,
                block.tau,
            )
            block_residuals = (
                jax.ops.segment_sum(
                    forwards,
                    block.trade_id,
                    num_segments=num_trades,
                )
                - q[block.quote_idx]
            )
        elif block.trade_type in ("OIS", "IRS"):
            if block.trade_type == "OIS":
                float_values = (
                    block.float_leg.notional
                    * discount_curve.compounded(
                        block.t_fix_start,
                        block.t_fix_end,
                    )
                    * discount_factors
                )
            else:
                forwards = forward_curve.forward(
                    block.t_fix_start,
                    block.t_fix_end,
                    block.tau_fix,
                )
                float_values = (
                    block.float_leg.notional
                    * forwards
                    * block.tau
                    * discount_factors
                )

            float_pv = jax.ops.segment_sum(
                float_values,
                block.trade_id,
                num_segments=num_trades,
            )
            annuity_values = (
                block.fixed_leg.notional
                * block.tau
                * discount_factors
            )
            annuity = jax.ops.segment_sum(
                annuity_values,
                block.trade_id,
                num_segments=num_trades,
            )
            block_residuals = float_pv / annuity - q[block.quote_idx]
        else:
            raise ValueError(f"Unsupported calibration trade type: {block.trade_type}")

        result = result.at[block.quote_idx].set(block_residuals)

    return result


__all__ = ["pv_swap", "residuals"]
