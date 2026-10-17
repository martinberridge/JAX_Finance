from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class LegArrays(eqx.Module):
    """Arrays used to represent one side of a swap or fixed coupon schedule."""

    start: jnp.ndarray = eqx.field(converter=jnp.asarray)
    end: jnp.ndarray = eqx.field(converter=jnp.asarray)
    rate: jnp.ndarray = eqx.field(converter=jnp.asarray)
    notional: jnp.ndarray = eqx.field(converter=jnp.asarray)


class FloatLegArrays(LegArrays):
    """Floating-leg arrays with the same pytree structure as a generic leg."""


class SwapArrays(eqx.Module):
    float_leg: FloatLegArrays
    spread: jnp.ndarray = eqx.field(converter=jnp.asarray)
    fixed_leg: LegArrays | None = None
    t_pay: jnp.ndarray = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.asarray([], dtype=jnp.float64),
    )
    trade_id: jnp.ndarray = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.asarray([], dtype=jnp.int64),
    )
    sign: jnp.ndarray = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.asarray([1.0], dtype=jnp.float64),
    )
    fixed_rate: jnp.ndarray = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.asarray(0.0, dtype=jnp.float64),
    )
    tau: jnp.ndarray = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.asarray([], dtype=jnp.float64),
    )
    t_fix_start: jnp.ndarray = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.asarray([], dtype=jnp.float64),
    )
    t_fix_end: jnp.ndarray = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.asarray([], dtype=jnp.float64),
    )
    tau_fix: jnp.ndarray = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.asarray([], dtype=jnp.float64),
    )


class CalibBlock(eqx.Module):
    trade_type: str = eqx.field(static=True)
    quote_idx: jnp.ndarray = eqx.field(converter=jnp.asarray)
    trade_id: jnp.ndarray = eqx.field(converter=jnp.asarray)
    float_leg: FloatLegArrays
    t_pay: jnp.ndarray = eqx.field(converter=jnp.asarray)
    tau: jnp.ndarray = eqx.field(converter=jnp.asarray)
    t_fix_start: jnp.ndarray = eqx.field(converter=jnp.asarray)
    t_fix_end: jnp.ndarray = eqx.field(converter=jnp.asarray)
    tau_fix: jnp.ndarray = eqx.field(converter=jnp.asarray)
    sign: jnp.ndarray = eqx.field(converter=jnp.asarray)
    fixed_leg: LegArrays | None = None


class CurveSpec(eqx.Module):
    name: str = eqx.field(static=True)
    param_slice: tuple[int, int] = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    interpolator: str = eqx.field(static=True)
    x_nodes: jnp.ndarray = eqx.field(converter=jnp.asarray)


class CalibrationProblem(eqx.Module):
    name: str = eqx.field(static=True)
    quote_labels: tuple[str, ...] = eqx.field(static=True)
    curves: tuple[CurveSpec, ...]
    blocks: tuple[CalibBlock, ...]
    q0: jnp.ndarray = eqx.field(converter=jnp.asarray)


__all__ = [
    "LegArrays",
    "FloatLegArrays",
    "SwapArrays",
    "CalibBlock",
    "CurveSpec",
    "CalibrationProblem",
]
