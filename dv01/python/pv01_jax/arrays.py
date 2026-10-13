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


class CalibBlock(eqx.Module):
    trade_type: str = eqx.field(static=True)
    quote_idx: int = eqx.field(static=True)
    float_leg: FloatLegArrays
    quote: jnp.ndarray = eqx.field(converter=jnp.asarray)
    fixed_leg: LegArrays | None = None
    metadata: tuple[str, ...] = eqx.field(static=True, default=("calibration", "block"))


class CurveSpec(eqx.Module):
    name: str = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    nodes: jnp.ndarray = eqx.field(converter=jnp.asarray)
    values: jnp.ndarray = eqx.field(converter=jnp.asarray)


class CalibrationProblem(eqx.Module):
    name: str = eqx.field(static=True)
    curve: CurveSpec = eqx.field()
    blocks: tuple[CalibBlock, ...] = eqx.field(static=True)
    quotes: jnp.ndarray = eqx.field(converter=jnp.asarray)


__all__ = [
    "LegArrays",
    "FloatLegArrays",
    "SwapArrays",
    "CalibBlock",
    "CurveSpec",
    "CalibrationProblem",
]
