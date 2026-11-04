"""Array containers for trades and calibration problems.

Each class subclasses ``eqx.Module``, which makes it a frozen dataclass and
registers it as a JAX pytree, so instances can be passed through ``jit``,
``grad``, ``jvp``, ``vmap``, ``custom_root`` and ``tree_map``. Modules are
immutable; use ``eqx.tree_at`` to build a modified copy.

Field options:

* ``converter=jnp.asarray`` converts lists, floats and NumPy arrays to JAX
  arrays at construction. It runs only in ``__init__``, not when JAX rebuilds
  a module with tracers.
* ``default_factory`` supplies empty-array defaults; dataclasses do not allow
  mutable default values.
* ``static=True`` stores the value in the pytree structure rather than its
  leaves, so it stays a plain Python value under ``jit``. Use it for anything
  read in Python at trace time: dict keys, slice bounds, dispatch strings and
  labels. Static values must be hashable (hence tuples, not lists), form part
  of the ``jit`` cache key (changing one recompiles) and must match across
  trades stacked by ``stack_trades``.
"""

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
    """Arrays describing one resolved swap for pricing.

    Optional schedule fields default to empty arrays. ``fixed_leg=None`` changes
    the pytree structure, so swaps with and without a fixed leg compile
    separately and cannot be stacked into one ``vmap`` batch.
    """

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
    """Calibration instruments of one trade type, priced together.

    ``trade_type`` is static because the pricers dispatch on it with Python
    ``if`` statements, which cannot branch on a traced value.
    """

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
    """Static description of one curve within the global parameter vector.

    ``param_slice`` is static because slice bounds fix output shapes, which must
    be known at compile time. ``interpolator`` is static because it is used as a
    dict key into ``INTERPOLATORS`` while tracing.
    """

    name: str = eqx.field(static=True)
    param_slice: tuple[int, int] = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    interpolator: str = eqx.field(static=True)
    x_nodes: jnp.ndarray = eqx.field(converter=jnp.asarray)


class CalibrationProblem(eqx.Module):
    """Curves, calibration blocks and initial quotes for one calibration group.

    Names and labels are static; ``curves`` and ``blocks`` are child pytrees,
    so the whole problem flattens into a single tree of arrays.
    """

    name: str = eqx.field(static=True)
    quote_labels: tuple[str, ...] = eqx.field(static=True)
    discount_curve: str = eqx.field(static=True)
    forward_curve: str = eqx.field(static=True)
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
