from __future__ import annotations

from collections.abc import Callable

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
    """Return swap PV sensitivities to market quotes in GBP per bp.

    One reverse-mode pass replaces the three hand-coded steps of the reference
    pipeline (``pv01_examples/multicurve2/calibration_interpolation_example.py``):

    1. ``DiscountingSwapTradePricer.present_value_sensitivity`` builds point
       sensitivities dPV/dz(t) per cashflow time, using
       ``ImmutableRatesProvider.zero_rate_point_sensitivity`` (``-df * t``) and
       ``ibor_point_sensitivities`` (dF/dz_L at period start and end). Here these
       are the derivatives of ``pv_swap`` through ``Curve.df`` and
       ``Curve.forward``; the ``EFFECTIVE_ZERO`` branches correspond to the
       ``jnp.where`` guards, whose constant branches have zero gradient.
    2. ``ImmutableRatesProvider.parameter_sensitivity`` maps points to curve
       nodes with ``y_value_parameter_sensitivity``. Here that is the derivative
       through ``INTERPOLATORS`` in ``Curve.zero``; the slice in ``curves_from``
       routes each curve's gradient into its section of the parameter vector.
    3. ``MarketQuoteSensitivityCalculator.sensitivity`` maps node sensitivities
       to quotes through the calibration Jacobian. Here ``calibrate`` uses
       ``custom_root``, whose implicit-function rule supplies dz/dq without
       differentiating through the Newton iterations.

    The per-cashflow point sensitivities of step 1 are never materialised;
    reverse mode goes straight from PV to curve nodes. For the step-2 result,
    call ``zero_rate_pv01`` with the calibrated parameters.

    ``1e-4`` converts per unit of quote (decimal rate) to per basis point.
    """
    return jax.grad(pv_from_quotes, argnums=2)(problem, swap, q) * 1e-4


def parallel_pv01(
    problem: CalibrationProblem,
    swap: SwapArrays,
    q: jax.Array,
) -> jax.Array:
    """
    Return the parallel market-quote PV01 in GBP per bp.
    autodiff pv_from_quotes  forward_mode, returning the tangent [1]     
    """
    return jax.jvp(
        lambda quotes: pv_from_quotes(problem, swap, quotes), # f: quotes → PV (scalar) - use lambda to make f dependent on quotes only, keeeping swap and quotes fixed
        (q,), # primals are quotes
        (jnp.ones_like(q),), # tangent is array of 1's same size & shape as q 
    )[1] * 1e-4


def cross_gamma(
    problem: CalibrationProblem,
    swap: SwapArrays,
    q: jax.Array,
) -> jax.Array:
    """Return market-quote cross-gamma in GBP per bp squared."""
    return jax.hessian(pv_from_quotes, argnums=2)(problem, swap, q) * 1e-8


def _bump_market_quote_pv01(
    problem: CalibrationProblem,
    swap: SwapArrays,
    q: jax.Array,
    h: float,
    map_over: Callable,
) -> jax.Array:
    q = jnp.asarray(q)
    bumps = h * jnp.eye(q.shape[0], dtype=q.dtype)
    shifts = jnp.concatenate([bumps, -bumps])
    pvs = map_over(lambda shift: pv_from_quotes(problem, swap, q + shift), shifts)
    n = q.shape[0]
    return (pvs[:n] - pvs[n:]) / (2.0 * h) * 1e-4


def bump_market_quote_pv01(
    problem: CalibrationProblem,
    swap: SwapArrays,
    q: jax.Array,
    h: float = 1e-4,
) -> jax.Array:
    """Return central bump-and-recalibrate PV01 per market quote in GBP per bp.

    Each quote is shifted by +h and -h, the curves are recalibrated and the swap
    repriced: the same compiled pricer and Newton calibrator as
    ``market_quote_pv01``, so the two differ only in how the derivative is
    taken. All 2N recalibrations run as one ``vmap`` batch. With the default
    h = 1bp the result is ``(PV_up - PV_down) / 2``, as in the legacy
    ``pv01_pricer.bump.bump_pv01``.
    """
    return _bump_market_quote_pv01(problem, swap, q, h, lambda f, xs: jax.vmap(f)(xs))


def bump_market_quote_pv01_sequential(
    problem: CalibrationProblem,
    swap: SwapArrays,
    q: jax.Array,
    h: float = 1e-4,
) -> jax.Array:
    """``bump_market_quote_pv01`` with the 2N recalibrations run one at a time (``lax.map``).

    This is how a conventional risk system bumps: one scenario after another.
    Comparing it with the ``vmap`` version separates the cost of the 2N
    recalibrations from the benefit of batching them.
    """
    return _bump_market_quote_pv01(problem, swap, q, h, jax.lax.map)


def stack_trades(swaps: list[SwapArrays]) -> SwapArrays:
    """Stack equal-shaped swaps into a leading trade dimension."""
    if not swaps:
        raise ValueError("at least one swap is required")

    reference_leaves, reference_structure = jax.tree_util.tree_flatten(swaps[0])
    for swap in swaps[1:]:
        leaves, structure = jax.tree_util.tree_flatten(swap)
        if structure != reference_structure or tuple(
            leaf.shape for leaf in leaves
        ) != tuple(leaf.shape for leaf in reference_leaves):
            raise ValueError(
                "all swaps must have identical pytree structures and leaf shapes"
            )

    return jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves),
        *swaps,
    )


def batch_market_quote_pv01(
    problem: CalibrationProblem,
    swaps_stacked: SwapArrays,
    q: jax.Array,
) -> jax.Array:
    """Return market-quote PV01 for each trade in a stacked swap batch."""
    return jax.vmap(market_quote_pv01, in_axes=(None, 0, None))(
        problem,
        swaps_stacked,
        q,
    )


def portfolio_market_quote_pv01(
    problem: CalibrationProblem,
    swaps: list[SwapArrays],
    q: jax.Array,
) -> jax.Array:
    """Return aggregate market-quote PV01 from a gradient of total portfolio PV."""
    if not swaps:
        raise ValueError("at least one swap is required")

    def portfolio_present_value(quotes: jax.Array) -> jax.Array:
        values = jnp.stack(
            [pv_from_quotes(problem, swap, quotes) for swap in swaps]
        )
        return jnp.sum(values)

    return jax.grad(portfolio_present_value)(q) * 1e-4


__all__ = [
    "zero_rate_pv01",
    "as_labelled",
    "pv_from_quotes",
    "market_quote_pv01",
    "parallel_pv01",
    "cross_gamma",
    "bump_market_quote_pv01",
    "bump_market_quote_pv01_sequential",
    "stack_trades",
    "batch_market_quote_pv01",
    "portfolio_market_quote_pv01",
]
