"""
Option risk to the FX smile quotes, taken through the SABR calibration.

The price depends on the quotes q = (ATM, RR25, BF25, RR10, BF10) only
through the calibrated parameters theta = (alpha, rho, nu):

    q -> strikes and vols -> calibrate theta -> Hagan vol at K -> Black price

Five quotes and three parameters have no exact fit, so calibration minimises
a loss L(theta, q) and its optimality condition is r(theta, q) = dL/dtheta = 0.
By the implicit function theorem,

    dtheta/dq = -(d2L/dtheta2)^-1 (d2L/dtheta dq),

with a 3x3 Hessian to solve. `lax.custom_root` applies this rule, so nothing
is differentiated through the Adam or Newton steps. `pv01_jax.calibrate`
does the same for curves, where r is the par-rate residual itself.

The rule holds only at the minimum. At an unconverged fit it returns
plausible but wrong numbers, so `calibrate_sabr_from_quotes` returns NaN
parameters there, and prices and risk are NaN with them.

Conventions, as in `jax_finance.fx`: forward deltas, premium unadjusted, a
delta-neutral ATM strike, and butterflies as smile (not market) strangles.
Quotes are decimals here (0.0659), not vol points (6.59).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.flatten_util import ravel_pytree
from jax.scipy.stats import norm

from .black import black_price
from .sabr import (
    calibrate_sabr,
    sabr_implied_vol,
    sabr_loss_for_points,
    sabr_params_from_raw,
    vega_weights,
)

# Order of the quote vector
QUOTE_LABELS = ("atm", "rr25", "bf25", "rr10", "bf10")

# N(d1) at each smile point: ATM, 25d call, 25d put, 10d call, 10d put
_FORWARD_CALL_DELTAS = (0.50, 0.25, 0.75, 0.10, 0.90)


def smile_from_quotes(quotes, F, T):
    """
    Strikes and vols of the five smile points, in the order of
    `_FORWARD_CALL_DELTAS`, from the quotes in the order of `QUOTE_LABELS`.

    The differentiable version of `jax_finance.fx.FXVolSurface.build_from_quotes`.
    """
    atm, rr25, bf25, rr10, bf10 = quotes
    vols = jnp.stack([
        atm,
        atm + bf25 + rr25 / 2.0,
        atm + bf25 - rr25 / 2.0,
        atm + bf10 + rr10 / 2.0,
        atm + bf10 - rr10 / 2.0,
    ])
    d1 = norm.ppf(jnp.array(_FORWARD_CALL_DELTAS))
    return F * jnp.exp(-d1 * vols * jnp.sqrt(T) + 0.5 * vols * vols * T), vols


def _quote_loss(raw, quotes, F, T, beta):
    """The loss `calibrate_sabr` minimises, as a function of the quotes."""
    K, vols = smile_from_quotes(quotes, F, T)
    F_vec, T_vec = jnp.full_like(K, F), jnp.full_like(K, T)
    weights = vega_weights(F_vec, K, T_vec, vols)
    return sabr_loss_for_points(raw, F_vec, K, T_vec, vols, beta, weights)[0]


def _tangent_solve(linearized, value):
    return jnp.linalg.solve(jax.jacfwd(linearized)(jnp.zeros_like(value)), value)


@partial(jit, static_argnames=("steps", "newton_steps"))
def calibrate_sabr_from_quotes(quotes, F, T, beta=1.0, lr=1e-2, steps=1200, newton_steps=20):
    """
    Calibrated (alpha, rho, nu) as an array, differentiable with respect to
    quotes, F, T and beta by the implicit function theorem. NaN if the
    calibration did not converge.
    """
    quotes = jnp.asarray(quotes, dtype=float)

    # The fit itself sees no tangents. All derivatives come from custom_root.
    q0, F0, T0, beta0, lr0 = lax.stop_gradient((quotes, F, T, beta, lr))
    K0, vols0 = smile_from_quotes(q0, F0, T0)
    fit = calibrate_sabr(F0, K0, T0, vols0, beta0, lr0, steps=steps, newton_steps=newton_steps)
    x0, unravel = ravel_pytree(fit["raw"])

    def optimality(x):
        return jax.grad(lambda v: _quote_loss(unravel(v), quotes, F, T, beta))(x)

    x = lax.custom_root(optimality, x0, lambda _, x_fit: x_fit, _tangent_solve)

    # Multiply rather than select with jnp.where: an unselected branch would
    # have zero, not NaN, derivatives.
    x = x * jnp.where(fit["converged"], 1.0, jnp.nan)
    alpha, _, rho, nu = sabr_params_from_raw(unravel(x), beta)
    return jnp.stack([alpha, rho, nu])


def pv_from_quotes(quotes, F, K, T, r_d, beta=1.0, is_call=True, **calibration):
    """
    Calibrate SABR to the quotes and price an option struck at K. `calibration`
    is passed to `calibrate_sabr_from_quotes`.
    """
    alpha, rho, nu = calibrate_sabr_from_quotes(quotes, F, T, beta, **calibration)
    return black_price(F, K, T, r_d, sabr_implied_vol(F, K, T, alpha, beta, rho, nu), is_call)


def market_quote_vega(quotes, F, K, T, r_d, beta=1.0, is_call=True, **calibration):
    """
    Price sensitivities to the quotes, in the order of `QUOTE_LABELS`, per 1
    vol point: how much of each quoted instrument's risk the option carries.
    One reverse-mode pass, whatever the number of quotes.
    """
    return 0.01 * jax.grad(pv_from_quotes)(
        jnp.asarray(quotes, dtype=float), F, K, T, r_d, beta, is_call, **calibration
    )
