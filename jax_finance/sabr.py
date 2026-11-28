"""
SABR implied volatility (Hagan et al. 2002) and calibration with Adam.

For forward F, strike K, expiry T and parameters alpha > 0, 0 <= beta <= 1,
-1 < rho < 1 and nu > 0, the Hagan lognormal implied vol is

    sigma_B = alpha / { (FK)^((1-beta)/2)
                        * [1 + (1-beta)^2/24 L^2 + (1-beta)^4/1920 L^4] }
              * z / x(z)
              * { 1 + [ (1-beta)^2/24 * alpha^2 / (FK)^(1-beta)
                        + rho beta nu alpha / (4 (FK)^((1-beta)/2))
                        + (2 - 3 rho^2)/24 * nu^2 ] T }

    L = ln(F/K),   z = (nu/alpha) (FK)^((1-beta)/2) L,
    x(z) = ln{ [sqrt(1 - 2 rho z + z^2) + z - rho] / (1 - rho) }

At K = F, z = 0 and z/x(z) -> 1, which gives the usual ATM formula. Near
z = 0, z/x(z) is computed from its series 1 - rho z/2 - (3 rho^2 - 2) z^2/12
(as QuantLib does), and the direct formula is evaluated at a dummy z there
(the "double where"), so neither the value nor its gradient sees 0/0.

This replaces the pre-refactor `SABR_JAX.sabr_implied_vol`, which returned
its eps floor for every K > F and had the (1-beta)^2/24 coefficients a
factor of 4 too small (notes/refactor-plan.md, Step 0.5). It agrees with
QuantLib's `sabrVolatility` to float64 precision.

Calibration fits (alpha, rho, nu) per expiry with beta fixed, by Adam on a
vega-weighted squared vol error, then a few guarded Newton steps on the same
loss. Adam gets near the minimum but, with a constant learning rate, never
settles on it; Newton converges to rounding error. Results carry a
`converged` flag. The Adam loop is a `jax.lax.scan`, so one calibration
compiles to one function, and expiries with the same number of strikes are
calibrated together under `vmap`.

Sensitivities to the market quotes, taken through the calibration, are in
`jax_finance.sabr_risk`.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, value_and_grad, vmap
from jax.flatten_util import ravel_pytree
from jax.nn import softplus

# |z| below this uses the series for z/x(z). It balances the series'
# O(z^3) truncation against cancellation in x(z), both about 1e-12 here.
_Z_SERIES = 1e-4

# A fit counts as converged when the next Newton step would move every
# unconstrained parameter by less than this.
_NEWTON_TOL = 1e-8


# -----------------------------------------------------------------------------
# Numerics and transforms
# -----------------------------------------------------------------------------
def clamp_pos(x, eps=1e-12):
    # Softplus + epsilon to keep strictly positive
    return softplus(x) + eps


def clamp_rho(x):
    # Map R -> (-1, 1)
    return jnp.tanh(x)


def black_vega(F, K, T, sigma, eps=1e-12):
    """Undiscounted Black forward vega, used to weight the calibration loss."""
    ln_fk = jnp.log((F + eps) / (K + eps))
    vol_sqrt_t = sigma * jnp.sqrt(jnp.maximum(T, eps))
    d1 = (ln_fk + 0.5 * sigma * sigma * jnp.maximum(T, eps)) / jnp.maximum(vol_sqrt_t, eps)
    nprime = jnp.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return F * nprime * jnp.sqrt(jnp.maximum(T, eps))


# -----------------------------------------------------------------------------
# Hagan (2002) implied volatility
# -----------------------------------------------------------------------------
def sabr_implied_vol(F, K, T, alpha, beta, rho, nu, eps=1e-12):
    """
    Hagan SABR lognormal implied vol (formula in the module docstring).

    F, K, T: arrays (broadcastable), strictly positive
    alpha > 0, nu > 0; beta in [0, 1]; rho in (-1, 1)

    beta is not clipped: at beta = 0 and beta = 1, the usual choices, a clip
    would sit on its own kink and JAX would return half the beta derivative.
    """
    F = jnp.clip(F, eps, None)
    K = jnp.clip(K, eps, None)
    T = jnp.clip(T, eps, None)

    one_m_beta = 1.0 - beta
    FK = F * K
    log_fk = jnp.log(F / K)
    FK_pow = FK ** (one_m_beta / 2.0)

    # z / x(z), with the series near z = 0 and a dummy z in the direct branch
    z = (nu / alpha) * FK_pow * log_fk
    use_series = jnp.abs(z) < _Z_SERIES
    z_safe = jnp.where(use_series, 1.0, z)
    xz = jnp.log((jnp.sqrt(1.0 - 2.0 * rho * z_safe + z_safe * z_safe) + z_safe - rho) / (1.0 - rho))
    zx_series = 1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0
    zx = jnp.where(use_series, zx_series, z_safe / xz)

    log_fk2 = log_fk * log_fk
    D = 1.0 + one_m_beta**2 * log_fk2 / 24.0 + one_m_beta**4 * log_fk2 * log_fk2 / 1920.0

    time_adj = 1.0 + (
        one_m_beta**2 * alpha * alpha / (24.0 * FK**one_m_beta)
        + rho * beta * nu * alpha / (4.0 * FK_pow)
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    ) * T

    sigma = alpha / (FK_pow * D) * zx * time_adj
    return jnp.clip(sigma, eps, None)


# -----------------------------------------------------------------------------
# Loss and parameterisation
# -----------------------------------------------------------------------------
def sabr_params_from_raw(raw, beta_fixed):
    """
    raw: dict with raw_alpha, raw_rho, raw_nu (unconstrained)
    returns: (alpha, beta, rho, nu) with constraints enforced
    """
    alpha = clamp_pos(raw["raw_alpha"])
    nu = clamp_pos(raw["raw_nu"])
    rho = clamp_rho(raw["raw_rho"])
    beta = jnp.asarray(beta_fixed)
    return alpha, beta, rho, nu


def sabr_loss_for_points(raw, F, K, T, market_iv, beta_fixed=0.5, weights=None):
    alpha, beta, rho, nu = sabr_params_from_raw(raw, beta_fixed)
    model_iv = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
    err2 = (model_iv - market_iv) ** 2
    if weights is not None:
        w = jnp.maximum(weights, 1e-12)
        err2 = w * err2
    return jnp.mean(err2), model_iv


def vega_weights(F, K, T, market_iv):
    """Black vega at each quote, scaled to mean 1: the calibration weights."""
    weights = black_vega(F, K, T, market_iv)
    return weights / jnp.maximum(jnp.mean(weights), 1e-12)


# -----------------------------------------------------------------------------
# Pure-JAX Adam
# -----------------------------------------------------------------------------
def adam_init(params):
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    t = jnp.array(0, dtype=jnp.int32)
    return {"m": m, "v": v, "t": t}


@jit
def adam_update(params, grads, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), state["v"], grads)
    m_hat = jax.tree.map(lambda m: m / (1.0 - b1 ** t), m)
    v_hat = jax.tree.map(lambda v: v / (1.0 - b2 ** t), v)
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return params, {"m": m, "v": v, "t": t}


# -----------------------------------------------------------------------------
# Newton polish - finishing an optimisation with a few Newton steps, 
# once a cheaper first-order method like Adam has got close to the minimum
# -----------------------------------------------------------------------------
def newton_direction(loss, x):
    """
    Newton step for a scalar loss of a flat vector, using |eigenvalues| of the
    Hessian so that it points downhill at saddles and maxima too. Returns the
    step and the smallest (signed) eigenvalue.
    """
    eigvals, Q = jnp.linalg.eigh(jax.hessian(loss)(x))
    scale = jnp.abs(eigvals)
    scale = jnp.maximum(scale, 1e-12 * jnp.max(scale) + jnp.finfo(x.dtype).tiny)
    return -Q @ ((Q.T @ jax.grad(loss)(x)) / scale), eigvals[0]


def newton_polish(loss, x, steps):
    """
    `steps` guarded Newton steps: each has length at most 1, and is rejected
    and retried at a quarter of the size if it raises the loss. The 1e-9
    allows for rounding in the loss, whose residuals are differences of
    nearly equal vols.
    """
    def step(carry, _):
        x, val, damping = carry
        direction, _ = newton_direction(loss, x)
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1.0)
        x_new = x + damping * direction
        val_new = loss(x_new)
        accept = val_new <= val + 1e-9 * jnp.abs(val)
        x = jnp.where(accept, x_new, x)
        val = jnp.where(accept, val_new, val)
        damping = jnp.where(accept, jnp.minimum(4.0 * damping, 1.0), damping / 4.0)
        return (x, val, damping), None

    (x, _, _), _ = lax.scan(step, (x, loss(x), jnp.ones((), x.dtype)), None, length=steps)
    return x


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
@partial(jit, static_argnames=("steps", "vega_weight", "newton_steps"))
def calibrate_sabr(F, Ks, T, market_ivs, beta=0.5, lr=1e-2, steps=1500, vega_weight=True,
                   newton_steps=20):
    """
    Calibrate SABR (alpha, rho, nu) for one expiry with fixed beta.

    Pure and jittable: the Adam loop is a `lax.scan` over `steps` updates,
    followed by `newton_steps` of `newton_polish` (0 for Adam alone).
    F and T are scalars; Ks and market_ivs are arrays of the same length.

    Returns a dict of arrays: alpha, beta, rho, nu, model_ivs, final_loss,
    raw (the unconstrained parameters), and

    - converged: the fit is at a minimum, to rounding. newton_step is the
      largest move the next Newton step would make in any raw parameter.
    - per-step Adam histories: loss_history (the loss before each update) and
      alpha_history, rho_history, nu_history (the parameters after it).
    """
    K = jnp.asarray(Ks)
    F_vec = jnp.full_like(K, F)
    T_vec = jnp.full_like(K, T)
    mkt = jnp.asarray(market_ivs)

    weights = None
    if vega_weight:
        weights = vega_weights(F_vec, K, T_vec, mkt)

    # Initialisation: median IV to guess alpha (scaled by F^(1-beta))
    init_alpha = jnp.median(mkt) * (F ** (1.0 - beta))
    raw = {
        "raw_alpha": jnp.log(jnp.maximum(init_alpha, 1e-4)),
        "raw_rho": jnp.array(0.0),
        "raw_nu": jnp.log(jnp.array(0.5)),
    }

    def loss_only(raw_params):
        loss_val, _ = sabr_loss_for_points(raw_params, F_vec, K, T_vec, mkt, beta, weights)
        return loss_val

    def step(carry, _):
        raw_params, state = carry
        loss_val, grads = value_and_grad(loss_only)(raw_params)
        raw_params, state = adam_update(raw_params, grads, state, lr=lr)
        alpha, _, rho, nu = sabr_params_from_raw(raw_params, beta)
        return (raw_params, state), (loss_val, alpha, rho, nu)

    (raw_params, _), (losses, alphas, rhos, nus) = lax.scan(
        step, (raw, adam_init(raw)), None, length=steps
    )

    x, unravel = ravel_pytree(raw_params)
    flat_loss = lambda v: loss_only(unravel(v))
    x = newton_polish(flat_loss, x, newton_steps)
    raw_params = unravel(x)
    direction, min_curvature = newton_direction(flat_loss, x)
    newton_step = jnp.max(jnp.abs(direction))

    final_loss, model_iv = sabr_loss_for_points(raw_params, F_vec, K, T_vec, mkt, beta, weights)
    alpha, beta_used, rho, nu = sabr_params_from_raw(raw_params, beta)
    return {
        "alpha": alpha,
        "beta": beta_used,
        "rho": rho,
        "nu": nu,
        "model_ivs": model_iv,
        "final_loss": final_loss,
        "raw": raw_params,
        "newton_step": newton_step,
        "converged": (newton_step < _NEWTON_TOL) & (min_curvature > 0.0),
        "loss_history": losses,
        "alpha_history": alphas,
        "rho_history": rhos,
        "nu_history": nus,
    }


@partial(jit, static_argnames=("steps", "vega_weight", "newton_steps"))
def calibrate_sabr_batch(F, Ks, T, market_ivs, beta=0.5, lr=1e-2, steps=1500, vega_weight=True,
                         newton_steps=20):
    """`calibrate_sabr` over a leading expiry axis: F, T of shape (n,), Ks, market_ivs of shape (n, m)."""
    return vmap(
        lambda f, k, t, m: calibrate_sabr(f, k, t, m, beta, lr, steps=steps, vega_weight=vega_weight,
                                         newton_steps=newton_steps)
    )(F, Ks, T, market_ivs)


def _as_result(out, steps, verbose):
    """Convert one calibration's arrays to the dict of floats the wrappers return."""
    if verbose:
        for i in range(steps):
            if i % 200 == 0 or i == steps - 1:
                print(f"[Adam {i:04d}] loss={float(out['loss_history'][i]):.6f} "
                      f"alpha={float(out['alpha_history'][i]):.4f} "
                      f"rho={float(out['rho_history'][i]):.4f} "
                      f"nu={float(out['nu_history'][i]):.4f}")
        print(f"[Final    ] loss={float(out['final_loss']):.6f} alpha={float(out['alpha']):.4f} "
              f"rho={float(out['rho']):.4f} nu={float(out['nu']):.4f} "
              f"converged={bool(out['converged'])}")
    return {
        "alpha": float(out["alpha"]),
        "beta": float(out["beta"]),
        "rho": float(out["rho"]),
        "nu": float(out["nu"]),
        "model_ivs": np.asarray(out["model_ivs"]),
        "final_loss": float(out["final_loss"]),
        "converged": bool(out["converged"]),
    }


def calibrate_sabr_single_expiry(F, Ks, T, market_ivs,
                                 beta=0.5,
                                 vega_weight=True,
                                 steps=1500,
                                 lr=1e-2,
                                 verbose=True,
                                 seed=0,
                                 newton_steps=20):
    """
    Calibrate SABR (alpha, rho, nu) for one expiry with fixed beta.

    Returns a dict of floats (alpha, beta, rho, nu, final_loss), the model
    IVs as a NumPy array, and whether the fit converged. With verbose, prints the loss and parameters
    every 200 steps. `seed` is accepted for compatibility; the
    initialisation is deterministic.
    """
    out = calibrate_sabr(F, jnp.asarray(Ks), T, jnp.asarray(market_ivs),
                         beta, lr, steps=steps, vega_weight=vega_weight, newton_steps=newton_steps)
    return _as_result(out, steps, verbose)


def calibrate_sabr_surface(forwards, strikes_list, maturities, market_ivs_list,
                           beta=0.5, steps=1200, lr=1e-2, vega_weight=True, verbose=False,
                           newton_steps=20):
    """
    Calibrate SABR for each expiry independently with fixed beta.

    When every expiry has the same number of strikes they are calibrated
    together with `calibrate_sabr_batch`; otherwise one at a time.
    Returns a list of result dicts, one per expiry.
    """
    if len({len(k) for k in strikes_list}) == 1:
        out = calibrate_sabr_batch(
            jnp.asarray(forwards, dtype=float), jnp.asarray(np.stack(strikes_list), dtype=float),
            jnp.asarray(maturities, dtype=float), jnp.asarray(np.stack(market_ivs_list), dtype=float),
            beta, lr, steps=steps, vega_weight=vega_weight, newton_steps=newton_steps,
        )
        per_expiry = [jax.tree.map(lambda x, i=i: x[i], out) for i in range(len(forwards))]
    else:
        per_expiry = [
            calibrate_sabr(F, jnp.asarray(Ks), T, jnp.asarray(ivs), beta, lr,
                           steps=steps, vega_weight=vega_weight, newton_steps=newton_steps)
            for F, Ks, T, ivs in zip(forwards, strikes_list, maturities, market_ivs_list)
        ]

    results = []
    for idx, (F, T, out) in enumerate(zip(forwards, maturities, per_expiry)):
        if verbose:
            print(f"\n=== Expiry {idx}: T={T:.6f}, F={F:.6f}, beta={beta} ===")
        results.append(_as_result(out, steps, verbose))
    return results


# -----------------------------------------------------------------------------
# Calibration from an FX market-data frame
# -----------------------------------------------------------------------------
_STRIKE_COLS = (
    "surface_atm_strike",
    "surface_25_delta_call_strike",
    "surface_25_delta_put_strike",
    "surface_10_delta_call_strike",
    "surface_10_delta_put_strike",
)
_VOL_COLS = (
    "surface_atm_vol",
    "surface_25_delta_call_vol",
    "surface_25_delta_put_vol",
    "surface_10_delta_call_vol",
    "surface_10_delta_put_vol",
)
_REQUIRED_DF_COLS = ("tenor", "tenor_years", "forward") + _STRIKE_COLS + _VOL_COLS


def _expiry_slices_from_df(row: dict) -> tuple[float, float, np.ndarray, np.ndarray]:
    F = row["forward"]
    T = row["tenor_years"]
    Ks = np.array([row[c] for c in _STRIKE_COLS], dtype=np.float64)
    ivs = np.array([row[c] for c in _VOL_COLS], dtype=np.float64) / 100.0
    return F, T, Ks, ivs


def calibrate_sabr_surface_from_df(
    market_df,
    beta: float = 1.0,
    steps: int = 1200,
    lr: float = 1e-2,
    vega_weight: bool = True,
    verbose: bool = False,
    newton_steps: int = 20,
) -> list[dict]:
    """
    Calibrate SABR per expiry from the Polars DataFrame produced by
    `jax_finance.fx.get_fx_market_data_for_calibration`. Polars is only
    needed to build the frame, not imported here.

    beta defaults to 1, the usual FX choice, unlike the generic calibrators'
    0.5.
    """
    missing = [c for c in _REQUIRED_DF_COLS if c not in market_df.columns]
    if missing:
        raise ValueError(f"market_df missing required columns: {missing}")

    rows = list(market_df.iter_rows(named=True))
    slices = [_expiry_slices_from_df(row) for row in rows]
    forwards = [s[0] for s in slices]
    maturities = [s[1] for s in slices]
    strikes_list = [s[2] for s in slices]
    market_ivs_list = [s[3] for s in slices]

    results = calibrate_sabr_surface(
        forwards,
        strikes_list,
        maturities,
        market_ivs_list,
        beta=beta,
        steps=steps,
        lr=lr,
        vega_weight=vega_weight,
        verbose=verbose,
        newton_steps=newton_steps,
    )

    for row, res in zip(rows, results):
        res["tenor"] = row["tenor"]
        res["tenor_years"] = row["tenor_years"]
        res["forward"] = row["forward"]

    return results
