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
vega-weighted squared vol error. The Adam loop is a `jax.lax.scan`, so one
calibration compiles to one function, and expiries with the same number of
strikes are calibrated together under `vmap`.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, value_and_grad, vmap
from jax.nn import softplus

# |z| below this uses the series for z/x(z). It balances the series'
# O(z^3) truncation against cancellation in x(z), both about 1e-12 here.
_Z_SERIES = 1e-4


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
    """
    F = jnp.clip(F, eps, None)
    K = jnp.clip(K, eps, None)
    T = jnp.clip(T, eps, None)
    beta = jnp.clip(beta, 0.0, 1.0)

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
# Calibration
# -----------------------------------------------------------------------------
@partial(jit, static_argnames=("steps", "vega_weight"))
def calibrate_sabr(F, Ks, T, market_ivs, beta=0.5, lr=1e-2, steps=1500, vega_weight=True):
    """
    Calibrate SABR (alpha, rho, nu) for one expiry with fixed beta.

    Pure and jittable: the Adam loop is a `lax.scan` over `steps` updates.
    F and T are scalars; Ks and market_ivs are arrays of the same length.

    Returns a dict of arrays: alpha, beta, rho, nu, model_ivs, final_loss,
    and per-step histories loss_history (the loss before each update) and
    alpha_history, rho_history, nu_history (the parameters after it).
    """
    K = jnp.asarray(Ks)
    F_vec = jnp.full_like(K, F)
    T_vec = jnp.full_like(K, T)
    mkt = jnp.asarray(market_ivs)

    weights = None
    if vega_weight:
        weights = black_vega(F_vec, K, T_vec, mkt)
        weights = weights / jnp.maximum(jnp.mean(weights), 1e-12)

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

    final_loss, model_iv = sabr_loss_for_points(raw_params, F_vec, K, T_vec, mkt, beta, weights)
    alpha, beta_used, rho, nu = sabr_params_from_raw(raw_params, beta)
    return {
        "alpha": alpha,
        "beta": beta_used,
        "rho": rho,
        "nu": nu,
        "model_ivs": model_iv,
        "final_loss": final_loss,
        "loss_history": losses,
        "alpha_history": alphas,
        "rho_history": rhos,
        "nu_history": nus,
    }


@partial(jit, static_argnames=("steps", "vega_weight"))
def calibrate_sabr_batch(F, Ks, T, market_ivs, beta=0.5, lr=1e-2, steps=1500, vega_weight=True):
    """`calibrate_sabr` over a leading expiry axis: F, T of shape (n,), Ks, market_ivs of shape (n, m)."""
    return vmap(
        lambda f, k, t, m: calibrate_sabr(f, k, t, m, beta, lr, steps=steps, vega_weight=vega_weight)
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
    return {
        "alpha": float(out["alpha"]),
        "beta": float(out["beta"]),
        "rho": float(out["rho"]),
        "nu": float(out["nu"]),
        "model_ivs": np.asarray(out["model_ivs"]),
        "final_loss": float(out["final_loss"]),
    }


def calibrate_sabr_single_expiry(F, Ks, T, market_ivs,
                                 beta=0.5,
                                 vega_weight=True,
                                 steps=1500,
                                 lr=1e-2,
                                 verbose=True,
                                 seed=0):
    """
    Calibrate SABR (alpha, rho, nu) for one expiry with fixed beta.

    Returns a dict of floats (alpha, beta, rho, nu, final_loss) and the
    model IVs as a NumPy array. With verbose, prints the loss and parameters
    every 200 steps. `seed` is accepted for compatibility; the
    initialisation is deterministic.
    """
    out = calibrate_sabr(F, jnp.asarray(Ks), T, jnp.asarray(market_ivs),
                         beta, lr, steps=steps, vega_weight=vega_weight)
    return _as_result(out, steps, verbose)


def calibrate_sabr_surface(forwards, strikes_list, maturities, market_ivs_list,
                           beta=0.5, steps=1200, lr=1e-2, vega_weight=True, verbose=False):
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
            beta, lr, steps=steps, vega_weight=vega_weight,
        )
        per_expiry = [jax.tree.map(lambda x, i=i: x[i], out) for i in range(len(forwards))]
    else:
        per_expiry = [
            calibrate_sabr(F, jnp.asarray(Ks), T, jnp.asarray(ivs), beta, lr,
                           steps=steps, vega_weight=vega_weight)
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
    beta: float = 0.5,
    steps: int = 1200,
    lr: float = 1e-2,
    vega_weight: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """
    Calibrate SABR per expiry from the Polars DataFrame produced by
    `jax_finance.fx.get_fx_market_data_for_calibration`. Polars is only
    needed to build the frame, not imported here.
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
    )

    for row, res in zip(rows, results):
        res["tenor"] = row["tenor"]
        res["tenor_years"] = row["tenor_years"]
        res["forward"] = row["forward"]

    return results
