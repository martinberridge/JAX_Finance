# JAX SABR calibration with automatic differentiation (Hagan 2002 approximation)
# -----------------------------------------------------------------------------
# Requirements: Python >= 3.8, JAX (pip install jax jaxlib) if not already present.
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
from jax.nn import softplus
# -----------------------------------------------------------------------------
# 1) Numerics and transforms
# -----------------------------------------------------------------------------
def clamp_pos(x, eps=1e-12):
    # Softplus + epsilon to keep strictly positive
    return softplus(x) + eps

def clamp_rho(x):
    # Map R -> (-1, 1)
    return jnp.tanh(x)

def near_atm(F, K, atol=1e-10, rtol=1e-6):
    return jnp.isclose(F, K, rtol=rtol, atol=atol)
# Black forward vega (for optional weighting)

def black_vega(F, K, T, sigma, eps=1e-12):
    ln_fk = jnp.log((F + eps) / (K + eps))
    vol_sqrt_t = sigma * jnp.sqrt(jnp.maximum(T, eps))
    d1 = (ln_fk + 0.5 * sigma * sigma * jnp.maximum(T, eps)) / jnp.maximum(vol_sqrt_t, eps)
    nprime = jnp.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    # Forward vega (no discounting)
    return F * nprime * jnp.sqrt(jnp.maximum(T, eps))
# -----------------------------------------------------------------------------
# 2) Hagan (2002) SABR implied volatility approximation (vectorized)
#    sigma_SABR(F, K, T; alpha, beta, rho, nu)
# -----------------------------------------------------------------------------
def sabr_implied_vol(F, K, T, alpha, beta, rho, nu, eps=1e-12):
    """
    Vectorized Hagan SABR implied vol approximation with ATM fallback.
    F, K, T: tensors (broadcastable)
    alpha>0, nu>0; beta in [0,1]; rho in (-1,1)
    """
    # Ensure Forwards, Strikes and Maturities are strictly positive
    F = jnp.clip(F, eps, None)
    K = jnp.clip(K, eps, None)
    T = jnp.clip(T, eps, None) 

    beta = jnp.clip(beta, 0.0, 1.0)
    FK = F * K
    log_fk = jnp.log(F / K)
    pow_ = (1.0 - beta) / 2.0
    FK_pow = FK ** pow_
    # z and x(z)
    z = (nu / alpha) * FK_pow * log_fk
    sqrt_term = jnp.sqrt(jnp.maximum(1.0 - 2.0 * rho * z + z * z, eps))
    xz = jnp.log(jnp.maximum((sqrt_term + z - rho) / (1.0 - rho), eps))
    # D term (log-moneyness series)
    log_fk2 = log_fk * log_fk
    log_fk4 = log_fk2 * log_fk2
    D = 1.0 + (pow_ * pow_) * log_fk2 / 24.0 + (pow_ ** 4) * log_fk4 / 1920.0
    base = alpha / jnp.maximum(FK_pow * D, eps)
    # z/x(z), safe near z~0
    zx = z / jnp.maximum(xz, eps)
    zx = jnp.where(jnp.abs(z) < 1e-8, 1.0, zx)
    # Time adjustment
    term1 = (pow_ * pow_) * alpha * alpha / (24.0 * jnp.maximum(FK ** (1.0 - beta), eps))
    term2 = (rho * beta * nu * alpha) / (4.0 * jnp.maximum(FK_pow, eps))
    term3 = ((2.0 - 3.0 * rho * rho) * nu * nu) / 24.0
    time_adj = 1.0 + (term1 + term2 + term3) * T
    sigma = base * zx * time_adj
    # ATM fallback when K ~ F (must stay traceable under jit — no Python if on tracers)
    atm_mask = near_atm(F, K)
    F_beta = jnp.maximum(F ** (1.0 - beta), eps)
    atm_base = alpha / F_beta
    atm_time_adj = 1.0 + (
        (pow_ * pow_) * alpha * alpha / (24.0 * jnp.maximum(F ** (2.0 * (1.0 - beta)), eps))
        + (rho * beta * nu * alpha) / (4.0 * jnp.maximum(F_beta, eps))
        + ((2.0 - 3.0 * rho * rho) * nu * nu) / 24.0
    ) * T
    sigma_atm = atm_base * atm_time_adj
    sigma = jnp.where(atm_mask, sigma_atm, sigma)
    return jnp.clip(sigma, eps, None)

# -----------------------------------------------------------------------------
# 3) Loss and parameterization
# -----------------------------------------------------------------------------
def sabr_params_from_raw(raw, beta_fixed):
    """
    raw: dict with raw_alpha, raw_rho, raw_nu (unconstrained)
    returns: (alpha, beta, rho, nu) with constraints enforced
    """
    alpha = clamp_pos(raw["raw_alpha"])
    nu    = clamp_pos(raw["raw_nu"])
    rho   = clamp_rho(raw["raw_rho"])
    beta  = jnp.array(beta_fixed, dtype=jnp.float32)
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
# 4) Pure-JAX Adam optimizer
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
# 5) Single-expiry calibration
# -----------------------------------------------------------------------------
def calibrate_sabr_single_expiry(F, Ks, T, market_ivs,
                                 beta=0.5,
                                 vega_weight=True,
                                 steps=1500,
                                 lr=1e-2,
                                 verbose=True,
                                 seed=0):
    """
    Calibrate SABR (alpha, rho, nu) for one expiry with fixed beta.
    Inputs:
      F: scalar forward
      Ks: array of strikes
      T: scalar maturity (in years)
      market_ivs: array of market implied vols aligned with Ks
    """
    key = jax.random.PRNGKey(seed)

    # casting 
    F_arr = jnp.array(F, dtype=jnp.float32)
    K_arr = jnp.array(Ks, dtype=jnp.float32)
    T_arr = jnp.array(T, dtype=jnp.float32)
    T_arr = jnp.full_like(K_arr, T_arr)  # broadcast T to strikes
    mkt   = jnp.array(market_ivs, dtype=jnp.float32)

    # Initialization: use median IV to guess alpha (scaled by F^(1-beta))
    init_alpha = jnp.median(mkt) * (F_arr ** (1.0 - beta))
    raw = {
        "raw_alpha": jnp.log(jnp.maximum(init_alpha, 1e-4)),      # log-init for stability
        "raw_rho":   jnp.array(0.0),                              # start at 0 correlation
        "raw_nu":    jnp.log(jnp.array(0.5)),                     # vol-of-vol init
    }
    state = adam_init(raw)
    # Optional vega weights (from market IVs)
    weights = None
    if vega_weight:
        weights = black_vega(jnp.full_like(K_arr, F_arr), K_arr, T_arr, mkt)
        weights = weights / jnp.maximum(jnp.mean(weights), 1e-12)
    # JIT-compiled loss+grad function

    def loss_only(raw_params):
        loss_val, _ = sabr_loss_for_points(raw_params, jnp.full_like(K_arr, F_arr), K_arr, T_arr, mkt, beta, weights)
        return loss_val
    
    loss_and_grad = jit(value_and_grad(loss_only))
    # Training loop
    raw_params = raw
    for i in range(steps):
        loss_val, grads = loss_and_grad(raw_params)
        raw_params, state = adam_update(raw_params, grads, state, lr=lr)
        if verbose and (i % 200 == 0 or i == steps - 1):
            alpha, beta_used, rho, nu = sabr_params_from_raw(raw_params, beta)
            print(f"[Adam {i:04d}] loss={float(loss_val):.6f} "
                  f"alpha={float(alpha):.4f} rho={float(rho):.4f} nu={float(nu):.4f}")
    # Final model IVs and params
    final_loss, model_iv = sabr_loss_for_points(raw_params, jnp.full_like(K_arr, F_arr), K_arr, T_arr, mkt, beta, weights)
    alpha, beta_used, rho, nu = sabr_params_from_raw(raw_params, beta)
    result = {
        "alpha": float(alpha),
        "beta":  float(beta_used),
        "rho":   float(rho),
        "nu":    float(nu),
        "model_ivs": np.array(model_iv),   # convert to numpy for convenience
        "final_loss": float(final_loss),
    }
    return result
# -----------------------------------------------------------------------------
# 6) Multi-expiry calibration (independent per expiry)
# -----------------------------------------------------------------------------
def calibrate_sabr_surface(forwards, strikes_list, maturities, market_ivs_list,
                           beta=0.5, steps=1200, lr=1e-2, vega_weight=True, verbose=False):
    """
    Calibrate SABR for each expiry independently with fixed beta.
    Returns a list of dicts with calibrated parameters per expiry.
    """
    results = []
    for idx, (F, Ks, T, ivs) in enumerate(zip(forwards, strikes_list, maturities, market_ivs_list)):
        if verbose:
            print(f"\n=== Expiry {idx}: T={T:.6f}, F={F:.6f}, beta={beta} ===")
        res = calibrate_sabr_single_expiry(F, Ks, T, ivs,
                                           beta=beta,
                                           steps=steps,
                                           lr=lr,
                                           vega_weight=vega_weight,
                                           verbose=verbose)
        results.append(res)
    return results
# -----------------------------------------------------------------------------
# 7) Synthetic demo (replace with your market data)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Single expiry synthetic example
    F = 100.0
    T = 0.5
    
    #  make an array of 17 strikes between 60 and 140
    Ks = np.linspace(60, 140, 17).astype(np.float32)
    
    # True params for generating our synthetic market data
    alpha_true = 0.25
    beta_true  = 0.5
    rho_true   = -0.3
    nu_true    = 0.6
    # Generate synthetic "market" IVs with small noise - make the maturities and forwards 
    F_vec = jnp.full_like(jnp.array(Ks), F)
    T_vec = jnp.full_like(jnp.array(Ks), T)
    iv_true = sabr_implied_vol(F_vec, jnp.array(Ks), T_vec,
                               jnp.array(alpha_true), jnp.array(beta_true),
                               jnp.array(rho_true), jnp.array(nu_true))
    market_ivs = np.array(iv_true) + 0.003 * np.random.randn(len(Ks))

    print("Calibrating single expiry...")
    result = calibrate_sabr_single_expiry(F, Ks, T, market_ivs,
                                          beta=beta_true, steps=1500, lr=8e-3, vega_weight=True, verbose=True)
    print("\nCalibrated parameters:")
    print(f"alpha={result['alpha']:.6f}, beta={result['beta']:.6f}, rho={result['rho']:.6f}, nu={result['nu']:.6f}")
    print(f"Final loss={result['final_loss']:.6e}")
    
    # Multi-expiry synthetic example
    forwards = [100.0, 102.5, 105.0]
    maturities = [0.25, 0.5, 1.0]
    strikes_list = [np.linspace(70, 130, 13).astype(np.float32) for _ in forwards]
    market_ivs_list = []
    for F_i, T_i, Ks_i in zip(forwards, maturities, strikes_list):
        iv_i = sabr_implied_vol(jnp.full_like(jnp.array(Ks_i), F_i), jnp.array(Ks_i), jnp.full_like(jnp.array(Ks_i), T_i),
                                jnp.array(alpha_true), jnp.array(beta_true),
                                jnp.array(rho_true), jnp.array(nu_true))
        market_ivs_list.append(np.array(iv_i) + 0.0025 * np.random.randn(len(Ks_i)))

    print("\nCalibrating multiple expiries...")
    results_surface = calibrate_sabr_surface(forwards, strikes_list, maturities, market_ivs_list,
                                             beta=beta_true, steps=1200, lr=8e-3, vega_weight=True, verbose=False)

    for i, r in enumerate(results_surface):
        print(f'{maturities[i]=}, {forwards[i]=}, {strikes_list[i]=}, {r["alpha"]=}, {r["beta"]=}, {r["rho"]=}, {r["nu"]=}, {r["final_loss"]=}')

# ### How to use this
# 1. Replace the **synthetic demo** at the bottom with your **market data**:
#     - `F` (forward), `Ks` (strikes), `T` (maturity in years), `market_ivs` (implied vols).
# 2. Choose a fixed **β (beta)** per expiry or across the surface (commonly 0.0–1.0; e.g., 0.5).
# 3. Tune `steps` and `lr`. Start with `steps=1500` and `lr=5e-3–1e-2`; increase steps for tougher fits.
# 4. If your smile has **noisy wings**, consider:
#     - Down‑weighting extreme strikes (custom `weights`),
#     - Adding a small **regularization** term to the loss (e.g., penalize large `ν` or extreme `ρ`).
# ### Notes
# - This example uses **Adam only**. For second‑order refinement (e.g., L‑BFGS), consider **JAXopt** (`pip install jaxopt`) and plug its solver into `loss_only`.
# - The **ATM fallback** improves stability when `K ≈ F`; you can relax the isclose tolerances if needed.
# - For **joint calibration** (shared parameters across expiries), stack all expiries’ points into one loss and optimize a single parameter set—just pass concatenated arrays to `sabr_loss_for_points`.