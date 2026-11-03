"""
SABR calibration with JAX automatic differentiation (Hagan 2002): a demo.

The model and calibrator live in `jax_finance.sabr` (formula in its
docstring), and the FX market-data helpers in `jax_finance.fx`, which needs
the `[fx]` extra. Importing `jax_finance` turns on float64 (x64).
Run from the repo root, or after `pip install -e .[fx]`.

The pre-refactor version of this file had two bugs in `sabr_implied_vol`
(every K > F returned its eps floor, and the (1-beta)^2/24 coefficients
were 4x too small), so its demo output was wrong; see
docs/refactor-plan.md, Step 0.5.
"""

import numpy as np
import jax.numpy as jnp

from jax_finance.fx import get_fx_market_data_for_calibration
from jax_finance.fx.data import atm, rrs, bfs, rates, spot, FWD_Points
from jax_finance.sabr import (
    calibrate_sabr_single_expiry,
    calibrate_sabr_surface,
    calibrate_sabr_surface_from_df,
    sabr_implied_vol,
)


# -----------------------------------------------------------------------------
# Synthetic demo (replace with your market data)
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

    print("\nSynthetic data - Calibrating multiple expiries...")
    results_surface = calibrate_sabr_surface(forwards, strikes_list, maturities, market_ivs_list,
                                             beta=beta_true, steps=1200, lr=8e-3, vega_weight=True, verbose=False)

    for i, r in enumerate(results_surface):
        print(f'{maturities[i]=}, {forwards[i]=}, {strikes_list[i]=}, {r["alpha"]=}, {r["beta"]=}, {r["rho"]=}, {r["nu"]=}, {r["final_loss"]=}')

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



    # FX Options versions using EURUSD market data.
    df = get_fx_market_data_for_calibration(atm, rrs, bfs, FWD_Points, spot, rates)
    results = calibrate_sabr_surface_from_df(df, beta=0.5, verbose=True)
    for r in results:
        print(
            f"{r['tenor']}: alpha={r['alpha']:.4f} rho={r['rho']:.4f} "
            f"nu={r['nu']:.4f} loss={r['final_loss']:.2e}"
        )

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
# - Near `K ≈ F`, `z/x(z)` switches to its series, so values and gradients stay finite at the money.
# - For **joint calibration** (shared parameters across expiries), stack all expiries’ points into one loss and optimize a single parameter set—just pass concatenated arrays to `sabr_loss_for_points`.