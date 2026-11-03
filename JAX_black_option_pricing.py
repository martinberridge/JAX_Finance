"""
FX option pricing with Black-76 and JAX automatic differentiation: a demo.

The pricer lives in `jax_finance.black` (formula and Greek conventions in
its docstring). Importing `jax_finance` turns on float64 (x64), so the
output differs from the pre-refactor float32 version in the last digits.
Run from the repo root, or after `pip install -e .`.
"""

import jax.numpy as jnp

# black_price and compute_greeks_single are unused here, but importing them
# keeps `from JAX_black_option_pricing import ...` working for old callers.
from jax_finance.black import (
    FXOptionPricer,
    batch_price_and_greeks,
    black_price,
    compute_greeks_single,
)


if __name__ == "__main__":
    
    print("=" * 70)
    print("FX OPTION PRICING WITH JAX AUTOMATIC DIFFERENTIATION (OPTIMIZED)")
    print("=" * 70)
    
    # Market data
    F = 1.0850        # Forward EUR/USD rate
    K = 1.0800        # Strike price
    T = 0.25          # 3 months to maturity
    r_d = 0.05        # USD interest rate (5%)
    sigma = 0.10      # Implied volatility (10%)
    
    print(f"\nMarket Parameters:")
    print(f"  Forward Rate (F):      {F}")
    print(f"  Strike Price (K):      {K}")
    print(f"  Time to Maturity (T):  {T} years")
    print(f"  Discount Rate (r_d):   {r_d}")
    print(f"  Volatility (sigma):    {sigma}")
    
    # Call Option
    call_pricer = FXOptionPricer(option_type='call')
    greeks = call_pricer.all_greeks(F, K, T, r_d, sigma)
    
    print(f"\nCALL OPTION GREEKS:")
    print(f"  Price:                 {greeks['price']:.6f}")
    print(f"  Delta (ΔPrice/ΔF):     {greeks['delta']:.6f}")
    print(f"  Gamma (Δ²Price/ΔF²):   {greeks['gamma']:.6f}")
    print(f"  Vega (ΔPrice/Δσ%):     {greeks['vega']:.6f}")
    print(f"  Theta (ΔPrice/Δt day): {greeks['theta']:.6f}")
    print(f"  Rho (ΔPrice/Δr%):      {greeks['rho']:.6f}")
    
    # Put Option
    put_pricer = FXOptionPricer(option_type='put')
    greeks_put = put_pricer.all_greeks(F, K, T, r_d, sigma)
    
    print(f"\nPUT OPTION GREEKS:")
    print(f"  Price:                 {greeks_put['price']:.6f}")
    print(f"  Delta (ΔPrice/ΔF):     {greeks_put['delta']:.6f}")
    print(f"  Gamma (Δ²Price/ΔF²):   {greeks_put['gamma']:.6f}")
    print(f"  Vega (ΔPrice/Δσ%):     {greeks_put['vega']:.6f}")
    print(f"  Theta (ΔPrice/Δt day): {greeks_put['theta']:.6f}")
    print(f"  Rho (ΔPrice/Δr%):      {greeks_put['rho']:.6f}")
    
    # Put-call parity check
    parity_diff = greeks['price'] - greeks_put['price'] - (F - K) * jnp.exp(-r_d * T)
    print(f"\nPUT-CALL PARITY CHECK:")
    print(f"  C - P - (F-K)e^(-rT) = {parity_diff:.10f} (should be ~0)")
    
    # Batch computation example
    print(f"\n" + "=" * 70)
    print("BATCH COMPUTATION (Multiple Options)")
    print("=" * 70)
    
    strikes = jnp.array([1.06, 1.07, 1.08, 1.09, 1.10])
    F_batch = jnp.full_like(strikes, F)
    T_batch = jnp.full_like(strikes, T)
    r_d_batch = jnp.full_like(strikes, r_d)
    sigma_batch = jnp.full_like(strikes, sigma)
    
    results = batch_price_and_greeks(F_batch, strikes, T_batch, r_d_batch, sigma_batch, 'call')
    
    print(f"\nStrike    Price      Delta      Gamma      Vega       Theta")
    print("-" * 70)
    for i, strike in enumerate(strikes):
        print(f"{strike:.4f}    {results['price'][i]:8.6f}   "
              f"{results['delta'][i]:8.6f}   {results['gamma'][i]:8.6f}   "
              f"{results['vega'][i]:8.6f}   {results['theta'][i]:8.6f}")