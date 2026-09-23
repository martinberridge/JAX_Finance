"""
FX Option Pricing using Black's Model with JAX (Optimized)
Automatically computes Price and all Greeks in a single JIT-compiled AD pass.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad, vmap
from jax.scipy.stats import norm
import numpy as np


# ============================================================================
# Core Black's Model Pricing Functions
# ============================================================================

def black_price(F, K, T, r_d, sigma, is_call=True):
    """
    Price a call or put option using Black's model.
    
    Args:
        F: Forward exchange rate
        K: Strike price
        T: Time to maturity (years)
        r_d: Domestic risk-free rate
        sigma: Implied volatility
        is_call: True for Call, False for Put
        
    Returns:
        Option price
    """
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * (sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r_d * T)
    
    # Standard normal CDF using jax.scipy.stats.norm
    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)
    
    call_price = df * (F * n_d1 - K * n_d2)
    put_price = df * (K * (1.0 - n_d2) - F * (1.0 - n_d1))
    
    return jnp.where(is_call, call_price, put_price)


def compute_greeks_single(F, K, T, r_d, sigma, is_call=True):
    """
    Computes Price and ALL Greeks in a SINGLE automatic differentiation pass.
    """
    # 1. First-order derivatives w.r.t (F, T, r_d, sigma) via reverse-mode AD
    price, (delta, theta_raw, rho_raw, vega_raw) = value_and_grad(
        black_price, argnums=(0, 2, 3, 4)
    )(F, K, T, r_d, sigma, is_call)

    # 2. Gamma: Second derivative d²Price / dF²
    gamma = grad(grad(black_price, argnums=0), argnums=0)(F, K, T, r_d, sigma, is_call)

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega_raw * 0.01,        # Scaled per 1% vol change
        'theta': -theta_raw / 365.0,    # Quoted per day decay
        'rho': rho_raw * 0.01,          # Scaled per 1% rate change
    }


# ============================================================================
# High-Performance FX Option Pricer
# ============================================================================

class FXOptionPricer:
    """
    FX Option pricer with single-pass JIT-compiled automatic differentiation.
    """
    
    def __init__(self, option_type='call'):
        self.is_call = (option_type.lower() == 'call')
        if option_type.lower() not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        
        # JIT-compile the single-pass price + all Greeks computation graph
        self._all_greeks_jit = jit(
            lambda F, K, T, r_d, sigma: compute_greeks_single(F, K, T, r_d, sigma, self.is_call)
        )
    
    def all_greeks(self, F, K, T, r_d, sigma):
        """Compute Price and all Greeks simultaneously in 1 JIT execution pass."""
        results = self._all_greeks_jit(F, K, T, r_d, sigma)
        # Convert JAX device buffers to float dict for Python output
        return {k: float(v) for k, v in results.items()}

    def price(self, F, K, T, r_d, sigma):
        return self.all_greeks(F, K, T, r_d, sigma)['price']
    
    def delta(self, F, K, T, r_d, sigma):
        return self.all_greeks(F, K, T, r_d, sigma)['delta']
    
    def gamma(self, F, K, T, r_d, sigma):
        return self.all_greeks(F, K, T, r_d, sigma)['gamma']
    
    def vega(self, F, K, T, r_d, sigma):
        return self.all_greeks(F, K, T, r_d, sigma)['vega']
    
    def theta(self, F, K, T, r_d, sigma):
        return self.all_greeks(F, K, T, r_d, sigma)['theta']
    
    def rho(self, F, K, T, r_d, sigma):
        return self.all_greeks(F, K, T, r_d, sigma)['rho']


# ============================================================================
# Efficient Batch Greeks Computation
# ============================================================================

@jit(static_argnames=['option_type'])
def batch_price_and_greeks(F, K, T, r_d, sigma, option_type='call'):
    """
    Computes prices and Greeks for batches of options in a single vmap step.
    """
    is_call = (option_type.lower() == 'call')
    
    # vmap the single-pass Greek function directly across the batch dimensions
    batch_fn = vmap(lambda f, k, t, r, s: compute_greeks_single(f, k, t, r, s, is_call))
    return batch_fn(F, K, T, r_d, sigma)


# ============================================================================
# Example Usage
# ============================================================================

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