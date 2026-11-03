"""
Black-76 option pricing with price and Greeks from automatic differentiation.

A European option on a forward F, struck at K, expiring in T years and
discounted at the continuously compounded rate r, is worth

    V = exp(-r T) * [ w F N(w d1) - w K N(w d2) ]

    d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)

with w = +1 for a call and -1 for a put, and N the standard normal CDF.

Greeks come from one reverse-mode pass (`value_and_grad`) for the first
derivatives and `grad(grad)` for gamma. They are quoted as:

    delta  dV/dF
    gamma  d2V/dF2
    vega   dV/dsigma * 0.01      per 1 vol point
    theta  -dV/dT / 365          per calendar day of decay
    rho    dV/dr * 0.01          per 1% rate move, with F held fixed

Holding F fixed means rho only sees the discount factor: rho = -0.01 T V.
"""

import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap
from jax.scipy.stats import norm


def black_price(F, K, T, r_d, sigma, is_call=True):
    """
    Black-76 price of a call or put.

    Args:
        F: Forward rate
        K: Strike
        T: Time to maturity (years)
        r_d: Domestic (discounting) rate, continuously compounded
        sigma: Implied volatility
        is_call: True for a call, False for a put

    Returns:
        Option price
    """
    w = jnp.where(is_call, 1.0, -1.0)
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r_d * T)
    return df * (w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2))


def compute_greeks_single(F, K, T, r_d, sigma, is_call=True):
    """
    Price and all Greeks for one option, scaled as in the module docstring.

    Returns:
        dict with keys price, delta, gamma, vega, theta, rho
    """
    # First-order derivatives w.r.t. (F, T, r_d, sigma) in one reverse-mode pass
    price, (delta, theta_raw, rho_raw, vega_raw) = value_and_grad(
        black_price, argnums=(0, 2, 3, 4)
    )(F, K, T, r_d, sigma, is_call)

    gamma = grad(grad(black_price, argnums=0), argnums=0)(F, K, T, r_d, sigma, is_call)

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega_raw * 0.01,
        'theta': -theta_raw / 365.0,
        'rho': rho_raw * 0.01,
    }


class FXOptionPricer:
    """Convenience wrapper: a JIT-compiled `compute_greeks_single` for one option type."""

    def __init__(self, option_type='call'):
        if option_type.lower() not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        self.is_call = (option_type.lower() == 'call')

        self._all_greeks_jit = jit(
            lambda F, K, T, r_d, sigma: compute_greeks_single(F, K, T, r_d, sigma, self.is_call)
        )

    def all_greeks(self, F, K, T, r_d, sigma):
        """Price and all Greeks as Python floats."""
        results = self._all_greeks_jit(F, K, T, r_d, sigma)
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


@jit(static_argnames=['option_type'])
def batch_price_and_greeks(F, K, T, r_d, sigma, option_type='call'):
    """Price and Greeks for a batch of options of one type, via `vmap`."""
    is_call = (option_type.lower() == 'call')
    batch_fn = vmap(lambda f, k, t, r, s: compute_greeks_single(f, k, t, r, s, is_call))
    return batch_fn(F, K, T, r_d, sigma)
