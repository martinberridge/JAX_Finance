"""
Correctness tests for jax_finance.black: AD Greeks against Black-76 closed
forms, put-call parity, and jit/vmap consistency.

The closed forms use math.erf, so they share no code with the JAX pricer.
With D = exp(-rT), N and n the standard normal CDF and PDF, w = +1 (call)
or -1 (put), and F held fixed:

    delta = w D N(w d1)
    gamma = D n(d1) / (F sigma sqrt(T))
    vega  = D F n(d1) sqrt(T) * 0.01
    rho   = -T V * 0.01
    theta = -(-r V + D F n(d1) sigma / (2 sqrt(T))) / 365
"""

import itertools
import math

import jax
import numpy as np
import pytest

from jax_finance.black import batch_price_and_greeks, black_price, compute_greeks_single

GREEKS = ["price", "delta", "gamma", "vega", "theta", "rho"]

# F, K/F, T, r, sigma: ITM to OTM, short to long expiry, low to high vol
GRID = list(itertools.product(
    [1.085, 100.0],
    [0.8, 0.95, 1.0, 1.05, 1.25],
    [0.1, 1.0, 5.0],
    [0.0, 0.04],
    [0.08, 0.35],
))


def _N(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _n(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def closed_form(F, K, T, r, sigma, is_call):
    w = 1.0 if is_call else -1.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    D = math.exp(-r * T)
    V = D * (w * F * _N(w * d1) - w * K * _N(w * d2))
    return {
        "price": V,
        "delta": w * D * _N(w * d1),
        "gamma": D * _n(d1) / (F * sigma * sqrt_T),
        "vega": D * F * _n(d1) * sqrt_T * 0.01,
        "rho": -T * V * 0.01,
        "theta": -(-r * V + D * F * _n(d1) * sigma / (2.0 * sqrt_T)) / 365.0,
    }


@pytest.mark.parametrize("is_call", [True, False], ids=["call", "put"])
def test_ad_greeks_match_closed_forms(is_call):
    for F, m, T, r, sigma in GRID:
        K = F * m
        ad = compute_greeks_single(F, K, T, r, sigma, is_call)
        cf = closed_form(F, K, T, r, sigma, is_call)
        for g in GREEKS:
            np.testing.assert_allclose(float(ad[g]), cf[g], rtol=1e-10,
                                       err_msg=f"{g} F={F} K/F={m} T={T} r={r} sigma={sigma}")


def test_put_call_parity():
    for F, m, T, r, sigma in GRID:
        K = F * m
        call = float(black_price(F, K, T, r, sigma, True))
        put = float(black_price(F, K, T, r, sigma, False))
        np.testing.assert_allclose(call - put, math.exp(-r * T) * (F - K), rtol=1e-12, atol=1e-14 * F)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_jit_and_vmap_match_scalar_calls(option_type):
    is_call = option_type == "call"
    F, m, T, r, sigma = (np.array(c, dtype=float) for c in zip(*GRID))
    K = F * m
    batch = batch_price_and_greeks(F, K, T, r, sigma, option_type)
    jitted = jax.jit(compute_greeks_single, static_argnums=5)
    # XLA fuses differently under jit, so deep-OTM prices near 1e-21 differ
    # by ~1e-33 absolute; atol covers that rounding, rtol everything else.
    for i in range(len(F)):
        scalar = compute_greeks_single(F[i], K[i], T[i], r[i], sigma[i], is_call)
        jit_one = jitted(F[i], K[i], T[i], r[i], sigma[i], is_call)
        for g in GREEKS:
            np.testing.assert_allclose(float(jit_one[g]), float(scalar[g]), rtol=1e-13, atol=1e-15, err_msg=g)
            np.testing.assert_allclose(float(batch[g][i]), float(scalar[g]), rtol=1e-13, atol=1e-15, err_msg=g)
