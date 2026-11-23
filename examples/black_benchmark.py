"""
Benchmark: price and Greeks for a batch of Black-76 options, five ways.

    1. Price only                      jit(black_price)
    2. JAX AD, jitted                  batch_price_and_greeks (vmap + value_and_grad + grad(grad))
    2b. JAX AD, jitted, static branch  as 2, but black_price_static picks call/put with a Python
                                       `if`, so the put branch is never traced (Part 4)
    3. Bump-and-reprice, central diffs jit of 9 repricings (base + 2 bumps each for F, sigma, T, r)
    4. JAX AD, not jitted              the same vmap'd AD function, run op by op

All runs are float64 on whatever device JAX picks (CPU by default). Each
timing is the median of RUNS calls after one warm-up call, and every call
waits for its result with block_until_ready, so async dispatch isn't
mistaken for speed.

Usage: python examples/black_benchmark.py [n_options]
"""

import sys
from statistics import median
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.stats import norm

from jax_finance.black import black_price, batch_price_and_greeks, compute_greeks_single

RUNS = 20
RUNS_SLOW = 5      # the un-jitted path is slow; fewer repeats keep the script quick
IS_CALL = True

# Bump sizes for central differences (absolute)
H_F = 1e-4
H_SIGMA = 1e-4
H_T = 1e-5
H_R = 1e-5


def make_batch(n, seed=0):
    """n EUR/USD-style options around the Part 3 example, deterministic."""
    kF, kK, kT, kS = jax.random.split(jax.random.PRNGKey(seed), 4)
    F = 1.0850 * jnp.exp(0.02 * jax.random.normal(kF, (n,)))
    K = jax.random.uniform(kK, (n,), minval=0.95, maxval=1.25)
    T = jax.random.uniform(kT, (n,), minval=0.05, maxval=2.0)
    r_d = jnp.full((n,), 0.05)
    sigma = jax.random.uniform(kS, (n,), minval=0.05, maxval=0.20)
    return F, K, T, r_d, sigma


# ── the four methods ──────────────────────────────────────────────────────────

@jit
def price_only(F, K, T, r_d, sigma):
    return black_price(F, K, T, r_d, sigma, IS_CALL)


def ad_jitted(F, K, T, r_d, sigma):
    return batch_price_and_greeks(F, K, T, r_d, sigma, 'call')


def black_price_static(F, K, T, r_d, sigma, is_call=True):
    """black_price with the call/put choice made in Python (Part 4).

    `is_call` must be a plain Python bool: only one branch is traced, so the
    put's N(-d1) and N(-d2) never appear in a call's program.
    """
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * (sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r_d * T)
    if is_call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def greeks_single_static(F, K, T, r_d, sigma, is_call=True):
    """Same outputs and scaling as compute_greeks_single, on black_price_static."""
    p = lambda F_, T_, r_, s_: black_price_static(F_, K, T_, r_, s_, is_call)
    price, (dF, dT, dr, ds) = jax.value_and_grad(p, argnums=(0, 1, 2, 3))(F, T, r_d, sigma)
    gamma = jax.grad(jax.grad(p, argnums=0), argnums=0)(F, T, r_d, sigma)
    return {
        'price': price,
        'delta': dF,
        'gamma': gamma,
        'vega': ds * 0.01,
        'theta': -dT / 365.0,
        'rho': dr * 0.01,
    }


ad_jitted_static = jit(vmap(lambda f, k, t, r, s: greeks_single_static(f, k, t, r, s, IS_CALL)))


@jit
def bump_and_reprice(F, K, T, r_d, sigma):
    """Same outputs and scaling as compute_greeks_single, by central differences."""
    p = lambda F_, T_, r_, s_: black_price(F_, K, T_, r_, s_, IS_CALL)

    v0 = p(F, T, r_d, sigma)
    v_F_up, v_F_dn = p(F + H_F, T, r_d, sigma), p(F - H_F, T, r_d, sigma)
    v_s_up, v_s_dn = p(F, T, r_d, sigma + H_SIGMA), p(F, T, r_d, sigma - H_SIGMA)
    v_T_up, v_T_dn = p(F, T + H_T, r_d, sigma), p(F, T - H_T, r_d, sigma)
    v_r_up, v_r_dn = p(F, T, r_d + H_R, sigma), p(F, T, r_d - H_R, sigma)

    return {
        'price': v0,
        'delta': (v_F_up - v_F_dn) / (2 * H_F),
        'gamma': (v_F_up - 2 * v0 + v_F_dn) / H_F**2,
        'vega': (v_s_up - v_s_dn) / (2 * H_SIGMA) * 0.01,
        'theta': -(v_T_up - v_T_dn) / (2 * H_T) / 365.0,
        'rho': (v_r_up - v_r_dn) / (2 * H_R) * 0.01,
    }


# Identical to batch_price_and_greeks but without the jit: every primitive
# in the forward sweep, backward sweep and nested gamma pass is dispatched
# from Python one at a time.
_ad_eager = vmap(lambda f, k, t, r, s: compute_greeks_single(f, k, t, r, s, IS_CALL))


def ad_not_jitted(F, K, T, r_d, sigma):
    return _ad_eager(F, K, T, r_d, sigma)


# ── timing ────────────────────────────────────────────────────────────────────

def time_it(fn, args, runs=RUNS):
    """(first-call seconds, median seconds over `runs` later calls)."""
    start = perf_counter()
    jax.block_until_ready(fn(*args))
    first = perf_counter() - start

    durations = []
    for _ in range(runs):
        start = perf_counter()
        jax.block_until_ready(fn(*args))
        durations.append(perf_counter() - start)
    return first, median(durations)


def main(n):
    args = make_batch(n)
    print(f"JAX {jax.__version__}, device {jax.devices()[0]}, "
          f"dtype {args[0].dtype}, {n:,} options\n")

    methods = [
        ("Price only", "price", price_only, RUNS),
        ("JAX AD, jitted", "price, Δ, Γ, vega, θ, ρ", ad_jitted, RUNS),
        ("JAX AD, jitted, static branch", "same", ad_jitted_static, RUNS),
        ("Bump-and-reprice, central differences", "same", bump_and_reprice, RUNS),
        ("JAX AD, *not* jitted", "same", ad_not_jitted, RUNS_SLOW),
    ]

    times = {}
    for name, _, fn, runs in methods:
        first, t = time_it(fn, args, runs)
        times[name] = t
        print(f"  {name:<40} first call {first * 1e3:8.1f} ms   median {t * 1e3:8.2f} ms")

    base = times["Price only"]
    print("\n| Method | What it computes | Time | Relative to price only |")
    print("|---|---|---|---|")
    for name, what, _, _ in methods:
        t = times[name]
        print(f"| {name} | {what} | {t * 1e3:.1f} ms | {t / base:.1f}× |")

    # The static-branch AD must agree with the jnp.where version
    ad = ad_jitted(*args)
    st = ad_jitted_static(*args)
    print("\nMax absolute difference, static-branch AD vs AD:")
    for key in ('price', 'delta', 'gamma', 'vega', 'theta', 'rho'):
        print(f"  {key:<6} {float(jnp.max(jnp.abs(st[key] - ad[key]))):.1e}")

    # XLA expands erf into polynomials, so count exponentials in the compiled
    # program instead: each N(x) evaluation brings its own e^(-x^2) terms, so
    # fewer exponentials means fewer normal-CDF evaluations were kept.
    def n_exp(fn):
        return jit(fn).lower(*args).compile().as_text().count("exponential(")
    print(f"\nexponentials in compiled program: price {n_exp(price_only)}, "
          f"AD {n_exp(ad_jitted)}, static-branch AD {n_exp(ad_jitted_static)}")

    # Accuracy of bumping against AD
    fd = bump_and_reprice(*args)
    print("\nMax relative difference, bump-and-reprice vs AD:")
    for key in ('price', 'delta', 'gamma', 'vega', 'theta', 'rho'):
        rel = jnp.abs(fd[key] - ad[key]) / jnp.maximum(jnp.abs(ad[key]), 1e-12)
        print(f"  {key:<6} {float(jnp.max(rel)):.1e}   (median {float(jnp.median(rel)):.1e})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
