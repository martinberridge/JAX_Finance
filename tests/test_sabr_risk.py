"""
Tests for jax_finance.sabr_risk. The reference for every derivative is
bump-and-recalibrate on the polished fit, which shares the calibrator but
not the implicit function theorem.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_finance.black import black_price
from jax_finance.sabr import calibrate_sabr, sabr_implied_vol
from jax_finance.sabr_risk import (
    QUOTE_LABELS,
    calibrate_sabr_from_quotes,
    market_quote_vega,
    pv_from_quotes,
    smile_from_quotes,
)

# EUR/USD quotes (jax_finance.fx.data), in the order of QUOTE_LABELS, with a
# strike about one 25-delta call out of the money.
CASES = {
    "on": dict(quotes=jnp.array([7.045, -0.4075, 0.1675, -0.77, 0.54]) / 100,
               F=1.085, T=1.0 / 365.0, K=1.088),
    "3m": dict(quotes=jnp.array([5.87, -0.4075, 0.2175, -0.77, 0.79]) / 100,
               F=1.08955, T=0.25, K=1.11),
    "1y": dict(quotes=jnp.array([6.59, -0.2075, 0.34, -0.3825, 1.25]) / 100,
               F=1.101577, T=1.0, K=1.15),
}
R_D = 0.05
BETAS = [0.5, 1.0]

_pv = jax.jit(pv_from_quotes, static_argnames="is_call")


def _bump(f, x, h):
    return jnp.stack([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in jnp.eye(x.shape[0])])


def test_smile_matches_fx_helpers():
    pytest.importorskip("QuantLib")
    from jax_finance.fx import FXVolSurface

    for c in CASES.values():
        surface = FXVolSurface(1.0, 0.0, 0.0, c["T"], forward=c["F"]).build_from_quotes(
            atm_vol=100 * c["quotes"][0], rr_25=100 * c["quotes"][1], bf_25=100 * c["quotes"][2],
            rr_10=100 * c["quotes"][3], bf_10=100 * c["quotes"][4],
        )
        K, vols = smile_from_quotes(c["quotes"], c["F"], c["T"])
        points = ["atm", "25_delta_call", "25_delta_put", "10_delta_call", "10_delta_put"]
        np.testing.assert_allclose(K, [float(surface[p]["strike"]) for p in points], rtol=1e-12)
        np.testing.assert_allclose(vols, [float(surface[p]["vol"]) / 100 for p in points], rtol=1e-12)


@pytest.mark.parametrize("beta", BETAS)
def test_pv_matches_direct_calibration(beta):
    c = CASES["1y"]
    K, vols = smile_from_quotes(c["quotes"], c["F"], c["T"])
    fit = calibrate_sabr(c["F"], K, c["T"], vols, beta, 1e-2, steps=1200)
    vol = sabr_implied_vol(c["F"], c["K"], c["T"], fit["alpha"], beta, fit["rho"], fit["nu"])
    np.testing.assert_allclose(
        float(_pv(c["quotes"], c["F"], c["K"], c["T"], R_D, beta)),
        float(black_price(c["F"], c["K"], c["T"], R_D, vol)), rtol=1e-12)


@pytest.mark.parametrize("tenor", CASES)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("is_call", [True, False])
def test_quote_vega_matches_bump_and_recalibrate(tenor, beta, is_call):
    c = CASES[tenor]
    ad = market_quote_vega(c["quotes"], c["F"], c["K"], c["T"], R_D, beta, is_call)
    bump = 0.01 * _bump(lambda q: _pv(q, c["F"], c["K"], c["T"], R_D, beta, is_call),
                        c["quotes"], 1e-6)
    assert ad.shape == (len(QUOTE_LABELS),)
    np.testing.assert_allclose(ad, bump, rtol=1e-5, atol=1e-8 * float(jnp.max(jnp.abs(ad))))


@pytest.mark.parametrize("tenor", CASES)
def test_parameter_jacobian_matches_bump(tenor):
    c = CASES[tenor]
    f = lambda q: calibrate_sabr_from_quotes(q, c["F"], c["T"])
    fd = _bump(f, c["quotes"], 1e-6).T
    for jac in (jax.jacfwd(f), jax.jacrev(f)):
        np.testing.assert_allclose(jac(c["quotes"]), fd, rtol=1e-4,
                                   atol=1e-6 * float(jnp.max(jnp.abs(fd))))


def test_gradients_to_the_other_inputs_match_bump():
    # F, T and beta also reach the price through the calibration.
    c = CASES["1y"]
    f = lambda x: _pv(c["quotes"], x[0], c["K"], x[1], x[2], x[3])
    x = jnp.array([c["F"], c["T"], R_D, 0.7])
    np.testing.assert_allclose(jax.grad(f)(x), _bump(f, x, 1e-5), rtol=1e-5)


def test_cross_gamma_matches_bumped_gradient():
    c = CASES["1y"]
    f = lambda q: _pv(q, c["F"], c["K"], c["T"], R_D)
    H = jax.hessian(f)(c["quotes"])
    np.testing.assert_allclose(H, _bump(jax.grad(f), c["quotes"], 1e-5), rtol=1e-5)
    np.testing.assert_allclose(H, H.T, rtol=1e-8)


def test_unconverged_calibration_gives_nan_price_and_risk():
    # 1200 Adam steps stop short of the overnight minimum, where the implicit
    # function theorem would return wrong numbers rather than fail.
    c = CASES["on"]
    args = (c["quotes"], c["F"], c["K"], c["T"], R_D)
    assert np.isnan(float(pv_from_quotes(*args, newton_steps=0)))
    assert np.all(np.isnan(market_quote_vega(*args, newton_steps=0)))
    assert np.all(np.isfinite(market_quote_vega(*args)))


def test_vmap_over_expiries():
    quotes, F, K, T = (jnp.stack([jnp.asarray(c[k]) for c in CASES.values()])
                       for k in ("quotes", "F", "K", "T"))
    batch = jax.vmap(market_quote_vega, in_axes=(0, 0, 0, 0, None))(quotes, F, K, T, R_D)
    for i in range(len(CASES)):
        np.testing.assert_allclose(batch[i], market_quote_vega(quotes[i], F[i], K[i], T[i], R_D),
                                   rtol=1e-7, atol=1e-10)
