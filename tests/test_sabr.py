"""
Correctness tests for jax_finance.sabr: gradients by check_grads, finite
derivatives at the money, recovery of known parameters, and agreement of the
batched calibration with the per-expiry one. Unlike test_regression.py, none
of these use the goldens.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from jax_finance.sabr import calibrate_sabr, calibrate_sabr_batch, sabr_implied_vol

# (F, T, alpha, beta, rho, nu)
PARAMS = [
    (100.0, 0.5, 1.0, 0.5, -0.3, 0.8),
    (1.085, 1.0, 0.10, 1.0, -0.2, 0.6),
    (0.03, 2.0, 0.01, 0.0, 0.3, 0.4),
]


@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("moneyness", [0.8, 0.97, 1.0, 1.03, 1.25])
def test_check_grads(p, moneyness):
    # check_grads steps every input by the same absolute eps, which is coarse
    # for F = 0.03 or alpha = 0.01, so perturb each input relative to its own
    # size. beta is held fixed, as in calibration: its clip to [0, 1] makes
    # beta = 0 and beta = 1 one-sided points.
    F, T, alpha, beta, rho, nu = p
    x0 = jnp.array([F, F * moneyness, T, alpha, rho, nu])

    def vol(u):
        F_, K_, T_, alpha_, rho_, nu_ = x0 * u
        return sabr_implied_vol(F_, K_, T_, alpha_, beta, rho_, nu_)

    check_grads(vol, (jnp.ones(6),), order=2, modes=["fwd", "rev"])


@pytest.mark.parametrize("p", PARAMS)
def test_derivatives_finite_at_atm(p):
    # Regression test for the double-where trap (gap 4): at K = F the direct
    # z/x(z) is 0/0, so value, gradient and Hessian must come from the series.
    F, T, alpha, beta, rho, nu = p
    args = (F, F, T, alpha, beta, rho, nu)
    assert np.isfinite(float(sabr_implied_vol(*args)))
    for order in (jax.grad, jax.hessian):
        derivs = order(sabr_implied_vol, argnums=(0, 1, 3, 5, 6))(*args)
        assert np.all(np.isfinite(np.asarray(jax.tree.leaves(derivs)))), order.__name__


def _smile(F, T, alpha, beta, rho, nu, n=13):
    atm_vol = alpha / F ** (1.0 - beta)
    K = F * jnp.exp(jnp.linspace(-2.5, 2.5, n) * atm_vol * jnp.sqrt(T))
    return K, sabr_implied_vol(F, K, T, alpha, beta, rho, nu)


@pytest.mark.parametrize("p", PARAMS[:2])
def test_calibration_recovers_parameters(p):
    F, T, alpha, beta, rho, nu = p
    K, vols = _smile(*p)
    out = calibrate_sabr(F, K, T, vols, beta, 8e-3, steps=3000)
    np.testing.assert_allclose([float(out["alpha"]), float(out["rho"]), float(out["nu"])],
                               [alpha, rho, nu], rtol=1e-8)


def test_batch_matches_per_expiry_calibration():
    expiries = [(100.0, 0.25, 1.0, 0.5, -0.3, 0.8),
                (102.5, 0.5, 0.95, 0.5, -0.25, 0.6),
                (105.0, 1.0, 0.9, 0.5, -0.2, 0.45)]
    smiles = [_smile(*e) for e in expiries]
    F = jnp.array([e[0] for e in expiries])
    T = jnp.array([e[1] for e in expiries])
    K = jnp.stack([s[0] for s in smiles])
    vols = jnp.stack([s[1] for s in smiles])
    batch = calibrate_sabr_batch(F, K, T, vols, 0.5, 8e-3, steps=1200)
    for i in range(len(expiries)):
        single = calibrate_sabr(F[i], K[i], T[i], vols[i], 0.5, 8e-3, steps=1200)
        for key in ("alpha", "rho", "nu", "final_loss"):
            np.testing.assert_allclose(float(batch[key][i]), float(single[key]),
                                       rtol=1e-10, atol=1e-20, err_msg=key)
