"""
Correctness tests for jax_finance.sabr: gradients by check_grads, finite
derivatives at the money, recovery of known parameters, convergence of the
Newton polish, and agreement of the batched calibration with the per-expiry
one. Unlike test_regression.py, none
of these use the goldens.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from jax_finance.sabr import calibrate_sabr, calibrate_sabr_batch, newton_polish, sabr_implied_vol

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
    # size, except beta, which can be 0.
    F, T, alpha, beta, rho, nu = p
    x0 = jnp.array([F, F * moneyness, T, alpha, rho, nu])

    def vol(u, dbeta):
        F_, K_, T_, alpha_, rho_, nu_ = x0 * u
        return sabr_implied_vol(F_, K_, T_, alpha_, beta + dbeta, rho_, nu_)

    check_grads(vol, (jnp.ones(6), 0.0), order=2, modes=["fwd", "rev"])


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_beta_gradient_at_the_ends_of_its_range(beta):
    # Regression test: a clip of beta to [0, 1] halved its gradient at both ends.
    def vol(b):
        return sabr_implied_vol(1.085, 1.15, 1.0, 0.1, b, -0.2, 0.6)

    h = 1e-6
    inside = -h if beta == 1.0 else h
    one_sided = (-3 * vol(beta) + 4 * vol(beta + inside) - vol(beta + 2 * inside)) / (2 * inside)
    np.testing.assert_allclose(float(jax.grad(vol)(beta)), float(one_sided), rtol=1e-6)


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


# EUR/USD overnight and 1y smiles (jax_finance.fx.data). Overnight is the hard
# one: nu is 9.7, far from its starting value, so 1200 Adam steps stop short.
OVERNIGHT = dict(
    F=1.085, T=1.0 / 365.0,
    vols=jnp.array([0.07045, 0.0700875, 0.0741625, 0.072, 0.0797]),
)
ONE_YEAR = dict(
    F=1.101577, T=1.0,
    vols=jnp.array([0.0659, 0.0682625, 0.0703375, 0.0764875, 0.0803125]),
)


def _delta_smile(F, T, vols):
    d1 = jax.scipy.stats.norm.ppf(jnp.array([0.50, 0.25, 0.75, 0.10, 0.90]))
    return F * jnp.exp(-d1 * vols * jnp.sqrt(T) + 0.5 * vols**2 * T)


def _fit(smile, beta, steps, newton_steps):
    K = _delta_smile(**smile)
    return calibrate_sabr(smile["F"], K, smile["T"], smile["vols"], beta, 1e-2,
                          steps=steps, newton_steps=newton_steps)


def test_converged_flag_catches_adam_stopping_short():
    adam = _fit(OVERNIGHT, 1.0, 1200, 0)
    polished = _fit(OVERNIGHT, 1.0, 1200, 20)
    assert not bool(adam["converged"])
    assert bool(polished["converged"])
    assert float(polished["final_loss"]) < 0.1 * float(adam["final_loss"])


@pytest.mark.parametrize("smile", [OVERNIGHT, ONE_YEAR])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_polished_fit_does_not_depend_on_adam_steps(smile, beta):
    # From 200 Adam steps an unguarded Newton goes to a spurious stationary
    # point of the overnight smile. At 6000 Adam alone has drifted.
    fits = [_fit(smile, beta, steps, 20) for steps in (200, 1200, 6000)]
    assert all(bool(f["converged"]) for f in fits)
    for f in fits[1:]:
        for key in ("alpha", "rho", "nu"):
            np.testing.assert_allclose(float(f[key]), float(fits[0][key]), rtol=1e-9, atol=1e-9,
                                       err_msg=key)


def test_polish_does_not_raise_the_loss():
    for steps in (1, 200, 1200):
        for smile in (OVERNIGHT, ONE_YEAR):
            before = float(_fit(smile, 0.5, steps, 0)["final_loss"])
            assert float(_fit(smile, 0.5, steps, 20)["final_loss"]) <= before * (1 + 1e-7)


def test_polish_leaves_a_saddle():
    # Minima at x = +-1, saddle at the origin: plain Newton goes to the saddle.
    loss = lambda v: v[0] ** 4 / 4 - v[0] ** 2 / 2 + v[1] ** 2 / 2
    np.testing.assert_allclose(newton_polish(loss, jnp.array([0.1, 1.0]), 20), [1.0, 0.0], atol=1e-8)


def test_polish_recovers_from_overshooting():
    # Plain Newton on sqrt(1 + u^2) diverges from any |u| > 1.
    loss = lambda v: jnp.sqrt(1.0 + (10.0 * v[0]) ** 2)
    np.testing.assert_allclose(newton_polish(loss, jnp.array([0.3]), 30), [0.0], atol=1e-7)
