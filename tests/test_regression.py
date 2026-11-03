"""
Regression tests for the jax_finance package against the Step 0.5 goldens
(docs/refactor-plan.md, Step 1). How each golden was made, and why, is in
tests/golden/capture_v0.py.
"""

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_finance
from jax_finance.black import FXOptionPricer, batch_price_and_greeks
from jax_finance.sabr import (
    adam_init,
    adam_update,
    black_vega,
    calibrate_sabr,
    calibrate_sabr_surface,
    sabr_implied_vol,
    sabr_loss_for_points,
    sabr_params_from_raw,
)

from conftest import read_golden

ROOT = Path(__file__).resolve().parents[1]
GREEKS = ["price", "delta", "gamma", "vega", "theta", "rho"]
SABR_INPUTS = ["F", "K", "T", "alpha", "beta", "rho", "nu"]


def test_x64_enabled():
    assert jax.config.jax_enable_x64
    assert jnp.asarray(1.0).dtype == jnp.float64


# -----------------------------------------------------------------------------
# Black
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def black_golden():
    return read_golden("black_v0.csv")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black_matches_v0(black_golden, option_type):
    g = black_golden
    sel = g["option_type"] == option_type
    out = batch_price_and_greeks(g["F"][sel], g["K"][sel], g["T"][sel], g["r"][sel],
                                 g["sigma"][sel], option_type)
    for col in GREEKS:
        np.testing.assert_allclose(np.asarray(out[col]), g[col][sel], rtol=1e-10, atol=1e-14,
                                   err_msg=f"{option_type} {col}")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_option_pricer_matches_v0(black_golden, option_type):
    g = black_golden
    i = int(np.flatnonzero(g["option_type"] == option_type)[0])
    greeks = FXOptionPricer(option_type).all_greeks(g["F"][i], g["K"][i], g["T"][i], g["r"][i], g["sigma"][i])
    for col in GREEKS:
        np.testing.assert_allclose(greeks[col], g[col][i], rtol=1e-10, atol=1e-14, err_msg=col)


# -----------------------------------------------------------------------------
# SABR implied vol
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sabr_ref():
    return read_golden("sabr_vol_ref.csv")


def test_sabr_vol_matches_quantlib(sabr_ref):
    s = sabr_ref
    vol = sabr_implied_vol(*(s[c] for c in SABR_INPUTS))
    np.testing.assert_allclose(np.asarray(vol), s["vol"], rtol=1e-10)


def _param_sets(s):
    keys = ["F", "T", "alpha", "beta", "rho", "nu"]
    rows = {tuple(float(s[k][i]) for k in keys) for i in range(len(s["F"]))}
    return [dict(zip(keys, r)) for r in sorted(rows)]


def test_sabr_continuous_at_atm(sabr_ref):
    h = 1e-8
    for p in _param_sets(sabr_ref):
        F = p["F"]
        K = jnp.array([F * (1 - h), F, F * (1 + h)])
        v = np.asarray(sabr_implied_vol(F, K, p["T"], p["alpha"], p["beta"], p["rho"], p["nu"]))
        np.testing.assert_allclose(v[1], 0.5 * (v[0] + v[2]), rtol=1e-12, err_msg=str(p))


def test_sabr_gradient_finite_at_atm(sabr_ref):
    for p in _param_sets(sabr_ref):
        grads = jax.grad(sabr_implied_vol, argnums=(1, 3, 5, 6))(
            p["F"], p["F"], p["T"], p["alpha"], p["beta"], p["rho"], p["nu"]
        )
        assert all(np.isfinite(float(g)) for g in grads), p


# -----------------------------------------------------------------------------
# SABR calibration
# -----------------------------------------------------------------------------
def _old_loop_reference(F, Ks, T, market_ivs, beta, steps, lr):
    """
    The pre-Step-1 SABR_JAX.calibrate_sabr_single_expiry: a Python for loop
    over jitted Adam steps. Ported in float64 on the new sabr_implied_vol.
    """
    K = jnp.asarray(Ks)
    F_vec = jnp.full_like(K, F)
    T_vec = jnp.full_like(K, T)
    mkt = jnp.asarray(market_ivs)

    weights = black_vega(F_vec, K, T_vec, mkt)
    weights = weights / jnp.maximum(jnp.mean(weights), 1e-12)

    init_alpha = jnp.median(mkt) * (F ** (1.0 - beta))
    raw = {
        "raw_alpha": jnp.log(jnp.maximum(init_alpha, 1e-4)),
        "raw_rho": jnp.array(0.0),
        "raw_nu": jnp.log(jnp.array(0.5)),
    }
    state = adam_init(raw)

    def loss_only(raw_params):
        loss_val, _ = sabr_loss_for_points(raw_params, F_vec, K, T_vec, mkt, beta, weights)
        return loss_val

    loss_and_grad = jax.jit(jax.value_and_grad(loss_only))
    for _ in range(steps):
        _, grads = loss_and_grad(raw)
        raw, state = adam_update(raw, grads, state, lr=lr)

    alpha, _, rho, nu = sabr_params_from_raw(raw, beta)
    return np.array([float(alpha), float(rho), float(nu)])


@pytest.fixture(scope="module")
def calib():
    data = read_golden("sabr_calib_input.csv")
    true = read_golden("sabr_calib_true.csv")
    expiries = []
    for j, expiry in enumerate(true["expiry"]):
        sel = data["expiry"] == expiry
        expiries.append(dict(
            F=float(data["F"][sel][0]), T=float(data["T"][sel][0]), K=data["K"][sel],
            market_vol=data["market_vol"][sel], beta=float(true["beta"][j]),
            steps=int(true["n_steps"][j]), lr=float(true["learning_rate"][j]),
            true=np.array([true["alpha"][j], true["rho"][j], true["nu"][j]]),
        ))
    for e in expiries:
        e["reference"] = _old_loop_reference(e["F"], e["K"], e["T"], e["market_vol"],
                                             e["beta"], e["steps"], e["lr"])
    return expiries


def test_calibration_scan_matches_old_loop(calib):
    for e in calib:
        out = calibrate_sabr(e["F"], e["K"], e["T"], e["market_vol"], e["beta"], e["lr"],
                             steps=e["steps"], vega_weight=True)
        got = np.array([float(out["alpha"]), float(out["rho"]), float(out["nu"])])
        np.testing.assert_allclose(got, e["reference"], rtol=1e-10)


def test_calibration_vmap_matches_old_loop(calib):
    beta, steps, lr = calib[0]["beta"], calib[0]["steps"], calib[0]["lr"]
    results = calibrate_sabr_surface([e["F"] for e in calib], [e["K"] for e in calib],
                                     [e["T"] for e in calib], [e["market_vol"] for e in calib],
                                     beta=beta, steps=steps, lr=lr, vega_weight=True)
    for e, r in zip(calib, results):
        np.testing.assert_allclose([r["alpha"], r["rho"], r["nu"]], e["reference"], rtol=1e-10)


def test_calibration_recovers_true_parameters(calib):
    for e in calib:
        out = calibrate_sabr(e["F"], e["K"], e["T"], e["market_vol"], e["beta"], e["lr"],
                             steps=e["steps"], vega_weight=True)
        got = np.array([float(out["alpha"]), float(out["rho"]), float(out["nu"])])
        np.testing.assert_allclose(got, e["true"], rtol=1e-8)


# -----------------------------------------------------------------------------
# Shims and demos
# -----------------------------------------------------------------------------
FX_UTILS_NAMES = ["tenor_to_year_fraction", "get_forwards", "FXVolSurface",
                  "flatten_tuple_to_dict", "build_vol_surfaces",
                  "get_fx_market_data_for_calibration"]
FX_DATA_NAMES = ["FWD_Points", "atm", "rrs", "bfs", "rates", "spot"]


def test_fx_shims_expose_old_names():
    pytest.importorskip("QuantLib")
    import fx_data
    import fx_utils
    import jax_finance.fx.data
    import jax_finance.fx.utils

    for name in FX_UTILS_NAMES:
        assert getattr(fx_utils, name) is getattr(jax_finance.fx.utils, name), name
    for name in FX_DATA_NAMES:
        assert getattr(fx_data, name) is getattr(jax_finance.fx.data, name), name


def _run_demo(script):
    proc = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_black_demo_runs():
    _run_demo("JAX_black_option_pricing.py")


def test_sabr_demo_runs():
    pytest.importorskip("QuantLib")
    _run_demo("SABR_JAX.py")
