"""
Tests for the FX market-data helpers in jax_finance.fx, which need the [fx]
extra. Each of the first four tests fails on one of the bugs fixed after
Step 1 (docs/refactor-plan.md).
"""

import numpy as np
import pytest

pytest.importorskip("QuantLib")

from jax_finance.fx import FXVolSurface, get_forwards, get_fx_market_data_for_calibration
from jax_finance.fx.data import FWD_Points, atm, bfs, rates, rrs, spot
from jax_finance.sabr import calibrate_sabr_surface_from_df

DELTAS = [(0.25, True), (0.10, True), (-0.25, False), (-0.10, False)]


@pytest.fixture(scope="module")
def market_df():
    return get_fx_market_data_for_calibration(atm, rrs, bfs, FWD_Points, spot, rates)


def test_forward_points_are_pips():
    fwds = get_forwards(spot, FWD_Points)
    assert fwds["1y"] == pytest.approx(spot + FWD_Points["1y"] * 1e-4)
    assert fwds["on"] == spot


def test_forward_points_match_rate_differential(market_df):
    # USD is domestic for EUR/USD, so the points imply r_dom - r_for close to
    # sofr - estr (1.58%). The flat overnight rates aren't a term structure,
    # so the implied differential ranges over 1.4%-1.8% across tenors.
    for row in market_df.iter_rows(named=True):
        if row["tenor"] == "on":
            continue
        implied = np.log(row["forward"] / spot) / row["tenor_years"]
        assert implied == pytest.approx(rates["sofr"] - rates["estr"], abs=5e-3), row["tenor"]


def test_surface_uses_quoted_forward(market_df):
    # The ATM strike is 50-delta about the same forward the calibration uses.
    for row in market_df.iter_rows(named=True):
        vol = row["surface_atm_vol"] / 100
        expected = row["forward"] * np.exp(vol**2 / 2 * row["tenor_years"])
        assert row["surface_atm_strike"] == pytest.approx(expected, rel=1e-14), row["tenor"]


@pytest.mark.parametrize("delta,is_call", DELTAS)
def test_strike_from_delta_inverts_delta_from_strike(delta, is_call):
    surface = FXVolSurface(spot, rates["sofr"], rates["estr"], tenor_years=1.0)
    K = surface.strike_from_delta(delta, 0.07, is_call=is_call)
    assert surface.delta_from_strike(K, 0.07, is_call=is_call) == pytest.approx(delta, rel=1e-12)
    # OTM: calls above the forward, puts below
    assert (K > surface.forward) == is_call


def test_atm_strike_is_50_delta():
    surface = FXVolSurface(spot, rates["sofr"], rates["estr"], tenor_years=1.0)
    K = surface.atm_strike(0.07)
    assert surface.delta_from_strike(K, 0.07, is_call=True) == pytest.approx(0.5, rel=1e-12)


def test_surface_strikes_ordered(market_df):
    cols = ["10_delta_put", "25_delta_put", "atm", "25_delta_call", "10_delta_call"]
    for row in market_df.iter_rows(named=True):
        strikes = [row[f"surface_{c}_strike"] for c in cols]
        assert strikes == sorted(strikes), row["tenor"]


def test_eurusd_sabr_calibration_fits_quotes(market_df):
    # Beyond overnight, SABR fits the five quoted vols to within 0.05 vol points.
    vol_cols = ["atm", "25_delta_call", "25_delta_put", "10_delta_call", "10_delta_put"]
    rows = {r["tenor"]: r for r in market_df.iter_rows(named=True)}
    for res in calibrate_sabr_surface_from_df(market_df, beta=0.5):
        if res["tenor"] == "on":
            continue
        quoted = np.array([rows[res["tenor"]][f"surface_{c}_vol"] for c in vol_cols])
        np.testing.assert_allclose(res["model_ivs"] * 100, quoted, atol=0.05, err_msg=res["tenor"])
