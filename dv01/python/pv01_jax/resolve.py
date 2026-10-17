from __future__ import annotations

import numpy as np

from pv01_basics.day_count import act_365f_between, act_365f_year_fraction
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import ResolvedSwapTrade
from pv01_jax.arrays import FloatLegArrays, LegArrays, SwapArrays


def _np_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _np_int_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def resolve_swap(swap: ResolvedSwapTrade, valuation_date) -> SwapArrays:
    """Convert a resolved swap trade into JAX-friendly arrays used for PV evaluation."""
    fixed_periods = swap.fixed_leg
    float_periods = swap.float_leg

    fixed_start = _np_float_array([act_365f_between(valuation_date, p.start) for p in fixed_periods])
    fixed_end = _np_float_array([act_365f_between(valuation_date, p.end) for p in fixed_periods])
    float_start = _np_float_array([act_365f_between(valuation_date, p.start) for p in float_periods])
    float_end = _np_float_array([act_365f_between(valuation_date, p.end) for p in float_periods])
    t_pay = _np_float_array([act_365f_between(valuation_date, p.payment) for p in float_periods])
    tau = _np_float_array([act_365f_year_fraction(p.start, p.end) for p in float_periods])
    tau_fix = _np_float_array([act_365f_year_fraction(p.start, p.end) for p in fixed_periods])
    t_fix_start = fixed_start.copy()
    t_fix_end = fixed_end.copy()

    sign = np.full(len(fixed_periods), 1.0 if swap.buy_sell == BuySell.BUY else -1.0, dtype=np.float64)
    fixed_rate = _np_float_array(swap.fixed_rate)
    fixed_notional = _np_float_array([p.notional for p in fixed_periods])
    float_notional = _np_float_array([p.notional for p in float_periods])

    fixed_leg = LegArrays(
        start=fixed_start,
        end=fixed_end,
        rate=np.full(len(fixed_periods), float(swap.fixed_rate), dtype=np.float64),
        notional=fixed_notional,
    )
    float_leg = FloatLegArrays(
        start=float_start,
        end=float_end,
        rate=np.zeros(len(float_periods), dtype=np.float64),
        notional=float_notional,
    )
    return SwapArrays(
        float_leg=float_leg,
        spread=np.zeros(len(float_periods), dtype=np.float64),
        fixed_leg=fixed_leg,
        t_pay=t_pay,
        trade_id=_np_int_array(np.zeros(len(float_periods), dtype=np.int64)),
        sign=sign,
        fixed_rate=fixed_rate,
        tau=tau,
        t_fix_start=t_fix_start,
        t_fix_end=t_fix_end,
        tau_fix=tau_fix,
    )


def _discount_factor(provider, t_pay: float) -> float:
    z = provider.discount_curve.y_value(t_pay)
    return float(np.exp(-t_pay * z))


def _forward_rate(provider, t_start: float, t_end: float, tau: float) -> float:
    z_start = provider.libor_curve.y_value(t_start)
    z_end = provider.libor_curve.y_value(t_end)
    base = np.exp(-t_start * z_start) / np.exp(-t_end * z_end)
    return float((base - 1.0) / tau)


def pv_from_arrays(arrays: SwapArrays, provider) -> float:
    """Reference NumPy PV evaluator mirroring the production pricer for a resolved swap."""
    pv_fixed = 0.0
    if arrays.fixed_leg is not None:
        for i in range(len(arrays.fixed_leg.start)):
            tau_fix = float(arrays.tau_fix[i] if arrays.tau_fix.size else arrays.fixed_leg.end[i] - arrays.fixed_leg.start[i])
            t_pay = float(arrays.t_pay[i])
            df = _discount_factor(provider, t_pay)
            pv_fixed += float(arrays.fixed_leg.notional[i]) * float(arrays.fixed_leg.rate[i]) * tau_fix * df

    pv_float = 0.0
    for i in range(len(arrays.float_leg.start)):
        tau = float(arrays.tau[i] if arrays.tau.size else arrays.float_leg.end[i] - arrays.float_leg.start[i])
        t_pay = float(arrays.t_pay[i])
        t_start = float(arrays.float_leg.start[i])
        t_end = float(arrays.float_leg.end[i])
        df = _discount_factor(provider, t_pay)
        fwd = _forward_rate(provider, t_start, t_end, tau)
        pv_float += float(arrays.float_leg.notional[i]) * fwd * tau * df

    sign = float(arrays.sign[0]) if np.asarray(arrays.sign).size else 1.0
    return float((pv_float - pv_fixed) * sign)


__all__ = ["resolve_swap", "pv_from_arrays"]
