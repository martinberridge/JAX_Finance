from __future__ import annotations

from datetime import date

import numpy as np

from pv01_basics.day_count import act_365f_between, act_365f_year_fraction
from pv01_loader.models import MarketData, RatesCurveGroupDefinition
from pv01_market.interpolated_nodal_curve import InterpolatedNodalCurve
from pv01_pricer.rates_provider import ImmutableRatesProvider
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import (
    ResolvedSwapTrade,
    ResolvedTrade,
    resolve_calibration_trade,
)
from pv01_jax.arrays import (
    CalibBlock,
    CalibrationProblem,
    CurveSpec,
    FloatLegArrays,
    LegArrays,
    SwapArrays,
)


def _np_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _np_int_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def resolve_swap(swap: ResolvedSwapTrade, valuation_date: date) -> SwapArrays:
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


def build_calibration_problem(
    group_defn: RatesCurveGroupDefinition,
    market_data: MarketData,
    valuation_date: date,
) -> CalibrationProblem:
    """Resolve curve calibration instruments and pack their period data into blocks."""
    curve_names = sorted(group_defn.curves)
    curve_names.sort(key=lambda name: (0 if "DSCON" in name else 1, name))

    q0_values: list[float] = []
    curve_specs: list[CurveSpec] = []
    trades_by_kind: dict[str, list[tuple[ResolvedTrade, int]]] = {
        "FIX": [],
        "OIS": [],
        "IRS": [],
    }

    for curve_name in curve_names:
        curve_definition = group_defn.curves[curve_name]
        start_idx = len(q0_values)
        curve_x_nodes = []

        for node in curve_definition.nodes:
            trade = resolve_calibration_trade(node, valuation_date, market_data)
            quote_idx = len(q0_values)
            q0_values.append(market_data.get_quote(node.quote_id))
            curve_x_nodes.append(act_365f_between(valuation_date, trade.node_date))
            try:
                trades_by_kind[trade.trade_type.value].append((trade, quote_idx))
            except KeyError as exc:
                raise ValueError(f"Unsupported calibration trade type: {trade.trade_type.value}") from exc

        curve_specs.append(
            CurveSpec(
                name=curve_name,
                param_slice=(start_idx, len(q0_values)),
                labels=tuple(node.label for node in curve_definition.nodes),
                interpolator=curve_definition.settings.interpolator,
                x_nodes=np.asarray(curve_x_nodes, dtype=np.float64),
            )
        )

    blocks = tuple(
        _build_calibration_block(kind, kind_trades, valuation_date)
        for kind, kind_trades in trades_by_kind.items()
        if kind_trades
    )
    return CalibrationProblem(
        name=group_defn.group_name,
        curves=tuple(curve_specs),
        blocks=blocks,
        q0=np.asarray(q0_values, dtype=np.float64),
    )


def _build_calibration_block(
    trade_type: str,
    trades_with_quote_indices: list[tuple[ResolvedTrade, int]],
    valuation_date: date,
) -> CalibBlock:
    float_periods = []
    fixed_periods = []
    trade_ids = []
    t_pay = []
    tau = []
    t_fix_start = []
    t_fix_end = []
    tau_fix = []
    signs = []
    quote_indices = []

    for trade_id, (trade, quote_idx) in enumerate(trades_with_quote_indices):
        quote_indices.append(quote_idx)
        sign = 1.0 if trade.buy_sell == BuySell.BUY else -1.0
        for period in trade.periods:
            float_periods.append(period)
            trade_ids.append(trade_id)
            t_pay.append(act_365f_between(valuation_date, period.payment))
            tau.append(act_365f_year_fraction(period.start, period.end))
            signs.append(sign)
            if trade_type != "FIX":
                fixed_periods.append(period)
                t_fix_start.append(act_365f_between(valuation_date, period.start))
                t_fix_end.append(act_365f_between(valuation_date, period.end))
                tau_fix.append(act_365f_year_fraction(period.start, period.end))

    float_leg = FloatLegArrays(
        start=np.asarray(
            [act_365f_between(valuation_date, period.start) for period in float_periods],
            dtype=np.float64,
        ),
        end=np.asarray(
            [act_365f_between(valuation_date, period.end) for period in float_periods],
            dtype=np.float64,
        ),
        rate=np.zeros(len(float_periods), dtype=np.float64),
        notional=np.asarray([period.notional for period in float_periods], dtype=np.float64),
    )

    fixed_leg = None
    if trade_type != "FIX":
        fixed_leg = LegArrays(
            start=np.asarray(t_fix_start, dtype=np.float64),
            end=np.asarray(t_fix_end, dtype=np.float64),
            rate=np.zeros(len(fixed_periods), dtype=np.float64),
            notional=np.asarray([period.notional for period in fixed_periods], dtype=np.float64),
        )

    return CalibBlock(
        trade_type=trade_type,
        quote_idx=np.asarray(quote_indices, dtype=np.int64),
        trade_id=np.asarray(trade_ids, dtype=np.int64),
        float_leg=float_leg,
        fixed_leg=fixed_leg,
        t_pay=np.asarray(t_pay, dtype=np.float64),
        tau=np.asarray(tau, dtype=np.float64),
        t_fix_start=np.asarray(t_fix_start, dtype=np.float64),
        t_fix_end=np.asarray(t_fix_end, dtype=np.float64),
        tau_fix=np.asarray(tau_fix, dtype=np.float64),
        sign=np.asarray(signs, dtype=np.float64),
    )


def par_spread_from_calibration_block(
    block: CalibBlock,
    trade_id: int,
    q: np.ndarray,
    provider: ImmutableRatesProvider,
) -> float:
    """Evaluate one trade's calibration par spread using packed arrays and quote indices."""
    period_indices = np.flatnonzero(np.asarray(block.trade_id) == trade_id)
    quote_idx = int(np.asarray(block.quote_idx)[trade_id])
    market_rate = float(q[quote_idx])
    sign = float(np.asarray(block.sign)[period_indices[0]])
    pv_float = 0.0
    pv_fixed = 0.0
    pvbp = 0.0

    for period_idx in period_indices:
        tau_period = float(np.asarray(block.tau)[period_idx])
        pay_time = float(np.asarray(block.t_pay)[period_idx])
        discount_factor = _discount_factor_at_time(provider.discount_curve, pay_time)
        start_time = float(np.asarray(block.float_leg.start)[period_idx])
        end_time = float(np.asarray(block.float_leg.end)[period_idx])
        notional = float(np.asarray(block.float_leg.notional)[period_idx])

        if block.trade_type == "FIX":
            forward = _forward_rate_at_times(
                provider.libor_curve, start_time, end_time, tau_period
            )
            return forward - market_rate

        if block.trade_type == "OIS":
            df_start = _discount_factor_at_time(provider.discount_curve, start_time)
            df_end = _discount_factor_at_time(provider.discount_curve, end_time)
            pv_float += notional * (df_start / df_end - 1.0) * discount_factor
        else:
            forward = _forward_rate_at_times(
                provider.libor_curve, start_time, end_time, tau_period
            )
            pv_float += notional * forward * tau_period * discount_factor

        fixed_notional = float(np.asarray(block.fixed_leg.notional)[period_idx])
        fixed_tau = float(np.asarray(block.tau_fix)[period_idx])
        pv_fixed += fixed_notional * market_rate * fixed_tau * discount_factor
        pvbp += fixed_notional * fixed_tau * discount_factor

    if abs(pvbp) < 1e-12:
        return (pv_float - pv_fixed) * sign
    return sign * ((pv_float - pv_fixed) * sign) / pvbp


def _discount_factor_at_time(curve: InterpolatedNodalCurve, time: float) -> float:
    if time <= 1e-10:
        return 1.0
    return float(np.exp(-time * curve.y_value(time)))


def _forward_rate_at_times(
    curve: InterpolatedNodalCurve, start: float, end: float, accrual: float
) -> float:
    if accrual <= 1e-10:
        return float(curve.y_value(end))
    start_df = _discount_factor_at_time(curve, start)
    end_df = _discount_factor_at_time(curve, end)
    return (start_df / end_df - 1.0) / accrual


__all__ = [
    "resolve_swap",
    "pv_from_arrays",
    "build_calibration_problem",
    "par_spread_from_calibration_block",
]
