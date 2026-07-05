"""Instrument pricers for calibration trades."""

from typing import List

import numpy as np

from pv01_basics.day_count import act_365f_year_fraction
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import RatePeriod, ResolvedTrade, TradeType
from pv01_pricer.rates_provider import ImmutableRatesProvider


def _sign(buy_sell: BuySell) -> float:
    return 1.0 if buy_sell == BuySell.BUY else -1.0


def _pv_fixed_leg(periods: List[RatePeriod], provider: ImmutableRatesProvider, fixed_rate: float) -> float:
    pv = 0.0
    for p in periods:
        accrual = act_365f_year_fraction(p.start, p.end)
        df = provider.discount_factor(p.payment)
        pv += p.notional * fixed_rate * accrual * df
    return pv


def _pv_float_leg_ibor(periods: List[RatePeriod], provider: ImmutableRatesProvider) -> float:
    pv = 0.0
    for p in periods:
        fwd = provider.forward_rate(p.start, p.end)
        accrual = act_365f_year_fraction(p.start, p.end)
        df = provider.discount_factor(p.payment)
        pv += p.notional * fwd * accrual * df
    return pv


def _pv_float_leg_overnight(periods: List[RatePeriod], provider: ImmutableRatesProvider) -> float:
    pv = 0.0
    for p in periods:
        # Compounded overnight approximation using period average rate
        rate = provider.overnight_rate(p.start)
        accrual = act_365f_year_fraction(p.start, p.end)
        df = provider.discount_factor(p.payment)
        compounded = (1.0 + rate * accrual) - 1.0
        pv += p.notional * compounded * df
    return pv


def _pv_fixing_deposit(trade: ResolvedTrade, provider: ImmutableRatesProvider) -> float:
    p = trade.periods[0]
    rate = provider.forward_rate(p.start, p.end)
    accrual = act_365f_year_fraction(p.start, p.end)
    df = provider.discount_factor(p.payment)
    return trade.notional * rate * accrual * df * _sign(trade.buy_sell)


def present_value(trade: ResolvedTrade, provider: ImmutableRatesProvider) -> float:
    if trade.trade_type == TradeType.FIX:
        return _pv_fixing_deposit(trade, provider)

    # Build fixed leg for swap calibration trades
    if trade.trade_type == TradeType.OIS:
        fixed_periods = [
            RatePeriod(p.start, p.end, p.payment, p.notional, fixed_rate=trade.fixed_rate, is_fixed=True)
            for p in trade.periods
        ]
        pv_fixed = _pv_fixed_leg(fixed_periods, provider, trade.fixed_rate)
        pv_float = _pv_float_leg_overnight(trade.periods, provider)
    else:  # IRS
        fixed_periods = [
            RatePeriod(p.start, p.end, p.payment, p.notional, fixed_rate=trade.fixed_rate, is_fixed=True)
            for p in trade.periods
        ]
        pv_fixed = _pv_fixed_leg(fixed_periods, provider, trade.fixed_rate)
        pv_float = _pv_float_leg_ibor(trade.periods, provider)

    return (pv_float - pv_fixed) * _sign(trade.buy_sell)


def pvbp(trade: ResolvedTrade, provider: ImmutableRatesProvider) -> float:
    """PV of a 1bp bump in fixed rate."""
    if trade.trade_type == TradeType.FIX:
        p = trade.periods[0]
        accrual = act_365f_year_fraction(p.start, p.end)
        df = provider.discount_factor(p.payment)
        return trade.notional * accrual * df * 1e-4

    total_accrual = sum(act_365f_year_fraction(p.start, p.end) for p in trade.periods)
    avg_payment = trade.periods[len(trade.periods) // 2].payment
    df = provider.discount_factor(avg_payment)
    return trade.notional * total_accrual * df * 1e-4 / max(len(trade.periods), 1) * len(trade.periods)


def _fixed_leg_pvbp(periods: List[RatePeriod], provider: ImmutableRatesProvider) -> float:
    bp = 0.0
    for p in periods:
        accrual = act_365f_year_fraction(p.start, p.end)
        df = provider.discount_factor(p.payment)
        bp += p.notional * accrual * df
    return bp


def par_spread(trade: ResolvedTrade, provider: ImmutableRatesProvider) -> float:
    """Par spread for calibration — should be zero at calibrated curve."""
    if trade.trade_type == TradeType.FIX:
        p = trade.periods[0]
        implied = provider.forward_rate(p.start, p.end)
        return implied - trade.market_rate

    fixed_periods = [
        RatePeriod(p.start, p.end, p.payment, p.notional, fixed_rate=trade.fixed_rate, is_fixed=True)
        for p in trade.periods
    ]
    pv = present_value(trade, provider)
    pvbp = _fixed_leg_pvbp(fixed_periods, provider)
    if abs(pvbp) < 1e-12:
        return pv
    return -pv / pvbp
