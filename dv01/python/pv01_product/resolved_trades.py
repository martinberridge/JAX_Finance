"""Resolved calibration and test trades."""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import List, Optional

from pv01_basics.calendar import HolidayCalendar, gblo
from pv01_basics.day_count import act_365f_year_fraction
from pv01_basics.schedule import SchedulePeriod, generate_schedule
from pv01_basics.tenor import Tenor
from pv01_loader.models import CurveNode, MarketData
from pv01_product.conventions import (
    CONVENTIONS,
    BuySell,
    ConventionInfo,
    spot_date,
)


class TradeType(Enum):
    OIS = "OIS"
    FIX = "FIX"
    IRS = "IRS"
    SWAP = "SWAP"


@dataclass
class RatePeriod:
    start: date
    end: date
    payment: date
    notional: float
    fixed_rate: Optional[float] = None
    is_fixed: bool = False
    is_overnight: bool = False


@dataclass
class ResolvedTrade:
    trade_type: TradeType
    convention: ConventionInfo
    periods: List[RatePeriod]
    notional: float
    buy_sell: BuySell
    market_rate: float
    node_date: date
    label: str
    curve_name: str
    fixed_rate: Optional[float] = None


@dataclass
class ResolvedSwapTrade:
    fixed_leg: List[RatePeriod]
    float_leg: List[RatePeriod]
    notional: float
    buy_sell: BuySell
    fixed_rate: float
    is_overnight_float: bool = False
    convention: Optional[ConventionInfo] = None


def _build_swap_periods(
    start: date,
    end: date,
    notional: float,
    fixed_rate: float,
    convention: ConventionInfo,
    buy_sell: BuySell,
) -> tuple[List[RatePeriod], List[RatePeriod]]:
    cal = convention.calendar
    fixed_periods = []
    float_periods = []
    for sp in generate_schedule(start, end, convention.fixed_frequency_months, cal):
        fixed_periods.append(RatePeriod(
            start=sp.start, end=sp.end, payment=sp.payment,
            notional=notional, fixed_rate=fixed_rate, is_fixed=True,
        ))
        float_periods.append(RatePeriod(
            start=sp.start, end=sp.end, payment=sp.payment,
            notional=notional, is_fixed=False,
        ))
    return fixed_periods, float_periods


def _build_ois_periods(
    start: date,
    end: date,
    notional: float,
    fixed_rate: float,
    convention: ConventionInfo,
) -> tuple[List[RatePeriod], List[RatePeriod]]:
    cal = convention.calendar
    fixed_periods = []
    float_periods = []
    for sp in generate_schedule(start, end, convention.ois_frequency_months, cal):
        fixed_periods.append(RatePeriod(
            start=sp.start, end=sp.end, payment=sp.payment,
            notional=notional, fixed_rate=fixed_rate, is_fixed=True,
        ))
        float_periods.append(RatePeriod(
            start=sp.start, end=sp.end, payment=sp.payment,
            notional=notional, is_fixed=False, is_overnight=True,
        ))
    return fixed_periods, float_periods


def resolve_calibration_trade(
    node: CurveNode,
    valuation_date: date,
    market_data: MarketData,
    notional: float = 1_000_000.0,
) -> ResolvedTrade:
    convention = CONVENTIONS[node.convention]
    market_rate = market_data.get_quote(node.quote_id)
    trade_date = valuation_date
    spot = spot_date(trade_date, convention)

    if node.node_type == "OIS":
        tenor = Tenor.parse(node.time)
        start = spot
        end = tenor.add_to(start)
        fixed_leg, float_leg = _build_ois_periods(start, end, notional, market_rate, convention)
        return ResolvedTrade(
            trade_type=TradeType.OIS,
            convention=convention,
            periods=float_leg,
            notional=notional,
            buy_sell=BuySell.BUY,
            market_rate=market_rate,
            node_date=end,
            label=node.label,
            curve_name=node.curve_name,
            fixed_rate=market_rate,
        )

    if node.node_type == "FIX":
        start = spot
        end = Tenor.parse("6M").add_to(start)
        return ResolvedTrade(
            trade_type=TradeType.FIX,
            convention=convention,
            periods=[RatePeriod(start, end, end, notional, is_fixed=False)],
            notional=notional,
            buy_sell=BuySell.BUY,
            market_rate=market_rate,
            node_date=end,
            label=node.label,
            curve_name=node.curve_name,
        )

    if node.node_type == "IRS":
        tenor = Tenor.parse(node.time)
        start = spot
        end = tenor.add_to(start)
        fixed_leg, float_leg = _build_swap_periods(start, end, notional, market_rate, convention, BuySell.BUY)
        return ResolvedTrade(
            trade_type=TradeType.IRS,
            convention=convention,
            periods=float_leg,
            notional=notional,
            buy_sell=BuySell.BUY,
            market_rate=market_rate,
            node_date=end,
            label=node.label,
            curve_name=node.curve_name,
            fixed_rate=market_rate,
        )

    raise ValueError(f"Unsupported node type: {node.node_type}")


def create_test_swap(
    valuation_date: date,
    period_to_start_months: int,
    tenor_years: int,
    buy_sell: BuySell,
    notional: float,
    fixed_rate: float,
) -> ResolvedSwapTrade:
    """8Y GBP fixed vs LIBOR 6M swap, 6M forward start."""
    convention = CONVENTIONS["GBP-FIXED-6M-LIBOR-6M"]
    spot = spot_date(valuation_date, convention)
    start = Tenor.parse(f"{period_to_start_months}M").add_to(spot)
    end = Tenor.parse(f"{tenor_years}Y").add_to(start)
    fixed_leg, float_leg = _build_swap_periods(start, end, notional, fixed_rate, convention, buy_sell)
    return ResolvedSwapTrade(
        fixed_leg=fixed_leg,
        float_leg=float_leg,
        notional=notional,
        buy_sell=buy_sell,
        fixed_rate=fixed_rate,
        convention=convention,
    )
