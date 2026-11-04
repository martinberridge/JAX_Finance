"""Bump-and-recalibrate reference PV01."""

from datetime import date
from typing import Dict

from pv01_loader.models import MarketData, RatesCurveGroupDefinition
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator
from pv01_pricer.swap_pricer import DiscountingSwapTradePricer
from pv01_product.resolved_trades import ResolvedSwapTrade


def bumped_market_data(market_data: MarketData, bumps: Dict) -> MarketData:
    """Copy of market_data with each quote_id in bumps shifted by its amount."""
    quotes = dict(market_data.quotes)
    for quote_id, shift in bumps.items():
        quotes[quote_id] = quotes[quote_id] + shift
    return MarketData(valuation_date=market_data.valuation_date, quotes=quotes)


def _recalibrated_pv(
    config: RatesCurveGroupDefinition,
    market_data: MarketData,
    swap: ResolvedSwapTrade,
    valuation_date: date,
) -> float:
    provider = RatesCurveCalibrator.standard().calibrate(config, market_data, valuation_date)
    return DiscountingSwapTradePricer.DEFAULT.present_value(swap, provider)["GBP"]


def bump_pv01(
    config: RatesCurveGroupDefinition,
    market_data: MarketData,
    swap: ResolvedSwapTrade,
    valuation_date: date,
    h: float = 1e-4,
) -> Dict[str, float]:
    """Central bump-and-recalibrate PV01 per curve node, in GBP per bp, keyed by node label.

    Only quotes referenced by curve nodes are bumped. Each node quote is shifted by +h and -h,
    the curves are recalibrated and the swap repriced; the bucket is (PV_up - PV_down) / 2.
    """
    result: Dict[str, float] = {}
    for curve_def in config.curves.values():
        for node in curve_def.nodes:
            pv_up = _recalibrated_pv(config, bumped_market_data(market_data, {node.quote_id: h}), swap, valuation_date)
            pv_down = _recalibrated_pv(config, bumped_market_data(market_data, {node.quote_id: -h}), swap, valuation_date)
            result[node.label] = (pv_up - pv_down) / 2.0
    return result
