"""Data models for calibration configuration."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from pv01_market.quote_id import QuoteId


@dataclass(frozen=True)
class CurveGroupEntry:
    group_name: str
    curve_type: str  # Discount or Forward
    reference: str
    curve_name: str


@dataclass(frozen=True)
class CurveNode:
    curve_name: str
    label: str
    quote_id: QuoteId
    node_type: str  # OIS, FIX, IRS
    convention: str
    time: Optional[str] = None


@dataclass(frozen=True)
class CurveSettings:
    curve_name: str
    value_type: str
    day_count: str
    interpolator: str
    left_extrapolator: str
    right_extrapolator: str


@dataclass
class CurveDefinition:
    name: str
    settings: CurveSettings
    nodes: List[CurveNode] = field(default_factory=list)


@dataclass
class RatesCurveGroupDefinition:
    group_name: str
    entries: List[CurveGroupEntry]
    curves: Dict[str, CurveDefinition]


@dataclass
class MarketData:
    valuation_date: date
    quotes: Dict[QuoteId, float]

    def get_quote(self, quote_id: QuoteId) -> float:
        if quote_id not in self.quotes:
            raise KeyError(f"Missing quote: {quote_id}")
        return self.quotes[quote_id]
