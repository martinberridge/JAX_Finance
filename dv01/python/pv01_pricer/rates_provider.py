"""Multi-curve rates provider."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from pv01_basics.day_count import act_365f_between
from pv01_market.curve_types import CurveParameterSize
from pv01_market.interpolated_nodal_curve import InterpolatedNodalCurve


EFFECTIVE_ZERO = 1e-10


@dataclass
class CurrencyParameterSensitivity:
    market_data_name: str
    currency: str
    sensitivity: np.ndarray
    labels: List[str]


@dataclass
class CurrencyParameterSensitivities:
    sensitivities: List[CurrencyParameterSensitivity] = field(default_factory=list)

    def combined_with(self, other: "CurrencyParameterSensitivities") -> "CurrencyParameterSensitivities":
        return CurrencyParameterSensitivities(self.sensitivities + other.sensitivities)

    @staticmethod
    def empty() -> "CurrencyParameterSensitivities":
        return CurrencyParameterSensitivities([])


@dataclass
class PointSensitivity:
    sensitivity_type: str  # ZeroRate, IborRate
    year_fraction: float
    currency: str
    value: float
    index: Optional[str] = None


@dataclass
class PointSensitivities:
    sensitivities: List[PointSensitivity] = field(default_factory=list)

    def combined_with(self, other: "PointSensitivities") -> "PointSensitivities":
        return PointSensitivities(self.sensitivities + other.sensitivities)


class ImmutableRatesProvider:
    """GBP multi-curve provider with discount and forward curves."""

    def __init__(
        self,
        valuation_date: date,
        discount_curve: InterpolatedNodalCurve,
        libor_curve: InterpolatedNodalCurve,
        curve_order: Optional[List[CurveParameterSize]] = None,
    ):
        self.valuation_date = valuation_date
        self.discount_curve = discount_curve
        self.libor_curve = libor_curve
        self.curves: Dict[str, InterpolatedNodalCurve] = {
            discount_curve.name: discount_curve,
            libor_curve.name: libor_curve,
        }
        self.curve_order = curve_order or [
            CurveParameterSize(discount_curve.name, discount_curve.parameter_count()),
            CurveParameterSize(libor_curve.name, libor_curve.parameter_count()),
        ]

    def year_fraction(self, d: date) -> float:
        return act_365f_between(self.valuation_date, d)

    def discount_factor(self, payment_date: date, curve: Optional[InterpolatedNodalCurve] = None) -> float:
        c = curve or self.discount_curve
        yf = self.year_fraction(payment_date)
        if yf <= EFFECTIVE_ZERO:
            return 1.0
        z = c.y_value(yf)
        return np.exp(-yf * z)

    def forward_rate(self, start: date, end: date) -> float:
        """LIBOR-6M forward from forward curve zero rates."""
        yf = self.year_fraction(end)
        return self.libor_curve.y_value(yf)

    def overnight_rate(self, period_start: date) -> float:
        yf = self.year_fraction(period_start)
        return self.discount_curve.y_value(yf)

    def zero_rate_point_sensitivity(self, payment_date: date, currency: str = "GBP") -> PointSensitivity:
        yf = self.year_fraction(payment_date)
        df = self.discount_factor(payment_date)
        if yf <= EFFECTIVE_ZERO:
            return PointSensitivity("ZeroRate", yf, currency, 0.0)
        return PointSensitivity("ZeroRate", yf, currency, -df * yf)

    def ibor_point_sensitivity(self, end: date, currency: str = "GBP") -> PointSensitivity:
        yf = self.year_fraction(end)
        return PointSensitivity("IborRate", yf, currency, 1.0, index="GBP-LIBOR-6M")

    def parameter_sensitivity(self, point_sens: PointSensitivities) -> CurrencyParameterSensitivities:
        result = CurrencyParameterSensitivities.empty()
        for pt in point_sens.sensitivities:
            if pt.sensitivity_type == "ZeroRate":
                curve = self.discount_curve
                unit = curve.y_value_parameter_sensitivity(pt.year_fraction)
                cps = CurrencyParameterSensitivity(
                    market_data_name=curve.name,
                    currency=pt.currency,
                    sensitivity=unit * pt.value,
                    labels=curve.node_labels,
                )
                result = result.combined_with(CurrencyParameterSensitivities([cps]))
            elif pt.sensitivity_type == "IborRate":
                curve = self.libor_curve
                unit = curve.y_value_parameter_sensitivity(pt.year_fraction)
                cps = CurrencyParameterSensitivity(
                    market_data_name=curve.name,
                    currency=pt.currency,
                    sensitivity=unit * pt.value,
                    labels=curve.node_labels,
                )
                result = result.combined_with(CurrencyParameterSensitivities([cps]))
        return result

    def find_curve(self, name: str) -> InterpolatedNodalCurve:
        return self.curves[name]
