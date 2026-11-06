# Derived from OpenGamma Strata (https://github.com/OpenGamma/Strata):
#   ImmutableRatesProvider, ZeroRateDiscountFactors
# Copyright (C) 2014 - present by OpenGamma Inc. and the OpenGamma group of companies
# Copyright (C) 2015 - present by OpenGamma Inc. and the OpenGamma group of companies
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.txt.
# Changed by Martin Berridge, 2026: translated from Java to Python and reduced to GBP
#   discount and LIBOR curves.

"""Multi-curve rates provider."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from pv01_basics.day_count import act_365f_between, act_365f_year_fraction
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
    sensitivity_type: str  # ZeroRate (discount curve z), IborRate (LIBOR curve z_L)
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

    def _libor_pseudo_df(self, yf: float) -> float:
        """Pseudo-discount factor P_L(t) = exp(-t * z_L(t)) on the LIBOR zero rates."""
        if yf <= EFFECTIVE_ZERO:
            return 1.0
        return np.exp(-yf * self.libor_curve.y_value(yf))

    def forward_rate(self, start: date, end: date) -> float:
        """LIBOR-6M simple forward F = (P_L(s) / P_L(e) - 1) / tau, tau = act/365F(s, e)."""
        tau = act_365f_year_fraction(start, end)
        if tau <= EFFECTIVE_ZERO:
            return self.libor_curve.y_value(self.year_fraction(end))
        ratio = self._libor_pseudo_df(self.year_fraction(start)) / self._libor_pseudo_df(self.year_fraction(end))
        return (ratio - 1.0) / tau

    def zero_rate_point_sensitivity(self, payment_date: date, currency: str = "GBP") -> PointSensitivity:
        yf = self.year_fraction(payment_date)
        df = self.discount_factor(payment_date)
        if yf <= EFFECTIVE_ZERO:
            return PointSensitivity("ZeroRate", yf, currency, 0.0)
        return PointSensitivity("ZeroRate", yf, currency, -df * yf)

    def ibor_point_sensitivities(
        self, start: date, end: date, dpv_dfwd: float, currency: str = "GBP"
    ) -> PointSensitivities:
        """Chain dPV/dF through F's dependence on z_L at the period start and end.

        dF/dz_L(t_e) = t_e * P_L(s) / P_L(e) / tau, dF/dz_L(t_s) = -t_s * P_L(s) / P_L(e) / tau.
        """
        tau = act_365f_year_fraction(start, end)
        yf_start = self.year_fraction(start)
        yf_end = self.year_fraction(end)
        if tau <= EFFECTIVE_ZERO:
            return PointSensitivities([PointSensitivity("IborRate", yf_end, currency, dpv_dfwd, index="GBP-LIBOR-6M")])
        ratio = self._libor_pseudo_df(yf_start) / self._libor_pseudo_df(yf_end)
        points = []
        if yf_start > EFFECTIVE_ZERO:
            points.append(
                PointSensitivity("IborRate", yf_start, currency, dpv_dfwd * (-yf_start * ratio / tau), index="GBP-LIBOR-6M")
            )
        if yf_end > EFFECTIVE_ZERO:
            points.append(
                PointSensitivity("IborRate", yf_end, currency, dpv_dfwd * (yf_end * ratio / tau), index="GBP-LIBOR-6M")
            )
        return PointSensitivities(points)

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
