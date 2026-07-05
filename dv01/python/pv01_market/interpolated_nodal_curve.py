"""Interpolated nodal zero-rate curve."""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import numpy as np

from pv01_basics.day_count import act_365f_between
from pv01_market.interpolators import BoundInterpolator, create_interpolator
from pv01_market.jacobian import JacobianCalibrationMatrix


@dataclass
class InterpolatedNodalCurve:
    name: str
    valuation_date: date
    node_dates: List[date]
    node_labels: List[str]
    y_values: np.ndarray
    interpolator_name: str
    jacobian: Optional[JacobianCalibrationMatrix] = None

    _bound: BoundInterpolator = field(init=False, repr=False)

    def __post_init__(self):
        x = np.array([act_365f_between(self.valuation_date, d) for d in self.node_dates])
        self._bound = create_interpolator(self.interpolator_name, x, self.y_values)

    @property
    def x_values(self) -> np.ndarray:
        return np.array([act_365f_between(self.valuation_date, d) for d in self.node_dates])

    def y_value(self, year_fraction: float) -> float:
        return self._bound.interpolate(year_fraction)

    def first_derivative(self, year_fraction: float) -> float:
        return self._bound.first_derivative(year_fraction)

    def y_value_parameter_sensitivity(self, year_fraction: float) -> np.ndarray:
        return self._bound.parameter_sensitivity(year_fraction)

    def with_y_values(self, y_values: np.ndarray) -> "InterpolatedNodalCurve":
        return InterpolatedNodalCurve(
            name=self.name,
            valuation_date=self.valuation_date,
            node_dates=self.node_dates,
            node_labels=self.node_labels,
            y_values=np.asarray(y_values, dtype=float),
            interpolator_name=self.interpolator_name,
            jacobian=self.jacobian,
        )

    def parameter_count(self) -> int:
        return len(self.y_values)
