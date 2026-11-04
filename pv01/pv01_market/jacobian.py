"""Jacobian calibration matrix."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from pv01_market.curve_types import CurveParameterSize


@dataclass
class JacobianCalibrationMatrix:
    """Inverse Jacobian: d(curve params) / d(market quotes)."""

    order: List[CurveParameterSize]
    jacobian_matrix: np.ndarray  # shape (n_params_curve, n_total_params)

    def split_values(self, market_quote_sens: np.ndarray) -> Dict[str, np.ndarray]:
        """Split sensitivity vector across curves in calibration order."""
        result: Dict[str, np.ndarray] = {}
        offset = 0
        for cps in self.order:
            count = cps.parameter_count
            result[cps.name] = np.asarray(market_quote_sens[offset:offset + count])
            offset += count
        return result

    @staticmethod
    def of(order: List[CurveParameterSize], matrix: np.ndarray) -> "JacobianCalibrationMatrix":
        return JacobianCalibrationMatrix(order=order, jacobian_matrix=matrix)
