"""RatesCurveCalibrator — Newton root-finding on curve zero rates."""

import logging
from datetime import date
from typing import List, Tuple

import numpy as np

from pv01_loader.models import CurveDefinition, MarketData, RatesCurveGroupDefinition
from pv01_market.curve_types import CurveParameterSize
from pv01_market.interpolated_nodal_curve import InterpolatedNodalCurve
from pv01_market.jacobian import JacobianCalibrationMatrix
from pv01_product.resolved_trades import ResolvedTrade, resolve_calibration_trade
from pv01_pricer.instrument_pricers import par_spread
from pv01_pricer.rates_provider import ImmutableRatesProvider

logger = logging.getLogger(__name__)


class RatesCurveCalibrator:
    def __init__(self, tol: float = 1e-14, max_iter: int = 50):
        self.tol = tol
        self.max_iter = max_iter

    @staticmethod
    def standard() -> "RatesCurveCalibrator":
        return RatesCurveCalibrator()

    def calibrate(
        self,
        group_defn: RatesCurveGroupDefinition,
        market_data: MarketData,
        valuation_date: date,
    ) -> ImmutableRatesProvider:
        trades: List[ResolvedTrade] = []
        curve_nodes_info: List[Tuple[str, CurveDefinition, int]] = []
        initial_guesses: List[float] = []

        # Order: discount curve first, then forward
        curve_names = sorted(group_defn.curves.keys())
        # Ensure OIS discount before LIBOR
        curve_names = sorted(curve_names, key=lambda n: (0 if "DSCON" in n else 1, n))

        for curve_name in curve_names:
            curve_def = group_defn.curves[curve_name]
            curve_nodes_info.append((curve_name, curve_def, len(curve_def.nodes)))
            for node in curve_def.nodes:
                trade = resolve_calibration_trade(node, valuation_date, market_data)
                trades.append(trade)
                initial_guesses.append(market_data.get_quote(node.quote_id))

        order = [
            CurveParameterSize(name, group_defn.curves[name].settings.curve_name and len(group_defn.curves[name].nodes))
            for name in curve_names
        ]
        # Fix order parameter counts
        order = [CurveParameterSize(cd.name, len(cd.nodes)) for _, cd, _ in
                 [(n, group_defn.curves[n], 0) for n in curve_names]]

        def build_provider(params: np.ndarray) -> ImmutableRatesProvider:
            idx = 0
            curves = {}
            node_dates_map = {}
            node_labels_map = {}
            interp_map = {}

            for curve_name in curve_names:
                cd = group_defn.curves[curve_name]
                n = len(cd.nodes)
                y_vals = params[idx:idx + n]
                idx += n
                dates = [t.node_date for t in trades if t.curve_name == cd.name]
                labels = [t.label for t in trades if t.curve_name == cd.name]
                curves[curve_name] = InterpolatedNodalCurve(
                    name=curve_name,
                    valuation_date=valuation_date,
                    node_dates=dates,
                    node_labels=labels,
                    y_values=y_vals,
                    interpolator_name=cd.settings.interpolator,
                )

            discount_name = next(e.curve_name for e in group_defn.entries if e.curve_type == "Discount")
            libor_name = next(e.curve_name for e in group_defn.entries if e.reference == "GBP-LIBOR-6M")

            return ImmutableRatesProvider(
                valuation_date=valuation_date,
                discount_curve=curves[discount_name],
                libor_curve=curves[libor_name],
                curve_order=order,
            )

        def objective(params: np.ndarray) -> np.ndarray:
            provider = build_provider(params)
            return np.array([par_spread(t, provider) for t in trades])

        def jacobian(params: np.ndarray) -> np.ndarray:
            n = len(params)
            eps = 1e-6
            jac = np.zeros((len(trades), n))
            for j in range(n):
                bumped_up = params.copy()
                bumped_down = params.copy()
                bumped_up[j] += eps
                bumped_down[j] -= eps
                jac[:, j] = (objective(bumped_up) - objective(bumped_down)) / (2.0 * eps)
            return jac

        calibrated_params = np.array(initial_guesses)
        final_jacobian = None
        max_residual = float("inf")
        for iteration in range(1, self.max_iter + 1):
            residuals = objective(calibrated_params)
            max_residual = float(np.max(np.abs(residuals)))
            final_jacobian = jacobian(calibrated_params)
            condition_number = float(np.linalg.cond(final_jacobian))
            logger.info(
                "Calibration iteration %d: max|R|=%.6e, cond(J)=%.6e",
                iteration,
                max_residual,
                condition_number,
            )
            if max_residual < self.tol:
                break
            calibrated_params -= np.linalg.solve(final_jacobian, residuals)
        else:
            residuals = objective(calibrated_params)
            max_residual = float(np.max(np.abs(residuals)))
            final_jacobian = jacobian(calibrated_params)
            condition_number = float(np.linalg.cond(final_jacobian))
            logger.info(
                "Calibration iteration %d: max|R|=%.6e, cond(J)=%.6e",
                self.max_iter,
                max_residual,
                condition_number,
            )
            if max_residual >= self.tol:
                raise RuntimeError(
                    f"Calibration failed to converge after {self.max_iter} iterations; "
                    f"last max|R|={max_residual:.6e}"
                )

        provider = build_provider(calibrated_params)

        # Compute inverse Jacobian for market quote sensitivity
        inv_jac = np.linalg.solve(final_jacobian, np.eye(len(trades)))

        idx = 0
        for curve_name in curve_names:
            cd = group_defn.curves[curve_name]
            n = len(cd.nodes)
            curve_inv = inv_jac[idx:idx + n, :]
            jacobian_info = JacobianCalibrationMatrix.of(order, curve_inv)
            curve = provider.find_curve(curve_name)
            curve.jacobian = jacobian_info
            idx += n

        return provider
