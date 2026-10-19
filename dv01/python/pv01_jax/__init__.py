import jax

jax.config.update("jax_enable_x64", True)

from .arrays import (
    CalibBlock,
    CalibrationProblem,
    CurveSpec,
    FloatLegArrays,
    LegArrays,
    SwapArrays,
)
from .calibrate import (
    calibrate,
    calibrate_checked,
    calibrate_with_info,
    newton_solve,
    newton_solve_with_info,
)
from .curves import Curve, curves_from
from .pricers import pv_swap, residuals
from .risk import (
    as_labelled,
    market_quote_pv01,
    parallel_pv01,
    pv_from_quotes,
    zero_rate_pv01,
)
from .resolve import (
    build_calibration_problem,
    par_spread_from_calibration_block,
    pv_from_arrays,
    resolve_calibration_problem,
    resolve_swap,
)

__all__ = [
    "CalibBlock",
    "CalibrationProblem",
    "CurveSpec",
    "FloatLegArrays",
    "LegArrays",
    "SwapArrays",
    "newton_solve",
    "newton_solve_with_info",
    "calibrate",
    "calibrate_checked",
    "calibrate_with_info",
    "Curve",
    "curves_from",
    "pv_swap",
    "residuals",
    "zero_rate_pv01",
    "as_labelled",
    "pv_from_quotes",
    "market_quote_pv01",
    "parallel_pv01",
    "resolve_swap",
    "pv_from_arrays",
    "build_calibration_problem",
    "resolve_calibration_problem",
    "par_spread_from_calibration_block",
]
