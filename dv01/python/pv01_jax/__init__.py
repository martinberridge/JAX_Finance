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
    batch_market_quote_pv01,
    bump_market_quote_pv01,
    bump_market_quote_pv01_sequential,
    cross_gamma,
    market_quote_pv01,
    parallel_pv01,
    portfolio_market_quote_pv01,
    pv_from_quotes,
    stack_trades,
    zero_rate_pv01,
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
    "cross_gamma",
    "stack_trades",
    "batch_market_quote_pv01",
    "bump_market_quote_pv01",
    "bump_market_quote_pv01_sequential",
    "portfolio_market_quote_pv01",
]
