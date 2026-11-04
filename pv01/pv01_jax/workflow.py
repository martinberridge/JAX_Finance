"""End-to-end workflow: loaded curve config, quotes and swap to PV and market-quote PV01.

Not re-exported from ``pv01_jax``: like ``pv01_jax.resolve``, which it calls,
it imports the Strata input layer. Import it as ``pv01_jax.workflow``.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_jax.risk import market_quote_pv01, pv_from_quotes

if TYPE_CHECKING:
    from pv01_loader.models import MarketData, RatesCurveGroupDefinition
    from pv01_product.resolved_trades import ResolvedSwapTrade


def price_and_risk(
    config: RatesCurveGroupDefinition,
    market_data: MarketData,
    swap: ResolvedSwapTrade,
    valuation_date: date,
) -> dict:
    """Calibrate the curves, price the swap and take market-quote PV01.

    Takes loaded objects, not file paths, so it works the same for the CSV
    example config and for curve definitions built in memory.

    Returns {"pv": GBP, "pv01": {curve_name: {quote_label: GBP per bp}}}.
    """
    problem = resolve_calibration_problem(config, market_data, valuation_date)
    swap_arrays = resolve_swap(swap, valuation_date)
    quotes = jnp.asarray(problem.q0)

    pv = float(pv_from_quotes(problem, swap_arrays, quotes))
    pv01 = np.asarray(market_quote_pv01(problem, swap_arrays, quotes))

    by_curve = {}
    for curve in problem.curves:
        start, end = curve.param_slice
        by_curve[curve.name] = {
            label: float(value) for label, value in zip(curve.labels, pv01[start:end])
        }
    return {"pv": pv, "pv01": by_curve}


__all__ = ["price_and_risk"]
