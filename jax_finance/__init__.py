"""
JAX pricing kernels for the "autodiff for Greeks" blog series: Black-76
price and Greeks, and SABR implied vol and calibration.

Importing this package turns on `jax_enable_x64`. That is a process-wide
JAX setting, so it also applies to the caller's own JAX code, the same as
importing `pv01_jax`.

The FX market-data helpers are in `jax_finance.fx`, which needs the `[fx]`
extra (QuantLib, SciPy, Polars). They are not imported here.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .black import (
    FXOptionPricer,
    batch_price_and_greeks,
    black_price,
    compute_greeks_single,
)
from .sabr import (
    calibrate_sabr,
    calibrate_sabr_batch,
    calibrate_sabr_single_expiry,
    calibrate_sabr_surface,
    calibrate_sabr_surface_from_df,
    sabr_implied_vol,
    sabr_loss_for_points,
    sabr_params_from_raw,
)

__all__ = [
    "black_price",
    "compute_greeks_single",
    "batch_price_and_greeks",
    "FXOptionPricer",
    "sabr_implied_vol",
    "sabr_params_from_raw",
    "sabr_loss_for_points",
    "calibrate_sabr",
    "calibrate_sabr_batch",
    "calibrate_sabr_single_expiry",
    "calibrate_sabr_surface",
    "calibrate_sabr_surface_from_df",
]
