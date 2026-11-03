"""
FX market-data helpers: vol surfaces from ATM/RR/BF quotes, forwards from
forward points, and a sample EUR/USD market (`jax_finance.fx.data`).

Needs the `[fx]` extra: `pip install -e .[fx]` (QuantLib, SciPy, Polars).
"""

from .utils import (
    FXVolSurface,
    build_vol_surfaces,
    flatten_tuple_to_dict,
    get_forwards,
    get_fx_market_data_for_calibration,
    tenor_to_year_fraction,
)

__all__ = [
    "FXVolSurface",
    "build_vol_surfaces",
    "flatten_tuple_to_dict",
    "get_forwards",
    "get_fx_market_data_for_calibration",
    "tenor_to_year_fraction",
]
