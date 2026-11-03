"""Re-export shim: the FX utilities now live in `jax_finance.fx.utils`."""

from jax_finance.fx.utils import (
    FXVolSurface,
    build_vol_surfaces,
    flatten_tuple_to_dict,
    get_forwards,
    get_fx_market_data_for_calibration,
    tenor_to_year_fraction,
)

if __name__ == "__main__":
    from jax_finance.fx.data import atm, rrs, bfs, rates, spot, FWD_Points

    df = get_fx_market_data_for_calibration(atm, rrs, bfs, FWD_Points, spot, rates)
    print(df)
