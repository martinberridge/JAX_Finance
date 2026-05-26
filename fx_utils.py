from scipy.stats import norm
import numpy as np
import QuantLib as ql
import polars as pl

_TENOR_ALIASES = {"on": "1D"}

def _tenor_to_ql_period(tenor: str) -> ql.Period:
    if tenor in _TENOR_ALIASES:
        return ql.PeriodParser.parse(_TENOR_ALIASES[tenor])
    return ql.PeriodParser.parse(tenor[:-1] + tenor[-1].upper())


def tenor_to_year_fraction(
    tenor: str,
    valuation_date: ql.Date | None = None,
    day_count: ql.DayCounter | None = None,
) -> float:
    if valuation_date is None:
        valuation_date = ql.Date.todaysDate()
    if day_count is None:
        day_count = ql.Actual365Fixed()

    period = _tenor_to_ql_period(tenor)
    return day_count.yearFraction(valuation_date, valuation_date + period)

def get_forwards(spot:float,points:dict) -> dict:
    fwds = { tenor : spot + (points/100) for tenor,  points in points.items() }
    fwds['on'] = spot 
    return fwds 

class FXVolSurface:
    """Complete FX vol surface from ATM/RR/BF quotes"""

    def __init__(self, spot: float, rate_dom: float, rate_for: float, tenor_years: float):
        self.spot = spot
        self.rate_dom = rate_dom
        self.rate_for = rate_for
        self.tenor_years = tenor_years

    def strike_from_delta(self, delta: float, vol: float | list[float], is_call: bool = True) -> float:
        """
        Calculate strike from delta using Black-Scholes

        Delta = N(d1) for calls
        Delta = N(d1) - 1 for puts
        """
        rate_diff = self.rate_dom - self.rate_for

        if is_call:
            d1 = norm.ppf(delta)
        else:
            d1 = norm.ppf(delta + 1)

        vol_sqrt_t = vol * np.sqrt(self.tenor_years)

        strike = self.spot * np.exp(
            (d1 * vol_sqrt_t) - (rate_diff * self.tenor_years) - (vol ** 2 / 2) * self.tenor_years
        )

        return strike

    def delta_from_strike(self, strike: float, vol: float, is_call: bool = True) -> float:
        """Calculate delta from strike"""
        rate_diff = self.rate_dom - self.rate_for

        d1 = (np.log(self.spot / strike) + (rate_diff + vol**2 / 2) * self.tenor_years) / (
            vol * np.sqrt(self.tenor_years)
        )

        if is_call:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def build_from_quotes(
        self,
        atm_vol: float,
        rr_25: float,
        bf_25: float,
        rr_10: float,
        bf_10: float,
    ) -> dict:
        """
        Build vol surface from standard quotes

        Returns vols at ATM, 25-delta, and 10-delta levels
        """
        call_25_vol = atm_vol + bf_25 + (rr_25 / 2)
        put_25_vol = atm_vol + bf_25 - (rr_25 / 2)
        call_10_vol = atm_vol + bf_10 + (rr_10 / 2)
        put_10_vol = atm_vol + bf_10 - (rr_10 / 2)

        strike_25_call = self.strike_from_delta(0.25, call_25_vol / 100, is_call=True)
        strike_25_put = self.strike_from_delta(-0.25, put_25_vol / 100, is_call=False)
        strike_10_call = self.strike_from_delta(0.10, call_10_vol / 100, is_call=True)
        strike_10_put = self.strike_from_delta(-0.10, put_10_vol / 100, is_call=False)

        return {
            'atm': {
                'strike': self.spot,
                'vol': atm_vol,
                'delta': 0.50,
            },
            '25_delta_call': {
                'strike': strike_25_call,
                'vol': call_25_vol,
                'delta': 0.25,
            },
            '25_delta_put': {
                'strike': strike_25_put,
                'vol': put_25_vol,
                'delta': -0.25,
            },
            '10_delta_call': {
                'strike': strike_10_call,
                'vol': call_10_vol,
                'delta': 0.10,
            },
            '10_delta_put': {
                'strike': strike_10_put,
                'vol': put_10_vol,
                'delta': -0.10,
            },
        }


def flatten_tuple_to_dict(tup, tenor_key='tenor', sep='_'):
    """Convert tuple (tenor, dict) to flattened dict"""
    
    def flatten_dict(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            elif isinstance(v, (np.integer, np.floating)):
                items.append((new_key, float(v) if isinstance(v, np.floating) else int(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    tenor_value, data_dict = tup
    flattened = flatten_dict(data_dict)
    return {tenor_key: tenor_value, **flattened}

def build_vol_surfaces(
    atm_quotes: dict,
    rr_quotes: dict,
    bf_quotes: dict,
    spot: float,
    rate_dom: float,
    rate_for: float,
    valuation_date: ql.Date | None = None,
) -> dict[str, dict]:
    surfaces = {}

    for tenor, atm_vol in atm_quotes.items():
        tenor_years = tenor_to_year_fraction(tenor, valuation_date)
        surface = FXVolSurface(spot, rate_dom, rate_for, tenor_years)
        surfaces[tenor] = {
            'tenor_years': tenor_years,
            'surface': surface.build_from_quotes(
                atm_vol=atm_vol,
                rr_25=rr_quotes['25d'][tenor],
                bf_25=bf_quotes['25d'][tenor],
                rr_10=rr_quotes['10d'][tenor],
                bf_10=bf_quotes['10d'][tenor],
            ),
        }

    return surfaces

def get_fx_market_data_for_calibration(atm_quotes: dict, rr_quotes: dict, bf_quotes: dict, FWD_Points:dict, spot: float, rates: dict) -> pl.DataFrame:
    surfaces = build_vol_surfaces(atm_quotes, rr_quotes, bf_quotes, spot, rates['estr'], rates['sofr'])
    fwds = get_forwards(spot,FWD_Points) # EUR/USD Forward Points
    surfaces_list = list(surfaces.items())
    flattened_dicts = [flatten_tuple_to_dict(tup) for tup in surfaces_list]
    fwds_df =  pl.DataFrame( {"tenor":list(fwds.keys()), "forward":list(fwds.values())})
    surface_df = pl.DataFrame(flattened_dicts)
    merged_df = surface_df.join(fwds_df, on="tenor", how="left")
    return merged_df

if __name__ == "__main__":
    from fx_data import atm, rrs, bfs, rates, spot, FWD_Points

    df = get_fx_market_data_for_calibration(atm, rrs, bfs, FWD_Points, spot, rates)
    print(df)