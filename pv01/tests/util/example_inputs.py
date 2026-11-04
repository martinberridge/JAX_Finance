"""The GBP example configuration, resolved once per interpolator and shared by tests."""

from functools import lru_cache

from pv01_examples.common import VALUATION_DATE, load_example_inputs
from pv01_examples.paths import SETTINGS_SUFFIXES
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap


@lru_cache(maxsize=None)
def example_problem(settings_suffix: str):
    config, market_data, _ = load_example_inputs(settings_suffix)
    return resolve_calibration_problem(config, market_data, VALUATION_DATE)


@lru_cache(maxsize=None)
def example_swap_arrays():
    _, _, swap = load_example_inputs(SETTINGS_SUFFIXES[0])
    return resolve_swap(swap, VALUATION_DATE)
