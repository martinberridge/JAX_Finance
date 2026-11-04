import unittest
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from pv01_examples.common import VALUATION_DATE, load_example_inputs
from pv01_examples.paths import SETTINGS_SUFFIXES
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_jax.risk import (
    bump_market_quote_pv01,
    bump_market_quote_pv01_sequential,
    market_quote_pv01,
)


@lru_cache(maxsize=None)
def _inputs(settings_suffix: str):
    config, market_data, swap = load_example_inputs(settings_suffix)
    problem = resolve_calibration_problem(config, market_data, VALUATION_DATE)
    return problem, resolve_swap(swap, VALUATION_DATE), jnp.asarray(problem.q0)


class TestPhase7Bump(unittest.TestCase):
    def test_bump_matches_market_quote_pv01(self):
        """JAX bump-and-recalibrate PV01 against AD on the real config.

        A central difference with step h has truncation error h^2/6 times the
        third derivative of PV in that quote. At h = 1bp the observed
        difference from AD is at most 2.5e-5 GBP/bp (on buckets up to about
        6,000 GBP/bp). The tolerance starts from the 0.01 GBP/bp of the
        opt-in legacy bump comparison in test_phase5_risk and is tightened to
        1e-3, still 40x the observed truncation error.
        """
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem, swap, q = _inputs(suffix)
                ad = market_quote_pv01(problem, swap, q)
                bumped = jax.jit(lambda q: bump_market_quote_pv01(problem, swap, q))(q)
                np.testing.assert_allclose(np.asarray(bumped), np.asarray(ad), atol=1e-3, rtol=0.0)

    def test_vmap_and_lax_map_variants_agree(self):
        """The two bump variants run the same 2N recalibrations.

        They agree to about 1e-8 GBP/bp rather than exactly, because batched
        (vmap) and unbatched (lax.map) linear solves in the Newton steps round
        differently; 1e-7 GBP/bp allows for that.
        """
        for suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=suffix):
                problem, swap, q = _inputs(suffix)
                batched = jax.jit(lambda q: bump_market_quote_pv01(problem, swap, q))(q)
                sequential = jax.jit(lambda q: bump_market_quote_pv01_sequential(problem, swap, q))(q)
                np.testing.assert_allclose(np.asarray(batched), np.asarray(sequential), atol=1e-7, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
