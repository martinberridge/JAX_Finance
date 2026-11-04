import unittest
from datetime import date
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_jax.interp import INTERPOLATORS
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.rates_curve_calibrator import RatesCurveCalibrator


VALUATION_DATE = date(2016, 8, 1)


@lru_cache(maxsize=None)
def _market_data():
    return load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )


@lru_cache(maxsize=None)
def _config(settings_suffix):
    config_dir = PATH_CONFIG / CONFIG_STR
    return load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}{settings_suffix}.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]


@lru_cache(maxsize=None)
def _provider(settings_suffix):
    return RatesCurveCalibrator.standard().calibrate(
        _config(settings_suffix),
        _market_data(),
        VALUATION_DATE,
    )


class TestInterpolators(unittest.TestCase):
    def test_values_jacobians_and_gradients_match_calibrated_curves(self):
        for settings_suffix in SETTINGS_SUFFIXES:
            with self.subTest(settings_suffix=settings_suffix):
                provider = _provider(settings_suffix)
                curves = (provider.discount_curve, provider.libor_curve)

                for curve in curves:
                    for interpolator_name, interp in INTERPOLATORS.items():
                        if curve.interpolator_name != interpolator_name:
                            continue
                        with self.subTest(
                            settings_suffix=settings_suffix,
                            curve=curve.name,
                            interpolator=interpolator_name,
                        ):
                            self._assert_interpolator_matches_curve(
                                interp,
                                curve.x_values,
                                curve.y_values,
                                curve,
                            )

    def _assert_interpolator_matches_curve(self, interp, x_values, z_values, curve):
        x = jnp.asarray(x_values, dtype=jnp.float64)
        z = jnp.asarray(z_values, dtype=jnp.float64)
        t_grid = np.concatenate(
            (
                np.linspace(-1.0, 35.0, 401),
                x_values,
                x_values - 1e-9,
                x_values + 1e-9,
            )
        )
        t = jnp.asarray(t_grid, dtype=jnp.float64)

        actual_values = np.asarray(interp(x, z, t))
        expected_values = np.asarray([curve.y_value(float(ti)) for ti in t_grid])
        np.testing.assert_allclose(actual_values, expected_values, rtol=0.0, atol=1e-13)

        actual_jacobian = np.asarray(jax.jacfwd(interp, argnums=1)(x, z, t))
        expected_jacobian = np.stack(
            [
                curve.y_value_parameter_sensitivity(float(ti))
                for ti in t_grid
            ],
            axis=0,
        )
        np.testing.assert_allclose(
            actual_jacobian,
            expected_jacobian,
            rtol=0.0,
            atol=1e-12,
        )

        gradient_z = jax.grad(lambda node_values: interp(x, node_values, t).sum())(z)
        gradient_t = jax.grad(lambda query_times: interp(x, z, query_times).sum())(t)
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient_z))))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient_t))))


if __name__ == "__main__":
    unittest.main()
