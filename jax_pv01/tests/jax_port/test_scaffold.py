import unittest

import pv01_jax
import jax.numpy as jnp


class TestJaxScaffold(unittest.TestCase):
    def test_default_float_dtype_is_float64(self):
        self.assertEqual(jnp.zeros(1).dtype, jnp.float64)
        self.assertEqual(jnp.asarray(1.0).dtype, jnp.float64)
