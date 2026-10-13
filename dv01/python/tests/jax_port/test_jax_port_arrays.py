import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pv01_jax.arrays import CalibBlock, CurveSpec, FloatLegArrays, LegArrays, SwapArrays


class TestPhase1ArrayPytree(unittest.TestCase):
    def test_phase1_arrays_are_jax_trees_with_static_metadata(self):
        fixed_leg = LegArrays(
            start=np.array([0.0, 0.5], dtype=np.float64),
            end=np.array([0.5, 1.0], dtype=np.float64),
            rate=np.array([0.015, 0.017], dtype=np.float64),
            notional=np.array([1.0, 1.0], dtype=np.float64),
        )
        float_leg = FloatLegArrays(
            start=np.array([0.0, 0.5, 1.0], dtype=np.float64),
            end=np.array([0.5, 1.0, 1.5], dtype=np.float64),
            rate=np.array([0.02, 0.021, 0.022], dtype=np.float64),
            notional=np.array([2.0, 2.0, 2.0], dtype=np.float64),
        )
        swap = SwapArrays(
            fixed_leg=fixed_leg,
            float_leg=float_leg,
            spread=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        )
        block = CalibBlock(
            trade_type="IRS",
            quote_idx=1,
            fixed_leg=fixed_leg,
            float_leg=float_leg,
            quote=np.array([0.5, 1.5, 2.5], dtype=np.float64),
            metadata=("irs", "calibration"),
        )
        block2 = CalibBlock(
            trade_type="FIX",
            quote_idx=0,
            fixed_leg=None,
            float_leg=float_leg,
            quote=np.array([0.25], dtype=np.float64),
            metadata=("fix",),
        )

        self.assertIsNone(block2.fixed_leg)

        leaves = jax.tree_util.tree_leaves(block)
        self.assertTrue(all(isinstance(x, jax.Array) for x in leaves))
        self.assertTrue(all(x.dtype in (jnp.float64, jnp.int64) for x in leaves))

        curve = CurveSpec(
            name="test-curve",
            labels=("0y", "1y", "2y"),
            nodes=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            values=np.array([0.01, 0.02, 0.03], dtype=np.float64),
        )
        self.assertIsNotNone(hash(jax.tree_util.tree_structure(curve)))
        self.assertIsNotNone(hash(jax.tree_util.tree_structure(block)))
        self.assertIsNotNone(hash(jax.tree_util.tree_structure(swap)))

        updated = eqx.tree_at(lambda x: x.quote, block, jnp.array([9.0, 9.0, 9.0]))
        self.assertTrue(jnp.all(updated.quote == jnp.array([9.0, 9.0, 9.0], dtype=jnp.float64)))
        self.assertTrue(jnp.all(block.quote == jnp.array([0.5, 1.5, 2.5], dtype=jnp.float64)))

        seen = []

        @jax.jit
        def fn(item):
            seen.append(1)
            return item.quote.sum()

        _ = fn(block)
        _ = fn(block)
        self.assertEqual(len(seen), 1)

        q = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
        grad = jax.grad(lambda qq: qq[block.quote_idx].sum())(q)
        self.assertTrue(jnp.all(grad == jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)))


if __name__ == "__main__":
    unittest.main()
