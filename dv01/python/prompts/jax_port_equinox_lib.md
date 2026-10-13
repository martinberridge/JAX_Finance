Create pv01_jax/arrays.py with the Equinox modules in the Phase 1 doc's "Pytree design"
section: LegArrays, FloatLegArrays(LegArrays), SwapArrays, CalibBlock, CurveSpec,
CalibrationProblem. Rules:
- every array field uses eqx.field(converter=jnp.asarray);
- strings, ints and tuples are eqx.field(static=True); static values must be hashable
  (tuples, never lists or arrays);
- legs are named fixed_leg / float_leg (never a field called `float`);
- CalibBlock.fixed_leg is Optional and None for FIX.
No pricing code. Add tests/jax_port/test_phase1_arrays.py (unittest) that builds small
instances by hand from numpy (e.g. an IRS block with 2 trades and 3 periods, a FIX block
with fixed_leg=None) and asserts:
(a) every leaf of jax.tree_util.tree_leaves is a jax.Array, float64 or int64;
(b) static fields are not leaves and hash(jax.tree_util.tree_structure(x)) works;
(c) eqx.tree_at returns a modified copy and leaves the original unchanged;
(d) a jax.jit function taking a CalibBlock runs twice with one trace (count calls);
(e) jax.grad(lambda q: q[block.quote_idx].sum())(q) is 1 at quote_idx and 0 elsewhere.