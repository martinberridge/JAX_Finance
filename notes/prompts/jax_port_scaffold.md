Add jax and equinox to the dependencies in pyproject.toml and to requirements.txt.
Create pv01_jax/__init__.py whose first statements are
    import jax
    jax.config.update("jax_enable_x64", True)
so any import of pv01_jax enables 64-bit before a jnp array exists. Export nothing else yet.
Create tests/jax_port/__init__.py (empty; do not name the folder tests/jax, it would
shadow the jax package) and tests/jax_port/test_phase1_scaffold.py with a
unittest.TestCase asserting that after `import pv01_jax`, jnp.zeros(1).dtype and
jnp.asarray(1.0).dtype are float64.
Acceptance: the new test passes and every existing test still passes.