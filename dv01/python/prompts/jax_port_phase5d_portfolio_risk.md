# Phase 5d — Risk for Many Trades

## Objective
Support market-quote PV01 for batches of equal-shaped swaps and calculate
aggregate portfolio PV01 with a single gradient of summed present value.

## Implementation

In `pv01_jax/risk.py`, add:

```python
stack_trades(swaps: list[SwapArrays]) -> SwapArrays
batch_market_quote_pv01(problem, swaps_stacked, q) -> jax.Array
portfolio_market_quote_pv01(problem, swaps: list[SwapArrays], q) -> jax.Array
```

`stack_trades` should use `jax.tree_util.tree_map` and `jnp.stack` to combine
corresponding `SwapArrays` leaves. If any corresponding leaf shapes differ,
raise a clear error rather than returning a malformed batch.

Implement per-trade PV01 as:

```python
jax.vmap(market_quote_pv01, in_axes=(None, 0, None))(
    problem,
    swaps_stacked,
    q,
)
```

Implement portfolio PV01 by summing `pv_from_quotes` over the supplied swaps,
taking `jax.grad` with respect to `q`, and multiplying by `1e-4`.

## Tests

Create `tests/jax_port/test_phase5_portfolio.py` using `unittest`, using the
linear interpolation setting:

1. Build 10 swaps with `resolve_swap` from `create_test_swap`, using coupons
   from `jnp.linspace(0.01, 0.04, 10)`.
2. Compare `batch_market_quote_pv01` with a Python loop that calls
   `market_quote_pv01` for each swap, to absolute tolerance `1e-10`.
3. Assert `portfolio_market_quote_pv01` equals the sum of the batch rows to
   `1e-8`.
4. Assert `stack_trades` raises a clear error when given swaps with different
   tenors and thus different leaf shapes.

## Copilot Instructions

- Preserve trade and quote order in the batched output.
- The portfolio result should be computed as a gradient of summed PVs, not by
  independently summing separately differentiated results in production code.
- Do not silently pad or truncate mismatched trade arrays.
- Keep tests in `unittest` style.

Do not modify any `pv01_*` packages other than `pv01_jax` files explicitly in
scope. Do not modify existing tests; add only
`tests/jax_port/test_phase5_portfolio.py`. Run
`python -m unittest discover -s tests -t .` without `RUN_SLOW`, and report the
results.

## Validation

Run test discovery without setting `RUN_SLOW`, but exclude only the NCS cases
from `tests.test_reference_correctness`. Keep the Linear and DoubleQuadratic
reference-correctness cases and all other discovered tests. Since unittest
discovery has no built-in per-test exclusion option, discover the suite and
filter its test IDs:

```python
import sys
import unittest


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


discovered = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
selected = [
    test
    for test in flatten(discovered)
    if not (
        test.id().startswith("tests.test_reference_correctness.")
        and "__ncs_settings" in test.id()
    )
]
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(selected))
sys.exit(not result.wasSuccessful())
```

Report the test results and note that the NCS reference-correctness cases were
excluded.
