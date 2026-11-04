# Phase 6d — Benchmark

## Objective
Time the JAX risk functions after compilation, report compile time separately,
and compare with the hand-coded bump-and-recalibrate PV01.

## Implementation

Create `pv01_jax/benchmark.py`, runnable as:

```bash
python -m pv01_jax.benchmark [--slow]
```

For each suffix in `SETTINGS_SUFFIXES`:

1. Build `problem`, `swap_arrays` and `q0 = problem.q0` as in earlier phases.
2. For each of `market_quote_pv01`, `parallel_pv01`, `cross_gamma` and
   `batch_market_quote_pv01`:
   - wrap it in `jax.jit` (or `eqx.filter_jit` where static fields need it);
   - report compile time as the wall time of the first call, synchronized with
     `block_until_ready()`;
   - after warm-up, report the median of 20 synchronized calls.
3. For `batch_market_quote_pv01`, use 100 swaps with coupons from
   `jnp.linspace(0.01, 0.04, 100)`, stacked with `stack_trades`.
4. Report the wall time of one hand-coded `bump_pv01` from
   `tests/util/bump.py`. Skip the NCS setting unless `--slow` is passed, and
   mark skipped rows clearly in the output.

Print a Markdown table (suffix, function, compile time, median run time) and
save the same table to `target/benchmark.md`.

As a guide, the prototype gave 1.5–3.4 ms for JAX market-quote PV01 against
about 23 s for the hand-coded bump (linear setting).

## Tests

None. This is a script, not a test. Run it once and include the table in your
report.

## Copilot Instructions

- Synchronize every timed call with `block_until_ready()`.
- Keep compile time separate from run time.
- Use `time.perf_counter` for timing and `statistics.median` for the median.
- Do not add test assertions or performance thresholds.
- Do not import `tests.*` at module import time in a way that breaks
  `pv01_jax` imports; import `bump_pv01` inside the function that uses it.
- Keep `pv01_jax` itself free of production pricer imports; the hand-coded
  comparison may use them only inside the benchmark entry point.

Do not modify any `pv01_*` packages other than adding `pv01_jax/benchmark.py`.
Do not modify existing tests. 
