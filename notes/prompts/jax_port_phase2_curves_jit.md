# Phase 2e — Curve JIT Behaviour and Performance Tests

## Objective
Extend `tests/jax_port/test_phase2_curves.py` with JIT‑related behavioural and performance tests for the `Curve` module and its interpolators.

These tests verify:

- JIT trace stability across different `z` values  
- JIT retracing when the interpolator changes  
- Real‑world performance characteristics of each interpolator  

---

## Part 1 — JIT Trace Stability Tests

### (5a) JIT trace count across multiple `z` values

Create a `jax.jit`‑wrapped function:

```
f(curve, t) -> curve.df(t)
```

Evaluate it over **1,000 query times**.

Instrument the function to **count how many times it is traced**.

Test requirement:

- Calling `f` three times with **different `z` values** (same interpolator) must result in **exactly one trace**.

This confirms that:

- The traced computation depends only on structure (interpolator, shapes)
- Changing curve parameters (`z`) does not trigger retracing

---

## Part 2 — JIT Retrace on Interpolator Change

### (5b) Changing interpolator triggers retrace

Construct a second `Curve` whose `spec.interpolator` differs from the first.

Requirement:

- Calling the same jitted function `f(curve, t)` with this new curve must **trace again**.

This confirms that:

- Interpolator choice is part of the traced structure
- JAX correctly retraces when the computation graph changes

---

## Part 3 — Performance Measurement (No Assertions)

### (5c) Median wall‑time benchmark

For each interpolator in `INTERPOLATORS`:

1. Warm up the jitted `df(t)` call using `block_until_ready()`.
2. Run **100 jitted evaluations** of `df(t)` over **1,000 query times**.
3. Measure wall‑time for each run.
4. Compute the **median wall‑time**.
5. Print the results.

This section has **no assertions** — it is diagnostic only.

### Reporting requirement

At the end of the test run, print a summary table:

- interpolator name  
- median wall‑time (in microseconds or milliseconds)  

This summary must appear in the test output.

---

## Copilot Instructions

- Add all tests to `tests/jax_port/test_phase2_curves.py`.
- Use **only** JAX (`jax.numpy`, `jax.jit`, `block_until_ready`) — no NumPy.
- Count traces by instrumenting the jitted function.
- Ensure:
  - Same interpolator → one trace across multiple `z` values.
  - Different interpolator → retrace occurs.
- Benchmark:
  - 1,000 query times
  - 100 jitted evaluations
  - Warm‑up before timing
  - Print median wall‑time per interpolator
- Do not add assertions to the performance test.
- Ensure all existing Phase‑2 tests continue to pass without modification.