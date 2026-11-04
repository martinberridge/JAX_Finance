# Working notes

The history of how this code was built. Nothing here is needed to run or review the
code, and it isn't part of the published site in `docs/`.

- [`refactor-plan.md`](refactor-plan.md): the audit, decisions and step-by-step record
  of the refactor, including the bugs it found and how each tolerance was measured.
- [`reference-corrections.md`](reference-corrections.md): the fixes to the NumPy reference
  and their effect on the numbers.
- [`prompts/`](prompts/): the prompts used for each phase of the JAX port. Their phase
  numbers map to the test modules through the table in
  [`jax_pv01/README.md`](../jax_pv01/README.md#tests).

These describe the repo as it was at the time, so file paths in them can be out of date.
The revision note at the top of `refactor-plan.md` maps the old paths to the new ones.
