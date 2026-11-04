# JAX_Finance refactor: audit and plan

## Context

The goal is to make JAX_Finance read like a production-grade companion to the
"autodiff for Greeks" blog series. That means pure JAX pricing kernels,
autodiff DV01, pytest validation against finite differences and closed forms,
a scaling benchmark that produces a chart, and an updated README.

This file is the audit of the repo as it stands (2026-10-28) and the plan.

**Revision (2026-10-28).** An earlier draft proposed a separate single-curve
"teaching" DV01 module. That is dropped. `dv01/python/pv01_jax` is now the
pricing and risk kernel for DV01. The non-JAX code under `dv01/` is kept only
to load the real market configuration and to provide the reference results
the benchmark and tests compare against. As a result, `dv01/` is no longer
left untouched. Step 2 restructures it.

**Revision (2026-11-01).** Step 1 had no regression tests. It moves the Black
and SABR code, switches it to float64, changes the SABR ATM branch and
rewrites SABR calibration as `lax.scan`, and the only check that behaviour
was preserved was comparing demo output by eye. A new Step 0.5 captures CSV
goldens from the current code before anything changes, and Step 1 now adds
`tests/test_regression.py`, which checks the new package against them.

**Revision (2026-11-01, later).** Restoring the dv01 goldens, untracked
since commit 667438d, moved from Step 3 to a new Step 0.25. Step 2 relies on
the goldens to show it changed nothing, so they must be tracked before it
starts. Step 0.25 has explicit stop points where a decision is needed.

**Revision (2026-11-03).** Step 0.25 is done. The committed `667438d^`
goldens turned out to be pre-bug-fix output, so the local 2026-10-05 copies
were tracked instead. The dv01 goldens are regression baselines from the
Python port, not Strata output; Step 0.25 records the details.

**Revision (2026-11-03, later).** Step 0.5 is done, but not as first
written. Its first float32 capture found two bugs in the old
`sabr_implied_vol`, a meaningless calibration golden, and float32 errors
far above Step 1's `rtol=1e-6`. The SABR goldens now come from QuantLib,
the Black golden is the old code in float64, Step 1 fixes both SABR bugs,
and Step 1's test bullets are amended to match. Step 0.5 records the details.

**Revision (2026-11-04).** All steps are done. The repo was then reorganised
so a reviewer can find their way around it. Paths in this document are as they
were when each step ran, so read them through this map:

| Then | Now |
|---|---|
| `dv01/python/` | `pv01/` |
| `dv01/examples/src/main/resources/example-calibration/` | `pv01/data/` |
| `JAX_black_option_pricing.py`, `SABR_JAX.py`, `JAX_PV01.py` | `examples/black.py`, `examples/sabr.py`, `examples/pv01.py` |
| `fx_utils.py`, `fx_data.py` (shims) | removed; use `jax_finance.fx` |
| `tests/util/bump.py` (shim) | removed; use `pv01_pricer.bump` |
| `tests/jax_port/test_phaseN_*.py` | named by subject; see `pv01/README.md` |
| `docs/refactor-plan.md`, `dv01/python/prompts/` | `notes/` |
| `docs/assets/dv01_scaling.png` | `docs/assets/pv01_scaling.png` |

This also settles the first "Still open" question: the demos moved to
`examples/`, without root-level shims.

**Running this plan.** Do one step per session, commit after each, and stop
at the end of the step. Wherever a step says **stop**, report what was found
and wait for a decision rather than choosing.

## Audit

### What exists

| Area | Location | State |
| --- | --- | --- |
| Black-76 pricer and Greeks | `JAX_black_option_pricing.py` | Pure `black_price` and `compute_greeks_single`, using `value_and_grad` plus `grad(grad)` for gamma, and a `vmap` batch. `FXOptionPricer` is a thin JIT wrapper class. Runs in float32 (no x64). Docstrings have no formulas. |
| SABR calibration | `SABR_JAX.py` | Pure Hagan (2002) `sabr_implied_vol` with a `jnp.where` ATM fallback. Adam is hand-written, but calibration runs as a **Python `for` loop** and the expiries are calibrated sequentially. Everything is forced to float32. |
| FX market data utilities | `fx_utils.py`, `fx_data.py` | Use QuantLib, SciPy and Polars, not JAX. This is data preparation, not kernels, so it can stay non-JAX. |
| JAX PV01 kernel | `dv01/python/pv01_jax/` | `risk.py` has `jax.grad` zero-rate and market-quote PV01, a `jax.jvp` parallel PV01, a `jax.hessian` cross-gamma, a `vmap` batch PV01 and a portfolio PV01. It differentiates implicitly through calibration (`jax.lax.custom_root`), using a jitted Newton solver that builds its Jacobian with `jacfwd`. Runs in x64. |
| Hand-coded reference | `dv01/python/pv01_{basics,loader,product,market,pricer}/` | A NumPy port of Strata's `CalibrationInterpolationExample`. Trade and curve sensitivities are hand-coded analytic (a reverse-mode chain: point sensitivities → node sensitivities → $J^{-1}$). The calibration Jacobian $J$ is **central finite differences** (`rates_curve_calibrator.py`), so market-quote sensitivities are not fully analytic. |
| DV01 tests | `dv01/python/tests/` | Around 20 unittest modules in `jax_port/`: gradient checks, stress tests, frozen goldens, an opt-in bump-and-recalibrate comparison, and comparisons against the hand-coded analytic path (`test_phase2_interp.py` at `atol=1e-12`, `test_phase3_risk.py` at `delta=1e-6` GBP/bp). |
| DV01 benchmark | `dv01/python/pv01_jax/benchmark.py` → `target/benchmark.md` | Compares the AD functions with the NumPy `bump_pv01` for three interpolators. The output is a table only, with no chart and a fixed curve size of 19 nodes. |
| Blog | `docs/` (Jekyll, minima) | Part 1 (Greeks via autodiff) and Part 2 (inside the autodiff engine) are published. The Black implementation and SABR posts are drafts. `docs/pv01_jax.md` documents the port. No post links to a Python file path. |

### Gaps against the brief

1. **Docstrings:** the Black and SABR functions have no mathematical
   definitions.
2. **Precision:** the Black and SABR kernels run in float32, which can't be
   validated to ~1e-7.
3. **SABR purity and speed:** the Python training loop isn't jittable as a
   whole, and expiries aren't vectorised.
4. **Possible NaN gradients in the SABR ATM branch:** `jnp.where` is applied to
   `z / x(z)` after the division. At K = F the masked-out branch can still pass
   NaN or inf into the gradient (the "double-where" trap). This needs a test and
   a safe-denominator fix.
5. **No top-level tests:** nothing outside `dv01/` is tested, and no pytest
   setup exists at the repo root.
6. **The benchmark comparison isn't like for like.** The NumPy
   `bump_pv01` takes 18–35s, most of it the Python calibrator. The AD path
   takes 0.7–1.5ms after jitting. The ratio mixes "Python versus XLA" with
   "2N calibrations versus one". The NCS row is skipped by default because
   the NumPy bump is too slow.
7. **No scaling benchmark or chart:** the curve size is fixed at 19 nodes by
   the Strata CSV config (10 OIS + 9 LIBOR-6M), so there's nothing to sweep,
   and `docs/assets/` doesn't exist.
8. **Dependencies run the wrong way in `dv01/python`:**
   - `pv01_jax/benchmark.py` imports `pv01_examples` and `tests.util.bump`,
     so the kernel package depends on examples and tests.
   - `pv01_jax/__init__.py` re-exports `resolve`, so `import pv01_jax` loads
     `pv01_basics`, `pv01_loader` and `pv01_product`, even though the
     numerical core needs only jax, equinox and numpy.
   - The JAX examples and the benchmark import the test-swap constants
     (`VALUATION_DATE`, `SWAP_COUPON`, ...) from another runnable example,
     `multicurve2/calibration_interpolation_example.py`.
   - `pv01_examples/jax/calibration_interpolation_example_jax.py` imports
     `bump_pv01` from `tests/util/bump.py`, which isn't packaged.
9. **Environment:** matplotlib isn't installed, and the root has no
   `pyproject.toml`.
10. **README:** two short paragraphs. There's no architecture section, roadmap
    or quickstart.
11. **No regression baseline for Black and SABR:** nothing records what the
    current code outputs, so there's no automated way to show that Step 1
    preserves behaviour. Step 4's tests check correctness against closed
    forms, not agreement with the old code, and they run after Step 1.

## Decisions

### Decision 1: `pv01_jax` is the DV01 pricing and risk kernel

**Decision.** All DV01 work in the refactor (tests, benchmark, chart,
README) is built on `pv01_jax`. No separate DV01 module is added at the
repo root.

**Why.**

- `pv01_jax` already has every autodiff technique the blog covers: reverse
  mode (`grad`), forward mode (`jvp`), second order (`hessian`), batching
  (`vmap`) and implicit differentiation through calibration (`custom_root`).
- It's validated against the hand-coded analytic sensitivities and
  bump-and-recalibrate, and regression-tested against the Python port's
  golden baselines (Step 0.25), so the benchmark's numbers come from a
  kernel already known to be correct.
- A second, simplified curve module would duplicate interpolation and
  discounting logic. It would also invite readers to confuse the two.

**What replaces the teaching module's two roles.**

- **Variable curve size for the scaling chart:** `CalibrationProblem` is
  made only of arrays and static metadata, so a new
  `pv01_jax/synthetic.py` can build an N-quote problem directly, without
  loaders, calendars or `resolve.py` (Step 2).
- **Closed-form tests:** the hand-coded analytic path in `pv01_pricer` is
  already the closed-form reference at the zero-rate level, with tests at
  `atol=1e-12` (interpolation) and `delta=1e-6` GBP/bp (zero-rate PV01).
  Market-quote PV01 has no closed form, because the hand-coded $J$ is itself
  finite-difference. That's inherent to calibration, and it's covered by
  bump-and-recalibrate and goldens. Step 4 adds structural tests on the
  synthetic problem that need no reference implementation.

**Cost.** The repo no longer has a DV01 example readers can take in at one
sitting. The README and `docs/pv01_jax.md` make up for this by walking
through `pv_from_quotes` → `market_quote_pv01` as the minimal path, and the
root `JAX_PV01.py` demo (Step 3) runs that path end to end.

### Decision 2: the main bump-and-reprice baseline runs on the same JAX kernel

**Decision.** Add `bump_market_quote_pv01` to `pv01_jax/risk.py`. It does
central bump-and-recalibrate by calling the jitted `pv_from_quotes` 2N
times. The benchmark's main comparison is this function against
`market_quote_pv01`. The NumPy `bump_pv01` stays as one "legacy" row on the
real configuration only.

```python
def bump_market_quote_pv01(problem, swap, q, h=1e-4):
    """Central bump-and-recalibrate PV01 in GBP per bp: 2N calibrations."""
    bumps = h * jnp.eye(q.shape[0])
    pv = jax.vmap(lambda dq: pv_from_quotes(problem, swap, q + dq))
    return (pv(bumps) - pv(-bumps)) / 2.0
```

With `h = 1e-4` (1bp) the central difference is already in GBP per bp, so no
rescaling is needed.

**Why.**

- Both methods run the same compiled pricer and calibrator. The only
  difference left is how many passes each one needs, which is what the blog
  is about.
- The NCS interpolator no longer has to be skipped: a jitted recalibration
  takes milliseconds rather than seconds.
- Use a `jax.lax.map` variant as well as `vmap`. `vmap` evaluates all bumps
  as one batch. `lax.map` evaluates them one at a time, which is how a
  conventional risk system runs. Reporting both shows how much of the gap
  is vectorisation and how much is pass count.

**What the chart should show.** Each Newton step in
`calibrate.newton_solve_with_info` builds its Jacobian with `jacfwd`, so a
single calibration already costs roughly N forward passes per iteration.
Therefore:

- `market_quote_pv01` costs about **one calibration**, plus one transposed
  linear solve in the `custom_root` backward pass.
- `bump_market_quote_pv01` costs **2N calibrations**.

The expected result is a gap that grows roughly linearly in N, not a flat
AD line. "AD costs about one extra calibration; bumping costs 2N" is the
honest headline once calibration is involved, and the README states it that
way.

**The non-JAX code that stays, and why.**

| Package | Role after the refactor |
| --- | --- |
| `pv01_basics`, `pv01_loader`, `pv01_product` | Input layer: loads the real GBP config and builds trades, which `pv01_jax/resolve.py` converts into arrays |
| `pv01_market`, `pv01_pricer` | Frozen reference: the hand-coded analytic sensitivities (closed-form reference for the zero-rate tests) and the NumPy calibrator |
| `pv01_pricer/bump.py` (moved from `tests/util/bump.py`) | Legacy bump-and-recalibrate: one benchmark row on the real config and the opt-in test comparison |

None of these take part in the scaling sweep. Building synthetic NumPy
configurations for every N isn't worth the effort, because the JAX bump
already covers the sweep.

### Decision 3: move the Black and SABR kernels into a `jax_finance/` package

**Decision.** Move the Black and SABR kernels into an installable
`jax_finance/` package (`pip install -e .` from a root `pyproject.toml`).
Keep `JAX_black_option_pricing.py` and `SABR_JAX.py` at their current paths
as thin `__main__` demos that import from the package. This gives the root
of the repo the same pattern `dv01/python` already has: packages for the
code, and example scripts that call them.

**Why.**

- Root-level pytest needs an importable package. Importing top-level scripts
  from tests only works through `sys.path` tweaks, and it runs any
  module-level demo code on import.
- The draft posts use the function names `black_price` and
  `compute_greeks_single`. The package keeps those names, so the drafts stay
  correct. No post links to a Python file path, so moving files breaks no
  links.

**Consequences.**

- **float64 everywhere.** `jax_finance/__init__.py` turns on
  `jax_enable_x64`, so the demo output changes in the last few digits
  compared with today's float32 run. Because x64 is a process-wide JAX
  setting, importing `jax_finance` also turns it on for the caller.
  `pv01_jax` behaves the same way, and the package docstring says so.
- **FX utilities become an optional extra.** They depend on QuantLib, SciPy
  and Polars, so they move to `jax_finance/fx/` behind
  `pip install -e .[fx]`. The core package and its tests then need only
  `jax` and `numpy`. The old `fx_utils.py` and `fx_data.py` paths become
  re-export shims.
- **Two Python roots.** `dv01/python` keeps its own `pyproject.toml` and
  unittest suite. The root `pyproject.toml` sets `testpaths = ["tests"]`, so
  a bare `pytest` at the root doesn't pick up the dv01 suite.

### Still open

- Should the root demo scripts (`JAX_black_option_pricing.py`, `SABR_JAX.py`
  and `JAX_PV01.py`) stay at the repo root, or move to an `examples/` folder
  with root-level shims?
- Should the shims raise a `DeprecationWarning`? The recommendation is no,
  because they're demo entry points, not a public API.
- Should the scaling sweep also time `batch_market_quote_pv01` against a
  portfolio size axis? Portfolio size is a second axis on which AD and
  bumping scale differently.
- Should the hand-coded analytic path (`present_value_sensitivity` →
  `parameter_sensitivity` → `MarketQuoteSensitivityCalculator`) get its own
  row on the real configuration, as "AD versus a human-written adjoint"? If
  it does, the row must note that its calibration Jacobian is
  finite-difference.

## Plan

### Step 0: save this document

Done. GitHub Pages renders `.md` files under `docs/` as site pages, as it
already does for `docs/pv01_jax.md`.

### Step 0.25: restore the dv01 goldens to version control

**Done (2026-11-03, commit 065f4f0).** The six golden CSVs in
`dv01/python/tests/golden/` are tracked again, and the `.gitignore` rule
that excluded them is removed.

What was found, and the decision taken:

- **The removal was deliberate.** 667438d ("Remove generated test results
  from source control") deleted the goldens and added
  `dv01/python/tests/golden/*.csv` to `.gitignore`. The next commit,
  b59d2fa, did the same for `target/example-output/*.csv`.
- **The local copies differ from `667438d^` in every file.** The committed
  versions came from the first commit (e70befd, 2026-07-05), before the
  Phase 0 corrections (fixed-leg discount-factor sign, OIS compounding,
  residual sign) and the LIBOR forward fix. Their PVs are the "before"
  column in `dv01/python/README.md`. The local copies were regenerated on
  2026-10-05 after those fixes, and the golden tests pass against them.
  Restoring `667438d^` would make the suite fail, with PVs off by about
  21,000 GBP.
- **Provenance correction.** An earlier version of this step said the
  goldens "came from the independent Strata reference". They didn't. Both
  versions are output from the Python port itself. They are regression
  baselines: they show that a change preserved the port's numbers, which is
  what Step 2 needs, but they are not an independent check of correctness.
  That comes from the hand-coded analytic tests and bump-and-recalibrate.
- **Decision (option A):** track the local 2026-10-05 copies unchanged.
  Removing the `.gitignore` rule is the narrowing the original step asked
  for, because the rule matched only these files. No `git add -f`, and no
  history rewrite.

**Rule for later steps:** never regenerate these goldens to make a test
pass. If a refactor step changes them, that step changed behaviour.

### Step 0.5: capture Black and SABR goldens from the current code

**Done (2026-11-03), with the changes below.** The goldens were captured on
commit 065f4f0 by `tests/golden/capture_v0.py`, which is committed with
them. Its docstring records the decisions, and
`tests/golden/v0_manifest.txt` records the commit, versions and settings.
`tests/conftest.py` has the reader, `read_golden(name)`: a dict of NumPy
arrays keyed by column, with text columns (`option_type`, `expiry`) kept as
strings. Nothing in `.gitignore` excludes `tests/golden/`.

What the first, float32 capture found:

- **The old `sabr_implied_vol` has two bugs.**
  - Every K > F returns the 1e-12 floor. `z / jnp.maximum(xz, eps)`
    replaces x(z), which is negative whenever z is, with `eps`.
  - The (1−β)²/24 coefficients in D, the time adjustment and the ATM
    branch are written `pow_ * pow_ / 24` with `pow_ = (1−β)/2`, a factor of
    4 too small. The error reaches 1.2% in the β = 0 wings.
  - With both fixed, the formula agrees with QuantLib's `sabrVolatility`
    to 6.5e-16. Any SABR results from this code, including the demos and
    the EURUSD calibration, are affected.
- **The calibration golden was meaningless.** The demo parameters give
  about 2.5% ATM vol, and with the K > F bug every strike above F was noise
  around zero, some of it negative. Only one or two strikes carried vega
  weight.
- **Float32 can't support Step 1's `rtol=1e-6`.** Against the same old code
  run in float64, Black gamma, vega and theta were off by up to 2e-4
  relative on ordinary values, price and rho by up to 0.65 on tiny OTM
  prices, and SABR vols by up to 2.2e-4 near ATM.

Decisions, and what was captured instead (all CSV with a header row, every
value written with `repr` of its float64 value):

1. **Step 1 fixes both SABR bugs**, so the SABR goldens come from QuantLib,
   not from the old code. `tests/golden/sabr_vol_ref.csv` (columns `F, K,
   T, alpha, beta, rho, nu, vol`) covers four parameter sets (β = 0, 0.5,
   0.7, 1) at 17 strikes from 0.5F to 2F. It includes K = F and strikes
   within 1e-4 of it.
2. **The Black golden is the old code run with x64 on.** `black_price` and
   `compute_greeks_single` have no hard float32 casts, so this is the old
   formula in float64. `tests/golden/black_v0.csv` (columns `option_type,
   F, K, T, sigma, r, price, delta, gamma, vega, theta, rho`) has 72 rows:
   calls and puts, K/F ∈ {0.95, 1, 1.05}, T ∈ {0.05, 0.25, 2}, σ ∈ {5%,
   30%}, r ∈ {0, 5%}. It agrees with an ω-form rewrite to 6e-17 absolute.
3. **The calibration smile is generated by QuantLib from known
   parameters**, at about 10% ATM vol and with no noise.
   `tests/golden/sabr_calib_input.csv` (columns `expiry, T, F, K,
   market_vol`) has three expiries with 13 strikes each, spaced ±2.5 ATM
   standard deviations, with the middle strike at K = F.
   `tests/golden/sabr_calib_true.csv` (columns `expiry, alpha, beta, rho,
   nu, n_steps, learning_rate`) holds the parameters, with 3000 steps at
   lr 8e-3. The old Adam algorithm on the corrected formula recovers them
   to about 1e-14 in float64 in that many steps, but only to about 1e-4 in
   the demo's 1200.

Only `black_v0.csv` depends on the old code. `capture_v0.py` reproduces it
only on 065f4f0 and must not be re-run after Step 1; it refuses to
overwrite existing files.

### Step 1: `jax_finance` package for Black and SABR

**Done (2026-11-03).** `jax_env/bin/pytest -v` from the root passes all 14
tests in about 12 s. The margins are wide: the worst Black error is 0.3% of
its tolerance, SABR vols match QuantLib to 4e-14, and calibration recovers
the true parameters to about 4e-15. Putting either SABR bug back fails
`test_sabr_vol_matches_quantlib` (32 and 44 of 68 rows). Notes:

- `sabr_implied_vol` has no separate ATM branch. Only z/x(z) is guarded:
  below |z| = 1e-4 it uses QuantLib's series 1 − ρz/2 − (3ρ² − 2)z²/12,
  and the direct formula is evaluated at a dummy z (the double where). At
  K = F the general formula then reduces exactly to Hagan's ATM formula.
  The `eps` clamps on quantities that are always positive are gone,
  including the `maximum(xz, eps)` behind the K > F bug.
- `calibrate_sabr` is the pure, jitted `scan` version and returns
  per-step histories; `calibrate_sabr_batch` is its `vmap`.
  `calibrate_sabr_single_expiry` and `calibrate_sabr_surface` keep their
  old signatures and print their verbose output from the histories.
- `calibrate_sabr_surface_from_df` stays in `sabr.py` and uses the
  DataFrame without importing Polars, so the core needs only `jax` and
  `numpy`.
- The root `pyproject.toml` sets `pythonpath = ["."]`, so the tests run
  without an install. `pip install -e .` also works; it needs network
  access for build isolation, because `jax_env` has no `setuptools`.
- **FX helper bugs, fixed in a follow-up commit.** Step 1 moved
  `fx_utils.py` unchanged, and the `SABR_JAX.py` EURUSD calibration showed
  it was wrong: ν barely moved from its starting value. In
  `jax_finance/fx/utils.py`:
  - `get_forwards` added `points/100` to spot, but the points are in pips
    (1/10000), so the 1y forward was 2.74 instead of 1.10.
  - `strike_from_delta` had every sign in its exponent reversed, so call
    strikes came out below spot and put strikes above. It's now the exact
    inverse of `delta_from_strike` (forward delta, premium unadjusted).
  - The "50-delta" ATM point was struck at spot. It's now the delta-neutral
    strike F·exp(σ²T/2).
  - `get_fx_market_data_for_calibration` passed ESTR as the domestic rate.
    For EURUSD, USD (SOFR) is domestic, as the forward points confirm.
  - Strikes came from the flat-rate forward while the calibration used the
    quoted forward, up to 0.76% apart at 5y. `FXVolSurface` and
    `build_vol_surfaces` now take an optional market forward, and the
    calibration helper passes the quoted ones.

  `tests/test_fx.py` has 10 tests, all of which fail on the old code. The
  EURUSD calibration now fits every tenor from 1w to 5y to within 0.05 vol
  points.

- Add `jax_finance/__init__.py`, which calls
  `jax.config.update("jax_enable_x64", True)`, the same pattern as
  `dv01/python/pv01_jax/__init__.py`.
- Add `jax_finance/black.py`.
  - `black_price`, `compute_greeks_single`, `batch_price_and_greeks`, moved
    from the current script.
  - Docstrings give Black-76: $V = e^{-rT}\,[\,\omega F N(\omega d_1) - \omega K N(\omega d_2)\,]$,
    with $d_{1,2} = \frac{\ln(F/K) \pm \tfrac12\sigma^2 T}{\sigma\sqrt T}$.
  - The docstring also states the Greek scaling conventions: vega per 1 vol
    point, theta per calendar day, and rho per 1% with F held fixed.
  - `FXOptionPricer` stays as a convenience wrapper.
- Add `jax_finance/sabr.py`.
  - `sabr_implied_vol`, with the Hagan formula in the docstring and a
    double-where safe denominator in the ATM branch.
  - Fix the two bugs Step 0.5 found: compute z/x(z) without clamping x(z)
    to be positive, so K > F works, and use (1−β)²/24, not ((1−β)/2)²/24,
    in D, the time adjustment and the ATM branch.
  - `sabr_params_from_raw` and the loss function.
  - Calibration rewritten as `jax.lax.scan` over Adam steps, so it compiles as
    one function, plus a `vmap` over expiries with equal strike counts.
  - The float32 casts are removed.
- Move `fx_utils.py` and `fx_data.py` to `jax_finance/fx/`, with re-export
  shims at the old paths.
- Reduce `JAX_black_option_pricing.py` and `SABR_JAX.py` to `__main__` demos
  that import from the package.
- **Add `tests/test_regression.py`**, which checks the new package against
  the Step 0.5 goldens. It needs the root `pyproject.toml` and `pytest`
  setup, so add those here rather than waiting for Step 4.
  - **Black:** price and all Greeks match `black_v0.csv` (the old code in
    float64) at `rtol=1e-10, atol=1e-14`. The `atol` covers the deep-OTM
    rows, where cancellation leaves the old formula about 6e-17 off. A
    changed formula, sign or Greek scaling convention fails.
  - **SABR implied vol:** matches `sabr_vol_ref.csv` (QuantLib) at
    `rtol=1e-10` on every row, including K = F and the strikes above F that
    the old code got wrong.
  - **SABR at K = F:** the vol is continuous across K = F (compare K = F
    with K = F(1 ± 1e-8)), and its gradient with respect to the parameters
    is finite there (the double-where fix).
  - **SABR calibration, against the old algorithm:** port the old Python
    `for` loop into the test file as a float64 reference, using the new
    `sabr_implied_vol`. Run both on `sabr_calib_input.csv` with the step
    count and learning rate from `sabr_calib_true.csv`, and assert the
    `scan` version matches the reference at `rtol≈1e-10`. Same algorithm,
    same steps, so they should agree almost exactly.
  - **SABR calibration, against the true parameters:** the `scan` version
    recovers `sabr_calib_true.csv` at `rtol≈1e-8`. A float64 prototype of
    the old algorithm got to about 1e-14.
  - **Shims:** the old import paths `fx_utils` and `fx_data` still expose
    the same public names. Skip with `pytest.importorskip("QuantLib")` when
    the `[fx]` extra isn't installed.
  - **Demos:** `JAX_black_option_pricing.py` and `SABR_JAX.py` run to
    completion under `subprocess` with exit code 0 (smoke test).
- Run `pytest tests/test_regression.py` before deleting the old code paths.
  Step 1 isn't done until it passes.

### Step 2: restructure `dv01/python`

These are moves and import changes, not rewrites. The goldens shouldn't
change.

**Done (2026-11-03).** `RUN_SLOW=1 python -m unittest discover -s tests
-t . -v` passes the same 94 tests before and after (64 and 65 minutes; logs
in `target/baseline-run.log` and `target/test-run.log`).
`git diff 065f4f0` on `tests/golden/` and `tests/jax_port/golden_jax/` is
empty. `import pv01_jax` loads no other `pv01_*` module and no third-party
package beyond jax, equinox and numpy. No package, example or benchmark
imports from `tests`. `python -m benchmarks.real_config` matches
`pre_refactor.md` to within run-to-run noise. A first run straight after the
test suite had the 100-swap batch rows 2–3× slow, but re-timing three times
gave 2.0–3.2 ms (linear) and 6.0–7.1 ms (dq), and a re-run is what's
committed. Notes:

- **`read_mqs` and `read_pv`** returned exactly what both old private
  readers returned, entries and order, on every golden and example-output
  CSV (9 of each).
- **`synthetic_problem`** converges in 3 Newton iterations with
  max |residual| ≤ 2e-16 for N ∈ {10, 20, 40, 80, 160} and all three
  interpolators. The test swap is 20y annual on a 30y curve, not one that
  spans the whole curve: Step 4's linear-interpolation check needs nodes
  beyond the last payment, and those buckets do come out exactly zero.
- **`bump_market_quote_pv01`** is within 2.5e-5 GBP/bp of
  `market_quote_pv01` on the real config for all three interpolators, and
  8e-5 on synthetic N = 20. That's 1bp central-difference truncation on
  buckets of up to about 1e4 GBP/bp.
- **Two Step 4 checks need rewording:**
  - The `vmap` and `lax.map` bump variants agree to about 1e-8 GBP/bp,
    not exactly, because batched and unbatched linear solves round
    differently. Use a tolerance around 1e-7.
  - The zero-rate buckets sum to the *zero-rate* parallel PV01 (a `jvp`
    of PV in z with a ones tangent), not to `parallel_pv01`, which shifts
    the *quotes*: 14,553 against 14,691 GBP/bp on synthetic N = 20.
- **The benchmark** moved to `benchmarks/real_config.py` with only import
  and output-path changes; Step 5 adds the JAX bump rows and
  `scaling.py`. The untracked `target/benchmark.md` was deleted after
  copying it to `benchmarks/results/pre_refactor.md`, the baseline for
  Step 5's timing comparison.

- **Move `tests/util/bump.py` to `pv01_pricer/bump.py`.** It holds two
  functions, and both move:
  - `bump_pv01`, imported by `pv01_jax/benchmark.py`,
    `pv01_examples/jax/calibration_interpolation_example_jax.py`,
    `tests/jax_port/test_phase5_risk.py` and
    `tests/test_reference_correctness.py`
  - `bumped_market_data`, imported by
    `tests/jax_port/test_phase3_residuals.py` and
    `tests/test_reference_correctness.py`

  `tests/util/bump.py` becomes a re-export of both names, so the tests keep
  working without edits. The example imports from `pv01_pricer.bump`
  directly, so no package or example imports from `tests` any more.
- **Move the shared test-swap constants** (`VALUATION_DATE`,
  `SWAP_TENOR_YEARS`, `SWAP_PERIOD_TO_START_MONTHS`, `SWAP_COUPON`,
  `SWAP_NOTIONAL`, `BP1`) into `pv01_examples/common.py`. The `multicurve2`
  example and all three JAX examples
  (`calibration_interpolation_example_jax.py`, `write_jax_goldens.py`,
  `zero_rate_pv01_example.py`) import them from there.
- **Put one golden-CSV reader in `pv01_examples/export_utils.py`.** There
  are two copies of the `_parse_mqs` reader today:
  - one private copy in `calibration_interpolation_example_jax.py`
  - one in `tests/test_calibration_interpolation_example.py` (alongside
    `_parse_pv`), which `test_phase1_jit_ordering.py`,
    `test_phase5_risk.py` and `test_phase5e_jax_example.py` import

  Add public `read_mqs` and `read_pv` next to `export_mqs` and `export_pv`.
  The example and the tests then use those, and the private copies go.
  Writer and reader then sit in one module, so the CSV format is defined in
  one place.
- **Stop re-exporting `resolve` from `pv01_jax/__init__.py`.** Remove the
  five `resolve` names from the imports and `__all__`. This is safe: every
  caller already imports `pv01_jax.resolve` directly, nothing uses
  `from pv01_jax import resolve_...`, and the scaffold test does only
  `import pv01_jax`. After this change, `import pv01_jax` loads only jax,
  equinox and numpy.
- **Add `bump_market_quote_pv01`** (vmap) and
  `bump_market_quote_pv01_sequential` (`lax.map`) to `pv01_jax/risk.py`.
- **Add `pv01_jax/synthetic.py`** with
  `synthetic_problem(n_quotes, interpolator) -> (CalibrationProblem, SwapArrays)`.
  - It builds two curves with N/2 nodes each: an OIS discount curve
    calibrated to an `OIS` block and a forward curve calibrated to an `IRS`
    block. These are the block types `pricers.residuals` already supports.
  - Tenors are evenly spaced out to 30 years. Year fractions are floats, and
    quotes come from a smooth upward-sloping curve so that Newton converges
    from `q0`.
  - It returns a test swap whose payments span the curve.
  - It uses only `pv01_jax.arrays`, with no Strata-layer imports.
- **Move `pv01_jax/benchmark.py`** into a new `dv01/python/benchmarks/`
  package (Step 5 splits it into two scripts).
  - `benchmarks/__init__.py` makes it a package. The scripts are run with
    `python -m benchmarks.real_config` and `python -m benchmarks.scaling`
    from `dv01/python`, so they don't need the `sys.path` insertion the
    example scripts use.
  - `benchmarks` is *not* added to the `pyproject.toml` package list. It's a
    script folder in the source tree, not part of the installed library.
- **Benchmark results are committed** to `dv01/python/benchmarks/results/`,
  not written to `target/`. The README quotes the headline numbers and
  embeds the chart, so the results the README refers to must be in version
  control. `target/` stays for regenerable example output. Delete
  `target/benchmark.md`.
- **Add `matplotlib`** as an optional `[bench]` dependency in
  `dv01/python/pyproject.toml`.

Layout after Step 2, with the files Steps 4 and 5 add (`[new]`, `[moved]`,
`[mod]`, `[gone]`; unmarked files are unchanged):

```
dv01/python/
├── pyproject.toml                          [mod]   [bench] extra
├── pv01_basics/  pv01_loader/  pv01_product/       input layer (unchanged)
├── pv01_market/                                    NumPy reference (frozen)
├── pv01_pricer/                                    NumPy reference (frozen)
│   └── bump.py                             [moved] from tests/util/bump.py
├── pv01_jax/                                       JAX kernel
│   ├── __init__.py                         [mod]   no resolve re-exports
│   ├── risk.py                             [mod]   + bump_market_quote_pv01{,_sequential}
│   ├── synthetic.py                        [new]
│   ├── resolve.py                                  Strata objects → arrays
│   ├── arrays.py  interp.py  curves.py  pricers.py  calibrate.py
│   └── benchmark.py                        [gone]  → benchmarks/
├── pv01_examples/
│   ├── common.py                           [new]   shared swap constants
│   ├── export_utils.py                     [mod]   + read_mqs, read_pv
│   ├── paths.py
│   ├── multicurve2/calibration_interpolation_example.py   [mod]
│   └── jax/{calibration_interpolation_example_jax,
│            write_jax_goldens, zero_rate_pv01_example}.py [mod]
├── benchmarks/                             [new]
│   ├── __init__.py  real_config.py  scaling.py
│   └── results/{real_config,scaling}.md
├── tests/
│   ├── util/bump.py                        [mod]   re-export
│   ├── test_calibration_interpolation_example.py  [mod]  uses read_mqs/read_pv
│   ├── test_phase5e_jax_example.py         [mod]   uses read_mqs/read_pv
│   └── jax_port/
│       ├── test_phase1_jit_ordering.py  test_phase5_risk.py   [mod]  read_mqs
│       └── test_phase7_bump.py  test_phase7_synthetic.py      [new]
└── target/
    └── benchmark.md                        [gone]
```

The calibration CSVs and quotes are not under `dv01/python`. They stay at
`dv01/examples/src/main/resources/example-calibration/` (see
`pv01_examples/paths.py`).

### Step 3: `JAX_PV01.py` demo at the repo root

**Done (2026-11-03).** `python JAX_PV01.py` prints each interpolator's
golden PV (linear −1,367,900.79, dq −1,366,361.40, ncs −1,368,040.96 GBP)
and the PV01 table. `-X importtime` shows no `pv01_pricer` or `tests.`
module. The two `test_phase7_workflow.py` tests pass, and the tests that
import `common.py` through the examples still pass (12 tests). Notes:

- `JAX_PV01.py`'s JIT note points to
  `dv01/python/benchmarks/results/real_config.md`, since Step 2 deleted
  `target/benchmark.md`.
- `pv01-python` is now editable-installed in `jax_env`
  (`pip install -e dv01/python --no-deps`), which `JAX_PV01.py` needs.
- `test_phase7_workflow.py` also checks that the PV01 is grouped by curve
  in the config's node order.

This gives PV01 the same shape as Black and SABR: a thin `__main__` script
at the root over a kernel package. It depends on Step 2 (`common.py`, and no
package importing `tests`), not on Step 1.

- **Add `pv01_jax/workflow.py`** with one function:

  ```python
  def price_and_risk(config, market_data, swap, valuation_date) -> dict:
      """Calibrate the curves, price the swap and take market-quote PV01.

      Returns {"pv": GBP, "pv01": {curve_name: {quote_label: GBP per bp}}}.
      """
  ```

  - It runs the sequence in today's
    `calibration_interpolation_example_jax.main()`: `resolve_swap`,
    `resolve_calibration_problem`, `pv_from_quotes` and `market_quote_pv01`.
    It then groups the PV01 by curve, using each curve's `param_slice` and
    `labels`.
  - It takes loaded objects, not file paths. That keeps `pv01_jax` free of
    any dependency on `pv01_examples`, and lets the API's in-memory curve
    definitions call it later.
  - It's the `price_and_risk` that [api-plan.md](../docs/api-plan.md) proposes, with
    the same return shape.
  - `pv01_jax/__init__.py` doesn't re-export it, for the same reason as
    `resolve` in Step 2: it imports the Strata input layer.
- **Add `load_example_inputs(settings_suffix)` to `pv01_examples/common.py`.**
  It returns `(config, market_data, swap)` for the GBP config, using
  `paths.py` and the swap constants.
- **Add `JAX_PV01.py`** at the repo root. For each interpolator it calls
  `load_example_inputs` and `price_and_risk`, then prints the PV and the
  PV01 table.
  - Its docstring says to run `pip install -e dv01/python` first.
  - It prints a note that the first run of each interpolator includes JIT
    compilation (2–20s in `target/benchmark.md`), so the run doesn't look
    stuck.
  - It imports nothing from `pv01_pricer` or `tests`. The comparison against
    the hand-coded bump stays in the dv01 example.
- **It imports `pv01_jax` directly, not through `jax_finance`.** A
  `jax_finance/pv01.py` re-export would make `jax_finance` depend on a
  package in the other Python root, which goes against Decision 3. A
  `pyproject.toml` also can't reliably declare a dependency on the relative
  path `dv01/python`.
- **The dv01 JAX example keeps its own `main()`.** Its timing needs the
  `CalibrationProblem`, which `price_and_risk` doesn't return.
- `paths.py` finds the CSVs relative to the dv01 source tree, and the CSVs
  aren't packaged. `JAX_PV01.py` therefore needs the editable install that
  the Quickstart uses.
- **Test:** `dv01/python/tests/jax_port/test_phase7_workflow.py` checks
  `price_and_risk` against the goldens in `tests/golden/` for all three
  interpolators. It uses the same `delta=0.01` as
  `test_phase5e_jax_example.py`. Those golden CSVs are back under version
  control from Step 0.25.

### Step 4: tests

**Done (2026-11-03).** The root suite passes 50 tests in 72 s
(`test_black.py` 5, `test_sabr.py` 21, plus Step 1's regression tests and
the FX tests). The new dv01 modules pass 6 tests: `test_phase7_bump.py` 2
and `test_phase7_synthetic.py` 4. Where the measurements differed from the
bullets below:

- **Black:** the AD Greeks match the closed forms (computed with
  `math.erf`, so independent of JAX) at `rtol=1e-10` on 120 grid points per
  option type. The jit and vmap check needs `atol=1e-15`: XLA fuses
  differently under jit, so a deep-OTM price of 1.5e-21 differs by 4e-33.
- **`check_grads`:** it steps every input by the same absolute amount,
  which is coarse for F = 0.03 or α = 0.01. The test therefore perturbs
  each input relative to its own size and holds β fixed: the clip to
  [0, 1] makes β = 0 and β = 1 one-sided points.
- **Finite at K = F:** the test checks the value, gradient and Hessian.
  Removing `z_safe` in a scratch copy leaves the value finite but makes the
  gradient NaN, so the test catches the double-where trap.
- **Bump against AD:** real config within `atol=1e-3` (2.5e-5 observed).
  On synthetic N = 20 the error scales with the bucket, reaching 1.03e-3
  on a 12,500 GBP/bp natural-cubic-spline bucket (8e-8 relative), so that
  check uses `rtol=1e-6, atol=1e-3`. Halving the step cuts the error by
  4.00 for every interpolator, which the test asserts: it's pure h²
  truncation.
- **`vmap` against `lax.map`:** `atol=1e-7`, not exact (see Step 2).
- **Bucket sum:** compared with the zero-rate parallel `jvp`, not with
  `parallel_pv01` (see Step 2).
- **Linear zero buckets:** "beyond the last payment" means after the first
  node at or past it. That node brackets the 20y payment, so it's nonzero;
  the test checks both.

Root `tests/` (pytest), for the Black and SABR kernels:

- `tests/test_black.py`
  - AD Greeks against the closed forms
    ($\Delta = e^{-rT}N(d_1)$, $\Gamma = e^{-rT}n(d_1)/(F\sigma\sqrt T)$, vega,
    rho, theta), at `rtol=1e-10`.
  - Put-call parity.
  - `jit` and `vmap` give the same result as scalar calls.
- `tests/test_sabr.py`
  - `jax.test_util.check_grads`.
  - Gradients are finite at K = F (the regression test for gap 4).
  - The calibration recovers synthetic parameters.
  - The `scan` and `vmap` calibration matches the per-expiry calibration.
- The root `pyproject.toml` (which defines the `jax_finance` package and
  `[tool.pytest.ini_options] testpaths = ["tests"]`) and
  `tests/test_regression.py` already exist from Step 1. The tests here sit
  beside them: the regression tests show the code is unchanged, and these
  show it's correct.

`dv01/python/tests/jax_port/` (unittest, alongside the existing suite):

- `test_phase7_bump.py`
  - `bump_market_quote_pv01` matches `market_quote_pv01` on the real config
    for all three interpolators. The tolerance comes from the truncation
    error of a 1bp central difference, and the test docstring explains it.
    Use the existing opt-in bump comparison's tolerance as the starting
    point.
  - The `vmap` and `lax.map` variants agree exactly.
- `test_phase7_synthetic.py`
  - The synthetic problem calibrates (max |residual| < 1e-12) for N ∈ {10,
    40, 160} and all three interpolators.
  - These structural checks need no reference implementation:
    - the zero-rate buckets sum to the parallel PV01 from `jvp` with a ones
      tangent
    - with linear interpolation, the zero-rate buckets for nodes beyond the
      swap's last payment date are exactly zero
    - `grad`, `vmap(jvp)` over the identity, and bump agree
- The existing closed-form-reference tests (`test_phase2_interp.py`,
  `test_phase3_risk.py`) and the goldens continue unchanged.

### Step 5: benchmark

**Done (2026-11-04).** Both scripts ran and their outputs are committed:
`benchmarks/results/real_config.md`, `benchmarks/results/scaling.md` and
`docs/assets/dv01_scaling.png`. The timing method is shared in
`benchmarks/timing.py`. Results and notes:

- **Real config, linear:** AD 0.70 ms; JAX bump 3.1 ms with `vmap`
  (4.5× AD) and 10.5 ms with `lax.map` (15×); legacy NumPy bump 18.4 s
  (about 26,000×). Most of the gap to the legacy bump is the
  implementation, not AD itself, which is what Decision 2 set out to show.
  AD rows match `pre_refactor.md` to within noise; the 100-swap batch row
  is noisy run to run (2.0–5.2 ms on linear).
- **`real_config.py` keeps the `cross_gamma` and 100-swap batch rows** as
  well as the rows listed below, and labels the legacy row
  `bump_pv01 (legacy NumPy)`.
- **Scaling (linear interpolation, the default; `--interpolator` picks
  another):** from N = 10 to 160, AD gets 23× slower (0.32 → 7.3 ms), the
  `lax.map` bump 657× and the `vmap` bump 3,836×. The `lax.map` bump/AD
  ratio grows from 8 to 242, roughly linear in N (about N^1.2) as the
  Verification expects. The `vmap` ratio grows faster (2 → 314): `vmap`
  wins at small N but is slower than `lax.map` at N = 160 (2.3 s against
  1.8 s), presumably from the memory of 320 batched Newton solves.
- **The chart** follows the dataviz method: categorical slots 1–4, run
  through the palette validator (all checks pass; the contrast warning on
  aqua and yellow is relieved by direct end labels, a legend and the
  table), solid lines for AD and dashed for bumping as a second cue, plain
  tick labels on the log axes, and a title computed from the results.
- **`matplotlib` 3.11.2 is installed in `jax_env`.**

The benchmark moves to `dv01/python/benchmarks/` and has two scripts.

- **`real_config.py`** writes `benchmarks/results/real_config.md`: a table
  for the 19-quote GBP config × {linear, dq, ncs}, with these rows:
  - `market_quote_pv01` (AD)
  - `parallel_pv01` (AD, `jvp`)
  - `bump_market_quote_pv01` (JAX, `vmap`)
  - `bump_market_quote_pv01_sequential` (JAX, `lax.map`)
  - legacy NumPy `bump_pv01`, with NCS still behind `--slow`
- **`scaling.py`** sweeps N ∈ {10, 20, 40, 80, 160} quotes using
  `synthetic_problem`. It times `market_quote_pv01`, `parallel_pv01` and the
  two JAX bump variants.
  - It writes a log-log matplotlib chart to `docs/assets/dv01_scaling.png`
    and a table to `benchmarks/results/scaling.md`.
- Both scripts report compile time separately and give the median over 20
  runs after warm-up, using `block_until_ready`. This is the same method as
  today's benchmark.

### Step 6: README

**Done (2026-11-04).** The root README has the four sections below, and
every relative link in it resolves to a tracked file. Notes:

- **The headline was checked before use.** On the real config AD takes
  1.4 calibrations (0.71 ms against 0.51 ms for one `pv_from_quotes`), and
  1.3 on synthetic N = 40 and 160. The `lax.map` bump takes 24, 44 and 297
  calibrations' worth at N = 19, 40 and 160, approaching 2N (38, 80, 320)
  as N grows. The README says "1.3–1.4 calibrations" and gives the 297 of
  320.
- **The architecture section** derives dz/dq = (∂par/∂z)⁻¹ from r = par − q,
  which is why AD costs one extra solve.
- **Roadmap status:** Part 1 is tracked and published. Part 2 has front
  matter dated 2026-09-29 but is untracked, so it's listed as a draft and
  not linked. The implementation and SABR drafts have no front matter, so
  their titles are working titles. The PV01 row points to
  `docs/pv01_jax.md`.
- **Quickstart** uses `pip install -e ".[fx,dev]"` at the root (the FX
  helpers and pytest) as well as `pip install -e "dv01/python[bench]"`, and
  runs all three demos.

Rewrite the root README with these sections:

- **Architecture:** kernel → transform → risk, with the formulas. It points
  to `pv01_jax` as the DV01 kernel and walks through `pv_from_quotes` →
  `market_quote_pv01` as the minimal path.
- **Benchmark result:** embed `docs/assets/dv01_scaling.png`, with the
  headline "AD costs about one extra calibration; bump-and-recalibrate costs
  2N". It states that the main comparison runs on the same JAX kernel, and
  that the NumPy row is legacy context.
- **Roadmap:** a table mapping each module to a blog chapter, with the
  chapters numbered in publication order:
  - Part 1 → `jax_finance/black.py` concepts
  - Part 2 → JVP/VJP in `pv01_jax/risk.py`
  - the implementation draft → `jax_finance/black.py`
  - the SABR draft → `jax_finance/sabr.py`
  - the PV01 port → `dv01/python/pv01_jax`, with `JAX_PV01.py` as its demo
- **Quickstart:** create the venv, run `pip install -e .` at the root and
  `pip install -e dv01/python[bench]`, then run `pytest`, the dv01 unittest
  suite, `python JAX_PV01.py` and the two benchmarks.

## Files

- **New:**
  - `jax_finance/{__init__,black,sabr}.py` and `jax_finance/fx/`
  - `tests/{conftest,test_black,test_sabr,test_regression}.py`
  - `tests/golden/capture_v0.py`, `tests/golden/v0_manifest.txt` and the
    goldens `tests/golden/{black_v0,sabr_vol_ref,sabr_calib_input,sabr_calib_true}.csv`
  - root `pyproject.toml`
  - `JAX_PV01.py`
  - `dv01/python/pv01_jax/{synthetic,workflow}.py`
  - `dv01/python/pv01_pricer/bump.py`
  - `dv01/python/pv01_examples/common.py`
  - `dv01/python/benchmarks/{__init__,real_config,scaling}.py` and
    `dv01/python/benchmarks/results/`
  - `dv01/python/tests/jax_port/{test_phase7_bump,test_phase7_synthetic,test_phase7_workflow}.py`
  - `docs/assets/dv01_scaling.png`
- **Modified:**
  - `JAX_black_option_pricing.py`, `SABR_JAX.py`, `fx_utils.py`, `fx_data.py`
    (become shims or demos)
  - `README.md`, `requirements.txt`
  - `.gitignore` (the rule excluding `dv01/python/tests/golden/*.csv` is
    removed; done in Step 0.25)
  - `dv01/python/tests/golden/*.csv` (tracked again from the local
    2026-10-05 copies; done in Step 0.25)
  - `dv01/python/pv01_jax/{__init__,risk}.py`
  - `dv01/python/pv01_examples/` (import constants from `common.py`, and
    `bump_pv01` from `pv01_pricer.bump`; `export_utils.py` gains `read_mqs`
    and `read_pv`)
  - `dv01/python/tests/util/bump.py` (becomes a re-export of `bump_pv01` and
    `bumped_market_data`)
  - `dv01/python/tests/` modules that use the golden-CSV readers:
    `test_calibration_interpolation_example.py`,
    `test_phase5e_jax_example.py`, `jax_port/test_phase1_jit_ordering.py`,
    `jax_port/test_phase5_risk.py`
  - `dv01/python/pyproject.toml` (`[bench]` extra)
- **Removed:**
  - `dv01/python/pv01_jax/benchmark.py`, which moves to `benchmarks/`
  - `dv01/python/target/benchmark.md`, replaced by `benchmarks/results/`

## Verification

- `jax_env/bin/pytest -q` from the repo root: all tests pass.
- `cd dv01/python && python -m unittest discover -s tests -t .`: the existing
  suite and the new phase 7 tests pass, and the goldens are unchanged.
- `python -c "import pv01_jax, sys; print([m for m in sys.modules if m.startswith('pv01_') and not m.startswith('pv01_jax')])"`
  prints an empty list.
- `git ls-files dv01/python/tests/golden` lists every dv01 golden, and
  `git diff 065f4f0 -- dv01/python/tests/golden/` is empty: nothing since
  Step 0.25 has changed them.
- `jax_env/bin/pytest -q tests/test_regression.py` passes: Black matches
  the old code's float64 golden, SABR vols match QuantLib, the `scan`
  calibration matches the ported loop and recovers the true parameters, the
  shims expose the old names, and both demo scripts run. This replaces
  comparing demo output by eye.
- `git ls-files tests/golden` lists the four golden CSVs, the manifest and
  `capture_v0.py`.
- `python JAX_PV01.py` prints the golden PV for each interpolator, and
  `python -X importtime JAX_PV01.py 2>&1 | grep -c "pv01_pricer\|tests\."`
  prints 0.
- `grep -rn "from tests" dv01/python/pv01_* dv01/python/benchmarks` finds
  nothing: no package, example or benchmark imports test code.
- `cd dv01/python && python -m benchmarks.real_config` reproduces the AD
  timings in today's `target/benchmark.md` to within run-to-run noise, and
  the JAX bump rows include NCS. Save a copy of `target/benchmark.md` before
  deleting it, for this comparison.
- `cd dv01/python && python -m benchmarks.scaling` writes the PNG. Check
  that the ratio of bump time to AD time grows roughly linearly in N.

### Monitoring long runs

The slow runs are the `RUN_SLOW` bump-and-recalibrate test in
`tests/jax_port/test_phase5_risk.py`, the full dv01 suite,
`benchmarks.real_config --slow` (the NCS NumPy bump) and the
`benchmarks.scaling` sweep up to N=160. Run them with per-test output
written to a log under `dv01/python/target/`, which is untracked:

```bash
cd dv01/python
RUN_SLOW=1 PYTHONUNBUFFERED=1 ../../jax_env/bin/python -m pytest tests -v -rA --durations=20 \
  2>&1 | tee target/test-run.log
```

and follow it from a second terminal with
`tail -f dv01/python/target/test-run.log`.

- `-v` prints a line as each test finishes, so the log shows progress.
- `--durations=20` lists the slowest tests at the end. Compare it before and
  after a step to catch a test that has got much slower.
- `-rA` summarises every outcome, including skips, so the log shows whether
  the `RUN_SLOW` test actually ran.
- `PYTHONUNBUFFERED=1` makes output appear immediately instead of in bursts.
  It matters most for the benchmark scripts, which print with plain `print`.
- Optional: `--junitxml=target/results.xml` keeps a machine-readable record,
  and `pytest-timeout` with `--timeout=600` fails a hung test instead of
  stalling the run.
- Until the TODO below ports the dv01 suite to pytest, the unittest runner
  gives the same live progress:
  `RUN_SLOW=1 python -m unittest discover -s tests -t . -v 2>&1 | tee target/test-run.log`.
- Benchmarks:
  `cd dv01/python && PYTHONUNBUFFERED=1 python -m benchmarks.scaling 2>&1 | tee target/scaling.log`.
  The first run of each interpolator includes JIT compilation (2–20s), so a
  pause there doesn't mean the run is stuck.

With `-v` the log names each test when it starts and adds the outcome when
it finishes, so `tail -f` shows which test is running. It doesn't show:

- **Progress inside one long test.** The `RUN_SLOW` test is a single
  unfinished line until it completes. pytest captures what tests print;
  add `-s` to send it to the log instead.
- **Progress during collection.** pytest imports JAX and every test module
  before running anything, so the log can sit on `collecting ...` for a
  while at the start.
- **The root-level tests.** The command above runs only
  `dv01/python/tests`, and the root `testpaths = ["tests"]` means a root run
  doesn't pick up the dv01 suite either. Run the root suite, including
  `tests/test_regression.py`, separately from the repo root:
  `jax_env/bin/pytest -v -rA --durations=20 2>&1 | tee test-run.log`.
- **Individual subtests.** The unittest `subTest` loops don't get their own
  progress lines under pytest, though failures inside them are still
  reported.

## TODO (after this refactor)

- [ ] **Port the dv01 unittest suite to pytest.** The root tests use pytest,
  and the dv01 suite uses unittest, so the repo has two test styles. This is
  out of scope for the refactor above, which keeps the dv01 tests unchanged
  so the goldens prove Step 2 changed nothing.
  - **First, switch the runner only.** pytest runs unittest tests unchanged,
    so check that `cd dv01/python && pytest tests` passes the same tests as
    `python -m unittest discover -s tests -t .`, and update the Quickstart
    and Verification commands to use it.
  - **Then convert module by module:** `setUpClass` → fixtures in a
    `dv01/python/tests/conftest.py`, `subTest` loops → `parametrize`,
    `self.assert*` → plain `assert` or `np.testing.assert_allclose`, and
    `unittest.skipUnless` (the opt-in bump comparison) → a pytest marker or
    `skipif`.
  - **Keep every tolerance exactly as it is** (`atol=1e-12`, `delta=1e-6`,
    `delta=0.01` and so on), and don't touch the golden CSVs. A port that
    loosens a tolerance isn't a port.
  - **Done when** the test count matches the unittest run, the suite passes
    under pytest, and no file under `dv01/python/tests` imports `unittest`.
- [ ] **Decide whether to move `dv01/python` up to `dv01/`.** Not part of
  this refactor: combining a directory move with Step 2's file moves would
  make the "goldens unchanged" check much harder to review.
  - **First, check what else is in `dv01/`** (`ls dv01`). The calibration
    CSVs sit in a Maven-style path
    (`dv01/examples/src/main/resources/example-calibration/`), which suggests
    the Strata Java example lives there too.
    - **If Java code is there: don't move.** The Java and Python roots would
      mix, and Maven's `dv01/target/` would collide with the Python
      examples' `target/`, so `mvn clean` could delete Python output.
    - **If `dv01/` holds only `python/` and the resources folder:** the move
      is reasonable. Consider moving the calibration CSVs into the Python
      tree at the same time, so the data and its loader (`paths.py`) live
      together.
  - **If moving, make it its own commit:** `git mv` plus path fixes, and
    nothing else. Run it after the refactor, once the new tests exist to
    catch breakage.
  - **Paths to update:** `paths.py`, the `sys.path` insertions in the
    example scripts, `.gitignore` entries, the Quickstart's `pip install -e dv01/python[bench]`,
    the Verification commands, `docs/pv01_jax.md`, and any CI config.
  - **Done when** the full dv01 suite and the root suite pass, both
    benchmarks run, `git log --follow` still traces the goldens' history,
    and `grep -rn "dv01/python" --exclude-dir=.git .` finds no stale
    references, other than in this plan's history.
