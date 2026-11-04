"""
One-off capture of the Black and SABR goldens for the Step 1 regression
tests (Step 0.5 of notes/refactor-plan.md).

It writes:

    black_v0.csv          price and Greeks from the pre-Step-1 Black code
    sabr_vol_ref.csv      Hagan (2002) implied vols from QuantLib
    sabr_calib_input.csv  a noise-free smile generated from known parameters
    sabr_calib_true.csv   those parameters, with the Adam step count and rate
    v0_manifest.txt       commit, versions, date and run settings

Every value is written with `repr` of its float64 value, so the CSVs
round-trip exactly. Never write with a fixed number of decimal places.

Decisions (2026-11-03), taken after a first float32 capture showed that the
plan's original goldens couldn't support the Step 1 tests:

1. The old SABR implied vol has two bugs, and Step 1 fixes them, so the SABR
   goldens come from QuantLib, not from the old code.
   - Every K > F returns the 1e-12 floor: `z / jnp.maximum(xz, eps)` in
     `SABR_JAX.sabr_implied_vol` replaces x(z), which is negative whenever
     z is (K > F), with eps.
   - The (1-beta)^2/24 coefficients in D, the time adjustment and the ATM
     branch are written `pow_ * pow_ / 24` with `pow_ = (1-beta)/2`, a
     factor of 4 too small. The error reaches 1.2% in the beta = 0 wings.
   With both fixed, the Hagan formula agrees with QuantLib's
   `sabrVolatility` to 6.5e-16 on the vol grid. `sabr_vol_ref.csv` includes
   K = F, which the old ATM branch got wrong too.
2. The Black golden is the old code run with x64 on. `black_price` and
   `compute_greeks_single` have no hard float32 casts, so this is the old
   formula in float64. A float32 capture differed from it by up to 2e-4
   relative on ordinary values (gamma, vega, theta) and 0.65 on tiny ones,
   so it couldn't support a tight tolerance. The float64 golden agrees with
   an omega-form rewrite to 6e-17 absolute, so Step 1 can check it at
   `rtol=1e-10, atol=1e-14`.
3. The calibration smile is generated from known parameters by QuantLib,
   at realistic vols (about 10% ATM), with no noise. The first capture used
   the SABR_JAX.py demo parameters (2.5% ATM vol) and the buggy formula, so
   every strike above F was noise around zero and the fit was meaningless.
   Step 1 checks that calibration recovers `sabr_calib_true.csv`. The old
   Adam algorithm on the corrected formula recovers it to about 1e-14 in
   float64 after 3000 steps at lr 8e-3 (only about 1e-4 after 1200).

Only `black_v0.csv` depends on the old code. It reproduces only on the
pre-Step-1 commit recorded in the manifest: Step 1 moves `compute_greeks_single`
into `jax_finance` and reduces the script to a demo, so running this
afterwards would capture the new code. The script has since moved to
examples/black.py, so the import below now fails, which is the safe outcome:
to reproduce the golden, check out that commit. Do not re-run it here. It
also refuses to overwrite existing output files.
"""

import csv
import datetime
import platform
import subprocess
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jaxlib
import numpy as np
import QuantLib as ql

GOLDEN_DIR = Path(__file__).resolve().parent
ROOT = GOLDEN_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from JAX_black_option_pricing import compute_greeks_single

OUTPUTS = (
    "black_v0.csv",
    "sabr_vol_ref.csv",
    "sabr_calib_input.csv",
    "sabr_calib_true.csv",
    "v0_manifest.txt",
)

# Black grid: moneyness K/F gives ITM, ATM and OTM for both calls and puts.
BLACK_F = 1.085
BLACK_MONEYNESS = (0.95, 1.0, 1.05)
BLACK_T = (0.05, 0.25, 2.0)
BLACK_SIGMA = (0.05, 0.30)
BLACK_R = (0.0, 0.05)

# SABR vol grid: (F, T, alpha, beta, rho, nu) per parameter set, each
# evaluated at the strikes F * m, including K = F.
SABR_SETS = (
    (100.0, 0.5, 0.25, 0.5, -0.3, 0.6),  # the SABR_JAX.py demo parameters
    (1.085, 0.25, 0.10, 1.0, -0.2, 0.8),  # lognormal, FX-like
    (0.03, 2.0, 0.01, 0.0, 0.3, 0.4),  # normal, rates-like
    (1.085, 1.0, 0.15, 0.7, 0.5, 1.5),  # positive rho, high vol-of-vol
)
SABR_MONEYNESS = (
    0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 1.0,
    1.0001, 1.001, 1.01, 1.05, 1.1, 1.25, 1.5, 2.0,
)

# Calibration smile: (expiry, T, F, alpha, rho, nu) with beta fixed. The 13
# strikes are F * exp(k * atm_vol * sqrt(T)) for k in [-2.5, 2.5], so the
# middle strike is K = F.
CALIB_EXPIRIES = (
    ("3M", 0.25, 100.0, 1.00, -0.30, 0.80),
    ("6M", 0.5, 102.5, 0.95, -0.25, 0.60),
    ("1Y", 1.0, 105.0, 0.90, -0.20, 0.45),
)
CALIB_BETA = 0.5
CALIB_K_SD = np.linspace(-2.5, 2.5, 13)
CALIB_STEPS = 3000
CALIB_LR = 8e-3


def fmt(v):
    """Full-precision text for a CSV cell."""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(np.float64(v)))


def write_csv(name, columns, rows):
    with open(GOLDEN_DIR / name, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([fmt(row[c]) for c in columns])
    print(f"wrote {name}: {len(rows)} rows")


def ql_sabr_vol(F, K, T, alpha, beta, rho, nu):
    # QuantLib's argument order is (strike, forward, expiry, alpha, beta, nu, rho).
    return ql.sabrVolatility(K, F, T, alpha, beta, nu, rho)


def capture_black():
    rows = []
    for option_type in ("call", "put"):
        for m in BLACK_MONEYNESS:
            for T in BLACK_T:
                for sigma in BLACK_SIGMA:
                    for r in BLACK_R:
                        F, K = BLACK_F, BLACK_F * m
                        g = compute_greeks_single(
                            F, K, T, r, sigma, option_type == "call"
                        )
                        rows.append(dict(
                            option_type=option_type, F=F, K=K, T=T,
                            sigma=sigma, r=r,
                            **{k: np.asarray(v) for k, v in g.items()},
                        ))
    columns = ["option_type", "F", "K", "T", "sigma", "r",
               "price", "delta", "gamma", "vega", "theta", "rho"]
    write_csv("black_v0.csv", columns, rows)
    return len(rows)


def capture_sabr_vol():
    rows = []
    for F, T, alpha, beta, rho, nu in SABR_SETS:
        for m in SABR_MONEYNESS:
            K = F * m
            rows.append(dict(F=F, K=K, T=T, alpha=alpha, beta=beta, rho=rho,
                             nu=nu, vol=ql_sabr_vol(F, K, T, alpha, beta, rho, nu)))
    columns = ["F", "K", "T", "alpha", "beta", "rho", "nu", "vol"]
    write_csv("sabr_vol_ref.csv", columns, rows)
    return len(rows)


def capture_sabr_calib():
    input_rows = []
    true_rows = []
    for expiry, T, F, alpha, rho, nu in CALIB_EXPIRIES:
        atm_vol = alpha / F ** (1.0 - CALIB_BETA)
        for k in CALIB_K_SD:
            K = F * float(np.exp(k * atm_vol * np.sqrt(T)))
            input_rows.append(dict(
                expiry=expiry, T=T, F=F, K=K,
                market_vol=ql_sabr_vol(F, K, T, alpha, CALIB_BETA, rho, nu),
            ))
        true_rows.append(dict(expiry=expiry, alpha=alpha, beta=CALIB_BETA,
                              rho=rho, nu=nu, n_steps=CALIB_STEPS,
                              learning_rate=CALIB_LR))
    write_csv("sabr_calib_input.csv",
              ["expiry", "T", "F", "K", "market_vol"], input_rows)
    write_csv("sabr_calib_true.csv",
              ["expiry", "alpha", "beta", "rho", "nu", "n_steps",
               "learning_rate"], true_rows)
    return len(input_rows), len(true_rows)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def write_manifest(counts):
    dirty = git("status", "--porcelain", "--", "JAX_black_option_pricing.py")
    lines = [
        f"commit: {git('rev-parse', 'HEAD')}",
        f"JAX_black_option_pricing.py clean at commit: {'no' if dirty else 'yes'}",
        f"capture date: {datetime.date.today().isoformat()}",
        f"python: {platform.python_version()}",
        f"jax: {jax.__version__}",
        f"jaxlib: {jaxlib.__version__}",
        f"numpy: {np.__version__}",
        f"QuantLib: {ql.__version__}",
        f"platform: {platform.platform()}",
        f"jax backend: {jax.default_backend()}",
        f"jax_enable_x64: {jax.config.jax_enable_x64}",
        f"black_v0.csv rows: {counts['black']} (old code, float64)",
        f"sabr_vol_ref.csv rows: {counts['sabr_vol']} (QuantLib sabrVolatility)",
        f"sabr_calib_input.csv rows: {counts['calib_input']} (QuantLib, no noise)",
        f"sabr_calib_true.csv rows: {counts['calib']}",
        f"calibration settings: beta={CALIB_BETA}, steps={CALIB_STEPS}, "
        f"lr={CALIB_LR}, vega_weight=True",
    ]
    (GOLDEN_DIR / "v0_manifest.txt").write_text("\n".join(lines) + "\n")
    print("wrote v0_manifest.txt")


def main():
    existing = [n for n in OUTPUTS if (GOLDEN_DIR / n).exists()]
    if existing:
        sys.exit(f"refusing to overwrite existing goldens: {existing}")
    counts = {"black": capture_black(), "sabr_vol": capture_sabr_vol()}
    counts["calib_input"], counts["calib"] = capture_sabr_calib()
    write_manifest(counts)


if __name__ == "__main__":
    main()
