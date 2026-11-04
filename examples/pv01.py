"""
Swap PV and market-quote PV01 by automatic differentiation through curve
calibration: a demo.

For each of three curve interpolators (linear, double quadratic, natural
cubic spline) it calibrates the GBP OIS discount and LIBOR 6M forward curves
from Strata's CalibrationInterpolationExample configuration, prices an 8y
swap, and takes the PV01 to every market quote in one reverse-mode pass
through the calibration. The kernel is `pv01_jax` in jax_pv01/ (see
docs/pv01_jax.md).

Setup: `pip install -e jax_pv01` first. The example CSVs are read from
the jax_pv01 source tree, so the install must be editable.
"""

from time import perf_counter

from pv01_examples.common import VALUATION_DATE, load_example_inputs
from pv01_examples.paths import SETTINGS_SUFFIXES
from pv01_jax.workflow import price_and_risk


def main() -> None:
    print("Market-quote PV01 of an 8y GBP swap, by AD through curve calibration")
    print("The first run of each interpolator includes JIT compilation (2-20s; see")
    print("jax_pv01/benchmarks/results/real_config.md), so it can pause before printing.")

    for suffix in SETTINGS_SUFFIXES:
        config, market_data, swap = load_example_inputs(suffix)
        start = perf_counter()
        result = price_and_risk(config, market_data, swap, VALUATION_DATE)
        elapsed = perf_counter() - start

        print(f"\n{suffix.strip('-')}: PV = {result['pv']:,.2f} GBP  "
              f"({elapsed:.1f}s including compilation)")
        print(f"  {'Curve':<18} {'Quote':<16} {'PV01 (GBP/bp)':>14}")
        for curve_name, buckets in result["pv01"].items():
            for label, value in buckets.items():
                print(f"  {curve_name:<18} {label:<16} {value:>14,.2f}")
        total = sum(v for buckets in result["pv01"].values() for v in buckets.values())
        print(f"  {'Total':<35} {total:>14,.2f}")


if __name__ == "__main__":
    main()
