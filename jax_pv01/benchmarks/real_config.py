"""Benchmark JAX market-quote risk functions against bump-and-recalibrate.

On the 19-quote GBP config for each interpolator: AD (reverse mode and
``jvp``), bump-and-recalibrate on the same JAX kernel (``vmap`` and
``lax.map``), cross-gamma, a 100-swap batch, and the legacy NumPy
``bump_pv01`` (NCS only with ``--slow``). JAX rows report compile time
separately and the median of 20 runs after warm-up.

Run from jax_pv01: ``python -m benchmarks.real_config [--slow]``. Writes
``benchmarks/results/real_config.md``.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import jax.numpy as jnp

from benchmarks.timing import time_jitted
from pv01_examples.common import (
    SWAP_COUPON,
    SWAP_NOTIONAL,
    SWAP_PERIOD_TO_START_MONTHS,
    SWAP_TENOR_YEARS,
    VALUATION_DATE,
)
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, PROJECT_DIR, SETTINGS_SUFFIXES
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_jax.risk import (
    batch_market_quote_pv01,
    bump_market_quote_pv01,
    bump_market_quote_pv01_sequential,
    cross_gamma,
    market_quote_pv01,
    parallel_pv01,
    stack_trades,
)
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_pricer.bump import bump_pv01
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import create_test_swap

OUTPUT_PATH = PROJECT_DIR / "benchmarks" / "results" / "real_config.md"


def _load_inputs(settings_suffix: str):
    config_dir = PATH_CONFIG / CONFIG_STR
    config = load_calibration(
        config_dir / f"{CONFIG_STR}-group.csv",
        config_dir / f"{CONFIG_STR}{settings_suffix}.csv",
        config_dir / f"{CONFIG_STR}-nodes.csv",
    )[CONFIG_STR]
    market_data = load_market_data(
        VALUATION_DATE,
        PATH_QUOTES / "MARKET-QUOTES-GBP-20160801.csv",
    )
    problem = resolve_calibration_problem(config, market_data, VALUATION_DATE)
    swap = create_test_swap(
        VALUATION_DATE,
        SWAP_PERIOD_TO_START_MONTHS,
        SWAP_TENOR_YEARS,
        BuySell.BUY,
        SWAP_NOTIONAL,
        SWAP_COUPON,
    )
    swap_arrays = resolve_swap(swap, VALUATION_DATE)
    portfolio = [
        resolve_swap(
            create_test_swap(
                VALUATION_DATE,
                SWAP_PERIOD_TO_START_MONTHS,
                SWAP_TENOR_YEARS,
                BuySell.BUY,
                SWAP_NOTIONAL,
                float(coupon),
            ),
            VALUATION_DATE,
        )
        for coupon in jnp.linspace(0.01, 0.04, 100)
    ]
    return config, market_data, swap, problem, swap_arrays, stack_trades(portfolio)


def _time_hand_coded_bump(config, market_data, swap) -> float:
    start = perf_counter()
    bump_pv01(config, market_data, swap, VALUATION_DATE)
    return perf_counter() - start


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.6f}s"


def run_benchmark(include_slow: bool = False) -> str:
    rows: list[tuple[str, str, str, str]] = []
    for suffix in SETTINGS_SUFFIXES:
        config, market_data, swap, problem, swap_arrays, swaps_stacked = _load_inputs(
            suffix
        )
        quotes = jnp.asarray(problem.q0)
        benchmarks = (
            (
                "market_quote_pv01",
                lambda q: market_quote_pv01(problem, swap_arrays, q),
                (quotes,),
            ),
            (
                "parallel_pv01",
                lambda q: parallel_pv01(problem, swap_arrays, q),
                (quotes,),
            ),
            (
                "bump_market_quote_pv01 (JAX, vmap)",
                lambda q: bump_market_quote_pv01(problem, swap_arrays, q),
                (quotes,),
            ),
            (
                "bump_market_quote_pv01_sequential (JAX, lax.map)",
                lambda q: bump_market_quote_pv01_sequential(problem, swap_arrays, q),
                (quotes,),
            ),
            (
                "cross_gamma",
                lambda q: cross_gamma(problem, swap_arrays, q),
                (quotes,),
            ),
            (
                "batch_market_quote_pv01 (100 swaps)",
                lambda q: batch_market_quote_pv01(problem, swaps_stacked, q),
                (quotes,),
            ),
        )
        for name, function, arguments in benchmarks:
            compile_seconds, median_seconds = time_jitted(function, arguments)
            rows.append(
                (
                    suffix,
                    name,
                    _format_seconds(compile_seconds),
                    _format_seconds(median_seconds),
                )
            )

        if suffix == "-ncs-settings" and not include_slow:
            rows.append((suffix, "bump_pv01 (legacy NumPy)", "—", "SKIPPED (pass --slow)"))
        else:
            elapsed = _time_hand_coded_bump(config, market_data, swap)
            rows.append((suffix, "bump_pv01 (legacy NumPy)", "—", _format_seconds(elapsed)))

    lines = [
        "| Settings suffix | Function | Compile time | Median / wall time |",
        "|---|---|---:|---:|",
    ]
    lines.extend(
        f"| {suffix} | {function} | {compile_time} | {run_time} |"
        for suffix, function, compile_time, run_time in rows
    )
    table = "\n".join(lines)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nSaved benchmark table to {OUTPUT_PATH}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slow",
        action="store_true",
        help="include the slow NCS hand-coded bump-and-recalibrate benchmark",
    )
    args = parser.parse_args()
    run_benchmark(include_slow=args.slow)


if __name__ == "__main__":
    main()
