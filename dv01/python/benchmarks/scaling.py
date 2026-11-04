"""How AD and bump-and-recalibrate PV01 scale with the number of curve quotes.

Sweeps N quotes on ``pv01_jax.synthetic.synthetic_problem`` and times
market-quote PV01 four ways on the same compiled pricer and calibrator: AD
in reverse mode (``market_quote_pv01``), AD in forward mode along a parallel
shift (``parallel_pv01``, one number rather than N), and 2N bump-and-
recalibrate scenarios batched with ``vmap`` or run one at a time with
``lax.map``. Compile time is reported separately from the median of 20 runs.

Run from dv01/python: ``python -m benchmarks.scaling``. Writes the table to
``benchmarks/results/scaling.md`` and a log-log chart to
``docs/assets/dv01_scaling.png`` at the repo root. Needs the ``[bench]``
extra (matplotlib).
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp

from benchmarks.timing import RUNS, time_jitted
from pv01_examples.paths import PYTHON_DIR
from pv01_jax.risk import (
    bump_market_quote_pv01,
    bump_market_quote_pv01_sequential,
    market_quote_pv01,
    parallel_pv01,
)
from pv01_jax.synthetic import synthetic_problem

SIZES = (10, 20, 40, 80, 160)
TABLE_PATH = PYTHON_DIR / "benchmarks" / "results" / "scaling.md"
CHART_PATH = PYTHON_DIR.parents[1] / "docs" / "assets" / "dv01_scaling.png"

# (key, label, function, line style); categorical slots 1-4 in this order
FUNCTIONS = (
    ("ad", "AD, reverse mode (market_quote_pv01)", market_quote_pv01, "-"),
    ("jvp", "AD, forward mode, parallel shift only (parallel_pv01)", parallel_pv01, "-"),
    ("bump_vmap", "Bump and recalibrate, vmap", bump_market_quote_pv01, "--"),
    ("bump_seq", "Bump and recalibrate, lax.map", bump_market_quote_pv01_sequential, "--"),
)
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def run_sweep(sizes, interpolator: str) -> dict[int, dict[str, tuple[float, float]]]:
    results = {}
    for n_quotes in sizes:
        problem, swap = synthetic_problem(n_quotes, interpolator)
        quotes = jnp.asarray(problem.q0)
        results[n_quotes] = {}
        for key, _, function, _ in FUNCTIONS:
            results[n_quotes][key] = time_jitted(
                lambda q, function=function: function(problem, swap, q), (quotes,)
            )
            compile_s, median_s = results[n_quotes][key]
            print(f"N={n_quotes:4d} {key:10s} compile {compile_s:7.2f}s  median {median_s * 1e3:10.3f} ms",
                  flush=True)
    return results


def write_table(results, interpolator: str) -> str:
    lines = [
        f"Synthetic two-curve problem, {interpolator} interpolation. Median of {RUNS} runs after",
        "compilation, in milliseconds; compile time (first call) in seconds.",
        "",
        "| N quotes | AD reverse (ms) | AD jvp parallel (ms) | Bump vmap (ms) | Bump lax.map (ms) "
        "| Bump vmap / AD | Bump lax.map / AD | Compile AD / bump vmap (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n_quotes, row in results.items():
        ad = row["ad"][1]
        lines.append(
            f"| {n_quotes} | {ad * 1e3:.3f} | {row['jvp'][1] * 1e3:.3f} "
            f"| {row['bump_vmap'][1] * 1e3:.1f} | {row['bump_seq'][1] * 1e3:.1f} "
            f"| {row['bump_vmap'][1] / ad:.0f} | {row['bump_seq'][1] / ad:.0f} "
            f"| {row['ad'][0]:.1f} / {row['bump_vmap'][0]:.1f} |"
        )
    table = "\n".join(lines) + "\n"
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table, encoding="utf-8")
    return table


def _spread_labels(values: list[float], min_ratio: float = 1.35) -> list[float]:
    """Nudge label heights apart on a log axis so end labels don't overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for previous, current in zip(order, order[1:]):
        placed[current] = max(placed[current], placed[previous] * min_ratio)
    return placed


def write_chart(results, interpolator: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    sizes = list(results)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    end_values = []
    for (key, label, _, style), color in zip(FUNCTIONS, SERIES_COLORS):
        seconds = [results[n][key][1] for n in sizes]
        ax.plot(sizes, [s * 1e3 for s in seconds], style, color=color, linewidth=2,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label, solid_capstyle="round", dash_capstyle="round")
        end_values.append(seconds[-1] * 1e3)

    short_labels = ("AD reverse", "AD jvp", "Bump vmap", "Bump lax.map")
    for text, y, value in zip(short_labels, _spread_labels(end_values), end_values):
        ax.annotate(f"{text}  {value:,.1f} ms" if value >= 1 else f"{text}  {value:.2f} ms",
                    xy=(sizes[-1], y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT_PRIMARY)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] / 1.15, sizes[-1] * 2.6)
    ax.set_xlabel("Number of market quotes (N)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Median time per PV01 (ms, log scale)", color=TEXT_SECONDARY, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.10g}"))
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)

    def growth(key):
        return results[sizes[-1]][key][1] / results[sizes[0]][key][1]

    bump_growth = sorted((growth("bump_vmap"), growth("bump_seq")))
    fig.suptitle(f"{sizes[0]} \u2192 {sizes[-1]} quotes: AD PV01 {growth('ad'):.0f}\u00d7 slower, "
                 f"bumping {bump_growth[0]:,.0f}\u2013{bump_growth[1]:,.0f}\u00d7 slower",
                 x=0.07, ha="left", fontsize=11.5, color=TEXT_PRIMARY, fontweight="bold")
    ax.set_title(f"Synthetic OIS + IRS curves, {interpolator} interpolation, same compiled JAX "
                 f"calibrator; median of {RUNS} runs after compile",
                 loc="left", fontsize=8.5, color=TEXT_SECONDARY, pad=10)
    legend = ax.legend(loc="upper left", fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interpolator", default="Linear",
                        choices=("Linear", "DoubleQuadratic", "NaturalCubicSpline"))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    args = parser.parse_args()

    results = run_sweep(args.sizes, args.interpolator)
    print()
    print(write_table(results, args.interpolator))
    write_chart(results, args.interpolator)
    print(f"Saved table to {TABLE_PATH}")
    print(f"Saved chart to {CHART_PATH}")


if __name__ == "__main__":
    main()
