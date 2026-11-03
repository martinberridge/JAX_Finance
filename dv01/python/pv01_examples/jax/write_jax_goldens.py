"""Compute and write frozen JAX market-quote risk values."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp
import numpy as np

from pv01_examples.common import (
    SWAP_COUPON,
    SWAP_NOTIONAL,
    SWAP_PERIOD_TO_START_MONTHS,
    SWAP_TENOR_YEARS,
    VALUATION_DATE,
)
from pv01_examples.paths import CONFIG_STR, PATH_CONFIG, PATH_QUOTES, SETTINGS_SUFFIXES
from pv01_jax.resolve import resolve_calibration_problem, resolve_swap
from pv01_jax.risk import cross_gamma, market_quote_pv01, pv_from_quotes
from pv01_loader.quotes_csv_loader import load_market_data
from pv01_loader.rates_calibration_csv_loader import load as load_calibration
from pv01_product.conventions import BuySell
from pv01_product.resolved_trades import create_test_swap

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "tests" / "jax_port" / "golden_jax"


def compute_values(
    settings_suffix: str,
) -> tuple[tuple[tuple[str, float, float], ...], float]:
    """Compute ordered quote PV01/gamma diagonal rows and swap PV."""
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
    quotes = jnp.asarray(problem.q0)

    pv01 = np.asarray(market_quote_pv01(problem, swap_arrays, quotes))
    gamma_diagonal = np.asarray(
        jnp.diag(cross_gamma(problem, swap_arrays, quotes))
    )
    pv = float(pv_from_quotes(problem, swap_arrays, quotes))
    rows = tuple(
        (label, float(mq_value), float(gamma_value))
        for label, mq_value, gamma_value in zip(
            problem.quote_labels,
            pv01,
            gamma_diagonal,
        )
    )
    return rows, pv


def golden_path(settings_suffix: str) -> Path:
    """Return the golden CSV path for a settings suffix."""
    return GOLDEN_DIR / f"{settings_suffix.lstrip('-')}.csv"


def write_golden(settings_suffix: str) -> Path:
    """Compute and write one settings suffix golden CSV."""
    rows, pv = compute_values(settings_suffix)
    path = golden_path(settings_suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("label", "mq_pv01", "gamma_diag"))
        writer.writerows(rows)
        writer.writerow(("PV", repr(pv), ""))
    return path


def main() -> None:
    for suffix in SETTINGS_SUFFIXES:
        print(f"Wrote {write_golden(suffix)}")


if __name__ == "__main__":
    main()
