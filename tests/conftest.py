"""Shared helpers for the root test suite."""

import csv
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _column(values):
    """An int, float64 or string array, whichever parses every value."""
    for dtype in (int, float):
        try:
            return np.array([dtype(v) for v in values])
        except ValueError:
            pass
    return np.array(values)


def read_golden(name):
    """
    Read tests/golden/<name> as a dict of NumPy arrays keyed by column.

    Numeric columns come back as float64 (or int when every value is an
    integer); text columns such as `option_type` and `expiry` stay strings.
    """
    with open(GOLDEN_DIR / name, newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    return {col: _column([r[i] for r in body]) for i, col in enumerate(header)}
