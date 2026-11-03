"""Test-swap and valuation constants shared by the examples and benchmarks."""

from datetime import date

VALUATION_DATE = date(2016, 8, 1)
SWAP_TENOR_YEARS = 8
SWAP_PERIOD_TO_START_MONTHS = 6
SWAP_COUPON = 0.025
SWAP_NOTIONAL = 10_000_000.0
BP1 = 1.0e-4

__all__ = [
    "VALUATION_DATE",
    "SWAP_TENOR_YEARS",
    "SWAP_PERIOD_TO_START_MONTHS",
    "SWAP_COUPON",
    "SWAP_NOTIONAL",
    "BP1",
]
