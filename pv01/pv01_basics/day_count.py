"""Day count conventions."""

from datetime import date


def act_365f_year_fraction(start: date, end: date) -> float:
    """Act/365F year fraction between two dates."""
    if end <= start:
        return 0.0
    return (end - start).days / 365.0


def act_365f_between(valuation: date, target: date) -> float:
    """Year fraction from valuation date to target date."""
    return act_365f_year_fraction(valuation, target)
