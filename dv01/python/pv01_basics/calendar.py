"""GBLO business-day calendar with UK bank holidays."""

from datetime import date, timedelta
from functools import lru_cache

# UK bank holidays 2010-2040 (GBLO subset, aligned with jollyday GBLO)
_UK_HOLIDAYS = {
    date(2010, 1, 1), date(2010, 4, 2), date(2010, 4, 5), date(2010, 5, 3),
    date(2010, 5, 31), date(2010, 8, 30), date(2010, 12, 27), date(2010, 12, 28),
    date(2011, 1, 3), date(2011, 4, 22), date(2011, 4, 25), date(2011, 4, 29),
    date(2011, 5, 2), date(2011, 5, 30), date(2011, 8, 29), date(2011, 12, 26),
    date(2011, 12, 27),
    date(2012, 1, 2), date(2012, 4, 6), date(2012, 4, 9), date(2012, 5, 7),
    date(2012, 6, 4), date(2012, 6, 5), date(2012, 8, 27), date(2012, 12, 25),
    date(2012, 12, 26),
    date(2013, 1, 1), date(2013, 3, 29), date(2013, 4, 1), date(2013, 5, 6),
    date(2013, 5, 27), date(2013, 8, 26), date(2013, 12, 25), date(2013, 12, 26),
    date(2014, 1, 1), date(2014, 4, 18), date(2014, 4, 21), date(2014, 5, 5),
    date(2014, 5, 26), date(2014, 8, 25), date(2014, 12, 25), date(2014, 12, 26),
    date(2015, 1, 1), date(2015, 4, 3), date(2015, 4, 6), date(2015, 5, 4),
    date(2015, 5, 25), date(2015, 8, 31), date(2015, 12, 25), date(2015, 12, 28),
    date(2016, 1, 1), date(2016, 3, 25), date(2016, 3, 28), date(2016, 5, 2),
    date(2016, 5, 30), date(2016, 8, 29), date(2016, 12, 26), date(2016, 12, 27),
    date(2017, 1, 2), date(2017, 4, 14), date(2017, 4, 17), date(2017, 5, 1),
    date(2017, 5, 29), date(2017, 8, 28), date(2017, 12, 25), date(2017, 12, 26),
    date(2018, 1, 1), date(2018, 3, 30), date(2018, 4, 2), date(2018, 5, 7),
    date(2018, 5, 28), date(2018, 8, 27), date(2018, 12, 25), date(2018, 12, 26),
    date(2019, 1, 1), date(2019, 4, 19), date(2019, 4, 22), date(2019, 5, 6),
    date(2019, 5, 27), date(2019, 8, 26), date(2019, 12, 25), date(2019, 12, 26),
    date(2020, 1, 1), date(2020, 4, 10), date(2020, 4, 13), date(2020, 5, 8),
    date(2020, 5, 25), date(2020, 8, 31), date(2020, 12, 25), date(2020, 12, 28),
    date(2021, 1, 1), date(2021, 4, 2), date(2021, 4, 5), date(2021, 5, 3),
    date(2021, 5, 31), date(2021, 8, 30), date(2021, 12, 27), date(2021, 12, 28),
    date(2022, 1, 3), date(2022, 4, 15), date(2022, 4, 18), date(2022, 5, 2),
    date(2022, 6, 2), date(2022, 6, 3), date(2022, 8, 29), date(2022, 9, 19),
    date(2022, 12, 26), date(2022, 12, 27),
    date(2023, 1, 2), date(2023, 4, 7), date(2023, 4, 10), date(2023, 5, 1),
    date(2023, 5, 8), date(2023, 5, 29), date(2023, 8, 28), date(2023, 12, 25),
    date(2023, 12, 26),
    date(2024, 1, 1), date(2024, 3, 29), date(2024, 4, 1), date(2024, 5, 6),
    date(2024, 5, 27), date(2024, 8, 26), date(2024, 12, 25), date(2024, 12, 26),
    date(2025, 1, 1), date(2025, 4, 18), date(2025, 4, 21), date(2025, 5, 5),
    date(2025, 5, 26), date(2025, 8, 25), date(2025, 12, 25), date(2025, 12, 26),
    date(2026, 1, 1), date(2026, 4, 3), date(2026, 4, 6), date(2026, 5, 4),
    date(2026, 5, 25), date(2026, 8, 31), date(2026, 12, 25), date(2026, 12, 28),
    date(2027, 1, 1), date(2027, 3, 26), date(2027, 3, 29), date(2027, 5, 3),
    date(2027, 5, 31), date(2027, 8, 30), date(2027, 12, 27), date(2027, 12, 28),
    date(2028, 1, 3), date(2028, 4, 14), date(2028, 4, 17), date(2028, 5, 1),
    date(2028, 5, 29), date(2028, 8, 28), date(2028, 12, 25), date(2028, 12, 26),
    date(2029, 1, 1), date(2029, 3, 30), date(2029, 4, 2), date(2029, 5, 7),
    date(2029, 5, 28), date(2029, 8, 27), date(2029, 12, 25), date(2029, 12, 26),
    date(2030, 1, 1), date(2030, 4, 19), date(2030, 4, 22), date(2030, 5, 6),
    date(2030, 5, 27), date(2030, 8, 26), date(2030, 12, 25), date(2030, 12, 26),
}


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def is_holiday(d: date) -> bool:
    return d in _UK_HOLIDAYS


def is_business_day(d: date) -> bool:
    return not is_weekend(d) and not is_holiday(d)


def adjust_following(d: date) -> date:
    """Following business day adjustment."""
    while not is_business_day(d):
        d += timedelta(days=1)
    return d


def adjust_modified_following(d: date) -> date:
    """Modified following business day adjustment."""
    original_month = d.month
    adjusted = adjust_following(d)
    if adjusted.month != original_month:
        d = d.replace(day=1) - timedelta(days=1)
        while not is_business_day(d):
            d -= timedelta(days=1)
        return d
    return adjusted


def add_business_days(d: date, days: int) -> date:
    """Add business days (can be zero)."""
    if days == 0:
        return adjust_modified_following(d)
    step = 1 if days > 0 else -1
    remaining = abs(days)
    current = d
    while remaining > 0:
        current += timedelta(days=step)
        if is_business_day(current):
            remaining -= 1
    return current


@lru_cache(maxsize=1)
def gblo() -> "HolidayCalendar":
    return HolidayCalendar("GBLO")


class HolidayCalendar:
    """GBLO holiday calendar."""

    def __init__(self, name: str):
        self.name = name

    def is_business_day(self, d: date) -> bool:
        return is_business_day(d)

    def adjust(self, d: date, convention: str = "ModifiedFollowing") -> date:
        if convention == "ModifiedFollowing":
            return adjust_modified_following(d)
        return adjust_following(d)

    def add_days(self, d: date, days: int) -> date:
        return add_business_days(d, days)
