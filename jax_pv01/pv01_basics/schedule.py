"""Periodic schedule generation."""

from dataclasses import dataclass
from datetime import date
from typing import List

from dateutil.relativedelta import relativedelta

from pv01_basics.calendar import HolidayCalendar, adjust_modified_following


@dataclass(frozen=True)
class SchedulePeriod:
    start: date
    end: date
    payment: date


def _add_months(d: date, months: int, roll_eom: bool = False) -> date:
    target = d + relativedelta(months=months)
    if roll_eom and d.day == _last_day_of_month(d):
        return target.replace(day=_last_day_of_month(target))
    return target


def _last_day_of_month(d: date) -> int:
    next_month = d.replace(day=28) + relativedelta(days=4)
    return (next_month - relativedelta(days=next_month.day)).day


def generate_schedule(
    start: date,
    end: date,
    frequency_months: int,
    calendar: HolidayCalendar,
    roll_eom: bool = True,
) -> List[SchedulePeriod]:
    """Generate regular coupon periods from start to end."""
    periods: List[SchedulePeriod] = []
    period_start = start
    while period_start < end:
        unadjusted_end = _add_months(period_start, frequency_months, roll_eom)
        if unadjusted_end >= end:
            period_end = end
        else:
            period_end = unadjusted_end
        payment = calendar.adjust(period_end)
        periods.append(SchedulePeriod(period_start, period_end, payment))
        period_start = period_end
    return periods
