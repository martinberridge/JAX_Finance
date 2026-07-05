"""Market conventions for GBP instruments."""

from dataclasses import dataclass
from datetime import date
from enum import Enum

from pv01_basics.calendar import HolidayCalendar, gblo
from pv01_basics.tenor import Tenor


class BuySell(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class ConventionInfo:
    name: str
    currency: str = "GBP"
    day_count: str = "Act/365F"
    calendar: HolidayCalendar = None  # type: ignore
    spot_lag_days: int = 0
    fixed_frequency_months: int = 6
    float_frequency_months: int = 6
    ois_frequency_months: int = 12

    def __post_init__(self):
        if self.calendar is None:
            object.__setattr__(self, "calendar", gblo())


GBP_FIXED_6M_LIBOR_6M = ConventionInfo(
    name="GBP-FIXED-6M-LIBOR-6M",
    fixed_frequency_months=6,
    float_frequency_months=6,
)

GBP_FIXED_1Y_SONIA_OIS = ConventionInfo(
    name="GBP-FIXED-1Y-SONIA-OIS",
    fixed_frequency_months=12,
    float_frequency_months=12,
    ois_frequency_months=12,
)

GBP_LIBOR_6M_FIX = ConventionInfo(
    name="GBP-LIBOR-6M",
    float_frequency_months=6,
)

CONVENTIONS = {
    "GBP-FIXED-6M-LIBOR-6M": GBP_FIXED_6M_LIBOR_6M,
    "GBP-FIXED-1Y-SONIA-OIS": GBP_FIXED_1Y_SONIA_OIS,
    "GBP-LIBOR-6M": GBP_LIBOR_6M_FIX,
}


def spot_date(trade_date: date, convention: ConventionInfo) -> date:
    cal = convention.calendar
    base = trade_date
    if convention.spot_lag_days == 0:
        return cal.adjust(base)
    return cal.add_days(base, convention.spot_lag_days)
