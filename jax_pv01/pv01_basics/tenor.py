"""Tenor parsing and date arithmetic."""

import re
from dataclasses import dataclass
from datetime import date

from dateutil.relativedelta import relativedelta

_TENOR_RE = re.compile(r"^P?(\d+)([DWMY])$", re.IGNORECASE)


@dataclass(frozen=True)
class Tenor:
    """Period tenor such as 3M, 1Y, 30Y."""

    period: str

    @staticmethod
    def parse(text: str) -> "Tenor":
        text = text.strip().upper()
        if text.startswith("P"):
            text = text[1:]
        return Tenor(text)

    def add_to(self, base: date) -> date:
        m = _TENOR_RE.match(self.period if self.period.startswith("P") else f"P{self.period}")
        if not m:
            raise ValueError(f"Invalid tenor: {self.period}")
        amount = int(m.group(1))
        unit = m.group(2).upper()
        if unit == "D":
            return base + relativedelta(days=amount)
        if unit == "W":
            return base + relativedelta(weeks=amount)
        if unit == "M":
            return base + relativedelta(months=amount)
        if unit == "Y":
            return base + relativedelta(years=amount)
        raise ValueError(f"Unknown tenor unit: {unit}")

    @property
    def months(self) -> int:
        m = _TENOR_RE.match(self.period if not self.period.startswith("P") else self.period[1:])
        if m and m.group(2).upper() == "M":
            return int(m.group(1))
        if m and m.group(2).upper() == "Y":
            return int(m.group(1)) * 12
        return 0
