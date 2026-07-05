"""Quote identifiers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuoteId:
    symbology: str
    ticker: str
    field_name: str = "MarketValue"

    def __str__(self) -> str:
        return f"{self.symbology}/{self.ticker}/{self.field_name}"
