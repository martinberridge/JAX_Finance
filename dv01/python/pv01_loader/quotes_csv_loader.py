"""QuotesCsvLoader — load market quotes from CSV."""

from datetime import date
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from pv01_loader.models import MarketData
from pv01_market.quote_id import QuoteId


def _parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def load(market_data_date: date, *resources: Union[str, Path]) -> Dict[QuoteId, float]:
    """Load quotes for a specific valuation date."""
    result: Dict[QuoteId, float] = {}
    for resource in resources:
        path = Path(resource)
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            if pd.isna(row.get("Valuation Date")) or str(row.get("Valuation Date", "")).strip() == "":
                continue
            row_date = _parse_date(str(row["Valuation Date"]))
            if row_date != market_data_date:
                continue
            symbology = str(row["Symbology"]).strip()
            ticker = str(row["Ticker"]).strip()
            field_name = str(row.get("Field Name", "MarketValue")).strip() or "MarketValue"
            value = float(row["Value"])
            qid = QuoteId(symbology, ticker, field_name)
            if qid in result:
                raise ValueError(f"Duplicate quote: {qid}")
            result[qid] = value
    return result


def load_market_data(market_data_date: date, *resources: Union[str, Path]) -> MarketData:
    return MarketData(valuation_date=market_data_date, quotes=load(market_data_date, *resources))
