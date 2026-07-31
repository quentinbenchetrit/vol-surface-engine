"""Parse Deribit option instruments into a tidy chain and persist snapshots."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import NamedTuple, Optional

import pandas as pd

from .deribit import DeribitClient

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
# Deribit option names look like "BTC-25DEC26-20000-P".
_PATTERN = re.compile(r"^([A-Z]+)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:\.\d+)?)-([CP])$")

# Deribit options settle at 08:00 UTC on the expiry date.
_EXPIRY_HOUR_UTC = 8
_YEAR_DAYS = 365.0


class Instrument(NamedTuple):
    underlying: str
    expiry: datetime
    strike: float
    option_type: str  # "C" or "P"


def parse_instrument(name: str) -> Instrument:
    m = _PATTERN.match(name)
    if m is None:
        raise ValueError(f"Unrecognized Deribit option name: {name!r}")
    underlying, day, mon, yy, strike, cp = m.groups()
    month = _MONTHS.get(mon)
    if month is None:
        raise ValueError(f"Unknown month {mon!r} in {name!r}")
    expiry = datetime(2000 + int(yy), month, int(day), _EXPIRY_HOUR_UTC, tzinfo=timezone.utc)
    return Instrument(underlying, expiry, float(strike), cp)


def load_chain(
    currency: str = "BTC",
    client: Optional[DeribitClient] = None,
    as_of: Optional[datetime] = None,
    min_volume: float = 0.0,
) -> pd.DataFrame:
    """Return the current option chain as a tidy frame.

    Premiums are quoted in coin (BTC) on Deribit; a USD column is added using
    the spot index. Rows with a missing or crossed quote are dropped. Time to
    expiry is an actual/365 year fraction from ``as_of`` (defaults to now UTC).
    """
    client = client or DeribitClient()
    as_of = as_of or datetime.now(timezone.utc)
    index = client.index_price(f"{currency.lower()}_usd")
    summary = client.book_summary(currency=currency, kind="option")

    rows = []
    for row in summary:
        inst = parse_instrument(row["instrument_name"])
        ttm = (inst.expiry - as_of).total_seconds() / (_YEAR_DAYS * 86400.0)
        if ttm <= 0:
            continue
        bid, ask = row.get("bid_price"), row.get("ask_price")
        mid = row.get("mid_price")
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            continue
        if (row.get("volume") or 0.0) < min_volume:
            continue
        rows.append(
            {
                "as_of": as_of,
                "instrument": row["instrument_name"],
                "underlying": inst.underlying,
                "expiry": inst.expiry,
                "T": ttm,
                "type": inst.option_type,
                "strike": inst.strike,
                "bid": bid,
                "ask": ask,
                "mid": mid if mid is not None else 0.5 * (bid + ask),
                "mark": row.get("mark_price"),
                "mark_iv": (row.get("mark_iv") or float("nan")) / 100.0,
                "forward": row.get("underlying_price"),  # Deribit per-expiry forward
                "index": index,
                "volume": row.get("volume") or 0.0,
                "open_interest": row.get("open_interest") or 0.0,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # USD premiums for downstream pricing (1 contract = 1 coin of notional).
    for col in ("bid", "ask", "mid", "mark"):
        df[f"{col}_usd"] = df[col] * df["index"]
    df = df.sort_values(["expiry", "strike", "type"]).reset_index(drop=True)
    return df


def store_snapshot(df: pd.DataFrame, path: str, table: str = "option_chain") -> int:
    """Append a snapshot to a DuckDB file so a history builds up over time."""
    import duckdb

    if df.empty:
        return 0
    con = duckdb.connect(path)
    try:
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df WHERE 1=0")
        con.execute(f"INSERT INTO {table} SELECT * FROM df")
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        con.close()
    return int(n)
