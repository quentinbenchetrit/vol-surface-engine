"""Fetch the current Deribit option chain and append it to a DuckDB file.

Run this on a schedule to accumulate a history of surfaces:

    python scripts/snapshot.py --currency BTC --db data/deribit_btc.duckdb
"""
from __future__ import annotations

import argparse
from pathlib import Path

from volsurface.data import load_chain, store_snapshot
from volsurface.data.forward import forward_curve


def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot a Deribit option chain into DuckDB")
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--db", default="data/deribit_btc.duckdb")
    ap.add_argument("--min-volume", type=float, default=0.0)
    args = ap.parse_args()

    chain = load_chain(currency=args.currency, min_volume=args.min_volume)
    if chain.empty:
        print("No quotes returned, nothing stored.")
        return

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    total = store_snapshot(chain, args.db)

    curve = forward_curve(chain)
    as_of = chain["as_of"].iloc[0]
    print(f"{as_of:%Y-%m-%d %H:%M UTC}  {args.currency}  "
          f"{len(chain)} quotes, {chain['expiry'].nunique()} expiries, "
          f"{total} rows total in {args.db}")
    with_ref = curve.dropna(subset=["forward_deribit"])
    if not with_ref.empty:
        gap = (with_ref["forward"] / with_ref["forward_deribit"] - 1.0).abs().max()
        print(f"max parity-vs-exchange forward gap: {gap * 1e4:.1f} bps")


if __name__ == "__main__":
    main()
