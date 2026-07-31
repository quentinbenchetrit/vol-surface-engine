# vol-surface-engine

An arbitrage-free implied volatility surface and stochastic-volatility calibration engine, built on live Deribit option data (BTC and ETH). The goal is a small but honest replica of a derivatives desk workflow: pull the market, clean it, pin the forward from the options themselves, fit a no-arbitrage surface, calibrate a model, price exotics by Monte Carlo, and check the whole thing with a hedging backtest.

The methods (SVI/SSVI, Heston, Dupire, Monte Carlo) are model-general. Crypto options are used because the full surface is available in one public API call, with mark IVs, greeks and a per-expiry forward, which removes the usual data-access friction.

## Data

Source: the Deribit v2 public market-data API. No key is required. One call to `get_book_summary_by_currency` returns the entire option cross-section (hundreds of instruments) with best bid/ask, mark price, mark implied vol and the per-expiry underlying forward.

Two points worth stating plainly:

- Premiums are quoted in coin (BTC). A USD column is derived from the spot index for downstream pricing.
- The public API gives the current surface, not a deep history. `scripts/snapshot.py` appends each pull to a DuckDB file, so running it on a schedule builds a time series of surfaces.

The forward and discount factor are not taken from the exchange. They are recovered from put-call parity: for one expiry, `C(K) - P(K) = DF * (F - K)`, so a linear fit of call-minus-put against strike gives `DF` (minus the slope) and `F` (intercept over `DF`), and `r = -ln(DF) / T`. Deribit's own forward is kept only as a cross-check, reported in basis points by the snapshot script.

## Roadmap

The project is built in increments. Each item lands when it works and is tested.

- [x] Deribit client and tidy option chain, with DuckDB snapshots
- [x] Forward and discount from put-call parity, cross-checked against the exchange
- [ ] Implied vol solver, robust in the wings
- [ ] SVI per-slice fit with a butterfly no-arbitrage constraint
- [ ] SSVI surface with a calendar no-arbitrage constraint
- [ ] Heston characteristic function, European pricing by the COS method and Carr-Madan FFT
- [ ] Heston calibration to the surface
- [ ] Dupire local vol from the SVI surface, by analytic differentiation
- [ ] Heston Monte Carlo with the Andersen QE scheme
- [ ] Variance reduction: antithetics and control variates
- [ ] Exotics: barriers, autocallable, cliquet
- [ ] Pathwise and likelihood-ratio greeks
- [ ] Delta-hedging P&L backtest
- [ ] Optional C++ Monte Carlo core with pybind11

## Layout

    src/volsurface/data/   market data: Deribit client, chain parsing, parity forward
    scripts/               snapshot utility that appends a surface to DuckDB
    tests/                 unit tests that run without network access

## Getting started

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

Take a snapshot of the current BTC surface:

    python scripts/snapshot.py --currency BTC --db data/deribit_btc.duckdb

Run the tests:

    pytest -q

## References

- Gatheral and Jacquier, Arbitrage-free SVI volatility surfaces (2014).
- Fang and Oosterlee, A novel pricing method based on Fourier-cosine series (2008).
- Andersen, Simple and efficient simulation of the Heston stochastic volatility model (2008).
- Carr and Madan, Option valuation using the fast Fourier transform (1999).
