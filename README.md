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
- [x] Black-76 pricer and implied vol solver, robust in the wings
- [x] SVI per-slice fit with a butterfly no-arbitrage constraint
- [x] SSVI surface with a calendar no-arbitrage constraint
- [x] Heston characteristic function, European pricing by the COS method and Carr-Madan FFT
- [x] Heston calibration to the surface
- [x] Dupire local vol from the SVI surface, by analytic differentiation
- [ ] Heston Monte Carlo with the Andersen QE scheme
- [ ] Variance reduction: antithetics and control variates
- [ ] Exotics: barriers, autocallable, cliquet
- [ ] Pathwise and likelihood-ratio greeks
- [ ] Delta-hedging P&L backtest
- [ ] Optional C++ Monte Carlo core with pybind11

## Results

Slice-by-slice SVI fits to the live BTC surface: market out-of-the-money mid vols against the arbitrage-free SVI curve. Raw SVI captures the skew and the wings across maturities; the steep far call wing on some slices is where a single arbitrage-free slice is hardest to match.

![SVI smiles, market vs fit](figures/svi_smiles.png)

Gatheral's density g(k) for the same slices stays non-negative everywhere, so each calibrated slice is free of butterfly arbitrage.

![Gatheral density stays non-negative](figures/svi_density.png)

Figures are a market snapshot; regenerate them with `python scripts/plot_svi.py`.

### SSVI surface

Tying the slices into one surface with SSVI. The at-the-money variance term structure is taken from the SVI slices and forced non-decreasing, then a single set of parameters (rho, eta, gamma) is fit to the whole surface under the Gatheral-Jacquier butterfly bound eta*(1 + |rho|) <= 2. Both no-arbitrage conditions hold by construction, so the surface is arbitrage-free everywhere.

![SSVI implied vol surface](figures/ssvi_surface.png)

Total variance is non-decreasing in maturity at every moneyness, which is exactly the no-calendar-arbitrage condition.

![Total variance rises with maturity](figures/ssvi_calendar.png)

One surface with three parameters is far more constrained than thirteen independent five-parameter slices, so the global fit trades some per-slice accuracy (about 3 vol points RMSE here) for a single self-consistent arbitrage-free surface. Regenerate with `python scripts/plot_ssvi.py`.

### Heston pricing

Moving from the static surface to a dynamic model. The Heston characteristic function is priced two independent ways, the COS method and the Carr-Madan FFT, and they agree to about 1e-3. The left panel is the implied vol smile the model produces (downward skew from rho < 0, flattening with maturity), priced by COS and inverted with the solver from the earlier step.

![Heston smile and COS vs Carr-Madan](figures/heston_smiles.png)

COS prices a strip of strikes in well under a millisecond, which is what makes the model fast to calibrate next. Regenerate with `python scripts/plot_heston.py`.

### Heston calibration

Fitting the five Heston parameters to the whole live surface in implied-vol space, pricing every strike by COS at each step. The fit is good at medium and long maturities but visibly struggles with the steep short-dated skew, which is the textbook limitation of one-factor Heston: matching it forces the vol of vol up until the Feller condition breaks, and it really wants jumps (Bates) instead. Shown, not hidden.

![Heston fit vs market across maturities](figures/heston_fit.png)

Regenerate with `python scripts/plot_heston_calibration.py`.

### Dupire local volatility

The third model: the one diffusion that reprices every vanilla exactly. Written in total variance, Dupire's denominator turns out to be exactly Gatheral's g(k) from the butterfly condition, so the formula collapses to

    sigma_loc^2 = w_T / g(k)

which makes the dependency explicit: butterfly (g >= 0) and calendar (w_T >= 0) are jointly what keep the local variance positive, and the SSVI fit enforces both. Under SSVI maturity enters only through theta(T), so w_T follows by the chain rule with dw/dtheta in closed form; theta is interpolated with PCHIP, which is C1 and monotonicity-preserving.

Deriving this from the fitted surface rather than raw quotes is the point. Dupire needs a second derivative in strike, which is unstable on noisy prices and can turn negative; on SSVI every derivative is analytic.

![Local volatility surface and slices](figures/dupire_local_vol.png)

Local vol is close to implied at the money and roughly twice as steep in skew nearby, which is the textbook relationship. Regenerate with `python scripts/plot_dupire.py`.

## Layout

    src/volsurface/       black-76 pricing, implied vol, svi/ssvi surfaces, heston, dupire
    src/volsurface/data/  market data: Deribit client, chain parsing, parity forward
    scripts/              snapshot to DuckDB, and svi, ssvi and heston figures (fit included)
    tests/                unit tests that run without network access

## Getting started

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,plot]"

Take a snapshot of the current BTC surface:

    python scripts/snapshot.py --currency BTC --db data/deribit_btc.duckdb

Fit SVI slices and draw the figures above:

    python scripts/plot_svi.py --currency BTC

Run the tests:

    pytest -q

## References

- Gatheral and Jacquier, Arbitrage-free SVI volatility surfaces (2014).
- Fang and Oosterlee, A novel pricing method based on Fourier-cosine series (2008).
- Andersen, Simple and efficient simulation of the Heston stochastic volatility model (2008).
- Carr and Madan, Option valuation using the fast Fourier transform (1999).
- Albrecher et al., The little Heston trap (2007).
- Dupire, Pricing with a smile (1994).
- Gatheral, The Volatility Surface (2006), chapter 1.
