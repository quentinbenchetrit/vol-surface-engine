"""Black-76 implied volatility inversion.

The undiscounted price is strictly increasing in sigma, from the intrinsic
value (sigma -> 0) to the forward or strike (sigma -> infinity), so the inverse
is unique. A Newton step on the vega converges in a handful of iterations near
the money, but the vega collapses in the wings, so we fall back to a bracketed
Brent solve there. Prices outside the no-arbitrage bounds return NaN.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _und_price(sigma: float, F: float, K: float, T: float, is_call: bool) -> float:
    vol = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol
    d2 = d1 - vol
    if is_call:
        return F * _norm_cdf(d1) - K * _norm_cdf(d2)
    return K * _norm_cdf(-d2) - F * _norm_cdf(-d1)


def _und_vega(sigma: float, F: float, K: float, T: float) -> float:
    vol = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol
    return F * math.exp(-0.5 * d1 * d1) / _SQRT_2PI * math.sqrt(T)


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    discount: float = 1.0,
    opt_type: str = "C",
    tol: float = 1e-8,
    max_iter: int = 60,
    sigma_max: float = 10.0,
) -> float:
    is_call = str(opt_type).upper().startswith("C")
    if T <= 0 or F <= 0 or K <= 0 or price <= 0 or discount <= 0:
        return float("nan")

    p = price / discount
    intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
    upper = F if is_call else K
    if p <= intrinsic + 1e-12 or p >= upper * (1.0 - 1e-12):
        return float("nan")

    # Brenner-Subrahmanyam style ATM seed, clipped to a sane range.
    sigma = min(5.0, max(1e-4, _SQRT_2PI / math.sqrt(T) * p / F))
    lo, hi = 1e-9, sigma_max
    for _ in range(max_iter):
        diff = _und_price(sigma, F, K, T, is_call) - p
        if abs(diff) < tol:
            return sigma
        vega = _und_vega(sigma, F, K, T)
        if vega < 1e-12:
            break  # wing: vega too small for a reliable Newton step
        nxt = sigma - diff / vega
        if not (lo < nxt < hi):
            break
        sigma = nxt

    try:
        return brentq(lambda s: _und_price(s, F, K, T, is_call) - p, lo, hi, xtol=tol, maxiter=200)
    except ValueError:
        return float("nan")


def implied_vol_surface(chain: pd.DataFrame, forwards: pd.DataFrame, price_col: str = "mid_usd") -> pd.DataFrame:
    """Attach implied vol, log-moneyness and total variance to a tidy chain.

    ``forwards`` is the term structure from :func:`volsurface.data.forward_curve`
    (columns ``expiry``, ``forward``, ``discount``). Vols are computed on the
    option's own quoted side; an ``otm`` flag marks the liquid out-of-the-money
    wing that surface fitting should use.
    """
    # The tidy chain already carries the exchange forward; keep it as a
    # cross-check and let the parity forward define moneyness.
    merged = chain.rename(columns={"forward": "forward_deribit"}).merge(
        forwards[["expiry", "forward", "discount"]], on="expiry", how="left"
    )
    ivs, otm = [], []
    for row in merged.itertuples(index=False):
        F, DF = row.forward, row.discount
        if pd.isna(F) or pd.isna(DF):
            ivs.append(float("nan"))
            otm.append(False)
            continue
        ivs.append(implied_vol(getattr(row, price_col), F, row.strike, row.T, DF, row.type))
        is_call = str(row.type).upper().startswith("C")
        otm.append((is_call and row.strike >= F) or (not is_call and row.strike < F))

    out = merged.copy()
    out["iv"] = ivs
    out["otm"] = otm
    out["log_moneyness"] = np.log(out["strike"] / out["forward"])
    out["total_var"] = out["iv"] ** 2 * out["T"]
    return out
