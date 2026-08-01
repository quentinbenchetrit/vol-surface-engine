"""Forward and discount factor from put-call parity.

For a single expiry, put-call parity in USD reads

    C(K) - P(K) = DF * (F - K),

so a linear regression of the call-minus-put price on the strike recovers the
discount factor DF (minus the slope) and the forward F (intercept / DF). The
implied rate is r = -ln(DF) / T. This is the principled way to pin the forward
from option prices alone; Deribit's own per-expiry forward is kept for a
cross-check.

Caveat on coin-margined instruments: Deribit options settle in coin, so a naive
USD parity slightly biases the discount factor, and the extracted rate is only
meaningful past the shortest expiries. The forward (the intercept term) stays
robust and matches the exchange forward to a few basis points near the money,
which is what the surface work downstream actually needs.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd


class Forward(NamedTuple):
    T: float
    forward: float
    discount: float
    rate: float
    n_pairs: int
    r2: float  # quality of the parity regression, flags noisy expiries
    forward_ref: Optional[float]  # exchange-reported forward, for comparison


def implied_forward(
    expiry_chain: pd.DataFrame,
    price_col: str = "mid_usd",
    moneyness_band: float = 0.15,
) -> Forward:
    """Extract (F, DF, r) for the options of one expiry.

    ``expiry_chain`` must hold both calls and puts for a single expiry with
    columns ``type`` ("C"/"P"), ``strike``, ``T`` and ``price_col``.
    """
    if expiry_chain["T"].nunique() != 1:
        raise ValueError("implied_forward expects a single expiry")
    T = float(expiry_chain["T"].iloc[0])

    calls = expiry_chain[expiry_chain["type"] == "C"].set_index("strike")[price_col]
    puts = expiry_chain[expiry_chain["type"] == "P"].set_index("strike")[price_col]
    pairs = pd.concat({"C": calls, "P": puts}, axis=1).dropna()
    if len(pairs) < 2:
        raise ValueError("need at least two strikes quoted on both sides")

    ref = None
    if "forward" in expiry_chain.columns and expiry_chain["forward"].notna().any():
        ref = float(expiry_chain["forward"].dropna().iloc[0])
    center = ref if ref is not None else float(pairs.index.to_series().median())

    band = pairs[np.abs(pairs.index / center - 1.0) <= moneyness_band]
    if len(band) < 2:
        band = pairs  # wings only: fall back to everything quoted on both sides

    k = band.index.to_numpy(dtype=float)
    y = (band["C"] - band["P"]).to_numpy(dtype=float)
    slope, intercept = np.polyfit(k, y, 1)
    discount = -slope
    if discount <= 0:
        raise ValueError("non-positive discount factor from parity regression")
    forward = intercept / discount
    rate = -math.log(discount) / T
    residual = y - (slope * k + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return Forward(T, forward, discount, rate, len(band), r2, ref)


def forward_curve(chain: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run :func:`implied_forward` on every expiry and return a term structure."""
    out = []
    for expiry, grp in chain.groupby("expiry"):
        try:
            fwd = implied_forward(grp, **kwargs)
        except ValueError:
            continue
        out.append(
            {
                "expiry": expiry,
                "T": fwd.T,
                "forward": fwd.forward,
                "discount": fwd.discount,
                "rate": fwd.rate,
                "n_pairs": fwd.n_pairs,
                "r2": fwd.r2,
                "forward_deribit": fwd.forward_ref,
            }
        )
    return pd.DataFrame(out).sort_values("T").reset_index(drop=True)
