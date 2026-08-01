import math

import numpy as np
import pandas as pd

from volsurface.data import implied_forward


def _synthetic_expiry(F=65000.0, r=0.05, T=0.25, seed=0):
    """Build calls and puts consistent with parity plus a small quote noise."""
    rng = np.random.default_rng(seed)
    df_disc = math.exp(-r * T)
    strikes = np.linspace(0.7 * F, 1.3 * F, 25)
    # An arbitrary but positive call price; the put is fixed by parity.
    intrinsic = np.maximum(F - strikes, 0.0) * df_disc
    call = intrinsic + 0.02 * F * df_disc + rng.normal(0, 1.0, size=strikes.size)
    put = call - df_disc * (F - strikes)
    rows = []
    for k, c, p in zip(strikes, call, put):
        rows.append({"type": "C", "strike": k, "T": T, "mid_usd": c, "forward": F})
        rows.append({"type": "P", "strike": k, "T": T, "mid_usd": p, "forward": F})
    return pd.DataFrame(rows)


def test_recovers_forward_and_rate():
    F, r, T = 65000.0, 0.05, 0.25
    chain = _synthetic_expiry(F=F, r=r, T=T)
    fwd = implied_forward(chain)
    assert abs(fwd.forward - F) / F < 1e-3
    assert abs(fwd.rate - r) < 1e-3
    assert 0.0 < fwd.discount < 1.0
    assert fwd.r2 > 0.99  # parity line fits the synthetic quotes tightly


def test_discount_matches_rate():
    chain = _synthetic_expiry(F=48000.0, r=0.03, T=0.5)
    fwd = implied_forward(chain)
    assert abs(math.exp(-fwd.rate * fwd.T) - fwd.discount) < 1e-9
