import math

import numpy as np
import pandas as pd

from volsurface import black76_price, implied_vol, implied_vol_surface


def test_round_trip_across_the_smile():
    F, T, DF = 65000.0, 0.5, 0.97
    sigma_true = 0.55
    # from deep in the wings on both sides to at the money
    for k in np.linspace(-1.2, 1.2, 25):
        K = F * math.exp(k)
        for opt in ("C", "P"):
            price = float(black76_price(F, K, T, sigma_true, DF, opt))
            if price <= 1e-6:
                continue
            iv = implied_vol(price, F, K, T, DF, opt)
            assert abs(iv - sigma_true) < 1e-6


def test_call_and_put_give_same_vol():
    F, K, T, DF, sigma = 48000.0, 52000.0, 0.25, 0.99, 0.7
    c = float(black76_price(F, K, T, sigma, DF, "C"))
    p = float(black76_price(F, K, T, sigma, DF, "P"))
    assert abs(implied_vol(c, F, K, T, DF, "C") - implied_vol(p, F, K, T, DF, "P")) < 1e-8


def test_price_below_intrinsic_is_nan():
    F, K, T, DF = 100.0, 80.0, 1.0, 1.0
    intrinsic = DF * (F - K)
    assert math.isnan(implied_vol(intrinsic * 0.9, F, K, T, DF, "C"))


def test_surface_recovers_flat_vol():
    F, T, DF, sigma = 30000.0, 0.4, 0.98, 0.65
    strikes = np.linspace(0.7 * F, 1.3 * F, 15)
    rows = []
    for K in strikes:
        opt = "C" if K >= F else "P"
        rows.append(
            {
                "expiry": pd.Timestamp("2026-12-25"),
                "T": T,
                "type": opt,
                "strike": K,
                "mid_usd": float(black76_price(F, K, T, sigma, DF, opt)),
            }
        )
    chain = pd.DataFrame(rows)
    forwards = pd.DataFrame({"expiry": [pd.Timestamp("2026-12-25")], "forward": [F], "discount": [DF]})
    surf = implied_vol_surface(chain, forwards)
    assert np.allclose(surf["iv"], sigma, atol=1e-6)
    assert np.allclose(surf["total_var"], sigma**2 * T, atol=1e-6)
