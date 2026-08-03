import numpy as np
import pandas as pd

from volsurface import black76_price, implied_vol
from volsurface.heston import (
    HestonParams,
    calibrate,
    feller_ok,
    price_cos,
    price_carr_madan,
    price_carr_madan_at,
)

P = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)


def test_feller():
    assert feller_ok(HestonParams(2.0, 0.04, 0.3, -0.5, 0.04))       # 2*2*0.04=0.16 >= 0.09
    assert not feller_ok(HestonParams(1.0, 0.04, 0.5, -0.5, 0.04))   # 0.08 < 0.25


def test_put_call_parity():
    S0, K, r, q, T = 100.0, 105.0, 0.02, 0.0, 1.0
    c = price_cos(S0, K, r, q, T, P, "C")
    put = price_cos(S0, K, r, q, T, P, "P")
    assert np.isclose(c - put, S0 * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-4)


def test_black_scholes_limit():
    # vanishing vol of vol with v0 = theta collapses Heston to Black-Scholes
    flat = HestonParams(kappa=1.5, theta=0.04, xi=1e-2, rho=0.0, v0=0.04)
    S0, K, r, q, T = 100.0, 100.0, 0.0, 0.0, 1.0
    price = price_cos(S0, K, r, q, T, flat, "C")
    iv = implied_vol(price, S0, K, T, 1.0, "C")
    assert abs(iv - 0.20) < 5e-3          # sqrt(0.04) = 0.20


def test_cos_matches_carr_madan():
    S0, r, q, T = 100.0, 0.01, 0.0, 0.75
    strikes = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
    cos = price_cos(S0, strikes, r, q, T, P, "C")
    cm = np.array([price_carr_madan_at(S0, K, r, q, T, P) for K in strikes])
    assert np.allclose(cos, cm, atol=0.02)


def test_call_bounds_and_monotonic():
    S0, r, q, T = 100.0, 0.0, 0.0, 0.5
    strikes = np.linspace(70, 140, 15)
    calls = price_cos(S0, strikes, r, q, T, P, "C")
    assert np.all(calls > 0) and np.all(calls <= S0 + 1e-8)
    assert np.all(np.diff(calls) < 0)          # calls fall as strike rises


def test_cos_convergence():
    S0, K, r, q, T = 100.0, 110.0, 0.0, 0.0, 1.0
    coarse = price_cos(S0, K, r, q, T, P, "C", N=64)
    fine = price_cos(S0, K, r, q, T, P, "C", N=512)
    assert abs(coarse - fine) < 1e-3


def _synthetic_surface(true, F=100.0):
    rows = []
    for T in (0.1, 0.25, 0.5, 1.0):
        kmax = 3.0 * np.sqrt(true.theta * T)
        for k in np.linspace(-kmax, kmax, 11):
            K = F * np.exp(k)
            call = price_cos(F, K, 0.0, 0.0, T, true, "C")
            iv = implied_vol(call, F, K, T, 1.0, "C")
            rows.append({
                "expiry": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(round(T * 365))),
                "forward": F, "discount": 1.0, "T": T, "strike": K,
                "iv": iv, "otm": True, "rel_spread": 0.05,
            })
    return pd.DataFrame(rows)


def test_calibration_recovers_synthetic_surface():
    true = HestonParams(kappa=2.5, theta=0.05, xi=0.4, rho=-0.6, v0=0.03)
    fit = calibrate(_synthetic_surface(true))
    assert fit.rmse_vol < 3e-3
    assert fit.feller_ok == feller_ok(true)
    assert abs(fit.params.theta - true.theta) < 0.01
    assert abs(fit.params.v0 - true.v0) < 0.01
    assert abs(fit.params.rho - true.rho) < 0.1
