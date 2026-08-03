"""Heston model: characteristic function and European pricing (COS and Carr-Madan).

The variance follows a CIR process,

    dS/S = (r - q) dt + sqrt(v) dW1,
    dv   = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1, W2> = rho dt,

and the log-price has a closed-form characteristic function. Knowing the
characteristic function is enough to price vanillas semi-analytically, which is
what makes Heston fast to calibrate. Two independent methods are provided:

  - the COS method (Fang-Oosterlee), a Fourier-cosine expansion of the density,
    with (near) exponential convergence,
  - the Carr-Madan FFT, which returns a whole strip of strikes at once.

The characteristic function uses the Albrecher "little trap" formulation, which
stays continuous for long maturities where the naive branch would jump.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import least_squares

from .impliedvol import implied_vol


class HestonParams(NamedTuple):
    kappa: float  # mean-reversion speed
    theta: float  # long-run variance
    xi: float     # vol of vol
    rho: float    # spot / variance correlation
    v0: float     # initial variance


def feller_ok(p: HestonParams) -> bool:
    """2 kappa theta >= xi^2 keeps the variance strictly positive."""
    return 2.0 * p.kappa * p.theta >= p.xi * p.xi


def cf_log_return(u, T: float, p: HestonParams, r: float = 0.0, q: float = 0.0):
    """Characteristic function of the log-return ln(S_T / S_0), trap-free."""
    u = np.asarray(u, dtype=complex)
    xi2 = p.xi * p.xi
    beta = p.kappa - 1j * p.rho * p.xi * u
    D = np.sqrt(beta * beta + xi2 * (1j * u + u * u))
    G = (beta - D) / (beta + D)
    eDT = np.exp(-D * T)
    C = (p.kappa * p.theta / xi2) * ((beta - D) * T - 2.0 * np.log((1.0 - G * eDT) / (1.0 - G)))
    Dt = (p.v0 / xi2) * (beta - D) * (1.0 - eDT) / (1.0 - G * eDT)
    return np.exp(1j * u * (r - q) * T + C + Dt)


def cf_log_spot(u, S0: float, T: float, p: HestonParams, r: float = 0.0, q: float = 0.0):
    """Characteristic function of ln(S_T)."""
    return np.exp(1j * np.asarray(u, dtype=complex) * np.log(S0)) * cf_log_return(u, T, p, r, q)


def _c1(T, p, r, q):
    return (r - q) * T + (1.0 - np.exp(-p.kappa * T)) * (p.theta - p.v0) / (2.0 * p.kappa) - 0.5 * p.theta * T


def _var_scale(T, p):
    # E[integrated variance] is a robust scale for the log-return dispersion
    return p.theta * T + (p.v0 - p.theta) * (1.0 - np.exp(-p.kappa * T)) / p.kappa


def _chi(u, a, c, d):
    return (1.0 / (1.0 + u * u)) * (
        np.cos(u * (d - a)) * np.exp(d) - np.cos(u * (c - a)) * np.exp(c)
        + u * np.sin(u * (d - a)) * np.exp(d) - u * np.sin(u * (c - a)) * np.exp(c)
    )


def _psi(u, a, c, d):
    out = np.empty_like(u)
    out[0] = d - c
    out[1:] = (np.sin(u[1:] * (d - a)) - np.sin(u[1:] * (c - a))) / u[1:]
    return out


def price_cos(S0, K, r, q, T, p: HestonParams, option: str = "C", N: int = 256, L: float = 12.0):
    """European option price by the COS method. K may be a scalar or an array."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    scale = np.sqrt(max(float(_var_scale(T, p)), 1e-10))
    c1 = float(_c1(T, p, r, q))
    a, b = c1 - L * scale, c1 + L * scale

    k = np.arange(N)
    u = k * np.pi / (b - a)
    cf = cf_log_return(u, T, p, r, q)

    is_call = option.upper().startswith("C")
    if is_call:
        coeff = _chi(u, a, 0.0, b) - _psi(u, a, 0.0, b)
    else:
        coeff = _psi(u, a, a, 0.0) - _chi(u, a, a, 0.0)

    x = np.log(S0 / K)                                  # (M,)
    F = (cf[None, :] * np.exp(1j * np.outer(x - a, u))).real  # (M, N)
    F[:, 0] *= 0.5
    Vk = (2.0 / (b - a)) * K[:, None] * coeff[None, :]        # (M, N)
    price = np.exp(-r * T) * np.sum(F * Vk, axis=1)
    return float(price[0]) if price.size == 1 else price


def price_carr_madan(S0, r, q, T, p: HestonParams, alpha: float = 1.5, N: int = 4096, eta: float = 0.25):
    """Call prices across a strip of strikes via the Carr-Madan FFT.

    Returns (strikes, call_prices). Prices at specific strikes come from
    interpolating this grid, or from the COS method for a single strike.
    """
    lam = 2.0 * np.pi / (N * eta)
    bb = N * lam / 2.0
    ku = -bb + lam * np.arange(N)          # log strikes
    v = eta * np.arange(N)

    phi = cf_log_spot(v - (alpha + 1.0) * 1j, S0, T, p, r, q)
    psi = np.exp(-r * T) * phi / (alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v)

    w = (eta / 3.0) * (3.0 + (-1.0) ** np.arange(1, N + 1))  # Simpson weights
    w[0] = eta / 3.0
    fft_in = np.exp(1j * bb * v) * psi * w
    call = (np.exp(-alpha * ku) / np.pi) * np.fft.fft(fft_in).real
    return np.exp(ku), call


def price_carr_madan_at(S0, K, r, q, T, p: HestonParams, **kwargs):
    """Call price at a single strike, by interpolating the Carr-Madan grid."""
    strikes, calls = price_carr_madan(S0, r, q, T, p, **kwargs)
    return float(np.interp(np.log(K), np.log(strikes), calls))


class HestonFit(NamedTuple):
    params: HestonParams
    rmse_vol: float
    feller_ok: bool
    n_points: int
    n_expiries: int


def _model_vols(F, K, T, p: HestonParams):
    """Heston implied vols on the forward: price calls by COS, invert to vol."""
    calls = np.atleast_1d(price_cos(F, K, 0.0, 0.0, T, p, "C"))
    return np.array([implied_vol(c, F, k, T, 1.0, "C") for c, k in zip(calls, np.atleast_1d(K))])


def calibrate(surface, use_otm: bool = True, weight_by_liquidity: bool = True, max_nfev: int = 600) -> HestonFit:
    """Fit Heston (kappa, theta, xi, rho, v0) to an implied-vol surface frame.

    Works on the forward (F, discount already in the surface columns), fits in
    implied-vol space so the residual is directly in vol points, and weights by
    the inverse relative spread when available.
    """
    sl = surface.dropna(subset=["iv", "forward", "T"])
    if use_otm and "otm" in sl.columns:
        sl = sl[sl["otm"]]

    groups, atm = [], []
    for _, g in sl.groupby("expiry"):
        F = float(g["forward"].iloc[0])
        T = float(g["T"].iloc[0])
        K = g["strike"].to_numpy(dtype=float)
        iv = g["iv"].to_numpy(dtype=float)
        if "rel_spread" in g.columns and weight_by_liquidity and g["rel_spread"].notna().any():
            sw = np.sqrt(1.0 / np.clip(g["rel_spread"].to_numpy(dtype=float), 1e-3, None))
        else:
            sw = np.ones_like(iv)
        groups.append((F, T, K, iv, sw))
        atm.append((T, float(iv[np.argmin(np.abs(K / F - 1.0))])))

    if len(groups) < 2:
        raise ValueError("need at least two expiries to calibrate Heston")

    atm.sort()
    v0_0 = atm[0][1] ** 2          # front at-the-money variance
    theta_0 = atm[-1][1] ** 2      # long at-the-money variance
    x0 = np.array([2.0, theta_0, 0.6, -0.5, v0_0])
    lb = np.array([0.1, 1e-4, 1e-2, -0.999, 1e-4])
    ub = np.array([20.0, 4.0, 10.0, 0.99, 4.0])
    x0 = np.clip(x0, lb, ub)

    def residuals(x):
        p = HestonParams(*x)
        out = []
        for F, T, K, iv_mkt, sw in groups:
            iv_mod = _model_vols(F, K, T, p)
            r = sw * (np.where(np.isfinite(iv_mod), iv_mod, iv_mkt) - iv_mkt)
            out.append(r)
        return np.concatenate(out)

    sol = least_squares(residuals, x0, bounds=(lb, ub), method="trf", max_nfev=max_nfev)
    p = HestonParams(*sol.x)

    # unweighted RMSE in vol points
    diffs = []
    for F, T, K, iv_mkt, _ in groups:
        iv_mod = _model_vols(F, K, T, p)
        diffs.append((np.where(np.isfinite(iv_mod), iv_mod, iv_mkt) - iv_mkt))
    diffs = np.concatenate(diffs)
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    return HestonFit(p, rmse, feller_ok(p), diffs.size, len(groups))
