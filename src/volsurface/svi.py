"""Raw SVI smile parametrization with a butterfly no-arbitrage constraint.

For a single maturity the total implied variance w = sigma_impl^2 * T is fitted
as a function of log-forward-moneyness k = ln(K / F) by the raw SVI form of
Gatheral (2004):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))

with a level a, wing slope b >= 0, skew |rho| < 1, horizontal shift m and ATM
curvature s > 0. No butterfly arbitrage is equivalent to a non-negative
risk-neutral density, which Gatheral encodes through a function g(k) >= 0.
The fit enforces g >= 0 on a grid, so the calibrated slice is arbitrage-free.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np
from scipy.optimize import least_squares, minimize


class SVIParams(NamedTuple):
    a: float
    b: float
    rho: float
    m: float
    s: float


class SVIFit(NamedTuple):
    params: SVIParams
    rmse_vol: float          # root mean square error in implied vol points
    min_g: float             # smallest density value on the grid (>= 0 is arb-free)
    arbitrage_free: bool
    n_points: int


def total_variance(k, p: SVIParams):
    k = np.asarray(k, dtype=float)
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.s * p.s))


def implied_vol(k, p: SVIParams, T: float):
    return np.sqrt(np.maximum(total_variance(k, p), 0.0) / T)


def _w_derivs(k, p: SVIParams):
    """Return w, w', w'' at k (analytic, since SVI is smooth)."""
    x = np.asarray(k, dtype=float) - p.m
    root = np.sqrt(x * x + p.s * p.s)
    w = p.a + p.b * (p.rho * x + root)
    wp = p.b * (p.rho + x / root)
    wpp = p.b * p.s * p.s / root ** 3
    return w, wp, wpp


def density_g(k, p: SVIParams):
    """Gatheral's g(k); the risk-neutral density is non-negative iff g >= 0."""
    w, wp, wpp = _w_derivs(k, p)
    return (1.0 - k * wp / (2.0 * w)) ** 2 - (wp * wp / 4.0) * (1.0 / w + 0.25) + wpp / 2.0


def min_total_variance(p: SVIParams) -> float:
    """Minimum of w over all k, reached at k = m - rho*s/sqrt(1-rho^2).

    Must be non-negative for the slice to be a valid variance curve.
    """
    return p.a + p.b * p.s * math.sqrt(max(0.0, 1.0 - p.rho * p.rho))


def atm_total_variance(p: SVIParams) -> float:
    """Total variance at the forward, k = 0. Feeds the SSVI term structure."""
    return float(total_variance(0.0, p))


def atm_vol(p: SVIParams, T: float) -> float:
    return math.sqrt(max(atm_total_variance(p), 0.0) / T)


def _bounds(k, w):
    return (
        [1e-8, 1e-6, -0.999, float(np.min(k)) - 1.0, 1e-4],
        [2.0 * float(np.max(w)) + 1e-3, 5.0, 0.999, float(np.max(k)) + 1.0, 2.0],
    )


def fit_slice(k, w, weights=None, enforce_butterfly: bool = True, T: Optional[float] = None, eps_w: float = 1e-8) -> SVIFit:
    """Calibrate one SVI slice to total-variance points (k, w)."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    base = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    # Fit in implied-vol space: a total-variance residual dw is a vol error of
    # dw / (2*sqrt(w*T)), so weighting by 1/(4 w T) stops the wings from
    # dominating and lines the objective up with the reported vol RMSE.
    if T is not None:
        base = base / (4.0 * np.maximum(w, 1e-8) * T)
    sqrt_w = np.sqrt(base)

    lb, ub = _bounds(k, w)
    i_min = int(np.argmin(w))
    x0 = [max(1e-6, 0.5 * float(np.min(w))), 0.1, -0.3, float(k[i_min]), 0.1]
    x0 = np.clip(x0, lb, ub)

    def residuals(theta):
        return sqrt_w * (total_variance(k, SVIParams(*theta)) - w)

    sol = least_squares(residuals, x0, bounds=(lb, ub), method="trf", max_nfev=4000)
    params = SVIParams(*sol.x)

    grid = np.linspace(k.min() - 0.2, k.max() + 0.2, 60)
    infeasible = float(np.min(density_g(grid, params))) < 0.0 or min_total_variance(params) < eps_w
    if enforce_butterfly and infeasible:
        def objective(theta):
            r = residuals(theta)
            return 0.5 * float(r @ r)

        cons = [
            {"type": "ineq", "fun": lambda theta: density_g(grid, SVIParams(*theta))},
            {"type": "ineq", "fun": lambda theta: min_total_variance(SVIParams(*theta)) - eps_w},
        ]
        pol = minimize(
            objective, sol.x, method="SLSQP",
            bounds=list(zip(lb, ub)), constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if pol.success:
            params = SVIParams(*pol.x)

    min_g = float(np.min(density_g(grid, params)))
    w_model = total_variance(k, params)
    if T is not None:
        err = np.sqrt(np.maximum(w_model, 0) / T) - np.sqrt(np.maximum(w, 0) / T)
    else:
        err = w_model - w
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return SVIFit(params, rmse, min_g, min_g >= -1e-8, len(k))


def calibrate_slices(surface, min_points: int = 5, use_otm: bool = True):
    """Fit an SVI slice per expiry from an implied-vol surface frame.

    Expects the columns produced by :func:`volsurface.implied_vol_surface`
    (``expiry``, ``T``, ``log_moneyness``, ``iv``, ``total_var``, ``otm`` and
    optionally ``rel_spread``). Returns a list of per-expiry fits.
    """
    out = []
    for expiry, grp in surface.groupby("expiry"):
        sl = grp.dropna(subset=["iv", "total_var", "log_moneyness"])
        if use_otm and "otm" in sl.columns:
            sl = sl[sl["otm"]]
        if len(sl) < min_points:
            continue
        weights = None
        if "rel_spread" in sl.columns and sl["rel_spread"].notna().any():
            weights = 1.0 / np.clip(sl["rel_spread"].to_numpy(), 1e-3, None)
        fit = fit_slice(sl["log_moneyness"].to_numpy(), sl["total_var"].to_numpy(),
                        weights=weights, T=float(sl["T"].iloc[0]))
        out.append({"expiry": expiry, "T": float(sl["T"].iloc[0]), **fit.params._asdict(),
                    "rmse_vol": fit.rmse_vol, "min_g": fit.min_g,
                    "arbitrage_free": fit.arbitrage_free, "n_points": fit.n_points})
    return out
