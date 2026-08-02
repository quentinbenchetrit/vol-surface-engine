"""SSVI: a single arbitrage-free surface tying the SVI slices together in time.

Following Gatheral and Jacquier (2014), the whole surface is written in terms of
the at-the-money total variance term structure theta(T) and one skew function
phi(theta):

    w(k, theta) = (theta / 2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + (1 - rho^2)))

with the power-law choice phi(theta) = eta * theta^(-gamma). Two closed-form
conditions make the whole surface arbitrage-free:

  - no butterfly (static) arbitrage:   eta * (1 + |rho|) <= 2
  - no calendar arbitrage:             theta(T) non-decreasing in T

theta(T) is taken from the per-slice SVI at-the-money variances of the previous
step and forced non-decreasing; (rho, eta, gamma) are then fit globally to the
whole surface under the butterfly constraint.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from .svi import SVIParams, calibrate_slices
from .svi import total_variance as svi_total_variance


class SSVIParams(NamedTuple):
    rho: float
    eta: float
    gamma: float
    T_knots: np.ndarray      # maturities of the ATM variance term structure
    theta_knots: np.ndarray  # ATM total variance at those maturities (non-decreasing)


class SSVIFit(NamedTuple):
    params: SSVIParams
    rmse_vol: float
    butterfly_ok: bool
    calendar_ok: bool
    n_points: int
    n_expiries: int


def phi_power(theta, eta: float, gamma: float):
    return eta * np.power(theta, -gamma)


def theta_at(T, params: SSVIParams):
    """ATM total variance at maturity T, linearly interpolated and flat-extrapolated."""
    return np.interp(np.asarray(T, dtype=float), params.T_knots, params.theta_knots)


def total_variance(k, T, params: SSVIParams):
    theta = theta_at(T, params)
    phi = phi_power(theta, params.eta, params.gamma)
    k = np.asarray(k, dtype=float)
    rho = params.rho
    return 0.5 * theta * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1.0 - rho * rho)))


def implied_vol(k, T, params: SSVIParams):
    return np.sqrt(np.maximum(total_variance(k, T, params), 0.0) / np.asarray(T, dtype=float))


def to_svi(theta: float, params: SSVIParams) -> SVIParams:
    """Express one SSVI slice as raw SVI, so the Gatheral density can be reused."""
    phi = params.eta * theta ** (-params.gamma)
    root = math.sqrt(max(0.0, 1.0 - params.rho * params.rho))
    return SVIParams(
        a=0.5 * theta * (1.0 - params.rho * params.rho),
        b=0.5 * theta * phi,
        rho=params.rho,
        m=-params.rho / phi,
        s=root / phi,
    )


def butterfly_ok(params: SSVIParams) -> bool:
    return params.eta * (1.0 + abs(params.rho)) <= 2.0 + 1e-9


def calendar_ok(params: SSVIParams) -> bool:
    return bool(np.all(np.diff(params.theta_knots) >= -1e-12))


def calibrate(surface, use_otm: bool = True, min_points_per_slice: int = 5) -> SSVIFit:
    """Fit an arbitrage-free SSVI surface to an implied-vol surface frame."""
    slices = calibrate_slices(surface, min_points=min_points_per_slice, use_otm=use_otm)
    if len(slices) < 2:
        raise ValueError("need at least two calibrated expiries for a surface")
    slices = sorted(slices, key=lambda d: d["T"])
    T_knots = np.array([d["T"] for d in slices], dtype=float)
    theta_knots = np.array(
        [float(svi_total_variance(0.0, SVIParams(d["a"], d["b"], d["rho"], d["m"], d["s"]))) for d in slices]
    )
    theta_knots = np.maximum.accumulate(theta_knots)  # calendar: ATM variance non-decreasing

    sl = surface.dropna(subset=["iv", "total_var", "log_moneyness"])
    if use_otm and "otm" in sl.columns:
        sl = sl[sl["otm"]]
    k = sl["log_moneyness"].to_numpy(dtype=float)
    w = sl["total_var"].to_numpy(dtype=float)
    T = sl["T"].to_numpy(dtype=float)
    theta_pts = np.interp(T, T_knots, theta_knots)

    # weight in vol space, and by liquidity when the spread is available
    weight = 1.0 / (4.0 * np.maximum(w, 1e-8) * np.maximum(T, 1e-6))
    if "rel_spread" in sl.columns and sl["rel_spread"].notna().any():
        weight = weight / np.clip(sl["rel_spread"].to_numpy(dtype=float), 1e-3, None)
    sqrt_w = np.sqrt(weight)

    def model(rho, eta, gamma):
        phi = eta * np.power(theta_pts, -gamma)
        return 0.5 * theta_pts * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1.0 - rho * rho)))

    def objective(x):
        r = sqrt_w * (model(*x) - w)
        return 0.5 * float(r @ r)

    x0 = [-0.5, 1.0, 0.4]
    bounds = [(-0.999, 0.999), (1e-3, 10.0), (0.01, 0.5)]
    cons = [{"type": "ineq", "fun": lambda x: 2.0 - x[1] * (1.0 + abs(x[0]))}]  # butterfly
    sol = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 800, "ftol": 1e-12})
    rho, eta, gamma = sol.x
    params = SSVIParams(float(rho), float(eta), float(gamma), T_knots, theta_knots)

    w_model = model(rho, eta, gamma)
    err = np.sqrt(np.maximum(w_model, 0) / T) - np.sqrt(np.maximum(w, 0) / T)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return SSVIFit(params, rmse, butterfly_ok(params), calendar_ok(params), len(k), len(slices))
