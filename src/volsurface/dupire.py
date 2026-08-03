"""Dupire local volatility, differentiated analytically from the SSVI surface.

Dupire's equation in total-variance form (Gatheral) reads

    sigma_loc^2(k, T) = w_T / [ (1 - k w_k / (2w))^2
                                - (w_k^2 / 4) (1/w + 1/4)
                                + w_kk / 2 ]

and the denominator is exactly Gatheral's g(k), the function whose positivity is
the butterfly no-arbitrage condition. So

    sigma_loc^2 = w_T / g(k),

which makes the whole chain explicit: butterfly (g >= 0) and calendar (w_T >= 0)
are precisely what keep the local variance positive. Both are enforced by the
SSVI fit, so the local volatility comes out well posed by construction.

Working off the fitted surface rather than raw quotes matters. Dupire needs a
second derivative in strike, which is unstable on noisy market prices and can go
negative; on SSVI every derivative is analytic.

Under SSVI the maturity enters only through theta(T), so the chain rule gives
w_T = (dw/dtheta) * theta'(T), with dw/dtheta available in closed form.
"""
from __future__ import annotations

import numpy as np

from . import ssvi
from .svi import density_g


def dw_dtheta(k, theta, params: ssvi.SSVIParams):
    """Partial derivative of SSVI total variance with respect to theta."""
    k = np.asarray(k, dtype=float)
    theta = np.asarray(theta, dtype=float)
    rho = params.rho
    phi = ssvi.phi_power(theta, params.eta, params.gamma)
    psi = phi * k
    root = np.sqrt((psi + rho) ** 2 + (1.0 - rho * rho))
    w = 0.5 * theta * (1.0 + rho * psi + root)
    # phi = eta * theta^-gamma, so dpsi/dtheta = -gamma * psi / theta
    return w / theta - 0.5 * params.gamma * psi * (rho + (psi + rho) / root)


def dw_dT(k, T, params: ssvi.SSVIParams):
    """Maturity derivative of total variance, by the chain rule through theta."""
    theta = ssvi.theta_at(T, params)
    return dw_dtheta(k, theta, params) * ssvi.dtheta_dT(T, params)


def local_variance(k, T, params: ssvi.SSVIParams, eps: float = 1e-8):
    """Dupire local variance sigma_loc^2 at log-moneyness k and maturity T."""
    k = np.asarray(k, dtype=float)
    theta = ssvi.theta_at(T, params)
    g = density_g(k, ssvi.to_svi_vec(theta, params))
    wT = dw_dT(k, T, params)
    return np.where(g > eps, wT / np.maximum(g, eps), np.nan)


def local_vol(k, T, params: ssvi.SSVIParams):
    """Dupire local volatility, the square root of the local variance."""
    return np.sqrt(np.maximum(local_variance(k, T, params), 0.0))
