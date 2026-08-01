"""Black-76 pricing for options on a forward.

Working on the forward rather than the spot keeps the discounting separate and
matches how the surface is quoted: the smile lives in log-forward-moneyness
k = ln(K / F). All functions are vectorized over numpy arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _d1_d2(F, K, T, sigma):
    F, K, T, sigma = np.broadcast_arrays(F, K, T, sigma)
    vol = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / vol
    d2 = d1 - vol
    return d1, d2


def _is_call(opt_type):
    arr = np.asarray(opt_type)
    return np.char.upper(arr.astype("U1")) == "C"


def black76_price(F, K, T, sigma, discount=1.0, opt_type="C"):
    """Undiscounted-then-discounted Black-76 price of a call or put on the forward."""
    d1, d2 = _d1_d2(F, K, T, sigma)
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    und = np.where(_is_call(opt_type), call, put)
    return np.asarray(discount) * und


def black76_vega(F, K, T, sigma, discount=1.0):
    """Sensitivity of the price to sigma. Identical for calls and puts."""
    d1, _ = _d1_d2(F, K, T, sigma)
    pdf = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    return np.asarray(discount) * F * pdf * np.sqrt(T)
