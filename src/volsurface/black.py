"""Black-76 pricing for options on a forward.

Working on the forward rather than the spot keeps the discounting separate and
matches how the surface is quoted: the smile lives in log-forward-moneyness
k = ln(K / F). All functions are vectorized over numpy arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _broadcast(F, K, T, sigma):
    return [np.asarray(x, dtype=float) for x in np.broadcast_arrays(F, K, T, sigma)]


def _d1_d2(F, K, T, sigma):
    vol = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / vol
    d2 = d1 - vol
    return d1, d2


def _is_call(opt_type):
    arr = np.asarray(opt_type)
    return np.char.upper(arr.astype("U1")) == "C"


def black76_price(F, K, T, sigma, discount=1.0, opt_type="C"):
    """Black-76 price of a call or put on the forward.

    Degenerate inputs (non-positive time or vol) return the discounted intrinsic
    value rather than a divide-by-zero, so the pricer is safe to call at expiry.
    """
    F, K, T, sigma = _broadcast(F, K, T, sigma)
    is_call = _is_call(opt_type)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    alive = (T > 0.0) & (sigma > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1_d2(F, K, T, sigma)
        call = F * ndtr(d1) - K * ndtr(d2)
        put = K * ndtr(-d2) - F * ndtr(-d1)
    und = np.where(is_call, call, put)
    und = np.where(alive, und, intrinsic)
    return np.asarray(discount) * und


def black76_vega(F, K, T, sigma, discount=1.0):
    """Sensitivity of the price to sigma. Identical for calls and puts."""
    F, K, T, sigma = _broadcast(F, K, T, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, _ = _d1_d2(F, K, T, sigma)
        vega = F * np.exp(-0.5 * d1 * d1) / _SQRT_2PI * np.sqrt(T)
    vega = np.where((T > 0.0) & (sigma > 0.0), vega, 0.0)
    return np.asarray(discount) * vega


def black76_delta(F, K, T, sigma, opt_type="C"):
    """Driftless (forward) delta N(d1) for a call, N(d1) - 1 for a put.

    This is the market convention for indexing a smile by delta, bounded in
    (0, 1) and (-1, 0). The discount factor is left out on purpose so the delta
    axis stays clean and does not inherit the short-expiry discount noise; the
    hedge ratio in forward terms is just this times the discount factor.
    """
    F, K, T, sigma = _broadcast(F, K, T, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, _ = _d1_d2(F, K, T, sigma)
    call = ndtr(d1)
    put = ndtr(d1) - 1.0
    return np.where(_is_call(opt_type), call, put)
