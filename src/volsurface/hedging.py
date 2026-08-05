"""Replay a discrete delta hedge and measure what is left over.

Selling an option and hedging it with the model's own delta is the sharpest
self-consistency check there is: if the pricer and the greeks agree, the
replication portfolio should track the payoff, and the leftover profit and loss
should sit at zero with a spread that shrinks as rebalancing gets finer.

The bookkeeping is the usual self-financing one. At inception the premium comes
in and delta shares are bought, so the cash account holds V0 - delta_0 S_0.
Between rebalancing dates cash earns the risk-free rate, and at each date the
change in the hedge is paid for out of cash. At maturity the shares are sold and
the payoff is paid away. What is left is the hedging error.

Only the engine lives here. It takes the paths and a delta function, so it works
just as well on Black-Scholes deltas, Heston deltas, or a deliberately wrong
delta when the point is to show what mispricing the hedge costs.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from .black import black76_delta, black76_price


class HedgeResult(NamedTuple):
    pnl: np.ndarray          # hedging error per path
    n_rebalances: int

    @property
    def mean(self) -> float:
        return float(self.pnl.mean())

    @property
    def std(self) -> float:
        return float(self.pnl.std(ddof=1))

    @property
    def stderr(self) -> float:
        return float(self.pnl.std(ddof=1) / np.sqrt(self.pnl.size))


def delta_hedge(paths, T, premium, payoff: Callable[[np.ndarray], np.ndarray],
                delta_fn: Callable[[np.ndarray, float], np.ndarray],
                r: float = 0.0) -> HedgeResult:
    """Replay a short option position hedged at the grid of ``paths``.

    ``paths`` has shape (n_steps + 1, n_paths). ``delta_fn(S, t)`` returns the
    hedge ratio at spot ``S`` and calendar time ``t``; it is never called at
    maturity, where the position is simply unwound. A positive result means the
    hedge made money against the option sold.
    """
    paths = np.asarray(paths, dtype=float)
    n_steps = paths.shape[0] - 1
    dt = T / n_steps
    growth = np.exp(r * dt)

    delta = np.asarray(delta_fn(paths[0], 0.0), dtype=float)
    cash = premium - delta * paths[0]

    for step in range(1, n_steps):
        cash *= growth
        new_delta = np.asarray(delta_fn(paths[step], step * dt), dtype=float)
        cash -= (new_delta - delta) * paths[step]
        delta = new_delta

    cash *= growth
    ST = paths[-1]
    return HedgeResult(cash + delta * ST - np.asarray(payoff(ST), dtype=float), n_steps)


def black_scholes_call(K, T, sigma, r=0.0, q=0.0):
    """Price and delta functions for a European call, for use with the engine.

    Black-76 on the forward is reused: with F = S e^{(r-q)(T-t)} it is the same
    model, and the delta it returns is already the driftless N(d1) the hedge
    needs, scaled to the spot.
    """
    def price(S, t):
        tau = max(T - t, 0.0)
        S = np.asarray(S, dtype=float)
        if tau <= 0.0:
            return np.maximum(S - K, 0.0)
        F = S * np.exp((r - q) * tau)
        return np.asarray(black76_price(F, K, tau, sigma, np.exp(-r * tau), "C"), dtype=float)

    def delta(S, t):
        tau = max(T - t, 0.0)
        S = np.asarray(S, dtype=float)
        if tau <= 0.0:
            return (S > K).astype(float)
        F = S * np.exp((r - q) * tau)
        # dV/dS = e^{-q tau} N(d1) for a spot-delta hedge
        return np.exp(-q * tau) * np.asarray(black76_delta(F, K, tau, sigma, "C"), dtype=float)

    return price, delta
