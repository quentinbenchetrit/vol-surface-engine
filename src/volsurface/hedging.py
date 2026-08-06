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

from . import mc
from .black import black76_delta, black76_price
from .heston import HestonParams, price_cos
from .impliedvol import implied_vol


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


def hedge_under_gbm(S0, K, T, sell_vol, hedge_vol, realized_vol, r=0.0,
                    n_paths=40_000, n_steps=256, seed=None) -> HedgeResult:
    """Sell at one volatility, hedge at another, let the market realize a third.

    This isolates what a volatility view earns. Selling at ``sell_vol`` when the
    market realizes less is profitable, and the expected edge is the difference
    of the two Black-Scholes values. Which volatility is used in the *delta*
    decides the shape of that profit, not its average:

      - hedging at the realized volatility makes the result deterministic in the
        continuous limit, so the spread collapses as rebalancing gets finer,
      - hedging at the implied volatility leaves a path-dependent result whose
        spread does not vanish, because the profit is then accumulated as a
        gamma-weighted integral along whichever path the market happens to take.
    """
    price_sell, _ = black_scholes_call(K, T, sell_vol, r)
    _, delta_fn = black_scholes_call(K, T, hedge_vol, r)
    premium = float(price_sell(np.array([S0]), 0.0)[0])
    _, paths = mc.simulate_gbm(S0, T, realized_vol, r, n_paths=n_paths,
                               n_steps=n_steps, seed=seed)
    return delta_hedge(paths, T, premium, lambda ST: np.maximum(ST - K, 0.0), delta_fn, r)


def hedge_under_heston(S0, K, T, params: HestonParams, r=0.0, q=0.0,
                       hedge_vol=None, n_paths=40_000, n_steps=256,
                       seed=None) -> HedgeResult:
    """Sell a Heston-priced call and hedge it with a Black-Scholes delta.

    The option is sold at its own model price, so the hedge is unbiased, but a
    delta on the spot alone cannot touch the variance risk. The leftover spread
    therefore flattens out instead of vanishing as rebalancing gets finer: that
    residual is model error, and no amount of rebalancing removes it. Killing it
    needs a second option in the hedge.
    """
    premium = float(price_cos(S0, K, r, q, T, params, "C"))
    if hedge_vol is None:
        hedge_vol = implied_vol(premium, S0 * np.exp((r - q) * T), K, T, np.exp(-r * T), "C")
    _, delta_fn = black_scholes_call(K, T, hedge_vol, r, q)
    _, paths = mc.simulate(S0, T, params, r, q, n_paths=n_paths, n_steps=n_steps,
                           seed=seed, return_paths=True)
    return delta_hedge(paths, T, premium, lambda ST: np.maximum(ST - K, 0.0), delta_fn, r)
