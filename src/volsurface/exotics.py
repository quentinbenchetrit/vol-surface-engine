"""Path-dependent and structured payoffs priced by Monte Carlo.

Three products, chosen because each exposes a different modelling issue:

  - **Barriers** are where discrete monitoring bites. A simulation only looks at
    the barrier on its grid, so it misses excursions between two steps and
    systematically overprices a knock-out. Two standard fixes are implemented:
    the Brownian-bridge crossing probability, which corrects in expectation, and
    the Broadie-Glasserman-Kou barrier shift, which moves the barrier by
    0.5826 * sigma * sqrt(dt) to absorb the bias.

  - **Autocallables** are the workhorse structured product: periodic observation
    dates, a coupon while the underlying holds up, early redemption, and capital
    at risk below a final barrier. Short volatility and short skew by nature.

  - **Cliquets** reset their strike at each period, so they price the *forward*
    smile rather than today's smile. That is exactly where a local volatility
    model and a stochastic volatility model calibrated to the same vanillas part
    company.
"""
from __future__ import annotations

import numpy as np

BGK = 0.5826  # Broadie, Glasserman and Kou barrier shift constant


def _bridge_survival(paths, barrier, variance, dt, direction="down"):
    """Probability a path never crossed the barrier between grid points.

    For a Brownian bridge between S_t and S_{t+dt} with variance v dt, the
    probability of having touched a barrier B while both endpoints stay on the
    same side is exp(-2 ln(S_t/B) ln(S_{t+dt}/B) / (v dt)).
    """
    s0, s1 = paths[:-1], paths[1:]
    v = np.maximum(variance, 1e-12)
    a = np.log(np.maximum(s0, 1e-300) / barrier)
    b = np.log(np.maximum(s1, 1e-300) / barrier)
    hit = np.exp(-2.0 * a * b / (v * dt))
    if direction == "down":
        alive = (s0 > barrier) & (s1 > barrier)
    else:
        alive = (s0 < barrier) & (s1 < barrier)
    step_survival = np.where(alive, 1.0 - hit, 0.0)
    return np.prod(step_survival, axis=0)


def barrier_payoff(paths, K, barrier, T, option="C", kind="do", variance=None,
                   correction="bridge", vol_hint=None):
    """Knock-out or knock-in barrier payoff on a simulated path grid.

    ``kind`` is one of "do", "di", "uo", "ui". ``correction`` is "bridge"
    (needs ``variance``, the per-step variance grid), "shift" (needs a scalar
    ``vol_hint``), or "none" for the raw discretely monitored payoff.
    """
    kind = kind.lower()
    down = kind[0] == "d"
    knock_out = kind[1] == "o"
    ST = paths[-1]
    vanilla = np.maximum(ST - K, 0.0) if str(option).upper().startswith("C") else np.maximum(K - ST, 0.0)
    n_steps = paths.shape[0] - 1
    dt = T / n_steps

    B = barrier
    if correction == "shift":
        if vol_hint is None:
            raise ValueError("the shift correction needs vol_hint")
        # Discrete monitoring misses excursions, so a knock-out survives too
        # often. Moving the barrier towards the spot makes the discrete grid
        # knock out as readily as continuous monitoring would.
        B = barrier * np.exp((1.0 if down else -1.0) * BGK * vol_hint * np.sqrt(dt))

    if correction == "bridge":
        if variance is None:
            raise ValueError("the bridge correction needs the variance grid")
        survival = _bridge_survival(paths, B, variance[:-1], dt, "down" if down else "up")
    else:
        breached = (paths[1:] <= B).any(axis=0) if down else (paths[1:] >= B).any(axis=0)
        survival = (~breached).astype(float)

    return vanilla * survival if knock_out else vanilla * (1.0 - survival)


def autocall_payoff(paths, obs_idx, coupon, autocall_barrier, protection_barrier,
                    notional=100.0, memory=True):
    """Payoff of a standard autocallable note.

    At each observation date the note redeems early at par plus accrued coupons
    if the underlying is at or above ``autocall_barrier`` (as a fraction of the
    initial level). If it survives to maturity, capital is returned in full when
    the final level is above ``protection_barrier``, and takes the downside one
    for one otherwise. With ``memory`` the missed coupons are paid on the first
    date the level recovers.
    """
    S0 = paths[0]
    n_paths = paths.shape[1]
    payoff = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)

    for i, idx in enumerate(obs_idx, start=1):
        level = paths[idx] / S0
        trigger = alive & (level >= autocall_barrier)
        if np.any(trigger):
            n_coupons = i if memory else 1
            payoff[trigger] = notional * (1.0 + coupon * n_coupons)
            alive &= ~trigger

    if np.any(alive):
        final = paths[-1] / S0
        safe = alive & (final >= protection_barrier)
        lost = alive & (final < protection_barrier)
        payoff[safe] = notional
        payoff[lost] = notional * final[lost]
    return payoff


def cliquet_payoff(paths, reset_idx, local_cap=0.08, local_floor=-0.08,
                   global_floor=0.0, notional=100.0):
    """Payoff of a cliquet: capped and floored periodic returns, summed.

    The strike resets at every observation, so the product depends on the
    distribution of *future* smiles rather than today's, which is why local and
    stochastic volatility models disagree on it.
    """
    idx = np.asarray(reset_idx)
    levels = paths[idx]
    returns = levels[1:] / levels[:-1] - 1.0
    clipped = np.clip(returns, local_floor, local_cap)
    total = np.maximum(clipped.sum(axis=0), global_floor)
    return notional * total
