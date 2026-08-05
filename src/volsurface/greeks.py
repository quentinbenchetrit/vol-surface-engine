"""Monte Carlo greeks by three routes, each with a different failure mode.

**Finite differences** reprice at bumped inputs. Simple and universal, but the
two runs must share their random draws: with independent draws the difference of
two noisy prices is dominated by that noise, and the estimator is useless. Using
common random numbers is what makes it work at all.

**Pathwise derivatives** differentiate the payoff along each path. Under Heston
the terminal spot scales with the initial spot, so dS_T/dS_0 = S_T / S_0 and the
call delta is E[e^{-rT} 1{S_T > K} S_T / S_0]. Very low variance, but it needs a
payoff that is differentiable almost everywhere with an integrable derivative.
A digital fails that test: its derivative is zero wherever it exists, so the
estimator collapses to zero and is simply wrong.

**Likelihood ratio** differentiates the density instead of the payoff, so it
does not care whether the payoff is continuous. Shifting ln S_0 translates the
first log-price step, which gives the score Z_1 / (S_0 sigma_1). It handles
digitals and barriers, at the cost of noticeably more variance.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from . import mc
from .heston import HestonParams


class GreekResult(NamedTuple):
    value: float
    stderr: float
    method: str

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.value - 1.96 * self.stderr, self.value + 1.96 * self.stderr)


def call_payoff(K):
    return lambda ST: np.maximum(ST - K, 0.0)


def digital_payoff(K):
    return lambda ST: (ST > K).astype(float)


def _summarise(sample, method) -> GreekResult:
    sample = np.asarray(sample, dtype=float)
    return GreekResult(float(sample.mean()),
                       float(sample.std(ddof=1) / np.sqrt(sample.size)),
                       method)


def delta_pathwise(S0, K, T, p: HestonParams, r=0.0, q=0.0, **kwargs) -> GreekResult:
    """Pathwise delta of a European call. Not valid for discontinuous payoffs."""
    ST = mc.simulate(S0, T, p, r, q, **kwargs)
    est = np.exp(-r * T) * (ST > K) * ST / S0
    return _summarise(est, "pathwise")


def delta_likelihood_ratio(S0, K, T, p: HestonParams, r=0.0, q=0.0,
                           payoff: Callable | None = None, **kwargs) -> GreekResult:
    """Likelihood-ratio delta, valid for any payoff including discontinuous ones."""
    ST, score = mc.simulate(S0, T, p, r, q, return_score=True, **kwargs)
    f = payoff if payoff is not None else call_payoff(K)
    est = np.exp(-r * T) * f(ST) * score
    return _summarise(est, "likelihood ratio")


def delta_finite_difference(S0, K, T, p: HestonParams, r=0.0, q=0.0, h=None,
                            common_random=True, payoff: Callable | None = None,
                            **kwargs) -> GreekResult:
    """Central finite-difference delta.

    ``common_random`` reuses the same uniforms for both bumped runs, which is
    what keeps the difference from being swamped by Monte Carlo noise.
    """
    h = h if h is not None else 0.01 * S0
    f = payoff if payoff is not None else call_payoff(K)
    n_steps = kwargs.get("n_steps", 32)
    n_paths = kwargs.get("n_paths", 100_000)

    if common_random:
        U = mc.draw_uniforms(n_paths, n_steps, kwargs.get("method", "pseudo"),
                             kwargs.get("seed"), kwargs.get("antithetic", False))
        up = mc.simulate(S0 + h, T, p, r, q, n_steps=n_steps, uniforms=U)
        dn = mc.simulate(S0 - h, T, p, r, q, n_steps=n_steps, uniforms=U)
        est = np.exp(-r * T) * (f(up) - f(dn)) / (2.0 * h)
        return _summarise(est, "finite difference (common randoms)")

    seed = kwargs.get("seed") or 0
    up = mc.simulate(S0 + h, T, p, r, q, **{**kwargs, "seed": seed})
    dn = mc.simulate(S0 - h, T, p, r, q, **{**kwargs, "seed": seed + 10_000})
    pu, pd_ = np.exp(-r * T) * f(up), np.exp(-r * T) * f(dn)
    value = (pu.mean() - pd_.mean()) / (2.0 * h)
    stderr = np.hypot(pu.std(ddof=1), pd_.std(ddof=1)) / np.sqrt(pu.size) / (2.0 * h)
    return GreekResult(float(value), float(stderr), "finite difference (independent)")


def vega_finite_difference(S0, K, T, p: HestonParams, r=0.0, q=0.0, h=1e-3,
                           payoff: Callable | None = None, **kwargs) -> GreekResult:
    """Sensitivity to the initial variance v0, by common-random-number bumping.

    Reported per unit of v0. Heston has several quantities a desk might call
    vega; this is the one the model itself exposes.
    """
    f = payoff if payoff is not None else call_payoff(K)
    n_steps = kwargs.get("n_steps", 32)
    n_paths = kwargs.get("n_paths", 100_000)
    U = mc.draw_uniforms(n_paths, n_steps, kwargs.get("method", "pseudo"),
                         kwargs.get("seed"), kwargs.get("antithetic", False))
    up = mc.simulate(S0, T, p._replace(v0=p.v0 + h), r, q, n_steps=n_steps, uniforms=U)
    dn = mc.simulate(S0, T, p._replace(v0=max(p.v0 - h, 1e-8)), r, q, n_steps=n_steps, uniforms=U)
    est = np.exp(-r * T) * (f(up) - f(dn)) / (2.0 * h)
    return _summarise(est, "vega, finite difference")
