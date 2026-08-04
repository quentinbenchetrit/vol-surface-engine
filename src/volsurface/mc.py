"""Monte Carlo simulation of Heston, with the Andersen QE scheme.

A plain Euler discretisation of the CIR variance goes negative, which breaks
sqrt(v) and biases prices. Andersen's Quadratic Exponential scheme instead
samples the next variance from a distribution matched to its first two
conditional moments: a shifted square of a normal when the variance is high
(psi <= psi_c), and a mass at zero plus an exponential tail when it is low.

The log-spot is then advanced with the broadly used central discretisation,

    ln S_{t+dt} = ln S_t + (r - q) dt + K0 + K1 v_t + K2 v_{t+dt}
                  + sqrt(K3 v_t + K4 v_{t+dt}) Z,

with Z drawn independently of the variance: the spot/variance correlation is
carried by the K1 and K2 terms rather than by correlating the normals. K0 is
replaced by the exact martingale correction, chosen so that

    E[S_{t+dt} | S_t, v_t] = S_t e^{(r - q) dt}

holds path by path, which makes the simulated forward unbiased.

A full-truncation Euler scheme is kept alongside as a baseline to show what the
QE scheme buys.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from .heston import HestonParams

PSI_C = 1.5


class MCResult(NamedTuple):
    price: float
    stderr: float
    n_paths: int
    n_steps: int

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr)


def simulate(S0, T, p: HestonParams, r=0.0, q=0.0, n_paths=100_000, n_steps=64,
             scheme="qe", seed=None, antithetic=False, return_paths=False):
    """Simulate Heston and return terminal spots, or the whole path grid.

    ``scheme`` is "qe" (Andersen) or "euler" (full truncation baseline).
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        if n_paths % 2:
            n_paths += 1
        half = n_paths // 2
    dt = T / n_steps

    v = np.full(n_paths, p.v0, dtype=float)
    x = np.full(n_paths, np.log(S0), dtype=float)
    paths = np.empty((n_steps + 1, n_paths)) if return_paths else None
    if return_paths:
        paths[0] = np.exp(x)

    rho, xi, kappa, theta = p.rho, p.xi, p.kappa, p.theta
    g1 = g2 = 0.5                       # central discretisation
    K1 = g1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    K2 = g2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)
    A = K2 + 0.5 * K4

    for step in range(n_steps):
        if scheme == "qe":
            e = np.exp(-kappa * dt)
            m = theta + (v - theta) * e
            s2 = (v * xi ** 2 * e / kappa) * (1.0 - e) + (theta * xi ** 2 / (2.0 * kappa)) * (1.0 - e) ** 2
            psi = np.where(m > 0, s2 / np.maximum(m * m, 1e-300), np.inf)
            quad = psi <= PSI_C

            v_next = np.empty_like(v)
            k0 = np.empty_like(v)

            if np.any(quad):
                inv_psi = 1.0 / psi[quad]
                b2 = 2.0 * inv_psi - 1.0 + np.sqrt(2.0 * inv_psi) * np.sqrt(np.maximum(2.0 * inv_psi - 1.0, 0.0))
                b = np.sqrt(b2)
                a = m[quad] / (1.0 + b2)
                if np.any(A * a >= 0.5):
                    raise ValueError("martingale correction undefined: reduce the step size")
                zv = rng.standard_normal(b.shape)
                v_next[quad] = a * (b + zv) ** 2
                # E[exp(A v')] for the quadratic branch
                k0[quad] = -(A * a * b2) / (1.0 - 2.0 * A * a) + 0.5 * np.log(1.0 - 2.0 * A * a)

            if np.any(~quad):
                psi_e = psi[~quad]
                pz = (psi_e - 1.0) / (psi_e + 1.0)
                beta = (1.0 - pz) / np.maximum(m[~quad], 1e-300)
                if np.any(A >= beta):
                    raise ValueError("martingale correction undefined: reduce the step size")
                u = rng.random(psi_e.shape)
                v_next[~quad] = np.where(u <= pz, 0.0,
                                         np.log(np.maximum((1.0 - pz) / (1.0 - u), 1e-300)) / beta)
                # E[exp(A v')] for the exponential branch
                k0[~quad] = -np.log(pz + (1.0 - pz) * beta / (beta - A))

            k0 = k0 - K1 * v - 0.5 * K3 * v
            zs = rng.standard_normal(n_paths)
            if antithetic:
                zs[half:] = -zs[:half]
            x = x + (r - q) * dt + k0 + K1 * v + K2 * v_next + np.sqrt(np.maximum(K3 * v + K4 * v_next, 0.0)) * zs
            v = v_next
        elif scheme == "euler":
            z1 = rng.standard_normal(n_paths)
            z2 = rho * z1 + np.sqrt(1.0 - rho * rho) * rng.standard_normal(n_paths)
            vp = np.maximum(v, 0.0)                 # full truncation
            sq = np.sqrt(vp * dt)
            x = x + (r - q - 0.5 * vp) * dt + sq * z2
            v = v + kappa * (theta - vp) * dt + xi * sq * z1
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

        if return_paths:
            paths[step + 1] = np.exp(x)

    return (np.exp(x), paths) if return_paths else np.exp(x)


def price_european(S0, K, T, p: HestonParams, r=0.0, q=0.0, option="C", **kwargs):
    """Monte Carlo price of a European option, with its standard error."""
    n_paths = kwargs.get("n_paths", 100_000)
    n_steps = kwargs.get("n_steps", 64)
    ST = simulate(S0, T, p, r, q, **kwargs)
    payoff = np.maximum(ST - K, 0.0) if str(option).upper().startswith("C") else np.maximum(K - ST, 0.0)
    disc = np.exp(-r * T) * payoff
    return MCResult(float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(disc.size)), len(ST), n_steps)
