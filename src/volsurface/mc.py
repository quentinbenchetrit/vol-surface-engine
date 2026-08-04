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

The whole simulation is driven by a fixed-dimension array of uniforms, two per
time step: one for the variance, one for the spot. In the quadratic branch the
uniform is mapped through the inverse normal, and the exponential branch
consumes it directly, so a single uniform advances the variance whichever
branch is taken. That is what makes the variance reduction work: antithetic
sampling is just U -> 1 - U, and quasi-Monte Carlo is just swapping the
pseudorandom uniforms for a scrambled Sobol sequence.

A full-truncation Euler scheme is kept alongside as a baseline to show what the
QE scheme buys.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .heston import HestonParams

PSI_C = 1.5
_EPS = 1e-12


class MCResult(NamedTuple):
    price: float
    stderr: float
    n_paths: int
    n_steps: int

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr)


def draw_uniforms(n_paths: int, n_steps: int, method: str = "pseudo",
                  seed=None, antithetic: bool = False) -> np.ndarray:
    """Uniforms driving the simulation, shaped (n_paths, 2 * n_steps).

    With ``antithetic`` only half the paths are drawn and mirrored to 1 - U.
    With ``method="sobol"`` a scrambled Sobol sequence replaces the pseudorandom
    draws; it is generated in a power-of-two block, which is what keeps the
    sequence balanced.
    """
    dim = 2 * n_steps
    n_draw = n_paths // 2 if antithetic else n_paths

    if method == "sobol":
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n_draw, 2))))
        U = sampler.random_base2(m)[:n_draw]
    elif method == "pseudo":
        U = np.random.default_rng(seed).random((n_draw, dim))
    else:
        raise ValueError(f"unknown sampling method {method!r}")

    if antithetic:
        U = np.concatenate([U, 1.0 - U], axis=0)
    return np.clip(U, _EPS, 1.0 - _EPS)


def simulate(S0, T, p: HestonParams, r=0.0, q=0.0, n_paths=100_000, n_steps=64,
             scheme="qe", seed=None, antithetic=False, method="pseudo",
             return_paths=False, return_variance=False,
             uniforms: Optional[np.ndarray] = None):
    """Simulate Heston and return terminal spots, or the whole path grid.

    ``scheme`` is "qe" (Andersen) or "euler" (full truncation baseline).
    ``method`` is "pseudo" or "sobol". Pass ``uniforms`` to supply the driving
    randomness directly, which is how paired estimators reuse a stream.
    """
    if uniforms is None:
        if antithetic and n_paths % 2:
            n_paths += 1
        uniforms = draw_uniforms(n_paths, n_steps, method, seed, antithetic)
    n_paths, dim = uniforms.shape
    if dim != 2 * n_steps:
        raise ValueError(f"uniforms must have {2 * n_steps} columns, got {dim}")

    dt = T / n_steps
    v = np.full(n_paths, p.v0, dtype=float)
    x = np.full(n_paths, np.log(S0), dtype=float)
    return_paths = return_paths or return_variance
    paths = np.empty((n_steps + 1, n_paths)) if return_paths else None
    vpaths = np.empty((n_steps + 1, n_paths)) if return_variance else None
    if return_paths:
        paths[0] = np.exp(x)
    if return_variance:
        vpaths[0] = v

    rho, xi, kappa, theta = p.rho, p.xi, p.kappa, p.theta
    g1 = g2 = 0.5                       # central discretisation
    K1 = g1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    K2 = g2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)
    A = K2 + 0.5 * K4
    e = np.exp(-kappa * dt)

    for step in range(n_steps):
        u_v = uniforms[:, 2 * step]
        u_s = uniforms[:, 2 * step + 1]

        if scheme == "qe":
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
                zv = ndtri(u_v[quad])
                v_next[quad] = a * (b + zv) ** 2
                # E[exp(A v')] for the quadratic branch
                k0[quad] = -(A * a * b2) / (1.0 - 2.0 * A * a) + 0.5 * np.log(1.0 - 2.0 * A * a)

            if np.any(~quad):
                psi_e = psi[~quad]
                pz = (psi_e - 1.0) / (psi_e + 1.0)
                beta = (1.0 - pz) / np.maximum(m[~quad], 1e-300)
                if np.any(A >= beta):
                    raise ValueError("martingale correction undefined: reduce the step size")
                u = u_v[~quad]
                v_next[~quad] = np.where(u <= pz, 0.0,
                                         np.log(np.maximum((1.0 - pz) / (1.0 - u), 1e-300)) / beta)
                # E[exp(A v')] for the exponential branch
                k0[~quad] = -np.log(pz + (1.0 - pz) * beta / (beta - A))

            k0 = k0 - K1 * v - 0.5 * K3 * v
            zs = ndtri(u_s)
            x = x + (r - q) * dt + k0 + K1 * v + K2 * v_next + np.sqrt(np.maximum(K3 * v + K4 * v_next, 0.0)) * zs
            v = v_next
        elif scheme == "euler":
            z1 = ndtri(u_v)
            z2 = rho * z1 + np.sqrt(1.0 - rho * rho) * ndtri(u_s)
            vp = np.maximum(v, 0.0)                 # full truncation
            sq = np.sqrt(vp * dt)
            x = x + (r - q - 0.5 * vp) * dt + sq * z2
            v = v + kappa * (theta - vp) * dt + xi * sq * z1
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

        if return_paths:
            paths[step + 1] = np.exp(x)
        if return_variance:
            vpaths[step + 1] = v

    if return_variance:
        return np.exp(x), paths, vpaths
    return (np.exp(x), paths) if return_paths else np.exp(x)


def simulate_local_vol(S0, T, ssvi_params, r=0.0, q=0.0, n_paths=100_000, n_steps=64,
                       seed=None, antithetic=False, method="pseudo",
                       return_paths=False, vol_bounds=(0.01, 5.0),
                       uniforms: Optional[np.ndarray] = None):
    """Simulate the Dupire local volatility diffusion off a fitted SSVI surface.

    The dynamics are dS/S = (r - q) dt + sigma_loc(S_t, t) dW, discretised in
    log space, with the local volatility read from the surface at the path's own
    log-forward-moneyness. This reprices every vanilla by construction, so it is
    the natural counterpart to Heston when asking whether two models that agree
    on vanillas also agree on exotics.

    One uniform per step is consumed, so the same antithetic and Sobol machinery
    applies. Local vol is clipped to ``vol_bounds`` because the surface is only
    meaningful where options trade.
    """
    from . import dupire

    if uniforms is None:
        if antithetic and n_paths % 2:
            n_paths += 1
        uniforms = draw_uniforms(n_paths, n_steps, method, seed, antithetic)[:, :n_steps]
    n_paths = uniforms.shape[0]

    dt = T / n_steps
    x = np.full(n_paths, np.log(S0), dtype=float)
    lnF0 = np.log(S0)
    paths = np.empty((n_steps + 1, n_paths)) if return_paths else None
    if return_paths:
        paths[0] = np.exp(x)

    lo, hi = vol_bounds
    for step in range(n_steps):
        t = step * dt
        lnF = lnF0 + (r - q) * t                     # forward at the current time
        k = x - lnF
        # the surface starts at its first knot; below that reuse the front slice
        t_eval = max(t, float(ssvi_params.T_knots[0]))
        sig = np.clip(dupire.local_vol(k, t_eval, ssvi_params), lo, hi)
        z = ndtri(uniforms[:, step])
        x = x + (r - q - 0.5 * sig * sig) * dt + sig * np.sqrt(dt) * z
        if return_paths:
            paths[step + 1] = np.exp(x)

    return (np.exp(x), paths) if return_paths else np.exp(x)


def control_variate(X, controls, means):
    """Combine a payoff with controls of known mean by the optimal coefficients.

    Solves for the c that minimises Var(X - c . (Y - E[Y])) by least squares on
    the centred controls, which is the multivariate version of
    c* = Cov(X, Y) / Var(Y). Returns the corrected sample.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(controls, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    elif Y.shape[0] != X.size:      # accept (n_controls, n_paths) too
        Y = Y.T
    means = np.atleast_1d(np.asarray(means, dtype=float))
    if Y.shape[1] != means.size:
        raise ValueError(f"got {Y.shape[1]} controls but {means.size} means")
    beta, *_ = np.linalg.lstsq(Y - Y.mean(axis=0), X - X.mean(), rcond=None)
    return X - (Y - means) @ beta


def _summarise(sample, n_steps, antithetic: bool = False) -> MCResult:
    """Mean and standard error of a discounted payoff sample.

    Antithetic paths come in negatively correlated pairs, so the independent
    unit is the pair average, not the path: averaging first is what lets the
    usual standard error formula apply and what makes the gain visible.
    """
    sample = np.asarray(sample, dtype=float)
    n = sample.size
    if antithetic and n % 2 == 0:
        half = n // 2
        units = 0.5 * (sample[:half] + sample[half:])
    else:
        units = sample
    return MCResult(float(sample.mean()),
                    float(units.std(ddof=1) / np.sqrt(units.size)),
                    n, n_steps)


def _rqmc_summarise(means, n_paths, n_steps) -> MCResult:
    """Randomised QMC: the error comes from the spread across scrambles.

    Sobol points are deliberately not independent, so the i.i.d. formula does
    not apply to them. Running several independent scrambles and looking at the
    dispersion of their means is the standard unbiased way to put an error bar
    on a quasi-Monte Carlo estimate.
    """
    means = np.asarray(means, dtype=float)
    return MCResult(float(means.mean()),
                    float(means.std(ddof=1) / np.sqrt(means.size)),
                    n_paths * means.size, n_steps)


def price_european(S0, K, T, p: HestonParams, r=0.0, q=0.0, option="C",
                   control=False, n_replicates: int = 1, **kwargs) -> MCResult:
    """Monte Carlo price of a European option, with its standard error.

    ``control=True`` uses the terminal spot as a control variate; its mean is
    the forward, known exactly, and the martingale correction makes that exact
    in the simulation too. ``n_replicates`` > 1 runs independent scrambles and
    reports a randomised quasi-Monte Carlo error, which is the meaningful error
    bar when ``method="sobol"``.
    """
    n_steps = kwargs.get("n_steps", 64)
    is_call = str(option).upper().startswith("C")

    def one(seed) -> np.ndarray:
        ST = simulate(S0, T, p, r, q, **{**kwargs, "seed": seed})
        payoff = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
        disc = np.exp(-r * T) * payoff
        return control_variate(disc, ST, S0 * np.exp((r - q) * T)) if control else disc

    if n_replicates > 1:
        base = kwargs.get("seed", 0) or 0
        means = [one(base + i).mean() for i in range(n_replicates)]
        return _rqmc_summarise(means, kwargs.get("n_paths", 100_000), n_steps)
    return _summarise(one(kwargs.get("seed")), n_steps, kwargs.get("antithetic", False))


def price_path_dependent(S0, T, p: HestonParams, payoff: Callable[[np.ndarray], np.ndarray],
                         r=0.0, q=0.0, controls: Optional[Callable] = None,
                         control_means=None, **kwargs) -> MCResult:
    """Price any payoff written as a function of the (n_steps + 1, n_paths) grid.

    ``controls`` maps the same grid to one or more control samples whose exact
    means are given in ``control_means``.
    """
    n_steps = kwargs.get("n_steps", 64)
    _, paths = simulate(S0, T, p, r, q, return_paths=True, **kwargs)
    disc = np.exp(-r * T) * np.asarray(payoff(paths), dtype=float)
    if controls is not None:
        disc = control_variate(disc, controls(paths), control_means)
    return _summarise(disc, n_steps, kwargs.get("antithetic", False))
