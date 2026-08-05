"""Three routes to a Monte Carlo delta, their variance, and where one breaks.

    python scripts/plot_greeks.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import greeks, mc
from volsurface.heston import HestonParams, price_cos

P = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.05)
S0, K, T = 100.0, 100.0, 1.0
PW, LR, CRN, IND = "#3fb950", "#e3b341", "#2f81f7", "#7d8590"


def reference_delta(h=0.01):
    up = float(price_cos(S0 + h, K, 0.0, 0.0, T, P, "C"))
    dn = float(price_cos(S0 - h, K, 0.0, 0.0, T, P, "C"))
    return (up - dn) / (2.0 * h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--paths", type=int, default=200_000)
    args = ap.parse_args()

    ref = reference_delta()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Left: estimates with their confidence intervals, against the exact value.
    kw = dict(n_paths=args.paths, n_steps=32, seed=7)
    entries = [
        ("pathwise", greeks.delta_pathwise(S0, K, T, P, **kw), PW),
        ("finite difference\ncommon randoms", greeks.delta_finite_difference(S0, K, T, P, common_random=True, **kw), CRN),
        ("likelihood ratio", greeks.delta_likelihood_ratio(S0, K, T, P, **kw), LR),
        ("finite difference\nindependent runs", greeks.delta_finite_difference(S0, K, T, P, common_random=False, **kw), IND),
    ]
    names = [e[0] for e in entries]
    vals = [e[1].value for e in entries]
    errs = [1.96 * e[1].stderr for e in entries]
    cols = [e[2] for e in entries]

    for n, v, e, c in zip(names, vals, errs, cols):
        axL.errorbar([v], [n], xerr=[e], fmt="o", ms=7, capsize=5, lw=2, color=c)
    axL.axvline(ref, color="#f85149", lw=1.6, ls="--", label=f"COS reference {ref:.4f}")
    axL.set_title(f"European call delta, {args.paths:,} paths (95% intervals)", fontsize=10.5)
    axL.set_xlabel("delta")
    axL.grid(alpha=0.2, axis="x")
    axL.legend(fontsize=8.5, loc="lower right")

    # Right: how the error shrinks with sample size, per estimator.
    counts = [2_000, 8_000, 32_000, 128_000, 512_000]
    # pathwise and common-random finite differences land almost on top of each
    # other, so the second is dashed to keep both visible
    series = [
        ("finite difference, common randoms", greeks.delta_finite_difference, {"common_random": True}, CRN, "-", 2.0),
        ("pathwise", greeks.delta_pathwise, {}, PW, "--", 2.4),
        ("likelihood ratio", greeks.delta_likelihood_ratio, {}, LR, "-", 2.0),
        ("finite difference, independent", greeks.delta_finite_difference, {"common_random": False}, IND, "-", 2.0),
    ]
    for label, fn, extra, color, ls, lw in series:
        errs_n = [fn(S0, K, T, P, n_paths=n, n_steps=32, seed=7, **extra).stderr for n in counts]
        axR.loglog(counts, errs_n, "o", ls=ls, color=color, lw=lw, ms=4, label=label)
    axR.set_title("Standard error against paths", fontsize=10.5)
    axR.set_xlabel("paths")
    axR.set_ylabel("standard error of delta")
    axR.grid(alpha=0.2, which="both")
    axR.legend(fontsize=8)

    # The digital: pathwise is not merely noisy, it is wrong.
    dig = greeks.digital_payoff(K)
    lr_d = greeks.delta_likelihood_ratio(S0, K, T, P, payoff=dig, n_paths=args.paths, n_steps=32, seed=7)
    ref_d = greeks.delta_finite_difference(S0, K, T, P, payoff=dig, h=0.5,
                                           n_paths=500_000, n_steps=32, seed=1)
    axR.annotate(
        f"digital call delta\nlikelihood ratio {lr_d.value:.4f}, reference {ref_d.value:.4f}\n"
        f"pathwise gives exactly 0, the indicator has no usable derivative",
        xy=(0.02, 0.03), xycoords="axes fraction", fontsize=7.8, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#e3b341", lw=1),
    )

    fig.suptitle("Monte Carlo greeks: three estimators, three failure modes", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "greeks.png", dpi=130)
    print(f"saved {outdir/'greeks.png'}")
    for n, v, e in zip(names, vals, errs):
        print(f"  {n.replace(chr(10), ' '):<36}{v:.5f} +/- {e / 1.96:.5f}")
    print(f"  digital: LR {lr_d.value:.5f}, reference {ref_d.value:.5f}, pathwise 0.00000")


if __name__ == "__main__":
    main()
