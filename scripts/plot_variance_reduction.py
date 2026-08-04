"""What antithetics, control variates and Sobol buy over plain Monte Carlo.

    python scripts/plot_variance_reduction.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import mc
from volsurface.heston import HestonParams, price_cos

P = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.05)
S0, K, R, Q, T = 100.0, 100.0, 0.0, 0.0, 1.0
N_STEPS, REPS = 32, 16

VARIANTS = [
    ("plain", dict(), "#7d8590"),
    ("antithetic", dict(antithetic=True), "#2f81f7"),
    ("control variate", dict(control=True), "#3fb950"),
    ("Sobol QMC", dict(method="sobol"), "#e3b341"),
    ("all three", dict(method="sobol", antithetic=True, control=True), "#f85149"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    ref = float(price_cos(S0, K, R, Q, T, P, "C"))
    path_counts = [512, 1024, 2048, 4096, 8192, 16384]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Left: error against path count, log-log.
    results = {}
    for name, kw, color in VARIANTS:
        errs = []
        for n in path_counts:
            res = mc.price_european(S0, K, T, P, R, Q, "C", n_paths=n, n_steps=N_STEPS,
                                    seed=100, n_replicates=REPS, **kw)
            errs.append(res.stderr)
        results[name] = errs
        axL.loglog(path_counts, errs, "o-", color=color, lw=2, ms=4, label=name)

    ref_line = np.array(results["plain"][0]) * np.sqrt(path_counts[0] / np.array(path_counts, float))
    axL.loglog(path_counts, ref_line, "--", color="#444", lw=1.2, label="1 / sqrt(N) reference")
    axL.set_title("Standard error against number of paths", fontsize=10.5)
    axL.set_xlabel("paths")
    axL.set_ylabel("standard error")
    axL.grid(alpha=0.2, which="both")
    axL.legend(fontsize=8.5)

    # Right: variance reduction factor at the largest sample.
    base = results["plain"][-1]
    names = [n for n, _, _ in VARIANTS]
    factors = [(base / results[n][-1]) ** 2 for n in names]
    colors = [c for _, _, c in VARIANTS]
    bars = axR.barh(names, factors, color=colors, alpha=0.85)
    for bar, f in zip(bars, factors):
        axR.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                 f"{f:.1f}x", va="center", fontsize=9.5)
    axR.set_title(f"Variance reduction at {path_counts[-1]:,} paths "
                  f"(equivalently, paths saved)", fontsize=10.5)
    axR.set_xlabel("variance reduction factor")
    axR.set_xlim(0, max(factors) * 1.25)
    axR.grid(alpha=0.2, axis="x")

    fig.suptitle(
        f"Variance reduction on a Heston European call    COS reference {ref:.4f}    "
        f"errors from {REPS} independent replicates",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "variance_reduction.png", dpi=130)
    print(f"saved {outdir/'variance_reduction.png'}   best factor {max(factors):.1f}x")


if __name__ == "__main__":
    main()
