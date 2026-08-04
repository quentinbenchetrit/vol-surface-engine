"""Validate the Heston Monte Carlo against COS, and show what QE buys over Euler.

    python scripts/plot_mc.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import mc
from volsurface.heston import HestonParams, feller_ok, price_cos

# Feller-violating parameters: the regime where a naive scheme hurts most.
P = HestonParams(kappa=1.0, theta=0.04, xi=0.9, rho=-0.7, v0=0.04)
S0, R, Q, T = 100.0, 0.0, 0.0, 1.0
QE, EULER, REF = "#2f81f7", "#f85149", "#3fb950"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--paths", type=int, default=200_000)
    args = ap.parse_args()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Left: discretisation bias against the number of time steps.
    ref = float(price_cos(S0, 100.0, R, Q, T, P, "C"))
    steps = [2, 4, 8, 16, 32, 64, 128]
    qe_bias, eu_bias, qe_err = [], [], []
    for ns in steps:
        a = mc.price_european(S0, 100.0, T, P, R, Q, "C", n_paths=args.paths, n_steps=ns, scheme="qe", seed=3)
        b = mc.price_european(S0, 100.0, T, P, R, Q, "C", n_paths=args.paths, n_steps=ns, scheme="euler", seed=3)
        qe_bias.append(abs(a.price - ref))
        eu_bias.append(abs(b.price - ref))
        qe_err.append(1.96 * a.stderr)

    axL.loglog(steps, eu_bias, "o-", color=EULER, lw=2, label="Euler, full truncation")
    axL.loglog(steps, qe_bias, "o-", color=QE, lw=2, label="Andersen QE")
    axL.loglog(steps, qe_err, "--", color="#7d8590", lw=1.3, label="Monte Carlo noise (95%)")
    axL.set_title("Discretisation bias against time steps", fontsize=10.5)
    axL.set_xlabel("time steps")
    axL.set_ylabel("absolute price error vs COS")
    axL.grid(alpha=0.2, which="both")
    axL.legend(fontsize=8.5)

    # Right: the simulation reprices the whole vanilla strip.
    strikes = np.linspace(70, 135, 14)
    cos = np.array([float(price_cos(S0, k, R, Q, T, P, "C")) for k in strikes])
    res = [mc.price_european(S0, k, T, P, R, Q, "C", n_paths=args.paths, n_steps=32, scheme="qe", seed=9)
           for k in strikes]
    px = np.array([r.price for r in res])
    er = np.array([1.96 * r.stderr for r in res])

    axR.plot(strikes, cos, color=REF, lw=2, label="COS reference")
    axR.errorbar(strikes, px, yerr=er, fmt="o", color=QE, ms=4, capsize=3, label="QE Monte Carlo (95% CI)")
    axR.set_title(f"Monte Carlo reprices the vanilla strip ({args.paths:,} paths)", fontsize=10.5)
    axR.set_xlabel("strike K")
    axR.set_ylabel("call price")
    axR.grid(alpha=0.2)
    axR.legend(fontsize=8.5, loc="lower left")
    inset = axR.inset_axes([0.5, 0.58, 0.46, 0.34])
    inset.axhline(0, color="#7d8590", lw=1)
    inset.errorbar(strikes, px - cos, yerr=er, fmt="o", color=QE, ms=3, capsize=2, lw=1)
    inset.set_title("MC minus COS", fontsize=8)
    inset.tick_params(labelsize=6.5)
    inset.grid(alpha=0.2)

    fig.suptitle(
        f"Heston Monte Carlo, Andersen QE    kappa={P.kappa}, theta={P.theta}, xi={P.xi}, "
        f"rho={P.rho}, v0={P.v0}    Feller {'holds' if feller_ok(P) else 'violated'}",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "mc_qe.png", dpi=130)
    print(f"saved {outdir/'mc_qe.png'}   QE bias at 8 steps {qe_bias[2]:.4f} vs Euler {eu_bias[2]:.4f}")


if __name__ == "__main__":
    main()
