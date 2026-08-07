"""The closing check: does a delta hedge actually replicate what we priced?

    python scripts/plot_hedging.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import hedging
from volsurface.heston import HestonParams

S0, K, T, R = 100.0, 100.0, 1.0, 0.0
SIGMA = 0.20
HES = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.05)
SELL, REALIZED = 0.25, 0.20
CONST, MODEL, REAL_H, IMPL_H = "#2f81f7", "#f85149", "#3fb950", "#e3b341"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--paths", type=int, default=60_000)
    args = ap.parse_args()

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 5.2))
    kw = dict(n_paths=args.paths, seed=1)

    # A: the hedging error tightens on zero as rebalancing gets finer.
    shades = ["#c6d9f5", "#7fb0ec", "#3d86e0", "#12508f"]
    for ns, c in zip((8, 32, 128, 512), shades):
        res = hedging.hedge_under_gbm(S0, K, T, SIGMA, SIGMA, SIGMA, R, n_steps=ns, **kw)
        axA.hist(res.pnl, bins=140, range=(-6, 6), histtype="step", lw=1.8, color=c, density=True,
                 label=f"{ns} rebalances, std {res.std:.2f}")
    axA.axvline(0.0, color="#444", lw=1, ls="--")
    axA.set_title("Hedging error, constant volatility", fontsize=10.5)
    axA.set_xlabel("profit and loss")
    axA.set_ylabel("density")
    axA.grid(alpha=0.2)
    axA.legend(fontsize=8)

    # B: discretisation error vanishes, model error does not.
    steps = [8, 16, 32, 64, 128, 256, 512, 1024]
    const_std, hes_std = [], []
    for ns in steps:
        const_std.append(hedging.hedge_under_gbm(S0, K, T, SIGMA, SIGMA, SIGMA, R, n_steps=ns, **kw).std)
        hes_std.append(hedging.hedge_under_heston(S0, K, T, HES, R, n_steps=ns, **kw).std)
    ref = const_std[0] * np.sqrt(steps[0] / np.array(steps, dtype=float))
    axB.loglog(steps, const_std, "o-", color=CONST, lw=2, ms=4, label="constant volatility")
    axB.loglog(steps, hes_std, "o-", color=MODEL, lw=2, ms=4, label="Heston, hedged with a BS delta")
    axB.loglog(steps, ref, "--", color="#444", lw=1.2, label="1 / sqrt(rebalances)")
    axB.text(0.42, 0.72, "model error: a spot hedge\ncannot touch variance risk",
             transform=axB.transAxes, fontsize=8, color=MODEL, ha="left", va="top")
    axB.set_title("Rebalancing removes discretisation error, not model error", fontsize=10.5)
    axB.set_xlabel("rebalances")
    axB.set_ylabel("standard deviation of the hedging error")
    axB.grid(alpha=0.2, which="both")
    axB.legend(fontsize=8, loc="lower left")

    # C: selling volatility the market does not deliver.
    edge = None
    for label, hv, color in [("hedged at realized vol", REALIZED, REAL_H),
                             ("hedged at implied vol", SELL, IMPL_H)]:
        res = hedging.hedge_under_gbm(S0, K, T, SELL, hv, REALIZED, R, n_steps=512, **kw)
        edge = res.mean if edge is None else edge
        axC.hist(res.pnl, bins=120, histtype="stepfilled", alpha=0.45, color=color, density=True,
                 label=f"{label}, std {res.std:.2f}")
    axC.axvline(0.0, color="#444", lw=1, ls="--")
    axC.axvline(edge, color="#111", lw=1.4,
                label=f"expected edge {edge:.2f}")
    axC.set_title(f"Selling {SELL:.0%} vol into a {REALIZED:.0%} market", fontsize=10.5)
    axC.set_xlabel("profit and loss")
    axC.set_ylabel("density")
    axC.grid(alpha=0.2)
    axC.legend(fontsize=8)

    fig.suptitle("Delta-hedging backtest: the pricer and the greeks checked against each other",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "hedging.png", dpi=130)
    print(f"saved {outdir/'hedging.png'}")
    print(f"  constant vol std: {const_std[0]:.3f} at {steps[0]} -> {const_std[-1]:.3f} at {steps[-1]}")
    print(f"  heston std:       {hes_std[0]:.3f} at {steps[0]} -> {hes_std[-1]:.3f} at {steps[-1]}")


if __name__ == "__main__":
    main()
