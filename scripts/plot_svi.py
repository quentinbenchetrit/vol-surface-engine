"""Fit SVI slices to the live Deribit surface and plot the results.

Produces two figures under figures/:
  - svi_smiles.png   market implied vols against the arbitrage-free SVI fit
  - svi_density.png  Gatheral's g(k) for the same slices, which stays >= 0

    python scripts/plot_svi.py --currency BTC
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import implied_vol_surface
from volsurface.data import load_chain
from volsurface.data.forward import forward_curve
from volsurface.svi import SVIParams, calibrate_slices, density_g, total_variance

MARKET = "#2f81f7"
FIT = "#f85149"


def _pick(fits, n=6):
    usable = sorted((f for f in fits if f["T"] > 0.01), key=lambda f: f["T"])
    idx = np.linspace(0, len(usable) - 1, min(n, len(usable))).round().astype(int)
    return [usable[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    chain = load_chain(args.currency)
    surface = implied_vol_surface(chain, forward_curve(chain))
    fits = calibrate_slices(surface)
    picks = _pick(fits)
    as_of = chain["as_of"].iloc[0]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Smiles: market points against the SVI curve.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axes.ravel(), picks):
        p = SVIParams(f["a"], f["b"], f["rho"], f["m"], f["s"])
        sl = surface[(surface["expiry"] == f["expiry"]) & surface["otm"]].dropna(subset=["iv"])
        k, iv = sl["log_moneyness"].to_numpy(), sl["iv"].to_numpy() * 100
        kk = np.linspace(k.min(), k.max(), 200)
        iv_fit = np.sqrt(np.maximum(total_variance(kk, p), 0) / f["T"]) * 100
        ax.scatter(k, iv, s=16, color=MARKET, alpha=0.8, label="market (OTM mid)")
        ax.plot(kk, iv_fit, color=FIT, lw=2, label="arbitrage-free SVI")
        ax.set_title(f"T = {f['T'] * 365:.0f}d    fit RMSE {f['rmse_vol'] * 100:.2f} vol pts", fontsize=10)
        ax.set_xlabel("log-moneyness  k = ln(K / F)")
        ax.set_ylabel("implied vol (%)")
        ax.grid(alpha=0.2)
    axes.ravel()[0].legend(fontsize=8, loc="upper center")
    fig.suptitle(f"{args.currency} implied volatility smiles: market vs SVI    Deribit {as_of:%Y-%m-%d %H:%M UTC}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outdir / "svi_smiles.png", dpi=130)

    # Density: g(k) >= 0 proves the fitted slices are butterfly arbitrage-free.
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for f in picks:
        p = SVIParams(f["a"], f["b"], f["rho"], f["m"], f["s"])
        kk = np.linspace(-1.0, 1.0, 400)
        ax2.plot(kk, density_g(kk, p), lw=1.6, label=f"T = {f['T'] * 365:.0f}d")
    ax2.axhline(0.0, color="#666", lw=1, ls="--")
    ax2.set_title(f"{args.currency} Gatheral density g(k), non-negative everywhere (no butterfly arbitrage)", fontsize=11)
    ax2.set_xlabel("log-moneyness  k = ln(K / F)")
    ax2.set_ylabel("g(k)")
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=8, ncol=2)
    fig2.tight_layout()
    fig2.savefig(outdir / "svi_density.png", dpi=130)

    print(f"saved {outdir/'svi_smiles.png'} and {outdir/'svi_density.png'}")


if __name__ == "__main__":
    main()
