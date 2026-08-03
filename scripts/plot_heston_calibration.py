"""Calibrate Heston to the live Deribit surface and plot the fit per maturity.

    python scripts/plot_heston_calibration.py --currency BTC
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import implied_vol, implied_vol_surface
from volsurface.data import load_chain
from volsurface.data.forward import forward_curve
from volsurface.heston import calibrate, price_cos

MARKET = "#2f81f7"
MODEL = "#f85149"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    chain = load_chain(args.currency)
    surface = implied_vol_surface(chain, forward_curve(chain))
    fit = calibrate(surface)
    p = fit.params
    as_of = chain["as_of"].iloc[0]

    sl = surface[surface["otm"]].dropna(subset=["iv"])
    by_T = sorted(sl["expiry"].unique(), key=lambda e: sl.loc[sl["expiry"] == e, "T"].iloc[0])
    by_T = [e for e in by_T if sl.loc[sl["expiry"] == e, "T"].iloc[0] > 0.01]
    picks = [by_T[i] for i in np.linspace(0, len(by_T) - 1, 6).round().astype(int)]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, e in zip(axes.ravel(), picks):
        g = sl[sl["expiry"] == e]
        F = float(g["forward"].iloc[0])
        T = float(g["T"].iloc[0])
        k = g["log_moneyness"].to_numpy()
        ax.scatter(k, g["iv"].to_numpy() * 100, s=16, color=MARKET, alpha=0.8, label="market (OTM mid)")
        kk = np.linspace(k.min(), k.max(), 80)
        Kk = F * np.exp(kk)
        calls = price_cos(F, Kk, 0.0, 0.0, T, p, "C")
        ivm = np.array([implied_vol(c, F, kv, T, 1.0, "C") for c, kv in zip(calls, Kk)]) * 100
        ax.plot(kk, ivm, color=MODEL, lw=2, label="Heston fit")
        ax.set_title(f"T = {T * 365:.0f}d", fontsize=10)
        ax.set_xlabel("log-moneyness  k = ln(K / F)")
        ax.set_ylabel("implied vol (%)")
        ax.grid(alpha=0.2)
    axes.ravel()[0].legend(fontsize=8)

    fig.suptitle(
        f"{args.currency} Heston fit    kappa={p.kappa:.2f}, theta={p.theta:.3f}, xi={p.xi:.2f}, "
        f"rho={p.rho:.2f}, v0={p.v0:.3f}    RMSE {fit.rmse_vol * 100:.2f} vol pts    "
        f"Feller {'holds' if fit.feller_ok else 'violated'}    Deribit {as_of:%Y-%m-%d}",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "heston_fit.png", dpi=130)
    print(f"saved {outdir/'heston_fit.png'}  (RMSE {fit.rmse_vol*100:.2f} vol pts, Feller_ok={fit.feller_ok})")


if __name__ == "__main__":
    main()
