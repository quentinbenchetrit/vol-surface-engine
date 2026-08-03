"""Local volatility from the fitted SSVI surface, next to the implied vol.

    python scripts/plot_dupire.py --currency BTC
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import dupire, implied_vol_surface, ssvi
from volsurface.data import load_chain
from volsurface.data.forward import forward_curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    chain = load_chain(args.currency)
    surface = implied_vol_surface(chain, forward_curve(chain))
    fit = ssvi.calibrate(surface)
    p = fit.params
    as_of = chain["as_of"].iloc[0]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Left: implied vs local vol slices at a few maturities.
    # skip the very front of the curve, where a few days to expiry leaves almost
    # no strike range and the slice degenerates to a spike
    T_min = max(float(p.T_knots[0]), 30.0 / 365.0)
    colors = ["#2f81f7", "#3fb950", "#e3b341", "#f85149"]
    Ts = np.linspace(T_min, p.T_knots[-1], 4)
    for T, c in zip(Ts, colors):
        kmax = 2.5 * np.sqrt(float(ssvi.theta_at(T, p)))
        k = np.linspace(-kmax, kmax, 120)
        iv = np.sqrt(ssvi.total_variance(k, T, p) / T) * 100
        lv = dupire.local_vol(k, T, p) * 100
        axL.plot(k, iv, color=c, lw=1.6, ls="--", label=f"implied, T = {T * 365:.0f}d")
        axL.plot(k, lv, color=c, lw=2.0, label=f"local, T = {T * 365:.0f}d")
    axL.set_title("Local vol is steeper than implied, and meets it at the money", fontsize=10.5)
    axL.set_xlabel("log-moneyness  k = ln(K / F)")
    axL.set_ylabel("volatility (%)")
    axL.grid(alpha=0.2)
    axL.legend(fontsize=7.5, ncol=2)

    # Right: the local vol surface itself.
    kk = np.linspace(-0.6, 0.6, 200)
    tt = np.linspace(T_min, p.T_knots[-1], 200)
    K, T = np.meshgrid(kk, tt)
    LV = dupire.local_vol(K, T, p) * 100
    LV = np.where(np.abs(K) <= 3.0 * np.sqrt(ssvi.theta_at(T, p)), LV, np.nan)
    lo, hi = np.nanpercentile(LV, [2, 98])
    im = axR.pcolormesh(K, T * 365, LV, cmap="viridis", shading="auto", vmin=lo, vmax=hi)
    fig.colorbar(im, ax=axR, label="local vol (%)")
    axR.set_title("Dupire local volatility surface", fontsize=10.5)
    axR.set_xlabel("log-moneyness  k = ln(K / F)")
    axR.set_ylabel("maturity (days)")

    fig.suptitle(
        f"{args.currency} local volatility from the arbitrage-free SSVI surface    "
        f"sigma_loc^2 = w_T / g(k)    Deribit {as_of:%Y-%m-%d}",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "dupire_local_vol.png", dpi=130)
    print(f"saved {outdir/'dupire_local_vol.png'}")


if __name__ == "__main__":
    main()
