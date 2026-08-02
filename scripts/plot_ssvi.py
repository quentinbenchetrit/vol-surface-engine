"""Calibrate the SSVI surface to live Deribit data and plot it.

Produces two figures under figures/:
  - ssvi_surface.png    the fitted implied vol surface with market points
  - ssvi_calendar.png    total variance rising with maturity at fixed moneyness

    python scripts/plot_ssvi.py --currency BTC
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from volsurface import implied_vol_surface, ssvi
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
    # Keep a moderate moneyness band: short-dated options do not trade in the
    # far wings, so drawing the surface out there is just extrapolation noise.
    kmax = 0.7
    pts = surface[surface["otm"] & (surface["log_moneyness"].abs() <= kmax)].dropna(subset=["iv"])
    as_of = chain["as_of"].iloc[0]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 3D surface with the market points sitting on it.
    kg = np.linspace(-kmax, kmax, 60)
    tg = np.linspace(p.T_knots.min(), p.T_knots.max(), 60)
    K, T = np.meshgrid(kg, tg)
    Z = ssvi.implied_vol(K, T, p) * 100.0
    # only draw where options actually trade: within about 3.5 standard
    # deviations of the forward, which narrows the surface at short maturities.
    Z = np.where(np.abs(K) <= 3.5 * np.sqrt(ssvi.theta_at(T, p)), Z, np.nan)

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, T * 365, Z, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.scatter(pts["log_moneyness"], pts["T"] * 365, pts["iv"] * 100,
               s=6, color="#f85149", depthshade=False, label="market (OTM mid)")
    ax.set_xlabel("log-moneyness  k = ln(K / F)")
    ax.set_ylabel("maturity (days)")
    ax.set_zlabel("implied vol (%)")
    ax.set_zlim(10, 90)
    ax.set_title(
        f"{args.currency} SSVI surface    rho={p.rho:.2f}, eta={p.eta:.2f}, gamma={p.gamma:.2f}    "
        f"arbitrage-free    Deribit {as_of:%Y-%m-%d}",
        fontsize=11,
    )
    ax.view_init(elev=22, azim=-60)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "ssvi_surface.png", dpi=130)

    # Calendar no-arbitrage: total variance is non-decreasing in maturity.
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    tt = np.linspace(p.T_knots.min(), p.T_knots.max(), 200)
    for k in (-0.3, -0.15, 0.0, 0.15, 0.3):
        ax2.plot(tt * 365, ssvi.total_variance(k, tt, p), lw=1.7, label=f"k = {k:+.2f}")
    ax2.plot(p.T_knots * 365, p.theta_knots, "o", color="#333", ms=4, label="ATM theta knots")
    ax2.set_title(f"{args.currency} total variance rises with maturity (no calendar arbitrage)", fontsize=11)
    ax2.set_xlabel("maturity (days)")
    ax2.set_ylabel("total variance  w = sigma^2 T")
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=8, ncol=3)
    fig2.tight_layout()
    fig2.savefig(outdir / "ssvi_calendar.png", dpi=130)

    print(f"saved {outdir/'ssvi_surface.png'} and {outdir/'ssvi_calendar.png'}  "
          f"(surface RMSE {fit.rmse_vol*100:.2f} vol pts, arb-free={fit.butterfly_ok and fit.calendar_ok})")


if __name__ == "__main__":
    main()
