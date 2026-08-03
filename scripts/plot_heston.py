"""Show what the Heston pricer produces: a smile, and COS vs Carr-Madan agreement.

    python scripts/plot_heston.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import implied_vol
from volsurface.heston import HestonParams, feller_ok, price_carr_madan_at, price_cos

P = HestonParams(kappa=3.0, theta=0.04, xi=0.4, rho=-0.7, v0=0.04)
S0, R, Q = 100.0, 0.0, 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: the implied vol smile Heston produces, priced by COS then inverted.
    # The moneyness range scales with sqrt(T) so short maturities are not drawn
    # out into the deep wings where the price is numerically zero.
    for T in (0.1, 0.25, 0.5, 1.0):
        kmax = 4.0 * np.sqrt(P.theta * T)
        k = np.linspace(-kmax, kmax, 41)
        K = S0 * np.exp(k)
        px = price_cos(S0, K, R, Q, T, P, "C")
        iv = np.array([implied_vol(p, S0, kk, T, 1.0, "C") for p, kk in zip(px, K)]) * 100
        iv[(iv <= 1.0) | ~np.isfinite(iv)] = np.nan  # drop failed deep-OTM inversions
        axL.plot(k, iv, lw=1.8, label=f"T = {T:.2f}")
    axL.set_title("Heston implied vol smile (priced by COS)", fontsize=11)
    axL.set_xlabel("log-moneyness  k = ln(K / F)")
    axL.set_ylabel("implied vol (%)")
    axL.grid(alpha=0.2)
    axL.legend(fontsize=9)

    # Right: COS and Carr-Madan on top of each other.
    T = 0.75
    Kg = np.linspace(80, 120, 25)
    cos = price_cos(S0, Kg, R, Q, T, P, "C")
    cm = np.array([price_carr_madan_at(S0, kk, R, Q, T, P) for kk in Kg])
    axR.plot(Kg, cos, color="#2f81f7", lw=2, label="COS")
    axR.plot(Kg, cm, "o", color="#f85149", ms=4, label="Carr-Madan FFT")
    axR.set_title(f"Two methods agree (max diff {np.max(np.abs(cos - cm)):.1e}), T = {T}", fontsize=11)
    axR.set_xlabel("strike K")
    axR.set_ylabel("call price")
    axR.grid(alpha=0.2)
    axR.legend(fontsize=9)

    fig.suptitle(
        f"Heston  kappa={P.kappa}, theta={P.theta}, xi={P.xi}, rho={P.rho}, v0={P.v0}"
        f"   Feller {'holds' if feller_ok(P) else 'violated'}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outdir / "heston_smiles.png", dpi=130)
    print(f"saved {outdir/'heston_smiles.png'}")


if __name__ == "__main__":
    main()
