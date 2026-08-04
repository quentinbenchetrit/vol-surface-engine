"""Exotics: the barrier monitoring bias, and where two calibrated models disagree.

The second panel is a controlled experiment. A Heston model is fixed, its own
implied vol surface is generated, SSVI is fitted to that surface and Dupire
local vol is read off it. By construction the two models agree on vanillas, so
any difference on an exotic is a pure statement about their dynamics.

    python scripts/plot_exotics.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volsurface import exotics, implied_vol, mc, ssvi
from volsurface.heston import HestonParams, price_cos

P = HestonParams(kappa=2.0, theta=0.045, xi=0.7, rho=-0.7, v0=0.045)
S0, T = 100.0, 1.0
HES, LOC = "#2f81f7", "#f85149"


def heston_surface():
    """Generate Heston's own implied vol surface, then fit SSVI to it."""
    rows = []
    for t in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5):
        kmax = 3.0 * np.sqrt(P.theta * t)
        for k in np.linspace(-kmax, kmax, 15):
            K = S0 * np.exp(k)
            iv = implied_vol(float(price_cos(S0, K, 0.0, 0.0, t, P, "C")), S0, K, t, 1.0, "C")
            if np.isfinite(iv):
                rows.append(dict(expiry=pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(t * 365)),
                                 T=t, log_moneyness=k, iv=iv, total_var=iv ** 2 * t,
                                 otm=True, rel_spread=0.02))
    return ssvi.calibrate(pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--paths", type=int, default=120_000)
    args = ap.parse_args()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Left: the discrete monitoring bias on a knock-out, and the two corrections.
    steps = [8, 16, 32, 64, 128, 256]
    raw, bridge, shift = [], [], []
    for ns in steps:
        _, paths, var = mc.simulate(S0, T, P, n_paths=args.paths, n_steps=ns, seed=11,
                                    return_variance=True)
        raw.append(exotics.barrier_payoff(paths, 100.0, 85.0, T, "C", "do", correction="none").mean())
        bridge.append(exotics.barrier_payoff(paths, 100.0, 85.0, T, "C", "do", variance=var).mean())
        shift.append(exotics.barrier_payoff(paths, 100.0, 85.0, T, "C", "do",
                                            correction="shift", vol_hint=0.22).mean())
    axL.semilogx(steps, raw, "o-", color="#7d8590", lw=2, label="discrete monitoring, uncorrected")
    axL.semilogx(steps, shift, "o-", color="#e3b341", lw=2, label="Broadie-Glasserman-Kou shift")
    axL.semilogx(steps, bridge, "o-", color=HES, lw=2, label="Brownian bridge")
    axL.set_title("Down-and-out call: the monitoring bias and its corrections", fontsize=10.5)
    axL.set_xlabel("monitoring steps")
    axL.set_ylabel("price")
    axL.grid(alpha=0.2, which="both")
    axL.legend(fontsize=8.5)

    # Right: same vanillas, different exotics.
    fit = heston_surface()
    n_steps = 48
    _, ph = mc.simulate(S0, T, P, n_paths=args.paths, n_steps=n_steps, seed=5, return_paths=True)
    _, pl = mc.simulate_local_vol(S0, T, fit.params, n_paths=args.paths, n_steps=n_steps,
                                  seed=5, return_paths=True)
    obs = [n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
    resets = [0] + obs

    products = [
        ("vanilla call", np.maximum(ph[-1] - 100.0, 0.0), np.maximum(pl[-1] - 100.0, 0.0)),
        ("down-and-out\ncall, B = 80",
         exotics.barrier_payoff(ph, 100.0, 80.0, T, "C", "do", correction="none"),
         exotics.barrier_payoff(pl, 100.0, 80.0, T, "C", "do", correction="none")),
        ("autocall\n7% coupon",
         exotics.autocall_payoff(ph, obs, 0.07, 1.0, 0.70),
         exotics.autocall_payoff(pl, obs, 0.07, 1.0, 0.70)),
        ("cliquet\ncap 8%",
         exotics.cliquet_payoff(ph, resets), exotics.cliquet_payoff(pl, resets)),
    ]

    names = [p[0] for p in products]
    diffs = [(p[1].mean() - p[2].mean()) / p[1].mean() * 100.0 for p in products]
    errs = [100.0 * np.hypot(p[1].std(ddof=1), p[2].std(ddof=1)) / np.sqrt(args.paths) / p[1].mean()
            for p in products]
    colors = ["#7d8590" if abs(d) < 2 else LOC for d in diffs]
    bars = axR.barh(names, diffs, xerr=errs, color=colors, alpha=0.9, capsize=4)
    for bar, d in zip(bars, diffs):
        axR.text(d + (0.35 if d >= 0 else -0.35), bar.get_y() + bar.get_height() / 2,
                 f"{d:+.1f}%", va="center", ha="left" if d >= 0 else "right", fontsize=9.5)
    axR.axvline(0, color="#444", lw=1)
    axR.set_title("Heston minus local vol, both fitted to the same vanillas", fontsize=10.5)
    axR.set_xlabel("price difference (%)")
    axR.set_xlim(min(diffs) - 3, max(diffs) + 3)
    axR.grid(alpha=0.2, axis="x")

    fig.suptitle(
        "Exotics under Heston and Dupire local volatility    "
        f"SSVI fits the Heston surface to {fit.rmse_vol * 100:.2f} vol pts",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "exotics.png", dpi=130)
    print("saved " + str(outdir / "exotics.png"))
    for n, d in zip(names, diffs):
        print(f"  {n.replace(chr(10), ' '):<28}{d:+7.2f}%")


if __name__ == "__main__":
    main()
