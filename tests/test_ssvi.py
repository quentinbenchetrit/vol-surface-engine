import math

import numpy as np
import pandas as pd

from volsurface import ssvi
from volsurface.svi import density_g, total_variance as svi_tv

TRUE = ssvi.SSVIParams(
    rho=-0.4, eta=1.2, gamma=0.35,
    T_knots=np.array([0.1, 0.3, 0.6, 1.0]),
    theta_knots=np.array([0.04, 0.10, 0.18, 0.28]),
)


def test_atm_total_variance_equals_theta():
    for T, th in zip(TRUE.T_knots, TRUE.theta_knots):
        assert np.isclose(float(ssvi.total_variance(0.0, T, TRUE)), th)


def test_to_svi_reproduces_the_slice():
    theta = 0.18
    p = ssvi.to_svi(theta, TRUE)
    k = np.linspace(-0.6, 0.6, 25)
    # the raw-SVI image must match the SSVI slice at that theta
    w_ssvi = 0.5 * theta * (1 + TRUE.rho * (TRUE.eta * theta ** -TRUE.gamma) * k
                            + np.sqrt((TRUE.eta * theta ** -TRUE.gamma * k + TRUE.rho) ** 2 + (1 - TRUE.rho ** 2)))
    assert np.allclose(svi_tv(k, p), w_ssvi, atol=1e-10)


def test_flags():
    assert ssvi.butterfly_ok(TRUE)            # 1.2 * 1.4 = 1.68 <= 2
    assert ssvi.calendar_ok(TRUE)
    bad = TRUE._replace(theta_knots=np.array([0.04, 0.10, 0.08, 0.28]))
    assert not ssvi.calendar_ok(bad)
    steep = TRUE._replace(eta=2.0)            # 2.0 * 1.4 = 2.8 > 2
    assert not ssvi.butterfly_ok(steep)


def test_density_nonneg_when_butterfly_holds():
    grid = np.linspace(-1.0, 1.0, 300)
    for theta in TRUE.theta_knots:
        assert np.min(density_g(grid, ssvi.to_svi(theta, TRUE))) > 0.0


def _synthetic_surface(params, ks, Ts):
    rows = []
    for T in Ts:
        for k in ks:
            w = float(ssvi.total_variance(k, T, params))
            rows.append({
                "expiry": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(round(T * 365))),
                "T": float(T), "log_moneyness": float(k), "iv": math.sqrt(max(w, 0) / T),
                "total_var": w, "otm": True, "rel_spread": 0.05,
            })
    return pd.DataFrame(rows)


def test_calibrate_recovers_synthetic_surface():
    surface = _synthetic_surface(TRUE, np.linspace(-0.5, 0.5, 15), TRUE.T_knots)
    fit = ssvi.calibrate(surface)
    assert fit.butterfly_ok and fit.calendar_ok
    assert fit.rmse_vol < 5e-3
    assert abs(fit.params.rho - TRUE.rho) < 0.1
    assert abs(fit.params.eta - TRUE.eta) < 0.2
    assert abs(fit.params.gamma - TRUE.gamma) < 0.12
