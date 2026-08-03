import numpy as np

from volsurface import dupire, ssvi

SURF = ssvi.SSVIParams(
    rho=-0.4, eta=1.0, gamma=0.3,
    T_knots=np.array([0.1, 0.3, 0.6, 1.0, 1.5]),
    theta_knots=np.array([0.04, 0.11, 0.20, 0.32, 0.47]),
)


def test_dw_dtheta_matches_finite_difference():
    k = np.linspace(-0.5, 0.5, 11)
    theta, h = 0.2, 1e-6
    fd = (ssvi.total_variance(k, 0.0, SURF._replace(  # evaluate w at theta directly
        T_knots=np.array([0.0, 1.0]), theta_knots=np.array([theta + h, theta + h]))) -
          ssvi.total_variance(k, 0.0, SURF._replace(
              T_knots=np.array([0.0, 1.0]), theta_knots=np.array([theta - h, theta - h])))) / (2 * h)
    assert np.allclose(dupire.dw_dtheta(k, theta, SURF), fd, rtol=1e-4, atol=1e-6)


def test_dw_dT_matches_finite_difference():
    k = np.linspace(-0.4, 0.4, 9)
    T, h = 0.7, 1e-5
    fd = (ssvi.total_variance(k, T + h, SURF) - ssvi.total_variance(k, T - h, SURF)) / (2 * h)
    assert np.allclose(dupire.dw_dT(k, T, SURF), fd, rtol=1e-3, atol=1e-5)


def test_local_variance_matches_finite_difference_dupire():
    """Cross-check the analytic formula against a brute-force Dupire."""
    k = np.linspace(-0.4, 0.4, 9)
    T, hk, hT = 0.6, 1e-4, 1e-5
    w = ssvi.total_variance(k, T, SURF)
    wk = (ssvi.total_variance(k + hk, T, SURF) - ssvi.total_variance(k - hk, T, SURF)) / (2 * hk)
    wkk = (ssvi.total_variance(k + hk, T, SURF) - 2 * w + ssvi.total_variance(k - hk, T, SURF)) / hk ** 2
    wT = (ssvi.total_variance(k, T + hT, SURF) - ssvi.total_variance(k, T - hT, SURF)) / (2 * hT)
    denom = (1 - k * wk / (2 * w)) ** 2 - (wk ** 2 / 4) * (1 / w + 0.25) + wkk / 2
    assert np.allclose(dupire.local_variance(k, T, SURF), wT / denom, rtol=1e-3, atol=1e-5)


def test_flat_surface_gives_flat_local_vol():
    # eta -> 0 kills the skew, so w = theta(T); with theta = sigma^2 T the local
    # vol must come back as the constant sigma.
    sigma = 0.35
    Ts = np.array([0.1, 0.5, 1.0, 2.0])
    flat = ssvi.SSVIParams(rho=0.0, eta=1e-9, gamma=0.3, T_knots=Ts, theta_knots=sigma ** 2 * Ts)
    for T in (0.3, 0.8, 1.5):
        lv = dupire.local_vol(np.linspace(-0.3, 0.3, 7), T, flat)
        assert np.allclose(lv, sigma, atol=1e-4)


def test_local_variance_positive_on_arbitrage_free_surface():
    assert ssvi.butterfly_ok(SURF) and ssvi.calendar_ok(SURF)
    for T in (0.15, 0.5, 1.0, 1.4):
        lv = dupire.local_variance(np.linspace(-0.6, 0.6, 25), T, SURF)
        assert np.all(np.isfinite(lv)) and np.all(lv > 0)


def test_local_vol_skew_steeper_than_implied():
    """Near the money local vol skew is about twice the implied vol skew."""
    T, h = 0.5, 1e-3
    lv_slope = (dupire.local_vol(h, T, SURF) - dupire.local_vol(-h, T, SURF)) / (2 * h)
    iv = lambda k: np.sqrt(ssvi.total_variance(k, T, SURF) / T)
    iv_slope = (iv(h) - iv(-h)) / (2 * h)
    ratio = float(lv_slope / iv_slope)
    assert 1.5 < ratio < 2.5
