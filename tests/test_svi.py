import numpy as np

from volsurface.svi import (
    SVIParams,
    total_variance,
    density_g,
    fit_slice,
    _w_derivs,
)

CALM = SVIParams(a=0.04, b=0.1, rho=-0.2, m=0.0, s=0.4)
SHARP = SVIParams(a=0.01, b=1.5, rho=-0.9, m=0.0, s=0.02)


def test_total_variance_at_the_money():
    # at k = m, w = a + b*s
    w = total_variance(CALM.m, CALM)
    assert np.isclose(w, CALM.a + CALM.b * CALM.s)


def test_derivatives_match_finite_difference():
    k = np.linspace(-0.5, 0.5, 11)
    _, wp, wpp = _w_derivs(k, CALM)
    h = 1e-5
    wp_fd = (total_variance(k + h, CALM) - total_variance(k - h, CALM)) / (2 * h)
    wpp_fd = (total_variance(k + h, CALM) - 2 * total_variance(k, CALM) + total_variance(k - h, CALM)) / h ** 2
    assert np.allclose(wp, wp_fd, atol=1e-6)
    assert np.allclose(wpp, wpp_fd, atol=1e-4)


def test_calm_slice_is_arbitrage_free():
    grid = np.linspace(-1.0, 1.0, 200)
    assert np.min(density_g(grid, CALM)) > 0.0


def test_sharp_slice_has_arbitrage():
    grid = np.linspace(-0.5, 0.5, 400)
    assert np.min(density_g(grid, SHARP)) < 0.0


def test_fit_recovers_a_known_smile():
    k = np.linspace(-0.6, 0.6, 21)
    w = total_variance(k, CALM)
    fit = fit_slice(k, w, T=0.5)
    assert fit.arbitrage_free
    assert np.allclose(total_variance(k, fit.params), w, atol=1e-4)
    assert fit.rmse_vol < 1e-2


def test_fit_stays_arbitrage_free_on_a_steep_smile():
    # a steep, noisy skew that a raw fit could push into arbitrage
    rng = np.random.default_rng(0)
    k = np.linspace(-0.7, 0.7, 25)
    w = 0.05 + 0.09 * (-0.7 * k + np.sqrt(k * k + 0.01)) + rng.normal(0, 1e-4, k.size)
    fit = fit_slice(k, np.maximum(w, 1e-4), T=0.25, enforce_butterfly=True)
    assert fit.min_g >= -1e-8
    assert fit.arbitrage_free
