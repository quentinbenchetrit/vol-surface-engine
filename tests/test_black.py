import numpy as np

from volsurface import black76_price, black76_vega


def test_put_call_parity():
    F, K, T, sigma, DF = 65000.0, 60000.0, 0.5, 0.6, 0.98
    call = black76_price(F, K, T, sigma, DF, "C")
    put = black76_price(F, K, T, sigma, DF, "P")
    assert np.isclose(call - put, DF * (F - K), rtol=1e-12, atol=1e-8)


def test_price_increasing_in_sigma():
    F, K, T = 100.0, 110.0, 1.0
    sig = np.array([0.1, 0.2, 0.4, 0.8])
    prices = black76_price(F, K, T, sig, 1.0, "C")
    assert np.all(np.diff(prices) > 0)


def test_vega_positive_and_peaks_atm():
    F, T, sigma = 100.0, 1.0, 0.3
    strikes = np.array([70.0, 100.0, 140.0])
    vega = black76_vega(F, strikes, T, sigma)
    assert np.all(vega > 0)
    assert vega[1] == vega.max()  # vega is largest at the money


def test_vectorized_shapes():
    F = 100.0
    K = np.array([80.0, 100.0, 120.0])
    prices = black76_price(F, K, 1.0, 0.25, 1.0, np.array(["C", "C", "P"]))
    assert prices.shape == (3,)
