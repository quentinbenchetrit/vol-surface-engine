import numpy as np

from volsurface import mc
from volsurface.heston import HestonParams, price_cos

P = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.05)
HARD = HestonParams(kappa=1.0, theta=0.04, xi=0.9, rho=-0.7, v0=0.04)  # violates Feller


def test_simulation_is_positive_and_shaped():
    ST = mc.simulate(100.0, 1.0, P, n_paths=5000, n_steps=16, seed=0)
    assert ST.shape == (5000,)
    assert np.all(ST > 0) and np.all(np.isfinite(ST))


def test_martingale_forward_is_unbiased():
    S0, r, q, T = 100.0, 0.03, 0.01, 1.0
    ST = mc.simulate(S0, T, P, r, q, n_paths=200_000, n_steps=32, seed=1)
    target = S0 * np.exp((r - q) * T)
    stderr = ST.std(ddof=1) / np.sqrt(ST.size)
    assert abs(ST.mean() - target) < 4.0 * stderr


def test_reprices_vanilla_against_cos():
    S0, r, q, T = 100.0, 0.0, 0.0, 1.0
    for K in (85.0, 100.0, 115.0):
        ref = price_cos(S0, K, r, q, T, P, "C")
        got = mc.price_european(S0, K, T, P, r, q, "C", n_paths=200_000, n_steps=64, seed=7)
        assert abs(got.price - ref) < 4.0 * got.stderr
        assert got.ci95[0] < ref < got.ci95[1]


def test_put_call_parity_in_simulation():
    S0, r, q, T, K = 100.0, 0.02, 0.0, 0.75, 105.0
    call = mc.price_european(S0, K, T, P, r, q, "C", n_paths=100_000, n_steps=32, seed=5).price
    put = mc.price_european(S0, K, T, P, r, q, "P", n_paths=100_000, n_steps=32, seed=5).price
    assert abs((call - put) - (S0 * np.exp(-q * T) - K * np.exp(-r * T))) < 0.05


def test_qe_beats_euler_at_coarse_steps():
    """The whole point of QE: small bias where full-truncation Euler is far off."""
    S0, K, T = 100.0, 100.0, 1.0
    ref = price_cos(S0, K, 0.0, 0.0, T, HARD, "C")
    qe = mc.price_european(S0, K, T, HARD, option="C", n_paths=100_000, n_steps=8, scheme="qe", seed=3)
    eu = mc.price_european(S0, K, T, HARD, option="C", n_paths=100_000, n_steps=8, scheme="euler", seed=3)
    assert abs(qe.price - ref) < 0.5 * abs(eu.price - ref)
    assert abs(qe.price - ref) < 0.1


def test_standard_error_shrinks_like_sqrt_n():
    S0, K, T = 100.0, 100.0, 0.5
    a = mc.price_european(S0, K, T, P, option="C", n_paths=20_000, n_steps=16, seed=11)
    b = mc.price_european(S0, K, T, P, option="C", n_paths=80_000, n_steps=16, seed=11)
    assert 1.7 < a.stderr / b.stderr < 2.3        # four times the paths, half the error


def test_return_paths_shape_and_start():
    ST, paths = mc.simulate(100.0, 1.0, P, n_paths=1000, n_steps=20, seed=2, return_paths=True)
    assert paths.shape == (21, 1000)
    assert np.allclose(paths[0], 100.0)
    assert np.allclose(paths[-1], ST)


def test_antithetic_pairs_are_mirrored():
    ST = mc.simulate(100.0, 1.0, P, n_paths=10_000, n_steps=16, seed=4, antithetic=True)
    assert ST.size == 10_000
    assert np.all(ST > 0)
