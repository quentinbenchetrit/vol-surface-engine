import numpy as np

from volsurface import mc
from volsurface.heston import HestonParams, price_cos

P = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.05)
S0, K, T = 100.0, 100.0, 1.0


def test_uniform_draws_are_shaped_and_bounded():
    U = mc.draw_uniforms(1000, 16, method="pseudo", seed=0)
    assert U.shape == (1000, 32)
    assert np.all(U > 0) and np.all(U < 1)


def test_antithetic_draws_are_mirrored():
    U = mc.draw_uniforms(1000, 8, method="pseudo", seed=0, antithetic=True)
    assert U.shape == (1000, 16)
    assert np.allclose(U[:500] + U[500:], 1.0)


def test_sobol_draws_are_shaped():
    U = mc.draw_uniforms(512, 8, method="sobol", seed=1)
    assert U.shape == (512, 16)
    assert np.all(U > 0) and np.all(U < 1)


def test_control_variate_is_unbiased_and_reduces_variance():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20_000)              # control, mean 0
    x = 2.0 * y + 0.5 * rng.standard_normal(20_000)   # strongly correlated payoff
    corrected = mc.control_variate(x, y, 0.0)
    assert abs(corrected.mean() - x.mean()) < 0.05        # same target
    assert corrected.std() < 0.35 * x.std()               # much tighter


def test_control_variate_rejects_mismatched_means():
    try:
        mc.control_variate(np.zeros(10), np.zeros((10, 2)), [0.0])
    except ValueError:
        return
    raise AssertionError("expected a ValueError on mismatched control means")


def test_each_technique_reduces_the_error():
    kw = dict(n_paths=8192, n_steps=16, seed=100, n_replicates=8)
    plain = mc.price_european(S0, K, T, P, option="C", **kw)
    anti = mc.price_european(S0, K, T, P, option="C", antithetic=True, **kw)
    ctrl = mc.price_european(S0, K, T, P, option="C", control=True, **kw)
    sobol = mc.price_european(S0, K, T, P, option="C", method="sobol", **kw)
    assert anti.stderr < plain.stderr
    assert ctrl.stderr < plain.stderr
    assert sobol.stderr < plain.stderr


def test_combined_techniques_stay_accurate():
    ref = float(price_cos(S0, K, 0.0, 0.0, T, P, "C"))
    res = mc.price_european(S0, K, T, P, option="C", method="sobol", antithetic=True,
                            control=True, n_paths=8192, n_steps=16, seed=100, n_replicates=8)
    assert abs(res.price - ref) < 4.0 * res.stderr


def test_asian_with_european_control():
    euro = float(price_cos(S0, K, 0.0, 0.0, T, P, "C"))
    asian = lambda paths: np.maximum(paths[1:].mean(axis=0) - K, 0.0)
    control = lambda paths: np.maximum(paths[-1] - K, 0.0)
    kw = dict(n_paths=16_384, n_steps=16, seed=42)
    plain = mc.price_path_dependent(S0, T, P, asian, **kw)
    with_cv = mc.price_path_dependent(S0, T, P, asian, controls=control, control_means=euro, **kw)
    assert with_cv.stderr < 0.75 * plain.stderr
    assert abs(with_cv.price - plain.price) < 4.0 * plain.stderr
    assert with_cv.price < euro          # averaging caps the vol, so the Asian is cheaper
