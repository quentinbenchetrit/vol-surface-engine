import numpy as np
import pytest

from volsurface import mc
from volsurface.heston import HestonParams, price_cos

P = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.05)
S0, T = 100.0, 1.0

needs_kernel = pytest.mark.skipif(not mc.has_fast_kernel(),
                                  reason="the C++ kernel is not built in this install")


def test_python_path_always_available():
    """The NumPy implementation must work whether or not the kernel is built."""
    ST = mc.simulate(S0, T, P, n_paths=2000, n_steps=16, seed=0, use_python=True)
    assert ST.shape == (2000,) and np.all(ST > 0)


@needs_kernel
def test_kernel_matches_python_path_by_path():
    """Same uniforms, same scheme, so the two implementations must agree closely."""
    kw = dict(n_paths=20_000, n_steps=64, seed=1)
    slow = mc.simulate(S0, T, P, **kw, use_python=True)
    fast = mc.simulate(S0, T, P, **kw)
    assert np.allclose(fast, slow, rtol=1e-9, atol=0.0)


@needs_kernel
@pytest.mark.parametrize("r,q", [(0.0, 0.0), (0.03, 0.01)])
def test_kernel_matches_python_with_carry(r, q):
    kw = dict(n_paths=10_000, n_steps=32, seed=4)
    slow = mc.simulate(S0, T, P, r, q, **kw, use_python=True)
    fast = mc.simulate(S0, T, P, r, q, **kw)
    assert np.allclose(fast, slow, rtol=1e-9, atol=0.0)


@needs_kernel
def test_kernel_agrees_across_thread_boundaries():
    """Path counts above and below the threading threshold must give the same answer."""
    for n_paths in (1000, 20_000, 60_000):
        U = mc.draw_uniforms(n_paths, 24, "pseudo", 7, False)
        slow = mc.simulate(S0, T, P, n_steps=24, uniforms=U, use_python=True)
        fast = mc.simulate(S0, T, P, n_steps=24, uniforms=U)
        assert np.allclose(fast, slow, rtol=1e-9, atol=0.0), n_paths


@needs_kernel
def test_kernel_reprices_a_vanilla():
    ref = float(price_cos(S0, 100.0, 0.0, 0.0, T, P, "C"))
    res = mc.price_european(S0, 100.0, T, P, option="C", n_paths=200_000, n_steps=64, seed=7)
    assert abs(res.price - ref) < 4.0 * res.stderr


@needs_kernel
def test_kernel_works_with_sobol_and_antithetics():
    kw = dict(n_paths=8192, n_steps=32, seed=3, method="sobol", antithetic=True)
    slow = mc.simulate(S0, T, P, **kw, use_python=True)
    fast = mc.simulate(S0, T, P, **kw)
    assert np.allclose(fast, slow, rtol=1e-9, atol=0.0)


@needs_kernel
def test_kernel_is_skipped_when_paths_are_requested():
    """Only terminal spots come from C++, so richer outputs stay on the Python path."""
    ST, paths = mc.simulate(S0, T, P, n_paths=500, n_steps=8, seed=2, return_paths=True)
    assert paths.shape == (9, 500)
    assert np.allclose(paths[-1], ST)


@needs_kernel
def test_kernel_rejects_malformed_uniforms():
    from volsurface import _core

    with pytest.raises(Exception):
        _core.simulate_qe(np.zeros((10, 7)), S0, T, 2.0, 0.04, 0.6, -0.7, 0.05, 0.0, 0.0)
