import numpy as np

from volsurface import exotics, mc
from volsurface.heston import HestonParams

P = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.05)
S0, K, T = 100.0, 100.0, 1.0


def _grid(n_paths=20_000, n_steps=32, seed=11):
    return mc.simulate(S0, T, P, n_paths=n_paths, n_steps=n_steps, seed=seed, return_variance=True)


def test_in_out_parity():
    _, paths, var = _grid()
    ko = exotics.barrier_payoff(paths, K, 80.0, T, "C", "do", variance=var).mean()
    ki = exotics.barrier_payoff(paths, K, 80.0, T, "C", "di", variance=var).mean()
    vanilla = np.maximum(paths[-1] - K, 0.0).mean()
    assert abs((ko + ki) - vanilla) < 1e-8


def test_knock_out_is_cheaper_than_vanilla():
    _, paths, var = _grid()
    ko = exotics.barrier_payoff(paths, K, 85.0, T, "C", "do", variance=var).mean()
    assert 0.0 < ko < np.maximum(paths[-1] - K, 0.0).mean()


def test_bridge_correction_lowers_the_knock_out():
    """Discrete monitoring misses crossings, so the raw payoff is too generous."""
    _, paths, var = _grid()
    raw = exotics.barrier_payoff(paths, K, 90.0, T, "C", "do", correction="none").mean()
    bridge = exotics.barrier_payoff(paths, K, 90.0, T, "C", "do", variance=var, correction="bridge").mean()
    shift = exotics.barrier_payoff(paths, K, 90.0, T, "C", "do", correction="shift", vol_hint=0.22).mean()
    assert bridge < raw
    assert shift < raw


def test_bridge_is_more_stable_across_step_counts_than_raw():
    raws, bridges = [], []
    for ns in (16, 64, 256):
        _, paths, var = _grid(n_paths=20_000, n_steps=ns)
        raws.append(exotics.barrier_payoff(paths, K, 85.0, T, "C", "do", correction="none").mean())
        bridges.append(exotics.barrier_payoff(paths, K, 85.0, T, "C", "do", variance=var).mean())
    assert np.ptp(bridges) < np.ptp(raws)


def test_barrier_correction_needs_its_inputs():
    _, paths, _ = _grid(n_paths=1000, n_steps=8)
    for kwargs in ({"correction": "bridge"}, {"correction": "shift"}):
        try:
            exotics.barrier_payoff(paths, K, 80.0, T, "C", "do", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected a ValueError for {kwargs}")


def test_autocall_redeems_at_par_plus_coupon():
    _, paths, _ = _grid()
    obs = [8, 16, 24, 32]
    pay = exotics.autocall_payoff(paths, obs, coupon=0.06, autocall_barrier=1.0,
                                  protection_barrier=0.7, notional=100.0)
    assert np.all(pay >= 0)
    # a path called on the first date pays exactly par plus one coupon
    called_first = paths[obs[0]] / paths[0] >= 1.0
    assert np.allclose(pay[called_first], 106.0)


def test_autocall_pays_less_when_the_coupon_is_smaller():
    _, paths, _ = _grid()
    obs = [8, 16, 24, 32]
    lo = exotics.autocall_payoff(paths, obs, 0.02, 1.0, 0.7).mean()
    hi = exotics.autocall_payoff(paths, obs, 0.10, 1.0, 0.7).mean()
    assert lo < hi


def test_cliquet_respects_its_caps_and_floors():
    _, paths, _ = _grid()
    resets = [0, 8, 16, 24, 32]
    pay = exotics.cliquet_payoff(paths, resets, local_cap=0.08, local_floor=-0.08,
                                 global_floor=0.0, notional=100.0)
    n_periods = len(resets) - 1
    assert np.all(pay >= 0.0)                      # global floor
    assert np.all(pay <= 100.0 * 0.08 * n_periods)  # every period capped


def test_cliquet_is_worth_more_with_a_higher_cap():
    _, paths, _ = _grid()
    resets = [0, 8, 16, 24, 32]
    tight = exotics.cliquet_payoff(paths, resets, local_cap=0.04).mean()
    loose = exotics.cliquet_payoff(paths, resets, local_cap=0.12).mean()
    assert tight < loose
