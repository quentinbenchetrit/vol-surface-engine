import numpy as np

from volsurface import greeks
from volsurface.heston import HestonParams, price_cos

P = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.05)
S0, K, T = 100.0, 100.0, 1.0
KW = dict(n_paths=100_000, n_steps=32, seed=7)


def _reference_delta(h=0.01):
    """Deterministic reference: bump the COS price, no Monte Carlo noise."""
    up = float(price_cos(S0 + h, K, 0.0, 0.0, T, P, "C"))
    dn = float(price_cos(S0 - h, K, 0.0, 0.0, T, P, "C"))
    return (up - dn) / (2.0 * h)


def test_all_estimators_agree_with_the_reference():
    ref = _reference_delta()
    for g in (greeks.delta_pathwise(S0, K, T, P, **KW),
              greeks.delta_likelihood_ratio(S0, K, T, P, **KW),
              greeks.delta_finite_difference(S0, K, T, P, **KW)):
        assert abs(g.value - ref) < 4.0 * g.stderr, g.method


def test_delta_is_between_zero_and_one():
    g = greeks.delta_pathwise(S0, K, T, P, **KW)
    assert 0.0 < g.value < 1.0


def test_pathwise_has_less_variance_than_likelihood_ratio():
    pw = greeks.delta_pathwise(S0, K, T, P, **KW)
    lr = greeks.delta_likelihood_ratio(S0, K, T, P, **KW)
    assert pw.stderr < 0.5 * lr.stderr


def test_common_random_numbers_beat_independent_runs():
    crn = greeks.delta_finite_difference(S0, K, T, P, common_random=True, **KW)
    ind = greeks.delta_finite_difference(S0, K, T, P, common_random=False, **KW)
    assert crn.stderr < 0.5 * ind.stderr


def test_pathwise_is_invalid_for_a_digital():
    """The indicator has zero derivative where it exists, so pathwise gives zero.

    Likelihood ratio differentiates the density instead and stays valid.
    """
    dig = greeks.digital_payoff(K)
    lr = greeks.delta_likelihood_ratio(S0, K, T, P, payoff=dig, **KW)
    ref = greeks.delta_finite_difference(S0, K, T, P, payoff=dig, h=0.5,
                                         n_paths=200_000, n_steps=32, seed=1)
    assert lr.value > 0.0
    assert abs(lr.value - ref.value) < 4.0 * np.hypot(lr.stderr, ref.stderr)


def test_deep_in_the_money_delta_approaches_one():
    g = greeks.delta_pathwise(S0, 50.0, T, P, **KW)
    assert g.value > 0.9


def test_vega_is_positive():
    g = greeks.vega_finite_difference(S0, K, T, P, **KW)
    assert g.value > 0.0


def test_greek_result_confidence_interval():
    g = greeks.delta_pathwise(S0, K, T, P, **KW)
    lo, hi = g.ci95
    assert lo < g.value < hi
