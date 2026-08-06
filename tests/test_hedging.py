import numpy as np

from volsurface import hedging, mc
from volsurface.heston import HestonParams

S0, K, T, SIGMA = 100.0, 100.0, 1.0, 0.20
CALL = lambda ST: np.maximum(ST - K, 0.0)


def _setup(n_steps, n_paths=40_000, r=0.0, seed=1):
    price, delta = hedging.black_scholes_call(K, T, SIGMA, r)
    premium = float(price(np.array([S0]), 0.0)[0])
    _, paths = mc.simulate_gbm(S0, T, SIGMA, r, n_paths=n_paths, n_steps=n_steps, seed=seed)
    return hedging.delta_hedge(paths, T, premium, CALL, delta, r), premium


def test_gbm_is_a_martingale_with_zero_rate():
    ST = mc.simulate_gbm(S0, T, SIGMA, n_paths=200_000, n_steps=32, seed=0, return_paths=False)
    stderr = ST.std(ddof=1) / np.sqrt(ST.size)
    assert abs(ST.mean() - S0) < 4.0 * stderr
    assert np.all(ST > 0)


def test_gbm_paths_start_at_spot():
    _, paths = mc.simulate_gbm(S0, T, SIGMA, n_paths=1000, n_steps=10, seed=0)
    assert paths.shape == (11, 1000)
    assert np.allclose(paths[0], S0)


def test_hedge_pnl_is_centred_on_zero():
    """If the price and the delta come from the same model, the hedge is unbiased."""
    res, _ = _setup(64)
    assert abs(res.mean) < 4.0 * res.stderr


def test_hedge_error_halves_like_sqrt_dt():
    """Four times the rebalancing should halve the spread of the hedging error."""
    coarse, _ = _setup(16)
    fine, _ = _setup(64)
    ratio = coarse.std / fine.std
    assert 1.7 < ratio < 2.3


def test_hedging_beats_doing_nothing():
    res, premium = _setup(64)
    _, paths = mc.simulate_gbm(S0, T, SIGMA, n_paths=40_000, n_steps=64, seed=1)
    naked = premium - CALL(paths[-1])
    assert res.std < 0.35 * naked.std(ddof=1)


def test_hedge_works_with_a_nonzero_rate():
    res, _ = _setup(64, r=0.03)
    assert abs(res.mean) < 4.0 * res.stderr


def test_delta_is_bounded_and_monotone():
    _, delta = hedging.black_scholes_call(K, T, SIGMA)
    spots = np.array([60.0, 90.0, 100.0, 110.0, 160.0])
    d = delta(spots, 0.0)
    assert np.all((d > 0.0) & (d < 1.0))
    assert np.all(np.diff(d) > 0)


def test_delta_at_maturity_is_the_exercise_indicator():
    _, delta = hedging.black_scholes_call(K, T, SIGMA)
    d = delta(np.array([90.0, 110.0]), T)
    assert np.allclose(d, [0.0, 1.0])


def test_result_reports_its_grid():
    res, _ = _setup(32)
    assert res.n_rebalances == 32
    assert res.pnl.size == 40_000


# selling volatility that the market does not deliver

SELL, REALIZED = 0.25, 0.20


def _edge():
    price_sell, _ = hedging.black_scholes_call(K, T, SELL)
    price_real, _ = hedging.black_scholes_call(K, T, REALIZED)
    return float(price_sell(np.array([S0]), 0.0)[0] - price_real(np.array([S0]), 0.0)[0])


def test_selling_rich_volatility_earns_the_value_difference():
    """Either hedge earns the same average: the gap between the two BS values."""
    edge = _edge()
    for hedge_vol in (REALIZED, SELL):
        res = hedging.hedge_under_gbm(S0, K, T, SELL, hedge_vol, REALIZED,
                                      n_paths=20_000, n_steps=256, seed=3)
        assert abs(res.mean - edge) < 4.0 * res.stderr


def test_hedging_at_realized_vol_is_the_tighter_of_the_two():
    kw = dict(n_paths=20_000, n_steps=512, seed=3)
    at_realized = hedging.hedge_under_gbm(S0, K, T, SELL, REALIZED, REALIZED, **kw)
    at_implied = hedging.hedge_under_gbm(S0, K, T, SELL, SELL, REALIZED, **kw)
    assert at_realized.std < at_implied.std


def test_hedging_at_realized_vol_becomes_deterministic():
    """Its spread is discretisation only, so it keeps shrinking with the grid."""
    kw = dict(n_paths=20_000, seed=3)
    coarse = hedging.hedge_under_gbm(S0, K, T, SELL, REALIZED, REALIZED, n_steps=64, **kw)
    fine = hedging.hedge_under_gbm(S0, K, T, SELL, REALIZED, REALIZED, n_steps=1024, **kw)
    assert fine.std < 0.35 * coarse.std


def test_hedging_at_implied_vol_never_loses_when_selling_rich():
    """The profit is a gamma-weighted integral with a fixed sign."""
    res = hedging.hedge_under_gbm(S0, K, T, SELL, SELL, REALIZED,
                                  n_paths=20_000, n_steps=512, seed=3)
    assert res.pnl.min() > 0.0


# model error, which rebalancing cannot fix

HES = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.05)


def test_heston_hedge_is_unbiased():
    res = hedging.hedge_under_heston(S0, K, T, HES, n_paths=20_000, n_steps=256, seed=5)
    assert abs(res.mean) < 4.0 * res.stderr


def test_model_error_does_not_vanish_with_finer_rebalancing():
    kw = dict(n_paths=20_000, seed=5)
    coarse = hedging.hedge_under_heston(S0, K, T, HES, n_steps=128, **kw)
    fine = hedging.hedge_under_heston(S0, K, T, HES, n_steps=1024, **kw)
    # a spot-only hedge cannot touch variance risk, so the spread plateaus
    assert fine.std > 0.8 * coarse.std
