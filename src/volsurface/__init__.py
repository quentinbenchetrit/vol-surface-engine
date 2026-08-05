"""Arbitrage-free implied volatility surface and stochastic-volatility calibration.

Typical use:

    from volsurface import implied_vol_surface, ssvi, heston, dupire
    from volsurface.data import load_chain, forward_curve

    chain = load_chain("BTC")
    surface = implied_vol_surface(chain, forward_curve(chain))
    fit = ssvi.calibrate(surface)              # arbitrage-free surface
    hes = heston.calibrate(surface)            # dynamic model
    lv = dupire.local_vol(0.0, 0.5, fit.params)
"""

from . import dupire, exotics, greeks, heston, mc, ssvi, svi
from .black import black76_price, black76_vega, black76_delta
from .impliedvol import implied_vol, implied_vol_surface

__version__ = "0.1.0"

__all__ = [
    "black76_price",
    "black76_vega",
    "black76_delta",
    "implied_vol",
    "implied_vol_surface",
    "svi",
    "ssvi",
    "heston",
    "dupire",
    "mc",
    "exotics",
    "greeks",
]
