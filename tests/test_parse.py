from datetime import datetime, timezone

import pytest

from volsurface.data import parse_instrument


def test_parse_two_digit_day():
    inst = parse_instrument("BTC-25DEC26-20000-P")
    assert inst.underlying == "BTC"
    assert inst.strike == 20000.0
    assert inst.option_type == "P"
    assert inst.expiry == datetime(2026, 12, 25, 8, tzinfo=timezone.utc)


def test_parse_single_digit_day_call():
    inst = parse_instrument("BTC-1AUG26-64000-C")
    assert inst.expiry == datetime(2026, 8, 1, 8, tzinfo=timezone.utc)
    assert inst.option_type == "C"
    assert inst.strike == 64000.0


def test_parse_eth_and_fractional_strike():
    inst = parse_instrument("ETH-7AUG26-3500.5-C")
    assert inst.underlying == "ETH"
    assert inst.strike == 3500.5


@pytest.mark.parametrize("bad", ["BTC-PERPETUAL", "BTC-1AUG26-64000", "garbage"])
def test_parse_rejects_non_options(bad):
    with pytest.raises(ValueError):
        parse_instrument(bad)
