"""Pricing, Greeks, and implied-volatility inversion."""

from __future__ import annotations

import math

import pytest
from impliedmove.blackscholes import bs_price, delta, implied_vol, norm_cdf


def test_norm_cdf_at_known_points() -> None:
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(-1.959964) == pytest.approx(0.025, abs=1e-6)
    assert norm_cdf(1.959964) == pytest.approx(0.975, abs=1e-6)


def test_put_call_parity_holds() -> None:
    """C - P = S - K e^{-rT}. Independent of the formula, so it is a real check."""
    spot, strike, years, sigma, rate = 100.0, 95.0, 0.5, 0.3, 0.04
    call = bs_price(spot, strike, years, sigma, call=True, rate=rate)
    put = bs_price(spot, strike, years, sigma, call=False, rate=rate)
    assert call - put == pytest.approx(spot - strike * math.exp(-rate * years))


def test_price_is_monotone_in_volatility() -> None:
    prices = [bs_price(100, 100, 1.0, v, call=True) for v in (0.1, 0.2, 0.4, 0.8)]
    assert prices == sorted(prices)


def test_at_expiry_price_is_intrinsic() -> None:
    assert bs_price(110, 100, 0.0, 0.3, call=True) == 10.0
    assert bs_price(90, 100, 0.0, 0.3, call=True) == 0.0
    assert bs_price(90, 100, 0.0, 0.3, call=False) == 10.0
    assert bs_price(110, 100, 0.0, 0.3, call=False) == 0.0


def test_zero_volatility_gives_intrinsic() -> None:
    assert bs_price(110, 100, 1.0, 0.0, call=True) == 10.0


@pytest.mark.parametrize(("spot", "strike"), [(0, 100), (100, 0), (-1, 100)])
def test_non_positive_inputs_are_rejected(spot: float, strike: float) -> None:
    # The log in d1 cannot take these, and a silent nan is much worse than a raise.
    with pytest.raises(ValueError, match="must be positive"):
        bs_price(spot, strike, 1.0, 0.3, call=True)


def test_delta_of_an_at_the_money_call_is_near_half() -> None:
    assert delta(100, 100, 1.0, 0.2, call=True) == pytest.approx(0.54, abs=0.02)


def test_call_and_put_delta_differ_by_one() -> None:
    call = delta(100, 95, 0.5, 0.3, call=True)
    put = delta(100, 95, 0.5, 0.3, call=False)
    assert call - put == pytest.approx(1.0)


def test_delta_at_expiry_is_a_step() -> None:
    assert delta(110, 100, 0.0, 0.3, call=True) == 1.0
    assert delta(90, 100, 0.0, 0.3, call=True) == 0.0
    assert delta(90, 100, 0.0, 0.3, call=False) == -1.0
    assert delta(110, 100, 0.0, 0.3, call=False) == 0.0


def test_delta_exactly_at_the_money_at_expiry_is_zero_by_convention() -> None:
    # Undefined in truth; documented as 0 so the function is total.
    assert delta(100, 100, 0.0, 0.3, call=True) == 0.0
    assert delta(100, 100, 0.0, 0.3, call=False) == 0.0


def test_delta_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        delta(0, 100, 1.0, 0.3, call=True)


@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.5, 1.5, 4.0])
@pytest.mark.parametrize("call", [True, False])
def test_implied_vol_round_trips(sigma: float, *, call: bool) -> None:
    price = bs_price(100, 105, 0.5, sigma, call=call)
    assert implied_vol(price, 100, 105, 0.5, call=call) == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_round_trips_with_a_rate() -> None:
    price = bs_price(100, 105, 0.5, 0.35, call=True, rate=0.05)
    assert implied_vol(price, 100, 105, 0.5, call=True, rate=0.05) == pytest.approx(0.35, abs=1e-5)


def test_implied_vol_returns_none_below_intrinsic() -> None:
    """A crossed or stale quote, not a low-volatility reading.

    Returning the nearest legal volatility here is how a data problem becomes a
    confident trade, so this returns None instead.
    """
    assert implied_vol(5.0, 110, 100, 0.5, call=True) is None


def test_implied_vol_returns_none_above_the_search_range() -> None:
    # A call can never be worth more than the underlying, so no volatility
    # reproduces this price. Note 99.0 would NOT do: at 1000% vol an at-the-money
    # call is worth very nearly spot, so the search range reaches almost to 100.
    assert implied_vol(105.0, 100, 100, 0.5, call=True) is None


@pytest.mark.parametrize(
    ("price", "spot", "strike", "years"),
    [
        (0.0, 100, 100, 1.0),
        (-1.0, 100, 100, 1.0),
        (5.0, 100, 100, 0.0),
        (5.0, 0, 100, 1.0),
        (5.0, 100, 0, 1.0),
    ],
)
def test_implied_vol_rejects_unusable_inputs(
    price: float, spot: float, strike: float, years: float
) -> None:
    assert implied_vol(price, spot, strike, years, call=True) is None


def test_implied_vol_accepts_a_price_exactly_at_intrinsic() -> None:
    """Deep in the money with no time value: a small vol, not a refusal.

    It converges near 1.8% rather than to the 0.01% floor, and that is a property
    of double precision rather than of the option: below roughly that volatility
    the time value of this contract is smaller than the last representable digit
    of a price near 10, so bisection cannot distinguish it from intrinsic. Worth
    knowing before reading a very low implied vol as meaningful.
    """
    result = implied_vol(10.0, 110, 100, 0.5, call=True)
    assert result is not None
    assert 0.0 < result < 0.05
