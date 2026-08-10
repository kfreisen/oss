"""Black-Scholes pricing, Greeks, and implied volatility, from `math` alone.

Computing our own implied volatility rather than reading a feed's `iv` field is
deliberate. On illiquid chains — small caps, far-dated expiries, anything with a
wide spread — vendor IV is frequently stale, inconsistent between strikes, or
simply absent, and the entire analysis downstream of it inherits that.

The risk-free rate defaults to zero. For the short-dated, event-driven horizons
this is built for, discounting over a few weeks moves the answer far less than the
bid-ask spread does, and carrying a rate people will not supply correctly is worse
than not carrying one. Pass `r` when it matters.
"""

from __future__ import annotations

import math

__all__ = ["bs_price", "delta", "implied_vol", "norm_cdf"]

_SQRT2 = math.sqrt(2.0)

# Bisection bounds for implied vol: 0.01% to 1000% annualized. The upper bound is
# absurd for a liquid name and routinely hit by a binary-event biotech.
_MIN_VOL = 1e-4
_MAX_VOL = 10.0
_VOL_TOLERANCE = 1e-8

# Bisection halves the interval every step, so the iteration count needed to reach
# the tolerance is known exactly rather than guessed at. Running a fixed number is
# both simpler and more predictable than looping with an early exit plus a large
# safety cap — the cap in that arrangement is unreachable, which means it is also
# untestable, and an untestable branch in numerical code is a place bugs live.
_ITERATIONS = math.ceil(math.log2((_MAX_VOL - _MIN_VOL) / _VOL_TOLERANCE))


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _d1(spot: float, strike: float, years: float, sigma: float, rate: float) -> float:
    """The `d1` term. Callers must ensure `years > 0` and `sigma > 0`."""
    return (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * years) / (
        sigma * math.sqrt(years)
    )


def bs_price(
    spot: float,
    strike: float,
    years: float,
    sigma: float,
    *,
    call: bool,
    rate: float = 0.0,
) -> float:
    """Black-Scholes price of a European option.

    Args:
        spot: Current underlying price.
        strike: Strike price.
        years: Time to expiry in years.
        sigma: Annualized volatility.
        call: True for a call, False for a put.
        rate: Continuously compounded risk-free rate.

    Returns:
        The option's theoretical value. At or past expiry, or with zero
        volatility, this is the intrinsic value — the limit the formula
        approaches, returned directly rather than dividing by zero on the way.

    Raises:
        ValueError: If `spot` or `strike` is not positive.
    """
    _require_positive(spot=spot, strike=strike)
    if years <= 0 or sigma <= 0:
        intrinsic = (spot - strike) if call else (strike - spot)
        return max(intrinsic, 0.0)

    d1 = _d1(spot, strike, years, sigma, rate)
    d2 = d1 - sigma * math.sqrt(years)
    discount = math.exp(-rate * years)
    if call:
        return spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    return strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)


def delta(
    spot: float,
    strike: float,
    years: float,
    sigma: float,
    *,
    call: bool,
    rate: float = 0.0,
) -> float:
    """Sensitivity of the option price to the underlying.

    At expiry this is the step function: 1 or 0 for a call, -1 or 0 for a put.
    Exactly at the money at expiry it returns 0, which is a convention rather than
    a fact — delta is undefined there.
    """
    _require_positive(spot=spot, strike=strike)
    if years <= 0 or sigma <= 0:
        if call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1 = _d1(spot, strike, years, sigma, rate)
    return norm_cdf(d1) if call else norm_cdf(d1) - 1.0


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    years: float,
    *,
    call: bool,
    rate: float = 0.0,
) -> float | None:
    """Invert Black-Scholes for volatility, by bisection.

    Bisection rather than Newton-Raphson on purpose. Newton is faster and it
    diverges on exactly the inputs this exists to handle — deep out-of-the-money
    options where vega is nearly zero, and crossed or stale quotes. Bisection on a
    monotone function cannot diverge, and the cost is microseconds.

    Args:
        price: Observed option price, typically the mid.
        spot: Current underlying price.
        strike: Strike price.
        years: Time to expiry in years.
        call: True for a call, False for a put.
        rate: Continuously compounded risk-free rate.

    Returns:
        The implied volatility, or **None** when no volatility reproduces the
        price. Returning None rather than a clamped bound is the point: a price
        below intrinsic is a bad quote, not a low-volatility signal, and quietly
        substituting the nearest legal number turns a data problem into a
        confident trade.
    """
    if price <= 0 or years <= 0 or spot <= 0 or strike <= 0:
        return None

    intrinsic = max((spot - strike) if call else (strike - spot), 0.0)
    # A tiny tolerance for rounding in the quote itself; anything further below
    # intrinsic is arbitrage or bad data.
    if price < intrinsic - 1e-6:
        return None
    if bs_price(spot, strike, years, _MAX_VOL, call=call, rate=rate) < price:
        return None

    low, high = _MIN_VOL, _MAX_VOL
    for _ in range(_ITERATIONS):
        mid = 0.5 * (low + high)
        if bs_price(spot, strike, years, mid, call=call, rate=rate) > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def _require_positive(**values: float) -> None:
    """Reject non-positive prices, which the log in `d1` cannot accept."""
    for name, value in values.items():
        if value <= 0:
            msg = f"{name} must be positive, got {value}"
            raise ValueError(msg)
