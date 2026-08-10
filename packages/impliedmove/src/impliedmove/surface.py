"""What the options market is saying, and what it implies about an event.

Two computations, and the second is unusual in that its most valuable output is a
refusal.

**The implied catalyst date.** When the market expects a binary event, at-the-money
implied volatility peaks at the first expiry that covers it. Taking the argmax of
ATM IV across expiries recovers the market's view of *when*, without anyone having
published a date.

**The two-outcome implied probability.** Given an up case and a down case, the
market price sits somewhere between them, and where it sits implies a probability.

That second one is where naive implementations manufacture edge from nothing. The
arithmetic `(spot - down) / (up - down)` always returns a number, including when
the assumed outcomes are inconsistent with what options are actually pricing. A
down case set just below spot yields an implied probability near zero — "the market
says this is hopeless" — which looks like enormous edge and is really a bad input.
So the down case is cross-checked against the options-implied move, and the
estimate is refused when they disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Expiry", "ImpliedView", "OptionsSurface", "implied_view"]

# Expiries closer than this are excluded when hunting the IV hump. Front-week IV
# is inflated by near-dated gamma and by whatever else lands that week, and it
# routinely wins the argmax spuriously — pointing the catalyst date at a thin
# contract that has nothing to do with the event.
MIN_HUMP_DTE = 10

# If the assumed downside move is smaller than this fraction of the
# options-implied move, the two disagree badly enough that the implied
# probability is not trustworthy.
FLOOR_MOVE_MISMATCH_RATIO = 0.5


@dataclass(frozen=True, slots=True)
class Expiry:
    """One expiry's at-the-money summary.

    Attributes:
        label: Identifier for the expiry — a date string, usually. Carried
            through untouched.
        dte: Calendar days to expiry.
        atm_iv: At-the-money implied volatility, annualized. `None` when it could
            not be computed, in which case this expiry is skipped rather than
            treated as zero.
        atm_straddle: Combined price of the at-the-money call and put. This is the
            market's expected absolute move by this expiry.
    """

    label: str
    dte: int
    atm_iv: float | None = None
    atm_straddle: float | None = None


@dataclass(frozen=True, slots=True)
class OptionsSurface:
    """A term structure of at-the-money quotes for one underlying."""

    symbol: str
    spot: float
    expiries: tuple[Expiry, ...]


@dataclass(frozen=True, slots=True)
class ImpliedView:
    """What the surface implies, and what cannot be trusted about it.

    Attributes:
        symbol: The underlying.
        spot: Spot price used.
        front_iv: ATM IV of the nearest expiry.
        peak_iv: Highest ATM IV among eligible expiries.
        hump_expiry: Label of the expiry where IV peaks — the market-implied
            catalyst window.
        implied_move: Expected absolute move by the hump expiry, as a fraction of
            spot.
        implied_probability: Two-outcome probability of the up case, or `None`
            when it cannot be trusted.
        caveats: Why, in words. Non-empty whenever `implied_probability` is None
            for a reason other than missing inputs, and sometimes when it is not
            None but deserves scrutiny.
    """

    symbol: str
    spot: float
    front_iv: float | None = None
    peak_iv: float | None = None
    hump_expiry: str | None = None
    implied_move: float | None = None
    implied_probability: float | None = None
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def trustworthy(self) -> bool:
        """Whether a probability came back with no reservations attached."""
        return self.implied_probability is not None and not self.caveats


def implied_view(
    surface: OptionsSurface,
    *,
    down_target: float | None = None,
    up_target: float | None = None,
) -> ImpliedView:
    """Read the market's view of an event off the surface.

    Args:
        surface: At-the-money quotes across expiries.
        down_target: Assumed price if the event fails.
        up_target: Assumed price if the event succeeds.

    Returns:
        An [`ImpliedView`][impliedmove.surface.ImpliedView]. Supplying no targets
        still yields the catalyst date and implied move; the probability needs
        both.

    Raises:
        ValueError: If spot is not positive.
    """
    if surface.spot <= 0:
        msg = f"spot must be positive, got {surface.spot}"
        raise ValueError(msg)

    caveats: list[str] = []
    dated = [e for e in surface.expiries if e.atm_iv is not None]

    front_iv = peak_iv = hump_label = implied_move = None
    if dated:
        front_iv = min(dated, key=lambda e: e.dte).atm_iv
        # Fall back to every expiry if none clears the near-dated exclusion —
        # better a suspect answer, flagged, than no answer at all.
        eligible = [e for e in dated if e.dte >= MIN_HUMP_DTE]
        if not eligible:
            eligible = dated
            caveats.append(
                f"Every expiry is under {MIN_HUMP_DTE} DTE, so the IV peak may be "
                f"near-dated noise rather than an event."
            )
        hump = max(eligible, key=lambda e: (e.atm_iv or 0.0, -e.dte))
        peak_iv = hump.atm_iv
        hump_label = hump.label
        if hump.atm_straddle is not None:
            implied_move = hump.atm_straddle / surface.spot

    probability = None
    if down_target is not None and up_target is not None:
        probability, probability_caveats = _two_outcome_probability(
            surface.spot, down_target, up_target, implied_move
        )
        caveats.extend(probability_caveats)

    return ImpliedView(
        symbol=surface.symbol,
        spot=surface.spot,
        front_iv=front_iv,
        peak_iv=peak_iv,
        hump_expiry=hump_label,
        implied_move=implied_move,
        implied_probability=probability,
        caveats=tuple(caveats),
    )


def _two_outcome_probability(
    spot: float,
    down: float,
    up: float,
    implied_move: float | None,
) -> tuple[float | None, list[str]]:
    """Probability of the up case, or None with a reason.

    Never clamps. Clamping is how a broken input becomes a confident reading: a
    down case at or above spot yields zero, which reads as "the market says this
    cannot succeed" and maximizes apparent edge at exactly the moment the inputs
    are wrong.
    """
    caveats: list[str] = []

    if up <= down:
        return None, [f"Up target {up} is not above down target {down}."]
    if not (down < spot < up):
        return None, [
            f"Spot {spot} is outside the bracket ({down}, {up}), so the market is "
            f"not pricing between the two outcomes and no probability follows."
        ]

    probability = (spot - down) / (up - down)

    # Cross-check the assumed downside against what options are pricing. If the
    # market expects a 40% move and the failure case is 8% below spot, the two
    # disagree, and the probability is an artifact of the bracket rather than a
    # reading of the market.
    if implied_move is not None and implied_move > 0:
        assumed_down_move = (spot - down) / spot
        if assumed_down_move < FLOOR_MOVE_MISMATCH_RATIO * implied_move:
            caveats.append(
                f"Assumed downside of {assumed_down_move:.1%} is far smaller than the "
                f"options-implied move of {implied_move:.1%}; the implied probability "
                f"of {probability:.0%} reflects the bracket, not the market."
            )

    return probability, caveats
