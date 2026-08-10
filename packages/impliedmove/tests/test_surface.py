"""Catalyst-date inference, implied move, and the self-invalidation check."""

from __future__ import annotations

import pytest
from impliedmove.surface import Expiry, OptionsSurface, implied_view


def surface(*expiries: Expiry, spot: float = 20.0) -> OptionsSurface:
    return OptionsSurface("BIO", spot, expiries)


def test_the_iv_hump_marks_the_catalyst_expiry() -> None:
    view = implied_view(
        surface(
            Expiry("2026-07", 30, atm_iv=0.8),
            Expiry("2026-09", 90, atm_iv=1.5),
            Expiry("2026-12", 180, atm_iv=0.9),
        )
    )
    assert view.hump_expiry == "2026-09"
    assert view.peak_iv == 1.5


def test_near_dated_expiries_are_excluded_from_the_hunt() -> None:
    """Front-week IV is inflated by gamma and by whatever else lands that week.

    Left in, it wins the argmax spuriously and points the catalyst date at a thin
    contract with nothing to do with the event.
    """
    view = implied_view(
        surface(
            Expiry("this-week", 3, atm_iv=2.5),
            Expiry("2026-09", 90, atm_iv=1.2),
        )
    )
    assert view.hump_expiry == "2026-09"


def test_when_everything_is_near_dated_it_answers_but_says_so() -> None:
    view = implied_view(surface(Expiry("a", 3, atm_iv=2.5), Expiry("b", 8, atm_iv=1.0)))
    assert view.hump_expiry == "a"
    assert any("near-dated" in c for c in view.caveats)


def test_front_iv_is_the_nearest_expiry_not_the_first_listed() -> None:
    view = implied_view(surface(Expiry("far", 90, atm_iv=1.5), Expiry("near", 20, atm_iv=0.7)))
    assert view.front_iv == 0.7


def test_ties_prefer_the_nearer_expiry() -> None:
    # Two expiries at the same IV: the nearer one is the more likely catalyst,
    # and a stable tie-break keeps the answer reproducible.
    view = implied_view(surface(Expiry("near", 30, atm_iv=1.0), Expiry("far", 120, atm_iv=1.0)))
    assert view.hump_expiry == "near"


def test_expiries_without_iv_are_skipped() -> None:
    view = implied_view(surface(Expiry("no-iv", 30), Expiry("has-iv", 60, atm_iv=0.9)))
    assert view.hump_expiry == "has-iv"


def test_a_surface_with_no_usable_expiries_yields_nothing() -> None:
    view = implied_view(surface(Expiry("no-iv", 30)))
    assert view.hump_expiry is None
    assert view.peak_iv is None
    assert view.implied_move is None


def test_an_empty_surface_is_not_an_error() -> None:
    view = implied_view(surface())
    assert view.hump_expiry is None


def test_implied_move_comes_from_the_hump_straddle() -> None:
    view = implied_view(surface(Expiry("2026-09", 90, atm_iv=1.5, atm_straddle=9.0)))
    assert view.implied_move == pytest.approx(0.45)


def test_no_straddle_means_no_implied_move() -> None:
    view = implied_view(surface(Expiry("2026-09", 90, atm_iv=1.5)))
    assert view.implied_move is None


def test_a_non_positive_spot_is_rejected() -> None:
    with pytest.raises(ValueError, match="spot must be positive"):
        implied_view(surface(spot=0.0))


def test_the_two_outcome_probability() -> None:
    view = implied_view(
        surface(Expiry("2026-09", 90, atm_iv=1.5, atm_straddle=9.0)),
        down_target=8.0,
        up_target=40.0,
    )
    # (20 - 8) / (40 - 8)
    assert view.implied_probability == pytest.approx(0.375)
    assert view.trustworthy


def test_no_targets_means_no_probability_and_no_complaint() -> None:
    view = implied_view(surface(Expiry("2026-09", 90, atm_iv=1.5, atm_straddle=9.0)))
    assert view.implied_probability is None
    assert view.caveats == ()


def test_only_one_target_is_not_enough() -> None:
    view = implied_view(surface(Expiry("a", 90, atm_iv=1.0)), down_target=8.0)
    assert view.implied_probability is None


def test_a_floor_inconsistent_with_the_implied_move_is_flagged() -> None:
    """The failure this package exists to catch.

    A down case just below spot yields a near-zero probability — which reads as
    "the market says this is hopeless" and looks like enormous edge. It is a bad
    input, and the options are pricing a 45% move that says so.
    """
    view = implied_view(
        surface(Expiry("2026-09", 90, atm_iv=1.5, atm_straddle=9.0)),
        down_target=19.0,
        up_target=40.0,
    )
    assert view.implied_probability is not None
    assert view.implied_probability < 0.06
    assert not view.trustworthy
    assert any("options-implied move" in c for c in view.caveats)


def test_spot_outside_the_bracket_yields_no_probability() -> None:
    """Never clamped.

    Clamping is how a stale share count becomes a top position: a floor at or
    above spot gives probability 0.0, i.e. maximum apparent edge, from a data bug.
    """
    view = implied_view(surface(Expiry("a", 90, atm_iv=1.0)), down_target=25.0, up_target=40.0)
    assert view.implied_probability is None
    assert any("outside the bracket" in c for c in view.caveats)


def test_spot_above_the_bracket_also_yields_nothing() -> None:
    view = implied_view(surface(Expiry("a", 90, atm_iv=1.0)), down_target=5.0, up_target=15.0)
    assert view.implied_probability is None


def test_an_inverted_bracket_is_rejected() -> None:
    view = implied_view(surface(Expiry("a", 90, atm_iv=1.0)), down_target=40.0, up_target=8.0)
    assert view.implied_probability is None
    assert any("not above" in c for c in view.caveats)


def test_without_an_implied_move_the_cross_check_is_skipped() -> None:
    # No straddle, so there is nothing to cross-check against. The probability is
    # returned without the mismatch caveat.
    view = implied_view(surface(Expiry("a", 90, atm_iv=1.0)), down_target=19.0, up_target=40.0)
    assert view.implied_probability is not None
    assert not any("options-implied move" in c for c in view.caveats)
