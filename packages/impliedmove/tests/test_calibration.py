"""Brier score, reliability curves, and directional hit rate."""

from __future__ import annotations

import pytest
from impliedmove.calibration import (
    brier_score,
    directional_hit_rate,
    reliability,
)


def test_a_perfect_forecaster_scores_zero() -> None:
    assert brier_score([1.0, 0.0, 1.0], [True, False, True]) == 0.0


def test_a_maximally_wrong_forecaster_scores_one() -> None:
    assert brier_score([0.0, 1.0], [True, False]) == 1.0


def test_always_saying_half_scores_a_quarter() -> None:
    # The reference point every other score should be read against.
    assert brier_score([0.5] * 4, [True, False, True, False]) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("predictions", "outcomes", "match"),
    [
        ([0.5], [True, False], "1 predictions and 2 outcomes"),
        ([], [], "no forecasts"),
        ([1.5], [True], r"must be in \[0, 1\]"),
        ([-0.5], [True], r"must be in \[0, 1\]"),
    ],
)
def test_unusable_input_is_rejected(
    predictions: list[float], outcomes: list[bool], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        brier_score(predictions, outcomes)


def test_reliability_buckets_and_compares() -> None:
    # Four forecasts at 0.9 of which two happened: predicted 90%, observed 50%.
    result = reliability([0.9] * 4, [True, True, False, False], n_buckets=10)
    assert len(result.buckets) == 1
    bucket = result.buckets[0]
    assert bucket.count == 4
    assert bucket.mean_prediction == pytest.approx(0.9)
    assert bucket.observed_rate == pytest.approx(0.5)
    assert bucket.gap == pytest.approx(0.4)


def test_a_well_calibrated_forecaster_has_a_small_gap() -> None:
    predictions = [0.2] * 10 + [0.8] * 10
    outcomes = [True] * 2 + [False] * 8 + [True] * 8 + [False] * 2
    result = reliability(predictions, outcomes, n_buckets=5)
    assert result.max_gap == pytest.approx(0.0, abs=1e-9)


def test_empty_buckets_are_dropped() -> None:
    """A band nobody forecast into says nothing.

    Reporting it as zero would drag any summary toward a meaningless number.
    """
    result = reliability([0.05, 0.95], [False, True], n_buckets=10)
    assert len(result.buckets) == 2


def test_a_forecast_of_exactly_one_stays_in_the_top_bucket() -> None:
    # Without the clamp this creates an extra one-wide band above the top.
    result = reliability([1.0], [True], n_buckets=10)
    assert len(result.buckets) == 1
    assert result.buckets[0].low == pytest.approx(0.9)


def test_buckets_come_back_in_ascending_order() -> None:
    result = reliability([0.9, 0.1, 0.5], [True, False, True], n_buckets=10)
    assert [b.low for b in result.buckets] == sorted(b.low for b in result.buckets)


def test_the_weighted_gap_respects_bucket_population() -> None:
    """A terrible band holding two forecasts matters less than a mediocre one
    holding two hundred, and the headline number should say so."""
    predictions = [0.9] * 2 + [0.5] * 100
    outcomes = [False] * 2 + [True] * 50 + [False] * 50
    result = reliability(predictions, outcomes, n_buckets=10)
    assert result.max_gap == pytest.approx(0.9)
    assert result.weighted_gap < 0.05


def test_a_single_bucket_is_allowed() -> None:
    result = reliability([0.3, 0.7], [True, False], n_buckets=1)
    assert len(result.buckets) == 1


def test_too_few_buckets_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_buckets must be at least 1"):
        reliability([0.5], [True], n_buckets=0)


def test_directional_hit_rate_counts_correct_sides() -> None:
    assert directional_hit_rate([0.9, 0.1, 0.8], [True, False, False]) == pytest.approx(2 / 3)


def test_forecasts_at_the_reference_express_no_direction() -> None:
    assert directional_hit_rate([0.5, 0.9], [False, True], reference=0.5) == 1.0


def test_no_directional_forecasts_scores_zero_rather_than_dividing_by_zero() -> None:
    assert directional_hit_rate([0.5, 0.5], [True, False]) == 0.0


def test_the_reference_can_be_a_market_price() -> None:
    """Beating a market is a different question from being right in the abstract.

    Against a market at 0.70, a forecast of 0.75 is a bet the event happens.
    """
    assert directional_hit_rate([0.75, 0.60], [True, True], reference=0.70) == pytest.approx(0.5)


def test_hit_rate_validates_its_input() -> None:
    with pytest.raises(ValueError, match="no forecasts"):
        directional_hit_rate([], [])
