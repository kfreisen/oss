"""Text normalization, the proper-noun penalty, and the composite scorer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from venuematch.records import Record
from venuematch.scoring import (
    CompositeScorer,
    jaccard,
    normalize,
    proper_nouns,
    tokenize,
)

WHEN = datetime(2026, 6, 1, tzinfo=UTC)


def record(text: str, *, source: str = "a", offset_days: int = 0, value: float | None = None):
    return Record(
        id=text[:12],
        source=source,
        text=text,
        timestamp=WHEN + timedelta(days=offset_days),
        value=value,
    )


def test_normalize_lowercases_and_strips_punctuation() -> None:
    assert normalize("Will Acme, Inc. ship?") == "acme inc ship"


def test_normalize_handles_empty_and_none() -> None:
    assert normalize("") == ""
    assert normalize(None) == ""


def test_tokenize_drops_short_tokens() -> None:
    assert tokenize("Acme a bc of X") == frozenset({"acme", "bc"})


def test_tokenize_handles_none() -> None:
    assert tokenize(None) == frozenset()


def test_jaccard_of_two_empty_sets_is_zero() -> None:
    # Not 1.0: two records with no usable tokens are not evidence of a match.
    assert jaccard([], []) == 0.0


def test_jaccard_is_intersection_over_union() -> None:
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_proper_nouns_finds_capitalized_names() -> None:
    assert proper_nouns("Will Acme beat Zenith today?") == frozenset({"Acme", "Zenith"})


def test_proper_nouns_ignores_sentence_openers() -> None:
    # Otherwise "Will Acme ship?" and "Does Acme ship?" are penalized for
    # disagreeing on their first word, which is exactly backwards.
    assert proper_nouns("Will Acme ship?") == proper_nouns("Does Acme ship?")


def test_proper_nouns_handles_none() -> None:
    assert proper_nouns(None) == frozenset()


def test_a_short_capitalized_word_is_not_a_proper_noun() -> None:
    # Three letters or fewer: too many false positives from acronyms and openers.
    assert proper_nouns("Buy IBM Now") == frozenset()


def test_the_proper_noun_penalty_separates_similar_titles() -> None:
    """The case the whole scorer exists for.

    These two share every token but one, so any purely frequency-agnostic metric
    rates them near-identical — and merging them would be a serious error.
    """
    scorer = CompositeScorer()
    iowa = record("Will the Iowa primary be won by Smith?", source="a")
    hampshire = record("Will the Hampshire primary be won by Smith?", source="b")

    with_penalty = scorer(iowa, hampshire)
    without = CompositeScorer(proper_noun_penalty=0.0)(iowa, hampshire)

    assert with_penalty is not None
    assert without is not None
    assert with_penalty.proper_noun_diff == 2
    assert with_penalty.confidence < without.confidence
    assert without.confidence > 0.65, "without the penalty these would merge"
    assert with_penalty.confidence < 0.65, "with it, they do not"


def test_identical_text_scores_highly() -> None:
    scorer = CompositeScorer()
    score = scorer(
        record("Acme ships widgets", source="a"), record("Acme ships widgets", source="b")
    )
    assert score is not None
    assert score.confidence > 0.9


def test_the_jaccard_prefilter_rejects_early() -> None:
    scorer = CompositeScorer()
    assert (
        scorer(record("Acme ships widgets", source="a"), record("Zenith buys land", source="b"))
        is None
    )


def test_empty_text_is_rejected() -> None:
    assert CompositeScorer()(record("", source="a"), record("Acme ships", source="b")) is None


def test_a_distant_timestamp_is_rejected() -> None:
    scorer = CompositeScorer()
    assert (
        scorer(
            record("Acme ships widgets", source="a"),
            record("Acme ships widgets", source="b", offset_days=60),
        )
        is None
    )


def test_a_missing_timestamp_is_rejected_when_time_matters() -> None:
    scorer = CompositeScorer()
    undated = Record("x", "b", "Acme ships widgets", None, None)
    assert scorer(record("Acme ships widgets", source="a"), undated) is None


def test_the_time_term_can_be_disabled_for_undated_records() -> None:
    scorer = CompositeScorer(max_time_delta_days=None)
    a = Record("x", "a", "Acme ships widgets", None, None)
    b = Record("y", "b", "Acme ships widgets", None, None)
    score = scorer(a, b)
    assert score is not None
    assert score.time_delta_days is None


def test_close_values_add_a_small_bonus() -> None:
    scorer = CompositeScorer()
    near = scorer(
        record("Acme ships widgets", source="a", value=0.50),
        record("Acme ships widgets", source="b", value=0.52),
    )
    far = scorer(
        record("Acme ships widgets", source="a", value=0.50),
        record("Acme ships widgets", source="b", value=0.90),
    )
    assert near is not None
    assert far is not None
    assert near.confidence > far.confidence


def test_confidence_is_clamped_to_the_unit_interval() -> None:
    # A large penalty must not produce a negative confidence.
    scorer = CompositeScorer(proper_noun_penalty=5.0)
    score = scorer(
        record("Will Acme beat Zenith?", source="a"),
        record("Will Acme beat Nadir?", source="b"),
    )
    assert score is not None
    assert score.confidence == 0.0


def test_a_naive_timestamp_is_assumed_utc() -> None:
    naive = Record("x", "a", "Acme ships", datetime(2026, 6, 1), None)
    assert naive.utc_timestamp() == datetime(2026, 6, 1, tzinfo=UTC)


def test_an_aware_timestamp_is_converted_not_reinterpreted() -> None:
    eastern = timezone(timedelta(hours=-5))
    aware = Record("x", "a", "Acme ships", datetime(2026, 6, 1, 12, tzinfo=eastern), None)
    assert aware.utc_timestamp() == datetime(2026, 6, 1, 17, tzinfo=UTC)


def test_a_missing_timestamp_stays_missing() -> None:
    assert Record("x", "a", "Acme ships", None, None).utc_timestamp() is None
