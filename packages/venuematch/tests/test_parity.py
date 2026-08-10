"""Blocking must not lose matches that all-pairs would have found.

This is the load-bearing test. Blocking buys speed by declining to compare some
pairs, and a key that separates a true match loses it **silently** — nothing in the
output records that a comparison never happened.

So parity here is not "identical output". It is recall against the all-pairs
baseline, measured rather than assumed, with the losing cases named.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from baselines.all_pairs import recall_against_all_pairs, resolve_all_pairs
from conftest import BASE, make_corpus, truth_pairs
from venuematch import Record, resolve
from venuematch.records import iso_week_blocker


def test_blocking_loses_nothing_on_a_well_dated_corpus(corpus: list[Record]) -> None:
    blocked = resolve(corpus)
    exhaustive, _ = resolve_all_pairs(corpus)
    recall, missed = recall_against_all_pairs(blocked.matched, exhaustive)
    assert recall == 1.0, f"blocking lost {len(missed)} pairs: {sorted(missed)[:5]}"


def test_blocking_actually_saves_work(corpus: list[Record]) -> None:
    blocked = resolve(corpus)
    _, exhaustive_comparisons = resolve_all_pairs(corpus)
    # If blocking were not reducing the comparison count there would be no reason
    # for any of this machinery to exist.
    assert blocked.comparisons < exhaustive_comparisons / 2


def test_both_find_the_known_truth(corpus: list[Record]) -> None:
    blocked = resolve(corpus)
    exhaustive, _ = resolve_all_pairs(corpus)
    truth = truth_pairs(corpus)

    assert {p.key() for p in blocked.matched} == truth
    assert {p.key() for p in exhaustive} == truth


def test_the_week_boundary_is_a_real_recall_loss() -> None:
    """The documented failure mode, demonstrated rather than asserted.

    Two descriptions of one event landing either side of an ISO week boundary are
    never compared. This is the honest cost of week blocking, and it is why the
    package ships forced overrides and reports recall.
    """
    # A Sunday and the following Monday are in different ISO weeks.
    sunday = BASE + timedelta(days=(6 - BASE.weekday()) % 7)
    monday = sunday + timedelta(days=1)
    straddling = [
        Record("a", "alpha", "Will Zetacorp report revenue above $5B?", sunday, 0.5),
        Record("b", "beta", "Zetacorp revenue over $5B?", monday, 0.5),
    ]

    blocked = resolve(straddling)
    exhaustive, _ = resolve_all_pairs(straddling)

    assert len(exhaustive) == 1, "all-pairs should find it"
    assert blocked.matched == [], "week blocking should miss it"
    recall, missed = recall_against_all_pairs(blocked.matched, exhaustive)
    assert recall == 0.0
    assert missed == {("a", "b")}


def test_a_forced_override_repairs_a_blocking_miss() -> None:
    """The escape valve for exactly the case above."""
    from venuematch.resolve import Overrides

    sunday = BASE + timedelta(days=(6 - BASE.weekday()) % 7)
    monday = sunday + timedelta(days=1)
    straddling = [
        Record("a", "alpha", "Will Zetacorp report revenue above $5B?", sunday, 0.5),
        Record("b", "beta", "Zetacorp revenue over $5B?", monday, 0.5),
    ]

    result = resolve(straddling, overrides=Overrides(forced=frozenset({("a", "b")})))
    assert result.cluster_of["a"] == result.cluster_of["b"]


def test_a_coarser_blocker_recovers_the_boundary_case() -> None:
    """Recall is a property of the blocking key, and the key is the caller's.

    Swapping the week blocker for one that puts everything in one bucket restores
    all-pairs recall at all-pairs cost. That trade is the whole design space, and
    the point of making the blocker a parameter.
    """
    sunday = BASE + timedelta(days=(6 - BASE.weekday()) % 7)
    monday = sunday + timedelta(days=1)
    straddling = [
        Record("a", "alpha", "Will Zetacorp report revenue above $5B?", sunday, 0.5),
        Record("b", "beta", "Zetacorp revenue over $5B?", monday, 0.5),
    ]
    result = resolve(straddling, blocker=lambda _r: ())
    assert len(result.matched) == 1


# Measured, not guessed. Week blocking loses a true pair whenever the two sources
# put the same event on opposite sides of a week boundary, and the chance of that
# rises roughly linearly with how far their timestamps disagree. These floors sit
# just under the measured values so the test pins the shape without being brittle.
DRIFT_RECALL_FLOORS = [(0, 1.00), (1, 0.85), (2, 0.72), (3, 0.60), (5, 0.30)]


@pytest.mark.parametrize(("drift_days", "floor"), DRIFT_RECALL_FLOORS)
def test_recall_degrades_predictably_as_timestamps_drift(drift_days: int, floor: float) -> None:
    """Week blocking is lossless only when sources agree on the date.

    This is the honest cost of the default blocker, and publishing the curve beats
    publishing a speedup and hoping nobody asks. If your sources disagree about
    timing by more than a day or two, use a coarser blocker and pay for it, or
    supply forced overrides for the pairs you know about.
    """
    corpus = make_corpus(n_events=25, drift_days=drift_days)
    blocked = resolve(corpus)
    exhaustive, _ = resolve_all_pairs(corpus)
    recall, missed = recall_against_all_pairs(blocked.matched, exhaustive)
    assert recall >= floor, f"recall {recall:.2%}, missed {sorted(missed)[:5]}"


def test_drift_only_ever_costs_recall_never_precision() -> None:
    """Blocking can only *skip* comparisons, never invent matches.

    So every pair blocking finds must also be found by all-pairs, at any drift.
    A violation would mean the two scorers disagree, which is a different and much
    worse bug than a recall loss.
    """
    for drift in (0, 1, 3, 5):
        corpus = make_corpus(n_events=25, drift_days=drift)
        blocked = {p.key() for p in resolve(corpus).matched}
        exhaustive = {p.key() for p in resolve_all_pairs(corpus)[0]}
        assert blocked <= exhaustive, f"drift={drift} produced matches all-pairs did not"


def test_an_extra_blocking_attribute_still_finds_the_truth() -> None:
    corpus = make_corpus(n_events=20, drift_days=0)
    tagged = [
        Record(r.id, r.source, r.text, r.timestamp, r.value, {"category": "earnings"})
        for r in corpus
    ]
    result = resolve(tagged, blocker=iso_week_blocker("category"))
    assert {p.key() for p in result.matched} == truth_pairs(tagged)
