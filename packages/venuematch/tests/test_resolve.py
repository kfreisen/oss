"""Blocking, clustering, and overrides."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from venuematch import Record, resolve
from venuematch.records import block_by, block_size_report, iso_week_blocker
from venuematch.resolve import Overrides

WHEN = datetime(2026, 6, 1, tzinfo=UTC)


def rec(rid: str, source: str, text: str, *, days: int = 0, attrs: dict | None = None) -> Record:
    return Record(rid, source, text, WHEN + timedelta(days=days), 0.5, attrs or {})


def test_blocking_groups_by_week() -> None:
    records = [
        rec("a", "x", "one", days=0),
        rec("b", "y", "two", days=1),
        rec("c", "z", "three", days=30),
    ]
    blocks = block_by(records, iso_week_blocker())
    assert len(blocks) == 2


def test_undated_records_form_their_own_block() -> None:
    # Dropping them would silently exclude records that can still match each other.
    records = [Record("a", "x", "text", None), Record("b", "y", "text", None)]
    blocks = block_by(records, iso_week_blocker())
    assert list(blocks) == [("undated",)]


def test_an_attribute_blocker_splits_further() -> None:
    records = [
        rec("a", "x", "one", attrs={"category": "sports"}),
        rec("b", "y", "one", attrs={"category": "politics"}),
    ]
    assert len(block_by(records, iso_week_blocker("category"))) == 2


def test_block_size_report_describes_the_split() -> None:
    records = [rec(str(i), "x", "text", days=i) for i in range(10)]
    report = block_size_report(block_by(records, iso_week_blocker()))
    assert "blocks over 10 records" in report
    assert "all-pairs" in report


def test_block_size_report_handles_no_blocks() -> None:
    assert block_size_report({}) == "no blocks"


def test_same_source_pairs_are_skipped_by_default() -> None:
    records = [rec("a", "same", "Acme ships widgets"), rec("b", "same", "Acme ships widgets")]
    assert resolve(records).comparisons == 0


def test_within_source_deduplication_can_be_enabled() -> None:
    records = [rec("a", "same", "Acme ships widgets"), rec("b", "same", "Acme ships widgets")]
    result = resolve(records, require_cross_source=False)
    assert len(result.matched) == 1


def test_clusters_are_transitive() -> None:
    """A~B and B~C puts all three together, even if A and C never matched.

    Documented, and worth pinning: it is how chains collapse, and it is also how
    one bad pair merges two real clusters.
    """
    records = [
        rec("a", "x", "Acme Corporation ships widgets"),
        rec("b", "y", "Acme Corporation ships widgets today"),
        rec("c", "z", "Acme Corporation ships widgets today and tomorrow"),
    ]
    result = resolve(records)
    assert len({result.cluster_of[i] for i in ("a", "b", "c")}) == 1


def test_singletons_are_not_clusters() -> None:
    result = resolve([rec("a", "x", "Acme ships"), rec("b", "y", "Zenith buys land")])
    assert result.clusters == {}
    assert result.cluster_of == {}


def test_cluster_ids_are_stable_across_runs() -> None:
    records = [rec("a", "x", "Acme ships widgets"), rec("b", "y", "Acme ships widgets")]
    assert resolve(records).clusters.keys() == resolve(records).clusters.keys()


def test_cluster_ids_do_not_depend_on_input_order() -> None:
    records = [rec("a", "x", "Acme ships widgets"), rec("b", "y", "Acme ships widgets")]
    assert list(resolve(records).clusters) == list(resolve(list(reversed(records))).clusters)


def test_a_blocked_override_prevents_a_merge() -> None:
    records = [rec("a", "x", "Acme ships widgets"), rec("b", "y", "Acme ships widgets")]
    assert len(resolve(records).clusters) == 1
    blocked = Overrides(blocked=frozenset({("a", "b")}))
    assert resolve(records, overrides=blocked).clusters == {}


def test_a_blocked_override_works_in_either_order() -> None:
    records = [rec("a", "x", "Acme ships widgets"), rec("b", "y", "Acme ships widgets")]
    blocked = Overrides(blocked=frozenset({("b", "a")}))
    assert resolve(records, overrides=blocked).clusters == {}


def test_a_forced_override_beats_a_non_match() -> None:
    records = [rec("a", "x", "Acme ships widgets"), rec("b", "y", "Zenith buys farmland")]
    forced = Overrides(forced=frozenset({("a", "b")}))
    result = resolve(records, overrides=forced)
    assert result.cluster_of["a"] == result.cluster_of["b"]


def test_near_misses_are_reported_for_tuning() -> None:
    # The pairs a threshold change would flip — the most useful thing to look at
    # when deciding whether the threshold is right.
    records = [
        rec("a", "x", "Will the Iowa primary be won by Smith?"),
        rec("b", "y", "Will the Hampshire primary be won by Smith?"),
    ]
    result = resolve(records, review_margin=0.5)
    assert result.near_misses
    assert result.matched == []


def test_a_higher_threshold_matches_less() -> None:
    records = [
        rec("a", "x", "Acme Corporation ships widgets"),
        rec("b", "y", "Acme Corporation ships some widgets"),
    ]
    assert len(resolve(records, threshold=0.6).matched) == 1
    assert len(resolve(records, threshold=0.99).matched) == 0


def test_an_empty_input_resolves_to_nothing() -> None:
    result = resolve([])
    assert result.clusters == {}
    assert result.comparisons == 0


def test_a_custom_scorer_is_honoured() -> None:
    from venuematch.scoring import PairScore

    def always_match(_a: Record, _b: Record) -> PairScore:
        return PairScore(1.0, 100.0, 1.0, 0, 0.0)

    records = [rec("a", "x", "totally"), rec("b", "y", "different")]
    assert len(resolve(records, scorer=always_match).matched) == 1


def test_a_scorer_returning_none_is_treated_as_no_match() -> None:
    records = [rec("a", "x", "totally"), rec("b", "y", "different")]
    assert resolve(records, scorer=lambda _a, _b: None).matched == []
