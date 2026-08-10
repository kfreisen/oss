"""Benchmark: blocked resolution against naive all-pairs.

Swept over corpus size, because that is where the two diverge.

A caveat the measurement forced, and which is easy to state wrongly: **blocking
does not change the complexity class here.** All-pairs is quadratic in the number
of records; blocking is quadratic *within each block*, so it is only linear overall
when block sizes stay bounded as the corpus grows. This corpus spreads events over
a fixed one-year range, so more events means bigger weekly blocks — and the
measured blocked timings grow superlinearly too (8000 events costs ~14x the 2000
case for 4x the records).

What blocking buys, then, is a large constant factor — roughly the number of
buckets — not a better asymptote. To get the better asymptote the key's cardinality
has to grow with the corpus, which for date blocking means a widening date range
rather than a denser one.

**Speed is only half the report.** Blocking buys its speed by not comparing some
pairs, so a run also records recall against the all-pairs baseline — the fraction
of achievable matches that blocking still found. A configuration that is 100x
faster and finds 96% of the matches is not obviously better than one that is 10x
faster and finds all of them, and a benchmark that printed only the speedup would
make that choice invisible.

Run with:

    make bench PKG=venuematch
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines.all_pairs import recall_against_all_pairs, resolve_all_pairs
from venuematch import Record, resolve

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def subject_name(index: int) -> str:
    """A unique, purely alphabetic, capitalized name.

    Alphabetic throughout on purpose: a trailing digit stops the proper-noun
    detector matching, which would silently disable the penalty this corpus is
    meant to exercise.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    return f"Corp{letters[index // 676 % 26].upper()}{letters[index // 26 % 26]}{letters[index % 26]}tech"


def make_corpus(n_events: int) -> list[Record]:
    """`n_events` true cross-source pairs spread over a year."""
    records: list[Record] = []
    for i in range(n_events):
        subject = subject_name(i)
        when = BASE + timedelta(days=i % 365)
        records.append(
            Record(f"a{i}", "alpha", f"Will {subject} report revenue above $5B?", when, 0.4)
        )
        records.append(Record(f"b{i}", "beta", f"{subject} revenue over $5B?", when, 0.41))
    return records


# All-pairs is quadratic, so its ceiling is low. 2000 events is 4000 records and
# ~4M cross-source comparisons, which is already tens of seconds.
BLOCKED_SIZES = [250, 1000, 2000, 8000]
ALL_PAIRS_SIZES = [250, 1000, 2000]


@pytest.mark.parametrize("n_events", BLOCKED_SIZES)
def test_blocked(benchmark, n_events: int) -> None:
    corpus = make_corpus(n_events)
    benchmark.extra_info["case"] = f"resolve/{n_events}events"
    benchmark.extra_info["impl"] = "blocked"
    benchmark.extra_info["params"] = {"events": n_events, "records": len(corpus)}
    result = benchmark(resolve, corpus)
    assert len(result.matched) > 0


@pytest.mark.parametrize("n_events", ALL_PAIRS_SIZES)
def test_all_pairs(benchmark, n_events: int) -> None:
    corpus = make_corpus(n_events)
    benchmark.extra_info["case"] = f"resolve/{n_events}events"
    benchmark.extra_info["impl"] = "all_pairs"
    benchmark.extra_info["params"] = {"events": n_events, "records": len(corpus)}
    matched, _ = benchmark.pedantic(resolve_all_pairs, args=(corpus,), rounds=3, iterations=1)
    assert len(matched) > 0


@pytest.mark.parametrize("n_events", ALL_PAIRS_SIZES)
def test_recall_is_reported_alongside_speed(n_events: int) -> None:
    """Not a timing test — the quality half of the same comparison.

    Printed rather than merely asserted, so a `make bench` run shows what blocking
    cost as well as what it saved.
    """
    corpus = make_corpus(n_events)
    blocked = resolve(corpus)
    exhaustive, exhaustive_comparisons = resolve_all_pairs(corpus)
    recall, missed = recall_against_all_pairs(blocked.matched, exhaustive)

    print(
        f"\n{n_events} events: blocked made {blocked.comparisons} comparisons vs "
        f"{exhaustive_comparisons} all-pairs "
        f"({blocked.comparisons / exhaustive_comparisons:.3%}), "
        f"recall {recall:.4f}, missed {len(missed)}"
    )
    assert recall == 1.0, f"blocking lost {len(missed)} pairs"
