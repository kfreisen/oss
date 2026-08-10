"""Shared fixtures."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from venuematch.records import Record

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

BASE = datetime(2026, 6, 1, tzinfo=UTC)


def pair_of_records(
    index: int, subject: str, *, day_offset: int = 0, drift_days: int = 0
) -> list[Record]:
    """The same event as described by two sources, worded differently."""
    when = BASE + timedelta(days=day_offset)
    return [
        Record(
            id=f"a{index}",
            source="alpha",
            text=f"Will {subject} report revenue above $5B in Q3?",
            timestamp=when,
            value=0.40 + (index % 5) / 100,
        ),
        Record(
            id=f"b{index}",
            source="beta",
            text=f"{subject} Q3 revenue over $5B?",
            timestamp=when + timedelta(days=drift_days),
            value=0.41 + (index % 5) / 100,
        ),
    ]


def subject_name(index: int) -> str:
    """A unique, purely alphabetic, capitalized company name.

    Purely alphabetic matters. The proper-noun detector looks for a capitalized
    run of letters ending at a word boundary, so a name like `Acorp7` is not a
    proper noun — the trailing digit swallows the boundary. A fixture built that
    way silently disables the very penalty these tests exist to exercise, and
    every event in a week then matches every other.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    return f"Corp{letters[index // 26].upper()}{letters[index % 26]}tech"


def make_corpus(n_events: int = 40, *, drift_days: int = 1) -> list[Record]:
    """`n_events` true pairs, spread across weeks, with distinct subjects.

    Subjects differ only by name, which is the case the proper-noun penalty exists
    for: without it every event in a week matches every other.
    """
    subjects = [subject_name(i) for i in range(n_events)]
    records: list[Record] = []
    for i, subject in enumerate(subjects):
        records.extend(pair_of_records(i, subject, day_offset=i % 30, drift_days=drift_days))
    return records


@pytest.fixture
def corpus() -> list[Record]:
    """A modest corpus with a known ground truth: `a{i}` matches `b{i}`.

    Timestamps agree exactly between the two sources. That is the case where week
    blocking is lossless, and it is the right default for tests about everything
    else. Drift is exercised deliberately, in the tests that are about drift.
    """
    return make_corpus(drift_days=0)


def truth_pairs(records: list[Record]) -> set[tuple[str, str]]:
    """The known-correct pairs for a corpus built by `make_corpus`."""
    ids = {r.id for r in records}
    return {(f"a{i}", f"b{i}") for i in range(len(ids) // 2)}
