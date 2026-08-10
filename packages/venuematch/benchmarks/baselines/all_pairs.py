"""Naive all-pairs resolution — the implementation blocking replaces.

Compares every record against every other, with no blocking at all. It is
quadratic and it is also, by construction, the **recall ceiling**: any true match
the scorer can recognize, this finds.

That second property is why it matters more than the timing. Blocking buys speed by
declining to compare some pairs, and a blocking key that separates a true match
loses it silently — nothing in the output says a comparison was skipped. So the
benchmark reports recall against this, not just a speedup. A scheme that is 100x
faster and drops 4% of matches is a regression, and without this baseline there is
no way to know.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from venuematch.resolve import ScoredPair
from venuematch.scoring import CompositeScorer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from venuematch.records import Record
    from venuematch.scoring import PairScore


def resolve_all_pairs(
    records: Iterable[Record],
    *,
    scorer: Callable[[Record, Record], PairScore | None] | None = None,
    threshold: float = 0.65,
    require_cross_source: bool = True,
) -> tuple[list[ScoredPair], int]:
    """Score every pair. Returns the matches and the comparison count.

    Deliberately does not cluster: clustering is identical either way, so keeping
    it out means the comparison is between the two candidate-generation strategies
    and nothing else.
    """
    scorer = scorer or CompositeScorer()
    items = list(records)

    matched: list[ScoredPair] = []
    comparisons = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if require_cross_source and a.source == b.source:
                continue
            comparisons += 1
            score = scorer(a, b)
            if score is not None and score.confidence >= threshold:
                matched.append(ScoredPair(a.id, b.id, score))
    return matched, comparisons


def recall_against_all_pairs(
    blocked_pairs: Iterable[ScoredPair], all_pairs: Iterable[ScoredPair]
) -> tuple[float, set[tuple[str, str]]]:
    """What fraction of the achievable matches blocking found, and what it missed.

    Returns `(recall, missed_pairs)`. Recall is 1.0 when nothing was lost; the
    missed set is what to look at when it is not, because the pattern in those
    pairs is usually the blocking key's flaw stated plainly.
    """
    found = {p.key() for p in blocked_pairs}
    achievable = {p.key() for p in all_pairs}
    if not achievable:
        return 1.0, set()
    missed = achievable - found
    return len(achievable & found) / len(achievable), missed
