"""Turning scored pairs into clusters.

Three steps, in order: block, score within blocks, then union-find the surviving
pairs into connected components.

The union-find step deserves a warning it usually does not get. Clustering by
transitive closure means A~B and B~C puts A, B and C together **even if A and C
would never have matched each other**. That is often what you want — it is how
chains of near-duplicates get collapsed — and it is also how one bad pair merges
two large clusters. Nothing here prevents that; what this module provides instead
is the ability to say "never merge these two" and have it stick, because in
practice that is the escape valve real deduplication systems need and most
libraries omit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from typing import TYPE_CHECKING

from venuematch.records import block_by, iso_week_blocker
from venuematch.scoring import CompositeScorer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from venuematch.records import BlockKey, Record
    from venuematch.scoring import PairScore

__all__ = ["Overrides", "ResolutionResult", "ScoredPair", "resolve"]

DEFAULT_THRESHOLD = 0.65


@dataclass(frozen=True, slots=True)
class ScoredPair:
    """A pair that was compared, and how it scored."""

    left: str
    right: str
    score: PairScore

    def key(self) -> tuple[str, str]:
        """Order-independent identity of the pair."""
        return (self.left, self.right) if self.left <= self.right else (self.right, self.left)


@dataclass(frozen=True, slots=True)
class Overrides:
    """Human decisions, which beat any score.

    Real deduplication accumulates a list of pairs someone has ruled on, and a
    system with nowhere to put them re-litigates the same wrong merge every time a
    threshold moves.

    Attributes:
        forced: Pairs to treat as matching regardless of score, and regardless of
            whether blocking would ever have compared them. This is also the
            repair for a blocking key that separates a true match.
        blocked: Pairs never to merge, whatever they score.
    """

    forced: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    blocked: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def is_forced(self, a: str, b: str) -> bool:
        """Whether this pair is pinned together."""
        return self._key(a, b) in {self._key(x, y) for x, y in self.forced}

    def is_blocked(self, a: str, b: str) -> bool:
        """Whether this pair is pinned apart."""
        return self._key(a, b) in {self._key(x, y) for x, y in self.blocked}


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Clusters, and enough detail to argue with them.

    Attributes:
        cluster_of: Record id to cluster id. Records in no cluster are absent.
        clusters: Cluster id to sorted member ids.
        matched: Pairs that met the threshold.
        near_misses: Pairs that scored within `review_margin` below the threshold.
            The single most useful output for tuning: these are the decisions the
            threshold is actually making.
        comparisons: How many pairs were scored. Compare against
            `n * (n - 1) / 2` to see what blocking saved.
    """

    cluster_of: dict[str, str]
    clusters: dict[str, list[str]]
    matched: list[ScoredPair]
    near_misses: list[ScoredPair]
    comparisons: int


class _UnionFind:
    """Union-find with path compression, iterative.

    Iterative on purpose: the recursive form is prettier and blows the stack on a
    long chain, which is exactly what a large near-duplicate run produces.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        """Representative of x's set, compressing the path on the way."""
        parent = self._parent
        if x not in parent:
            parent[x] = x
            return x
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        """Merge the sets containing a and b."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b

    def groups(self) -> dict[str, list[str]]:
        """Every set, keyed by representative."""
        out: dict[str, list[str]] = {}
        for node in list(self._parent):
            out.setdefault(self.find(node), []).append(node)
        return out


def _cluster_id(members: Sequence[str]) -> str:
    """A stable id derived from membership.

    Deriving it from the members means the same cluster keeps its id across runs,
    which lets downstream storage be keyed on it. It also means adding one member
    changes the id — correct, since it is a different cluster, but worth knowing
    before treating these as durable primary keys.
    """
    digest = blake2b("|".join(members).encode(), digest_size=8).hexdigest()[:10]
    return f"vm:{digest}"


def resolve(
    records: Iterable[Record],
    *,
    scorer: Callable[[Record, Record], PairScore | None] | None = None,
    blocker: Callable[[Record], BlockKey] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    review_margin: float = 0.10,
    require_cross_source: bool = True,
    overrides: Overrides | None = None,
) -> ResolutionResult:
    """Group records that describe the same entity.

    Args:
        records: The records to resolve.
        scorer: Pair scorer. Defaults to
            [`CompositeScorer`][venuematch.scoring.CompositeScorer].
        blocker: Blocking function. Defaults to
            [`iso_week_blocker`][venuematch.records.iso_week_blocker] with no
            extra attributes.
        threshold: Minimum confidence to treat a pair as matching.
        review_margin: How far below the threshold to keep pairs as near misses.
        require_cross_source: Only compare records from different sources. True
            suits cross-venue matching; set it False to deduplicate within a
            source.
        overrides: Human decisions, applied after scoring and beating it.

    Returns:
        A [`ResolutionResult`][venuematch.resolve.ResolutionResult].

    Example:
        ```python
        from venuematch import Record, resolve

        result = resolve(records, threshold=0.7)
        for cluster_id, members in result.clusters.items():
            print(cluster_id, members)
        ```
    """
    scorer = scorer or CompositeScorer()
    blocker = blocker or iso_week_blocker()
    overrides = overrides or Overrides()

    by_id = {r.id: r for r in records}
    blocks = block_by(by_id.values(), blocker)

    matched: list[ScoredPair] = []
    near_misses: list[ScoredPair] = []
    comparisons = 0

    for members in blocks.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if require_cross_source and a.source == b.source:
                    continue
                comparisons += 1
                score = scorer(a, b)
                if score is None:
                    continue
                pair = ScoredPair(a.id, b.id, score)
                if score.confidence >= threshold:
                    matched.append(pair)
                elif score.confidence >= threshold - review_margin:
                    near_misses.append(pair)

    union = _UnionFind()
    for pair in matched:
        if overrides.is_blocked(pair.left, pair.right):
            continue
        union.union(pair.left, pair.right)
    # Forced pairs are applied last and are not subject to blocking or to whether
    # the blocker ever put them in the same bucket — that is the whole point.
    for forced_left, forced_right in overrides.forced:
        union.union(forced_left, forced_right)

    clusters: dict[str, list[str]] = {}
    cluster_of: dict[str, str] = {}
    for group in union.groups().values():
        if len(group) < 2:
            continue
        ordered = sorted(group)
        identifier = _cluster_id(ordered)
        clusters[identifier] = ordered
        for member in ordered:
            cluster_of[member] = identifier

    return ResolutionResult(
        cluster_of=cluster_of,
        clusters=clusters,
        matched=matched,
        near_misses=near_misses,
        comparisons=comparisons,
    )
