"""What a record is, and how records are grouped before comparison.

The original of this code matched prediction markets across venues. Nothing about
that is essential: the problem is "two sources describe the same thing in their own
words, with no shared identifier", and the shape of the solution is the same for
products, companies, papers, or events.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = ["BlockKey", "Record", "block_by", "iso_week_blocker"]

BlockKey = tuple[Hashable, ...]
"""A blocking key. Records with different keys are never compared."""


@dataclass(frozen=True, slots=True)
class Record:
    """One entity as described by one source.

    Attributes:
        id: Unique within the whole input. Cluster membership is reported by id.
        source: Which system this came from. Used by `require_cross_source`, and
            by nothing else — a matcher that only ever compares across sources
            cannot deduplicate within one.
        text: The free-text description that carries the meaning. Matching is
            driven almost entirely by this.
        timestamp: When the described thing happens or happened. Used both for
            blocking and as a similarity signal; `None` disables both for this
            record.
        value: An optional numeric attribute — a price, a score, a quantity. Weak
            corroborating evidence when two records agree closely on it.
        attributes: Anything else, carried through untouched for the caller's
            benefit.
    """

    id: str
    source: str
    text: str
    timestamp: datetime | None = None
    value: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict, compare=False)

    def utc_timestamp(self) -> datetime | None:
        """Timestamp as an aware UTC datetime, or None.

        Naive datetimes are assumed UTC rather than rejected. Sources are
        inconsistent about this, and refusing them would push the same guess onto
        every caller — but comparing a naive to an aware datetime raises, so the
        guess has to happen somewhere.
        """
        if self.timestamp is None:
            return None
        if self.timestamp.tzinfo is None:
            return self.timestamp.replace(tzinfo=UTC)
        return self.timestamp.astimezone(UTC)


def iso_week_blocker(
    *attribute_keys: str,
) -> Callable[[Record], BlockKey]:
    """Block by ISO year-week of the timestamp, plus any attributes named.

    The default blocking strategy, and a good one when records carry a date: two
    descriptions of the same event rarely disagree about which week it falls in,
    and the key caps bucket size hard.

    It has a real failure mode, and it is worth naming: an event near a week
    boundary can land in different weeks in two sources, and this blocker will
    never compare them. That is a recall loss, it is measurable, and
    `benchmarks/` measures it against the all-pairs baseline rather than assuming
    it away.

    Args:
        *attribute_keys: Record attribute names to include in the key. Adding a
            category makes buckets smaller and the recall risk larger.

    Returns:
        A blocking function suitable for [`resolve`][venuematch.resolve.resolve].
    """

    def blocker(record: Record) -> BlockKey:
        extras = tuple(record.attributes.get(k) for k in attribute_keys)
        moment = record.utc_timestamp()
        if moment is None:
            # Undated records form their own bucket rather than being dropped:
            # they can still match each other.
            return (*extras, "undated")
        iso_year, iso_week, _ = moment.isocalendar()
        return (*extras, f"{iso_year:04d}W{iso_week:02d}")

    return blocker


def block_by(
    records: Iterable[Record], blocker: Callable[[Record], BlockKey]
) -> dict[BlockKey, list[Record]]:
    """Group records into blocks.

    Returns:
        Blocks in first-seen key order, each preserving input order — so a run is
        reproducible without sorting anything.
    """
    blocks: dict[BlockKey, list[Record]] = {}
    for record in records:
        blocks.setdefault(blocker(record), []).append(record)
    return blocks


def block_size_report(blocks: dict[BlockKey, Sequence[Record]]) -> str:
    """Summarize the blocking, because a bad key is invisible otherwise.

    Two failure modes to look for: one enormous block, which means the key is not
    discriminating and the comparison count is back to quadratic; and blocks of
    size one everywhere, which means the key is too specific and nothing will ever
    be compared.
    """
    sizes = sorted((len(v) for v in blocks.values()), reverse=True)
    if not sizes:
        return "no blocks"
    comparisons = sum(n * (n - 1) // 2 for n in sizes)
    total = sum(sizes)
    all_pairs = total * (total - 1) // 2
    singletons = sum(1 for n in sizes if n == 1)
    return (
        f"{len(sizes)} blocks over {total} records; "
        f"largest {sizes[0]}, median {sizes[len(sizes) // 2]}, "
        f"{singletons} singleton(s); "
        f"{comparisons} comparisons vs {all_pairs} all-pairs "
        f"({comparisons / all_pairs:.4%})"
        if all_pairs
        else f"{len(sizes)} blocks over {total} records"
    )
