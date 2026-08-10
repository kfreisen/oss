"""Hoisting per-entity lookups out of the inner loop.

The least glamorous part of this package and frequently the largest speedup.

A Monte Carlo study over entities — players, customers, assets — usually reaches
for that entity's parameters on every trial. If the parameters live in a DataFrame
and the reach is a `.loc`, the study spends most of its time in pandas indexing
rather than in the simulation, and parallelizing it buys a fraction of what it
should because every worker pays the same overhead.

`EntityCache` does the lookup once, into flat columns indexed by a dense integer,
and ships to workers as arrays rather than as a DataFrame — which is also far
cheaper to serialize.

**One measured correction to the obvious version of this advice.** "Hoist the
lookups into NumPy arrays" is only good advice when the consumer is vectorized.
Indexing an array one element at a time is *slower* than indexing a dict, because
each access boxes a NumPy scalar. So the cache stores arrays and hands out plain
Python lists for scalar access, which measured fastest of the three. See
[`field_list`][mcharness.cache.EntityCache.field_list].

The benchmarks measure the cache separately from parallelism, because attributing
a combined speedup to "we used Ray" when most of it came from fixing the lookup
would be the wrong lesson — and, as it turns out, so would the reverse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

__all__ = ["EntityCache"]


class EntityCache:
    """Per-entity parameters as flat arrays, indexed by dense integer id.

    Example:
        ```python
        cache = EntityCache.from_records(records, key="player_id", fields=("mean", "stddev"))
        index = cache.index_of("abc123")
        mean = cache.field("mean")[index]
        ```
    """

    __slots__ = ("_columns", "_index", "_keys", "_lists")

    def __init__(self, keys: Sequence[Hashable], columns: Mapping[str, np.ndarray]) -> None:
        """Build a cache from parallel keys and columns."""
        self._keys = list(keys)
        self._index = {key: position for position, key in enumerate(self._keys)}
        if len(self._index) != len(self._keys):
            msg = "duplicate entity keys; each entity must appear once"
            raise ValueError(msg)
        for name, column in columns.items():
            if len(column) != len(self._keys):
                msg = (
                    f"column {name!r} has {len(column)} entries but there are "
                    f"{len(self._keys)} entities"
                )
                raise ValueError(msg)
        self._columns = dict(columns)
        self._lists: dict[str, list[float]] = {}

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        key: str,
        fields: Sequence[str],
        dtype: Any = np.float64,
    ) -> EntityCache:
        """Build from a sequence of per-entity mappings.

        Raises:
            KeyError: If a record is missing the key or a named field.
        """
        keys: list[Hashable] = []
        for position, record in enumerate(records):
            if key not in record:
                msg = f"record {position} has no {key!r}"
                raise KeyError(msg)
            keys.append(record[key])

        columns: dict[str, np.ndarray] = {}
        for field in fields:
            for position, record in enumerate(records):
                if field not in record:
                    msg = f"record {position} has no {field!r}"
                    raise KeyError(msg)
            columns[field] = np.asarray([r[field] for r in records], dtype=dtype)
        return cls(keys, columns)

    def __len__(self) -> int:
        """Number of entities."""
        return len(self._keys)

    def index_of(self, key: Hashable) -> int:
        """Dense integer position of an entity.

        Call this once, outside the inner loop. Calling it per trial reintroduces
        a dict lookup where an array index was the point.

        Raises:
            KeyError: If the entity is unknown.
        """
        try:
            return self._index[key]
        except KeyError:
            msg = f"unknown entity {key!r}"
            raise KeyError(msg) from None

    def field(self, name: str) -> np.ndarray:
        """The flat array for a field.

        Use this for **vectorized** access — `means[positions]` for a whole
        batch. For scalar access inside a Python loop, use
        [`field_list`][mcharness.cache.EntityCache.field_list] instead; see the
        note there.

        Raises:
            KeyError: If the field was not cached.
        """
        return self._column(name)

    def field_list(self, name: str) -> list[float]:
        """The field as a Python list, for scalar access in a loop.

        This exists because of a measurement that contradicted the obvious
        assumption. Indexing a NumPy array one element at a time is **slower**
        than indexing a dict, because each access boxes a NumPy scalar; a plain
        Python list is faster than both:

            dict of dicts    0.87 ms
            numpy array      0.94 ms
            python list      0.77 ms

        (2000 scalar lookups plus a draw each, on the reference machine.)

        So "hoist the lookups into arrays" is only good advice when the consumer
        is vectorized. Inside a per-trial Python loop, arrays lose. The cache
        stores arrays because that is what ships cheaply to workers and what a
        vectorized simulation wants, and hands out lists for the scalar case.

        The result is cached after the first call, so this is safe to hoist out
        of a loop and pointless to call inside one.

        Raises:
            KeyError: If the field was not cached.
        """
        if name not in self._lists:
            self._lists[name] = self._column(name).tolist()
        return self._lists[name]

    def _column(self, name: str) -> np.ndarray:
        try:
            return self._columns[name]
        except KeyError:
            available = ", ".join(sorted(self._columns))
            msg = f"field {name!r} was not cached; available: {available}"
            raise KeyError(msg) from None

    @property
    def keys(self) -> tuple[Hashable, ...]:
        """Entity keys, in index order."""
        return tuple(self._keys)

    @property
    def fields(self) -> tuple[str, ...]:
        """Cached field names."""
        return tuple(self._columns)
