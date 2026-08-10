"""The two implementations mcharness replaces, kept working and kept tested.

Two baselines, not one, because the speedup has two independent sources and
reporting them together would teach the wrong lesson.

`run_serial_uncached` is where most studies start: a single process, a loop over
trials, and a DataFrame lookup for the entity's parameters on every iteration.

`run_serial_cached` fixes only the lookup. Comparing against it isolates how much
came from hoisting the lookup out of the inner loop, which in the original was more
than parallelism contributed — and would have been invisible if only the combined
number were published.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcharness.cache import EntityCache


class FrameLookupSimulation:
    """A trial that resolves its entity through a DataFrame `.loc` on every call.

    This is where studies actually start, and it is the thing the cache replaces.
    It is slow for a reason that has nothing to do with the simulation: every
    `.loc` walks a pandas index and constructs a Series.
    """

    def __init__(self, frame: Any, entity_ids: Sequence[str]) -> None:
        self.frame = frame
        self.entity_ids = list(entity_ids)

    def run_trial(self, rng: np.random.Generator, index: int) -> float:
        entity = self.entity_ids[index % len(self.entity_ids)]
        row = self.frame.loc[entity]
        return float(rng.normal(row["mean"], row["stddev"]))


class LookupSimulation:
    """A trial that resolves its entity through a dict-of-dicts on every call.

    The middle ground, and a fair opponent: a dict lookup is already far faster
    than pandas indexing, so measuring against this rather than against the frame
    understates what the cache is worth in the wild — deliberately.
    """

    def __init__(self, table: dict[str, dict[str, float]], entity_ids: Sequence[str]) -> None:
        self.table = table
        self.entity_ids = list(entity_ids)

    def run_trial(self, rng: np.random.Generator, index: int) -> float:
        entity = self.entity_ids[index % len(self.entity_ids)]
        row = self.table[entity]
        return float(rng.normal(row["mean"], row["stddev"]))


class CachedSimulation:
    """The same trial, resolving through the cache's scalar-access lists.

    Note this uses `field_list`, not `field`. Scalar indexing into a NumPy array
    is slower than into a dict — see `EntityCache.field_list` — so the array form
    would make this benchmark a demonstration of the wrong thing.
    """

    def __init__(self, cache: EntityCache, entity_positions: Sequence[int]) -> None:
        self.means = cache.field_list("mean")
        self.stddevs = cache.field_list("stddev")
        self.entity_positions = list(entity_positions)

    def run_trial(self, rng: np.random.Generator, index: int) -> float:
        position = self.entity_positions[index % len(self.entity_positions)]
        return float(rng.normal(self.means[position], self.stddevs[position]))


def run_serial_loop(simulation: Any, n_trials: int, rng: np.random.Generator) -> list[float]:
    """One process, one loop, one generator. No batching, no dispatch."""
    return [simulation.run_trial(rng, index) for index in range(n_trials)]


class CostlySimulation:
    """A trial that does real work, to show where parallelism starts paying.

    Parallelism is not free: every batch is serialized, shipped, scheduled and
    shipped back. That overhead is fixed per batch, so whether Ray wins depends
    entirely on how much work a trial does — not on how many trials there are.

    This trial does a small amount of numerical work per call, which is the regime
    a real simulation lives in and the regime the cheap benchmarks above do not
    represent.
    """

    def __init__(self, inner_steps: int = 400) -> None:
        self.inner_steps = inner_steps

    def run_trial(self, rng: np.random.Generator, index: int) -> float:  # noqa: ARG002
        total = 0.0
        for _ in range(self.inner_steps):
            total += float(rng.random()) ** 0.5
        return total
