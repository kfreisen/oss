"""Deriving independent, reproducible random streams per batch.

This is the part that quietly goes wrong, so it is worth being explicit about.

The obvious approach — `default_rng(run_seed + batch_index)` — gives adjacent
seeds. NumPy's `SeedSequence` exists precisely because adjacent integer seeds do
**not** guarantee independent streams: they are hashed into state, and while
collisions are unlikely, the guarantee is not there and correlation between
"nearby" streams is a documented hazard of naive seeding. In a Monte Carlo study,
correlated streams show up as variance that is quietly too low — the estimate looks
more precise than it is, which is the worst way for a bug to present.

`SeedSequence.spawn` is the supported primitive for this. It derives child
sequences that are statistically independent by construction.

The second property is that the derivation depends only on `(run_seed,
batch_index)` — never on scheduling order, worker identity, or how many workers
happen to exist. So a run reproduces at any level of parallelism, including
serial, which is what makes a parallel bug bisectable.

The cost of that choice, stated plainly: **batch size is part of the contract.**
Streams are per batch, so changing the batch size changes which trials share a
stream and therefore changes the answer. Reproducing a run means matching the seed
and the batch size. Making batch size irrelevant would mean spawning a stream per
trial, which for most simulations costs more than the trial does.
"""

from __future__ import annotations

import numpy as np

__all__ = ["batch_bounds", "spawn_generators"]


def spawn_generators(run_seed: int, n_batches: int) -> list[np.random.Generator]:
    """One independent generator per batch, determined by `(run_seed, index)`.

    Args:
        run_seed: The run's seed. The same seed always yields the same generators.
        n_batches: How many to derive.

    Returns:
        Generators in batch order.

    Raises:
        ValueError: If `n_batches` is negative.
    """
    if n_batches < 0:
        msg = f"n_batches must be non-negative, got {n_batches}"
        raise ValueError(msg)
    root = np.random.SeedSequence(run_seed)
    return [np.random.default_rng(child) for child in root.spawn(n_batches)]


def batch_bounds(n_trials: int, batch_size: int) -> list[tuple[int, int]]:
    """Split `n_trials` into `(start, stop)` half-open ranges.

    The final batch is short rather than padded, so the trial count is exactly what
    was asked for. Padding and discarding would make the answer depend on the batch
    size, which is a tuning knob and must not change results.

    Raises:
        ValueError: If `batch_size` is not positive or `n_trials` is negative.
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if n_trials < 0:
        msg = f"n_trials must be non-negative, got {n_trials}"
        raise ValueError(msg)
    return [(start, min(start + batch_size, n_trials)) for start in range(0, n_trials, batch_size)]
