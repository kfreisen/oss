"""Benchmark: where the speedup actually comes from.

The interesting question is not "is Ray faster than a loop" — it is how much of
the gain came from parallelism and how much from fixing the inner loop. Reporting
a single combined number teaches the wrong lesson, because in the original of this
code the lookup fix was worth more than the parallelism.

So four implementations of the same study:

- `serial_dataframe` — one loop, DataFrame `.loc` per trial. Where studies start.
- `serial_uncached` — one loop, dict lookup per trial. A fair middle ground.
- `serial_cached` — one loop, cache lookup. Isolates the cache.
- `ray_uncached` — parallel, dict lookup per trial. Isolates parallelism.
- `ray_cached` — both.

Measuring against the dict rather than only against the DataFrame is deliberate: it
is the harder comparison, and it is what turned up the fact that NumPy scalar
indexing is slower than a dict — which changed the cache's implementation.

## What the numbers actually said

Measured at 100,000 trials, medians on the reference machine:

    serial_dataframe   1516 ms
    serial_uncached      49 ms
    serial_cached        48 ms
    ray_uncached        187 ms
    ray_cached          117 ms

Two results, and neither is the one this package was expected to demonstrate.

**Almost all of the win is DataFrame to dict — about 30x.** The array cache adds a
few percent on top. If a study is slow and it does a `.loc` per trial, that is the
whole problem, and no amount of parallelism addresses it.

**Ray is slower than a serial loop here, by 2-4x.** These trials are one normal
draw each, so batching, serializing, scheduling and returning costs far more than
the work. That is not a criticism of Ray; it is what fixed per-batch overhead means.

The `costly/` case exists to find the crossover, and it is further out than
intuition suggests: at 400 inner operations per trial, Ray reaches roughly parity
(521 ms against 545 ms) rather than winning outright. Parallelism pays when a trial
is expensive, and "expensive" here means substantially more than a few hundred
arithmetic operations.

The reason to publish this rather than a flattering subset is that it is the
actual advice: fix the lookup first, measure, and reach for a cluster only once a
trial is heavy enough to earn one.

Run with:

    make bench PKG=mcharness
"""

from __future__ import annotations

import numpy as np
import pytest
from mcharness import EntityCache, run

from benchmarks.baselines.serial_uncached import (
    CachedSimulation,
    CostlySimulation,
    FrameLookupSimulation,
    LookupSimulation,
    run_serial_loop,
)

N_ENTITIES = 500
TRIAL_COUNTS = [20_000, 100_000]


def make_world(n_entities: int = N_ENTITIES):
    """Entity parameters, plus both lookup shapes over them. Deterministic."""
    records = [
        {"player_id": f"p{i:05d}", "mean": 10.0 + (i % 17), "stddev": 1.0 + (i % 5)}
        for i in range(n_entities)
    ]
    table = {r["player_id"]: r for r in records}
    entity_ids = [f"p{i % n_entities:05d}" for i in range(n_entities * 4)]

    cache = EntityCache.from_records(records, key="player_id", fields=("mean", "stddev"))
    positions = [cache.index_of(e) for e in entity_ids]
    return table, entity_ids, cache, positions


def make_frame(records):
    """The same parameters as a pandas DataFrame indexed by entity."""
    import pandas as pd

    return pd.DataFrame(records).set_index("player_id")


@pytest.mark.parametrize("n_trials", TRIAL_COUNTS)
def test_serial_frame(benchmark, n_trials: int) -> None:
    """Where studies start: a DataFrame .loc per trial."""
    records = [
        {"player_id": f"p{i:05d}", "mean": 10.0 + (i % 17), "stddev": 1.0 + (i % 5)}
        for i in range(N_ENTITIES)
    ]
    _, entity_ids, _, _ = make_world()
    simulation = FrameLookupSimulation(make_frame(records), entity_ids)
    benchmark.extra_info["case"] = f"study/{n_trials}trials"
    benchmark.extra_info["impl"] = "serial_dataframe"
    benchmark.extra_info["params"] = {"trials": n_trials, "entities": N_ENTITIES}
    result = benchmark.pedantic(
        run_serial_loop,
        args=(simulation, n_trials, np.random.default_rng(0)),
        rounds=3,
        iterations=1,
    )
    assert len(result) == n_trials


@pytest.mark.parametrize("n_trials", TRIAL_COUNTS)
def test_serial_uncached(benchmark, n_trials: int) -> None:
    table, entity_ids, _, _ = make_world()
    simulation = LookupSimulation(table, entity_ids)
    benchmark.extra_info["case"] = f"study/{n_trials}trials"
    benchmark.extra_info["impl"] = "serial_uncached"
    benchmark.extra_info["params"] = {"trials": n_trials, "entities": N_ENTITIES}
    result = benchmark.pedantic(
        run_serial_loop,
        args=(simulation, n_trials, np.random.default_rng(0)),
        rounds=3,
        iterations=1,
    )
    assert len(result) == n_trials


@pytest.mark.parametrize("n_trials", TRIAL_COUNTS)
def test_serial_cached(benchmark, n_trials: int) -> None:
    _, _, cache, positions = make_world()
    simulation = CachedSimulation(cache, positions)
    benchmark.extra_info["case"] = f"study/{n_trials}trials"
    benchmark.extra_info["impl"] = "serial_cached"
    benchmark.extra_info["params"] = {"trials": n_trials, "entities": N_ENTITIES}
    result = benchmark.pedantic(
        run_serial_loop,
        args=(simulation, n_trials, np.random.default_rng(0)),
        rounds=3,
        iterations=1,
    )
    assert len(result) == n_trials


@pytest.mark.parametrize("n_trials", TRIAL_COUNTS)
def test_ray_uncached(benchmark, ray_bench, n_trials: int) -> None:  # noqa: ARG001
    table, entity_ids, _, _ = make_world()
    simulation = LookupSimulation(table, entity_ids)
    benchmark.extra_info["case"] = f"study/{n_trials}trials"
    benchmark.extra_info["impl"] = "ray_uncached"
    benchmark.extra_info["params"] = {"trials": n_trials, "entities": N_ENTITIES}
    result = benchmark.pedantic(
        run,
        args=(simulation, n_trials),
        kwargs={"batch_size": 5000, "backend": "ray"},
        rounds=3,
        iterations=1,
    )
    assert result.n_trials == n_trials


@pytest.mark.parametrize("n_trials", TRIAL_COUNTS)
def test_ray_cached(benchmark, ray_bench, n_trials: int) -> None:  # noqa: ARG001
    _, _, cache, positions = make_world()
    simulation = CachedSimulation(cache, positions)
    benchmark.extra_info["case"] = f"study/{n_trials}trials"
    benchmark.extra_info["impl"] = "ray_cached"
    benchmark.extra_info["params"] = {"trials": n_trials, "entities": N_ENTITIES}
    result = benchmark.pedantic(
        run,
        args=(simulation, n_trials),
        kwargs={"batch_size": 5000, "backend": "ray"},
        rounds=3,
        iterations=1,
    )
    assert result.n_trials == n_trials


# Where parallelism starts paying. The cheap cases above are dominated by dispatch
# overhead; this one does real work per trial, which is the regime a real study
# lives in.
COSTLY_TRIALS = 4000


def test_costly_serial(benchmark) -> None:
    benchmark.extra_info["case"] = f"costly/{COSTLY_TRIALS}trials"
    benchmark.extra_info["impl"] = "serial"
    benchmark.extra_info["params"] = {"trials": COSTLY_TRIALS, "inner_steps": 400}
    result = benchmark.pedantic(
        run,
        args=(CostlySimulation(), COSTLY_TRIALS),
        kwargs={"batch_size": 200, "backend": "serial"},
        rounds=3,
        iterations=1,
    )
    assert result.n_trials == COSTLY_TRIALS


def test_costly_ray(benchmark, ray_bench) -> None:  # noqa: ARG001
    benchmark.extra_info["case"] = f"costly/{COSTLY_TRIALS}trials"
    benchmark.extra_info["impl"] = "ray"
    benchmark.extra_info["params"] = {"trials": COSTLY_TRIALS, "inner_steps": 400}
    result = benchmark.pedantic(
        run,
        args=(CostlySimulation(), COSTLY_TRIALS),
        kwargs={"batch_size": 200, "backend": "ray"},
        rounds=3,
        iterations=1,
    )
    assert result.n_trials == COSTLY_TRIALS
