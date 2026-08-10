"""The Ray backend must produce exactly what the serial backend produces.

This is the load-bearing test. The whole argument for the harness is that a
distributed run is debuggable — that `backend="serial"` reproduces whatever the
cluster did, on one core, with a debugger attached. If the two ever disagree, that
argument is gone and every speedup measured against the serial baseline is
measuring two different computations.

Exact equality, not approximate. Both backends run the same trials with the same
per-batch streams, so anything other than identical output is a bug.
"""

from __future__ import annotations

import pytest
from baselines.serial_uncached import CachedSimulation, LookupSimulation, run_serial_loop
from conftest import Coin, Draw, Indexed
from mcharness import EntityCache, run


def test_backends_agree_on_a_coin(ray_session) -> None:
    serial = run(Coin(0.3), 500, batch_size=100, run_seed=7, backend="serial")
    parallel = run(Coin(0.3), 500, batch_size=100, run_seed=7, backend="ray")
    assert serial.value == parallel.value


def test_backends_agree_on_raw_draws(ray_session) -> None:
    # Floats, so any difference in stream derivation shows up immediately rather
    # than being hidden behind a threshold.
    serial = run(Draw(), 400, batch_size=50, run_seed=13, backend="serial")
    parallel = run(Draw(), 400, batch_size=50, run_seed=13, backend="ray")
    assert serial.value == parallel.value


def test_ray_preserves_trial_order_despite_out_of_order_completion(ray_session) -> None:
    """The failure this package was nearly shipped with.

    `ray.wait` returns whichever batch finishes first. Folding results in as they
    arrive, without carrying the batch index, produces a correctly-sized result in
    scheduling order — which passes a length check, passes a mean check, and is
    wrong.
    """
    result = run(Indexed(), 350, batch_size=25, run_seed=1, backend="ray")
    assert result.value == list(range(350))


@pytest.mark.parametrize("batch_size", [1, 7, 100, 1000])
def test_backends_agree_at_every_batch_size(ray_session, batch_size: int) -> None:
    serial = run(Draw(), 200, batch_size=batch_size, run_seed=4, backend="serial")
    parallel = run(Draw(), 200, batch_size=batch_size, run_seed=4, backend="ray")
    assert serial.value == parallel.value


def test_a_batch_size_larger_than_the_run_is_one_batch(ray_session) -> None:
    result = run(Coin(), 10, batch_size=1000, run_seed=2, backend="ray")
    assert result.n_batches == 1
    assert len(result.value) == 10


def test_ray_reports_progress(ray_session) -> None:
    seen: list[int] = []
    run(
        Coin(),
        300,
        batch_size=100,
        backend="ray",
        on_progress=lambda done, _total: seen.append(done),
    )
    # Arrival order is not guaranteed, but the totals must be and the last must
    # be everything.
    assert sorted(seen) == [100, 200, 300]


def test_zero_trials_on_ray_is_not_an_error(ray_session) -> None:
    assert run(Coin(), 0, backend="ray").value == []


def test_the_cached_and_uncached_simulations_agree() -> None:
    """The cache is an optimization, so it must not change the answer.

    Without this, the cached benchmark could be measuring a different computation
    and the speedup would be meaningless.
    """
    import numpy as np

    records = [{"player_id": f"p{i}", "mean": 10.0 + i, "stddev": 1.0 + i % 3} for i in range(20)]
    table = {r["player_id"]: r for r in records}
    entity_ids = [f"p{i % 20}" for i in range(50)]

    cache = EntityCache.from_records(records, key="player_id", fields=("mean", "stddev"))
    positions = [cache.index_of(e) for e in entity_ids]

    uncached = run_serial_loop(LookupSimulation(table, entity_ids), 50, np.random.default_rng(5))
    cached = run_serial_loop(CachedSimulation(cache, positions), 50, np.random.default_rng(5))
    assert uncached == pytest.approx(cached)
