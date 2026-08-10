"""Batching, seeding, reduction, and argument validation."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import Coin, Draw, Indexed
from mcharness import CollectReducer, run
from mcharness.seeding import batch_bounds, spawn_generators


def test_it_runs_the_requested_number_of_trials() -> None:
    result = run(Coin(), 250, batch_size=100)
    assert result.n_trials == 250
    assert len(result.value) == 250
    assert result.n_batches == 3


def test_the_last_batch_is_short_rather_than_padded() -> None:
    # Padding and discarding would make the answer depend on batch size in a
    # second, sneakier way than the seeding already does.
    assert batch_bounds(250, 100) == [(0, 100), (100, 200), (200, 250)]


def test_results_come_back_in_trial_order() -> None:
    result = run(Indexed(), 55, batch_size=10)
    assert result.value == list(range(55))


def test_the_same_seed_gives_the_same_answer() -> None:
    a = run(Draw(), 200, batch_size=50, run_seed=11)
    b = run(Draw(), 200, batch_size=50, run_seed=11)
    assert a.value == b.value


def test_a_different_seed_gives_a_different_answer() -> None:
    a = run(Draw(), 200, batch_size=50, run_seed=11)
    b = run(Draw(), 200, batch_size=50, run_seed=12)
    assert a.value != b.value


def test_batch_size_is_part_of_the_contract() -> None:
    """Documented, and therefore pinned.

    Streams are per batch, so changing the batch size changes which trials share
    a stream. If this ever stopped being true the documentation would be wrong in
    the more dangerous direction — people would rely on a guarantee not given.
    """
    a = run(Draw(), 200, batch_size=50, run_seed=11)
    b = run(Draw(), 200, batch_size=100, run_seed=11)
    assert a.value != b.value


def test_batch_streams_are_independent() -> None:
    """Adjacent integer seeds are why SeedSequence.spawn exists.

    Correlated streams show up as variance that is quietly too low, so this checks
    that two batches do not produce suspiciously similar draws.
    """
    first, second = spawn_generators(run_seed=5, n_batches=2)
    a = first.random(2000)
    b = second.random(2000)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.1


def test_the_estimate_converges_to_the_truth() -> None:
    result = run(Coin(0.3), 20_000, batch_size=1000, run_seed=3)
    assert sum(result.value) / len(result.value) == pytest.approx(0.3, abs=0.02)


def test_zero_trials_is_not_an_error() -> None:
    result = run(Coin(), 0)
    assert result.value == []
    assert result.n_batches == 0


def test_progress_is_reported_per_batch() -> None:
    seen: list[tuple[int, int]] = []
    run(Coin(), 250, batch_size=100, on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(100, 250), (200, 250), (250, 250)]


def test_a_custom_reducer_keeps_memory_flat() -> None:
    class MeanReducer:
        def initial(self) -> tuple[float, int]:
            return (0.0, 0)

        def add(self, acc, batch_index, results):
            values = list(results)
            return (acc[0] + sum(values), acc[1] + len(values))

        def finalize(self, acc):
            return acc[0] / acc[1] if acc[1] else 0.0

    result = run(Coin(0.3), 10_000, batch_size=1000, run_seed=3, reducer=MeanReducer())
    assert result.value == pytest.approx(0.3, abs=0.03)


def test_the_result_carries_what_is_needed_to_reproduce_it() -> None:
    result = run(Coin(), 10, batch_size=5, run_seed=99)
    assert result.run_seed == 99
    assert result.backend == "serial"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_trials": -1}, "n_trials must be non-negative"),
        ({"n_trials": 10, "batch_size": 0}, "batch_size must be positive"),
        ({"n_trials": 10, "backend": "dask"}, "unknown backend"),
    ],
)
def test_bad_arguments_are_rejected(kwargs: dict, match: str) -> None:
    n_trials = kwargs.pop("n_trials")
    with pytest.raises(ValueError, match=match):
        run(Coin(), n_trials, **kwargs)


def test_spawn_generators_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError, match="n_batches must be non-negative"):
        spawn_generators(0, -1)


def test_batch_bounds_rejects_negative_trials() -> None:
    with pytest.raises(ValueError, match="n_trials must be non-negative"):
        batch_bounds(-1, 10)


def test_collect_reducer_restores_batch_order() -> None:
    """The default reducer must be immune to arrival order.

    Batches land out of order under Ray, and appending in arrival order would make
    the output depend on scheduling — reproducible in serial, quietly not in
    parallel.
    """
    reducer: CollectReducer[int] = CollectReducer()
    acc = reducer.initial()
    acc = reducer.add(acc, 2, [5, 6])
    acc = reducer.add(acc, 0, [1, 2])
    acc = reducer.add(acc, 1, [3, 4])
    assert reducer.finalize(acc) == [1, 2, 3, 4, 5, 6]
