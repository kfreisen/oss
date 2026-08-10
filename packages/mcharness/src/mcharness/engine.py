"""Running a Monte Carlo study, serially or on Ray.

Two backends behind one function, and they are required to agree exactly. That
requirement is what makes the parallel version debuggable: when a distributed run
produces something surprising, `backend="serial"` reproduces it on one core with a
debugger attached, and if it does not, the harness is at fault rather than the
simulation.

Collection uses `ray.wait` rather than `ray.get` on the whole list. `ray.get`
materializes every batch result before the reduction starts, so peak memory scales
with the number of trials and nothing is reported until the slowest batch lands.
Waiting for whichever batch finishes next and folding it in immediately bounds
memory by the accumulator and gives progress as it happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from mcharness.seeding import batch_bounds, spawn_generators
from mcharness.simulation import CollectReducer

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from mcharness.simulation import Reducer, Simulation

__all__ = ["Backend", "RunResult", "run"]

TrialT = TypeVar("TrialT")
AccT = TypeVar("AccT")

Backend = str
"""``"serial"`` or ``"ray"``."""


@dataclass(frozen=True, slots=True)
class RunResult(Generic[AccT]):
    """The outcome of a study.

    Attributes:
        value: The reducer's accumulator.
        n_trials: Trials actually run.
        n_batches: Batches dispatched.
        backend: Which backend ran it.
        run_seed: The seed, so a result carries what is needed to reproduce it.
    """

    value: AccT
    n_trials: int
    n_batches: int
    backend: Backend
    run_seed: int


def _run_batch(
    simulation: Simulation[TrialT],
    rng: np.random.Generator,
    start: int,
    stop: int,
) -> list[TrialT]:
    """Run one batch's trials. The unit of work both backends dispatch."""
    return [simulation.run_trial(rng, index) for index in range(start, stop)]


def _run_indexed_batch(
    batch_index: int,
    simulation: Simulation[TrialT],
    rng: np.random.Generator,
    start: int,
    stop: int,
) -> tuple[int, list[TrialT]]:
    """Run a batch and carry its index back with it.

    Ray returns whichever batch finishes first, so the index has to travel with
    the payload — the future's position in the submitted list is not recoverable
    from what `ray.wait` hands back.
    """
    return batch_index, _run_batch(simulation, rng, start, stop)


def run(
    simulation: Simulation[TrialT],
    n_trials: int,
    *,
    reducer: Reducer[TrialT, AccT] | None = None,
    batch_size: int = 1000,
    run_seed: int = 0,
    backend: Backend = "serial",
    ray_options: dict[str, Any] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> RunResult[Any]:
    """Run `n_trials` trials of `simulation`.

    Args:
        simulation: The trial. Must be picklable for the Ray backend.
        n_trials: How many trials to run.
        reducer: How to combine results. Defaults to collecting them all, which
            makes memory scale with `n_trials` — replace it for a large run.
        batch_size: Trials per dispatched unit. **Part of the reproducibility
            contract, not a free tuning knob:** streams are derived per batch, so
            changing this changes which trials share a stream and therefore
            changes the result. Reproducing a run means matching `run_seed` and
            `batch_size` both. Making it irrelevant would mean deriving a stream
            per trial, which costs more than most trials do.
        run_seed: Determines every batch's random stream.
        backend: `"serial"` or `"ray"`.
        ray_options: Passed to `ray.init` if Ray is not already initialized. A
            session Ray is left alone, and left running afterwards — shutting down
            something this function did not start would break every other user in
            the process.
        on_progress: Called with `(completed_trials, total_trials)` as each batch
            lands.

    Returns:
        A [`RunResult`][mcharness.engine.RunResult].

    Raises:
        ValueError: If `n_trials` is negative or the backend is unknown.
        ImportError: If the Ray backend is requested and Ray is not installed.

    Example:
        ```python
        from mcharness import run

        result = run(MySimulation(), 100_000, backend="ray", run_seed=1)
        ```
    """
    if n_trials < 0:
        msg = f"n_trials must be non-negative, got {n_trials}"
        raise ValueError(msg)
    if backend not in {"serial", "ray"}:
        msg = f"unknown backend {backend!r}; expected 'serial' or 'ray'"
        raise ValueError(msg)

    active: Reducer[TrialT, Any] = reducer if reducer is not None else CollectReducer()
    bounds = batch_bounds(n_trials, batch_size)
    generators = spawn_generators(run_seed, len(bounds))

    runner = _run_serial if backend == "serial" else _run_ray
    accumulator = runner(simulation, bounds, generators, active, ray_options, on_progress)

    return RunResult(
        value=accumulator,
        n_trials=n_trials,
        n_batches=len(bounds),
        backend=backend,
        run_seed=run_seed,
    )


def _run_serial(
    simulation: Simulation[TrialT],
    bounds: list[tuple[int, int]],
    generators: list[np.random.Generator],
    reducer: Reducer[TrialT, AccT],
    _ray_options: dict[str, Any] | None,
    on_progress: Callable[[int, int], None] | None,
) -> AccT:
    """Run every batch in this process, in order."""
    total = bounds[-1][1] if bounds else 0
    accumulator = reducer.initial()
    completed = 0
    for index, ((start, stop), rng) in enumerate(zip(bounds, generators, strict=True)):
        accumulator = reducer.add(accumulator, index, _run_batch(simulation, rng, start, stop))
        completed += stop - start
        if on_progress is not None:
            on_progress(completed, total)
    return reducer.finalize(accumulator)


def _run_ray(
    simulation: Simulation[TrialT],
    bounds: list[tuple[int, int]],
    generators: list[np.random.Generator],
    reducer: Reducer[TrialT, AccT],
    ray_options: dict[str, Any] | None,
    on_progress: Callable[[int, int], None] | None,
) -> AccT:
    """Dispatch batches to Ray and fold results in as they land."""
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - exercised only without ray
        msg = "the 'ray' backend requires ray to be installed"
        raise ImportError(msg) from exc

    started_here = False
    if not ray.is_initialized():
        ray.init(**(ray_options or {}))
        started_here = True

    remote_batch: Any = ray.remote(_run_indexed_batch)
    total = bounds[-1][1] if bounds else 0
    accumulator = reducer.initial()
    completed = 0

    try:
        pending = [
            remote_batch.remote(index, simulation, rng, start, stop)
            for index, ((start, stop), rng) in enumerate(zip(bounds, generators, strict=True))
        ]
        # Reduce as results arrive. `ray.get(pending)` would hold every batch in
        # memory at once and report nothing until the last one landed.
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            for reference in ready:
                batch_index, results = ray.get(reference)
                accumulator = reducer.add(accumulator, batch_index, results)
                completed += len(results)
                if on_progress is not None:
                    on_progress(completed, total)
    finally:
        # Only tear down a Ray this call started. A session Ray belongs to whoever
        # started it, and shutting it down here would break them.
        if started_here:
            ray.shutdown()

    return reducer.finalize(accumulator)
