"""What the caller supplies, and what the harness promises in return.

The harness owns batching, seeding, dispatch, collection and progress. The caller
owns the simulation and the reduction. Nothing about a domain appears here, which
is the whole point — the original of this code had a baseball game welded into the
dispatch loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

__all__ = ["CollectReducer", "Reducer", "Simulation"]

TrialT = TypeVar("TrialT")
# Variance matters on these protocols: a Simulation only ever *produces* a trial
# result, and a Reducer only ever *consumes* one while producing an accumulator.
# Declaring that lets a Reducer[object, ...] accept a Simulation[int] without a
# cast, which is the ordinary way these get combined.
TrialCo = TypeVar("TrialCo", covariant=True)
TrialContra = TypeVar("TrialContra", contravariant=True)
AccT = TypeVar("AccT")


@runtime_checkable
class Simulation(Protocol[TrialCo]):
    """One trial's worth of work.

    Implementations must be **picklable**: Ray ships this object to each worker.
    That rules out lambdas and closures over open files, and it is worth knowing
    before a confusing serialization error rather than after.

    The random generator is supplied rather than created. A simulation that reaches
    for a module-level `random` or `np.random` breaks reproducibility silently —
    the harness can no longer guarantee the run repeats, and nothing will say so.
    """

    def run_trial(self, rng: np.random.Generator, index: int) -> TrialCo:
        """Run one trial.

        Args:
            rng: A generator private to this batch. Independent of every other
                batch's, and determined by the run seed and the batch index.
            index: Global trial index, for simulations that vary by trial.

        Returns:
            Whatever one trial produces.
        """
        ...


class Reducer(Protocol[TrialContra, AccT]):
    """Combines trial results incrementally.

    Incremental on purpose. Collecting every result and reducing at the end is
    simpler and makes memory scale with trial count, which is what breaks first on
    a long run.

    **Batches arrive out of order.** The Ray backend folds in whichever batch
    finishes next, so completion order depends on scheduling and is not
    reproducible. Two consequences, and ignoring either produces a harness that is
    reproducible in serial and quietly is not in parallel:

    * `add` receives the batch index. An order-sensitive reducer must use it —
      that is what [`CollectReducer`][mcharness.simulation.CollectReducer] does.
    * `finalize` runs once, after every batch, and is where any ordering is
      restored.
    """

    def initial(self) -> AccT:
        """The empty accumulator."""
        ...

    def add(self, accumulator: AccT, batch_index: int, results: Iterable[TrialContra]) -> AccT:
        """Fold a completed batch in. May be called in any batch order."""
        ...

    def finalize(self, accumulator: AccT) -> AccT:
        """Produce the final value once every batch has been folded in."""
        ...


class CollectReducer(Generic[TrialT]):
    """Collects every trial result, in trial order.

    The default, and the first thing to replace on a large run: it keeps
    everything, so memory grows with the number of trials. Supply a reducer that
    accumulates moments or a histogram and memory stays flat.

    Results are buffered per batch and concatenated in batch order by `finalize`,
    which is what makes the parallel backend produce the same list as the serial
    one. Appending in arrival order would be simpler and would silently make the
    output depend on scheduling.
    """

    def initial(self) -> dict[int, list[TrialT]]:
        """An empty per-batch buffer."""
        return {}

    def add(
        self, accumulator: dict[int, list[TrialT]], batch_index: int, results: Iterable[TrialT]
    ) -> dict[int, list[TrialT]]:
        """Store a completed batch under its index."""
        accumulator[batch_index] = list(results)
        return accumulator

    def finalize(self, accumulator: dict[int, list[TrialT]]) -> list[TrialT]:
        """Concatenate the batches in batch order."""
        return [item for index in sorted(accumulator) for item in accumulator[index]]
