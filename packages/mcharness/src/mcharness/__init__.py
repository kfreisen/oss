"""Reproducible batched Monte Carlo on Ray.

The harness owns batching, per-batch seed derivation, dispatch, streaming
collection and progress. You supply a simulation and, for a large run, a reducer.

```python
from mcharness import run

result = run(MySimulation(), 1_000_000, backend="ray", run_seed=1)
```

Three things it is careful about, each of which is a way parallel Monte Carlo
usually goes wrong:

- **Seeds** come from `SeedSequence.spawn`, not from `seed + index`. Adjacent
  integer seeds are not guaranteed to give independent streams, and correlated
  streams show up as variance that is quietly too low.
- **Collection** streams via `ray.wait`, so memory is bounded by the accumulator
  rather than by the trial count, and progress arrives as it happens.
- **Order** is explicit. Batches complete out of order, so reducers receive the
  batch index and restore any ordering in `finalize` — which is what makes the
  serial and Ray backends produce identical results.
"""

from __future__ import annotations

from mcharness.cache import EntityCache
from mcharness.engine import RunResult, run
from mcharness.seeding import batch_bounds, spawn_generators
from mcharness.simulation import CollectReducer, Reducer, Simulation

__all__ = [
    "CollectReducer",
    "EntityCache",
    "Reducer",
    "RunResult",
    "Simulation",
    "__version__",
    "batch_bounds",
    "run",
    "spawn_generators",
]

__version__ = "0.0.1.dev0"
