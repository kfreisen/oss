"""Fast roster optimization.

`slatekit` builds valid lineups under salary, position, and group constraints. The
hot path is Rust; this package is the typed, documented surface over it.

```python
from slatekit import PlayerPool, build_lineups
from slatekit.presets import DK_MLB_CLASSIC

pool = PlayerPool.from_records(records, DK_MLB_CLASSIC)
lineups = build_lineups(pool, DK_MLB_CLASSIC, num_lineups=500, seed=1)
```

The compiled kernel lives in `slatekit._native`, which is private: it takes flat
arrays chosen for cheap marshalling and gives no diagnostics. Everything supported
is re-exported here, so pinning to these names insulates you from changes to the
boundary.
"""

from __future__ import annotations

from slatekit._native import __version__ as _native_version
from slatekit._native import active_isa
from slatekit.greedy import CONTRARIAN, STANDARD, JitterProfile, build_lineups
from slatekit.pool import PlayerPool
from slatekit.spec import GroupConstraint, RosterSpec, Slot

__all__ = [
    "CONTRARIAN",
    "STANDARD",
    "GroupConstraint",
    "JitterProfile",
    "PlayerPool",
    "RosterSpec",
    "Slot",
    "__version__",
    "active_isa",
    "build_lineups",
    "native_version",
]

__version__ = "0.0.1.dev0"


def native_version() -> str:
    """Return the version of the compiled kernel.

    Tracks the Python package version but is reported separately: a mismatch means
    a stale build artifact is on the path, which otherwise presents as baffling
    behavior rather than an import error.
    """
    return str(_native_version)
