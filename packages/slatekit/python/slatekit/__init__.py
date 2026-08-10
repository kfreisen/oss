"""Fast roster optimization.

`slatekit` builds valid rosters under salary, position, and group constraints, and
selects a diverse portfolio of them against simulated outcomes. The hot paths are
implemented in Rust; this module is the typed, documented surface over them.

The compiled kernel is reached through :mod:`slatekit._native`, which is private.
Everything supported is re-exported here, so pinning to the public names insulates
you from changes to the FFI layer.
"""

from __future__ import annotations

from slatekit._native import __version__ as _native_version
from slatekit._native import active_isa

__all__ = ["__version__", "active_isa", "native_version"]

__version__ = "0.0.1.dev0"


def native_version() -> str:
    """Return the version of the compiled kernel.

    This tracks the Python package version but is reported separately: a mismatch
    means a stale build artifact is on the path, which otherwise presents as
    baffling behavior rather than an import error.
    """
    return str(_native_version)
