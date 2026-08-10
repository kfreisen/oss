"""Options-implied move, catalyst-date inference, and calibrated event probability.

Everything in this package is closed-form arithmetic over stdlib :mod:`math`. The
absence of dependencies is part of the contract: this is meant to be droppable into
an environment where adding scipy is not worth it, and to be readable end to end.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.0.1.dev0"
