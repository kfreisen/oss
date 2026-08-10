"""Blocked fuzzy entity resolution.

Group records that describe the same real-world entity across sources, using a
caller-supplied blocking key to keep the comparison count tractable and a scorer that
penalizes disagreement on discriminative tokens rather than on filler.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.0.1.dev0"
