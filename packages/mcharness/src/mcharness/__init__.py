"""Reproducible batched Monte Carlo on Ray.

The harness owns batching, per-batch seed derivation, dispatch, and streaming
collection; the caller supplies a simulation. Results depend on the run seed and the
batch layout, never on scheduling order — a run reproduces at any level of
parallelism, including serial.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.0.1.dev0"
