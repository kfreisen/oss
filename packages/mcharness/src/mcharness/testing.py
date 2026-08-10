"""Small example simulations, for tests, documentation, and benchmarks.

These live in the installed package rather than in a test file for a reason worth
knowing about: **Ray unpickles the simulation in a worker process**, and a worker
cannot import your `conftest.py`. A simulation defined there fails with
`ModuleNotFoundError: No module named 'conftest'` at dispatch, which is a confusing
way to learn the picklability rule.

The same applies to your own code. A simulation has to be importable by name from
somewhere the workers can reach — a module in your package, not a closure, not a
lambda, not something defined in `__main__` or a test file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Coin", "Draw", "Indexed", "Normal"]


class Coin:
    """Returns 1.0 with probability `p`. The mean converges to `p`."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def run_trial(self, rng: np.random.Generator, index: int) -> float:  # noqa: ARG002
        """Flip once."""
        return float(rng.random() < self.p)


class Draw:
    """Returns a raw uniform draw.

    Useful for checking stream behavior, because any difference in seeding shows
    up directly instead of being hidden behind a threshold.
    """

    def run_trial(self, rng: np.random.Generator, index: int) -> float:  # noqa: ARG002
        """Draw once."""
        return float(rng.random())


class Indexed:
    """Returns the trial index, making result ordering directly observable."""

    def run_trial(self, rng: np.random.Generator, index: int) -> int:  # noqa: ARG002
        """Return the index."""
        return index


class Normal:
    """Draws from a normal distribution."""

    def __init__(self, mean: float = 0.0, stddev: float = 1.0) -> None:
        self.mean = mean
        self.stddev = stddev

    def run_trial(self, rng: np.random.Generator, index: int) -> float:  # noqa: ARG002
        """Draw once."""
        return float(rng.normal(self.mean, self.stddev))
