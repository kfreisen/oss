"""Shared fixtures.

Ray is started once for the whole session and reused. Starting and stopping it per
test costs seconds each time and is the main reason Ray test suites get skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mcharness import testing as _testing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))


# Simulations come from the installed package, not from this file: Ray unpickles
# the simulation in a worker, and a worker cannot import conftest. Defining them
# here fails at dispatch with ModuleNotFoundError, which is a confusing way to
# discover the picklability rule.
Coin = _testing.Coin
Draw = _testing.Draw
Indexed = _testing.Indexed


@pytest.fixture(scope="session")
def ray_session():
    """A small local Ray, started once."""
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, configure_logging=False)
    yield ray
    ray.shutdown()
