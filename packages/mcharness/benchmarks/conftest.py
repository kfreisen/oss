"""Benchmark fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ray_bench():
    """A Ray started once for the whole benchmark session.

    Per-run startup would dominate every measurement and make the parallel
    numbers meaningless.
    """
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        # The benchmark simulations live under benchmarks/baselines/, which a Ray
        # worker cannot import by default — it fails at dispatch with
        # ModuleNotFoundError. Shipping the tree as the runtime working_dir is what
        # makes an out-of-package simulation dispatchable, and is the same thing a
        # user with code outside their installed package needs.
        #
        # It has to be the *package* directory, not benchmarks/: uv's Ray hook
        # requires pyproject.toml to be inside the working_dir, and imports are
        # therefore rooted there — hence `benchmarks.baselines...` in the bench
        # module rather than the `baselines...` the other packages use.
        ray.init(
            include_dashboard=False,
            log_to_driver=False,
            configure_logging=False,
            runtime_env={"working_dir": str(Path(__file__).resolve().parent.parent)},
        )
    yield ray
    ray.shutdown()
