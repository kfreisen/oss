"""Benchmark: greedy construction against MILP and against pure Python.

Three implementations of the same job, on the same slate:

* `slatekit` — the Rust kernel.
* `reference` — the pure-Python transcription in `baselines/reference.py`, which
  is also the parity oracle. This is the honest measure of what the port bought.
* `milp_pulp` / `milp_ortools` — solver formulations from `baselines/milp.py`,
  producing distinct lineups via no-good cuts.

Reported as **lineups per second**, not seconds, because the solvers scale
differently: each lineup is a fresh solve against a problem one constraint larger,
so their total is superlinear while the greedy builder is linear. A single
wall-clock number at one portfolio size would hide that.

Quality is measured too, in `bench_quality.py` — speed alone would be a misleading
thing to publish, since the solver produces *better individual lineups* and this
comparison would look like a win on every axis if only time were reported.

Run with:

    make bench PKG=slatekit
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines.milp import solve_portfolio_pulp
from baselines.reference import build_lineups_reference
from slatekit import build_lineups
from slatekit.pool import PlayerPool
from slatekit.presets import DK_MLB_CLASSIC

# PuLP 3.x warns that constructing LpVariable directly is deprecated in favour of a
# 4.0 API that does not exist in 3.x. The package pins `pulp<4` precisely because
# 4.0 removes PULP_CBC_CMD, so there is nothing to migrate to yet and the warning
# is noise. Scoped to this module rather than relaxed globally.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pulp.*")


def make_slate(per_position: int = 10) -> PlayerPool:
    """A realistic-shaped DraftKings MLB slate.

    Deterministic: a benchmark whose input changes between runs cannot be compared
    against a committed result.
    """
    records: list[dict[str, object]] = []
    for position in ("P", "C", "1B", "2B", "3B", "SS", "OF"):
        count = per_position * 3 if position == "OF" else per_position
        for k in range(count):
            records.append(
                {
                    "name": f"{position}-{k}",
                    "positions": (position,),
                    "salary": 2500 + (k % 12) * 750,
                    "projection": 4.0 + (k % 12) * 1.2,
                    "stddev": 3.0 + (k % 4),
                    "ownership": ((k * 7) % 30) / 100.0,
                    "team": f"TM{k % 10}",
                }
            )
    return PlayerPool.from_records(records, DK_MLB_CLASSIC)


# Solvers are orders of magnitude slower, so they are measured at portfolio sizes
# that finish in reasonable time; lineups-per-second is what gets compared.
#
# The MILP ceiling is deliberately low. Each additional lineup is a fresh solve
# against a problem carrying one more no-good cut, so the cost grows superlinearly
# — 20 lineups did not finish in fifteen minutes on the reference machine. That is
# the result, not an obstacle to it, and it is visible at 3 versus 10 without
# making `make bench` unrunnable.
KERNEL_SIZES = [20, 150, 1000]
REFERENCE_SIZES = [20, 150]
MILP_SIZES = [3, 10]


@pytest.mark.parametrize("num_lineups", KERNEL_SIZES)
def test_kernel(benchmark, num_lineups: int) -> None:
    pool = make_slate()
    benchmark.extra_info["case"] = f"build/{num_lineups}"
    benchmark.extra_info["impl"] = "slatekit_rust"
    benchmark.extra_info["params"] = {"num_lineups": num_lineups, "pool": len(pool)}

    result = benchmark(build_lineups, pool, DK_MLB_CLASSIC, num_lineups=num_lineups, seed=1)
    assert len(result) > 0


@pytest.mark.parametrize("num_lineups", REFERENCE_SIZES)
def test_reference_python(benchmark, num_lineups: int) -> None:
    pool = make_slate()
    benchmark.extra_info["case"] = f"build/{num_lineups}"
    benchmark.extra_info["impl"] = "reference_python"
    benchmark.extra_info["params"] = {"num_lineups": num_lineups, "pool": len(pool)}

    result = benchmark(
        build_lineups_reference, pool, DK_MLB_CLASSIC, num_lineups=num_lineups, seed=1
    )
    assert len(result) > 0


@pytest.mark.parametrize("num_lineups", MILP_SIZES)
def test_milp_pulp(benchmark, num_lineups: int) -> None:
    pool = make_slate()
    benchmark.extra_info["case"] = f"build/{num_lineups}"
    benchmark.extra_info["impl"] = "milp_pulp_cbc"
    benchmark.extra_info["params"] = {"num_lineups": num_lineups, "pool": len(pool)}

    # Few rounds: each is a full solve, and pytest-benchmark's default calibration
    # would otherwise spend minutes here for no extra precision.
    result = benchmark.pedantic(
        solve_portfolio_pulp,
        args=(pool, DK_MLB_CLASSIC),
        kwargs={"num_lineups": num_lineups},
        rounds=3,
        iterations=1,
    )
    assert len(result) > 0
