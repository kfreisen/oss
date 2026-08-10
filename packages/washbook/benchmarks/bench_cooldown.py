"""Benchmark: vectorized cooldown gating against the serial per-symbol loop.

Swept over symbol count rather than over total rows, because that is the axis the
two implementations differ on. The serial version pays a fixed cost per symbol —
a frame filter, a materialization out of Arrow, a Python loop — so its total grows
with the number of partitions even when the row count is held constant. The
vectorized version sees one job regardless.

Both produce identical output; `tests/test_parity.py` is what holds that.

Run with:

    make bench PKG=washbook
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines.serial import suppress_after_losses_serial
from washbook.cooldown import suppress_after_losses


def make_panel(n_symbols: int, n_days: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A signal panel and its realized losses. Deterministic."""
    start = date(2024, 1, 1)
    rows: dict[str, list] = {"symbol": [], "date": [], "entry": []}
    loss_rows: dict[str, list] = {"symbol": [], "loss_date": []}

    for si in range(n_symbols):
        symbol = f"S{si:05d}"
        for d in range(n_days):
            rows["symbol"].append(symbol)
            rows["date"].append(start + timedelta(days=d))
            rows["entry"].append((d + si) % 11 == 0)
        for d in range(0, n_days, 37):
            loss_rows["symbol"].append(symbol)
            loss_rows["loss_date"].append(start + timedelta(days=(d + si) % n_days))

    return pl.DataFrame(rows), pl.DataFrame(loss_rows)


# Row count is held roughly constant across the first three so the sweep isolates
# partition count. The last one is a realistic full-universe panel.
SHAPES = [
    (10, 5000),
    (100, 500),
    (1000, 50),
    (3000, 500),
]


@pytest.mark.parametrize(("n_symbols", "n_days"), SHAPES)
def test_vectorized(benchmark, n_symbols: int, n_days: int) -> None:
    signals, losses = make_panel(n_symbols, n_days)
    benchmark.extra_info["case"] = f"cooldown/{n_symbols}sym_x_{n_days}bar"
    benchmark.extra_info["impl"] = "polars_asof"
    benchmark.extra_info["params"] = {
        "symbols": n_symbols,
        "bars": n_days,
        "rows": len(signals),
    }
    result = benchmark(suppress_after_losses, signals, losses, cooldown_days=31)
    assert result.suppressed > 0


@pytest.mark.parametrize(("n_symbols", "n_days"), SHAPES)
def test_serial(benchmark, n_symbols: int, n_days: int) -> None:
    signals, losses = make_panel(n_symbols, n_days)
    benchmark.extra_info["case"] = f"cooldown/{n_symbols}sym_x_{n_days}bar"
    benchmark.extra_info["impl"] = "baseline_serial"
    benchmark.extra_info["params"] = {
        "symbols": n_symbols,
        "bars": n_days,
        "rows": len(signals),
    }
    # The 3000-symbol case takes seconds per round; the default calibration would
    # spend minutes for no additional precision.
    _, suppressed = benchmark.pedantic(
        suppress_after_losses_serial,
        args=(signals, losses),
        kwargs={"cooldown_days": 31},
        rounds=3 if n_symbols >= 1000 else 5,
        iterations=1,
    )
    assert suppressed > 0
