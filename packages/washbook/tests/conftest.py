"""Shared fixtures.

`benchmarks/baselines/` is not part of the installed package — it is the "before"
the "after" is measured against, and shipping a deliberately slow implementation to
users would be odd. Tests still need it, so it goes on the path here.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))


def make_panel(
    n_symbols: int = 20, n_days: int = 250, *, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A signal panel and a set of realized losses.

    Deterministic without a random generator: entries and losses fall on fixed
    arithmetic patterns, so a failure reproduces exactly and a benchmark run is
    comparable to the one before it.
    """
    start = date(2024, 1, 1)
    symbols = [f"S{i:04d}" for i in range(n_symbols)]

    rows = {
        "symbol": [],
        "date": [],
        "entry": [],
    }
    for si, symbol in enumerate(symbols):
        for d in range(n_days):
            rows["symbol"].append(symbol)
            rows["date"].append(start + timedelta(days=d))
            # Roughly one entry every eleven bars, offset per symbol so the panel
            # is not synchronized.
            rows["entry"].append((d + si + seed) % 11 == 0)
    signals = pl.DataFrame(rows)

    loss_rows = {"symbol": [], "loss_date": []}
    for si, symbol in enumerate(symbols):
        for d in range(0, n_days, 37):
            loss_rows["symbol"].append(symbol)
            loss_rows["loss_date"].append(start + timedelta(days=(d + si) % n_days))
    losses = pl.DataFrame(loss_rows)

    return signals, losses


@pytest.fixture
def panel() -> tuple[pl.DataFrame, pl.DataFrame]:
    """A small panel, fast enough for every test to build its own."""
    return make_panel()
