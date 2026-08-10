"""The vectorized cooldown and the serial loop must produce the same answer.

This is the load-bearing test in the package. Every speed claim washbook makes is
a comparison against `benchmarks/baselines/serial.py`, and a comparison is only
meaningful if both implementations do the same job.

Unlike a heuristic, this one admits exact equality — the two are computing the same
predicate over the same rows, so anything short of identical output is a bug rather
than a tolerable difference.
"""

from __future__ import annotations

import polars as pl
import pytest
from baselines.serial import suppress_after_losses_serial
from conftest import make_panel
from washbook.cooldown import suppress_after_losses


def assert_identical(signals: pl.DataFrame, losses: pl.DataFrame, cooldown_days: int) -> None:
    """Both implementations must agree row for row."""
    fast = suppress_after_losses(signals, losses, cooldown_days=cooldown_days)
    slow_frame, slow_suppressed = suppress_after_losses_serial(
        signals, losses, cooldown_days=cooldown_days
    )

    columns = ["symbol", "date", "entry", "blocked"]
    left = fast.signals.select(columns).sort(["symbol", "date"])
    right = slow_frame.select(columns).sort(["symbol", "date"])

    assert left.equals(right), "vectorized and serial gating disagree"
    assert fast.suppressed == slow_suppressed


def test_agreement_on_a_typical_panel(panel: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    signals, losses = panel
    assert_identical(signals, losses, 31)


@pytest.mark.parametrize("cooldown_days", [0, 1, 7, 31, 90, 400])
def test_agreement_across_window_lengths(
    panel: tuple[pl.DataFrame, pl.DataFrame], cooldown_days: int
) -> None:
    # A window longer than the panel suppresses nearly everything; a window of
    # zero suppresses nothing. Both are boundaries worth pinning.
    signals, losses = panel
    assert_identical(signals, losses, cooldown_days)


def test_agreement_when_a_symbol_has_no_losses(
    panel: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """An as-of join produces nulls for unmatched partitions.

    Treating a null "days since loss" as zero rather than as "no loss" would
    suppress every entry for exactly those symbols that never lost money, which is
    both wrong and the sort of thing a summary statistic hides.
    """
    signals, losses = panel
    trimmed = losses.filter(pl.col("symbol") != "S0000")
    assert_identical(signals, trimmed, 31)


def test_agreement_when_no_symbol_has_losses(
    panel: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    signals, losses = panel
    empty = losses.clear()
    assert_identical(signals, empty, 31)


def test_agreement_when_losses_precede_the_panel(
    panel: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    from datetime import date

    signals, _ = panel
    early = pl.DataFrame(
        {
            "symbol": ["S0000", "S0001"],
            "loss_date": [date(2020, 1, 1), date(2020, 6, 1)],
        }
    )
    assert_identical(signals, early, 31)


def test_agreement_on_unsorted_input(panel: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    """An as-of join on unsorted input is silently wrong, not loudly wrong.

    The implementation sorts rather than documenting a precondition, and this is
    what holds it to that.
    """
    signals, losses = panel
    shuffled = signals.sample(fraction=1.0, shuffle=True, seed=7)
    assert_identical(shuffled, losses, 31)


def test_agreement_with_duplicate_loss_dates(
    panel: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    signals, losses = panel
    doubled = pl.concat([losses, losses])
    assert_identical(signals, doubled, 31)


def test_agreement_on_a_single_symbol() -> None:
    signals, losses = make_panel(n_symbols=1, n_days=120)
    assert_identical(signals, losses, 31)


def test_agreement_on_a_wide_panel() -> None:
    # Many symbols, few bars — the shape where the serial version's per-symbol
    # overhead dominates and where a partitioning bug would show up.
    signals, losses = make_panel(n_symbols=200, n_days=20)
    assert_identical(signals, losses, 31)
