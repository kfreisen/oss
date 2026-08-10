"""Wash-sale avoidance: suppress re-entry after a realized loss.

The other way to model the rule. Rather than accounting for disallowed losses,
refuse to re-enter a symbol within the window, so no wash sale ever happens. This
changes the equity curve rather than the tax line, and the cost shows up as
suppressed entries.

It is a *conservative* model, and knowingly so: the blackout blocks every entry in
the window, whereas §1091 only disallows a loss when a replacement is actually
purchased. The suppressed set is a strict superset of what the statute forbids, so
a backtest run this way reports a lower bound on achievable return.

## Why this is the part worth optimizing

It runs over the whole signal panel — every symbol, every bar — while the deferral
ledger runs over closed trades, which is smaller by orders of magnitude. The
obvious implementation is a loop over symbols with an inner scan over bars, and
that is what this replaces.

The vectorized form asks, for every bar, "when was the most recent losing exit in
this symbol?" — which is exactly a backward as-of join partitioned by symbol. One
join over the whole panel, in one lazy query, instead of one pass per symbol.

`benchmarks/baselines/serial.py` keeps the loop, and `tests/test_parity.py` asserts
the two agree exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["CooldownResult", "suppress_after_losses"]

# The source implementations used 31 rather than 30: the statute's window is 30
# days *after* the sale, and re-entering on day 31 is the first safe day.
DEFAULT_COOLDOWN_DAYS = 31


class CooldownResult:
    """Gated signals plus a count of what the gate cost.

    Attributes:
        signals: The input frame with `blocked` added and the entry column gated.
        suppressed: How many entry signals were removed. This is the whole point
            of reporting anything: a high-turnover strategy can lose most of its
            entries to the blackout, and a backtest that only shows the resulting
            Sharpe hides why it moved.
    """

    __slots__ = ("signals", "suppressed")

    def __init__(self, signals: pl.DataFrame, suppressed: int) -> None:
        self.signals = signals
        self.suppressed = suppressed


def suppress_after_losses(
    signals: pl.DataFrame | pl.LazyFrame,
    losses: pl.DataFrame | pl.LazyFrame,
    *,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
    symbol_col: str = "symbol",
    date_col: str = "date",
    entry_col: str = "entry",
    loss_date_col: str = "loss_date",
) -> CooldownResult:
    """Blank out entry signals within `cooldown_days` of a realized loss.

    Args:
        signals: Long-format panel with a symbol, a date, and a boolean entry
            column. Need not be sorted.
        losses: Realized-loss dates, with a symbol column and `loss_date_col`.
        cooldown_days: Days after a loss during which entries are suppressed. A
            signal exactly `cooldown_days` after a loss is allowed — that is the
            first safe day.
        symbol_col: Partition column, present in both frames.
        date_col: Date column in `signals`.
        entry_col: Boolean entry column in `signals`.
        loss_date_col: Date column in `losses`.

    Returns:
        A [`CooldownResult`][washbook.cooldown.CooldownResult] whose `signals`
        frame carries `blocked` alongside the gated entry column, so a caller can
        see which bars were suppressed and not just that some were.

    Raises:
        ValueError: If a required column is missing, or `cooldown_days` is
            negative.
    """
    if cooldown_days < 0:
        msg = f"cooldown_days must be non-negative, got {cooldown_days}"
        raise ValueError(msg)

    signal_lf = signals.lazy() if isinstance(signals, pl.DataFrame) else signals
    loss_lf = losses.lazy() if isinstance(losses, pl.DataFrame) else losses

    _require(signal_lf, [symbol_col, date_col, entry_col], "signals")
    _require(loss_lf, [symbol_col, loss_date_col], "losses")

    # An as-of join needs both sides sorted by the join key. Doing it here rather
    # than documenting a precondition: an unsorted input produces wrong answers
    # silently, which is the worst possible failure mode for an accounting tool.
    ordered_signals = signal_lf.sort([symbol_col, date_col])
    ordered_losses = (
        loss_lf.select([symbol_col, pl.col(loss_date_col).alias(date_col)])
        .unique()
        .sort([symbol_col, date_col])
    )

    gated = (
        ordered_signals.join_asof(
            ordered_losses.with_columns(pl.col(date_col).alias("_last_loss")),
            on=date_col,
            by=symbol_col,
            strategy="backward",
        )
        .with_columns(
            (pl.col(date_col) - pl.col("_last_loss")).dt.total_days().alias("_days_since_loss")
        )
        .with_columns(
            (
                pl.col("_days_since_loss").is_not_null()
                & (pl.col("_days_since_loss") < cooldown_days)
                & pl.col(entry_col).fill_null(value=False)
            ).alias("blocked")
        )
        .with_columns(
            (pl.col(entry_col).fill_null(value=False) & ~pl.col("blocked")).alias(entry_col)
        )
        .drop("_last_loss", "_days_since_loss")
    )

    frame = gated.collect()
    return CooldownResult(frame, int(frame.get_column("blocked").sum()))


def _require(frame: pl.LazyFrame, columns: Sequence[str], name: str) -> None:
    """Fail with the missing column named, rather than a KeyError from polars."""
    available = frame.collect_schema().names()
    missing = [c for c in columns if c not in available]
    if missing:
        msg = f"{name} is missing column(s) {missing}; it has {available}"
        raise ValueError(msg)
