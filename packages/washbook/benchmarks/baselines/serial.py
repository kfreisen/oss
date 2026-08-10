"""Serial per-symbol cooldown gating — the implementation that was replaced.

This is the obvious way to write it, and it is what the vectorized version in
`washbook.cooldown` is measured against. It is kept working and kept tested, so
the speedup is a comparison between two things that produce the same answer rather
than an assertion.

The shape of the cost is the interesting part. Per symbol it filters the panel,
pulls the rows into Python, and walks them forward against a pointer into that
symbol's loss dates. The per-row work is small; what dominates at scale is doing it
once per symbol — thousands of small frame filters, thousands of round trips out of
Arrow into Python objects, and no opportunity for the query engine to see the whole
job at once.

Note this is not a straw man. The inner loop is already O(rows + losses) with a
two-pointer walk rather than a scan per row, which is the version someone would
arrive at after noticing the naive one was slow. The remaining cost is structural.
"""

from __future__ import annotations

from bisect import bisect_right

import polars as pl

DEFAULT_COOLDOWN_DAYS = 31


def suppress_after_losses_serial(
    signals: pl.DataFrame,
    losses: pl.DataFrame,
    *,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
    symbol_col: str = "symbol",
    date_col: str = "date",
    entry_col: str = "entry",
    loss_date_col: str = "loss_date",
) -> tuple[pl.DataFrame, int]:
    """Gate entries after realized losses, one symbol at a time.

    Returns the gated frame — sorted by `(symbol, date)`, matching the vectorized
    implementation — and the number of suppressed entries.
    """
    loss_dates: dict[str, list] = {}
    for row in losses.iter_rows(named=True):
        loss_dates.setdefault(row[symbol_col], []).append(row[loss_date_col])
    for dates in loss_dates.values():
        dates.sort()

    frames: list[pl.DataFrame] = []
    suppressed = 0

    for symbol in sorted(signals.get_column(symbol_col).unique().to_list()):
        per_symbol = signals.filter(pl.col(symbol_col) == symbol).sort(date_col)
        dates = loss_dates.get(symbol, [])

        blocked: list[bool] = []
        for row in per_symbol.iter_rows(named=True):
            when = row[date_col]
            entry = bool(row[entry_col])
            if not entry or not dates:
                blocked.append(False)
                continue
            # Most recent loss at or before this bar.
            position = bisect_right(dates, when) - 1
            if position < 0:
                blocked.append(False)
                continue
            blocked.append((when - dates[position]).days < cooldown_days)

        gated = per_symbol.with_columns(pl.Series("blocked", blocked, dtype=pl.Boolean))
        gated = gated.with_columns(
            (pl.col(entry_col).fill_null(value=False) & ~pl.col("blocked")).alias(entry_col)
        )
        suppressed += sum(blocked)
        frames.append(gated)

    if not frames:
        return signals.with_columns(pl.lit(value=False).alias("blocked")), 0
    return pl.concat(frames), suppressed
