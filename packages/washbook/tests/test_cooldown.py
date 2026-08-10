"""The cooldown gate's own behavior and its input validation."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from washbook.cooldown import suppress_after_losses
from washbook.ledger import TaxedTrade, Trade

SIGNALS = pl.DataFrame(
    {
        "symbol": ["A", "A", "A", "A"],
        "date": [date(2026, 1, 1), date(2026, 1, 10), date(2026, 2, 15), date(2026, 3, 1)],
        "entry": [True, True, True, True],
    }
)
LOSSES = pl.DataFrame({"symbol": ["A"], "loss_date": [date(2026, 1, 5)]})


def test_entries_inside_the_window_are_suppressed() -> None:
    result = suppress_after_losses(SIGNALS, LOSSES, cooldown_days=31)
    gated = result.signals.sort("date")
    # 2026-01-01 predates the loss; 2026-01-10 is 5 days after, suppressed;
    # 2026-02-15 is 41 days after, allowed.
    assert gated.get_column("entry").to_list() == [True, False, True, True]
    assert result.suppressed == 1


def test_the_first_safe_day_is_allowed() -> None:
    # With a 31-day cooldown, an entry exactly 31 days after the loss is allowed —
    # that is the first day a repurchase is outside the statutory 30-day window.
    signals = pl.DataFrame({"symbol": ["A"], "date": [date(2026, 2, 5)], "entry": [True]})
    result = suppress_after_losses(signals, LOSSES, cooldown_days=31)
    assert result.signals.get_column("entry").to_list() == [True]


def test_the_day_before_is_not() -> None:
    signals = pl.DataFrame({"symbol": ["A"], "date": [date(2026, 2, 4)], "entry": [True]})
    result = suppress_after_losses(signals, LOSSES, cooldown_days=31)
    assert result.signals.get_column("entry").to_list() == [False]


def test_a_non_entry_bar_is_never_counted_as_suppressed() -> None:
    signals = SIGNALS.with_columns(pl.lit(value=False).alias("entry"))
    result = suppress_after_losses(signals, LOSSES, cooldown_days=31)
    assert result.suppressed == 0
    assert not result.signals.get_column("blocked").any()


def test_lazy_frames_are_accepted() -> None:
    result = suppress_after_losses(SIGNALS.lazy(), LOSSES.lazy(), cooldown_days=31)
    assert result.suppressed == 1


def test_a_negative_cooldown_is_rejected() -> None:
    with pytest.raises(ValueError, match="cooldown_days must be non-negative"):
        suppress_after_losses(SIGNALS, LOSSES, cooldown_days=-1)


def test_a_missing_signal_column_names_itself() -> None:
    with pytest.raises(ValueError, match=r"signals is missing column\(s\) \['entry'\]"):
        suppress_after_losses(SIGNALS.drop("entry"), LOSSES)


def test_a_missing_loss_column_names_itself() -> None:
    with pytest.raises(ValueError, match=r"losses is missing column\(s\) \['loss_date'\]"):
        suppress_after_losses(SIGNALS, LOSSES.rename({"loss_date": "when"}))


def test_custom_column_names_are_honoured() -> None:
    signals = SIGNALS.rename({"symbol": "ticker", "date": "bar", "entry": "go_long"})
    losses = LOSSES.rename({"symbol": "ticker", "loss_date": "sold_at"})
    result = suppress_after_losses(
        signals,
        losses,
        cooldown_days=31,
        symbol_col="ticker",
        date_col="bar",
        entry_col="go_long",
        loss_date_col="sold_at",
    )
    assert result.suppressed == 1


def test_taxed_trade_reports_its_taxable_amount() -> None:
    trade = Trade("A", date(2026, 1, 1), date(2026, 2, 1), 1100.0, 1000.0)
    taxed = TaxedTrade(
        trade=trade,
        allowed_loss=0.0,
        disallowed_loss=0.0,
        gain=100.0,
        is_wash_sale=False,
        long_term=False,
        adjusted_basis=1000.0,
        tacked_days=0,
    )
    assert taxed.taxable == pytest.approx(100.0)
