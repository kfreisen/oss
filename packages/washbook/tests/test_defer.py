"""§1091 deferral.

Rates are deliberately absent here — this layer decides *what is deductible when*,
not what it is worth. Tax rates live in `netting.py`.
"""

from __future__ import annotations

from datetime import date

import pytest
from washbook import Trade, apply_wash_sales
from washbook.defer import WashSaleConfig
from washbook.ledger import ChainScope, HoldingPeriod


def trade(
    opened: str,
    closed: str,
    proceeds: float,
    basis: float = 1000.0,
    *,
    symbol: str = "AAA",
    account: str | None = None,
    bars: int = 0,
) -> Trade:
    """A trade written compactly, dates as ISO strings."""
    return Trade(
        symbol=symbol,
        opened=date.fromisoformat(opened),
        closed=date.fromisoformat(closed),
        proceeds=proceeds,
        basis=basis,
        account=account,
        bars_held=bars,
    )


def test_a_lone_loss_is_deductible() -> None:
    result = apply_wash_sales([trade("2026-01-05", "2026-02-01", 900.0)])
    assert result.wash_sale_count == 0
    assert result.total_allowed_loss == pytest.approx(-100.0)
    assert result.deferred_at_end == 0.0
    result.check_invariant()


def test_a_gain_is_never_a_wash_sale() -> None:
    # §1091 disallows *losses*. A gain repurchased the next day is just a gain.
    result = apply_wash_sales(
        [trade("2026-01-05", "2026-02-01", 1100.0), trade("2026-02-02", "2026-03-01", 1200.0)]
    )
    assert result.wash_sale_count == 0
    assert result.total_gain == pytest.approx(300.0)
    result.check_invariant()


def test_a_repurchase_inside_the_window_washes_the_loss() -> None:
    result = apply_wash_sales(
        [
            trade("2026-01-05", "2026-02-01", 900.0),  # -100, washed
            trade("2026-02-10", "2026-06-01", 1300.0),  # repurchase, +300 before basis
        ]
    )
    assert result.wash_sale_count == 1
    assert result.total_allowed_loss == 0.0
    # The disallowed 100 raises the replacement's basis, so the gain is 300 - 100.
    assert result.total_gain == pytest.approx(200.0)
    result.check_invariant()


def test_a_repurchase_outside_the_window_does_not() -> None:
    result = apply_wash_sales(
        [trade("2026-01-05", "2026-02-01", 900.0), trade("2026-04-01", "2026-06-01", 1300.0)]
    )
    assert result.wash_sale_count == 0
    assert result.total_allowed_loss == pytest.approx(-100.0)
    assert result.total_gain == pytest.approx(300.0)
    result.check_invariant()


def test_the_window_is_symmetric() -> None:
    """A purchase *before* the loss sale is a wash sale too.

    This is the case every implementation this package was extracted from got
    wrong, and it is not an edge case: scaling into a position and then selling
    the older lot at a loss is an ordinary thing to do.
    """
    # Second position opened 2026-01-20, first sold at a loss 2026-02-01 — the
    # acquisition precedes the disposal by 12 days, well inside 30.
    trades = [
        trade("2025-11-01", "2026-02-01", 900.0),
        trade("2026-01-20", "2026-09-01", 1300.0),
    ]
    assert apply_wash_sales(trades).wash_sale_count == 1

    forward_only = WashSaleConfig(symmetric=False)
    assert apply_wash_sales(trades, config=forward_only).wash_sale_count == 0


def test_the_window_boundary_is_inclusive() -> None:
    inside = apply_wash_sales(
        [trade("2026-01-01", "2026-02-01", 900.0), trade("2026-03-03", "2026-06-01", 1000.0)]
    )
    outside = apply_wash_sales(
        [trade("2026-01-01", "2026-02-01", 900.0), trade("2026-03-04", "2026-06-01", 1000.0)]
    )
    # 2026-03-03 is exactly 30 days after 2026-02-01.
    assert inside.wash_sale_count == 1
    assert outside.wash_sale_count == 0


def test_a_trade_does_not_wash_against_its_own_purchase() -> None:
    """A three-week round trip at a loss is deductible.

    Its own acquisition is inside its own window, so without an explicit exclusion
    every short losing trade would report as a wash sale — which would be both
    wrong and very loud.
    """
    result = apply_wash_sales([trade("2026-01-05", "2026-01-20", 900.0)])
    assert result.wash_sale_count == 0


def test_two_positions_opened_the_same_day_do_replace_each_other() -> None:
    # The exclusion above must not swallow a genuine duplicate.
    result = apply_wash_sales(
        [
            trade("2026-01-05", "2026-01-20", 900.0),
            trade("2026-01-05", "2026-09-01", 1200.0),
        ]
    )
    assert result.wash_sale_count == 1


def test_chains_roll_through_several_replacements() -> None:
    result = apply_wash_sales(
        [
            trade("2026-01-01", "2026-02-01", 900.0),  # -100 washed
            trade("2026-02-05", "2026-03-01", 900.0),  # -100 more, washed
            trade("2026-03-05", "2026-11-01", 1400.0),  # finally sold
        ]
    )
    assert result.wash_sale_count == 2
    # Basis of the last leg is 1000 + 100 + 100; proceeds 1400 → gain 200.
    assert result.total_gain == pytest.approx(200.0)
    result.check_invariant()


def test_holding_period_tacks_onto_the_replacement() -> None:
    """§1223(3): the washed position's holding period carries forward.

    Without tacking the replacement below is short-term. With it, the eleven
    months already served push the combined period past a year.
    """
    result = apply_wash_sales(
        [
            trade("2025-01-10", "2025-12-15", 900.0),  # ~11 months, loss, washed
            trade("2025-12-20", "2026-03-01", 1300.0),  # ~2 months alone
        ]
    )
    assert result.wash_sale_count == 1
    assert result.trades[1].long_term


def test_without_tacking_the_replacement_would_be_short_term() -> None:
    # The same second leg, with no washed predecessor to inherit from.
    result = apply_wash_sales([trade("2025-12-20", "2026-03-01", 1300.0)])
    assert not result.trades[0].long_term


def test_a_deferral_open_at_ledger_end_is_reported() -> None:
    """A loss washed by a position that never closes is never deducted.

    Economically suffered, tax credit never received inside the window studied.
    Reporting it is the difference between an honest tax-drag number and a
    flattering one.
    """
    result = apply_wash_sales(
        [trade("2026-01-01", "2026-02-01", 900.0)],
        open_positions=[("AAA", date(2026, 2, 10))],
    )
    assert result.wash_sale_count == 1
    assert result.deferred_at_end == pytest.approx(100.0)
    assert result.total_allowed_loss == 0.0
    result.check_invariant()


def test_an_open_position_outside_the_window_does_not_wash() -> None:
    result = apply_wash_sales(
        [trade("2026-01-01", "2026-02-01", 900.0)],
        open_positions=[("AAA", date(2026, 6, 1))],
    )
    assert result.wash_sale_count == 0
    assert result.total_allowed_loss == pytest.approx(-100.0)


def test_different_symbols_do_not_interact() -> None:
    result = apply_wash_sales(
        [
            trade("2026-01-01", "2026-02-01", 900.0, symbol="AAA"),
            trade("2026-02-05", "2026-06-01", 1300.0, symbol="BBB"),
        ]
    )
    assert result.wash_sale_count == 0
    result.check_invariant()


def test_account_scope_narrows_the_chain() -> None:
    """Per-account scope understates, and is documented as doing so."""
    trades = [
        trade("2026-01-01", "2026-02-01", 900.0, account="taxable"),
        trade("2026-02-05", "2026-06-01", 1300.0, account="ira"),
    ]
    assert apply_wash_sales(trades).wash_sale_count == 1
    narrow = WashSaleConfig(scope=ChainScope.SYMBOL_AND_ACCOUNT)
    assert apply_wash_sales(trades, config=narrow).wash_sale_count == 0


def test_bars_can_stand_in_for_days() -> None:
    config = WashSaleConfig(holding_period=HoldingPeriod.BARS, long_term_after=252)
    result = apply_wash_sales([trade("2025-01-02", "2026-01-02", 1300.0, bars=253)], config=config)
    assert result.trades[0].long_term


def test_input_order_is_preserved_in_the_output() -> None:
    # Chains are processed independently and internally sorted; the caller should
    # not have to care.
    trades = [
        trade("2026-03-01", "2026-04-01", 1100.0, symbol="BBB"),
        trade("2026-01-01", "2026-02-01", 900.0, symbol="AAA"),
    ]
    result = apply_wash_sales(trades)
    assert [t.trade.symbol for t in result.trades] == ["BBB", "AAA"]


def test_an_empty_ledger_is_not_an_error() -> None:
    result = apply_wash_sales([])
    assert result.trades == ()
    assert result.wash_sale_count == 0
    result.check_invariant()


def test_unsorted_input_is_handled() -> None:
    ordered = apply_wash_sales(
        [trade("2026-01-01", "2026-02-01", 900.0), trade("2026-02-05", "2026-06-01", 1300.0)]
    )
    reversed_input = apply_wash_sales(
        [trade("2026-02-05", "2026-06-01", 1300.0), trade("2026-01-01", "2026-02-01", 900.0)]
    )
    assert ordered.wash_sale_count == reversed_input.wash_sale_count == 1


def test_a_zero_window_disables_the_rule() -> None:
    config = WashSaleConfig(window_days=0)
    result = apply_wash_sales(
        [trade("2026-01-01", "2026-02-01", 900.0), trade("2026-02-02", "2026-06-01", 1300.0)],
        config=config,
    )
    assert result.wash_sale_count == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window_days": -1}, "window_days must be non-negative"),
        ({"long_term_after": -1}, "long_term_after must be non-negative"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        WashSaleConfig(**kwargs)  # type: ignore[arg-type]


def test_a_trade_closing_before_it_opens_is_rejected() -> None:
    with pytest.raises(ValueError, match="precedes opened"):
        Trade("AAA", date(2026, 2, 1), date(2026, 1, 1), 1.0, 1.0)


def test_a_non_positive_quantity_is_rejected() -> None:
    with pytest.raises(ValueError, match="quantity must be positive"):
        Trade("AAA", date(2026, 1, 1), date(2026, 2, 1), 1.0, 1.0, quantity=0)


def test_the_invariant_catches_a_broken_result() -> None:
    from washbook.defer import DeferralResult

    good = apply_wash_sales([trade("2026-01-05", "2026-02-01", 900.0)])
    broken = DeferralResult(
        trades=good.trades,
        wash_sale_count=0,
        deferred_at_end=0.0,
        total_allowed_loss=0.0,  # the loss has been dropped on the floor
        total_gain=0.0,
    )
    with pytest.raises(AssertionError, match="does not balance"):
        broken.check_invariant()
