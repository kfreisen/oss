"""IRS §1091 wash-sale deferral.

Sell at a loss, buy substantially identical stock within the window, and the loss
is *disallowed* — not forgiven. It rolls into the replacement's cost basis, and the
old holding period tacks onto the new one (§1223(3)). Chains roll: a replacement
that is itself sold at a loss into another replacement carries everything forward.
A chain still open when the ledger ends is a deduction the taxpayer economically
suffered and never got.

## What this implements, and where it stops

**Implemented.** Loss disallowance, basis adjustment into the replacement, holding
period tacking, chains of arbitrary length, the deferral outstanding at ledger end,
and the symmetric 61-day window.

**The window is symmetric**, which is the thing most implementations get wrong. The
rule is 30 days *before* through 30 days *after* the loss sale — buy first, sell at
a loss second, and it is still a wash sale. Every implementation this package was
extracted from looked only forward, which understates wash sales for any strategy
that scales into a position.

**Not implemented, deliberately.** "Substantially identical" is decided by exact
symbol match: no ETF-for-ETF equivalence, no options, no convertibles. Positions
are whole — no partial-lot pro-rata disallowance, no FIFO/LIFO/specific-ID
selection. No IRA repurchase rule (Rev. Rul. 2008-5), no spouse or controlled-entity
attribution, no §1091(e) short-sale specialization, no $3,000 ordinary-income
limit. This is a tool for measuring tax drag in a backtest. It is not tax advice
and it will not produce a correct Form 8949.

## Reconciling the sources

Four near-duplicate implementations were merged to produce this, and they disagreed.
Each disagreement was resolved rather than parameterized away where one was simply
right:

| Divergence | Resolution |
| --- | --- |
| Holding period in bars vs calendar days | Calendar days by default — §1222 says days. Bars remain available, and are wrong for daily bars over a year (≈252 bars, not 365). |
| Open position at ledger end handled / ignored | Handled. §1091 triggers on *acquisition*, so a position still open is a replacement. Ignoring it silently overstates deductions. |
| Chain scope per-symbol vs per-(strategy, symbol) | Per symbol by default. Narrower scope understates wash sales; it stays available because comparing strategies in isolation sometimes wants it. |
| Washed trade dropped vs kept with a flag | Kept and flagged, so that realized + deferred = economic, which is the invariant worth testing. |
| Forward-only 30-day window | Replaced by the symmetric 61-day window. |
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from washbook.ledger import ChainScope, HoldingPeriod, TaxedTrade, Trade

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import date

__all__ = ["DeferralResult", "WashSaleConfig", "apply_wash_sales"]

# §1091: 30 days before through 30 days after, inclusive of the sale date itself.
WASH_WINDOW_DAYS = 30
# §1222: "more than one year".
LONG_TERM_DAYS = 365


@dataclass(frozen=True, slots=True)
class WashSaleConfig:
    """How to apply the rule.

    Attributes:
        window_days: Days either side of the disposal in which an acquisition
            counts as a replacement. The statutory value is 30, giving a 61-day
            window in total.
        symmetric: Whether acquisitions *before* the loss sale count. True is the
            statute. False reproduces the forward-only approximation the source
            implementations used, and understates wash sales.
        holding_period: Whether long-term status is decided in calendar days or in
            bars.
        long_term_after: Threshold for long-term treatment, in whichever unit
            `holding_period` names.
        scope: How widely to look for a replacement.
    """

    window_days: int = WASH_WINDOW_DAYS
    symmetric: bool = True
    holding_period: HoldingPeriod = HoldingPeriod.CALENDAR_DAYS
    long_term_after: int = LONG_TERM_DAYS
    scope: ChainScope = ChainScope.SYMBOL

    def __post_init__(self) -> None:
        """Reject a configuration that cannot mean anything."""
        if self.window_days < 0:
            msg = f"window_days must be non-negative, got {self.window_days}"
            raise ValueError(msg)
        if self.long_term_after < 0:
            msg = f"long_term_after must be non-negative, got {self.long_term_after}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DeferralResult:
    """The outcome of applying §1091 to a ledger.

    Attributes:
        trades: Every input trade, in input order, with its tax treatment.
        wash_sale_count: How many disposals were wash sales.
        deferred_at_end: Loss still deferred when the ledger ended — economically
            suffered, never deducted inside the window studied. Reported as a
            positive number.
        total_allowed_loss: Losses deducted, as a negative number.
        total_gain: Gains realized.
    """

    trades: tuple[TaxedTrade, ...]
    wash_sale_count: int
    deferred_at_end: float
    total_allowed_loss: float
    total_gain: float

    @property
    def total_economic_pnl(self) -> float:
        """Economic result, ignoring tax treatment entirely."""
        return sum(t.trade.pnl for t in self.trades)

    def check_invariant(self, tolerance: float = 1e-6) -> None:
        """Assert that no economic result was created or destroyed.

        Wash-sale treatment moves *when* a loss is deducted, never whether it
        happened. So allowed losses plus gains plus what is still deferred must
        equal the economic result. Every one of the four source implementations
        could have been checked against this, and the one that dropped washed rows
        entirely would have failed it.

        Raises:
            AssertionError: If the books do not balance.
        """
        accounted = self.total_allowed_loss + self.total_gain - self.deferred_at_end
        difference = abs(accounted - self.total_economic_pnl)
        if difference > tolerance:
            msg = (
                f"wash-sale accounting does not balance: allowed {self.total_allowed_loss:.6f} "
                f"+ gains {self.total_gain:.6f} - deferred {self.deferred_at_end:.6f} "
                f"= {accounted:.6f}, but economic P&L is "
                f"{self.total_economic_pnl:.6f} (difference {difference:.6g})"
            )
            raise AssertionError(msg)


def _chain_key(trade: Trade, scope: ChainScope) -> tuple[str, ...]:
    """The identity within which replacements are looked for."""
    if scope is ChainScope.SYMBOL_AND_ACCOUNT:
        return (trade.symbol, trade.account or "")
    return (trade.symbol,)


def apply_wash_sales(
    trades: Iterable[Trade],
    *,
    config: WashSaleConfig | None = None,
    open_positions: Sequence[tuple[str, date]] = (),
) -> DeferralResult:
    """Apply §1091 to a ledger of closed trades.

    Args:
        trades: Closed round-trips, in any order. They are grouped and sorted
            internally, so a ledger straight out of a backtester is fine.
        config: How to apply the rule. Defaults to the statute.
        open_positions: `(symbol, acquired_date)` for positions still open when the
            ledger ends. These are replacements too — §1091 triggers on acquisition,
            not on the replacement's eventual disposal — and omitting them
            overstates deductions for any strategy still holding at the end.

    Returns:
        A [`DeferralResult`][washbook.defer.DeferralResult] whose `trades` are in
        the same order as the input.

    Example:
        ```python
        from datetime import date
        from washbook import Trade, apply_wash_sales

        result = apply_wash_sales(
            [
                Trade("AAPL", date(2026, 1, 5), date(2026, 2, 1), 900.0, 1000.0),
                Trade("AAPL", date(2026, 2, 10), date(2026, 6, 1), 1300.0, 1200.0),
            ]
        )
        result.wash_sale_count  # 1 — the February repurchase is inside the window
        ```
    """
    config = config or WashSaleConfig()
    ledger = list(trades)
    if not ledger:
        return DeferralResult((), 0, 0.0, 0.0, 0.0)

    # Acquisition dates per chain, sorted, so "was anything bought near this
    # disposal" is a binary search rather than a scan. This is what keeps the
    # symmetric window from costing a second pass over the whole ledger per trade.
    acquisitions: dict[tuple[str, ...], list[date]] = {}
    for trade in ledger:
        acquisitions.setdefault(_chain_key(trade, config.scope), []).append(trade.opened)
    for symbol, opened in open_positions:
        # An open position has no account attached, so under account scope it joins
        # the unlabelled chain — the same one a Trade with account=None lands in.
        open_key: tuple[str, ...] = (symbol,) if config.scope is ChainScope.SYMBOL else (symbol, "")
        acquisitions.setdefault(open_key, []).append(opened)
    for dates in acquisitions.values():
        dates.sort()

    # Group and order within each chain. Chains are independent, so this is the
    # only ordering that matters; input order is restored at the end.
    order: dict[tuple[str, ...], list[int]] = {}
    for index, trade in enumerate(ledger):
        order.setdefault(_chain_key(trade, config.scope), []).append(index)
    for indices in order.values():
        indices.sort(key=lambda i: (ledger[i].closed, ledger[i].opened, i))

    taxed: list[TaxedTrade | None] = [None] * len(ledger)
    wash_count = 0
    deferred_at_end = 0.0

    for key, indices in order.items():
        chain_dates = acquisitions[key]
        # Disallowed loss inherited by the next disposal in this chain, and the
        # holding period tacked along with it.
        carry_loss = 0.0
        carry_days = 0

        for position, index in enumerate(indices):
            trade = ledger[index]
            # A disallowed loss *raises* the replacement's basis, which is what
            # defers the deduction rather than destroying it: the larger basis
            # shrinks a future gain or deepens a future loss by the same amount.
            adjusted_basis = trade.basis + carry_loss
            pnl = trade.proceeds - adjusted_basis

            if config.holding_period is HoldingPeriod.BARS:
                held = trade.bars_held + carry_days
            else:
                held = trade.days_held + carry_days
            long_term = held >= config.long_term_after

            washed = pnl < 0 and _has_replacement(chain_dates, trade, config, exclude=trade.opened)
            # A loss on the final disposal of a chain with nothing left to buy it
            # back is deductible; one that is washed by a still-open position is not.
            if washed:
                wash_count += 1
                carry_loss = -pnl
                carry_days = held
                taxed[index] = TaxedTrade(
                    trade=trade,
                    allowed_loss=0.0,
                    disallowed_loss=pnl,
                    gain=0.0,
                    is_wash_sale=True,
                    long_term=long_term,
                    adjusted_basis=adjusted_basis,
                    tacked_days=carry_days
                    - (
                        trade.bars_held
                        if config.holding_period is HoldingPeriod.BARS
                        else trade.days_held
                    ),
                )
                if position == len(indices) - 1:
                    deferred_at_end += -pnl
            else:
                taxed[index] = TaxedTrade(
                    trade=trade,
                    allowed_loss=min(pnl, 0.0),
                    disallowed_loss=0.0,
                    gain=max(pnl, 0.0),
                    is_wash_sale=False,
                    long_term=long_term,
                    adjusted_basis=adjusted_basis,
                    tacked_days=carry_days,
                )
                carry_loss = 0.0
                carry_days = 0

    settled = tuple(t for t in taxed if t is not None)
    return DeferralResult(
        trades=settled,
        wash_sale_count=wash_count,
        deferred_at_end=deferred_at_end,
        total_allowed_loss=sum(t.allowed_loss for t in settled),
        total_gain=sum(t.gain for t in settled),
    )


def _has_replacement(
    chain_dates: list[date],
    trade: Trade,
    config: WashSaleConfig,
    *,
    exclude: date,
) -> bool:
    """Whether anything in this chain was acquired inside the window.

    The disposal's own acquisition does not count as its own replacement, which is
    why `exclude` exists — without it every single trade whose holding period is
    under 30 days would wash against itself.
    """
    window = timedelta(days=config.window_days)
    start = trade.closed - window if config.symmetric else trade.closed
    end = trade.closed + window

    low = bisect_left(chain_dates, start)
    high = bisect_right(chain_dates, end)
    for acquired in chain_dates[low:high]:
        if acquired != exclude:
            return True
    # The excluded date may appear more than once — two positions opened the same
    # day — and a duplicate genuinely is a replacement.
    return chain_dates[low:high].count(exclude) > 1
