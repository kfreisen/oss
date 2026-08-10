"""Trade records, and what the engine needs to know about them.

The source implementations all operated on a backtester's internal ledger, keyed
by bar index into a price frame. That is a backtest artifact, not a ledger, and it
makes the accounting unusable anywhere else. `washbook` takes closed trades with
real dates and real dollars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

__all__ = ["ChainScope", "HoldingPeriod", "TaxedTrade", "Trade"]


class ChainScope(StrEnum):
    """How widely a replacement purchase is looked for.

    §1091 asks whether *you* acquired substantially identical stock, without
    regard to which strategy or sub-account did the acquiring. Narrower scopes are
    approximations that **understate** wash sales, and are offered because a
    backtest comparing strategies in isolation often wants them.
    """

    SYMBOL = "symbol"
    """Per symbol, across everything. The faithful reading of §1091."""

    SYMBOL_AND_ACCOUNT = "symbol_and_account"
    """Per symbol within an account. Understates: a repurchase in another account
    of yours is still a wash sale, and in an IRA it is worse than one."""


class HoldingPeriod(StrEnum):
    """What "held for more than a year" is measured in."""

    CALENDAR_DAYS = "calendar_days"
    """Calendar days, which is what §1222 actually says."""

    BARS = "bars"
    """Bars held. Only meaningful for a backtest on regular bars, and wrong for
    daily bars over a year because weekends and holidays are not bars — roughly
    252 bars to the year, not 365."""


@dataclass(frozen=True, slots=True)
class Trade:
    """One closed round-trip position.

    Attributes:
        symbol: Instrument identifier. Matching is exact — see the README on what
            "substantially identical" would require and why this does not attempt it.
        opened: Acquisition date.
        closed: Disposal date.
        proceeds: What the disposal brought in, net of fees.
        basis: What the position cost, net of fees.
        quantity: Shares. Carried for reporting and for partial-replacement
            arithmetic; a trade is otherwise treated as one indivisible lot.
        account: Optional account label, used by
            [`ChainScope.SYMBOL_AND_ACCOUNT`][washbook.ledger.ChainScope].
        bars_held: Bars the position was open. Only needed for
            [`HoldingPeriod.BARS`][washbook.ledger.HoldingPeriod].
    """

    symbol: str
    opened: date
    closed: date
    proceeds: float
    basis: float
    quantity: float = 1.0
    account: str | None = None
    bars_held: int = 0

    def __post_init__(self) -> None:
        """Reject a trade that cannot be accounted for."""
        if self.closed < self.opened:
            msg = f"{self.symbol}: closed {self.closed} precedes opened {self.opened}"
            raise ValueError(msg)
        if self.quantity <= 0:
            msg = f"{self.symbol}: quantity must be positive, got {self.quantity}"
            raise ValueError(msg)

    @property
    def pnl(self) -> float:
        """Economic gain or loss, before any tax treatment."""
        return self.proceeds - self.basis

    @property
    def days_held(self) -> int:
        """Calendar days between acquisition and disposal."""
        return (self.closed - self.opened).days


@dataclass(frozen=True, slots=True)
class TaxedTrade:
    """A trade after §1091 treatment.

    Attributes:
        trade: The trade as supplied.
        allowed_loss: Loss deductible this year. Zero for a gain, and zero for a
            loss fully disallowed as a wash sale.
        disallowed_loss: Loss disallowed and rolled into the replacement's basis.
            Negative or zero.
        gain: Taxable gain, zero for a loss.
        is_wash_sale: Whether this disposal was a wash sale.
        long_term: Whether the (possibly tacked) holding period exceeded a year.
        adjusted_basis: The trade's basis after absorbing any inherited
            disallowed loss from the chain.
        tacked_days: Holding period inherited from washed predecessors, per
            §1223(3).
    """

    trade: Trade
    allowed_loss: float
    disallowed_loss: float
    gain: float
    is_wash_sale: bool
    long_term: bool
    adjusted_basis: float
    tacked_days: int

    @property
    def taxable(self) -> float:
        """Signed amount entering the year's tax computation."""
        return self.gain + self.allowed_loss
