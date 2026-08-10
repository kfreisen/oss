"""After-tax backtest accounting: wash-sale blackout and IRS §1091 deferral.

Two models of the same rule, answering different questions:

- **Deferral** ([`apply_wash_sales`][washbook.defer.apply_wash_sales]) — trade
  freely and account for §1091 as a taxable account actually experiences it.
  Disallowed losses roll into the replacement's basis, holding periods tack, and
  chains still open at the end are reported rather than quietly forgiven.
- **Cooldown** ([`apply_cooldown`][washbook.cooldown.apply_cooldown]) — refuse to
  re-enter within the window so no wash sale ever occurs. This changes the equity
  curve; the cost shows up as suppressed entries.

Neither is tax advice, and the boundaries of what is modelled are documented on
each function rather than left to be discovered.
"""

from __future__ import annotations

from washbook.defer import DeferralResult, WashSaleConfig, apply_wash_sales
from washbook.ledger import ChainScope, HoldingPeriod, TaxedTrade, Trade

__all__ = [
    "ChainScope",
    "DeferralResult",
    "HoldingPeriod",
    "TaxedTrade",
    "Trade",
    "WashSaleConfig",
    "__version__",
    "apply_wash_sales",
]

__version__ = "0.0.1.dev0"
