"""The baseline this package is measured against: don't adjust at all.

Every other package here publishes a speedup. This one cannot honestly: it computes
closed-form arithmetic in microseconds, and a "10x faster than what?" number would
be theatre.

What it can be measured on is **accuracy**. The bounded-factor bridge exists to
improve on a base rate by applying capped adjustments, so the question worth
answering is whether the adjusted estimate is better calibrated than the base rate
it started from. If it is not, the whole apparatus is decoration.

The comparison needs a third leg to be meaningful, so there are two baselines:

- `unadjusted_prior` — always predict the base rate. The thing to beat.
- `uncapped_adjustment` — apply the same factors with the caps removed. This is
  what you get by trusting a proposer directly, and it is the reason the caps
  exist: it wins when the factors are good and loses badly when they are not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from impliedmove.bridge import Factor


def unadjusted_prior(base_rate: float, _factors: Sequence[Factor]) -> float:
    """Ignore every factor and predict the base rate."""
    return base_rate


def uncapped_adjustment(base_rate: float, factors: Sequence[Factor]) -> float:
    """Apply every factor at face value, clamped only to stay a probability.

    No per-factor cap, no per-category cap, no total cap — the estimate a proposer
    produces when nothing constrains it.
    """
    total = sum(f.delta for f in factors)
    return max(0.0, min(1.0, base_rate + total))
