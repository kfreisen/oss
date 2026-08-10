"""Adjusting a base rate by named factors, under hard caps.

The problem this solves is narrow and increasingly common: you want a language
model's judgement in a probability estimate, and you cannot have it produce the
probability.

Ask a model for "the probability this succeeds" and you get a number with no audit
trail, no bound on how far it can move, and no way to tell a considered 0.7 from a
fluent one. Ask it instead for *named factors with signed weights* and let code
enforce the limits, and you get an estimate that can be read back as an argument:
base rate was 0.35, three factors moved it, none by more than 0.10, total movement
capped at 0.25.

The caps are the mechanism. Nothing here validates that a factor is *sensible* —
that is not checkable. What it guarantees is that no single factor, no category of
factors, and no combination can move the estimate further than you allowed. The
worst case is bounded even when the reasoning is nonsense.

None of this is specific to language models. It is the same discipline you would
want from a committee of analysts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Adjustment", "Bridge", "Factor"]


@dataclass(frozen=True, slots=True)
class Factor:
    """One named reason to move away from the base rate.

    Attributes:
        name: What this factor is. Appears in the audit trail.
        delta: Signed adjustment in probability points. Positive raises.
        category: Optional grouping, so related factors can be capped together —
            five separate factors all restating "management is good" should not
            compound into a large move.
        rationale: Free text, carried through for the record.
    """

    name: str
    delta: float
    category: str | None = None
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class Adjustment:
    """The result, with everything needed to argue with it.

    Attributes:
        base_rate: Where it started.
        probability: Where it ended, after caps and clamping.
        applied: Each factor's delta after per-factor and per-category capping.
        total_adjustment: Net movement actually applied.
        capped: Names of factors, categories, or `"total"` that hit a limit. An
            empty tuple means nothing was constrained; a long one means the
            estimate is being held back and the reasoning deserves a look.
    """

    base_rate: float
    probability: float
    applied: tuple[tuple[str, float], ...]
    total_adjustment: float
    capped: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Bridge:
    """Applies factors to a base rate under caps.

    Attributes:
        max_per_factor: Largest absolute move any single factor may contribute.
        max_per_category: Largest absolute net move any one category may
            contribute, after per-factor capping.
        max_total: Largest absolute net move overall.
        floor: Lowest probability that may be returned.
        ceiling: Highest probability that may be returned. Kept strictly inside
            `[0, 1]` by default: an estimate of exactly 0 or 1 asserts certainty,
            which this procedure is in no position to establish.
    """

    max_per_factor: float = 0.10
    max_per_category: float = 0.15
    max_total: float = 0.25
    floor: float = 0.01
    ceiling: float = 0.99

    def __post_init__(self) -> None:
        """Reject caps that cannot mean anything."""
        for name in ("max_per_factor", "max_per_category", "max_total"):
            value = getattr(self, name)
            if value < 0:
                msg = f"{name} must be non-negative, got {value}"
                raise ValueError(msg)
        if not 0.0 <= self.floor <= self.ceiling <= 1.0:
            msg = (
                f"floor and ceiling must satisfy 0 <= floor <= ceiling <= 1, "
                f"got floor={self.floor}, ceiling={self.ceiling}"
            )
            raise ValueError(msg)

    def apply(self, base_rate: float, factors: list[Factor]) -> Adjustment:
        """Adjust `base_rate` by `factors`, enforcing every cap.

        Args:
            base_rate: Starting probability, in `[0, 1]`.
            factors: Proposed adjustments.

        Returns:
            An [`Adjustment`][impliedmove.bridge.Adjustment] recording what was
            applied and what was capped.

        Raises:
            ValueError: If `base_rate` is outside `[0, 1]`.
        """
        if not 0.0 <= base_rate <= 1.0:
            msg = f"base_rate must be in [0, 1], got {base_rate}"
            raise ValueError(msg)

        capped: list[str] = []

        # 1. Cap each factor individually.
        per_factor: list[tuple[str, float, str | None]] = []
        for factor in factors:
            limited = _clamp(factor.delta, -self.max_per_factor, self.max_per_factor)
            if limited != factor.delta:
                capped.append(factor.name)
            per_factor.append((factor.name, limited, factor.category))

        # 2. Cap each category's net contribution, scaling its members
        #    proportionally so the audit trail still adds up to what was applied.
        by_category: dict[str, float] = {}
        for _name, delta, category in per_factor:
            if category is not None:
                by_category[category] = by_category.get(category, 0.0) + delta

        scale = {
            category: self.max_per_category / abs(total)
            for category, total in by_category.items()
            if abs(total) > self.max_per_category
        }
        capped.extend(scale)

        applied = [
            (name, delta * scale.get(category, 1.0) if category else delta)
            for name, delta, category in per_factor
        ]

        # 3. Cap the total, again scaling proportionally.
        raw_total = sum(delta for _, delta in applied)
        total = _clamp(raw_total, -self.max_total, self.max_total)
        if total != raw_total:
            capped.append("total")
            factor_scale = total / raw_total if raw_total else 0.0
            applied = [(name, delta * factor_scale) for name, delta in applied]

        probability = _clamp(base_rate + total, self.floor, self.ceiling)

        return Adjustment(
            base_rate=base_rate,
            probability=probability,
            applied=tuple(applied),
            total_adjustment=total,
            capped=tuple(capped),
        )


def _clamp(value: float, low: float, high: float) -> float:
    """Constrain `value` to `[low, high]`."""
    return max(low, min(high, value))
