"""Was the probability any good?

An uncalibrated probability is a number with a percent sign. These are the cheapest
checks that turn a log of forecasts and outcomes into an answer.

Three of them, because they fail differently:

**Brier score** is the mean squared error of the forecast. It rewards being both
calibrated and confident, and it is a single number, which makes it good for
tracking and useless for diagnosing.

**Reliability** buckets forecasts and compares the mean prediction in each bucket
against the observed frequency. This is where you see *how* a forecaster is wrong —
overconfident at the extremes, hedging toward the middle, or fine.

**Directional hit rate** ignores magnitude and asks whether the forecast landed on
the correct side of a reference. A forecaster can be badly calibrated and still
profitable if it gets the direction right, and well calibrated and useless if it
does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Bucket", "Reliability", "brier_score", "directional_hit_rate", "reliability"]


@dataclass(frozen=True, slots=True)
class Bucket:
    """One band of the reliability curve.

    Attributes:
        low: Lower edge of the band, inclusive.
        high: Upper edge, inclusive only for the topmost band.
        count: Forecasts falling in it.
        mean_prediction: Their average forecast.
        observed_rate: The fraction that actually happened.
    """

    low: float
    high: float
    count: int
    mean_prediction: float
    observed_rate: float

    @property
    def gap(self) -> float:
        """Predicted minus observed. Positive means overconfident in this band."""
        return self.mean_prediction - self.observed_rate


@dataclass(frozen=True, slots=True)
class Reliability:
    """A reliability curve and its summary.

    Attributes:
        buckets: Non-empty bands, in ascending order.
        max_gap: Largest absolute gap across bands, weighted by nothing — the
            worst single band.
        weighted_gap: Mean absolute gap weighted by bucket population, which is
            the honest headline: a terrible band holding two forecasts matters
            less than a mediocre one holding two hundred.
    """

    buckets: tuple[Bucket, ...]
    max_gap: float
    weighted_gap: float


def _validate(predictions: Sequence[float], outcomes: Sequence[bool]) -> None:
    """Reject inputs that cannot be scored."""
    if len(predictions) != len(outcomes):
        msg = f"got {len(predictions)} predictions and {len(outcomes)} outcomes"
        raise ValueError(msg)
    if not predictions:
        msg = "no forecasts to score"
        raise ValueError(msg)
    for p in predictions:
        if not 0.0 <= p <= 1.0:
            msg = f"predictions must be in [0, 1], got {p}"
            raise ValueError(msg)


def brier_score(predictions: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Mean squared error of the forecasts. Lower is better.

    For orientation: always saying 0.5 scores 0.25. A forecaster scoring worse than
    the base rate of the outcomes is worse than a constant, which is worth knowing
    before tuning anything.

    Raises:
        ValueError: If the inputs are empty, mismatched, or out of range.
    """
    _validate(predictions, outcomes)
    return sum((p - float(o)) ** 2 for p, o in zip(predictions, outcomes, strict=True)) / len(
        predictions
    )


def reliability(
    predictions: Sequence[float], outcomes: Sequence[bool], *, n_buckets: int = 10
) -> Reliability:
    """Bucket forecasts and compare predicted against observed frequency.

    Empty buckets are dropped rather than reported as zero — a band nobody
    forecast into says nothing, and including it drags any summary toward a
    meaningless number.

    Args:
        predictions: Forecast probabilities.
        outcomes: What happened.
        n_buckets: How many equal-width bands to divide `[0, 1]` into.

    Raises:
        ValueError: If the inputs are unusable or `n_buckets` is below 1.
    """
    _validate(predictions, outcomes)
    if n_buckets < 1:
        msg = f"n_buckets must be at least 1, got {n_buckets}"
        raise ValueError(msg)

    width = 1.0 / n_buckets
    grouped: dict[int, list[tuple[float, bool]]] = {}
    for p, o in zip(predictions, outcomes, strict=True):
        # min() keeps a forecast of exactly 1.0 in the top bucket rather than
        # creating an extra one-wide band above it.
        index = min(int(p / width), n_buckets - 1)
        grouped.setdefault(index, []).append((p, o))

    buckets: list[Bucket] = []
    for index in sorted(grouped):
        entries = grouped[index]
        buckets.append(
            Bucket(
                low=index * width,
                high=(index + 1) * width,
                count=len(entries),
                mean_prediction=sum(p for p, _ in entries) / len(entries),
                observed_rate=sum(float(o) for _, o in entries) / len(entries),
            )
        )

    total = sum(b.count for b in buckets)
    return Reliability(
        buckets=tuple(buckets),
        max_gap=max(abs(b.gap) for b in buckets),
        weighted_gap=sum(abs(b.gap) * b.count for b in buckets) / total,
    )


def directional_hit_rate(
    predictions: Sequence[float],
    outcomes: Sequence[bool],
    *,
    reference: float = 0.5,
) -> float:
    """Fraction of forecasts that landed on the correct side of `reference`.

    Forecasts exactly at the reference express no direction and are excluded. If
    every forecast sits at the reference there is nothing to score, and this
    returns 0.0 rather than dividing by zero.

    Args:
        predictions: Forecast probabilities.
        outcomes: What happened.
        reference: The dividing line — a market price, or 0.5 for a bare
            better-than-even call.
    """
    _validate(predictions, outcomes)
    directional = [(p, o) for p, o in zip(predictions, outcomes, strict=True) if p != reference]
    if not directional:
        return 0.0
    hits = sum(1 for p, o in directional if (p > reference) == o)
    return hits / len(directional)
