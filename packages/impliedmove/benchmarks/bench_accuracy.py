"""Benchmark: is a bounded adjustment better calibrated than no adjustment?

This package's claim is accuracy, not speed, so that is what gets measured — see
`baselines/unadjusted.py` for why a timing comparison here would be theatre.

The scenario is a synthetic forecasting log in which factors carry **real but
noisy** signal, and a minority of them are badly wrong. That is the honest setting:
if every factor were accurate, capping would only ever hurt, and if none were,
ignoring them all would win. The interesting question is what happens in between,
which is where every real proposer lives.

Three forecasters, scored by Brier (lower is better):

- `unadjusted_prior` — always the base rate.
- `uncapped_adjustment` — every factor at face value.
- `bounded_bridge` — the same factors under per-factor, per-category, and total
  caps.

Measured, at 4000 events:

    bad factors   prior      uncapped   bounded
    0%            0.22892    0.20244    0.20560
    17%           0.22892    0.25958    0.21336
    50%           0.22892    0.39591    0.24687

Read all three rows. At 17% the bridge is the best of the three and uncapped
adjustment is *worse than doing nothing* — that is the case the caps are for. At
0% the caps cost a little, because they hold back an adjustment that was correct.

And at 50%, the bridge is still better than uncapped by a wide margin and is
**worse than not adjusting at all**. Capping bounds the damage; it does not turn
bad factors into good ones. A package that published only the middle row would be
selling something.

Run with:

    make bench PKG=impliedmove
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines.unadjusted import unadjusted_prior, uncapped_adjustment
from impliedmove.bridge import Bridge, Factor
from impliedmove.calibration import brier_score, reliability

BASE_RATE = 0.35
N_EVENTS = 4000
# One factor in six is badly wrong — a proposer confidently asserting the opposite
# of the truth. Below this rate capping never earns its keep; well above it,
# nothing helps.
BAD_FACTOR_RATE = 1 / 6


def make_log(seed: int = 20260810) -> tuple[list[float], list[list[Factor]], list[bool]]:
    """A forecasting log with informative-but-noisy factors.

    Deterministic: a benchmark whose input moves cannot be compared against the
    committed result.
    """
    rng = random.Random(seed)
    base_rates: list[float] = []
    factor_sets: list[list[Factor]] = []
    outcomes: list[bool] = []

    for _ in range(N_EVENTS):
        # The true probability for this event, which the forecaster cannot see.
        true_p = min(0.95, max(0.05, rng.gauss(BASE_RATE, 0.18)))
        happened = rng.random() < true_p

        # Factors point in the direction of the truth, with noise, and sometimes
        # confidently the wrong way.
        signal = true_p - BASE_RATE
        factors: list[Factor] = []
        for k in range(3):
            if rng.random() < BAD_FACTOR_RATE:
                delta = -signal * rng.uniform(1.0, 3.0)
            else:
                delta = signal * rng.uniform(0.2, 0.7)
            factors.append(Factor(f"f{k}", delta, category="evidence"))

        base_rates.append(BASE_RATE)
        factor_sets.append(factors)
        outcomes.append(happened)

    return base_rates, factor_sets, outcomes


def score(predictions: list[float], outcomes: list[bool]) -> tuple[float, float]:
    """Brier score and population-weighted calibration gap."""
    return brier_score(predictions, outcomes), reliability(predictions, outcomes).weighted_gap


def test_bounded_adjustment_beats_both_baselines(benchmark) -> None:
    """The accuracy result, and a genuine timing of the bridge itself.

    The timing is small and honest — this is closed-form arithmetic — and is
    recorded so the benchmark harness has something in the same schema as the
    other packages. The number that matters is printed below it.
    """
    base_rates, factor_sets, outcomes = make_log()
    bridge = Bridge()

    def run_bridge() -> list[float]:
        return [
            bridge.apply(b, f).probability for b, f in zip(base_rates, factor_sets, strict=True)
        ]

    benchmark.extra_info["case"] = f"bridge/{N_EVENTS}events"
    benchmark.extra_info["impl"] = "bounded_bridge"
    benchmark.extra_info["params"] = {"events": N_EVENTS, "bad_factor_rate": BAD_FACTOR_RATE}
    bounded = benchmark(run_bridge)

    prior = [unadjusted_prior(b, f) for b, f in zip(base_rates, factor_sets, strict=True)]
    uncapped = [uncapped_adjustment(b, f) for b, f in zip(base_rates, factor_sets, strict=True)]

    bounded_brier, bounded_gap = score(bounded, outcomes)
    prior_brier, prior_gap = score(prior, outcomes)
    uncapped_brier, uncapped_gap = score(uncapped, outcomes)

    print(
        f"\nBrier (lower is better) over {N_EVENTS} events, "
        f"{BAD_FACTOR_RATE:.0%} of factors badly wrong:\n"
        f"  unadjusted_prior     {prior_brier:.5f}   calibration gap {prior_gap:.4f}\n"
        f"  uncapped_adjustment  {uncapped_brier:.5f}   calibration gap {uncapped_gap:.4f}\n"
        f"  bounded_bridge       {bounded_brier:.5f}   calibration gap {bounded_gap:.4f}"
    )

    assert bounded_brier < prior_brier, "capped adjustment should beat doing nothing"
    assert bounded_brier < uncapped_brier, "caps should beat trusting factors outright"


@pytest.mark.parametrize("bad_rate", [0.0, 0.5])
def test_the_caps_matter_most_when_factors_are_unreliable(bad_rate: float) -> None:
    """Where capping helps, and where it costs — stated rather than assumed.

    With perfect factors, caps only hold back a correct adjustment and uncapped
    wins. With half the factors wrong, uncapped is worse than doing nothing at all
    and the caps are what keep the estimate usable. Publishing only the favourable
    half of that would be dishonest.
    """
    global BAD_FACTOR_RATE
    original = BAD_FACTOR_RATE
    try:
        BAD_FACTOR_RATE = bad_rate
        base_rates, factor_sets, outcomes = make_log()
        bridge = Bridge()
        bounded = [
            bridge.apply(b, f).probability for b, f in zip(base_rates, factor_sets, strict=True)
        ]
        uncapped = [uncapped_adjustment(b, f) for b, f in zip(base_rates, factor_sets, strict=True)]
        prior = [unadjusted_prior(b, f) for b, f in zip(base_rates, factor_sets, strict=True)]

        print(
            f"\nbad factor rate {bad_rate:.0%}: "
            f"prior {brier_score(prior, outcomes):.5f}  "
            f"uncapped {brier_score(uncapped, outcomes):.5f}  "
            f"bounded {brier_score(bounded, outcomes):.5f}"
        )
        if bad_rate == 0.0:
            assert brier_score(uncapped, outcomes) < brier_score(bounded, outcomes)
        else:
            assert brier_score(bounded, outcomes) < brier_score(uncapped, outcomes)
    finally:
        BAD_FACTOR_RATE = original
