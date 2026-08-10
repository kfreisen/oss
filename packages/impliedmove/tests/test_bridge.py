"""The bounded-factor probability bridge.

The caps are the whole mechanism, so most of these test that a cap binds. Nothing
here checks whether a factor is *sensible* — that is not checkable. What is
guaranteed is that the worst case is bounded even when the reasoning is nonsense.
"""

from __future__ import annotations

import pytest
from impliedmove.bridge import Bridge, Factor


def test_no_factors_leaves_the_base_rate_alone() -> None:
    result = Bridge().apply(0.35, [])
    assert result.probability == pytest.approx(0.35)
    assert result.total_adjustment == 0.0
    assert result.capped == ()


def test_factors_move_the_estimate() -> None:
    result = Bridge().apply(0.35, [Factor("strong data", 0.08)])
    assert result.probability == pytest.approx(0.43)


def test_negative_factors_move_it_down() -> None:
    result = Bridge().apply(0.35, [Factor("weak endpoint", -0.06)])
    assert result.probability == pytest.approx(0.29)


def test_a_single_factor_cannot_exceed_its_cap() -> None:
    result = Bridge(max_per_factor=0.10).apply(0.35, [Factor("wild claim", 0.90)])
    assert result.total_adjustment == pytest.approx(0.10)
    assert "wild claim" in result.capped


def test_a_negative_factor_is_capped_symmetrically() -> None:
    result = Bridge(max_per_factor=0.10).apply(0.35, [Factor("doom", -0.90)])
    assert result.total_adjustment == pytest.approx(-0.10)


def test_a_category_cap_binds_across_related_factors() -> None:
    """Five restatements of one idea must not compound into a large move."""
    factors = [Factor(f"mgmt-{i}", 0.05, category="management") for i in range(5)]
    result = Bridge(max_per_category=0.10, max_total=1.0).apply(0.30, factors)
    assert result.total_adjustment == pytest.approx(0.10)
    assert "management" in result.capped


def test_category_capping_scales_members_proportionally() -> None:
    # The audit trail must still add up to what was applied, or it is not an
    # audit trail.
    factors = [Factor("a", 0.08, category="c"), Factor("b", 0.04, category="c")]
    result = Bridge(max_per_category=0.06, max_total=1.0).apply(0.30, factors)
    assert sum(d for _, d in result.applied) == pytest.approx(result.total_adjustment)
    assert result.applied[0][1] == pytest.approx(0.04)
    assert result.applied[1][1] == pytest.approx(0.02)


def test_uncategorized_factors_escape_the_category_cap() -> None:
    factors = [Factor("a", 0.09), Factor("b", 0.09)]
    result = Bridge(max_per_category=0.05, max_total=1.0).apply(0.30, factors)
    assert result.total_adjustment == pytest.approx(0.18)


def test_the_total_cap_binds_last() -> None:
    factors = [Factor(f"f{i}", 0.10) for i in range(6)]
    result = Bridge(max_total=0.25).apply(0.30, factors)
    assert result.total_adjustment == pytest.approx(0.25)
    assert "total" in result.capped


def test_total_capping_also_scales_the_audit_trail() -> None:
    factors = [Factor(f"f{i}", 0.10) for i in range(6)]
    result = Bridge(max_total=0.25).apply(0.30, factors)
    assert sum(d for _, d in result.applied) == pytest.approx(0.25)


def test_opposing_factors_can_net_to_nothing() -> None:
    result = Bridge().apply(0.40, [Factor("up", 0.08), Factor("down", -0.08)])
    assert result.total_adjustment == pytest.approx(0.0)
    assert result.probability == pytest.approx(0.40)
    assert result.capped == ()


def test_the_floor_and_ceiling_bound_the_result() -> None:
    bridge = Bridge(max_per_factor=1.0, max_total=1.0, floor=0.01, ceiling=0.99)
    assert bridge.apply(0.02, [Factor("doom", -1.0)]).probability == pytest.approx(0.01)
    assert bridge.apply(0.98, [Factor("boom", 1.0)]).probability == pytest.approx(0.99)


def test_the_default_ceiling_stops_short_of_certainty() -> None:
    # An estimate of exactly 1.0 asserts certainty, which this procedure is in no
    # position to establish.
    assert Bridge().ceiling < 1.0
    assert Bridge().floor > 0.0


@pytest.mark.parametrize("base_rate", [-0.1, 1.1])
def test_an_out_of_range_base_rate_is_rejected(base_rate: float) -> None:
    with pytest.raises(ValueError, match=r"base_rate must be in \[0, 1\]"):
        Bridge().apply(base_rate, [])


@pytest.mark.parametrize(
    "kwargs",
    [{"max_per_factor": -0.1}, {"max_per_category": -0.1}, {"max_total": -0.1}],
)
def test_negative_caps_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        Bridge(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"floor": 0.8, "ceiling": 0.2}, {"floor": -0.1}, {"ceiling": 1.5}],
)
def test_an_impossible_floor_or_ceiling_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="floor and ceiling"):
        Bridge(**kwargs)


def test_the_rationale_is_carried_but_not_used() -> None:
    factor = Factor("data", 0.05, rationale="phase 2 hit its primary endpoint")
    result = Bridge().apply(0.3, [factor])
    assert result.probability == pytest.approx(0.35)
    assert factor.rationale


def test_a_zero_total_cap_freezes_the_base_rate() -> None:
    result = Bridge(max_total=0.0).apply(0.42, [Factor("anything", 0.10)])
    assert result.probability == pytest.approx(0.42)
    assert "total" in result.capped
