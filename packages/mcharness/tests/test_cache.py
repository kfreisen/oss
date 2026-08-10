"""The entity cache."""

from __future__ import annotations

import numpy as np
import pytest
from mcharness import EntityCache
from mcharness.testing import Normal

RECORDS = [
    {"player_id": "a", "mean": 10.0, "stddev": 1.0},
    {"player_id": "b", "mean": 20.0, "stddev": 2.0},
    {"player_id": "c", "mean": 30.0, "stddev": 3.0},
]


def cache() -> EntityCache:
    return EntityCache.from_records(RECORDS, key="player_id", fields=("mean", "stddev"))


def test_it_indexes_entities_densely() -> None:
    c = cache()
    assert len(c) == 3
    assert [c.index_of(k) for k in ("a", "b", "c")] == [0, 1, 2]


def test_fields_come_back_as_flat_arrays() -> None:
    c = cache()
    assert c.field("mean").tolist() == [10.0, 20.0, 30.0]
    assert c.field("mean").dtype == np.float64


def test_keys_and_fields_are_reported_in_order() -> None:
    c = cache()
    assert c.keys == ("a", "b", "c")
    assert c.fields == ("mean", "stddev")


def test_an_unknown_entity_says_so() -> None:
    with pytest.raises(KeyError, match="unknown entity 'zzz'"):
        cache().index_of("zzz")


def test_an_uncached_field_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="available: mean, stddev"):
        cache().field("variance")


def test_duplicate_entities_are_rejected() -> None:
    # Silently keeping the last would make the cache disagree with its source.
    doubled = [*RECORDS, RECORDS[0]]
    with pytest.raises(ValueError, match="duplicate entity keys"):
        EntityCache.from_records(doubled, key="player_id", fields=("mean",))


def test_a_record_missing_the_key_is_reported_by_position() -> None:
    with pytest.raises(KeyError, match="record 1 has no 'player_id'"):
        EntityCache.from_records([RECORDS[0], {"mean": 1.0}], key="player_id", fields=("mean",))


def test_a_record_missing_a_field_is_reported_by_position() -> None:
    with pytest.raises(KeyError, match="record 1 has no 'stddev'"):
        EntityCache.from_records(
            [RECORDS[0], {"player_id": "b", "mean": 1.0}],
            key="player_id",
            fields=("mean", "stddev"),
        )


def test_a_mismatched_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="column 'mean' has 2 entries but there are 3"):
        EntityCache(["a", "b", "c"], {"mean": np.array([1.0, 2.0])})


def test_a_custom_dtype_is_honoured() -> None:
    c = EntityCache.from_records(RECORDS, key="player_id", fields=("mean",), dtype=np.float32)
    assert c.field("mean").dtype == np.float32


def test_the_normal_example_simulation_draws() -> None:
    rng = np.random.default_rng(0)
    values = [Normal(5.0, 0.1).run_trial(rng, i) for i in range(500)]
    assert float(np.mean(values)) == pytest.approx(5.0, abs=0.05)
