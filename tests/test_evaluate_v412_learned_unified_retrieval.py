from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.evaluate_v412_learned_unified_retrieval import (
    _metric,
    _parse_pool,
)


def test_parse_pool_normalizes_and_enforces_the_absolute_ceiling() -> None:
    assert _parse_pool(json.dumps(["1234567890001", "98765432100001"])) == [
        "01234567890001",
        "98765432100001",
    ]
    with pytest.raises(ValueError, match="Duplicate SIRET"):
        _parse_pool(json.dumps(["12345678900001", "12345678900001"]))
    with pytest.raises(ValueError, match="exceeds ceiling"):
        _parse_pool(json.dumps([f"{value:014d}" for value in range(101)]))


def test_metric_publishes_raw_counts_and_both_wilson_intervals() -> None:
    metric = _metric(pd.Series([True, True, False, True]))
    assert metric["successes"] == 3
    assert metric["total"] == 4
    assert metric["rate"] == 0.75
    assert len(metric["wilson_95"]) == 2
    assert len(metric["wilson_99"]) == 2
