from __future__ import annotations

import pandas as pd
import pytest

from src.xgb_matcher.v412_service_parity import (
    _assert_exact,
    evaluate_paired_gate,
    nearest_rank,
)


def _summary(mode: str, pid: int, p95: int) -> dict:
    return {
        "mode": mode,
        "pid": pid,
        "query_count": 1456,
        "peak_rss_bytes": 2 * 1024**3,
        "latency": {
            "p95_ns": p95,
            "evidence_guard_p95_ns": 0 if mode == "v411" else 20,
            "retrieval_lookup_ns_p95_ns": 100,
        },
        "counters": {
            "lookup_missing_count": 0,
            "sealed_key_miss_count": 0,
            "cache_rebuild_count": 0,
            "cache_write_count": 0,
            "cache_rebuild_api_absent": True,
            "cache_write_api_absent": True,
        },
        "model_load_count": 1,
        "store_load_count": 1,
        "_total_wall_ns": [p95] * 1456,
    }


def test_nearest_rank_is_contractual() -> None:
    assert nearest_rank([5, 1, 4, 2, 3], 0.50) == 3
    assert nearest_rank([5, 1, 4, 2, 3], 0.95) == 5


def test_exact_parity_accepts_only_equivalent_text_storage() -> None:
    observed = pd.DataFrame(
        {
            "query_id": pd.Series(["1", None], dtype="string"),
            "score": pd.Series([1.0, 2.0], dtype="float32"),
        }
    )
    expected = pd.DataFrame(
        {
            "query_id": pd.Series(["1", None], dtype=object),
            "score": pd.Series([1.0, 2.0], dtype="float32"),
        }
    )
    _assert_exact(observed, expected, label="text storage")

    changed_value = expected.copy()
    changed_value.loc[0, "query_id"] = "2"
    with pytest.raises(ValueError, match="parity changed"):
        _assert_exact(observed, changed_value, label="text value")

    changed_numeric_dtype = expected.copy()
    changed_numeric_dtype["score"] = changed_numeric_dtype["score"].astype(
        "float64"
    )
    with pytest.raises(ValueError, match="parity changed"):
        _assert_exact(
            observed,
            changed_numeric_dtype,
            label="numeric dtype",
        )


def test_paired_gate_distinguishes_go_pivot_and_integrity() -> None:
    v411 = _summary("v411", 101, 100)
    v412g = _summary("v412g", 202, 150)
    assert evaluate_paired_gate(v411, v412g)["verdict"] == (
        "GO_V412_SERVICE_FREEZE"
    )

    v412g["latency"]["p95_ns"] = 200
    pivot = evaluate_paired_gate(v411, v412g)
    assert pivot["verdict"] == "PIVOT_V412_SERVICE_IMPLEMENTATION"
    assert "full_latency_ratio" in pivot["cost_reasons"]

    v412g["latency"]["p95_ns"] = 150
    v412g["counters"]["cache_write_count"] = 1
    with pytest.raises(ValueError, match="cache_write_count is nonzero"):
        evaluate_paired_gate(v411, v412g)

    v412g["counters"]["cache_write_count"] = 0
    v412g["model_load_count"] = 2
    pivot = evaluate_paired_gate(v411, v412g)
    assert "v412g:model_load_count" in pivot["persistence_reasons"]

    v412g["pid"] = v411["pid"]
    with pytest.raises(ValueError, match="worker identity changed"):
        evaluate_paired_gate(v411, v412g)
