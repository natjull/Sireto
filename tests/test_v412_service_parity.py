from __future__ import annotations

import pytest

from src.xgb_matcher.v412_service_parity import (
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
    pivot = evaluate_paired_gate(v411, v412g)
    assert "v412g:cache_write_count" in pivot["persistence_reasons"]

    v412g["counters"]["cache_write_count"] = 0
    v412g["model_load_count"] = 2
    pivot = evaluate_paired_gate(v411, v412g)
    assert "v412g:model_load_count" in pivot["persistence_reasons"]

    v412g["pid"] = v411["pid"]
    with pytest.raises(ValueError, match="worker identity changed"):
        evaluate_paired_gate(v411, v412g)
