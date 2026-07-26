import json

import pandas as pd
import pytest

from scripts.finalize_v4_retrieval_gate import (
    _candidate_result,
    _metric,
    _verdict,
)


def test_candidate_result_finds_exact_siret_rank() -> None:
    result = _candidate_result(
        query_id="q1",
        truth="12345678900002",
        candidates=["11111111100001", "12345678900002"],
        split="dev",
        subset="dev_new",
        provenance="fresh_frozen_retrieval",
        oracle_hit=True,
    )
    assert result["hit_at_100"] is True
    assert result["ground_truth_rank"] == 2
    assert json.loads(result["candidate_sirets_json"])[1] == "12345678900002"


def test_candidate_result_rejects_budget_and_duplicates() -> None:
    with pytest.raises(ValueError, match="more than 100"):
        _candidate_result(
            query_id="q1",
            truth="12345678900002",
            candidates=[str(value).zfill(14) for value in range(101)],
            split="dev",
            subset="dev_new",
            provenance="fresh_frozen_retrieval",
            oracle_hit=True,
        )
    with pytest.raises(ValueError, match="duplicate"):
        _candidate_result(
            query_id="q2",
            truth="12345678900002",
            candidates=["12345678900002", "12345678900002"],
            split="dev",
            subset="dev_new",
            provenance="fresh_frozen_retrieval",
            oracle_hit=True,
        )


def test_metric_reports_counts_and_wilson_interval() -> None:
    metric = _metric(pd.DataFrame({"hit_at_100": [True, True, False]}))
    assert metric["successes"] == 2
    assert metric["total"] == 3
    assert metric["rate"] == pytest.approx(2 / 3)
    assert metric["wilson_95"][0] < metric["rate"] < metric["wilson_95"][1]


def test_gate_verdicts_are_explicit() -> None:
    assert (
        _verdict(
            fit_rate=0.995,
            dev_rate=0.99,
            fresh_oracle_rate=1.0,
            controls_pass=True,
        )
        == "GO_RANKER_V4"
    )
    assert (
        _verdict(
            fit_rate=0.995,
            dev_rate=0.98,
            fresh_oracle_rate=1.0,
            controls_pass=True,
        )
        == "PIVOT_RETRIEVAL_V4"
    )
    assert (
        _verdict(
            fit_rate=0.995,
            dev_rate=0.98,
            fresh_oracle_rate=0.98,
            controls_pass=True,
        )
        == "STOP_V4_DATA"
    )
