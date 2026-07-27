from __future__ import annotations

import pandas as pd
import pytest

from scripts.evaluate_v42_representative_retrieval import summarize_results


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "service_id": "A",
                "sampling_stratum": "RANDOM_POPULATION",
                "baseline_a_top10_hit": True,
                "hit_at_100": True,
                "candidate_count": 100,
                "closed_candidate_count": 0,
                "truth_state": "A",
                "positive_injected": False,
                "latency_ms": 10.0,
            },
            {
                "service_id": "B",
                "sampling_stratum": "TARGETED",
                "baseline_a_top10_hit": False,
                "hit_at_100": True,
                "candidate_count": 3,
                "closed_candidate_count": 0,
                "truth_state": "A",
                "positive_injected": False,
                "latency_ms": 20.0,
            },
        ]
    )


def test_summarize_v42_results_passes_all_integrity_gates() -> None:
    summary = summarize_results(_results())

    assert summary["recall_at_100"] == {
        "successes": 2,
        "total": 2,
        "rate": 1.0,
    }
    assert summary["random_population_recall_at_100"]["rate"] == 1.0
    assert summary["baseline_a_regression_count"] == 0
    assert summary["max_candidate_count"] == 100
    assert summary["verdict"] == "GO_HARD_LABELS"
    assert all(summary["gates"].values())


def test_summarize_v42_results_detects_regression_and_closed_candidate() -> None:
    results = _results()
    results.loc[0, "hit_at_100"] = False
    results.loc[1, "closed_candidate_count"] = 1

    summary = summarize_results(results)

    assert summary["baseline_a_regression_count"] == 1
    assert summary["closed_candidate_count"] == 1
    assert summary["miss_service_ids"] == ["A"]
    assert summary["verdict"] == "PIVOT"


def test_summarize_v42_results_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_results(pd.DataFrame())
