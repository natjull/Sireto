from __future__ import annotations

import pandas as pd

from scripts.build_v412_neural_text_corpus import _baseline_metrics


def test_baseline_metrics_counts_missing_prediction_as_error() -> None:
    labels = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100011",
                "label_is_human_validated": True,
                "ground_truth_state": "A",
                "oof_fold": 0,
            },
            {
                "query_id": "q2",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "22222222200022",
                "label_is_human_validated": False,
                "ground_truth_state": "F",
                "oof_fold": 1,
            },
            {
                "query_id": "q3",
                "label_kind": "AMBIGUOUS",
                "ground_truth_siret": None,
                "label_is_human_validated": True,
                "ground_truth_state": "",
                "oof_fold": 0,
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "candidate_siret": "11111111100011",
                "ranker_rank": 1,
            }
        ]
    )

    metrics = _baseline_metrics(labels, predictions)
    global_exact = metrics[(metrics["fold"] == "ALL") & (metrics["segment"] == "exact")].iloc[0]
    fold_one = metrics[(metrics["fold"] == "1") & (metrics["segment"] == "exact")].iloc[0]

    assert (global_exact["correct"], global_exact["total"]) == (1, 2)
    assert (fold_one["correct"], fold_one["total"]) == (0, 1)
