from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.evaluate_v412_neural_reranker import _evaluate


def test_evaluate_keeps_absent_truth_as_end_to_end_error(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        [
            {
                "query_id": "a",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100011",
                "label_is_human_validated": True,
                "ground_truth_state": "A",
                "oof_fold": 0,
            },
            {
                "query_id": "b",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "22222222200022",
                "label_is_human_validated": False,
                "ground_truth_state": "F",
                "oof_fold": 0,
            },
        ]
    )
    labels.to_parquet(tmp_path / "labels.parquet", index=False)
    scored = pd.DataFrame(
        [
            {
                "query_id": "a",
                "candidate_siret": "11111111100011",
                "candidate_siren": "111111111",
                "retrieval_rank": 2,
                "neural_score": 1.0,
            },
            {
                "query_id": "b",
                "candidate_siret": "99999999900099",
                "candidate_siren": "999999999",
                "retrieval_rank": 1,
                "neural_score": 1.0,
            },
        ]
    )

    metrics, detail = _evaluate(tmp_path, 0, scored, {"a", "b"})

    exact = metrics[metrics["segment"].eq("exact")].iloc[0]
    assert (exact["correct"], exact["total"]) == (1, 2)
    assert detail.set_index("query_id").loc["b", "correct"] == False  # noqa: E712
