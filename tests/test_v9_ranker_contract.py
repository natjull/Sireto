import pandas as pd

import pytest

from scripts.train_v9_ranker import (
    eligible_ranker_rows,
    fold_for_query,
    validate_training_positive_rows,
)


def test_ranker_training_excludes_queries_without_retrieved_positive():
    labels = pd.DataFrame(
        {
            "query_id": ["found", "miss", "negative"],
            "label_kind": ["MATCH_EXACT", "MATCH_EXACT", "NO_MATCH"],
        }
    )
    candidates = pd.DataFrame(
        {
            "query_id": ["found", "found", "miss", "negative"],
            "is_ground_truth": [1, 0, 0, 0],
        }
    )
    kept = eligible_ranker_rows(candidates, labels)
    assert set(kept["query_id"]) == {"found"}


def test_fold_assignment_is_stable():
    assert fold_for_query("q1", 42, 5) == fold_for_query("q1", 42, 5)


def test_positive_injection_is_train_only():
    labels = pd.DataFrame(
        {
            "query_id": ["train", "test"],
            "split": ["train", "test"],
            "ground_truth_siret": ["11111111100011", "22222222200022"],
        }
    )
    with pytest.raises(ValueError, match="forbidden outside"):
        validate_training_positive_rows(
            pd.DataFrame(
                {
                    "query_id": ["test"],
                    "candidate_siret": ["22222222200022"],
                    "feature": [1.0],
                }
            ),
            labels,
            ["feature"],
        )
