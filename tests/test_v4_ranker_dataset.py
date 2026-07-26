import pandas as pd

from scripts.build_v4_ranker_dataset import (
    OUTPUT_CANDIDATE_COLUMNS,
    relabel_historical_candidates,
    validate_candidate_rows,
)


def _candidate(query_id: str, siret: str) -> dict:
    row = {column: 0.0 for column in OUTPUT_CANDIDATE_COLUMNS}
    row.update(
        {
            "query_id": query_id,
            "candidate_siret": siret,
            "candidate_siren": siret[:9],
            "split": "dev",
            "is_ground_truth": 0,
            "retrieval_source": "fixture",
        }
    )
    return row


def test_relabel_historical_candidates_uses_v4_truth() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("q1", "11111111100001"),
            _candidate("q1", "22222222200002"),
        ]
    )
    labels = pd.DataFrame(
        {
            "query_id": ["q1"],
            "ground_truth_siret": ["22222222200002"],
        }
    )
    output = relabel_historical_candidates(candidates, labels)
    assert output["split"].eq("train").all()
    assert output["is_ground_truth"].tolist() == [0, 1]


def test_validate_candidate_rows_requires_one_positive_per_query() -> None:
    labels = pd.DataFrame({"query_id": ["q1"]})
    valid = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "candidate_siret": ["11111111100001", "22222222200002"],
            "is_ground_truth": [0, 1],
        }
    )
    assert validate_candidate_rows(valid, labels)["pass"] is True
    invalid = valid.assign(is_ground_truth=0)
    result = validate_candidate_rows(invalid, labels)
    assert result["pass"] is False
    assert result["checks"]["exactly_one_positive_per_query"] is False
