import pandas as pd

from scripts.build_v4_acceptor_dataset import (
    OUTPUT_CANDIDATE_COLUMNS,
    relabel_ambiguous_candidates,
    validate_acceptor_candidates,
)


def test_relabel_ambiguous_candidates_never_creates_positive() -> None:
    row = {column: 0.0 for column in OUTPUT_CANDIDATE_COLUMNS}
    row.update(
        {
            "query_id": "q1",
            "candidate_siret": "11111111100001",
            "candidate_siren": "111111111",
            "split": "dev",
            "is_ground_truth": 1,
            "retrieval_source": "fixture",
        }
    )
    output = relabel_ambiguous_candidates(pd.DataFrame([row]), {"q1"})
    assert output.loc[0, "split"] == "train"
    assert output.loc[0, "is_ground_truth"] == 0


def test_validate_acceptor_candidates_distinguishes_label_kinds() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": ["exact", "ambiguous"],
            "candidate_siret": ["11111111100001", "22222222200002"],
            "is_ground_truth": [1, 0],
        }
    )
    labels = pd.DataFrame(
        {
            "query_id": ["exact", "ambiguous"],
            "label_kind": ["MATCH_EXACT", "AMBIGUOUS"],
        }
    )
    assert validate_acceptor_candidates(candidates, labels)["pass"] is True
