from pathlib import Path

from scripts.build_v412_trusted_review_labels import build


def test_trusted_review_labels_are_complete_and_canonical() -> None:
    assignments = Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
        "v4_11_input_blind/ec4326ec57e4411d/split_assignments.parquet"
    )
    frame = build(assignments)
    assert len(frame) == 279
    assert frame["query_id"].is_unique
    assert frame["label_kind"].value_counts().to_dict() == {
        "MATCH_EXACT": 254,
        "AMBIGUOUS": 25,
    }
