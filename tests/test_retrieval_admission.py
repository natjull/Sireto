import pandas as pd

from scripts.evaluate_retrieval_admission import (
    load_frozen_baseline,
    select_candidates,
)


def test_admission_preserves_overlay_quota_and_hard_budget() -> None:
    v7 = {
        "current_sparse": ["A", "B", "C", "D"],
        "name_word": ["A", "C"],
        "name_char": ["A", "B"],
        "address_word": [],
        "siren_head": [],
        "name_exact": [],
        "address_exact": [],
    }
    overlay = {
        "name_word": ["X"],
        "name_char": ["Y", "Z"],
    }
    selected = select_candidates(
        v7_channels=v7,
        overlay_channels=overlay,
        budget=4,
        internal_k=10,
    )
    assert selected[:3] == ["X", "Y", "Z"]
    assert selected == ["X", "Y", "Z", "A"]


def test_admission_tie_break_is_deterministic() -> None:
    channels = {
        "current_sparse": [],
        "name_word": ["B", "A"],
        "name_char": ["A", "B"],
        "address_word": [],
        "siren_head": [],
        "name_exact": [],
        "address_exact": [],
    }
    selected = select_candidates(
        v7_channels=channels,
        overlay_channels={},
        budget=2,
        internal_k=10,
    )
    assert selected == ["A", "B"]


def test_audited_current_sparse_can_be_the_train_baseline(tmp_path) -> None:
    path = tmp_path / "channels.parquet"
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "current_sparse_sirets_json": ['["A","B"]'],
        }
    ).to_parquet(path, index=False)

    baseline = load_frozen_baseline(path)

    assert baseline.to_dict("records") == [
        {"query_id": "q1", "candidate_sirets_json": '["A","B"]'}
    ]
