from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.evaluate_retrieval_ltr_streaming import iter_complete_query_frames
from scripts.train_retrieval_ltr_streaming import (
    _contiguous_qid,
    presample_hard_candidates,
)
from src.xgb_matcher.retrieval_ltr_admission import AdmissionConfig


def test_query_iterator_preserves_groups_across_arrow_batches(tmp_path: Path) -> None:
    source = tmp_path / "candidates.parquet"
    pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2", "q2", "q2", "q3"],
            "siret": [f"{index:014d}" for index in range(7)],
        }
    ).to_parquet(source, index=False, row_group_size=3)

    groups = list(iter_complete_query_frames(source, batch_size=3))

    assert [group.iloc[0]["query_id"] for group in groups] == ["q1", "q2", "q3"]
    assert [len(group) for group in groups] == [2, 4, 1]


def test_query_iterator_rejects_non_contiguous_queries(tmp_path: Path) -> None:
    source = tmp_path / "candidates.parquet"
    pd.DataFrame(
        {"query_id": ["q1", "q2", "q1"], "siret": ["1", "2", "3"]}
    ).to_parquet(source, index=False)

    try:
        list(iter_complete_query_frames(source, batch_size=1))
    except ValueError as error:
        assert "Non-contiguous" in str(error)
    else:  # pragma: no cover
        raise AssertionError("non-contiguous queries must fail closed")


def test_contiguous_qid_is_stable_and_rejects_reappearing_groups() -> None:
    assert _contiguous_qid(pd.Series(["a", "a", "b", "c", "c"])).tolist() == [
        0,
        0,
        1,
        2,
        2,
    ]
    with pytest.raises(ValueError, match="not contiguous"):
        _contiguous_qid(pd.Series(["a", "b", "a"]))


def test_hard_negative_presample_spans_channels_and_keeps_natural_truth() -> None:
    config = AdmissionConfig.load("config/retrieval_ltr_admission_official_v6.json")
    frame = pd.DataFrame(
        {
            "query_id": ["q"] * 20,
            "siret": [f"{index:014d}" for index in range(20)],
            "gt_siret": [f"{19:014d}"] * 20,
            "union_rank": list(range(1, 21)),
            "legal_name_exact_rank": [None] * 10 + [1] + [None] * 9,
            "address_char_rank": [None] * 11 + [1] + [None] * 8,
        }
    )

    sampled = presample_hard_candidates(
        frame, config, cap=6, union_head=2, channel_head=1
    )

    assert len(sampled) == 5
    assert set(sampled["siret"]) == {
        f"{0:014d}",
        f"{1:014d}",
        f"{10:014d}",
        f"{11:014d}",
        f"{19:014d}",
    }
