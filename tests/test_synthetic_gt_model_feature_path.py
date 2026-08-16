from __future__ import annotations

import json

import pandas as pd

from scripts.build_synthetic_gt_model_features import (
    ORACLE_CHANNELS,
    _admission,
    _select_negative_indices,
)
from scripts.prepare_synthetic_gt_model_retrieval_input import _stable_train_fold


def test_synthetic_fold_is_siren_stable_and_train_only() -> None:
    assert _stable_train_fold("123456789") == _stable_train_fold("123456789")
    assert _stable_train_fold("123456789") in {2, 3, 4}
    observed = {_stable_train_fold(f"{value:09d}") for value in range(100)}
    assert observed == {2, 3, 4}


def _channel_row(query_id: str, values: dict[str, list[str]]) -> dict[str, str]:
    return {
        "query_id": query_id,
        **{
            f"{channel}_sirets_json": json.dumps(values.get(channel, []))
            for channel in ORACLE_CHANNELS
        },
    }


def test_admission_reuses_frozen_fusion_without_injecting_truth() -> None:
    v7 = pd.DataFrame(
        [_channel_row("q1", {"name_word": ["1", "2"], "name_char": ["2", "3"]})]
    )
    overlay = pd.DataFrame(
        [_channel_row("q1", {"name_word": ["9"], "name_char": ["8"]})]
    )

    output = _admission(v7, overlay)
    selected = json.loads(output.iloc[0]["candidate_sirets_json"])

    assert selected[:2] == ["9", "8"]
    assert set(selected) == {"1", "2", "3", "8", "9"}
    assert "77777777777777" not in selected


def test_bge_negative_mining_excludes_same_siren_same_official_site() -> None:
    rows = [
        {
            "candidate_siret": "11111111100001",
            "candidate_siren": "111111111",
            "address_id": "SITE-A",
            "candidate_state": "F",
            "is_ground_truth": 1,
        },
        {
            "candidate_siret": "11111111100002",
            "candidate_siren": "111111111",
            "address_id": "SITE-A",
            "candidate_state": "A",
            "is_ground_truth": 0,
        },
    ]
    for index in range(2, 20):
        rows.append(
            {
                "candidate_siret": f"{index:014d}",
                "candidate_siren": f"{index:09d}",
                "address_id": f"SITE-{index}",
                "candidate_state": "A",
                "is_ground_truth": 0,
            }
        )
    frame = pd.DataFrame(rows)
    frame["business_ranker_rank"] = range(1, len(frame) + 1)
    frame["retrieval_rank"] = range(1, len(frame) + 1)
    frame["source_name_score"] = 0.0
    frame["name_jaro_max"] = 0.0
    frame["addr_jaro"] = 0.0
    frame["postcode_match"] = 0

    selected, excluded = _select_negative_indices(frame)

    assert excluded == 1
    assert 1 not in selected
    assert len(selected) == 15
