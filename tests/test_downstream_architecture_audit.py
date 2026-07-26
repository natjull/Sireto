from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.audit_downstream_architecture import (
    candidate_dataset_summary,
    legacy_reference_summary,
    profile_inventory,
)


def test_candidate_audit_checks_positive_siren_split_leakage():
    frame = pd.DataFrame(
        {
            "query_id": ["a", "a", "b", "b", "c", "c"],
            "label": [1, 0, 1, 0, 1, 0],
            "siren": ["111", "900", "222", "901", "333", "902"],
            "split": ["train", "train", "dev", "dev", "test", "test"],
            "feature": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )

    result = candidate_dataset_summary(frame, ["feature"])

    assert result["positive_count_per_query"] == {"1": 3}
    assert result["positive_siren_split_leakage"] == {
        "train_dev": 0,
        "train_test": 0,
        "dev_test": 0,
    }


def test_candidate_audit_detects_positive_siren_leakage():
    frame = pd.DataFrame(
        {
            "query_id": ["a", "b"],
            "label": [1, 1],
            "siren": ["111", "111"],
            "split": ["train", "dev"],
            "feature": [1.0, 1.0],
        }
    )

    result = candidate_dataset_summary(frame, ["feature"])

    assert result["positive_siren_split_leakage"]["train_dev"] == 1


def test_legacy_reference_separates_duplicated_and_distinct_scenes():
    topk = pd.DataFrame(
        {
            "crm_id": ["1", "1", "2", "2"],
            "siret_candidate": [
                "11111111100001",
                "11111111100001",
                "22222222200002",
                "99999999900009",
            ],
            "rank": [0, 1, 0, 1],
        }
    )
    truth = pd.DataFrame(
        {
            "crm_id": ["1", "2"],
            "siret_gt": ["11111111100001", "99999999900009"],
        }
    )
    routed = pd.DataFrame(
        {
            "crm_id": ["1", "2"],
            "xgb_status": ["AUTO_RISK", "AUTO_RISK"],
            "chosen_siret_final": [
                "11111111100001",
                "22222222200002",
            ],
        }
    )

    result = legacy_reference_summary(topk, truth, routed)

    assert result["duplicate_top1_top2"]["query_count"] == 1
    assert result["auto_duplicate_scenes"]["precision"] == 1.0
    assert result["auto_distinct_scenes"]["precision"] == 0.0
    assert result["candidate_recall"]["successes"] == 2
    assert result["top1_exact"]["successes"] == 1


def test_candidate_audit_rejects_unknown_split():
    frame = pd.DataFrame(
        {
            "query_id": ["a"],
            "label": [1],
            "siren": ["111"],
            "split": ["holdout"],
            "feature": [1.0],
        }
    )

    with pytest.raises(ValueError, match="unsupported split"):
        candidate_dataset_summary(frame, ["feature"])


def test_profile_inventory_exposes_lexicographic_selection_and_invalid_bundle(
    tmp_path,
):
    older = tmp_path / "xgb_two_stage_meta_20260221_224040.json"
    older.write_text(
        json.dumps(
            {
                "ranker_model": "same.json",
                "decider_model": "same.json",
            }
        ),
        encoding="utf-8",
    )
    legacy = tmp_path / "xgb_two_stage_meta_v5fast.json"
    legacy.write_text(
        json.dumps(
            {
                "ranker_model": "ranker.json",
                "decider_model": "decider.json",
            }
        ),
        encoding="utf-8",
    )

    result = profile_inventory(tmp_path)

    assert result["lexicographic_latest"] == str(legacy)
    assert result["invalid_same_ranker_and_decider_metas"] == [str(older)]
