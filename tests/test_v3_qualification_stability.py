from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_v3_qualification_stability import (
    audit_train_dev,
    no_evidence_nearness,
    qualification_decomposition,
)


def _qualified(split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["1", "2", "3", "4"],
            "split": [split] * 4,
            "label_kind": [
                "MATCH_EXACT",
                "UNRESOLVED",
                "AMBIGUOUS",
                "MATCH_EXACT",
            ],
            "v2_label_kind": [
                "MATCH_EXACT",
                "MATCH_EXACT",
                "AMBIGUOUS",
                "MATCH_EXACT",
            ],
            "direct_evidence_class": [
                "NAME_ONLY",
                "NO_DIRECT_EVIDENCE",
                "NAME_AND_ADDRESS",
                "ADDRESS_ONLY",
            ],
            "ground_truth_state": ["A", "F", "F", "A"],
            "location_match_type": ["insee", "insee", "cp_only", "insee"],
            "name_jaro_max": [0.9, 0.8, 0.2, 0.3],
            "postcode_match": [1.0, 1.0, 0.0, 1.0],
            "street_name_jaro": [0.9, 0.4, 0.2, 0.9],
        }
    )


def _flags() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["1", "2", "3", "4"],
            "mega_base_pool": [False, True, False, True],
            "multi_site_siren": [False, False, True, True],
        }
    )


def test_qualification_decomposition_separates_two_exclusion_stages():
    result = qualification_decomposition(_qualified("train"))

    assert result["query_count"] == 4
    assert result["v2_exact_count"] == 3
    assert result["v3_exact_count"] == 2
    assert result["structural_excluded_count"] == 1
    assert result["direct_evidence_excluded_count"] == 1
    assert result["v2_coverage"] == 0.75
    assert result["direct_evidence_retention"] == pytest.approx(2 / 3)
    assert result["v3_coverage"] == 0.5


def test_nearness_is_descriptive_for_v3_exclusions_only():
    result = no_evidence_nearness(_qualified("train"))

    assert result["query_count"] == 1
    assert result["buckets"]["NAME_NEAR__ADDRESS_FAR"]["count"] == 1


def test_audit_rejects_any_test_split():
    with pytest.raises(ValueError, match="Expected only split 'train'"):
        audit_train_dev(_qualified("test"), _qualified("dev"), _flags())


def test_audit_adds_dev_scene_segments_without_test_data():
    result = audit_train_dev(
        _qualified("train"),
        _qualified("dev"),
        _flags(),
    )

    assert result["scope"] == "TRAIN_DEV_ONLY"
    assert result["test_read"] is False
    assert result["dev"]["segments"]["mega"]["query_count"] == 2
    assert result["dev"]["segments"]["multi_site"]["query_count"] == 2
