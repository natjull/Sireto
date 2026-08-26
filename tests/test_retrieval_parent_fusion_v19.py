from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.evaluate_retrieval_parent_fusion_v19_oof import _variant_report
from src.xgb_matcher.retrieval_parent_fusion_v19 import (
    assert_identifier_free_features,
    feature_names,
    prepare_training_frame,
    validate_fold_isolation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/retrieval_all_human_v19_parent_fusion.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def test_feature_order_is_fixed_and_identifier_free() -> None:
    config = _config()
    names = feature_names(config["channels"])
    assert len(names) == 36
    assert names[:6] == (
        "channel_count",
        "best_rank_reciprocal",
        "rrf_score",
        "best_seed_rank_reciprocal",
        "structured_channel_count",
        "native_channel_count",
    )
    assert_identifier_free_features(names)
    with pytest.raises(ValueError, match="identifier-derived"):
        assert_identifier_free_features(["candidate_siren"])
    with pytest.raises(ValueError, match="identifier-derived"):
        assert_identifier_free_features(["query_id"])


def test_fold_isolation_requires_disjoint_ground_truth_parents() -> None:
    good = pd.DataFrame(
        {
            "query_id": ["a", "b", "c"],
            "fold": [2, 3, 4],
            "gt_siren": ["111111111", "222222222", "333333333"],
        }
    )
    validate_fold_isolation(good)
    bad = good.copy()
    bad.loc[2, "gt_siren"] = "111111111"
    with pytest.raises(ValueError, match="share"):
        validate_fold_isolation(bad)


def test_training_sampling_keeps_natural_positive_without_injection() -> None:
    features = ("rrf_score", "channel_count")
    rows = []
    for query in ("a", "b"):
        for index in range(6):
            rows.append(
                {
                    "query_id": query,
                    "is_positive": int(index == 5),
                    "negative_rank": 0 if index == 5 else index + 1,
                    "rrf_score": float(10 - index),
                    "channel_count": float(index + 1),
                }
            )
    X, y, qid, diagnostics = prepare_training_frame(
        pd.DataFrame(rows), features=features, negative_limit=3
    )
    assert X.shape == (8, 2)
    assert y.sum() == 2
    assert list(pd.Series(qid).value_counts(sort=False)) == [4, 4]
    assert diagnostics["positive_injection"] is False
    assert diagnostics["natural_positive_count"] == 2


def _outcomes(primary_hits: int, controls: int = 512) -> pd.DataFrame:
    rows = []
    for variant in ("FIXED_RRF", "OOF_LAMBDAMART"):
        for index in range(522):
            rows.append(
                {
                    "query_id": f"m{index}",
                    "fold": 2 + index % 3,
                    "cohort_kind": "BASE_TOP1750_MISS",
                    "data_origin": "REAL_CRM_20260817" if index < 389 else "REAL_CRM_HISTORICAL",
                    "variant": variant,
                    "parent_hit": index < primary_hits,
                    "parent_rank": 1 if index < primary_hits else None,
                }
            )
        for index in range(512):
            rows.append(
                {
                    "query_id": f"c{index}",
                    "fold": 2 + index % 3,
                    "cohort_kind": "BASE_TOP1750_CONTROL",
                    "data_origin": "REAL_CRM_20260817",
                    "variant": variant,
                    "parent_hit": index < controls,
                    "parent_rank": 1 if index < controls else None,
                }
            )
    return pd.DataFrame(rows)


def test_parent_2000_gate_requires_334_and_all_controls() -> None:
    gates = _config()["gates"]
    # Make the origin guard neutral for this exact boundary test.
    gates = {**gates, "baseline_v17_hits_by_origin": {"REAL_CRM_20260817": 0, "REAL_CRM_HISTORICAL": 0}}
    assert _variant_report(_outcomes(334), gates=gates)["decision"] == "PASS_PARENT_2000_MASTER_99_2_DEVELOPMENT"
    assert _variant_report(_outcomes(333), gates=gates)["decision"] == "PIVOT_PARENT_2000"
    assert _variant_report(_outcomes(334, controls=511), gates=gates)["decision"] == "STOP_PARENT_2000_CONTROL_REGRESSION"


def test_config_freezes_outer_folds_cap_and_no_external_claim() -> None:
    config = _config()
    assert config["fusion"]["parent_cap"] == 2000
    assert config["fusion"]["outer_protocol"] == {"2": [3, 4], "3": [2, 4], "4": [2, 3]}
    assert config["fusion"]["positive_injection"] is False
    assert config["population"]["fold1_locked"] is True
    assert config["claims"]["cross_crm_generalization"] is False
    assert config["claims"]["national_run_authorized"] is False
