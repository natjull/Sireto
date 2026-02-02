from __future__ import annotations

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    FAST_RANKER_FEATURE_NAMES,
    SEMANTIC_FEATURE_NAMES,
)
from pathlib import Path

from src.xgb_matcher.profile import InferenceProfile


def test_fast_ranker_features_subset_and_order() -> None:
    assert FAST_RANKER_FEATURE_NAMES
    assert len(set(FAST_RANKER_FEATURE_NAMES)) == len(FAST_RANKER_FEATURE_NAMES)
    assert all(name in FEATURE_NAMES for name in FAST_RANKER_FEATURE_NAMES)
    assert not set(FAST_RANKER_FEATURE_NAMES) & set(SEMANTIC_FEATURE_NAMES)

    positions = [FEATURE_NAMES.index(name) for name in FAST_RANKER_FEATURE_NAMES]
    assert positions == sorted(positions)

    for name in ("name_jaro_max", "addr_jaro", "geo_exact_match"):
        assert name in FAST_RANKER_FEATURE_NAMES


def test_profile_stage1_feature_order_prefers_fast() -> None:
    profile = InferenceProfile(
        feature_order=["full_a", "full_b"],
        ranker_feature_order=["ranker_a"],
        ranker_fast_feature_order=["fast_a"],
        ranker_fast_path=Path("models/xgbranker_fast.json"),
        use_ranker_fast=True,
    )
    assert profile.get_stage1_feature_order() == ["fast_a"]

    profile.use_ranker_fast = False
    assert profile.get_stage1_feature_order() == ["ranker_a"]

    profile.ranker_feature_order = []
    assert profile.get_stage1_feature_order() == ["full_a", "full_b"]
