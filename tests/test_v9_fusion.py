import pytest

from src.xgb_matcher.fusion import (
    annotate_fused_candidate,
    reciprocal_rank_fusion,
)
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.v9_features import inject_retrieval_siren_features


def test_rrf_preserves_budget_and_rewards_channel_agreement():
    hits = reciprocal_rank_fusion(
        {
            "sparse": ["a", "b", "c", "d"],
            "dense": ["c", "a", "e", "f"],
            "rescue": ["g"],
        },
        budget=3,
        rrf_k=60,
    )

    assert len(hits) == 3
    assert [hit.key for hit in hits[:2]] == ["a", "c"]
    assert hits[0].channel_ranks == {"sparse": 1, "dense": 2}


def test_rrf_annotation_and_siren_aggregates():
    hits = reciprocal_rank_fusion(
        {"sparse": [0, 1], "dense": [1, 0]},
        budget=2,
    )
    candidates = [
        annotate_fused_candidate({"siret": "12345678900011", "siren": "123456789"}, hits[0]),
        annotate_fused_candidate({"siret": "12345678900022", "siren": "123456789"}, hits[1]),
    ]
    rows = [{}, {}]
    inject_retrieval_siren_features(rows, candidates)

    assert rows[0]["retrieval_agreement"] == 1.0
    assert rows[0]["siren_candidate_count"] == 2.0
    assert rows[0]["siren_rrf_max"] == pytest.approx(
        max(candidate["rrf_score"] for candidate in candidates)
    )


def test_rrf_config_roundtrip_and_validation():
    config = RetrievalConfigV1(
        fusion_mode="rrf",
        retrieval_budget=50,
        dense_retrieval_enabled=True,
    )
    restored = RetrievalConfigV1.from_dict(config.to_dict())
    assert restored == config
    assert restored.signature().hash == config.signature().hash
    with pytest.raises(ValueError, match="fusion_mode"):
        RetrievalConfigV1(fusion_mode="unknown")
