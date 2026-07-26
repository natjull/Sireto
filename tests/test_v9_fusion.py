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


def test_sparse_channel_fusion_mode_is_signed_but_does_not_rebuild_matrices():
    legacy = RetrievalConfigV1(sparse_channel_fusion_mode="max_score")
    separated = RetrievalConfigV1(sparse_channel_fusion_mode="separate_rrf")

    assert RetrievalConfigV1.from_dict(separated.to_dict()) == separated
    assert legacy.signature().hash != separated.signature().hash
    assert legacy.tfidf_artifact_hash() == separated.tfidf_artifact_hash()
    with pytest.raises(ValueError, match="sparse_channel_fusion_mode"):
        RetrievalConfigV1(sparse_channel_fusion_mode="unknown")


def test_separate_sparse_annotation_uses_best_sparse_rank_and_dense_agreement():
    hit = reciprocal_rank_fusion(
        {
            "sparse_name": ["truth", "other"],
            "sparse_address": ["other", "truth"],
            "dense": ["truth"],
        },
        budget=2,
    )[0]

    annotated = annotate_fused_candidate({"siret": "truth"}, hit)

    assert annotated["sparse_rank"] == 1
    assert annotated["sparse_name_rank"] == 1
    assert annotated["sparse_address_rank"] == 2
    assert annotated["retrieval_agreement"] == 1


def test_fr029212_synthetic_regression_separate_channels_recover_truth():
    """Model the audited ranks without using or tuning against holdout data."""

    truth = "truth"
    common = [f"mall-{index:03d}" for index in range(200)]
    name = [*common[:28], truth, *common[28:]]
    address = [*common[:127], truth, *common[127:]]
    rescue = [*common[:132], truth, *common[132:]]

    legacy = reciprocal_rank_fusion(
        {"sparse": address, "rescue": rescue},
        budget=100,
    )
    separated = reciprocal_rank_fusion(
        {
            "sparse_name": name,
            "sparse_address": address,
            "rescue": rescue,
        },
        budget=100,
    )

    assert truth not in {hit.key for hit in legacy}
    separated_truth = next(hit for hit in separated if hit.key == truth)
    assert separated_truth.rank <= 100
    assert separated_truth.channel_ranks == {
        "sparse_name": 29,
        "sparse_address": 128,
        "rescue": 133,
    }
