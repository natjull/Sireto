from __future__ import annotations

import pandas as pd
import pytest

from src.xgb_matcher.v412_service_retrieval import (
    V412RetrievalFeatureService,
)
from src.xgb_matcher.v412_unit_retrieval import (
    UnitRetrievalContext,
    UnitRetrievalResult,
)


FEATURES = [*[f"feature_{index}" for index in range(44)], "retrieval_rank_recip"]


def _candidate(siret: str) -> dict[str, object]:
    return {
        "siret": siret,
        "siren": siret[:9],
        "denomination": f"Candidate {siret}",
        "numeroVoie": "1",
        "typeVoie": "RUE",
        "libelleVoie": "EXEMPLE",
        "postcode": "75001",
        "city": "PARIS",
        "insee": "75101",
        "etat_admin": "A",
    }


def _context(*, missing: int = 0) -> UnitRetrievalContext:
    candidates = tuple(
        _candidate(siret)
        for siret in (
            "00000000100001",
            "00000000200002",
            "00000000300003",
            "00000000400004",
        )
    )
    result = UnitRetrievalResult(
        partition_key="75101_",
        candidate_sirets=("00000000100001", "00000000200002"),
        raw_pool_count=4,
        aligned_pool_count=4,
        lookup_missing_count=missing,
    )
    return UnitRetrievalContext(
        result=result,
        aligned_pool=candidates,
        selected_indices=(0, 1, 2),
        scores_by_index={0: 0.3, 1: 0.2, 2: 0.1},
        channel_ranks_by_index={
            0: {"sparse_name": 1, "sparse_address": 2},
            1: {},
            2: {"rescue": 1},
        },
        snapshot_details={
            "00000000100001": {
                "candidate_state": "A",
                "enseigne1": "ONE",
                "enseigne2": None,
                "enseigne3": None,
                "denomination_usuelle": None,
                "activity_code": "00.00Z",
            },
            "00000000200002": {
                "candidate_state": "A",
                "enseigne1": "TWO",
                "enseigne2": None,
                "enseigne3": None,
                "denomination_usuelle": None,
                "activity_code": "00.00Z",
            },
            "00000000300003": {
                "candidate_state": "F",
                "enseigne1": "THREE",
                "enseigne2": None,
                "enseigne3": None,
                "denomination_usuelle": None,
                "activity_code": "00.00Z",
            },
        },
    )


def _service(
    context: UnitRetrievalContext,
    captured: list[list[dict[str, object]]],
) -> V412RetrievalFeatureService:
    def retriever(**kwargs):
        return context

    def feature_builder(crm, candidates, *, include_semantic):
        assert include_semantic is False
        captured.append([dict(candidate) for candidate in candidates])
        return [
            {name: float(index) for index, name in enumerate(FEATURES[:-1])}
            for _candidate_row in candidates
        ]

    return V412RetrievalFeatureService(
        partition_store=object(),
        tfidf_cache=object(),
        lookup=object(),
        ranker_feature_order=FEATURES,
        retriever=retriever,
        feature_builder=feature_builder,
        idf_builder=lambda pool: ({"CANDIDATE": 1.0}, 2.0),
    )


def test_build_uses_pre_lookup_selection_for_density_and_final_active_order() -> None:
    captured: list[list[dict[str, object]]] = []
    output = _service(_context(), captured).build(
        {
            "query_id": "q0",
            "crm_name": "Candidate",
            "crm_address": "1 rue exemple",
            "crm_postcode": "75001",
            "crm_city": "Paris",
            "crm_insee": "75101",
        }
    )

    assert output.selected_pre_lookup_count == 3
    assert output.lookup_missing_count == 0
    assert output.candidates["candidate_siret"].tolist() == [
        "00000000100001",
        "00000000200002",
    ]
    assert output.candidates["retrieval_source"].tolist() == [
        "sparse_address+sparse_name",
        "padding",
    ]
    assert output.candidates["retrieval_channel_count"].tolist() == [2, 0]
    assert output.candidates["retrieval_rank_recip"].tolist() == [1.0, 0.5]
    assert [row["_xgb_addr_density_insee"] for row in captured[0]] == [3, 3]
    assert list(output.candidates[FEATURES].dtypes.unique()) == [
        output.candidates["feature_0"].dtype
    ]


def test_forbidden_query_field_is_rejected_before_retrieval() -> None:
    captured: list[list[dict[str, object]]] = []
    with pytest.raises(ValueError, match="forbidden"):
        _service(_context(), captured).build(
            {"query_id": "q0", "ground_truth_siret": "00000000100001"}
        )
    assert captured == []


def test_snapshot_lookup_miss_is_fail_closed() -> None:
    captured: list[list[dict[str, object]]] = []
    with pytest.raises(ValueError, match="snapshot lookup miss"):
        _service(_context(missing=1), captured).build({"query_id": "q0"})
    assert captured == []


def test_ranker_feature_order_requires_45_unique_features() -> None:
    with pytest.raises(ValueError, match="feature order"):
        V412RetrievalFeatureService(
            partition_store=object(),
            tfidf_cache=object(),
            lookup=object(),
            ranker_feature_order=["retrieval_rank_recip"],
        )
