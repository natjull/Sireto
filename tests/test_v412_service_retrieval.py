from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pandas as pd
import pytest

from src.xgb_matcher import features as feature_module
from src.xgb_matcher.v411_scene import V411_ACCEPTOR_FEATURE_NAMES
from src.xgb_matcher.v412_service import V412DownstreamService
from src.xgb_matcher.v412_service_retrieval import (
    OUTPUT_COLUMNS,
    RANKER_C_FEATURE_ORDER,
    V412RetrievalFeatureService,
)
from src.xgb_matcher.v412_unit_retrieval import (
    UnitRetrievalContext,
    UnitRetrievalResult,
)


FEATURES = list(RANKER_C_FEATURE_ORDER)


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
    assert output.candidates["enseigne2"].tolist() == [None, None]
    assert str(output.candidates["enseigne2"].dtype) == "object"
    assert list(output.candidates[FEATURES].dtypes.unique()) == [
        output.candidates[FEATURES[0]].dtype
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


def test_unknown_45_feature_order_is_rejected() -> None:
    forged = [*[f"feature_{index}" for index in range(44)], "retrieval_rank_recip"]
    with pytest.raises(ValueError, match="feature order"):
        V412RetrievalFeatureService(
            partition_store=object(),
            tfidf_cache=object(),
            lookup=object(),
            ranker_feature_order=forged,
        )


def test_empty_final_pool_preserves_frozen_schema_and_dtypes() -> None:
    context = _context()
    empty = UnitRetrievalContext(
        result=UnitRetrievalResult(
            partition_key=context.result.partition_key,
            candidate_sirets=(),
            raw_pool_count=context.result.raw_pool_count,
            aligned_pool_count=context.result.aligned_pool_count,
            lookup_missing_count=0,
        ),
        aligned_pool=context.aligned_pool,
        selected_indices=context.selected_indices,
        scores_by_index=context.scores_by_index,
        channel_ranks_by_index=context.channel_ranks_by_index,
        snapshot_details=context.snapshot_details,
    )
    output = _service(empty, []).build({"query_id": "empty"})

    assert output.candidates.empty
    assert tuple(output.candidates.columns) == OUTPUT_COLUMNS
    assert str(output.candidates["retrieval_rank"].dtype) == "int64"
    assert str(output.candidates["retrieval_channel_count"].dtype) == "int32"
    assert str(output.candidates["retrieval_agreement"].dtype) == "int32"
    assert all(
        str(output.candidates[name].dtype) == "float32"
        for name in RANKER_C_FEATURE_ORDER
    )


def test_missing_frozen_feature_is_fail_closed() -> None:
    service = _service(_context(), [])

    def incomplete_builder(crm, candidates, *, include_semantic):
        return [
            {
                name: 0.0
                for name in FEATURES[:-1]
                if name != "idf_name"
            }
            for _candidate_row in candidates
        ]

    service.feature_builder = incomplete_builder
    with pytest.raises(ValueError, match="feature missing"):
        service.build({"query_id": "q0"})


def test_idf_context_is_isolated_between_concurrent_queries() -> None:
    state_lock = threading.Lock()
    active_builders = 0
    maximum_active_builders = 0

    def run(default_idf: float) -> float:
        nonlocal active_builders, maximum_active_builders
        captured: list[list[dict[str, object]]] = []
        service = _service(_context(), captured)
        service.idf_builder = lambda pool: ({"TARGET": default_idf}, default_idf)

        def concurrent_builder(crm, candidates, *, include_semantic):
            nonlocal active_builders, maximum_active_builders
            with state_lock:
                active_builders += 1
                maximum_active_builders = max(
                    maximum_active_builders,
                    active_builders,
                )
            try:
                time.sleep(0.03)
                value = feature_module._idf_overlap("TARGET", "TARGET")
                return [
                    {
                        name: value if name == "idf_name" else 0.0
                        for name in FEATURES[:-1]
                    }
                    for _candidate_row in candidates
                ]
            finally:
                with state_lock:
                    active_builders -= 1

        service.feature_builder = concurrent_builder
        result = service.build({"query_id": f"q{default_idf}"})
        return float(result.candidates.iloc[0]["idf_name"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        observed = sorted(executor.map(run, (2.0, 7.0)))
    assert observed == [2.0, 7.0]
    assert maximum_active_builders == 1


def test_empty_retrieval_output_routes_to_no_candidate_review() -> None:
    context = _context()
    empty = UnitRetrievalContext(
        result=UnitRetrievalResult(
            partition_key=context.result.partition_key,
            candidate_sirets=(),
            raw_pool_count=context.result.raw_pool_count,
            aligned_pool_count=context.result.aligned_pool_count,
            lookup_missing_count=0,
        ),
        aligned_pool=context.aligned_pool,
        selected_indices=context.selected_indices,
        scores_by_index=context.scores_by_index,
        channel_ranks_by_index=context.channel_ranks_by_index,
        snapshot_details=context.snapshot_details,
    )
    candidates = _service(empty, []).build({"query_id": "empty"}).candidates

    class ExplodingRanker:
        def predict(self, matrix):
            raise AssertionError("ranker must not be called for an empty pool")

    class Acceptor:
        def predict_proba(self, matrix):
            return [[0.9, 0.1]]

    def scene_builder(query, scored, taxonomy):
        assert scored.empty
        return {
            "predicted_siret": None,
            "predicted_siren": None,
            **{name: 0.0 for name in V411_ACCEPTOR_FEATURE_NAMES},
        }

    trace = V412DownstreamService(
        ranker=ExplodingRanker(),
        acceptor=Acceptor(),
        taxonomy=object(),
        ranker_feature_order=RANKER_C_FEATURE_ORDER,
        scene_builder=scene_builder,
    ).infer_one(
        query={"query_id": "empty"},
        candidates=candidates,
        direct_evidence={
            "query_id": "empty",
            "direct_candidate_count": 0,
            "sole_direct_siret": None,
        },
    )

    assert trace.decision_v412 == "REVIEW"
    assert trace.review_reason_v412 == "NO_CANDIDATE"
