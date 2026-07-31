from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.xgb_matcher.v412_evidence_service import (
    EvidenceResult,
    EvidenceTimings,
)
from src.xgb_matcher.v412_service import (
    ServiceTimings,
    ServiceTrace,
    V411Trace,
)
from src.xgb_matcher.v412_service_retrieval import (
    RetrievalFeatureResult,
    RetrievalFeatureTimings,
)
from src.xgb_matcher.v412_service_worker import PersistentV412Worker


class _Retrieval:
    def build(self, query):
        return RetrievalFeatureResult(
            candidates=pd.DataFrame({"query_id": [query["query_id"]]}),
            partition_key="75056_",
            raw_pool_count=1,
            aligned_pool_count=1,
            lookup_missing_count=0,
            selected_pre_lookup_count=1,
            timings=RetrievalFeatureTimings(
                retrieval_lookup_ns=10,
                hydrate_feature_ns=20,
            ),
        )


class _Evidence:
    def __init__(self):
        self.calls = 0

    def build(self, query):
        self.calls += 1
        return EvidenceResult(
            query={
                "query_id": query["query_id"],
                "direct_candidate_count": 0,
                "sole_direct_siret": None,
            },
            candidates=(),
            timings=EvidenceTimings(
                route_load_index_ns=30,
                search_aggregate_ns=40,
            ),
        )


class _Downstream:
    def rank_and_accept_one(self, *, query, candidates):
        return V411Trace(
            query_id=query["query_id"],
            predicted_siret=None,
            predicted_siren=None,
            acceptor_score=0.1,
            threshold=0.8,
            decision_v411="REVIEW",
            review_reason_v411="NO_CANDIDATE",
            scored_candidates=candidates,
            scene={},
            ranker_ns=50,
            scene_acceptor_ns=60,
        )

    def apply_guard_to_trace(self, *, trace, direct_evidence):
        return ServiceTrace(
            query_id=trace.query_id,
            predicted_siret=trace.predicted_siret,
            predicted_siren=trace.predicted_siren,
            acceptor_score=trace.acceptor_score,
            threshold=trace.threshold,
            decision_v411=trace.decision_v411,
            review_reason_v411=trace.review_reason_v411,
            decision_v412="REVIEW",
            review_reason_v412="NO_CANDIDATE",
            scored_candidates=trace.scored_candidates,
            scene=trace.scene,
            timings=ServiceTimings(
                ranker_ns=trace.ranker_ns,
                scene_acceptor_ns=trace.scene_acceptor_ns,
                guard_ns=70,
            ),
        )


def _bundle(*, evidence):
    return SimpleNamespace(
        retrieval=_Retrieval(),
        downstream=_Downstream(),
        evidence=evidence,
    )


def test_v411_worker_never_loads_or_calls_evidence() -> None:
    worker = PersistentV412Worker(
        bundle=_bundle(evidence=None),
        mode="v411",
    )
    result = worker.process({"query_id": "q0"})

    assert result.evidence is None
    assert result.v412 is None
    assert result.timings.evidence_guard_ns == 0
    assert worker.counters() == {
        "query_count": 1,
        "lookup_missing_count": 0,
        "maximum_candidate_count": 1,
        "sealed_key_miss_count": 0,
        "cache_rebuild_count": 0,
        "cache_write_count": 0,
    }


def test_v412g_worker_measures_evidence_and_guard_separately() -> None:
    evidence = _Evidence()
    worker = PersistentV412Worker(
        bundle=_bundle(evidence=evidence),
        mode="v412g",
    )
    result = worker.process({"query_id": "q0"})

    assert evidence.calls == 1
    assert result.v412 is not None
    assert result.timings.evidence_route_load_index_ns == 30
    assert result.timings.evidence_search_aggregate_ns == 40
    assert result.timings.guard_ns == 70
    assert result.timings.evidence_guard_ns == 140
    assert result.timings.total_wall_ns > 0


def test_worker_mode_and_bundle_shape_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid worker mode"):
        PersistentV412Worker(bundle=_bundle(evidence=None), mode="other")
    with pytest.raises(ValueError, match="loaded evidence"):
        PersistentV412Worker(bundle=_bundle(evidence=_Evidence()), mode="v411")
    with pytest.raises(ValueError, match="evidence service absent"):
        PersistentV412Worker(bundle=_bundle(evidence=None), mode="v412g")
