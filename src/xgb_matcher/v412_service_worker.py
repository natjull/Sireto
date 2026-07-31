"""Single-process persistent request worker for frozen V4.11/V4.12-G."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal, Mapping

from .v412_evidence_service import EvidenceResult
from .v412_service import ServiceTrace, V411Trace
from .v412_service_bundle import (
    FrozenV412ServiceBundle,
    validate_frozen_v412_service_bundle,
)
from .v412_service_retrieval import RetrievalFeatureResult
from .v412_unit_retrieval import UnitRetrievalError


WorkerMode = Literal["v411", "v412g"]


@dataclass(frozen=True)
class WorkerTimings:
    retrieval_lookup_ns: int
    hydrate_feature_ns: int
    ranker_ns: int
    scene_acceptor_ns: int
    evidence_route_load_index_ns: int
    evidence_search_aggregate_ns: int
    guard_ns: int
    total_wall_ns: int

    @property
    def evidence_guard_ns(self) -> int:
        return (
            self.evidence_route_load_index_ns
            + self.evidence_search_aggregate_ns
            + self.guard_ns
        )


@dataclass(frozen=True)
class WorkerResult:
    mode: WorkerMode
    query_id: str
    retrieval: RetrievalFeatureResult
    v411: V411Trace
    evidence: EvidenceResult | None
    v412: ServiceTrace | None
    timings: WorkerTimings


class PersistentV412Worker:
    """Execute many requests while retaining all frozen assets and caches."""

    def __init__(
        self,
        *,
        bundle: FrozenV412ServiceBundle,
        mode: WorkerMode,
    ) -> None:
        validate_frozen_v412_service_bundle(bundle)
        if mode not in {"v411", "v412g"}:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid worker mode"
            )
        if mode == "v411" and bundle.evidence is not None:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: V4.11 loaded evidence service"
            )
        if mode == "v412g" and bundle.evidence is None:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: V4.12-G evidence service absent"
            )
        self.bundle = bundle
        self.mode = mode
        self.query_count = 0
        self.lookup_missing_count = 0
        self.maximum_candidate_count = 0
        self.route_sealed_key_miss_count = 0

    def process(self, query: Mapping[str, Any]) -> WorkerResult:
        validate_frozen_v412_service_bundle(self.bundle)
        wall_started = time.perf_counter_ns()
        try:
            retrieval = self.bundle.retrieval.build(query)
        except UnitRetrievalError as exc:
            if "no frozen partition for query" in str(exc):
                self.route_sealed_key_miss_count += 1
            raise
        self.lookup_missing_count += retrieval.lookup_missing_count
        self.maximum_candidate_count = max(
            self.maximum_candidate_count,
            len(retrieval.candidates),
        )
        v411 = self.bundle.downstream.rank_and_accept_one(
            query=query,
            candidates=retrieval.candidates,
        )

        evidence: EvidenceResult | None = None
        v412: ServiceTrace | None = None
        evidence_route_load_index_ns = 0
        evidence_search_aggregate_ns = 0
        guard_ns = 0
        if self.mode == "v412g":
            evidence_service = self.bundle.evidence
            if evidence_service is None:
                raise AssertionError("validated evidence service disappeared")
            evidence = evidence_service.build(query)
            v412 = self.bundle.downstream.apply_guard_to_trace(
                trace=v411,
                direct_evidence=evidence.query,
            )
            evidence_route_load_index_ns = (
                evidence.timings.route_load_index_ns
            )
            evidence_search_aggregate_ns = (
                evidence.timings.search_aggregate_ns
            )
            guard_ns = v412.timings.guard_ns

        total_wall_ns = time.perf_counter_ns() - wall_started
        self.query_count += 1
        return WorkerResult(
            mode=self.mode,
            query_id=v411.query_id,
            retrieval=retrieval,
            v411=v411,
            evidence=evidence,
            v412=v412,
            timings=WorkerTimings(
                retrieval_lookup_ns=(
                    retrieval.timings.retrieval_lookup_ns
                ),
                hydrate_feature_ns=(
                    retrieval.timings.hydrate_feature_ns
                ),
                ranker_ns=v411.ranker_ns,
                scene_acceptor_ns=v411.scene_acceptor_ns,
                evidence_route_load_index_ns=evidence_route_load_index_ns,
                evidence_search_aggregate_ns=evidence_search_aggregate_ns,
                guard_ns=guard_ns,
                total_wall_ns=total_wall_ns,
            ),
        )

    def counters(self) -> dict[str, int | bool]:
        evidence = self.bundle.evidence
        sealed_misses = (
            self.route_sealed_key_miss_count
            + self.bundle.partition_store.sealed_key_miss_count
            + self.bundle.tfidf_cache.sealed_key_miss_count
        )
        return {
            "query_count": self.query_count,
            "lookup_missing_count": self.lookup_missing_count,
            "maximum_candidate_count": self.maximum_candidate_count,
            "sealed_key_miss_count": sealed_misses,
            "cache_rebuild_count": (
                self.bundle.tfidf_cache.cache_rebuild_count
            ),
            "cache_write_count": (
                self.bundle.tfidf_cache.cache_write_count
            ),
            "cache_rebuild_api_absent": (
                self.bundle.tfidf_cache.rebuild_api_absent
            ),
            "cache_write_api_absent": (
                self.bundle.tfidf_cache.write_api_absent
            ),
            "evidence_cache_hit_count": (
                evidence.cache_hit_count if evidence is not None else 0
            ),
            "evidence_cache_miss_count": (
                evidence.cache_miss_count if evidence is not None else 0
            ),
            "evidence_cache_eviction_count": (
                evidence.cache_eviction_count if evidence is not None else 0
            ),
        }

    def reset_counters(self) -> None:
        """Exclude warm-up requests without clearing persistent caches."""
        self.query_count = 0
        self.lookup_missing_count = 0
        self.maximum_candidate_count = 0
        self.route_sealed_key_miss_count = 0
        self.bundle.partition_store.sealed_key_miss_count = 0
        self.bundle.tfidf_cache.sealed_key_miss_count = 0
        self.bundle.tfidf_cache.cache_rebuild_count = 0
        self.bundle.tfidf_cache.cache_write_count = 0
        evidence = self.bundle.evidence
        if evidence is not None:
            evidence.cache_hit_count = 0
            evidence.cache_miss_count = 0
            evidence.cache_eviction_count = 0


__all__ = [
    "PersistentV412Worker",
    "WorkerMode",
    "WorkerResult",
    "WorkerTimings",
]
