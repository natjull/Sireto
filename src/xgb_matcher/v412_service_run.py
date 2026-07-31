"""Collect frozen persistent service traces without reading expected outputs."""

from __future__ import annotations

from dataclasses import dataclass
import os
import resource
import sys
from typing import Any, Mapping, Sequence

import pandas as pd

from .v411_scene import V411_ACCEPTOR_FEATURE_NAMES
from .v412_direct_evidence import (
    CANDIDATE_EVIDENCE_COLUMNS,
    QUERY_EVIDENCE_COLUMNS,
)
from .v412_service import FORBIDDEN_FIELDS
from .v412_service_retrieval import OUTPUT_COLUMNS
from .v412_service_worker import PersistentV412Worker, WorkerMode


SAFE_QUERY_COLUMNS = (
    "query_id",
    "crm_record_id",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
    "crm_name_norm",
    "crm_address_norm",
    "crm_city_norm",
)
RANKER_COLUMNS = (
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "retrieval_rank",
    "ranker_score",
    "prediction_origin",
    "oof_fold",
    "ranker_rank",
)
SCENE_COLUMNS = (
    "query_id",
    "crm_record_id",
    "predicted_siret",
    "predicted_siren",
    *V411_ACCEPTOR_FEATURE_NAMES,
)
ACCEPTOR_COLUMNS = (
    "model_family",
    "evaluation_partition",
    "query_id",
    "predicted_siret",
    "score",
    "threshold",
    "decision",
)
GUARD_COLUMNS = (
    "query_id",
    "predicted_siret",
    "predicted_siren",
    "acceptor_score",
    "acceptor_threshold",
    "decision_v411",
    "direct_candidate_count",
    "direct_siren_count",
    "sole_direct_siret",
    "sole_direct_siren",
    "decision_v412",
    "review_reason_v412",
)
TIMING_COLUMNS = (
    "query_id",
    "retrieval_lookup_ns",
    "hydrate_feature_ns",
    "ranker_ns",
    "scene_acceptor_ns",
    "evidence_route_load_index_ns",
    "evidence_search_aggregate_ns",
    "guard_ns",
    "evidence_guard_ns",
    "total_wall_ns",
)


@dataclass(frozen=True)
class CollectedServiceRun:
    mode: WorkerMode
    candidates: pd.DataFrame
    ranker: pd.DataFrame
    scenes: pd.DataFrame
    acceptor: pd.DataFrame
    query_evidence: pd.DataFrame
    candidate_evidence: pd.DataFrame
    guard: pd.DataFrame
    timings: pd.DataFrame
    manifest: dict[str, Any]


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _validate_queries(queries: pd.DataFrame) -> None:
    if tuple(queries.columns) != SAFE_QUERY_COLUMNS:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: safe query schema changed"
        )
    if len(queries) == 0:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: safe query population empty"
        )
    if FORBIDDEN_FIELDS & set(queries.columns):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: safe queries contain truth"
        )
    query_ids = queries["query_id"].astype(str)
    if query_ids.eq("").any() or query_ids.duplicated().any():
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: invalid safe query identities"
        )


def _frame(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(columns))


def collect_persistent_run(
    *,
    worker: PersistentV412Worker,
    queries: pd.DataFrame,
    model_load_count: int,
    store_load_count: int,
) -> CollectedServiceRun:
    """Warm one request, reset counters only, then collect canonical traces."""
    _validate_queries(queries)
    if (
        type(model_load_count) is not int
        or model_load_count < 0
        or type(store_load_count) is not int
        or store_load_count < 0
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: invalid observed load counters"
        )
    query_rows = queries.to_dict(orient="records")
    warmup_query_id = str(query_rows[0]["query_id"])
    worker.process(query_rows[0])
    worker.reset_counters()

    candidate_frames: list[pd.DataFrame] = []
    ranker_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    acceptor_rows: list[dict[str, Any]] = []
    query_evidence_rows: list[dict[str, Any]] = []
    candidate_evidence_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for query in query_rows:
        result = worker.process(query)
        query_id = result.query_id
        candidate_frames.append(result.retrieval.candidates)
        scored = result.v411.scored_candidates
        ranker_rows.extend(
            {
                "query_id": query_id,
                "candidate_siret": str(row.candidate_siret),
                "candidate_siren": str(row.candidate_siren),
                "retrieval_rank": int(row.retrieval_rank),
                "ranker_score": float(row.ranker_score),
                "prediction_origin": "ranker_c_dev",
                "oof_fold": pd.NA,
                "ranker_rank": int(row.ranker_rank),
            }
            for row in scored.itertuples(index=False)
        )
        scene_rows.append(
            {
                "query_id": query_id,
                "crm_record_id": str(query.get("crm_record_id") or ""),
                "predicted_siret": result.v411.predicted_siret,
                "predicted_siren": result.v411.predicted_siren,
                **{
                    name: float(result.v411.scene[name])
                    for name in V411_ACCEPTOR_FEATURE_NAMES
                },
            }
        )
        acceptor_rows.append(
            {
                "model_family": "COMPACT_LOGIT",
                "evaluation_partition": "threshold_dev",
                "query_id": query_id,
                "predicted_siret": result.v411.predicted_siret,
                "score": result.v411.acceptor_score,
                "threshold": result.v411.threshold,
                "decision": result.v411.decision_v411,
            }
        )
        if result.mode == "v412g":
            if result.evidence is None or result.v412 is None:
                raise AssertionError("V4.12-G trace lost evidence or guard")
            query_evidence_rows.append(dict(result.evidence.query))
            candidate_evidence_rows.extend(
                dict(row) for row in result.evidence.candidates
            )
            evidence_query = result.evidence.query
            guard_rows.append(
                {
                    "query_id": query_id,
                    "predicted_siret": result.v412.predicted_siret,
                    "predicted_siren": result.v412.predicted_siren,
                    "acceptor_score": result.v412.acceptor_score,
                    "acceptor_threshold": result.v412.threshold,
                    "decision_v411": result.v412.decision_v411,
                    "direct_candidate_count": int(
                        evidence_query["direct_candidate_count"]
                    ),
                    "direct_siren_count": int(
                        evidence_query["direct_siren_count"]
                    ),
                    "sole_direct_siret": evidence_query[
                        "sole_direct_siret"
                    ],
                    "sole_direct_siren": evidence_query[
                        "sole_direct_siren"
                    ],
                    "decision_v412": result.v412.decision_v412,
                    "review_reason_v412": (
                        result.v412.review_reason_v412
                    ),
                }
            )
        timing_rows.append(
            {
                "query_id": query_id,
                **{
                    field: int(getattr(result.timings, field))
                    for field in TIMING_COLUMNS[1:-2]
                },
                "evidence_guard_ns": int(
                    result.timings.evidence_guard_ns
                ),
                "total_wall_ns": int(result.timings.total_wall_ns),
            }
        )

    candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
    )[list(OUTPUT_COLUMNS)]
    ranker = _frame(ranker_rows, RANKER_COLUMNS).astype(
        {
            "query_id": "string",
            "candidate_siret": "string",
            "candidate_siren": "string",
            "retrieval_rank": "int64",
            "ranker_score": "float32",
            "prediction_origin": "string",
            "oof_fold": "Int8",
            "ranker_rank": "int64",
        }
    )
    scenes = _frame(scene_rows, SCENE_COLUMNS).astype(
        {name: "float64" for name in V411_ACCEPTOR_FEATURE_NAMES}
    )
    acceptor = _frame(acceptor_rows, ACCEPTOR_COLUMNS).astype(
        {"score": "float64", "threshold": "float64"}
    )
    query_evidence = _frame(
        query_evidence_rows,
        QUERY_EVIDENCE_COLUMNS,
    )
    candidate_evidence = _frame(
        candidate_evidence_rows,
        CANDIDATE_EVIDENCE_COLUMNS,
    )
    guard = _frame(guard_rows, GUARD_COLUMNS)
    timings = _frame(timing_rows, TIMING_COLUMNS).astype(
        {name: "int64" for name in TIMING_COLUMNS[1:]}
    )
    counters = worker.counters()
    evidence_service = worker.bundle.evidence
    manifest = {
        "schema_version": "sireto-v4.12-persistent-service-run-1",
        "mode": worker.mode,
        "pid": os.getpid(),
        "warmup_query_id": warmup_query_id,
        "warmup_excluded": True,
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "peak_rss_bytes": peak_rss_bytes(),
        "counters": counters,
        "counter_basis": {
            "sealed_key_miss_count": (
                "observed route, StrictPartitionStore and "
                "StrictVerifiedTfidfCache missing-key exceptions"
            ),
            "cache_rebuild_count": (
                "observed operations; rebuild API structurally absent"
            ),
            "cache_write_count": (
                "observed operations; write API structurally absent"
            ),
        },
        "evidence_cache": {
            "hit_count": (
                evidence_service.cache_hit_count
                if evidence_service is not None
                else 0
            ),
            "miss_count": (
                evidence_service.cache_miss_count
                if evidence_service is not None
                else 0
            ),
            "eviction_count": (
                evidence_service.cache_eviction_count
                if evidence_service is not None
                else 0
            ),
        },
        "asset_hashes": dict(worker.bundle.asset_hashes),
        "model_load_count": model_load_count,
        "store_load_count": store_load_count,
    }
    if (
        counters["query_count"] != len(queries)
        or counters["lookup_missing_count"] != 0
        or counters["maximum_candidate_count"] > 100
        or counters["sealed_key_miss_count"] != 0
        or counters["cache_rebuild_count"] != 0
        or counters["cache_write_count"] != 0
        or counters["cache_rebuild_api_absent"] is not True
        or counters["cache_write_api_absent"] is not True
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: persistent worker counters failed"
        )
    return CollectedServiceRun(
        mode=worker.mode,
        candidates=candidates,
        ranker=ranker,
        scenes=scenes,
        acceptor=acceptor,
        query_evidence=query_evidence,
        candidate_evidence=candidate_evidence,
        guard=guard,
        timings=timings,
        manifest=manifest,
    )


__all__ = [
    "ACCEPTOR_COLUMNS",
    "CollectedServiceRun",
    "GUARD_COLUMNS",
    "RANKER_COLUMNS",
    "SAFE_QUERY_COLUMNS",
    "SCENE_COLUMNS",
    "TIMING_COLUMNS",
    "collect_persistent_run",
    "peak_rss_bytes",
]
