"""Persistent label-blind V4.12 retrieval and Ranker-C feature serving."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .blocking import attach_address_density
from .candidates import compute_name_idf_map
from .features import (
    make_feature_rows_from_preprocessed,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from .v412_service import (
    CANDIDATE_CEILING,
    FORBIDDEN_FIELDS,
    RANKER_C_FEATURE_ORDER,
    RANKER_C_FEATURE_ORDER_SHA256,
)
from .v412_unit_retrieval import (
    UnitRetrievalContext,
    retrieve_unit_query_context,
)


ROLE_COLUMNS = (
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "denomination_usuelle",
    "activity_code",
)
OUTPUT_COLUMNS = (
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "candidate_state",
    "retrieval_rank",
    "retrieval_source",
    "retrieval_channel_count",
    "retrieval_agreement",
    *ROLE_COLUMNS,
    *RANKER_C_FEATURE_ORDER,
)


@dataclass(frozen=True)
class RetrievalFeatureTimings:
    retrieval_lookup_ns: int
    hydrate_feature_ns: int

    @property
    def total_ns(self) -> int:
        return self.retrieval_lookup_ns + self.hydrate_feature_ns


@dataclass(frozen=True)
class RetrievalFeatureResult:
    candidates: pd.DataFrame
    partition_key: str
    raw_pool_count: int
    aligned_pool_count: int
    lookup_missing_count: int
    selected_pre_lookup_count: int
    timings: RetrievalFeatureTimings


Retriever = Callable[..., UnitRetrievalContext]
FeatureBuilder = Callable[..., list[dict[str, Any]]]
IdfBuilder = Callable[
    [dict[str, dict[str, Any]]],
    tuple[dict[str, float], float],
]


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"STOP_V412_SERVICE_INTEGRITY: non-numeric {label}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(
            f"STOP_V412_SERVICE_INTEGRITY: non-finite {label}"
        )
    return result


def _crm_preprocessed(query: Mapping[str, Any]) -> dict[str, Any]:
    query_id = str(query.get("query_id") or "")
    return preprocess_crm_row(
        {
            "query_id": query_id,
            "crm_id": query_id,
            "crm_name": str(query.get("crm_name") or ""),
            "crm_address": str(query.get("crm_address") or ""),
            "crm_city": str(query.get("crm_city") or ""),
            "postcode": str(query.get("crm_postcode") or ""),
            "insee": str(query.get("crm_insee") or ""),
        }
    )


class V412RetrievalFeatureService:
    """Build the exact active top-100 and 45 frozen Ranker-C features."""

    def __init__(
        self,
        *,
        partition_store: Any,
        tfidf_cache: Any,
        lookup: Any,
        ranker_feature_order: Sequence[str],
        retriever: Retriever = retrieve_unit_query_context,
        feature_builder: FeatureBuilder = make_feature_rows_from_preprocessed,
        idf_builder: IdfBuilder = compute_name_idf_map,
    ) -> None:
        if tuple(ranker_feature_order) != RANKER_C_FEATURE_ORDER:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: Ranker C feature order changed"
            )
        self.partition_store = partition_store
        self.tfidf_cache = tfidf_cache
        self.lookup = lookup
        self.ranker_feature_order = tuple(ranker_feature_order)
        self.retriever = retriever
        self.feature_builder = feature_builder
        self.idf_builder = idf_builder

    def build(self, query: Mapping[str, Any]) -> RetrievalFeatureResult:
        if FORBIDDEN_FIELDS & set(query):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: query contains forbidden fields"
            )
        query_id = str(query.get("query_id") or "")
        if not query_id:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: empty query identity"
            )

        retrieval_started = time.perf_counter_ns()
        context = self.retriever(
            query=query,
            partition_store=self.partition_store,
            tfidf_cache=self.tfidf_cache,
            lookup=self.lookup,
        )
        retrieval_lookup_ns = time.perf_counter_ns() - retrieval_started
        result = context.result
        if result.lookup_missing_count != 0:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: snapshot lookup miss"
            )
        if len(result.candidate_sirets) > CANDIDATE_CEILING:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: candidate ceiling exceeded"
            )

        feature_started = time.perf_counter_ns()
        aligned_by_siret: dict[str, dict[str, Any]] = {}
        for raw in context.aligned_pool:
            candidate = dict(raw)
            siret = str(candidate.get("siret") or "")
            if not siret or siret in aligned_by_siret:
                raise ValueError(
                    "STOP_V412_SERVICE_INTEGRITY: aligned pool identity changed"
                )
            aligned_by_siret[siret] = candidate
        idf_map, default_idf = self.idf_builder(aligned_by_siret)
        if not math.isfinite(float(default_idf)) or any(
            not math.isfinite(float(value)) for value in idf_map.values()
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid IDF context"
            )

        selected: list[dict[str, Any]] = []
        selected_index_by_siret: dict[str, int] = {}
        for index in context.selected_indices:
            if index < 0 or index >= len(context.aligned_pool):
                raise ValueError(
                    "STOP_V412_SERVICE_INTEGRITY: selected index out of bounds"
                )
            candidate = dict(context.aligned_pool[index])
            siret = str(candidate.get("siret") or "")
            if not siret or siret in selected_index_by_siret:
                raise ValueError(
                    "STOP_V412_SERVICE_INTEGRITY: selected identity changed"
                )
            selected_index_by_siret[siret] = index
            selected.append(candidate)

        # Historical V4.11 density is computed on the pre-lookup top-100.
        attach_address_density(selected)
        selected_by_siret = {
            str(candidate["siret"]): candidate for candidate in selected
        }
        final_candidates: list[dict[str, Any]] = []
        for siret in result.candidate_sirets:
            candidate = dict(selected_by_siret.get(siret) or {})
            detail = context.snapshot_details.get(siret)
            if not candidate or not isinstance(detail, Mapping):
                raise ValueError(
                    "STOP_V412_SERVICE_INTEGRITY: hydration identity changed"
                )
            if detail.get("candidate_state") != "A":
                raise ValueError(
                    "STOP_V412_SERVICE_INTEGRITY: inactive final candidate"
                )
            candidate.update(
                {
                    "siret": siret,
                    "siren": str(candidate.get("siren") or siret[:9]),
                    "etat_admin": "A",
                    "enseigne1": detail.get("enseigne1"),
                    "enseigne2": detail.get("enseigne2"),
                    "enseigne3": detail.get("enseigne3"),
                    "denomination_usuelle": detail.get(
                        "denomination_usuelle"
                    ),
                    "activity_code": detail.get("activity_code"),
                }
            )
            final_candidates.append(candidate)

        set_global_name_idf_map(idf_map, float(default_idf))
        feature_rows = self.feature_builder(
            _crm_preprocessed(query),
            final_candidates,
            include_semantic=False,
        )
        if len(feature_rows) != len(final_candidates):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: feature row count changed"
            )

        rows: list[dict[str, Any]] = []
        for rank, (candidate, features) in enumerate(
            zip(final_candidates, feature_rows, strict=True),
            start=1,
        ):
            siret = str(candidate["siret"])
            index = selected_index_by_siret[siret]
            channel_ranks = context.channel_ranks_by_index.get(index, {})
            sources = sorted(
                channel
                for channel in channel_ranks
                if channel != "dense"
            )
            source = "+".join(sources) if sources else "padding"
            missing_features = set(self.ranker_feature_order[:-1]) - set(
                features
            )
            if missing_features:
                raise ValueError(
                    "STOP_V412_SERVICE_INTEGRITY: frozen Ranker C feature "
                    f"missing: {sorted(missing_features)}"
                )
            feature_values = {
                name: _finite(features[name], name)
                for name in self.ranker_feature_order[:-1]
            }
            feature_values["retrieval_rank_recip"] = 1.0 / rank
            rows.append(
                {
                    "query_id": query_id,
                    "candidate_siret": siret,
                    "candidate_siren": str(
                        candidate.get("siren") or siret[:9]
                    ),
                    "candidate_state": "A",
                    "retrieval_rank": rank,
                    "retrieval_source": source,
                    "retrieval_channel_count": len(sources),
                    "retrieval_agreement": int(len(sources) >= 2),
                    **{
                        column: candidate.get(column)
                        for column in ROLE_COLUMNS
                    },
                    **feature_values,
                }
            )
        frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        frame = frame.astype(
            {
                "query_id": "string",
                "candidate_siret": "string",
                "candidate_siren": "string",
                "candidate_state": "string",
                "retrieval_rank": "int64",
                "retrieval_source": "string",
                "retrieval_channel_count": "int32",
                "retrieval_agreement": "int32",
                **{
                    name: "float32"
                    for name in self.ranker_feature_order
                },
            }
        )
        if len(frame):
            matrix = frame[list(self.ranker_feature_order)].to_numpy(
                dtype=np.float32
            )
            if not np.isfinite(matrix).all():
                raise ValueError(
                    "STOP_V412_SERVICE_INTEGRITY: non-finite feature matrix"
                )
        hydrate_feature_ns = time.perf_counter_ns() - feature_started
        return RetrievalFeatureResult(
            candidates=frame,
            partition_key=result.partition_key,
            raw_pool_count=result.raw_pool_count,
            aligned_pool_count=result.aligned_pool_count,
            lookup_missing_count=result.lookup_missing_count,
            selected_pre_lookup_count=len(context.selected_indices),
            timings=RetrievalFeatureTimings(
                retrieval_lookup_ns=retrieval_lookup_ns,
                hydrate_feature_ns=hydrate_feature_ns,
            ),
        )


__all__ = [
    "OUTPUT_COLUMNS",
    "RANKER_C_FEATURE_ORDER",
    "RANKER_C_FEATURE_ORDER_SHA256",
    "ROLE_COLUMNS",
    "RetrievalFeatureResult",
    "RetrievalFeatureTimings",
    "V412RetrievalFeatureService",
]
