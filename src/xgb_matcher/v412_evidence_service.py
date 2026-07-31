"""Persistent full-universe V4.12 direct-evidence service."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping

import pandas as pd

from .blocking import normalize_code
from .v412_direct_evidence import (
    CANDIDATE_EVIDENCE_COLUMNS,
    QUERY_EVIDENCE_COLUMNS,
    candidate_evidence_record,
    query_evidence_record,
    validate_evidence,
)
from .v412_service import FORBIDDEN_FIELDS

# Frozen from data/candidates_v7_all/manifest/insee_counts.parquet
# (SHA-256 a07bf9cd084f2f8e4842c30b545a913a93da7096a5d9c9a81d3f48c8b866ab0a)
# under the V4.12 snapshot policy (row_count > 100_000). Pinning the set in
# code avoids consulting mutable legacy manifests while preserving the route.
_FROZEN_MEGA_INSEE = frozenset(
    {
        "06029", "06088", "13001", "21231", "30189", "31555", "33063",
        "34172", "35238", "38185", "42218", "44109", "51454", "59350",
        "66136", "67482", "69266", "69383", "74010", "75101", "75102",
        "75107", "75108", "75109", "75110", "75111", "75112", "75113",
        "75114", "75115", "75116", "75117", "75118", "75119", "75120",
        "76540", "83137", "92012", "97209",
    }
)


@dataclass(frozen=True)
class EvidenceTimings:
    route_load_index_ns: int
    search_aggregate_ns: int

    @property
    def total_ns(self) -> int:
        return self.route_load_index_ns + self.search_aggregate_ns


@dataclass(frozen=True)
class EvidenceResult:
    query: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    timings: EvidenceTimings


class V412DirectEvidenceService:
    """Cache complete active geographic indexes, never top-100 candidates."""

    def __init__(
        self,
        *,
        partition_store: Any,
        max_index_cache_entries: int = 5,
        route: Callable[..., str] | None = None,
        load_partition: Callable[..., list[dict[str, Any]]] | None = None,
        build_index: Callable[..., Any] | None = None,
        search: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> None:
        if (
            type(max_index_cache_entries) is not int
            or max_index_cache_entries <= 0
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid evidence cache bound"
            )
        supplied = (route, load_partition, build_index, search)
        if any(function is not None for function in supplied) and any(
            function is None for function in supplied
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: evidence functions must be "
                "fully injected or fully frozen"
            )
        if all(function is None for function in supplied):
            from scripts.build_benchmark_v4_current_snapshot import (
                build_active_partition_index,
                find_direct_active_candidates,
            )

            route = _strict_partition_key
            load_partition = _load_strict_partition
            build_index = build_active_partition_index
            search = find_direct_active_candidates
        self.partition_store = partition_store
        self.max_index_cache_entries = max_index_cache_entries
        self.route = route
        self.load_partition = load_partition
        self.build_index = build_index
        self.search = search
        self._indexes: OrderedDict[str, Any] = OrderedDict()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.cache_eviction_count = 0

    def build(self, query: Mapping[str, Any]) -> EvidenceResult:
        if FORBIDDEN_FIELDS & set(query):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: evidence query contains forbidden fields"
            )
        query_id = str(query.get("query_id") or "")
        if not query_id:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: empty evidence query identity"
            )
        work = {
            **dict(query),
            "split": "v412_service_label_free",
            "postcode": str(query.get("crm_postcode") or ""),
            "insee": str(query.get("crm_insee") or ""),
        }

        route_started = time.perf_counter_ns()
        partition_key = self.route(work, self.partition_store)
        if partition_key in self._indexes:
            self.cache_hit_count += 1
            self._indexes.move_to_end(partition_key)
            index = self._indexes[partition_key]
        else:
            self.cache_miss_count += 1
            rows = self.load_partition(partition_key, self.partition_store)
            index = self.build_index(rows)
            while len(self._indexes) >= self.max_index_cache_entries:
                self._indexes.popitem(last=False)
                self.cache_eviction_count += 1
            self._indexes[partition_key] = index
        route_load_index_ns = time.perf_counter_ns() - route_started

        search_started = time.perf_counter_ns()
        direct = self.search(
            work,
            index,
            partition_key=partition_key,
        )
        candidate_rows = tuple(
            candidate_evidence_record(query_id, row) for row in direct
        )
        query_row = query_evidence_record(
            query_id=query_id,
            partition_key=partition_key,
            active_universe_count=int(index.active_count),
            candidates=candidate_rows,
        )
        query_frame = pd.DataFrame(
            [query_row], columns=QUERY_EVIDENCE_COLUMNS
        )
        candidate_frame = pd.DataFrame(
            candidate_rows, columns=CANDIDATE_EVIDENCE_COLUMNS
        )
        validate_evidence(query_frame, candidate_frame)
        search_aggregate_ns = time.perf_counter_ns() - search_started
        return EvidenceResult(
            query=query_row,
            candidates=candidate_rows,
            timings=EvidenceTimings(
                route_load_index_ns=route_load_index_ns,
                search_aggregate_ns=search_aggregate_ns,
            ),
        )


def _strict_partition_key(
    row: Mapping[str, Any],
    store: Any,
) -> str:
    """Apply the frozen evidence policy through the sealed strict store."""
    keys = frozenset(store.partition_keys)
    insee = normalize_code(row.get("insee"))
    postcode = normalize_code(row.get("postcode"))
    if insee:
        strict_insee_key = f"{insee}_"
        if strict_insee_key in keys:
            if insee in _FROZEN_MEGA_INSEE and postcode:
                return f"insee_cp:{insee}:{postcode}"
            return f"insee:{insee}"
    if postcode:
        strict_postcode_key = f"_{postcode}"
        if strict_postcode_key in keys:
            return f"cp:{postcode}"
    return "none"


def _load_strict_partition(
    key: str,
    store: Any,
) -> list[dict[str, Any]]:
    """Load only files verified and allow-listed by ``StrictPartitionStore``."""
    if key == "none":
        return []
    kind, payload = key.split(":", maxsplit=1)
    if kind == "insee":
        return store.load(f"{payload}_")
    if kind == "cp":
        return store.load(f"_{payload}")
    if kind == "insee_cp":
        insee, postcode = payload.split(":", maxsplit=1)
        return [
            row
            for row in store.load(f"{insee}_")
            if normalize_code(row.get("postcode")) == postcode
        ]
    raise ValueError(
        "STOP_V412_SERVICE_INTEGRITY: unsupported evidence partition key"
    )


__all__ = [
    "EvidenceResult",
    "EvidenceTimings",
    "V412DirectEvidenceService",
]
