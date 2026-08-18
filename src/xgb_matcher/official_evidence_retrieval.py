"""Detailed pre-admission union over the national and official-evidence indices.

This adapter is intentionally not wired into the legacy top-100 path.  It
exposes raw per-channel ranks/scores/sources for downstream admission research,
with a hard internal-union cap of 2,000 and no RRF or model-derived fusion.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .hierarchical_retrieval import (
    IndexedEstablishment,
    RetrievalBackend,
    RetrievalQuery,
    TantivyBackend,
)
from .official_evidence_tantivy import OfficialEvidenceTantivyBackend


OFFICIAL_EVIDENCE_UNION_SCHEMA_VERSION = "sireto-official-evidence-union-v1"


class SirenRelationBackend(Protocol):
    def linked_sirens(self, siren: str, limit: int = 256) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class OfficialEvidenceRetrievalConfig:
    max_union_candidates: int = 2000
    exact_limit: int = 500
    word_limit: int = 1000
    character_limit: int = 1000
    siren_limit: int = 10
    sites_per_siren: int = 32
    relation_seed_limit: int = 500
    relation_limit: int = 500
    search_workers: int = 12

    def __post_init__(self) -> None:
        if not 1 <= self.max_union_candidates <= 2000:
            raise ValueError("max_union_candidates must be between 1 and 2000")
        if not 1 <= self.sites_per_siren <= 32:
            raise ValueError("sites_per_siren must be between 1 and 32")
        for name in (
            "exact_limit",
            "word_limit",
            "character_limit",
            "siren_limit",
            "relation_seed_limit",
            "relation_limit",
            "search_workers",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class OfficialEvidenceSignal:
    source: str
    channel: str
    rank: int
    score: float
    parent_identifier: str = ""
    parent_rank: int | None = None
    site_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "channel": self.channel,
            "rank": self.rank,
            "score": self.score,
            "parent_identifier": self.parent_identifier,
            "parent_rank": self.parent_rank,
            "site_rank": self.site_rank,
        }


@dataclass(frozen=True)
class OfficialEvidenceCandidate:
    union_rank: int
    siret: str
    siren: str
    record: IndexedEstablishment
    signals: tuple[OfficialEvidenceSignal, ...]
    official_evidence_ids: tuple[str, ...] = ()
    official_evidence_sources: tuple[str, ...] = ()

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({signal.source for signal in self.signals}))

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(sorted({signal.channel for signal in self.signals}))

    def to_dict(
        self,
        *,
        query_id: str,
        raw_union_size: int,
        union_truncated: bool,
        retrieval_latency_ms: float,
    ) -> dict[str, Any]:
        best_signal = min(
            self.signals,
            key=lambda signal: (
                _CHANNEL_PRIORITY[signal.channel], signal.rank, signal.source
            ),
        )
        channel_values: dict[str, Any] = {}
        for channel in LTR_CHANNELS:
            values = [signal for signal in self.signals if signal.channel == channel]
            best = min(values, key=lambda signal: (signal.rank, -signal.score)) if values else None
            channel_values[f"{channel}_rank"] = best.rank if best else None
            channel_values[f"{channel}_score"] = best.score if best else None
        return {
            "schema_version": OFFICIAL_EVIDENCE_UNION_SCHEMA_VERSION,
            "query_id": str(query_id),
            "union_rank": self.union_rank,
            "siret": self.siret,
            "siren": self.siren,
            "sources": list(self.sources),
            "channels": list(self.channels),
            "signals": [signal.to_dict() for signal in self.signals],
            "official_evidence_ids": list(self.official_evidence_ids),
            "official_evidence_sources": list(self.official_evidence_sources),
            "candidate_names": list(self.record.names),
            "candidate_addresses": list(self.record.addresses),
            "candidate_insee": self.record.insee,
            "candidate_postcode": self.record.postcode,
            "candidate_number": self.record.number,
            "candidate_state": "A" if self.record.active else "F",
            "candidate_is_siege": self.record.is_siege,
            "retrieval_source": "+".join(self.sources),
            "retrieval_rank": self.union_rank,
            "retrieval_score": best_signal.score,
            **channel_values,
            "raw_union_size": raw_union_size,
            "union_truncated": union_truncated,
            "retrieval_latency_ms": retrieval_latency_ms,
            "candidate_present": True,
        }


@dataclass(frozen=True)
class OfficialEvidenceRetrievalResult:
    query: RetrievalQuery
    candidates: tuple[OfficialEvidenceCandidate, ...]
    raw_union_size: int
    union_truncated: bool
    retrieval_latency_ms: float

    def candidate_rows(
        self,
        query_id: str,
        query_metadata: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        rows = [
            {
                **dict(query_metadata or {}),
                **candidate.to_dict(
                    query_id=query_id,
                    raw_union_size=self.raw_union_size,
                    union_truncated=self.union_truncated,
                    retrieval_latency_ms=self.retrieval_latency_ms,
                ),
            }
            for candidate in self.candidates
        ]
        if rows:
            return rows
        # No fake/SIRET-empty candidate is emitted.  Missing-query accounting
        # belongs in retrieval diagnostics, not in an LTR candidate table.
        return []


@dataclass
class _MutableCandidate:
    record: IndexedEstablishment
    signals: list[OfficialEvidenceSignal] = field(default_factory=list)
    names: set[str] = field(default_factory=set)
    addresses: set[str] = field(default_factory=set)
    evidence_ids: set[str] = field(default_factory=set)
    evidence_sources: set[str] = field(default_factory=set)


_CHANNEL_PRIORITY: Mapping[str, int] = {
    "name_exact": 0,
    "address_exact": 1,
    "name_word": 2,
    "address_word": 3,
    "name_char": 4,
    "address_char": 5,
    "siren_expansion": 6,
    "establishment_relation": 7,
    "siren_relation_expansion": 8,
    "rne_name": 2,
    "rne_address": 3,
    "bodacc_name": 2,
    "bodacc_address": 3,
    "hierarchical": 6,
    "official_successor": 7,
    "bodacc_relation": 8,
}

LTR_CHANNELS: tuple[str, ...] = (
    "name_exact",
    "address_exact",
    "name_word",
    "address_word",
    "name_char",
    "address_char",
    "rne_name",
    "rne_address",
    "bodacc_name",
    "bodacc_address",
    "hierarchical",
    "official_successor",
    "bodacc_relation",
)


def _mapped_direct_channels(
    backend_name: str,
    channel: str,
    record: IndexedEstablishment,
) -> tuple[tuple[str, str], ...]:
    if backend_name != "official_overlay":
        return ((channel, f"{backend_name}:{channel}"),)
    raw_sources = record.payload.get("official_evidence_sources") or []
    if isinstance(raw_sources, str):
        raw_sources = [raw_sources]
    sources = {str(value).upper() for value in raw_sources}
    family = "name" if channel.startswith("name") else "address"
    mapped: list[tuple[str, str]] = []
    if "RNE" in sources:
        mapped.append((f"rne_{family}", f"official_overlay:RNE:{channel}"))
    if "BODACC" in sources:
        mapped.append((f"bodacc_{family}", f"official_overlay:BODACC:{channel}"))
    # SIRENE-only evidence remains comparable to the base lexical channel.
    if not mapped:
        mapped.append((channel, f"official_overlay:{channel}"))
    return tuple(mapped)


def _record_matches_query_geo(
    record: IndexedEstablishment, query: RetrievalQuery
) -> bool:
    if query.insee and record.insee:
        return record.insee == query.insee
    if query.postcode and record.postcode:
        return record.postcode == query.postcode
    return False


class OfficialEvidenceRetriever:
    """Union adapter over base Tantivy and a separately built evidence overlay."""

    def __init__(
        self,
        base_backend: RetrievalBackend,
        overlay_backend: RetrievalBackend | None = None,
        config: OfficialEvidenceRetrievalConfig | None = None,
    ) -> None:
        self.config = config or OfficialEvidenceRetrievalConfig()
        self.backends: tuple[tuple[str, RetrievalBackend], ...] = tuple(
            [("base", base_backend)]
            + ([ ("official_overlay", overlay_backend) ] if overlay_backend else [])
        )

    @classmethod
    def from_index_paths(
        cls,
        base_index_path: Path | str,
        overlay_index_path: Path | str | None = None,
        config: OfficialEvidenceRetrievalConfig | None = None,
    ) -> "OfficialEvidenceRetriever":
        return cls(
            TantivyBackend(base_index_path),
            (
                OfficialEvidenceTantivyBackend(overlay_index_path)
                if overlay_index_path
                else None
            ),
            config,
        )

    def retrieve(
        self, row: RetrievalQuery | Mapping[str, Any]
    ) -> OfficialEvidenceRetrievalResult:
        started = time.perf_counter()
        query = row if isinstance(row, RetrievalQuery) else RetrievalQuery.from_mapping(row)
        if not query.insee and not query.postcode:
            return OfficialEvidenceRetrievalResult(
                query, (), 0, False, (time.perf_counter() - started) * 1000.0
            )
        candidates: dict[str, _MutableCandidate] = {}
        direct_jobs: list[tuple[str, RetrievalBackend, str, int]] = []
        for backend_name, backend in self.backends:
            for channel, limit in (
                ("name_exact", self.config.exact_limit),
                ("address_exact", self.config.exact_limit),
                ("name_word", self.config.word_limit),
                ("address_word", self.config.word_limit),
                ("name_char", self.config.character_limit),
                ("address_char", self.config.character_limit),
            ):
                direct_jobs.append((backend_name, backend, channel, limit))

        # Search calls are independent and Tantivy searchers are immutable.
        with ThreadPoolExecutor(max_workers=self.config.search_workers) as pool:
            futures = {
                pool.submit(backend.search, query, channel, limit): (
                    backend_name,
                    channel,
                )
                for backend_name, backend, channel, limit in direct_jobs
            }
            direct_results: dict[tuple[str, str], Any] = {}
            for future in as_completed(futures):
                direct_results[futures[future]] = future.result()

        # Registration order is fixed even though searches completed in parallel.
        for backend_name, _backend in self.backends:
            for channel in (
                "name_exact",
                "address_exact",
                "name_word",
                "address_word",
                "name_char",
                "address_char",
            ):
                for rank, hit in enumerate(
                    direct_results[(backend_name, channel)], start=1
                ):
                    for mapped_channel, source in _mapped_direct_channels(
                        backend_name, channel, hit.record
                    ):
                        self._register(
                            candidates,
                            hit.record,
                            OfficialEvidenceSignal(
                                source=source,
                                channel=mapped_channel,
                                rank=rank,
                                score=float(hit.score),
                            ),
                        )

        siren_hits: dict[str, list[tuple[str, int, float]]] = {}
        for backend_name, backend in self.backends:
            for rank, (siren, score) in enumerate(
                backend.search_sirens(query, self.config.siren_limit), start=1
            ):
                siren_hits.setdefault(siren, []).append(
                    (backend_name, rank, float(score))
                )

        site_expansion_rank = 0
        for siren in sorted(
            siren_hits,
            key=lambda value: (
                min(item[1] for item in siren_hits[value]),
                value,
            ),
        ):
            for siren_backend_name, parent_rank, parent_score in sorted(
                siren_hits[siren], key=lambda item: (item[1], item[0])
            ):
                for site_backend_name, site_backend in self.backends:
                    sites = site_backend.sites_for_siren(
                        siren, query, self.config.sites_per_siren
                    )
                    for site_rank, site in enumerate(sites, start=1):
                        site_expansion_rank += 1
                        self._register(
                            candidates,
                            site,
                            OfficialEvidenceSignal(
                                source=(
                                    f"{siren_backend_name}:siren->"
                                    f"{site_backend_name}:site"
                                ),
                                channel="hierarchical",
                                rank=site_expansion_rank,
                                score=parent_score,
                                parent_identifier=siren,
                                parent_rank=parent_rank,
                                site_rank=site_rank,
                            ),
                        )

        self._expand_establishment_relations(query, candidates)
        self._expand_siren_relations(query, siren_hits, candidates)

        ordered = sorted(
            candidates.items(), key=lambda item: self._union_sort_key(item[0], item[1])
        )
        raw_union_size = len(ordered)
        selected = ordered[: self.config.max_union_candidates]
        output = tuple(
            OfficialEvidenceCandidate(
                union_rank=rank,
                siret=siret,
                siren=mutable.record.siren,
                record=self._merged_record(mutable),
                signals=tuple(
                    sorted(
                        mutable.signals,
                        key=lambda signal: (
                            _CHANNEL_PRIORITY[signal.channel],
                            signal.rank,
                            signal.source,
                        ),
                    )
                ),
                official_evidence_ids=tuple(sorted(mutable.evidence_ids)),
                official_evidence_sources=tuple(sorted(mutable.evidence_sources)),
            )
            for rank, (siret, mutable) in enumerate(selected, start=1)
        )
        return OfficialEvidenceRetrievalResult(
            query=query,
            candidates=output,
            raw_union_size=raw_union_size,
            union_truncated=raw_union_size > len(output),
            retrieval_latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _register(
        candidates: dict[str, _MutableCandidate],
        record: IndexedEstablishment,
        signal: OfficialEvidenceSignal,
    ) -> None:
        if not record.siret:
            return
        candidate = candidates.setdefault(record.siret, _MutableCandidate(record))
        candidate.names.update(record.names)
        candidate.addresses.update(record.addresses)
        evidence_ids = record.payload.get("official_evidence_ids") or []
        evidence_sources = record.payload.get("official_evidence_sources") or []
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        if isinstance(evidence_sources, str):
            evidence_sources = [evidence_sources]
        candidate.evidence_ids.update(str(item) for item in evidence_ids if item)
        candidate.evidence_sources.update(str(item) for item in evidence_sources if item)
        if signal not in candidate.signals:
            candidate.signals.append(signal)
        # Registration is deliberately base-first.  Never let a later overlay
        # document replace the national record (and its official succession
        # links); overlay payload provenance is accumulated in the sets above.

    @staticmethod
    def _merged_record(mutable: _MutableCandidate) -> IndexedEstablishment:
        record = mutable.record
        payload = {
            **record.payload,
            "official_evidence_ids": sorted(mutable.evidence_ids),
            "official_evidence_sources": sorted(mutable.evidence_sources),
        }
        return IndexedEstablishment(
            siret=record.siret,
            siren=record.siren,
            insee=record.insee,
            postcode=record.postcode,
            names=tuple(sorted(mutable.names or set(record.names))),
            addresses=tuple(sorted(mutable.addresses or set(record.addresses))),
            number=record.number,
            active=record.active,
            is_siege=record.is_siege,
            linked_sirets=record.linked_sirets,
            payload=payload,
        )

    def _lookup_siret(self, siret: str) -> IndexedEstablishment | None:
        for _backend_name, backend in self.backends:
            record = backend.by_siret(siret)
            if record is not None:
                return record
        return None

    def _expand_establishment_relations(
        self,
        query: RetrievalQuery,
        candidates: dict[str, _MutableCandidate],
    ) -> None:
        seeds = sorted(
            candidates.items(), key=lambda item: self._union_sort_key(item[0], item[1])
        )[: self.config.relation_seed_limit]
        relation_rank = 0
        seen_pairs: set[tuple[str, str]] = set()
        for source_siret, mutable in seeds:
            for target_siret in mutable.record.linked_sirets:
                pair = (source_siret, target_siret)
                if pair in seen_pairs or source_siret == target_siret:
                    continue
                seen_pairs.add(pair)
                target = self._lookup_siret(target_siret)
                if target is None or not _record_matches_query_geo(target, query):
                    continue
                relation_rank += 1
                self._register(
                    candidates,
                    target,
                    OfficialEvidenceSignal(
                        source="official_relation:siret",
                        channel="official_successor",
                        rank=relation_rank,
                        score=0.0,
                        parent_identifier=source_siret,
                    ),
                )
                if relation_rank >= self.config.relation_limit:
                    return

    def _expand_siren_relations(
        self,
        query: RetrievalQuery,
        siren_hits: Mapping[str, Sequence[tuple[str, int, float]]],
        candidates: dict[str, _MutableCandidate],
    ) -> None:
        relation_rank = 0
        visited: set[tuple[str, str]] = set()
        for source_siren in sorted(siren_hits):
            for relation_backend_name, relation_backend in self.backends:
                linked_method = getattr(relation_backend, "linked_sirens", None)
                if not callable(linked_method):
                    continue
                for target_siren in linked_method(source_siren):
                    pair = (source_siren, target_siren)
                    if pair in visited or source_siren == target_siren:
                        continue
                    visited.add(pair)
                    for site_backend_name, site_backend in self.backends:
                        sites = site_backend.sites_for_siren(
                            target_siren, query, self.config.sites_per_siren
                        )
                        for site_rank, site in enumerate(sites, start=1):
                            relation_rank += 1
                            self._register(
                                candidates,
                                site,
                                OfficialEvidenceSignal(
                                    source=(
                                        f"{relation_backend_name}:siren_relation->"
                                        f"{site_backend_name}:site"
                                    ),
                                    channel="bodacc_relation",
                                    rank=relation_rank,
                                    score=0.0,
                                    parent_identifier=source_siren,
                                    site_rank=site_rank,
                                ),
                            )
                            if relation_rank >= self.config.relation_limit:
                                return

    @staticmethod
    def _union_sort_key(
        siret: str, mutable: _MutableCandidate
    ) -> tuple[int, int, int, str]:
        best = min(
            (
                _CHANNEL_PRIORITY[signal.channel],
                signal.rank,
            )
            for signal in mutable.signals
        )
        # Raw BM25 scores are not comparable across fields/indices and are not
        # fused here.  The deterministic order only makes the bounded artifact
        # stable; admission gets every per-signal value separately.
        return (best[0], best[1], -len(mutable.signals), siret)


def official_evidence_union_arrow_schema() -> pa.Schema:
    fields: list[tuple[str, Any]] = [
            ("schema_version", pa.string()),
            ("query_id", pa.string()),
            ("fold", pa.int8()),
            ("gt_siret", pa.string()),
            ("ground_truth_siret", pa.string()),
            ("ground_truth_siren", pa.string()),
            ("ground_truth_state", pa.string()),
            ("historical_ground_truth_siret", pa.string()),
            ("identifiable_exact", pa.bool_()),
            ("label_kind", pa.string()),
            ("v2_exact", pa.bool_()),
            ("v2_label_kind", pa.string()),
            ("qualification_v2", pa.string()),
            ("v3_exact", pa.bool_()),
            ("v3_label_kind", pa.string()),
            ("qualification_v3", pa.string()),
            ("source_kind", pa.string()),
            ("pool_size", pa.int64()),
            ("mega_base_pool", pa.bool_()),
            ("unseen_siren", pa.bool_()),
            ("is_synthetic", pa.bool_()),
            ("crm_name", pa.string()),
            ("crm_address", pa.string()),
            ("crm_postcode", pa.string()),
            ("crm_insee", pa.string()),
            ("crm_number", pa.string()),
            ("identifiable", pa.bool_()),
            ("acceptable_sirets_operational", pa.list_(pa.string())),
            ("acceptable_sirets_operational_json", pa.string()),
            ("query_metadata_json", pa.string()),
            ("union_rank", pa.int32()),
            ("siret", pa.string()),
            ("siren", pa.string()),
            ("sources", pa.list_(pa.string())),
            ("channels", pa.list_(pa.string())),
            (
                "signals",
                pa.list_(
                    pa.struct(
                        [
                            ("source", pa.string()),
                            ("channel", pa.string()),
                            ("rank", pa.int32()),
                            ("score", pa.float64()),
                            ("parent_identifier", pa.string()),
                            ("parent_rank", pa.int32()),
                            ("site_rank", pa.int32()),
                        ]
                    )
                ),
            ),
            ("official_evidence_ids", pa.list_(pa.string())),
            ("official_evidence_sources", pa.list_(pa.string())),
            ("candidate_names", pa.list_(pa.string())),
            ("candidate_addresses", pa.list_(pa.string())),
            ("candidate_insee", pa.string()),
            ("candidate_postcode", pa.string()),
            ("candidate_number", pa.string()),
            ("candidate_state", pa.string()),
            ("candidate_is_siege", pa.bool_()),
            ("retrieval_source", pa.string()),
            ("retrieval_rank", pa.int32()),
            ("retrieval_score", pa.float64()),
            ("raw_union_size", pa.int32()),
            ("union_truncated", pa.bool_()),
            ("retrieval_latency_ms", pa.float64()),
            ("candidate_present", pa.bool_()),
    ]
    for channel in LTR_CHANNELS:
        fields.extend(
            [
                (f"{channel}_rank", pa.int32()),
                (f"{channel}_score", pa.float64()),
            ]
        )
    return pa.schema(fields)


def _first_metadata(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        return value
    return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    normalized = str(value).strip().upper()
    if normalized in {"1", "TRUE", "YES", "Y", "OUI", "IDENTIFIABLE"}:
        return True
    if normalized in {"0", "FALSE", "NO", "N", "NON", "UNIDENTIFIABLE"}:
        return False
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def _required_fold(value: Any) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("candidate export requires an integer fold")
    try:
        fold = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("candidate export requires an integer fold") from error
    if fold not in {0, 1, 2, 3, 4}:
        raise ValueError(f"unsupported retrieval fold: {fold}")
    return fold


def _operational_sirets(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [item.strip() for item in value.split(",")]
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return sorted(
        {
            digits
            for item in value
            if len(digits := "".join(character for character in str(item) if character.isdigit())) == 14
        }
    )


def _query_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    query = RetrievalQuery.from_mapping(row)
    ground_truth_siret = str(
        _first_metadata(row, ["ground_truth_siret", "gt_siret", "siret_gt"]) or ""
    )
    ground_truth_siren = str(
        _first_metadata(row, ["ground_truth_siren", "gt_siren", "siren_gt"])
        or ground_truth_siret[:9]
    )
    operational = _operational_sirets(
        _first_metadata(
            row,
            [
                "acceptable_sirets_operational",
                "acceptable_sirets_operational_json",
                "operational_sirets",
            ],
        )
    )
    identifiable = _optional_bool(
        _first_metadata(
            row,
            [
                "identifiable_exact",
                "identifiable",
                "is_identifiable",
                "v3_identifiable",
                "v2_identifiable",
            ],
        )
    )
    selected = {
        "fold": _required_fold(
            _first_metadata(row, ["fold", "oof_fold", "partition", "split"])
        ),
        "gt_siret": ground_truth_siret,
        "ground_truth_siret": ground_truth_siret,
        "ground_truth_siren": ground_truth_siren,
        "ground_truth_state": str(
            _first_metadata(row, ["ground_truth_state", "gt_state", "etat_admin_gt"])
            or ""
        ),
        "historical_ground_truth_siret": str(
            _first_metadata(
                row,
                [
                    "historical_ground_truth_siret",
                    "historical_gt_siret",
                    "gt_siret_historical",
                ],
            )
            or ground_truth_siret
        ),
        "identifiable_exact": identifiable,
        "label_kind": str(_first_metadata(row, ["label_kind"]) or ""),
        "v2_exact": _optional_bool(_first_metadata(row, ["v2_exact"])),
        "v2_label_kind": str(_first_metadata(row, ["v2_label_kind"]) or ""),
        "qualification_v2": str(
            _first_metadata(row, ["qualification_v2"]) or ""
        ),
        "v3_exact": _optional_bool(_first_metadata(row, ["v3_exact"])),
        "v3_label_kind": str(_first_metadata(row, ["v3_label_kind"]) or ""),
        "qualification_v3": str(
            _first_metadata(row, ["qualification_v3"]) or ""
        ),
        "source_kind": str(
            _first_metadata(row, ["source_kind", "label_source", "provenance"])
            or ""
        ),
        "pool_size": _optional_int(_first_metadata(row, ["pool_size"])),
        "mega_base_pool": _optional_bool(
            _first_metadata(row, ["mega_base_pool"])
        ),
        "unseen_siren": _optional_bool(
            _first_metadata(row, ["unseen_siren", "is_unseen_siren"])
        ),
        "is_synthetic": _optional_bool(_first_metadata(row, ["is_synthetic"])),
        "crm_name": query.name,
        "crm_address": query.address,
        "crm_postcode": query.postcode,
        "crm_insee": query.insee,
        "crm_number": query.number,
        "identifiable": identifiable,
        "acceptable_sirets_operational": operational,
        "acceptable_sirets_operational_json": json.dumps(
            operational, separators=(",", ":")
        ),
    }
    selected["query_metadata_json"] = json.dumps(
        dict(row), sort_keys=True, separators=(",", ":"), default=str
    )
    return selected


def retrieve_official_evidence_union_to_parquet(
    retriever: OfficialEvidenceRetriever,
    rows: Iterable[Mapping[str, Any]],
    output_path: Path | str,
    *,
    query_id_field: str = "query_id",
    batch_size: int = 4096,
) -> Path:
    """Stream one candidate row per query/SIRET for the admission stage."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = official_evidence_union_arrow_schema()
    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    buffered: list[Mapping[str, Any]] = []
    try:
        for ordinal, row in enumerate(rows):
            query_id = str(row.get(query_id_field) or ordinal)
            result = retriever.retrieve(row)
            buffered.extend(result.candidate_rows(query_id, _query_metadata(row)))
            if len(buffered) >= batch_size:
                writer.write_table(pa.Table.from_pylist(buffered, schema=schema))
                buffered.clear()
        if buffered:
            writer.write_table(pa.Table.from_pylist(buffered, schema=schema))
        writer.close()
        return output_path
    except Exception:
        writer.close()
        output_path.unlink(missing_ok=True)
        raise
