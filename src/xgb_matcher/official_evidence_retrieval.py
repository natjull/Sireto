"""Detailed pre-admission union over the national and official-evidence indices.

This adapter is intentionally not wired into the legacy top-100 path.  It
exposes raw per-channel ranks/scores/sources for downstream admission research,
with a hard internal-union cap of 2,000 and no RRF or model-derived fusion.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
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


OFFICIAL_EVIDENCE_UNION_SCHEMA_VERSION = "sireto-official-evidence-union-v2"


class SirenRelationBackend(Protocol):
    def linked_sirens(self, siren: str, limit: int = 256) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class OfficialEvidenceRetrievalConfig:
    max_union_candidates: int = 2000
    exact_limit: int = 500
    word_limit: int = 1000
    character_limit: int = 1000
    historical_limit: int = 500
    supporting_limit: int = 250
    siren_limit: int = 10
    sites_per_siren: int = 32
    relation_seed_limit: int = 500
    relation_limit: int = 500
    balanced_prefix_per_channel: int = 32
    search_workers: int = 12
    require_temporal_complete: bool = True
    require_temporal_overlay: bool = False
    typed_portfolio: bool | None = None
    fielded_name_boosts: Mapping[str, float] = field(
        default_factory=lambda: {
            "legal_names": 1.25,
            "trade_names": 1.50,
            "site_names": 1.75,
        }
    )

    def __post_init__(self) -> None:
        if not 1 <= self.max_union_candidates <= 2000:
            raise ValueError("max_union_candidates must be between 1 and 2000")
        if not 1 <= self.sites_per_siren <= 32:
            raise ValueError("sites_per_siren must be between 1 and 32")
        for name in (
            "exact_limit",
            "word_limit",
            "character_limit",
            "historical_limit",
            "supporting_limit",
            "siren_limit",
            "relation_seed_limit",
            "relation_limit",
            "balanced_prefix_per_channel",
            "search_workers",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OfficialEvidenceRetrievalConfig":
        retrieval = raw.get("retrieval") or raw
        return cls(
            max_union_candidates=int(retrieval.get("max_union_candidates", 2000)),
            exact_limit=int(retrieval.get("exact_limit", 500)),
            word_limit=int(retrieval.get("word_limit", 1000)),
            character_limit=int(retrieval.get("character_limit", 1000)),
            historical_limit=int(retrieval.get("historical_limit", 500)),
            supporting_limit=int(retrieval.get("supporting_limit", 250)),
            siren_limit=int(retrieval.get("siren_limit", 5)),
            sites_per_siren=int(retrieval.get("sites_per_siren", 32)),
            relation_seed_limit=int(retrieval.get("relation_seed_limit", 500)),
            relation_limit=int(retrieval.get("relation_limit", 500)),
            balanced_prefix_per_channel=int(
                retrieval.get("balanced_prefix_per_channel", 32)
            ),
            search_workers=int(retrieval.get("search_workers", 12)),
            require_temporal_complete=bool(
                retrieval.get("require_temporal_complete", True)
            ),
            require_temporal_overlay=bool(
                retrieval.get("require_temporal_overlay", False)
            ),
            typed_portfolio=(
                bool(retrieval["typed_portfolio"])
                if "typed_portfolio" in retrieval else None
            ),
            fielded_name_boosts=dict(
                retrieval.get("fielded_name_boosts")
                or {"legal_names": 1.25, "trade_names": 1.50, "site_names": 1.75}
            ),
        )

    @classmethod
    def load(cls, path: Path | str) -> "OfficialEvidenceRetrievalConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


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
            "candidate_geo_scaffold": bool(self.record.payload.get("geo_scaffold", False)),
            "temporal_valid_from": str(self.record.payload.get("temporal_valid_from") or ""),
            "temporal_valid_to": str(self.record.payload.get("temporal_valid_to") or ""),
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
    temporal_valid_from: set[str] = field(default_factory=set)
    temporal_valid_to: set[str] = field(default_factory=set)
    geo_scaffold: bool = False
    linked_sirets: set[str] = field(default_factory=set)
    linked_sirens: set[str] = field(default_factory=set)


_CHANNEL_PRIORITY: Mapping[str, int] = {
    "legal_name_exact": 0,
    "trade_name_exact": 0,
    "site_name_exact": 0,
    "number_exact": 1,
    "fielded_name_bm25": 2,
    "legal_name_word": 2,
    "trade_name_word": 2,
    "site_name_word": 2,
    "current_name_char": 4,
    "historical_name_word": 5,
    "historical_address_word": 5,
    "supporting_name_word": 6,
    "supporting_address_word": 6,
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
    "legal_name_exact",
    "trade_name_exact",
    "site_name_exact",
    "number_exact",
    "fielded_name_bm25",
    "legal_name_word",
    "trade_name_word",
    "site_name_word",
    "current_name_char",
    "historical_name_word",
    "historical_address_word",
    "supporting_name_word",
    "supporting_address_word",
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
    if backend_name == "sirene_temporal":
        # Historical exact lexical equality is valuable recall evidence but is
        # deliberately not an exact-protected current-identity signal.
        if channel.startswith("name"):
            return (("historical_name_word", f"sirene_temporal:{channel}"),)
        return ()
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
        temporal_overlay_backend: RetrievalBackend | None = None,
    ) -> None:
        self.config = config or OfficialEvidenceRetrievalConfig()
        self.backends: tuple[tuple[str, RetrievalBackend], ...] = tuple(
            [("base", base_backend)]
            + ([ ("official_overlay", overlay_backend) ] if overlay_backend else [])
            + ([ ("sirene_temporal", temporal_overlay_backend) ] if temporal_overlay_backend else [])
        )
        base_manifest = getattr(base_backend, "manifest", {}) or {}
        self.typed_portfolio = (
            bool(self.config.typed_portfolio)
            if self.config.typed_portfolio is not None
            else bool(base_manifest.get("typed_name_portfolio"))
        )

    @classmethod
    def from_index_paths(
        cls,
        base_index_path: Path | str,
        overlay_index_path: Path | str | None = None,
        config: OfficialEvidenceRetrievalConfig | None = None,
        temporal_overlay_index_path: Path | str | None = None,
    ) -> "OfficialEvidenceRetriever":
        resolved_config = config or OfficialEvidenceRetrievalConfig()
        base_backend = TantivyBackend(
            base_index_path,
            fielded_name_boosts=resolved_config.fielded_name_boosts,
        )
        overlay_backend = (
            OfficialEvidenceTantivyBackend(overlay_index_path)
            if overlay_index_path else None
        )
        temporal_backend = (
            OfficialEvidenceTantivyBackend(temporal_overlay_index_path)
            if temporal_overlay_index_path else None
        )
        if resolved_config.require_temporal_overlay and temporal_backend is None:
            raise RuntimeError("Official retrieval requires the SIRENE temporal rescue overlay")
        if resolved_config.require_temporal_complete:
            missing = set(base_backend.manifest.get("missing_temporal_sources") or [])
            if temporal_backend is not None:
                temporal_manifest = temporal_backend.manifest
                if int(temporal_manifest.get("num_documents") or 0) <= 0:
                    raise RuntimeError("SIRENE temporal overlay contains zero documents")
                if temporal_manifest.get("overlay_kind") != "SIRENE_TEMPORAL_RESCUE":
                    raise RuntimeError("Unexpected temporal overlay kind")
                missing -= {"SIRENE_HISTORY", "SIRENE_SUCCESSION"}
            if base_backend.manifest.get("temporal_complete") is not True and missing:
                raise RuntimeError(
                    "Official retrieval still lacks temporal sources: " + ",".join(sorted(missing))
                )
        return cls(
            base_backend,
            overlay_backend,
            resolved_config,
            temporal_backend,
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
        typed_channels = (
            ("legal_name_exact", self.config.exact_limit),
            ("trade_name_exact", self.config.exact_limit),
            ("site_name_exact", self.config.exact_limit),
            ("address_exact", self.config.exact_limit),
            ("number_exact", self.config.exact_limit),
            ("fielded_name_bm25", self.config.word_limit),
            ("legal_name_word", self.config.word_limit),
            ("trade_name_word", self.config.word_limit),
            ("site_name_word", self.config.word_limit),
            ("address_word", self.config.word_limit),
            ("current_name_char", self.config.character_limit),
            ("address_char", self.config.character_limit),
            ("historical_name_word", self.config.historical_limit),
            ("historical_address_word", self.config.historical_limit),
            ("supporting_name_word", self.config.supporting_limit),
            ("supporting_address_word", self.config.supporting_limit),
        )
        legacy_channels = (
            ("name_exact", self.config.exact_limit),
            ("address_exact", self.config.exact_limit),
            ("name_word", self.config.word_limit),
            ("address_word", self.config.word_limit),
            ("name_char", self.config.character_limit),
            ("address_char", self.config.character_limit),
        )
        for backend_name, backend in self.backends:
            backend_typed = backend_name == "base" and self.typed_portfolio
            backend_channels = (
                typed_channels if backend_typed else
                tuple(item for item in legacy_channels if item[0].startswith("name"))
                if backend_name == "sirene_temporal" else legacy_channels
            )
            for channel, limit in backend_channels:
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
            backend_typed = backend_name == "base" and self.typed_portfolio
            backend_channels = (
                typed_channels if backend_typed else
                tuple(item for item in legacy_channels if item[0].startswith("name"))
                if backend_name == "sirene_temporal" else legacy_channels
            )
            for channel, _limit in backend_channels:
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
            candidates.items(),
            key=lambda item: self._balanced_union_sort_key(item[0], item[1]),
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
        if record.payload.get("temporal_valid_from"):
            candidate.temporal_valid_from.add(str(record.payload["temporal_valid_from"]))
        if record.payload.get("temporal_valid_to"):
            candidate.temporal_valid_to.add(str(record.payload["temporal_valid_to"]))
        candidate.geo_scaffold = candidate.geo_scaffold or bool(record.payload.get("geo_scaffold"))
        candidate.linked_sirets.update(record.linked_sirets)
        candidate.linked_sirens.update(record.linked_sirens)
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
            "temporal_valid_from": min(mutable.temporal_valid_from, default=""),
            "temporal_valid_to": max(mutable.temporal_valid_to, default=""),
            "geo_scaffold": mutable.geo_scaffold,
        }
        return IndexedEstablishment(
            siret=record.siret,
            siren=record.siren,
            insee=record.insee,
            postcode=record.postcode,
            names=tuple(sorted(mutable.names or set(record.names))),
            addresses=tuple(sorted(mutable.addresses or set(record.addresses))),
            legal_names=record.legal_names,
            trade_names=record.trade_names,
            site_names=record.site_names,
            historical_names=record.historical_names,
            supporting_names=record.supporting_names,
            historical_addresses=record.historical_addresses,
            supporting_addresses=record.supporting_addresses,
            number=record.number,
            active=record.active,
            is_siege=record.is_siege,
            linked_sirets=tuple(sorted(set(record.linked_sirets) | mutable.linked_sirets)),
            linked_sirens=tuple(sorted(set(record.linked_sirens) | mutable.linked_sirens)),
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
            candidates.items(),
            key=lambda item: self._balanced_union_sort_key(item[0], item[1]),
        )[: self.config.relation_seed_limit]
        relation_rank = 0
        seen_pairs: set[tuple[str, str]] = set()
        for source_siret, mutable in seeds:
            linked_by_source: dict[str, set[str]] = {}
            for target in set(mutable.record.linked_sirets) | mutable.linked_sirets:
                linked_by_source.setdefault(target, set()).add("record")
            for backend_name, backend in self.backends:
                linked_method = getattr(backend, "linked_sirets", None)
                if callable(linked_method):
                    for target in linked_method(source_siret):
                        linked_by_source.setdefault(target, set()).add(backend_name)
            for target_siret in sorted(linked_by_source):
                pair = (source_siret, target_siret)
                if pair in seen_pairs or source_siret == target_siret:
                    continue
                seen_pairs.add(pair)
                target = self._lookup_siret(target_siret)
                if target is None or not _record_matches_query_geo(target, query):
                    continue
                relation_sources = linked_by_source[target_siret]
                is_sirene_successor = "sirene_temporal" in relation_sources or (
                    "SIRENE_SUCCESSION" in {str(value).upper() for value in mutable.evidence_sources}
                )
                relation_rank += 1
                self._register(
                    candidates,
                    target,
                    OfficialEvidenceSignal(
                        source=(
                            "SIRENE_SUCCESSION:structured_siret_relation"
                            if is_sirene_successor
                            else "BODACC:structured_siret_relation"
                        ),
                        channel=("official_successor" if is_sirene_successor else "bodacc_relation"),
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

    def _channel_depth(self, channel: str) -> int:
        if channel in {
            "legal_name_exact",
            "trade_name_exact",
            "site_name_exact",
            "name_exact",
            "address_exact",
            "number_exact",
        }:
            return self.config.exact_limit
        if channel in {
            "fielded_name_bm25",
            "legal_name_word",
            "trade_name_word",
            "site_name_word",
            "name_word",
            "address_word",
        }:
            return self.config.word_limit
        if channel in {
            "current_name_char",
            "name_char",
            "address_char",
            "rne_name",
            "rne_address",
            "bodacc_name",
            "bodacc_address",
        }:
            # Source-specific overlay channels combine exact, word and
            # character searches.  The largest configured direct depth is the
            # conservative denominator for their ranks.
            return max(
                self.config.exact_limit,
                self.config.word_limit,
                self.config.character_limit,
            )
        if channel in {"historical_name_word", "historical_address_word"}:
            return self.config.historical_limit
        if channel in {"supporting_name_word", "supporting_address_word"}:
            return self.config.supporting_limit
        if channel == "hierarchical":
            return max(1, self.config.siren_limit * self.config.sites_per_siren)
        if channel in {"official_successor", "bodacc_relation"}:
            return self.config.relation_limit
        return self.config.max_union_candidates

    def _balanced_union_sort_key(
        self, siret: str, mutable: _MutableCandidate
    ) -> tuple[int, float, int, int, str]:
        exact_channels = {
            "legal_name_exact",
            "trade_name_exact",
            "site_name_exact",
            "name_exact",
            "address_exact",
        }
        has_exact = any(
            signal.channel in exact_channels for signal in mutable.signals
        )
        has_preserved_prefix = any(
            signal.rank <= self.config.balanced_prefix_per_channel
            for signal in mutable.signals
        )
        best_fraction = min(
            signal.rank / max(1, self._channel_depth(signal.channel))
            for signal in mutable.signals
        )
        best_rank = min(signal.rank for signal in mutable.signals)
        channel_count = len({signal.channel for signal in mutable.signals})
        # The previous order compared a hard channel priority before rank.  In
        # large communes that allowed thousands of name hits to evict even a
        # top-10 address hit.  Exact evidence remains protected, then a fixed
        # prefix from every channel is preserved, and the rest is compared by
        # rank relative to the depth requested from that channel.  Raw BM25
        # scores are deliberately not mixed across fields or indices.
        bucket = 0 if has_exact else (1 if has_preserved_prefix else 2)
        return (bucket, best_fraction, -channel_count, best_rank, siret)


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
            ("new_site_known_siren", pa.bool_()),
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
            ("candidate_geo_scaffold", pa.bool_()),
            ("temporal_valid_from", pa.string()),
            ("temporal_valid_to", pa.string()),
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
        "new_site_known_siren": _optional_bool(
            _first_metadata(
                row, ["new_site_known_siren", "is_new_site_known_siren"]
            )
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
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists():
        raise FileExistsError(output_path)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = official_evidence_union_arrow_schema()
    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    buffered: list[Mapping[str, Any]] = []
    query_count = 0
    candidate_count = 0
    empty_query_count = 0
    try:
        for ordinal, row in enumerate(rows):
            query_count += 1
            query_id = str(row.get(query_id_field) or ordinal)
            result = retriever.retrieve(row)
            buffered.extend(result.candidate_rows(query_id, _query_metadata(row)))
            candidate_count += len(result.candidates)
            empty_query_count += int(not result.candidates)
            if len(buffered) >= batch_size:
                writer.write_table(pa.Table.from_pylist(buffered, schema=schema))
                buffered.clear()
        if buffered:
            writer.write_table(pa.Table.from_pylist(buffered, schema=schema))
        writer.close()
        digest = hashlib.sha256()
        with output_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        backend_manifests = []
        for backend_name, backend in retriever.backends:
            manifest = getattr(backend, "manifest", None)
            index_path = getattr(backend, "index_path", None)
            backend_manifests.append(
                {
                    "backend": backend_name,
                    "index_path": str(Path(index_path).resolve()) if index_path else None,
                    "build_hash": manifest.get("build_hash") if manifest else None,
                    "schema_version": manifest.get("schema_version") if manifest else None,
                    "temporal_complete": manifest.get("temporal_complete") if manifest else None,
                }
            )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "sireto-official-evidence-union-manifest-v2",
                    "candidate_parquet": output_path.name,
                    "candidate_parquet_sha256": digest.hexdigest(),
                    "query_count": query_count,
                    "candidate_count": candidate_count,
                    "empty_query_count": empty_query_count,
                    "maximum_union_candidates": retriever.config.max_union_candidates,
                    "positive_injection": False,
                    "synthetic_rows": 0,
                    "retrieval_config": asdict(retriever.config),
                    "official_indices": backend_manifests,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path
    except Exception:
        writer.close()
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
