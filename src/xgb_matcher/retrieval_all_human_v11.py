"""Label-blind SIREN-first retrieval for the all-human v11 contract.

The historical official retriever remains byte-compatible.  This module asks
it for a wider, reference-only prefix, adds compact geographic and local-site
sidecars, then applies a deterministic SIREN-balanced cap of 2,000.  Human
labels are deliberately absent from every public type and code path here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

from .hierarchical_retrieval import IndexedEstablishment, RetrievalQuery, normalize_text
from .official_evidence_retrieval import (
    LTR_CHANNELS,
    OfficialEvidenceCandidate,
    OfficialEvidenceRetriever,
    OfficialEvidenceSignal,
)
from .retrieval_sidecars_v11 import LocalSite, RetrievalSidecarReader


V11_RETRIEVAL_SCHEMA_VERSION = "sireto-retrieval-all-human-v11-union-1"
V11_CHANNELS = (
    "geo_insee_census",
    "geo_exact_cell",
    "geo_number_cell",
    "v11_parent_expansion",
)
_LEADING_NUMBER = re.compile(r"^\s*\d{1,5}(?:\s*(?:BIS|TER|QUATER|[A-Z]))?\s+")

_NAME_EXACT = {
    "legal_name_exact",
    "trade_name_exact",
    "site_name_exact",
    "name_exact",
}
_NAME_EVIDENCE = _NAME_EXACT | {
    "fielded_name_bm25",
    "legal_name_word",
    "trade_name_word",
    "site_name_word",
    "current_name_char",
    "historical_name_word",
    "supporting_name_word",
    "rne_name",
    "bodacc_name",
    "legal_entity_role_name",
    "sector_name",
}
_ADDRESS_EVIDENCE = {
    "address_exact",
    "number_exact",
    "address_word",
    "address_char",
    "historical_address_word",
    "supporting_address_word",
    "rne_address",
    "bodacc_address",
    "sector_address",
    "geo_insee_census",
    "geo_exact_cell",
    "geo_number_cell",
}
_EXACT_ADDRESS = {"address_exact", "number_exact", "geo_exact_cell"}


@dataclass(frozen=True)
class RetrievalV11Config:
    raw_reference_cap: int = 10_000
    internal_cap: int = 2_000
    parent_candidate_budget: int = 700
    parent_siren_limit: int = 64
    relation_seed_limit: int = 32

    @classmethod
    def load(cls, path: Path | str) -> "RetrievalV11Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        retrieval = raw.get("retrieval") or {}
        budgets = raw.get("budgets") or {}
        return cls(
            raw_reference_cap=int(retrieval.get("raw_reference_cap", 10_000)),
            internal_cap=int(
                retrieval.get("maximum_internal_candidates", 2_000)
            ),
            parent_candidate_budget=int(
                retrieval.get(
                    "parent_candidate_budget",
                    budgets.get("parent_expansion_candidates", 700),
                )
            ),
            parent_siren_limit=int(
                retrieval.get(
                    "parent_siren_limit", budgets.get("parent_siren_limit", 64)
                )
            ),
            relation_seed_limit=int(
                retrieval.get(
                    "relation_seed_limit", budgets.get("relation_seed_limit", 32)
                )
            ),
        )

    def __post_init__(self) -> None:
        if not 1 <= self.internal_cap <= 2_000:
            raise ValueError("v11 internal cap must be between 1 and 2000")
        if not self.internal_cap <= self.raw_reference_cap <= 10_000:
            raise ValueError("v11 raw reference cap must be between internal cap and 10000")
        if not 1 <= self.parent_candidate_budget <= self.internal_cap:
            raise ValueError("parent candidate budget must fit in the internal cap")
        if not 1 <= self.parent_siren_limit <= 512:
            raise ValueError("parent_siren_limit must be between 1 and 512")
        if not 1 <= self.relation_seed_limit <= 32:
            raise ValueError("relation_seed_limit must be between 1 and 32")


@dataclass
class _Reference:
    siret: str
    siren: str
    record: IndexedEstablishment | None = None
    signals: list[OfficialEvidenceSignal] = field(default_factory=list)
    evidence_ids: set[str] = field(default_factory=set)
    evidence_sources: set[str] = field(default_factory=set)

    @property
    def channels(self) -> set[str]:
        return {signal.channel for signal in self.signals}


@dataclass(frozen=True)
class RetrievalV11Result:
    query: RetrievalQuery
    candidates: tuple[OfficialEvidenceCandidate, ...]
    raw_reference_count: int
    raw_reference_truncated: bool
    retrieval_latency_ms: float
    strategy_counts: Mapping[str, Any]

    def candidate_rows(
        self, query_id: str, query_metadata: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Serialize the label-blind union without calling label-side helpers."""
        rows: list[dict[str, Any]] = []
        for candidate in self.candidates:
            best = min(
                candidate.signals,
                key=lambda signal: (signal.rank, -float(signal.score), signal.channel),
            )
            channel_values: dict[str, Any] = {}
            for channel in (*LTR_CHANNELS, *V11_CHANNELS):
                signals = [
                    signal for signal in candidate.signals if signal.channel == channel
                ]
                selected = (
                    min(signals, key=lambda signal: (signal.rank, -signal.score))
                    if signals
                    else None
                )
                channel_values[f"{channel}_rank"] = selected.rank if selected else None
                channel_values[f"{channel}_score"] = selected.score if selected else None
            rows.append(
                {
                    **dict(query_metadata or {}),
                    "schema_version": V11_RETRIEVAL_SCHEMA_VERSION,
                    "query_id": str(query_id),
                    "union_rank": candidate.union_rank,
                    "siret": candidate.siret,
                    "siren": candidate.siren,
                    "sources": list(candidate.sources),
                    "channels": list(candidate.channels),
                    "signals": [signal.to_dict() for signal in candidate.signals],
                    "official_evidence_ids": list(candidate.official_evidence_ids),
                    "official_evidence_sources": list(
                        candidate.official_evidence_sources
                    ),
                    "candidate_names": list(candidate.record.names),
                    "candidate_addresses": list(candidate.record.addresses),
                    "candidate_insee": candidate.record.insee,
                    "candidate_postcode": candidate.record.postcode,
                    "candidate_number": candidate.record.number,
                    "candidate_state": "A" if candidate.record.active else "F",
                    "candidate_is_siege": candidate.record.is_siege,
                    "temporal_valid_from": str(
                        candidate.record.payload.get("temporal_valid_from") or ""
                    ),
                    "temporal_valid_to": str(
                        candidate.record.payload.get("temporal_valid_to") or ""
                    ),
                    "retrieval_source": "+".join(candidate.sources),
                    "retrieval_rank": candidate.union_rank,
                    "retrieval_score": float(best.score),
                    **channel_values,
                    "pool_size": int(
                        self.strategy_counts.get("geographic_pool_size") or 0
                    ),
                    "mega_base_pool": int(
                        self.strategy_counts.get("geographic_pool_size") or 0
                    )
                    >= 100_000,
                    "raw_union_size": self.raw_reference_count,
                    "union_truncated": self.raw_reference_truncated,
                    "retrieval_latency_ms": self.retrieval_latency_ms,
                    "candidate_present": True,
                    "retrieval_strategy_json": json.dumps(
                        dict(self.strategy_counts), sort_keys=True
                    ),
                }
            )
        return rows


def _street_query(query: RetrievalQuery) -> str:
    return normalize_text(_LEADING_NUMBER.sub("", query.address))


def _local_record(site: LocalSite) -> IndexedEstablishment:
    return IndexedEstablishment.from_mapping(
        {
            "siret": site.siret,
            "siren": site.siren,
            "insee": site.insee,
            "postcode": site.postcode,
            "number": site.street_number,
            "addresses": [site.current_address_normalized],
            "active": site.active,
            "is_siege": site.is_headquarters,
            "state": site.administrative_state,
        }
    )


def _joint_anchor(reference: _Reference) -> bool:
    channels = reference.channels
    has_name = bool(channels & _NAME_EXACT)
    has_name_evidence = bool(channels & _NAME_EVIDENCE)
    has_geo = bool(channels & _ADDRESS_EVIDENCE)
    has_address_exact = bool(channels & _EXACT_ADDRESS)
    return (has_name and has_geo) or (has_address_exact and has_name_evidence)


def _reference_priority(reference: _Reference) -> tuple[Any, ...]:
    channels = reference.channels
    independent_sources = len({signal.source for signal in reference.signals})
    parent_support = max(
        (signal.parent_support_count or 0 for signal in reference.signals),
        default=0,
    )
    parent_rank = min(
        (signal.parent_rank for signal in reference.signals if signal.parent_rank),
        default=2_147_483_647,
    )
    site_rank = min(
        (signal.site_rank for signal in reference.signals if signal.site_rank),
        default=2_147_483_647,
    )
    best_rank = min((signal.rank for signal in reference.signals), default=2_147_483_647)
    best_score = max((float(signal.score) for signal in reference.signals), default=0.0)
    consensus = len(channels)
    return (
        0 if _joint_anchor(reference) else 1,
        -independent_sources,
        -consensus,
        -parent_support,
        parent_rank,
        site_rank,
        best_rank,
        -best_score,
        reference.siret,
    )


class AllHumanV11Retriever:
    """Compose official search with geographic census and SIREN-site expansion."""

    def __init__(
        self,
        official: OfficialEvidenceRetriever,
        sidecars: RetrievalSidecarReader,
        config: RetrievalV11Config | None = None,
    ) -> None:
        self.official = official
        self.sidecars = sidecars
        self.config = config or RetrievalV11Config()
        official_config = getattr(official, "config", None)
        official_relation_seed_limit = getattr(
            official_config, "relation_seed_limit", self.config.relation_seed_limit
        )
        if int(official_relation_seed_limit) > self.config.relation_seed_limit:
            raise ValueError(
                "official retriever relation_seed_limit exceeds the v11 one-hop seed budget"
            )
        self._canonical_backend = official.backends[0][1]

    @staticmethod
    def _register_signal(
        references: dict[str, _Reference],
        site: LocalSite,
        signal: OfficialEvidenceSignal,
    ) -> None:
        reference = references.setdefault(
            site.siret,
            _Reference(site.siret, site.siren, record=_local_record(site)),
        )
        reference.signals.append(signal)

    def _merge_official(
        self,
        references: dict[str, _Reference],
        candidates: Sequence[OfficialEvidenceCandidate],
    ) -> None:
        for candidate in candidates:
            reference = references.setdefault(
                candidate.siret,
                _Reference(candidate.siret, candidate.siren),
            )
            reference.record = candidate.record
            reference.signals.extend(candidate.signals)
            reference.evidence_ids.update(candidate.official_evidence_ids)
            reference.evidence_sources.update(candidate.official_evidence_sources)

    def _add_geographic_candidates(
        self,
        query: RetrievalQuery,
        references: dict[str, _Reference],
    ) -> tuple[str, int, int]:
        if not query.insee:
            return "NO_INSEE", 0, 0
        result = self.sidecars.geo_candidates(
            insee=query.insee,
            postcode=query.postcode,
            street_number=query.number,
            street=_street_query(query),
            max_sites=self.config.internal_cap,
        )
        channel = {
            "INSEE_ALL": "geo_insee_census",
            "EXACT_GEO_CELL": "geo_exact_cell",
            "NUMBER_CELL": "geo_number_cell",
        }.get(result.strategy)
        if channel is None:
            return result.strategy, 0, result.matched_count
        for rank, site in enumerate(result.sites, start=1):
            self._register_signal(
                references,
                site,
                OfficialEvidenceSignal(
                    source="siren_dossier_v3:geo_sidecar",
                    channel=channel,
                    rank=rank,
                    score=1.0 / rank,
                    parent_identifier=site.siren,
                    site_rank=rank,
                ),
            )
        return result.strategy, len(result.sites), result.matched_count

    def _parent_order(self, references: Mapping[str, _Reference]) -> list[str]:
        by_parent: dict[str, list[_Reference]] = {}
        for reference in references.values():
            if reference.siren:
                by_parent.setdefault(reference.siren, []).append(reference)
        ranked = sorted(
            by_parent,
            key=lambda siren: (
                min(_reference_priority(item) for item in by_parent[siren]),
                -len(by_parent[siren]),
                siren,
            ),
        )
        return ranked[: self.config.parent_siren_limit]

    def _expand_parents(
        self,
        query: RetrievalQuery,
        references: dict[str, _Reference],
    ) -> int:
        parent_support = {
            siren: sum(reference.siren == siren for reference in references.values())
            for siren in self._parent_order(references)
        }
        batch_method = getattr(self.sidecars, "sites_by_sirens_geo", None)
        batch_results = (
            batch_method(
                sirens=list(parent_support),
                insee=query.insee,
                postcode=query.postcode,
                address=query.address,
                street_number=query.number,
                street=_street_query(query),
                include_active=True,
                include_closed=True,
                max_sites_per_siren=self.config.parent_candidate_budget,
            )
            if callable(batch_method) and parent_support
            else {}
        )
        queues: list[tuple[int, str, tuple[LocalSite, ...]]] = []
        for parent_rank, siren in enumerate(parent_support, start=1):
            result = batch_results.get(siren)
            if result is None:
                result = self.sidecars.sites_by_siren_geo(
                    siren=siren,
                    insee=query.insee,
                    postcode=query.postcode,
                    address=query.address,
                    street_number=query.number,
                    street=_street_query(query),
                    include_active=True,
                    include_closed=True,
                    max_sites=self.config.parent_candidate_budget,
                )
            if result.sites:
                queues.append((parent_rank, siren, result.sites))
        positions = [0] * len(queues)
        emitted = 0
        while emitted < self.config.parent_candidate_budget:
            progressed = False
            for queue_index, (parent_rank, siren, sites) in enumerate(queues):
                position = positions[queue_index]
                if position >= len(sites):
                    continue
                progressed = True
                positions[queue_index] += 1
                emitted += 1
                site = sites[position]
                self._register_signal(
                    references,
                    site,
                    OfficialEvidenceSignal(
                        source="v11_parent:siren->geo_sidecar:site",
                        channel="v11_parent_expansion",
                        rank=emitted,
                        score=1.0 / parent_rank,
                        parent_identifier=siren,
                        parent_rank=parent_rank,
                        parent_support_count=parent_support[siren],
                        site_rank=position + 1,
                    ),
                )
                if emitted >= self.config.parent_candidate_budget:
                    break
            if not progressed:
                break
        return emitted

    @staticmethod
    def _fair_select(
        references: Mapping[str, _Reference], cap: int
    ) -> list[_Reference]:
        anchors = sorted(
            (reference for reference in references.values() if _joint_anchor(reference)),
            key=_reference_priority,
        )
        selected = anchors[:cap]
        selected_ids = {reference.siret for reference in selected}
        if len(selected) >= cap:
            return selected

        queues: dict[str, list[_Reference]] = {}
        for reference in references.values():
            if reference.siret in selected_ids:
                continue
            queues.setdefault(reference.siren or reference.siret, []).append(reference)
        for queue in queues.values():
            queue.sort(key=_reference_priority)
        parents = sorted(
            queues,
            key=lambda parent: (_reference_priority(queues[parent][0]), parent),
        )
        positions = {parent: 0 for parent in parents}
        while len(selected) < cap:
            progressed = False
            for parent in parents:
                position = positions[parent]
                queue = queues[parent]
                if position >= len(queue):
                    continue
                progressed = True
                reference = queue[position]
                positions[parent] += 1
                selected.append(reference)
                if len(selected) >= cap:
                    break
            if not progressed:
                break
        return selected

    def retrieve(self, row: Mapping[str, Any] | RetrievalQuery) -> RetrievalV11Result:
        started = time.perf_counter()
        query = row if isinstance(row, RetrievalQuery) else RetrievalQuery.from_mapping(row)
        # The wider prefix is explicitly reference-only.  The v11 selector
        # below is the only component allowed to publish the 2,000 candidates.
        official_started = time.perf_counter()
        official_result = self.official.retrieve(
            query, candidate_cap=self.config.raw_reference_cap
        )
        official_ms = (time.perf_counter() - official_started) * 1000.0
        references: dict[str, _Reference] = {}
        self._merge_official(references, official_result.candidates)
        geo_started = time.perf_counter()
        geo_strategy, geo_count, geo_pool_size = self._add_geographic_candidates(
            query, references
        )
        geo_ms = (time.perf_counter() - geo_started) * 1000.0
        parent_started = time.perf_counter()
        parent_count = self._expand_parents(query, references)
        parent_ms = (time.perf_counter() - parent_started) * 1000.0
        raw_count = len(references)
        raw_reference_truncated = (
            official_result.raw_union_size > self.config.raw_reference_cap
            or raw_count > self.config.raw_reference_cap
        )
        if raw_count > self.config.raw_reference_cap:
            references = {
                reference.siret: reference
                for reference in self._fair_select(
                    references, self.config.raw_reference_cap
                )
            }
        selection_started = time.perf_counter()
        selected = self._fair_select(references, self.config.internal_cap)
        candidates: list[OfficialEvidenceCandidate] = []
        for rank, reference in enumerate(selected, start=1):
            record = reference.record or self._canonical_backend.by_siret(reference.siret)
            if record is None:
                continue
            candidates.append(
                OfficialEvidenceCandidate(
                    union_rank=rank,
                    siret=reference.siret,
                    siren=reference.siren or record.siren,
                    record=record,
                    signals=tuple(
                        sorted(
                            reference.signals,
                            key=lambda signal: (signal.rank, signal.channel, signal.source),
                        )
                    ),
                    official_evidence_ids=tuple(sorted(reference.evidence_ids)),
                    official_evidence_sources=tuple(sorted(reference.evidence_sources)),
                )
            )
        if len(candidates) > self.config.internal_cap:
            raise AssertionError("v11 internal candidate cap exceeded")
        return RetrievalV11Result(
            query=query,
            candidates=tuple(candidates),
            raw_reference_count=raw_count,
            raw_reference_truncated=raw_reference_truncated,
            retrieval_latency_ms=(time.perf_counter() - started) * 1000.0,
            strategy_counts={
                "official_reference_prefix": len(official_result.candidates),
                "geographic_candidates": geo_count,
                "geographic_pool_size": geo_pool_size,
                "parent_expansion_emissions": parent_count,
                "selected": len(candidates),
                "geo_strategy": geo_strategy,
                "official_ms": official_ms,
                "geo_ms": geo_ms,
                "parent_ms": parent_ms,
                "selection_hydration_ms": (
                    time.perf_counter() - selection_started
                )
                * 1000.0,
            },
        )


__all__ = [
    "AllHumanV11Retriever",
    "RetrievalV11Config",
    "RetrievalV11Result",
    "V11_RETRIEVAL_SCHEMA_VERSION",
    "V11_CHANNELS",
]
