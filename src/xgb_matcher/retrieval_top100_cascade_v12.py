"""Top-100-first, label-blind retrieval cascade for all-human v12.

The cascade consumes the *lightweight* references emitted by retrieval v11;
it does not call the national index again and it never hydrates a candidate.
It aggregates evidence at SIREN level, expands the most credible parents with
the official v11 sidecar and produces three bounded views:

* at most 2,000 master SIRET references scored by the site LambdaMART,
* at most 64 parent SIREN references,
* an at-most-500 upstream heuristic shortlist retained only as a feature and
  diagnostic; it never controls learned PreRank@500 eligibility.

Identifiers are grouping keys and deterministic tie-breakers only.  Human
truth, CRM aliases and synthetic data are deliberately absent from the public
types and scoring functions.  A conditional SIREN rescue can be injected via
the small :class:`ParentRescueProvider` protocol; the cascade calls it at most
once and only when the frozen, label-blind evidence sufficiency rule fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .hierarchical_retrieval import RetrievalQuery, normalize_text
from .official_evidence_retrieval import LTR_CHANNELS
from .retrieval_sidecars_v11 import LocalSite, SiteQueryResult


V12_CASCADE_SCHEMA_VERSION = "sireto-retrieval-top100-cascade-v12-1"
V12_PARENT_SCHEMA_VERSION = "sireto-retrieval-top100-parent-v12-1"
V12_OUTPUT_CHANNELS = tuple(
    dict.fromkeys(
        (
            *LTR_CHANNELS,
            "geo_insee_census",
            "geo_exact_cell",
            "geo_number_cell",
            "v11_parent_expansion",
            "v12_conditional_siren_rescue",
            "v12_parent_local_expansion",
            "v12_rescue_local_expansion",
        )
    )
)

# This exact order is shared by parent-ranker fit, inner-OOF prediction and
# serving-time expansion.  Adding/reordering a feature requires a new schema.
CASCADE_PARENT_FEATURES = (
    "exact_name_and_address",
    "name_and_exact_address",
    "exact_name_evidence",
    "name_evidence",
    "exact_address_evidence",
    "address_evidence",
    "geographic_support",
    "joint_site_log1p",
    "official_source_count",
    "parent_source_registry_present",
    "parent_source_rne_present",
    "parent_source_bodacc_present",
    "parent_source_sector_present",
    "channel_count",
    "support_site_log1p",
    "channel_rrf",
    "rescue_evidence",
    "rescue_score",
)

_SIRET = re.compile(r"^[0-9]{14}$")
_SIREN = re.compile(r"^[0-9]{9}$")
_LEADING_NUMBER = re.compile(r"^\s*\d{1,5}(?:\s*(?:BIS|TER|QUATER|[A-Z]))?\s+")

_NAME_EXACT = frozenset(
    {"legal_name_exact", "trade_name_exact", "site_name_exact", "name_exact"}
)
_NAME_EVIDENCE = _NAME_EXACT | frozenset(
    {
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
        "contrastive_siren_rescue",
        "v12_conditional_siren_rescue",
    }
)
_ADDRESS_EXACT = frozenset({"address_exact", "number_exact", "geo_exact_cell"})
_ADDRESS_EVIDENCE = _ADDRESS_EXACT | frozenset(
    {
        "address_word",
        "address_char",
        "historical_address_word",
        "supporting_address_word",
        "rne_address",
        "bodacc_address",
        "sector_address",
        "geo_insee_census",
        "geo_number_cell",
    }
)
_RELATION_CHANNELS = frozenset({"official_successor", "bodacc_relation"})
DEFERRED_RESCUE_CHANNELS = frozenset(
    {"contrastive_siren_rescue", "v12_conditional_siren_rescue"}
)
_GEO_CHANNELS = frozenset(
    {"geo_insee_census", "geo_exact_cell", "geo_number_cell"}
)


def _source_families(values: Iterable[str]) -> set[str]:
    families: set[str] = set()
    for value in values:
        for token in filter(None, re.split(r"[+,|;\s]+", normalize_text(value))):
            if "BODACC" in token:
                families.add("BODACC")
            elif token in {"RNE", "INPI"} or "RNE" in token:
                families.add("RNE")
            elif token in {"SIRENE", "SIREN", "INSEE"} or "SIRENE" in token:
                families.add("SIRENE")
            elif "SECTOR" in token or "SECTEUR" in token:
                families.add("SECTOR")
    return families

# These columns may exist in an audit artefact.  Their values are never read,
# copied or compared by this module.  The CLI also projects them out at the
# Parquet scanner boundary.
_FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "ground_truth_siret",
        "historical_ground_truth_siret",
        "human_ground_truth_siret",
        "human_ground_truth_siren",
        "human_ground_truth_source",
        "acceptable_sirets_operational",
        "acceptable_sirets_operational_json",
        "crm_alias",
        "crm_aliases",
        "alias_crm",
        "label_kind",
        "source_kind",
        "unseen_siren",
        "new_site_known_siren",
        "human_siret",
        "gt_siret",
        "target_siret",
        "label_siret",
        "correct_siret",
        "expected_siret",
        "operational_sirets",
        "operational_sirets_json",
        "is_positive",
        "relevance",
        "match_label",
    }
)


def is_forbidden_input_key(name: str) -> bool:
    """Return whether ``name`` is truth-side or a private CRM alias field."""

    value = str(name).strip().lower()
    return (
        value in _FORBIDDEN_EXACT_KEYS
        or value.startswith("ground_truth_")
        or value.startswith("human_ground_truth_")
        or value.startswith("acceptable_siret")
        or value.startswith("crm_alias")
        or value.startswith("gt_")
        or value.startswith("label_")
        or value.endswith("_ground_truth_siret")
    )


def production_projection(columns: Iterable[str]) -> tuple[str, ...]:
    """Project an input schema without ever loading truth-side columns."""

    return tuple(str(column) for column in columns if not is_forbidden_input_key(column))


def _clean_row(row: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        str(key): value
        for key, value in row.items()
        if not is_forbidden_input_key(str(key))
    }
    return values


def _first(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("v12 budgets must be positive integers")
    return parsed


@dataclass(frozen=True)
class RetrievalTop100CascadeV12Config:
    """Frozen serving budgets; the JSON loader accepts contract aliases."""

    master_reference_cap: int = 2_000
    hydration_shortlist_cap: int = 500
    parent_shortlist_cap: int = 64
    small_insee_threshold: int = 100
    parent_expansion_budget: int = 700
    max_sites_per_parent: int = 2_000
    relation_seed_limit: int = 32
    relation_reference_budget: int = 64
    rescue_parent_budget: int = 16
    rescue_parent_reserve: int = 8
    rescue_min_name_parents: int = 12
    rescue_min_strong_parents: int = 2
    rrf_constant: float = 60.0

    @classmethod
    def load(cls, path: Path | str) -> "RetrievalTop100CascadeV12Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        architecture = _mapping(raw.get("architecture"))
        retrieval = _mapping(raw.get("retrieval"))
        cascade = _mapping(
            _first(
                raw.get("retrieval_top100_cascade_v12"),
                raw.get("cascade"),
                retrieval.get("cascade"),
                default={},
            )
        )
        budgets = _mapping(_first(cascade.get("budgets"), raw.get("budgets"), default={}))
        rescue = _mapping(
            _first(
                cascade.get("conditional_rescue"),
                retrieval.get("conditional_rescue"),
                default={},
            )
        )

        def get(*keys: str, default: Any) -> Any:
            for source in (cascade, budgets, architecture, retrieval, raw):
                for key in keys:
                    if source.get(key) is not None:
                        return source[key]
            return default

        return cls(
            master_reference_cap=_positive_int(
                get(
                    "master_reference_cap",
                    "master_siret_reference_cap",
                    "maximum_internal_candidates",
                    default=2_000,
                ),
                2_000,
            ),
            hydration_shortlist_cap=_positive_int(
                get(
                    "hydration_shortlist_cap",
                    "maximum_hydrated_sirets",
                    "normal_shortlist_siret_cap",
                    default=500,
                ),
                500,
            ),
            parent_shortlist_cap=_positive_int(
                get("parent_shortlist_cap", "parent_siren_limit", default=64), 64
            ),
            small_insee_threshold=_positive_int(
                get("small_insee_threshold", "small_commune_threshold", default=100),
                100,
            ),
            parent_expansion_budget=_positive_int(
                get("parent_expansion_budget", "parent_expansion_candidates", default=700),
                700,
            ),
            max_sites_per_parent=_positive_int(
                get("max_sites_per_parent", default=2_000), 2_000
            ),
            relation_seed_limit=_positive_int(
                get("relation_seed_limit", default=32), 32
            ),
            relation_reference_budget=_positive_int(
                get("relation_reference_budget", default=64), 64
            ),
            rescue_parent_budget=_positive_int(
                _first(rescue.get("parent_budget"), rescue.get("limit"), default=16),
                16,
            ),
            rescue_parent_reserve=_positive_int(
                _first(rescue.get("parent_reserve"), default=8), 8
            ),
            rescue_min_name_parents=_positive_int(
                _first(rescue.get("minimum_name_parents"), default=12), 12
            ),
            rescue_min_strong_parents=_positive_int(
                _first(rescue.get("minimum_strong_parents"), default=2), 2
            ),
            rrf_constant=float(get("rrf_constant", default=60.0)),
        )

    def __post_init__(self) -> None:
        if not 1 <= self.master_reference_cap <= 2_000:
            raise ValueError("v12 master_reference_cap must be in [1, 2000]")
        if not 1 <= self.hydration_shortlist_cap <= 500:
            raise ValueError("v12 hydration_shortlist_cap must be in [1, 500]")
        if self.hydration_shortlist_cap > self.master_reference_cap:
            raise ValueError("v12 hydration shortlist must fit in master references")
        if not 1 <= self.parent_shortlist_cap <= 64:
            raise ValueError("v12 parent_shortlist_cap must be in [1, 64]")
        if not 1 <= self.small_insee_threshold <= 100:
            raise ValueError("v12 small_insee_threshold must be in [1, 100]")
        if not 1 <= self.parent_expansion_budget <= self.master_reference_cap:
            raise ValueError("v12 parent expansion budget must fit in master cap")
        if not 1 <= self.max_sites_per_parent <= 2_000:
            raise ValueError("v12 max_sites_per_parent must be in [1, 2000]")
        if not 1 <= self.relation_seed_limit <= 32:
            raise ValueError("v12 relation_seed_limit must be in [1, 32]")
        if self.relation_reference_budget > self.master_reference_cap:
            raise ValueError("v12 relation references must fit in master cap")
        if self.rescue_parent_budget > self.parent_shortlist_cap:
            raise ValueError("v12 rescue parent budget must fit in parent shortlist")
        if self.rescue_parent_reserve > self.rescue_parent_budget:
            raise ValueError("v12 rescue reserve must fit in rescue budget")
        if self.rrf_constant <= 0:
            raise ValueError("v12 rrf_constant must be positive")


@dataclass(frozen=True)
class CascadeSignalV12:
    source: str
    channel: str
    rank: int
    score: float = 0.0
    parent_identifier: str = ""
    parent_rank: int | None = None
    parent_support_count: int | None = None
    site_rank: int | None = None
    relation_direction: str = ""
    relation_date: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CascadeSignalV12":
        return cls(
            source=str(value.get("source") or ""),
            channel=str(value.get("channel") or ""),
            rank=max(1, int(value.get("rank") or 2_147_483_647)),
            score=float(value.get("score") or 0.0),
            parent_identifier=str(value.get("parent_identifier") or ""),
            parent_rank=(
                int(value["parent_rank"])
                if value.get("parent_rank") not in (None, "")
                else None
            ),
            parent_support_count=(
                int(value["parent_support_count"])
                if value.get("parent_support_count") not in (None, "")
                else None
            ),
            site_rank=(
                int(value["site_rank"])
                if value.get("site_rank") not in (None, "")
                else None
            ),
            relation_direction=str(value.get("relation_direction") or ""),
            relation_date=str(
                value.get("relation_date")
                or value.get("event_date")
                or value.get("date")
                or ""
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "channel": self.channel,
            "rank": self.rank,
            "score": self.score,
            "parent_identifier": self.parent_identifier,
            "parent_rank": self.parent_rank,
            "parent_support_count": self.parent_support_count,
            "site_rank": self.site_rank,
            "relation_direction": self.relation_direction,
            "relation_date": self.relation_date,
        }


@dataclass(frozen=True)
class RescueParentHitV12:
    siren: str
    rank: int
    score: float
    source: str = "contrastive_siren_rescue"


class ParentRescueProvider(Protocol):
    """Inference-only rescue.  Implementations must be frozen before OOF."""

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        limit: int,
        query_id: str = "",
    ) -> Sequence[RescueParentHitV12 | Mapping[str, Any] | Sequence[Any]]: ...


class ParentScorerV12(Protocol):
    """Frozen inner-OOF scorer receiving features without identifiers."""

    def score(self, features: Mapping[str, float]) -> float: ...


@dataclass(frozen=True)
class XGBoostParentScorerV12:
    """Thin adapter for the same frozen parent ranker used by admission."""

    model: Any

    @classmethod
    def from_model_path(cls, path: Path | str) -> "XGBoostParentScorerV12":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        return cls(booster)

    def score(self, features: Mapping[str, float]) -> float:
        if tuple(features) != CASCADE_PARENT_FEATURES:
            raise ValueError("v12 parent feature schema/order drift")
        import numpy as np

        values = np.asarray(
            [[float(features[name]) for name in CASCADE_PARENT_FEATURES]],
            dtype=np.float32,
        )
        # Booster and sklearn wrappers expose different predict signatures.
        if self.model.__class__.__name__ == "Booster":
            import xgboost as xgb

            output = self.model.predict(
                xgb.DMatrix(values, feature_names=list(CASCADE_PARENT_FEATURES))
            )
        else:
            output = self.model.predict(values)
        return float(np.asarray(output).reshape(-1)[0])


class SidecarReaderV12(Protocol):
    def all_sites_by_insee_if_fits(
        self, insee: str, *, max_sites: int
    ) -> SiteQueryResult: ...

    def sites_by_siren_geo(self, **kwargs: Any) -> SiteQueryResult: ...


@dataclass
class _Reference:
    siret: str
    siren: str
    channels: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    signals: list[CascadeSignalV12] = field(default_factory=list)
    evidence_ids: set[str] = field(default_factory=set)
    evidence_sources: set[str] = field(default_factory=set)
    lanes: set[str] = field(default_factory=set)
    reasons: set[str] = field(default_factory=set)
    channel_ranks: dict[str, int] = field(default_factory=dict)
    names: tuple[str, ...] = ()
    addresses: tuple[str, ...] = ()
    insee: str = ""
    postcode: str = ""
    number: str = ""
    state: str = ""
    is_headquarters: bool = False
    temporal_valid_from: str = ""
    temporal_valid_to: str = ""
    input_rank: int = 2_147_483_647
    parent_rank_hint: int | None = None
    parent_score_hint: float = 0.0
    site_score: float = 0.0

    def merge_signal(self, signal: CascadeSignalV12) -> None:
        if not signal.channel:
            return
        key = (
            signal.source,
            signal.channel,
            signal.rank,
            signal.parent_identifier,
            signal.parent_support_count,
            signal.relation_direction,
            signal.relation_date,
        )
        existing = {
            (
                item.source,
                item.channel,
                item.rank,
                item.parent_identifier,
                item.parent_support_count,
                item.relation_direction,
                item.relation_date,
            )
            for item in self.signals
        }
        if key not in existing:
            self.signals.append(signal)
        self.channels.add(signal.channel)
        if signal.source:
            self.sources.add(signal.source)
        self.channel_ranks[signal.channel] = min(
            self.channel_ranks.get(signal.channel, signal.rank), signal.rank
        )


@dataclass(frozen=True)
class ParentReferenceV12:
    siren: str
    parent_rank: int
    parent_score: float
    support_sites: int
    channels: tuple[str, ...]
    sources: tuple[str, ...]
    reasons: tuple[str, ...]
    name_evidence: bool
    address_evidence: bool
    exact_name_evidence: bool
    exact_address_evidence: bool
    geographic_support: bool
    rescue: bool
    feature_values: tuple[float, ...]

    def to_row(self, query_id: str) -> dict[str, Any]:
        return {
            "schema_version": V12_PARENT_SCHEMA_VERSION,
            "query_id": str(query_id),
            "siren": self.siren,
            "parent_rank": self.parent_rank,
            "parent_score": self.parent_score,
            "support_sites": self.support_sites,
            "channels": list(self.channels),
            "sources": list(self.sources),
            "selection_reasons": list(self.reasons),
            "name_evidence": self.name_evidence,
            "address_evidence": self.address_evidence,
            "exact_name_evidence": self.exact_name_evidence,
            "exact_address_evidence": self.exact_address_evidence,
            "geographic_support": self.geographic_support,
            "rescue": self.rescue,
            "feature_names": list(CASCADE_PARENT_FEATURES),
            "feature_values": list(self.feature_values),
        }


@dataclass(frozen=True)
class CascadeCandidateV12:
    siret: str
    siren: str
    master_rank: int
    hydration_rank: int | None
    parent_rank: int | None
    site_rank: int
    parent_score: float
    parent_feature_names: tuple[str, ...]
    parent_feature_values: tuple[float, ...]
    site_score: float
    selection_lane: str
    lanes: tuple[str, ...]
    reasons: tuple[str, ...]
    channels: tuple[str, ...]
    sources: tuple[str, ...]
    signals: tuple[CascadeSignalV12, ...]
    official_evidence_ids: tuple[str, ...]
    official_evidence_sources: tuple[str, ...]
    candidate_names: tuple[str, ...]
    candidate_addresses: tuple[str, ...]
    candidate_insee: str
    candidate_postcode: str
    candidate_number: str
    candidate_state: str
    candidate_is_headquarters: bool
    temporal_valid_from: str
    temporal_valid_to: str

    def to_row(
        self,
        query_id: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        union_rank: int | None = None,
    ) -> dict[str, Any]:
        safe_metadata = {
            str(key): value
            for key, value in (metadata or {}).items()
            if not is_forbidden_input_key(str(key))
        }
        best_by_channel: dict[str, CascadeSignalV12] = {}
        for signal in self.signals:
            current = best_by_channel.get(signal.channel)
            if current is None or (signal.rank, -signal.score, signal.source) < (
                current.rank,
                -current.score,
                current.source,
            ):
                best_by_channel[signal.channel] = signal
        channel_values: dict[str, Any] = {}
        for channel in V12_OUTPUT_CHANNELS:
            signal = best_by_channel.get(channel)
            channel_values[f"{channel}_rank"] = signal.rank if signal else None
            channel_values[f"{channel}_score"] = signal.score if signal else None
        return {
            **safe_metadata,
            "schema_version": V12_CASCADE_SCHEMA_VERSION,
            "query_id": str(query_id),
            "siret": self.siret,
            "siren": self.siren,
            "union_rank": int(
                union_rank
                if union_rank is not None
                else (self.hydration_rank or self.master_rank)
            ),
            "master_rank": self.master_rank,
            "hydration_rank": self.hydration_rank,
            "parent_rank": self.parent_rank,
            "site_rank": self.site_rank,
            "parent_score": self.parent_score,
            "parent_feature_names": list(self.parent_feature_names),
            "parent_feature_values": list(self.parent_feature_values),
            "site_score": self.site_score,
            "selection_lane": self.selection_lane,
            "lanes": list(self.lanes),
            "selection_reasons": list(self.reasons),
            "channels": list(self.channels),
            "sources": list(self.sources),
            "signals": [signal.to_dict() for signal in self.signals],
            "official_evidence_ids": list(self.official_evidence_ids),
            "official_evidence_sources": list(self.official_evidence_sources),
            "candidate_names": list(self.candidate_names),
            "candidate_addresses": list(self.candidate_addresses),
            "candidate_insee": self.candidate_insee,
            "candidate_postcode": self.candidate_postcode,
            "candidate_number": self.candidate_number,
            "candidate_state": self.candidate_state,
            "candidate_is_headquarters": self.candidate_is_headquarters,
            "temporal_valid_from": self.temporal_valid_from,
            "temporal_valid_to": self.temporal_valid_to,
            **channel_values,
        }


@dataclass(frozen=True)
class RetrievalTop100CascadeV12Result:
    query: RetrievalQuery
    master_references: tuple[CascadeCandidateV12, ...]
    hydration_shortlist: tuple[CascadeCandidateV12, ...]
    parent_candidates: tuple[ParentReferenceV12, ...]
    parent_shortlist: tuple[ParentReferenceV12, ...]
    rescue_triggered: bool
    rescue_executed: bool
    strategy_counts: Mapping[str, Any]

    def master_rows(
        self, query_id: str, metadata: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return [
            candidate.to_row(query_id, metadata, union_rank=candidate.master_rank)
            for candidate in self.master_references
        ]

    def admission_rows(
        self, query_id: str, metadata: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Serialize the legacy heuristic shortlist (diagnostic only).

        The historical method name is retained for artifact compatibility;
        v12 learned admission consumes :meth:`master_rows` instead.
        """
        return [
            candidate.to_row(
                query_id, metadata, union_rank=candidate.hydration_rank
            )
            for candidate in self.hydration_shortlist
        ]

    def parent_rows(self, query_id: str) -> list[dict[str, Any]]:
        return [parent.to_row(query_id) for parent in self.parent_shortlist]

    def parent_candidate_rows(self, query_id: str) -> list[dict[str, Any]]:
        """All bounded pre-expansion parents for OOF ranker fit/scoring."""

        return [parent.to_row(query_id) for parent in self.parent_candidates]


@dataclass
class _Parent:
    siren: str
    references: list[_Reference] = field(default_factory=list)
    channels: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    score: float = 0.0
    rescue: bool = False
    rescue_rank: int | None = None
    rescue_score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)

    @property
    def name_evidence(self) -> bool:
        return bool(self.channels & _NAME_EVIDENCE)

    @property
    def address_evidence(self) -> bool:
        return bool(self.channels & _ADDRESS_EVIDENCE)

    @property
    def exact_name_evidence(self) -> bool:
        return bool(self.channels & _NAME_EXACT)

    @property
    def exact_address_evidence(self) -> bool:
        return bool(self.channels & _ADDRESS_EXACT)

    @property
    def geographic_support(self) -> bool:
        return any(reference.channels & _GEO_CHANNELS for reference in self.references)

    @property
    def strong(self) -> bool:
        return self.exact_name_evidence or (
            self.name_evidence and self.address_evidence
        )


def _as_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if item not in (None, ""))
    return (str(value),)


def _valid_siret(value: Any) -> str:
    text = re.sub(r"\D", "", str(value or ""))
    return text if _SIRET.fullmatch(text) else ""


def _valid_siren(value: Any) -> str:
    text = re.sub(r"\D", "", str(value or ""))
    return text if _SIREN.fullmatch(text) else ""


def _street_query(query: RetrievalQuery) -> str:
    return normalize_text(_LEADING_NUMBER.sub("", query.address))


def _geo_compatible(reference: _Reference, query: RetrievalQuery) -> bool:
    if query.insee:
        return bool(reference.insee and reference.insee == query.insee)
    if query.postcode:
        return bool(reference.postcode and reference.postcode == query.postcode)
    return False


def _structured_relation(signal: CascadeSignalV12) -> bool:
    if signal.channel not in _RELATION_CHANNELS:
        return False
    parent = re.sub(r"\D", "", signal.parent_identifier)
    if len(parent) not in (9, 14):
        return False
    source = signal.source.upper()
    establishment_relation = "STRUCTURED" in source and (
        "BODACC" in source or "SIRENE" in source
    )
    # official_evidence_retrieval emits linked_sirens() traversal as
    # ``<bodacc backend>:siren_relation-><site backend>:site``.  The typed
    # backend method and the 9-digit parent make this structured even though
    # the source label does not literally contain the word ``structured``.
    siren_relation = (
        len(parent) == 9
        and "BODACC" in source
        and ":SIREN_RELATION->" in source
    )
    return establishment_relation or siren_relation


def _signals_from_row(row: Mapping[str, Any]) -> list[CascadeSignalV12]:
    output: list[CascadeSignalV12] = []
    for raw in row.get("signals") or ():
        if isinstance(raw, Mapping):
            output.append(CascadeSignalV12.from_mapping(raw))
    known_channels = set(_as_strings(row.get("channels")))
    known_channels.update(
        key[:-5]
        for key, value in row.items()
        if key.endswith("_rank")
        and value not in (None, "")
        and key not in {"union_rank", "retrieval_rank", "parent_rank", "site_rank"}
    )
    existing = {signal.channel for signal in output}
    for channel in sorted(known_channels - existing):
        rank_value = row.get(f"{channel}_rank")
        rank = int(rank_value or row.get("union_rank") or row.get("retrieval_rank") or 2_147_483_647)
        output.append(
            CascadeSignalV12(
                source=str(row.get("retrieval_source") or "v11_lightweight_union"),
                channel=channel,
                rank=max(1, rank),
                score=float(row.get(f"{channel}_score") or 0.0),
            )
        )
    return output


def _relation_seed_parents(
    rows: Sequence[Mapping[str, Any]], query: RetrievalQuery, limit: int
) -> set[str]:
    ranked: list[tuple[int, str, str]] = []
    for raw in rows:
        row = _clean_row(raw)
        reference = _reference_shell(row)
        if reference is None or not _geo_compatible(reference, query):
            continue
        for signal in _signals_from_row(row):
            if _structured_relation(signal):
                ranked.append((signal.rank, signal.parent_identifier, reference.siret))
    output: list[str] = []
    for _rank, parent, _siret in sorted(ranked):
        if parent not in output:
            output.append(parent)
        if len(output) >= limit:
            break
    return set(output)


def _reference_shell(row: Mapping[str, Any]) -> _Reference | None:
    siret = _valid_siret(row.get("siret") or row.get("candidate_siret"))
    if not siret:
        return None
    siren = _valid_siren(row.get("siren") or row.get("candidate_siren")) or siret[:9]
    return _Reference(
        siret=siret,
        siren=siren,
        names=_as_strings(row.get("candidate_names") or row.get("names")),
        addresses=_as_strings(row.get("candidate_addresses") or row.get("addresses")),
        insee=str(row.get("candidate_insee") or row.get("insee_etablissement") or ""),
        postcode=str(row.get("candidate_postcode") or row.get("postcode_etablissement") or ""),
        number=str(row.get("candidate_number") or row.get("numeroVoie") or ""),
        state=str(row.get("candidate_state") or row.get("etat_admin") or ""),
        is_headquarters=bool(row.get("candidate_is_siege") or row.get("is_siege") or False),
        temporal_valid_from=str(row.get("temporal_valid_from") or ""),
        temporal_valid_to=str(row.get("temporal_valid_to") or ""),
        input_rank=max(1, int(row.get("union_rank") or row.get("retrieval_rank") or 2_147_483_647)),
    )


def _merge_reference(target: _Reference, incoming: _Reference) -> None:
    target.channels.update(incoming.channels)
    target.sources.update(incoming.sources)
    for signal in incoming.signals:
        target.merge_signal(signal)
    target.evidence_ids.update(incoming.evidence_ids)
    target.evidence_sources.update(incoming.evidence_sources)
    target.lanes.update(incoming.lanes)
    target.reasons.update(incoming.reasons)
    for channel, rank in incoming.channel_ranks.items():
        target.channel_ranks[channel] = min(target.channel_ranks.get(channel, rank), rank)
    target.names = tuple(dict.fromkeys((*target.names, *incoming.names)))
    target.addresses = tuple(dict.fromkeys((*target.addresses, *incoming.addresses)))
    target.insee = target.insee or incoming.insee
    target.postcode = target.postcode or incoming.postcode
    target.number = target.number or incoming.number
    target.state = target.state or incoming.state
    target.is_headquarters = target.is_headquarters or incoming.is_headquarters
    target.temporal_valid_from = target.temporal_valid_from or incoming.temporal_valid_from
    target.temporal_valid_to = target.temporal_valid_to or incoming.temporal_valid_to
    target.input_rank = min(target.input_rank, incoming.input_rank)
    if incoming.parent_rank_hint is not None:
        target.parent_rank_hint = (
            incoming.parent_rank_hint
            if target.parent_rank_hint is None
            else min(target.parent_rank_hint, incoming.parent_rank_hint)
        )
    target.parent_score_hint = max(target.parent_score_hint, incoming.parent_score_hint)


def _reference_from_row(
    raw: Mapping[str, Any],
    *,
    query: RetrievalQuery,
    allowed_relation_parents: set[str],
) -> _Reference | None:
    row = _clean_row(raw)
    reference = _reference_shell(row)
    if reference is None:
        return None
    relation_seen = False
    non_relation_seen = False
    for signal in _signals_from_row(row):
        if signal.channel in _RELATION_CHANNELS:
            relation_seen = True
            if (
                not _structured_relation(signal)
                or signal.parent_identifier not in allowed_relation_parents
                or not _geo_compatible(reference, query)
            ):
                continue
            reference.lanes.add("official_relation_1hop")
            reference.reasons.add("STRUCTURED_OFFICIAL_RELATION_GEO_COMPATIBLE")
        else:
            non_relation_seen = True
        reference.merge_signal(signal)
    if relation_seen and not non_relation_seen and not (reference.channels & _RELATION_CHANNELS):
        return None
    reference.sources.update(_as_strings(row.get("sources")))
    reference.evidence_ids.update(_as_strings(row.get("official_evidence_ids")))
    reference.evidence_sources.update(
        _as_strings(row.get("official_evidence_sources"))
    )
    reference.lanes.add("v11_lightweight")
    reference.reasons.add("V11_NATURAL_REFERENCE")
    return reference


def _site_reference(
    site: LocalSite,
    *,
    signal: CascadeSignalV12,
    lane: str,
    reason: str,
) -> _Reference:
    reference = _Reference(
        siret=site.siret,
        siren=site.siren,
        addresses=(site.current_address_normalized,) if site.current_address_normalized else (),
        insee=site.insee,
        postcode=site.postcode,
        number=site.street_number,
        state=site.administrative_state,
        is_headquarters=site.is_headquarters,
        lanes={lane},
        reasons={reason},
        input_rank=signal.rank,
        parent_rank_hint=signal.parent_rank,
    )
    reference.merge_signal(signal)
    return reference


def _joint_evidence(reference: _Reference) -> bool:
    return bool(reference.channels & _NAME_EVIDENCE) and bool(
        reference.channels & _ADDRESS_EVIDENCE
    )


def _parent_features(
    parent: _Parent, query: RetrievalQuery, rrf_constant: float
) -> dict[str, float]:
    ranks: dict[str, int] = {}
    geo_compatible = 0
    joint = 0
    evidence_sources: set[str] = set(parent.sources)
    for reference in parent.references:
        geo_compatible += int(_geo_compatible(reference, query))
        joint += int(_joint_evidence(reference))
        evidence_sources.update(reference.evidence_sources)
        for channel, rank in reference.channel_ranks.items():
            ranks[channel] = min(ranks.get(channel, rank), rank)
    rrf = sum(1.0 / (rrf_constant + max(1, rank)) for rank in ranks.values())
    source_families = _source_families(evidence_sources)
    values = {
        "exact_name_and_address": float(
            parent.exact_name_evidence and parent.address_evidence
        ),
        "name_and_exact_address": float(
            parent.name_evidence and parent.exact_address_evidence
        ),
        "exact_name_evidence": float(parent.exact_name_evidence),
        "name_evidence": float(parent.name_evidence),
        "exact_address_evidence": float(parent.exact_address_evidence),
        "address_evidence": float(parent.address_evidence),
        "geographic_support": float(geo_compatible > 0),
        "joint_site_log1p": float(math.log1p(joint)),
        "official_source_count": float(len(source_families)),
        "parent_source_registry_present": float("SIRENE" in source_families),
        "parent_source_rne_present": float("RNE" in source_families),
        "parent_source_bodacc_present": float("BODACC" in source_families),
        "parent_source_sector_present": float("SECTOR" in source_families),
        "channel_count": float(len(ranks)),
        "support_site_log1p": float(math.log1p(len(parent.references))),
        "channel_rrf": float(rrf),
        "rescue_evidence": float(parent.rescue),
        "rescue_score": float(parent.rescue_score if parent.rescue else 0.0),
    }
    return {name: values[name] for name in CASCADE_PARENT_FEATURES}


def _default_parent_score(features: Mapping[str, float]) -> float:
    score = 0.0
    score += 8.0 * features["exact_name_and_address"]
    score += 6.0 * features["name_and_exact_address"]
    score += 4.0 * features["exact_name_evidence"]
    score += 2.5 * features["name_evidence"]
    score += 2.0 * features["exact_address_evidence"]
    score += 1.25 * features["address_evidence"]
    score += 2.0 * features["geographic_support"]
    score += min(3.0, features["joint_site_log1p"])
    score += min(2.0, 0.35 * features["official_source_count"])
    score += min(2.0, 0.20 * features["channel_count"])
    score += min(1.5, 0.25 * features["support_site_log1p"])
    score += 100.0 * features["channel_rrf"]
    # Rescue evidence is deliberately weaker than joint sparse evidence;
    # reserved parent slots, not score inflation, guarantee its audit lane.
    score += min(1.0, max(0.0, features["rescue_score"]))
    return float(score)


def _site_score(reference: _Reference, parent_score: float, query: RetrievalQuery) -> float:
    channels = reference.channels
    best_rank = min(reference.channel_ranks.values(), default=reference.input_rank)
    source_count = len(reference.sources | reference.evidence_sources)
    score = 0.20 * parent_score
    score += 10.0 * int(_joint_evidence(reference))
    score += 4.0 * int(bool(channels & _NAME_EXACT))
    score += 3.0 * int(bool(channels & _ADDRESS_EXACT))
    score += 2.0 * int(bool(channels & _NAME_EVIDENCE))
    score += 1.0 * int(bool(channels & _ADDRESS_EVIDENCE))
    score += 2.0 * int(_geo_compatible(reference, query))
    score += min(2.0, 0.25 * len(channels))
    score += min(1.5, 0.30 * source_count)
    score += 4.0 / math.log2(max(2, best_rank + 1))
    score += 0.10 * int(reference.state == "A")
    return float(score)


def _normalise_rescue_hit(value: Any, fallback_rank: int) -> RescueParentHitV12 | None:
    if isinstance(value, RescueParentHitV12):
        hit = value
    elif isinstance(value, Mapping):
        hit = RescueParentHitV12(
            siren=str(value.get("siren") or ""),
            rank=int(value.get("rank") or fallback_rank),
            score=float(value.get("score") or 0.0),
            source=str(value.get("source") or "contrastive_siren_rescue"),
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            hit = RescueParentHitV12(str(value[0]), fallback_rank, float(value[1]))
        elif len(value) >= 3:
            hit = RescueParentHitV12(str(value[0]), int(value[1]), float(value[2]))
        else:
            return None
    else:
        return None
    siren = _valid_siren(hit.siren)
    if not siren:
        return None
    return replace(hit, siren=siren, rank=max(1, int(hit.rank)))


def _soft_round_robin(
    references: Iterable[_Reference],
    *,
    parent_order: Mapping[str, tuple[int, float]],
    query: RetrievalQuery,
    cap: int,
    protected_lanes: frozenset[str] = frozenset(),
) -> list[_Reference]:
    values = list(references)
    for reference in values:
        rank, score = parent_order.get(reference.siren, (2_147_483_647, 0.0))
        reference.parent_rank_hint = rank if rank < 2_147_483_647 else None
        reference.parent_score_hint = max(reference.parent_score_hint, score)
        reference.site_score = _site_score(reference, reference.parent_score_hint, query)

    def site_key(reference: _Reference) -> tuple[Any, ...]:
        return (
            -reference.site_score,
            reference.input_rank,
            reference.siret,
        )

    selected: list[_Reference] = []
    seen: set[str] = set()
    protected = sorted(
        (reference for reference in values if reference.lanes & protected_lanes),
        key=site_key,
    )
    for reference in protected:
        if len(selected) >= cap:
            return selected
        if reference.siret not in seen:
            selected.append(reference)
            seen.add(reference.siret)

    queues: dict[str, list[_Reference]] = {}
    for reference in values:
        if reference.siret not in seen:
            queues.setdefault(reference.siren, []).append(reference)
    for queue in queues.values():
        queue.sort(key=site_key)
    parents = sorted(
        queues,
        key=lambda siren: (
            parent_order.get(siren, (2_147_483_647, 0.0))[0],
            -parent_order.get(siren, (2_147_483_647, 0.0))[1],
            site_key(queues[siren][0]),
            siren,
        ),
    )
    positions = {siren: 0 for siren in parents}
    while len(selected) < cap:
        progressed = False
        for siren in parents:
            position = positions[siren]
            if position >= len(queues[siren]):
                continue
            progressed = True
            reference = queues[siren][position]
            positions[siren] += 1
            selected.append(reference)
            seen.add(reference.siret)
            if len(selected) >= cap:
                break
        if not progressed:
            break
    return selected


_LANE_PRIORITY = (
    "small_insee_census",
    "joint_official_evidence",
    "targeted_parent_expansion",
    "conditional_siren_rescue",
    "official_relation_1hop",
    "v11_lightweight",
)


def _primary_lane(reference: _Reference) -> str:
    for lane in _LANE_PRIORITY:
        if lane in reference.lanes:
            return lane
    return min(reference.lanes, default="fair_fill")


def _parent_reference(parent: _Parent, rank: int) -> ParentReferenceV12:
    return ParentReferenceV12(
        siren=parent.siren,
        parent_rank=rank,
        parent_score=parent.score,
        support_sites=len(parent.references),
        channels=tuple(sorted(parent.channels)),
        sources=tuple(sorted(parent.sources)),
        reasons=tuple(
            filter(
                None,
                (
                    "JOINT_PARENT_EVIDENCE" if parent.strong else "",
                    "CONDITIONAL_RESCUE_PARENT" if parent.rescue else "",
                    "GEOGRAPHIC_PARENT_SUPPORT" if parent.geographic_support else "",
                ),
            )
        ),
        name_evidence=parent.name_evidence,
        address_evidence=parent.address_evidence,
        exact_name_evidence=parent.exact_name_evidence,
        exact_address_evidence=parent.exact_address_evidence,
        geographic_support=parent.geographic_support,
        rescue=parent.rescue,
        feature_values=tuple(
            float(parent.features.get(name, 0.0))
            for name in CASCADE_PARENT_FEATURES
        ),
    )


class RetrievalTop100CascadeV12:
    """Deterministic SIREN-first cascade operating on natural v11 refs."""

    def __init__(
        self,
        sidecars: SidecarReaderV12,
        config: RetrievalTop100CascadeV12Config | None = None,
        *,
        rescue_provider: ParentRescueProvider | None = None,
        parent_scorer: (
            ParentScorerV12
            | Callable[[Mapping[str, float]], float]
            | Mapping[str, float]
            | None
        ) = None,
    ) -> None:
        self.sidecars = sidecars
        self.config = config or RetrievalTop100CascadeV12Config()
        self.rescue_provider = rescue_provider
        self.parent_scorer = parent_scorer

    def _score_parent(self, parent: _Parent, query: RetrievalQuery) -> float:
        features = _parent_features(parent, query, self.config.rrf_constant)
        parent.features = dict(features)
        scorer = self.parent_scorer
        if scorer is None:
            return _default_parent_score(features)
        if isinstance(scorer, Mapping):
            # Weight mappings are convenient for frozen linear/OOF exports.
            # Unknown weights are rejected so a misspelled feature cannot be
            # silently treated as scientific evidence.
            unknown = set(scorer).difference(features)
            if unknown:
                raise ValueError(
                    f"v12 parent scorer has unknown features: {sorted(unknown)}"
                )
            return float(
                sum(float(scorer.get(name, 0.0)) * value for name, value in features.items())
            )
        method = getattr(scorer, "score", None)
        value = method(features) if callable(method) else scorer(features)
        return float(value)

    @staticmethod
    def _merge(references: dict[str, _Reference], incoming: _Reference) -> None:
        existing = references.get(incoming.siret)
        if existing is None:
            references[incoming.siret] = incoming
        else:
            _merge_reference(existing, incoming)

    def _small_commune_census(
        self, query: RetrievalQuery, references: dict[str, _Reference]
    ) -> tuple[int, int, str]:
        if not query.insee:
            return 0, 0, "NO_INSEE"
        method = getattr(self.sidecars, "all_sites_by_insee_if_fits", None)
        if callable(method):
            result = method(query.insee, max_sites=self.config.small_insee_threshold)
        else:
            fallback = getattr(self.sidecars, "geo_candidates", None)
            if not callable(fallback):
                return 0, 0, "UNAVAILABLE"
            result = fallback(
                insee=query.insee,
                postcode=query.postcode,
                street_number=query.number,
                street=_street_query(query),
                max_sites=self.config.small_insee_threshold,
            )
        if result.strategy != "INSEE_ALL" or result.matched_count > self.config.small_insee_threshold:
            return 0, int(result.matched_count), str(result.strategy)
        for rank, site in enumerate(result.sites, start=1):
            self._merge(
                references,
                _site_reference(
                    site,
                    signal=CascadeSignalV12(
                        source="siren_dossier_v3:geo_sidecar",
                        channel="geo_insee_census",
                        rank=rank,
                        score=1.0 / rank,
                        parent_identifier=site.siren,
                        site_rank=rank,
                    ),
                    lane="small_insee_census",
                    reason="COMPLETE_OFFICIAL_INSEE_CENSUS_LE_100",
                ),
            )
        return len(result.sites), int(result.matched_count), str(result.strategy)

    def _build_parents(
        self, references: Mapping[str, _Reference], query: RetrievalQuery
    ) -> dict[str, _Parent]:
        parents: dict[str, _Parent] = {}
        for reference in references.values():
            parent = parents.setdefault(reference.siren, _Parent(reference.siren))
            parent.references.append(reference)
            parent.channels.update(reference.channels)
            parent.sources.update(reference.sources | reference.evidence_sources)
        for parent in parents.values():
            parent.score = self._score_parent(parent, query)
        return parents

    def _rescue_required(self, parents: Mapping[str, _Parent]) -> bool:
        name_parents = sum(parent.name_evidence for parent in parents.values())
        strong_parents = sum(parent.strong for parent in parents.values())
        return (
            name_parents < self.config.rescue_min_name_parents
            or strong_parents < self.config.rescue_min_strong_parents
        )

    def _execute_rescue(
        self,
        query: RetrievalQuery,
        parents: dict[str, _Parent],
        *,
        query_id: str,
    ) -> int:
        if self.rescue_provider is None:
            return 0
        method = getattr(self.rescue_provider, "retrieve")
        try:
            raw_hits = method(
                query,
                limit=self.config.rescue_parent_budget,
                query_id=query_id,
            )
        except TypeError:
            raw_hits = method(query, limit=self.config.rescue_parent_budget)
        added = 0
        seen: set[str] = set()
        for fallback_rank, raw in enumerate(raw_hits, start=1):
            hit = _normalise_rescue_hit(raw, fallback_rank)
            if hit is None or hit.siren in seen:
                continue
            seen.add(hit.siren)
            parent = parents.setdefault(hit.siren, _Parent(hit.siren))
            parent.rescue = True
            parent.rescue_rank = hit.rank
            parent.rescue_score = max(parent.rescue_score, hit.score)
            parent.channels.add("v12_conditional_siren_rescue")
            parent.sources.add(hit.source)
            parent.score = self._score_parent(parent, query)
            added += 1
            if added >= self.config.rescue_parent_budget:
                break
        return added

    def _select_parents(self, parents: Mapping[str, _Parent]) -> list[_Parent]:
        key = lambda parent: (-parent.score, min((r.input_rank for r in parent.references), default=2_147_483_647), parent.siren)
        normal = sorted((parent for parent in parents.values() if not parent.rescue), key=key)
        rescue = sorted(
            (parent for parent in parents.values() if parent.rescue),
            key=lambda parent: (
                parent.rescue_rank or 2_147_483_647,
                -parent.rescue_score,
                *key(parent),
            ),
        )
        reserve = min(self.config.rescue_parent_reserve, len(rescue))
        selected = normal[: max(0, self.config.parent_shortlist_cap - reserve)]
        selected.extend(rescue[:reserve])
        selected_ids = {parent.siren for parent in selected}
        remainder = sorted(
            (parent for parent in (*normal, *rescue) if parent.siren not in selected_ids),
            key=key,
        )
        selected.extend(remainder[: self.config.parent_shortlist_cap - len(selected)])
        return selected[: self.config.parent_shortlist_cap]

    def _bounded_parent_candidates(
        self, parents: Mapping[str, _Parent]
    ) -> list[_Parent]:
        """Keep every natural parent unless the absolute master cap is hit."""

        key = lambda parent: (
            -parent.score,
            min(
                (reference.input_rank for reference in parent.references),
                default=2_147_483_647,
            ),
            parent.siren,
        )
        values = list(parents.values())
        if len(values) <= self.config.master_reference_cap:
            return sorted(values, key=key)
        rescue = sorted((parent for parent in values if parent.rescue), key=key)
        normal = sorted((parent for parent in values if not parent.rescue), key=key)
        reserve = min(self.config.rescue_parent_reserve, len(rescue))
        kept = normal[: self.config.master_reference_cap - reserve]
        kept.extend(rescue[:reserve])
        kept_ids = {parent.siren for parent in kept}
        remainder = sorted(
            (parent for parent in values if parent.siren not in kept_ids), key=key
        )
        kept.extend(remainder[: self.config.master_reference_cap - len(kept)])
        return sorted(kept, key=key)

    def _expand_parents(
        self,
        query: RetrievalQuery,
        selected_parents: Sequence[_Parent],
        references: dict[str, _Reference],
    ) -> int:
        if not selected_parents or (not query.insee and not query.postcode):
            return 0
        sirens = [parent.siren for parent in selected_parents]
        batch = getattr(self.sidecars, "sites_by_sirens_geo", None)
        if callable(batch):
            results = batch(
                sirens=sirens,
                insee=query.insee,
                postcode=query.postcode,
                address=query.address,
                street_number=query.number,
                street=_street_query(query),
                include_active=True,
                include_closed=True,
                max_sites_per_siren=self.config.max_sites_per_parent,
            )
        else:
            results = {
                siren: self.sidecars.sites_by_siren_geo(
                    siren=siren,
                    insee=query.insee,
                    postcode=query.postcode,
                    address=query.address,
                    street_number=query.number,
                    street=_street_query(query),
                    include_active=True,
                    include_closed=True,
                    max_sites=self.config.max_sites_per_parent,
                )
                for siren in sirens
            }
        queues = [tuple(results.get(parent.siren, SiteQueryResult((), "EMPTY", 0, 1, False)).sites) for parent in selected_parents]
        positions = [0] * len(queues)
        emitted = 0
        while emitted < self.config.parent_expansion_budget:
            progressed = False
            for parent_rank, (parent, sites) in enumerate(zip(selected_parents, queues), start=1):
                position = positions[parent_rank - 1]
                if position >= len(sites):
                    continue
                progressed = True
                positions[parent_rank - 1] += 1
                emitted += 1
                site = sites[position]
                lane = "conditional_siren_rescue" if parent.rescue else "targeted_parent_expansion"
                channel = "v12_rescue_local_expansion" if parent.rescue else "v12_parent_local_expansion"
                incoming = _site_reference(
                    site,
                    signal=CascadeSignalV12(
                        source="v12_parent:siren->official_local_site",
                        channel=channel,
                        rank=emitted,
                        score=1.0 / parent_rank,
                        parent_identifier=parent.siren,
                        parent_rank=parent_rank,
                        site_rank=position + 1,
                    ),
                    lane=lane,
                    reason=(
                        "CONDITIONAL_RESCUE_PARENT_LOCAL_EXPANSION"
                        if parent.rescue
                        else "TOP_PARENT_LOCAL_EXPANSION"
                    ),
                )
                incoming.parent_score_hint = parent.score
                self._merge(references, incoming)
                if emitted >= self.config.parent_expansion_budget:
                    break
            if not progressed:
                break
        return emitted

    def cascade(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        query: RetrievalQuery | Mapping[str, Any] | None = None,
        query_id: str = "",
    ) -> RetrievalTop100CascadeV12Result:
        if not rows:
            raise ValueError("v12 cascade requires at least one natural v11 reference")
        upstream_rescue_channels = {
            signal.channel
            for raw in rows
            for signal in _signals_from_row(_clean_row(raw))
            if signal.channel in DEFERRED_RESCUE_CHANNELS
        }
        if upstream_rescue_channels:
            raise ValueError(
                "initial v12 official stack refuses deferred upstream rescue "
                f"channels: {sorted(upstream_rescue_channels)}"
            )
        safe_first = _clean_row(rows[0])
        normalized_query = (
            query
            if isinstance(query, RetrievalQuery)
            else RetrievalQuery.from_mapping(_clean_row(query or safe_first))
        )
        allowed_relation_parents = _relation_seed_parents(
            rows, normalized_query, self.config.relation_seed_limit
        )
        references: dict[str, _Reference] = {}
        relation_references = 0
        for raw in rows:
            reference = _reference_from_row(
                raw,
                query=normalized_query,
                allowed_relation_parents=allowed_relation_parents,
            )
            if reference is None:
                continue
            if reference.channels & _RELATION_CHANNELS:
                if relation_references >= self.config.relation_reference_budget:
                    # Keep non-relation evidence on a mixed candidate.
                    reference.signals = [
                        signal
                        for signal in reference.signals
                        if signal.channel not in _RELATION_CHANNELS
                    ]
                    reference.channels.difference_update(_RELATION_CHANNELS)
                    reference.lanes.discard("official_relation_1hop")
                    reference.reasons.discard(
                        "STRUCTURED_OFFICIAL_RELATION_GEO_COMPATIBLE"
                    )
                    if not reference.channels:
                        continue
                else:
                    relation_references += 1
            if _joint_evidence(reference):
                reference.lanes.add("joint_official_evidence")
                reference.reasons.add("JOINT_NAME_AND_GEOGRAPHIC_EVIDENCE")
            self._merge(references, reference)

        small_count, small_pool, small_strategy = self._small_commune_census(
            normalized_query, references
        )
        parents = self._build_parents(references, normalized_query)
        parent_count_before_rescue = len(parents)
        rescue_triggered = self._rescue_required(parents)
        rescue_hits = (
            self._execute_rescue(
                normalized_query, parents, query_id=str(query_id)
            )
            if rescue_triggered
            else 0
        )
        rescue_executed = rescue_hits > 0
        # Every natural parent has already been scored by the injected scorer.
        # Keep this complete rank map separate from the <=64 expansion choice:
        # admission must receive the pass-A score/rank for *every* master and
        # hydration candidate, including parents that were not expanded.
        all_scored_parents = sorted(
            parents.values(),
            key=lambda parent: (
                -parent.score,
                min(
                    (
                        reference.input_rank
                        for reference in parent.references
                    ),
                    default=2_147_483_647,
                ),
                parent.siren,
            ),
        )
        all_parent_order = {
            parent.siren: (rank, parent.score)
            for rank, parent in enumerate(all_scored_parents, start=1)
        }
        all_parent_features = {
            parent.siren: tuple(
                float(parent.features.get(name, 0.0))
                for name in CASCADE_PARENT_FEATURES
            )
            for parent in all_scored_parents
        }
        bounded_parent_candidates = self._bounded_parent_candidates(parents)
        frozen_parent_candidates = tuple(
            _parent_reference(parent, rank)
            for rank, parent in enumerate(bounded_parent_candidates, start=1)
        )
        selected_parents = self._select_parents(
            {parent.siren: parent for parent in bounded_parent_candidates}
        )
        parent_expansions = self._expand_parents(
            normalized_query, selected_parents, references
        )

        # Expansion happens *after* the frozen parent score.  Refresh support
        # fields for reporting, but never rerank parents on evidence created by
        # their own expansion (that circularity would bias large chains).
        parents_after = self._build_parents(references, normalized_query)
        refreshed_parents: list[_Parent] = []
        for parent in selected_parents:
            refreshed = parents_after.get(parent.siren)
            if refreshed is None:
                continue
            refreshed.rescue = parent.rescue
            refreshed.rescue_rank = parent.rescue_rank
            refreshed.rescue_score = parent.rescue_score
            refreshed.channels.update(parent.channels)
            refreshed.sources.update(parent.sources)
            refreshed.score = parent.score
            refreshed.features = dict(parent.features)
            refreshed_parents.append(refreshed)
        selected_parents = refreshed_parents
        parent_order = all_parent_order

        master = _soft_round_robin(
            references.values(),
            parent_order=parent_order,
            query=normalized_query,
            cap=self.config.master_reference_cap,
            protected_lanes=frozenset({"small_insee_census"}),
        )
        master_ids = {reference.siret for reference in master}
        shortlist_pool = [
            reference
            for reference in master
            if reference.siren in parent_order
            or "small_insee_census" in reference.lanes
            or "official_relation_1hop" in reference.lanes
        ]
        shortlist = _soft_round_robin(
            shortlist_pool,
            parent_order=parent_order,
            query=normalized_query,
            cap=self.config.hydration_shortlist_cap,
            protected_lanes=frozenset({"small_insee_census"}),
        )
        if len(shortlist) < self.config.hydration_shortlist_cap:
            remainder = [
                reference
                for reference in master
                if reference.siret not in {item.siret for item in shortlist}
            ]
            fill = _soft_round_robin(
                remainder,
                parent_order=parent_order,
                query=normalized_query,
                cap=self.config.hydration_shortlist_cap - len(shortlist),
            )
            shortlist.extend(fill)
        shortlist_ids = {reference.siret for reference in shortlist}
        hydration_ranks = {
            reference.siret: rank for rank, reference in enumerate(shortlist, start=1)
        }

        site_ranks: dict[str, int] = {}
        by_parent: dict[str, list[_Reference]] = {}
        for reference in master:
            by_parent.setdefault(reference.siren, []).append(reference)
        for values in by_parent.values():
            for rank, reference in enumerate(
                sorted(values, key=lambda item: (-item.site_score, item.siret)),
                start=1,
            ):
                site_ranks[reference.siret] = rank

        candidates: list[CascadeCandidateV12] = []
        for master_rank, reference in enumerate(master, start=1):
            rank_score = parent_order.get(reference.siren)
            candidates.append(
                CascadeCandidateV12(
                    siret=reference.siret,
                    siren=reference.siren,
                    master_rank=master_rank,
                    hydration_rank=hydration_ranks.get(reference.siret),
                    parent_rank=rank_score[0] if rank_score else None,
                    site_rank=site_ranks[reference.siret],
                    parent_score=rank_score[1] if rank_score else reference.parent_score_hint,
                    parent_feature_names=tuple(CASCADE_PARENT_FEATURES),
                    parent_feature_values=all_parent_features.get(
                        reference.siren,
                        tuple(0.0 for _ in CASCADE_PARENT_FEATURES),
                    ),
                    site_score=reference.site_score,
                    selection_lane=_primary_lane(reference),
                    lanes=tuple(sorted(reference.lanes)),
                    reasons=tuple(sorted(reference.reasons)),
                    channels=tuple(sorted(reference.channels)),
                    sources=tuple(sorted(reference.sources)),
                    signals=tuple(
                        sorted(
                            reference.signals,
                            key=lambda signal: (
                                signal.rank,
                                signal.channel,
                                signal.source,
                            ),
                        )
                    ),
                    official_evidence_ids=tuple(sorted(reference.evidence_ids)),
                    official_evidence_sources=tuple(
                        sorted(reference.evidence_sources)
                    ),
                    candidate_names=reference.names,
                    candidate_addresses=reference.addresses,
                    candidate_insee=reference.insee,
                    candidate_postcode=reference.postcode,
                    candidate_number=reference.number,
                    candidate_state=reference.state,
                    candidate_is_headquarters=reference.is_headquarters,
                    temporal_valid_from=reference.temporal_valid_from,
                    temporal_valid_to=reference.temporal_valid_to,
                )
            )
        candidate_by_id = {candidate.siret: candidate for candidate in candidates}
        hydration = tuple(candidate_by_id[item.siret] for item in shortlist)

        parent_rows = [
            _parent_reference(parent, parent_order[parent.siren][0])
            for parent in selected_parents
        ]

        if len(candidates) > self.config.master_reference_cap:
            raise AssertionError("v12 master reference cap exceeded")
        if len(hydration) > self.config.hydration_shortlist_cap:
            raise AssertionError("v12 hydration shortlist cap exceeded")
        if len(parent_rows) > self.config.parent_shortlist_cap:
            raise AssertionError("v12 parent shortlist cap exceeded")
        if len(frozen_parent_candidates) > self.config.master_reference_cap:
            raise AssertionError("v12 parent candidate cap exceeded")
        if not shortlist_ids.issubset(master_ids):
            raise AssertionError("v12 hydration shortlist escaped master references")

        return RetrievalTop100CascadeV12Result(
            query=normalized_query,
            master_references=tuple(candidates),
            hydration_shortlist=hydration,
            parent_candidates=frozen_parent_candidates,
            parent_shortlist=tuple(parent_rows),
            rescue_triggered=rescue_triggered,
            rescue_executed=rescue_executed,
            strategy_counts={
                "input_references": len(rows),
                "small_insee_sites": small_count,
                "small_insee_pool": small_pool,
                "small_insee_strategy": small_strategy,
                "structured_relation_references": relation_references,
                "parent_candidates_before_rescue": parent_count_before_rescue,
                "parent_candidates": len(frozen_parent_candidates),
                "rescue_triggered": rescue_triggered,
                "rescue_executed": rescue_executed,
                "rescue_parent_hits": rescue_hits,
                "upstream_rescue_reference_count": 0,
                "parent_shortlist": len(parent_rows),
                "parent_expansion_emissions": parent_expansions,
                "master_references": len(candidates),
                "hydration_shortlist": len(hydration),
                "positive_injection": False,
                "human_labels_opened": False,
                "hard_sites_per_siren_cap": None,
                "relation_hops": 1,
            },
        )


__all__ = [
    "CascadeCandidateV12",
    "CascadeSignalV12",
    "CASCADE_PARENT_FEATURES",
    "ParentReferenceV12",
    "ParentRescueProvider",
    "ParentScorerV12",
    "RescueParentHitV12",
    "RetrievalTop100CascadeV12",
    "RetrievalTop100CascadeV12Config",
    "RetrievalTop100CascadeV12Result",
    "SidecarReaderV12",
    "V12_CASCADE_SCHEMA_VERSION",
    "V12_PARENT_SCHEMA_VERSION",
    "V12_OUTPUT_CHANNELS",
    "XGBoostParentScorerV12",
    "is_forbidden_input_key",
    "production_projection",
]
