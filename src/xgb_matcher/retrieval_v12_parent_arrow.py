"""Exact parent-only Arrow compiler for the retrieval-v12 bootstrap pass.

The normal cascade needs rich, immutable SIRET references because it later
expands and serialises them.  Parent fitting does not: it only consumes the
18 pre-expansion SIREN features, the minimum input rank, and a handful of
scene counters.  Building every rich reference (and first converting an
entire Arrow table to Python dictionaries) dominated the national parent
pass.

This module reproduces the parent sufficient statistics directly from Arrow
columns.  It intentionally returns a :class:`FoldFusedSceneV12` with an empty
``references`` tuple; callers must use it only for parent bootstrap fitting.
The small-commune census and structured one-hop relation rules remain exactly
the same as the canonical compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import pyarrow as pa

from .hierarchical_retrieval import RetrievalQuery
from .retrieval_fold_fused_v12 import (
    ALLOWED_FOLDS_V12,
    FoldFusedSceneV12,
    SceneParentV12,
)
from .retrieval_top100_cascade_v12 import (
    CASCADE_PARENT_FEATURES,
    DEFERRED_RESCUE_CHANNELS,
    RetrievalTop100CascadeV12Config,
    SidecarReaderV12,
    _ADDRESS_EVIDENCE,
    _ADDRESS_EXACT,
    _GEO_CHANNELS,
    _NAME_EVIDENCE,
    _NAME_EXACT,
    _RELATION_CHANNELS,
    _source_families,
    _structured_relation,
    _valid_siren,
    _valid_siret,
    is_forbidden_input_key,
)


# Columns sufficient when the source has an independently sealed proof that
# ``signals`` contains every channel represented by ``channels`` and by every
# non-null ``*_rank`` column.  Query aliases are retained to preserve the
# canonical RetrievalQuery precedence rules on fixtures as well as production.
PARENT_SIGNAL_COMPLETE_PROJECTION_V12 = (
    "query_id",
    "fold",
    "oof_fold",
    "crm_name",
    "crm_name_raw",
    "name",
    "nom",
    "crm_addr",
    "crm_address",
    "crm_address_raw",
    "address",
    "adresse",
    "numeroVoie",
    "number",
    "numero",
    "postcode",
    "crm_cp",
    "crm_postcode",
    "crm_postcode_raw",
    "insee",
    "crm_insee",
    "crm_insee_raw",
    "siret",
    "candidate_siret",
    "siren",
    "candidate_siren",
    "sources",
    "channels",
    "signals",
    "official_evidence_sources",
    "candidate_insee",
    "insee_etablissement",
    "candidate_postcode",
    "postcode_etablissement",
    "union_rank",
    "retrieval_rank",
    "retrieval_source",
)

_RESERVED_RANK_COLUMNS = frozenset(
    {"union_rank", "retrieval_rank", "parent_rank", "site_rank"}
)
_MAX_RANK = 2_147_483_647


def parent_signal_complete_projection_v12(
    available_columns: Sequence[str],
) -> tuple[str, ...]:
    """Return the strongly projected parent-only scanner schema."""

    available = set(str(value) for value in available_columns)
    selected = tuple(
        name for name in PARENT_SIGNAL_COMPLETE_PROJECTION_V12 if name in available
    )
    if "query_id" not in selected or not ({"fold", "oof_fold"} & set(selected)):
        raise ValueError("parent Arrow projection requires query_id and fold")
    if "signals" not in selected or "channels" not in selected:
        raise ValueError("parent Arrow projection requires signals and channels")
    if not ({"siret", "candidate_siret"} & set(selected)):
        raise ValueError("parent Arrow projection requires a SIRET column")
    return selected


def _column(table: pa.Table, name: str, default: Any = None) -> list[Any]:
    if name not in table.column_names:
        return [default] * table.num_rows
    return table[name].combine_chunks().to_pylist()


def _first_column(
    columns: Mapping[str, list[Any]], index: int, *names: str, default: Any = None
) -> Any:
    for name in names:
        values = columns.get(name)
        if values is not None:
            value = values[index]
            # Canonical row readers use ``row.get(a) or row.get(b)`` for all
            # of these fields, so empty strings and numeric zero fall through.
            if value:
                return value
    return default


def _value(
    columns: Mapping[str, list[Any]], name: str, index: int, default: Any = None
) -> Any:
    values = columns.get(name)
    return default if values is None else values[index]


def _as_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    try:
        return tuple(str(item) for item in value if item not in (None, ""))
    except TypeError:
        return (str(value),)


def _geo_compatible(insee: str, postcode: str, query: RetrievalQuery) -> bool:
    if query.insee:
        return bool(insee and insee == query.insee)
    if query.postcode:
        return bool(postcode and postcode == query.postcode)
    return False


@dataclass
class _ParentOnlyReference:
    siret: str
    siren: str
    channels: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    evidence_sources: set[str] = field(default_factory=set)
    channel_ranks: dict[str, int] = field(default_factory=dict)
    insee: str = ""
    postcode: str = ""
    input_rank: int = _MAX_RANK

    def merge(self, incoming: "_ParentOnlyReference") -> None:
        # Exact projection of _merge_reference for fields used by the parent
        # features.  In particular, SIREN and geography remain first-nonempty.
        self.channels.update(incoming.channels)
        self.sources.update(incoming.sources)
        self.evidence_sources.update(incoming.evidence_sources)
        for channel, rank in incoming.channel_ranks.items():
            self.channel_ranks[channel] = min(
                self.channel_ranks.get(channel, rank), rank
            )
        self.insee = self.insee or incoming.insee
        self.postcode = self.postcode or incoming.postcode
        self.input_rank = min(self.input_rank, incoming.input_rank)


@dataclass(frozen=True, slots=True)
class _ParentSignal:
    """Only the four signal fields consumed by parent aggregation."""

    source: str
    channel: str
    rank: int
    parent_identifier: str = ""


def _signals_for_row(
    index: int,
    *,
    columns: Mapping[str, list[Any]],
    dynamic_rank_columns: Sequence[str],
    signals_complete: bool,
) -> list[_ParentSignal]:
    raw_signals = _value(columns, "signals", index) or ()
    output = [
        _ParentSignal(
            source=str(value.get("source") or ""),
            channel=str(value.get("channel") or ""),
            rank=max(1, int(value.get("rank") or _MAX_RANK)),
            parent_identifier=str(value.get("parent_identifier") or ""),
        )
        for value in raw_signals
        if isinstance(value, Mapping)
    ]
    known_channels = set(_as_strings(_value(columns, "channels", index)))
    if not signals_complete:
        for name in dynamic_rank_columns:
            value = columns[name][index]
            if value not in (None, ""):
                known_channels.add(name[:-5])
    existing = {signal.channel for signal in output}
    missing = sorted(known_channels - existing)
    if signals_complete and missing:
        raise ValueError(
            "sealed parent signal-complete projection contradicted by scene: "
            f"{missing[:8]}"
        )
    union_rank = _first_column(
        columns, index, "union_rank", "retrieval_rank", default=_MAX_RANK
    )
    retrieval_source = _first_column(
        columns,
        index,
        "retrieval_source",
        default="v11_lightweight_union",
    )
    for channel in missing:
        rank_values = columns.get(f"{channel}_rank")
        rank_value = rank_values[index] if rank_values is not None else None
        output.append(
            _ParentSignal(
                source=str(retrieval_source or "v11_lightweight_union"),
                channel=channel,
                rank=max(1, int(rank_value or union_rank or _MAX_RANK)),
            )
        )
    return output


def _query_from_columns(columns: Mapping[str, list[Any]]) -> RetrievalQuery:
    names = (
        "crm_name",
        "crm_name_raw",
        "name",
        "nom",
        "crm_addr",
        "crm_address",
        "crm_address_raw",
        "address",
        "adresse",
        "numeroVoie",
        "number",
        "numero",
        "postcode",
        "crm_cp",
        "crm_postcode",
        "crm_postcode_raw",
        "insee",
        "crm_insee",
        "crm_insee_raw",
    )
    return RetrievalQuery.from_mapping(
        {name: columns[name][0] for name in names if name in columns}
    )


def _make_reference(
    index: int,
    *,
    columns: Mapping[str, list[Any]],
    signals: Sequence[_ParentSignal],
    query: RetrievalQuery,
    allowed_relation_parents: set[str],
) -> _ParentOnlyReference | None:
    siret = _valid_siret(
        _first_column(columns, index, "siret", "candidate_siret", default="")
    )
    if not siret:
        return None
    siren = _valid_siren(
        _first_column(columns, index, "siren", "candidate_siren", default="")
    ) or siret[:9]
    insee = str(
        _first_column(
            columns,
            index,
            "candidate_insee",
            "insee_etablissement",
            default="",
        )
        or ""
    )
    postcode = str(
        _first_column(
            columns,
            index,
            "candidate_postcode",
            "postcode_etablissement",
            default="",
        )
        or ""
    )
    input_rank = max(
        1,
        int(
            _first_column(
                columns,
                index,
                "union_rank",
                "retrieval_rank",
                default=_MAX_RANK,
            )
            or _MAX_RANK
        ),
    )
    reference = _ParentOnlyReference(
        siret=siret,
        siren=siren,
        insee=insee,
        postcode=postcode,
        input_rank=input_rank,
    )
    relation_seen = False
    non_relation_seen = False
    compatible = _geo_compatible(insee, postcode, query)
    for signal in signals:
        if signal.channel in _RELATION_CHANNELS:
            relation_seen = True
            if (
                not _structured_relation(signal)
                or signal.parent_identifier not in allowed_relation_parents
                or not compatible
            ):
                continue
        else:
            non_relation_seen = True
        if not signal.channel:
            continue
        reference.channels.add(signal.channel)
        if signal.source:
            reference.sources.add(signal.source)
        reference.channel_ranks[signal.channel] = min(
            reference.channel_ranks.get(signal.channel, signal.rank), signal.rank
        )
    if relation_seen and not non_relation_seen and not (
        reference.channels & _RELATION_CHANNELS
    ):
        return None
    reference.sources.update(_as_strings(_value(columns, "sources", index)))
    reference.evidence_sources.update(
        _as_strings(_value(columns, "official_evidence_sources", index))
    )
    return reference


def _parent_feature_values(
    references: Sequence[_ParentOnlyReference],
    *,
    query: RetrievalQuery,
    rrf_constant: float,
    source_family_cache: dict[tuple[str, ...], frozenset[str]],
) -> tuple[float, ...]:
    parent_channels: set[str] = set()
    parent_sources: set[str] = set()
    ranks: dict[str, int] = {}
    geographic = 0
    joint = 0
    for reference in references:
        parent_channels.update(reference.channels)
        parent_sources.update(reference.sources)
        parent_sources.update(reference.evidence_sources)
        geographic += int(
            _geo_compatible(reference.insee, reference.postcode, query)
        )
        joint += int(
            bool(reference.channels & _NAME_EVIDENCE)
            and bool(reference.channels & _ADDRESS_EVIDENCE)
        )
        for channel, rank in reference.channel_ranks.items():
            ranks[channel] = min(ranks.get(channel, rank), rank)
    source_key = tuple(sorted(parent_sources))
    source_families = source_family_cache.get(source_key)
    if source_families is None:
        source_families = frozenset(_source_families(parent_sources))
        source_family_cache[source_key] = source_families
    exact_name = bool(parent_channels & _NAME_EXACT)
    name = bool(parent_channels & _NAME_EVIDENCE)
    exact_address = bool(parent_channels & _ADDRESS_EXACT)
    address = bool(parent_channels & _ADDRESS_EVIDENCE)
    values = {
        "exact_name_and_address": float(exact_name and address),
        "name_and_exact_address": float(name and exact_address),
        "exact_name_evidence": float(exact_name),
        "name_evidence": float(name),
        "exact_address_evidence": float(exact_address),
        "address_evidence": float(address),
        "geographic_support": float(geographic > 0),
        "joint_site_log1p": float(math.log1p(joint)),
        "official_source_count": float(len(source_families)),
        "parent_source_registry_present": float("SIRENE" in source_families),
        "parent_source_rne_present": float("RNE" in source_families),
        "parent_source_bodacc_present": float("BODACC" in source_families),
        "parent_source_sector_present": float("SECTOR" in source_families),
        "channel_count": float(len(ranks)),
        "support_site_log1p": float(math.log1p(len(references))),
        "channel_rrf": float(
            sum(
                1.0 / (rrf_constant + max(1, rank))
                for rank in ranks.values()
            )
        ),
        "rescue_evidence": 0.0,
        "rescue_score": 0.0,
    }
    return tuple(float(values[name]) for name in CASCADE_PARENT_FEATURES)


def compile_parent_arrow_scene_v12(
    table: pa.Table,
    *,
    query_id: str,
    sidecars: SidecarReaderV12,
    config: RetrievalTop100CascadeV12Config,
    signals_complete: bool = False,
) -> FoldFusedSceneV12:
    """Compile exact pre-expansion parent features without rich references.

    ``signals_complete=True`` may be used only after validating a capability
    manifest bound to the exact source bytes.  In that mode omitted rank/score
    columns are never consulted.  Without it, all rank columns present in the
    table participate exactly like ``_signals_from_row``.
    """

    if not isinstance(table, pa.Table) or table.num_rows < 1:
        raise ValueError("parent Arrow compiler requires a non-empty table")
    if not query_id:
        raise ValueError("parent Arrow compiler requires a query_id")
    forbidden = [name for name in table.column_names if is_forbidden_input_key(name)]
    if forbidden:
        raise ValueError(f"parent Arrow compiler refuses truth columns: {forbidden}")

    names = set(table.column_names)
    fold_name = "fold" if "fold" in names else "oof_fold"
    if fold_name not in names or "query_id" not in names:
        raise ValueError("parent Arrow compiler requires query_id and fold")
    folds = {int(value) for value in _column(table, fold_name)}
    query_ids = {str(value or "") for value in _column(table, "query_id")}
    if (
        len(folds) != 1
        or not folds.issubset(ALLOWED_FOLDS_V12)
        or query_ids != {str(query_id)}
    ):
        raise ValueError("parent Arrow scene mixes query identifiers or folds")

    dynamic_rank_columns = tuple(
        name
        for name in table.column_names
        if name.endswith("_rank") and name not in _RESERVED_RANK_COLUMNS
    )
    required = {
        "query_id",
        fold_name,
        "signals",
        "channels",
        "sources",
        "official_evidence_sources",
        "siret",
        "candidate_siret",
        "siren",
        "candidate_siren",
        "candidate_insee",
        "insee_etablissement",
        "candidate_postcode",
        "postcode_etablissement",
        "union_rank",
        "retrieval_rank",
        "retrieval_source",
        *dynamic_rank_columns,
        *PARENT_SIGNAL_COMPLETE_PROJECTION_V12,
    }
    columns = {
        name: _column(table, name)
        for name in required
        if name in table.column_names
    }
    query = _query_from_columns(columns)
    signals_by_row = [
        _signals_for_row(
            index,
            columns=columns,
            dynamic_rank_columns=dynamic_rank_columns,
            signals_complete=signals_complete,
        )
        for index in range(table.num_rows)
    ]
    deferred = {
        signal.channel
        for signals in signals_by_row
        for signal in signals
        if signal.channel in DEFERRED_RESCUE_CHANNELS
    }
    if deferred:
        raise ValueError(
            "initial v12 fold-fused stack refuses deferred upstream rescue "
            f"channels: {sorted(deferred)}"
        )

    relation_ranked: list[tuple[int, str, str]] = []
    for index, signals in enumerate(signals_by_row):
        siret = _valid_siret(
            _first_column(columns, index, "siret", "candidate_siret", default="")
        )
        if not siret:
            continue
        insee = str(
            _first_column(
                columns,
                index,
                "candidate_insee",
                "insee_etablissement",
                default="",
            )
            or ""
        )
        postcode = str(
            _first_column(
                columns,
                index,
                "candidate_postcode",
                "postcode_etablissement",
                default="",
            )
            or ""
        )
        if not _geo_compatible(insee, postcode, query):
            continue
        for signal in signals:
            if _structured_relation(signal):
                relation_ranked.append(
                    (signal.rank, signal.parent_identifier, siret)
                )
    allowed_relation_parents: list[str] = []
    for _rank, parent, _siret in sorted(relation_ranked):
        if parent not in allowed_relation_parents:
            allowed_relation_parents.append(parent)
        if len(allowed_relation_parents) >= config.relation_seed_limit:
            break
    allowed_relation_set = set(allowed_relation_parents)

    references: dict[str, _ParentOnlyReference] = {}
    relation_references = 0
    for index, signals in enumerate(signals_by_row):
        reference = _make_reference(
            index,
            columns=columns,
            signals=signals,
            query=query,
            allowed_relation_parents=allowed_relation_set,
        )
        if reference is None:
            continue
        relation_channels = reference.channels & _RELATION_CHANNELS
        if relation_channels:
            if relation_references >= config.relation_reference_budget:
                # Canonical code removes signals/channels, but deliberately
                # leaves channel_ranks and sources in place.  Preserve that
                # slightly surprising historical behavior exactly.
                reference.channels.difference_update(relation_channels)
                if not reference.channels:
                    continue
            else:
                relation_references += 1
        existing = references.get(reference.siret)
        if existing is None:
            references[reference.siret] = reference
        else:
            existing.merge(reference)

    small_count = 0
    small_pool = 0
    small_strategy = "NO_INSEE"
    if query.insee:
        method = getattr(sidecars, "all_sites_by_insee_if_fits", None)
        if callable(method):
            result = method(query.insee, max_sites=config.small_insee_threshold)
            small_pool = int(result.matched_count)
            small_strategy = str(result.strategy)
            if (
                result.strategy == "INSEE_ALL"
                and result.matched_count <= config.small_insee_threshold
            ):
                small_count = len(result.sites)
                for rank, site in enumerate(result.sites, start=1):
                    incoming = _ParentOnlyReference(
                        siret=site.siret,
                        siren=site.siren,
                        channels={"geo_insee_census"},
                        sources={"siren_dossier_v3:geo_sidecar"},
                        channel_ranks={"geo_insee_census": rank},
                        insee=site.insee,
                        postcode=site.postcode,
                        input_rank=rank,
                    )
                    existing = references.get(incoming.siret)
                    if existing is None:
                        references[incoming.siret] = incoming
                    else:
                        existing.merge(incoming)
        else:
            # Parent training uses the official v11 reader, which always
            # exposes the method above.  Failing closed avoids silently
            # changing the legacy geo-census semantics in a custom caller.
            raise ValueError("parent Arrow compiler requires INSEE census support")
    if not references:
        raise ValueError("parent Arrow compiler produced no references")

    by_parent: dict[str, list[_ParentOnlyReference]] = {}
    for reference in references.values():
        by_parent.setdefault(reference.siren, []).append(reference)
    source_family_cache: dict[tuple[str, ...], frozenset[str]] = {}
    parents = tuple(
        SceneParentV12(
            siren=siren,
            feature_values=_parent_feature_values(
                current,
                query=query,
                rrf_constant=config.rrf_constant,
                source_family_cache=source_family_cache,
            ),
            minimum_input_rank=min(value.input_rank for value in current),
        )
        for siren, current in sorted(by_parent.items())
    )
    parent_channels = [
        set().union(*(value.channels for value in current))
        for current in by_parent.values()
    ]
    return FoldFusedSceneV12(
        query_id=str(query_id),
        query_fold=next(iter(folds)),
        query=query,
        references=(),
        parents=parents,
        input_reference_count=table.num_rows,
        parent_count_before_rescue=len(parents),
        rescue_triggered=(
            sum(bool(channels & _NAME_EVIDENCE) for channels in parent_channels)
            < config.rescue_min_name_parents
            or sum(
                bool(channels & _NAME_EXACT)
                or (
                    bool(channels & _NAME_EVIDENCE)
                    and bool(channels & _ADDRESS_EVIDENCE)
                )
                for channels in parent_channels
            )
            < config.rescue_min_strong_parents
        ),
        small_insee_sites=small_count,
        small_insee_pool=small_pool,
        small_insee_strategy=small_strategy,
        structured_relation_references=relation_references,
    )


__all__ = [
    "PARENT_SIGNAL_COMPLETE_PROJECTION_V12",
    "compile_parent_arrow_scene_v12",
    "parent_signal_complete_projection_v12",
]
