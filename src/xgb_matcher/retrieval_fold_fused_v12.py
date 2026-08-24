"""Fold-fused, leakage-safe parent/context kernel for retrieval v12.

The historical v12 runner executes the same label-blind scene three times,
once for every outer OOF context.  This module factors the common work without
changing the logical cascade:

* input references, relation filtering and the small-commune census are built
  once into an immutable :class:`FoldFusedSceneV12`;
* the pre-expansion parent feature matrix is scored in three *separate* OOF
  contexts, but every model receives one float32 matrix rather than thousands
  of one-row ``DMatrix`` objects;
* the union of the three context-specific parent shortlists is looked up once
  in the official sidecar, then filtered back through immutable context masks;
* each context replays the original expansion, round-robin and caps exactly.

No human label is accepted by the scene compiler.  SIREN/SIRET remain grouping
and output identifiers and are never present in the model matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .hierarchical_retrieval import RetrievalQuery
from .retrieval_sidecars_v11 import LocalSite, SiteQueryResult
from .retrieval_top100_cascade_v12 import (
    CASCADE_PARENT_FEATURES,
    DEFERRED_RESCUE_CHANNELS,
    CascadeCandidateV12,
    CascadeSignalV12,
    ParentReferenceV12,
    RetrievalTop100CascadeV12Config,
    RetrievalTop100CascadeV12Result,
    SidecarReaderV12,
    _Parent,
    _Reference,
    _clean_row,
    _joint_evidence,
    _merge_reference,
    _parent_features,
    _parent_reference,
    _primary_lane,
    _reference_from_row,
    _relation_seed_parents,
    _signals_from_row,
    _site_reference,
    _soft_round_robin,
    _street_query,
    is_forbidden_input_key,
)


ALLOWED_FOLDS_V12 = frozenset({2, 3, 4})
PARENT_MODEL_KEYS_V12 = frozenset(
    {"fit_2", "fit_3", "fit_4", "fit_23", "fit_24", "fit_34"}
)

# This is the exact mapping sealed by ``_OUTER_PARENT_MAPS`` in the admission
# runner.  MappingProxyType prevents an accidental runtime rewrite.
PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12: Mapping[int, Mapping[int, str]] = (
    MappingProxyType(
        {
            2: MappingProxyType({2: "fit_34", 3: "fit_4", 4: "fit_3"}),
            3: MappingProxyType({2: "fit_4", 3: "fit_24", 4: "fit_2"}),
            4: MappingProxyType({2: "fit_3", 3: "fit_2", 4: "fit_23"}),
        }
    )
)


class ParentMatrixScorerV12(Protocol):
    """Pointwise parent scorer over the frozen 18-column float32 matrix."""

    def score_matrix(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class XGBoostParentMatrixScorerV12:
    """Batch adapter preserving the old Booster/DMatrix prediction semantics."""

    model: Any

    @classmethod
    def from_model_path(cls, path: Path | str) -> "XGBoostParentMatrixScorerV12":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        return cls(booster)

    def score_matrix(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32, order="C")
        if matrix.ndim != 2 or matrix.shape[1] != len(CASCADE_PARENT_FEATURES):
            raise ValueError("v12 parent matrix feature schema/order drift")
        if self.model.__class__.__name__ == "Booster":
            import xgboost as xgb

            prediction = self.model.predict(
                xgb.DMatrix(
                    matrix,
                    feature_names=list(CASCADE_PARENT_FEATURES),
                )
            )
        else:
            prediction = self.model.predict(matrix)
        output = np.asarray(prediction).reshape(-1)
        if len(output) != len(matrix) or not np.isfinite(output).all():
            raise ValueError("v12 parent batch scorer returned invalid predictions")
        return output


@dataclass(frozen=True)
class LinearParentMatrixScorerV12:
    """Deterministic vectorized scorer used by parity fixtures and smokes."""

    weights: tuple[float, ...]
    intercept: float = 0.0

    def __post_init__(self) -> None:
        if len(self.weights) != len(CASCADE_PARENT_FEATURES):
            raise ValueError("linear v12 parent scorer feature schema/order drift")

    def score_matrix(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32, order="C")
        if matrix.ndim != 2 or matrix.shape[1] != len(self.weights):
            raise ValueError("linear v12 parent matrix has the wrong shape")
        weights = np.asarray(self.weights, dtype=np.float32)
        return matrix @ weights + np.float32(self.intercept)


@dataclass(frozen=True)
class FrozenReferenceV12:
    """Deeply immutable representation of the mutable legacy reference."""

    siret: str
    siren: str
    channels: tuple[str, ...]
    sources: tuple[str, ...]
    signals: tuple[CascadeSignalV12, ...]
    evidence_ids: tuple[str, ...]
    evidence_sources: tuple[str, ...]
    lanes: tuple[str, ...]
    reasons: tuple[str, ...]
    channel_ranks: tuple[tuple[str, int], ...]
    names: tuple[str, ...]
    addresses: tuple[str, ...]
    insee: str
    postcode: str
    number: str
    state: str
    is_headquarters: bool
    temporal_valid_from: str
    temporal_valid_to: str
    input_rank: int
    parent_rank_hint: int | None
    parent_score_hint: float
    site_score: float

    @classmethod
    def freeze(cls, value: _Reference) -> "FrozenReferenceV12":
        return cls(
            siret=value.siret,
            siren=value.siren,
            channels=tuple(sorted(value.channels)),
            sources=tuple(sorted(value.sources)),
            signals=tuple(value.signals),
            evidence_ids=tuple(sorted(value.evidence_ids)),
            evidence_sources=tuple(sorted(value.evidence_sources)),
            lanes=tuple(sorted(value.lanes)),
            reasons=tuple(sorted(value.reasons)),
            channel_ranks=tuple(sorted(value.channel_ranks.items())),
            names=tuple(value.names),
            addresses=tuple(value.addresses),
            insee=value.insee,
            postcode=value.postcode,
            number=value.number,
            state=value.state,
            is_headquarters=bool(value.is_headquarters),
            temporal_valid_from=value.temporal_valid_from,
            temporal_valid_to=value.temporal_valid_to,
            input_rank=int(value.input_rank),
            parent_rank_hint=value.parent_rank_hint,
            parent_score_hint=float(value.parent_score_hint),
            site_score=float(value.site_score),
        )

    def thaw(self) -> _Reference:
        return _Reference(
            siret=self.siret,
            siren=self.siren,
            channels=set(self.channels),
            sources=set(self.sources),
            signals=list(self.signals),
            evidence_ids=set(self.evidence_ids),
            evidence_sources=set(self.evidence_sources),
            lanes=set(self.lanes),
            reasons=set(self.reasons),
            channel_ranks=dict(self.channel_ranks),
            names=self.names,
            addresses=self.addresses,
            insee=self.insee,
            postcode=self.postcode,
            number=self.number,
            state=self.state,
            is_headquarters=self.is_headquarters,
            temporal_valid_from=self.temporal_valid_from,
            temporal_valid_to=self.temporal_valid_to,
            input_rank=self.input_rank,
            parent_rank_hint=self.parent_rank_hint,
            parent_score_hint=self.parent_score_hint,
            site_score=self.site_score,
        )


@dataclass(frozen=True)
class SceneParentV12:
    siren: str
    feature_values: tuple[float, ...]
    minimum_input_rank: int

    def __post_init__(self) -> None:
        if len(self.feature_values) != len(CASCADE_PARENT_FEATURES):
            raise ValueError("v12 scene parent feature schema/order drift")


@dataclass(frozen=True)
class FoldFusedSceneV12:
    """Label-free immutable scene shared by all three outer contexts."""

    query_id: str
    query_fold: int
    query: RetrievalQuery
    references: tuple[FrozenReferenceV12, ...]
    parents: tuple[SceneParentV12, ...]
    input_reference_count: int
    parent_count_before_rescue: int
    rescue_triggered: bool
    small_insee_sites: int
    small_insee_pool: int
    small_insee_strategy: str
    structured_relation_references: int

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("v12 fold-fused scene requires a query_id")
        if self.query_fold not in ALLOWED_FOLDS_V12:
            raise ValueError("v12 fold-fused scene refuses fold 0/1")
        if len({value.siret for value in self.references}) != len(self.references):
            raise ValueError("v12 fold-fused scene references must be unique")
        if len({value.siren for value in self.parents}) != len(self.parents):
            raise ValueError("v12 fold-fused scene parents must be unique")

    def thaw_references(self) -> dict[str, _Reference]:
        return {value.siret: value.thaw() for value in self.references}

    def parent_matrix(self) -> np.ndarray:
        return np.asarray(
            [value.feature_values for value in self.parents],
            dtype=np.float32,
            order="C",
        )


@dataclass(frozen=True)
class ParentScoreV12:
    siren: str
    score: float
    rank: int


@dataclass(frozen=True)
class FoldFusedContextMaskV12:
    """Immutable boundary preventing evidence from crossing outer contexts."""

    outer_fold: int
    query_fold: int
    model_key: str
    parent_scores: tuple[ParentScoreV12, ...]
    bounded_parent_sirens: tuple[str, ...]
    selected_parent_sirens: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.outer_fold not in ALLOWED_FOLDS_V12:
            raise ValueError("v12 context outer fold must be 2, 3 or 4")
        expected = PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12[self.outer_fold][
            self.query_fold
        ]
        if self.model_key != expected:
            raise ValueError("v12 context uses the wrong nested parent model")
        score_ids = tuple(value.siren for value in self.parent_scores)
        if len(set(score_ids)) != len(score_ids):
            raise ValueError("v12 context parent scores must be unique")
        if not set(self.selected_parent_sirens).issubset(
            self.bounded_parent_sirens
        ):
            raise ValueError("v12 selected parent escaped its context bound")

    def score_by_siren(self) -> dict[str, tuple[int, float]]:
        return {
            value.siren: (value.rank, value.score)
            for value in self.parent_scores
        }


@dataclass(frozen=True)
class ExpandedParentSitesV12:
    siren: str
    result: SiteQueryResult


@dataclass(frozen=True)
class PreparedFoldFusedSceneV12:
    scene: FoldFusedSceneV12
    contexts: tuple[FoldFusedContextMaskV12, ...]
    lookup_parent_sirens: tuple[str, ...]
    expanded_sites: tuple[ExpandedParentSitesV12, ...]

    def context(self, outer_fold: int) -> FoldFusedContextMaskV12:
        for value in self.contexts:
            if value.outer_fold == int(outer_fold):
                return value
        raise KeyError(outer_fold)

    def expansion_by_siren(self) -> dict[str, SiteQueryResult]:
        return {value.siren: value.result for value in self.expanded_sites}


def _aggregate_parents(
    references: Mapping[str, _Reference],
) -> dict[str, _Parent]:
    """Aggregate reporting fields without computing features or scores."""

    parents: dict[str, _Parent] = {}
    for reference in references.values():
        parent = parents.setdefault(reference.siren, _Parent(reference.siren))
        parent.references.append(reference)
        parent.channels.update(reference.channels)
        parent.sources.update(reference.sources | reference.evidence_sources)
    return parents


def _parents_from_references(
    references: Mapping[str, _Reference],
    query: RetrievalQuery,
    *,
    rrf_constant: float,
) -> dict[str, _Parent]:
    """Aggregate and feature parents once without invoking a model scorer."""

    parents = _aggregate_parents(references)
    for parent in parents.values():
        parent.features = _parent_features(parent, query, rrf_constant)
    return parents


def _merge_into(
    references: dict[str, _Reference], incoming: _Reference
) -> None:
    existing = references.get(incoming.siret)
    if existing is None:
        references[incoming.siret] = incoming
    else:
        _merge_reference(existing, incoming)


def _small_commune_census_once(
    sidecars: SidecarReaderV12,
    query: RetrievalQuery,
    references: dict[str, _Reference],
    config: RetrievalTop100CascadeV12Config,
) -> tuple[int, int, str]:
    if not query.insee:
        return 0, 0, "NO_INSEE"
    method = getattr(sidecars, "all_sites_by_insee_if_fits", None)
    if callable(method):
        result = method(query.insee, max_sites=config.small_insee_threshold)
    else:
        fallback = getattr(sidecars, "geo_candidates", None)
        if not callable(fallback):
            return 0, 0, "UNAVAILABLE"
        result = fallback(
            insee=query.insee,
            postcode=query.postcode,
            street_number=query.number,
            street=_street_query(query),
            max_sites=config.small_insee_threshold,
        )
    if (
        result.strategy != "INSEE_ALL"
        or result.matched_count > config.small_insee_threshold
    ):
        return 0, int(result.matched_count), str(result.strategy)
    for rank, site in enumerate(result.sites, start=1):
        _merge_into(
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


def compile_label_blind_scene_v12(
    rows: Sequence[Mapping[str, Any]],
    *,
    query_id: str,
    sidecars: SidecarReaderV12,
    config: RetrievalTop100CascadeV12Config,
    query: RetrievalQuery | Mapping[str, Any] | None = None,
) -> FoldFusedSceneV12:
    """Compile common cascade state exactly once, without retaining labels."""

    if not rows:
        raise ValueError("v12 fold-fused scene requires natural references")
    if not query_id:
        raise ValueError("v12 fold-fused scene requires a query_id")
    observed_folds = {
        int(row.get("fold", row.get("oof_fold", -1))) for row in rows
    }
    if len(observed_folds) != 1 or not observed_folds.issubset(ALLOWED_FOLDS_V12):
        raise ValueError("v12 fold-fused scene refuses fold 0/1 or mixed folds")
    for row in rows:
        row_query_id = row.get("query_id")
        if row_query_id not in (None, "", query_id):
            raise ValueError("v12 fold-fused scene mixes query identifiers")
    # Truth-side values are projected out exactly like the legacy cascade.
    # This assertion protects the immutable IR from accidentally retaining a
    # new forbidden field if the upstream projection evolves.
    cleaned = tuple(_clean_row(row) for row in rows)
    if any(
        is_forbidden_input_key(key)
        for row in cleaned
        for key in row
    ):
        raise AssertionError("v12 scene compiler retained a forbidden label key")
    upstream_rescue_channels = {
        signal.channel
        for row in cleaned
        for signal in _signals_from_row(row)
        if signal.channel in DEFERRED_RESCUE_CHANNELS
    }
    if upstream_rescue_channels:
        raise ValueError(
            "initial v12 fold-fused stack refuses deferred upstream rescue "
            f"channels: {sorted(upstream_rescue_channels)}"
        )
    first = cleaned[0]
    normalized_query = (
        query
        if isinstance(query, RetrievalQuery)
        else RetrievalQuery.from_mapping(_clean_row(query or first))
    )
    allowed_relation_parents = _relation_seed_parents(
        cleaned, normalized_query, config.relation_seed_limit
    )
    references: dict[str, _Reference] = {}
    relation_references = 0
    for row in cleaned:
        reference = _reference_from_row(
            row,
            query=normalized_query,
            allowed_relation_parents=allowed_relation_parents,
        )
        if reference is None:
            continue
        relation_channels = {
            signal.channel
            for signal in reference.signals
            if signal.channel in {"official_successor", "bodacc_relation"}
        }
        if relation_channels:
            if relation_references >= config.relation_reference_budget:
                reference.signals = [
                    signal
                    for signal in reference.signals
                    if signal.channel not in relation_channels
                ]
                reference.channels.difference_update(relation_channels)
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
        _merge_into(references, reference)

    small_count, small_pool, small_strategy = _small_commune_census_once(
        sidecars, normalized_query, references, config
    )
    if not references:
        raise ValueError("v12 fold-fused scene produced no references")
    parents = _parents_from_references(
        references,
        normalized_query,
        rrf_constant=config.rrf_constant,
    )
    scene_parents = tuple(
        SceneParentV12(
            siren=parent.siren,
            feature_values=tuple(
                float(parent.features[name]) for name in CASCADE_PARENT_FEATURES
            ),
            minimum_input_rank=min(
                (reference.input_rank for reference in parent.references),
                default=2_147_483_647,
            ),
        )
        for parent in sorted(parents.values(), key=lambda value: value.siren)
    )
    return FoldFusedSceneV12(
        query_id=str(query_id),
        query_fold=next(iter(observed_folds)),
        query=normalized_query,
        references=tuple(
            FrozenReferenceV12.freeze(value) for value in references.values()
        ),
        parents=scene_parents,
        input_reference_count=len(rows),
        parent_count_before_rescue=len(parents),
        rescue_triggered=(
            sum(parent.name_evidence for parent in parents.values())
            < config.rescue_min_name_parents
            or sum(parent.strong for parent in parents.values())
            < config.rescue_min_strong_parents
        ),
        small_insee_sites=small_count,
        small_insee_pool=small_pool,
        small_insee_strategy=small_strategy,
        structured_relation_references=relation_references,
    )


def validate_parent_scorer_catalog_v12(
    scorers: Mapping[str, ParentMatrixScorerV12],
) -> None:
    if set(scorers) != set(PARENT_MODEL_KEYS_V12):
        raise ValueError("v12 fold-fused scorer catalog must contain six models")
    if any(not callable(getattr(value, "score_matrix", None)) for value in scorers.values()):
        raise TypeError("v12 parent scorer must expose score_matrix")


def _scored_parents_for_context(
    scene: FoldFusedSceneV12,
    scores: np.ndarray,
    config: RetrievalTop100CascadeV12Config,
) -> tuple[tuple[ParentScoreV12, ...], tuple[str, ...], tuple[str, ...]]:
    if len(scores) != len(scene.parents):
        raise ValueError("v12 parent score count does not match scene")
    values = [
        (parent, float(score))
        for parent, score in zip(scene.parents, scores, strict=True)
    ]
    ordered = sorted(
        values,
        key=lambda item: (
            -item[1],
            item[0].minimum_input_rank,
            item[0].siren,
        ),
    )
    parent_scores = tuple(
        ParentScoreV12(parent.siren, score, rank)
        for rank, (parent, score) in enumerate(ordered, start=1)
    )
    # Rescue is disabled in the initial official v12 stack, so the legacy
    # bounded and selected sets are exactly the score order prefixes.
    bounded = tuple(
        parent.siren
        for parent, _score in ordered[: config.master_reference_cap]
    )
    selected = bounded[: config.parent_shortlist_cap]
    return parent_scores, bounded, selected


def score_three_outer_contexts_v12(
    scene: FoldFusedSceneV12,
    *,
    scorers: Mapping[str, ParentMatrixScorerV12],
    config: RetrievalTop100CascadeV12Config,
) -> tuple[FoldFusedContextMaskV12, ...]:
    """Score the three exact nested contexts with three matrix predictions."""

    validate_parent_scorer_catalog_v12(scorers)
    matrix = scene.parent_matrix()
    contexts: list[FoldFusedContextMaskV12] = []
    for outer_fold in sorted(ALLOWED_FOLDS_V12):
        model_key = PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12[outer_fold][
            scene.query_fold
        ]
        scores = np.asarray(scorers[model_key].score_matrix(matrix)).reshape(-1)
        if len(scores) != len(scene.parents) or not np.isfinite(scores).all():
            raise ValueError("v12 parent matrix scorer returned invalid predictions")
        parent_scores, bounded, selected = _scored_parents_for_context(
            scene, scores, config
        )
        contexts.append(
            FoldFusedContextMaskV12(
                outer_fold=outer_fold,
                query_fold=scene.query_fold,
                model_key=model_key,
                parent_scores=parent_scores,
                bounded_parent_sirens=bounded,
                selected_parent_sirens=selected,
            )
        )
    return tuple(contexts)


def _lookup_union_once(
    scene: FoldFusedSceneV12,
    contexts: Sequence[FoldFusedContextMaskV12],
    sidecars: SidecarReaderV12,
    config: RetrievalTop100CascadeV12Config,
) -> tuple[tuple[str, ...], tuple[ExpandedParentSitesV12, ...]]:
    union = tuple(
        sorted(
            {
                siren
                for context in contexts
                for siren in context.selected_parent_sirens
            }
        )
    )
    if len(union) > 3 * config.parent_shortlist_cap:
        raise AssertionError("v12 fold-fused parent union exceeds three contexts")
    if not union or (not scene.query.insee and not scene.query.postcode):
        return union, ()
    batch = getattr(sidecars, "sites_by_sirens_geo", None)
    if not callable(batch):
        raise TypeError("v12 fold-fused execution requires batched sidecar lookup")
    results = batch(
        sirens=list(union),
        insee=scene.query.insee,
        postcode=scene.query.postcode,
        address=scene.query.address,
        street_number=scene.query.number,
        street=_street_query(scene.query),
        include_active=True,
        include_closed=True,
        max_sites_per_siren=config.max_sites_per_parent,
    )
    if set(results) != set(union):
        raise ValueError("v12 fold-fused sidecar did not return every requested parent")
    return union, tuple(
        ExpandedParentSitesV12(siren, results[siren]) for siren in union
    )


def prepare_fold_fused_scene_v12(
    scene: FoldFusedSceneV12,
    *,
    scorers: Mapping[str, ParentMatrixScorerV12],
    sidecars: SidecarReaderV12,
    config: RetrievalTop100CascadeV12Config,
) -> PreparedFoldFusedSceneV12:
    contexts = score_three_outer_contexts_v12(
        scene, scorers=scorers, config=config
    )
    union, expanded = _lookup_union_once(scene, contexts, sidecars, config)
    return PreparedFoldFusedSceneV12(scene, contexts, union, expanded)


def _expand_context_without_rescore(
    references: dict[str, _Reference],
    selected_parents: Sequence[_Parent],
    expansion_by_siren: Mapping[str, SiteQueryResult],
    query: RetrievalQuery,
    config: RetrievalTop100CascadeV12Config,
) -> int:
    if not selected_parents or (not query.insee and not query.postcode):
        return 0
    queues = [
        tuple(
            expansion_by_siren.get(
                parent.siren,
                SiteQueryResult((), "EMPTY", 0, 1, False),
            ).sites
        )
        for parent in selected_parents
    ]
    positions = [0] * len(queues)
    emitted = 0
    while emitted < config.parent_expansion_budget:
        progressed = False
        for parent_rank, (parent, sites) in enumerate(
            zip(selected_parents, queues, strict=True), start=1
        ):
            position = positions[parent_rank - 1]
            if position >= len(sites):
                continue
            progressed = True
            positions[parent_rank - 1] += 1
            emitted += 1
            site = sites[position]
            incoming = _site_reference(
                site,
                signal=CascadeSignalV12(
                    source="v12_parent:siren->official_local_site",
                    channel="v12_parent_local_expansion",
                    rank=emitted,
                    score=1.0 / parent_rank,
                    parent_identifier=parent.siren,
                    parent_rank=parent_rank,
                    site_rank=position + 1,
                ),
                lane="targeted_parent_expansion",
                reason="TOP_PARENT_LOCAL_EXPANSION",
            )
            incoming.parent_score_hint = parent.score
            _merge_into(references, incoming)
            if emitted >= config.parent_expansion_budget:
                break
        if not progressed:
            break
    return emitted


def materialize_outer_context_v12(
    prepared: PreparedFoldFusedSceneV12,
    *,
    outer_fold: int,
    config: RetrievalTop100CascadeV12Config,
) -> RetrievalTop100CascadeV12Result:
    """Replay one context from immutable state without any model rescore."""

    scene = prepared.scene
    context = prepared.context(outer_fold)
    score_by_siren = context.score_by_siren()
    references = scene.thaw_references()
    initial_parents = _aggregate_parents(references)
    feature_by_siren = {
        value.siren: value.feature_values for value in scene.parents
    }
    for siren, parent in initial_parents.items():
        parent.score = score_by_siren[siren][1]
        parent.features = dict(
            zip(CASCADE_PARENT_FEATURES, feature_by_siren[siren], strict=True)
        )
    bounded_parents = [
        initial_parents[siren] for siren in context.bounded_parent_sirens
    ]
    selected_parents = [
        initial_parents[siren] for siren in context.selected_parent_sirens
    ]
    frozen_parent_candidates = tuple(
        _parent_reference(parent, rank)
        for rank, parent in enumerate(bounded_parents, start=1)
    )
    parent_expansions = _expand_context_without_rescore(
        references,
        selected_parents,
        prepared.expansion_by_siren(),
        scene.query,
        config,
    )

    # The legacy implementation calls the parent scorer again here, then
    # immediately overwrites score/features with their pre-expansion values.
    # Aggregate reporting fields only and retain the sealed score/features.
    parents_after = _aggregate_parents(references)
    refreshed_parents: list[_Parent] = []
    for original in selected_parents:
        refreshed = parents_after[original.siren]
        refreshed.score = original.score
        refreshed.features = dict(original.features)
        refreshed_parents.append(refreshed)

    master = _soft_round_robin(
        references.values(),
        parent_order=score_by_siren,
        query=scene.query,
        cap=config.master_reference_cap,
        protected_lanes=frozenset({"small_insee_census"}),
    )
    shortlist_pool = [
        reference
        for reference in master
        if reference.siren in score_by_siren
        or "small_insee_census" in reference.lanes
        or "official_relation_1hop" in reference.lanes
    ]
    shortlist = _soft_round_robin(
        shortlist_pool,
        parent_order=score_by_siren,
        query=scene.query,
        cap=config.hydration_shortlist_cap,
        protected_lanes=frozenset({"small_insee_census"}),
    )
    shortlist_ids = {reference.siret for reference in shortlist}
    if len(shortlist) < config.hydration_shortlist_cap:
        remainder = [
            reference for reference in master if reference.siret not in shortlist_ids
        ]
        shortlist.extend(
            _soft_round_robin(
                remainder,
                parent_order=score_by_siren,
                query=scene.query,
                cap=config.hydration_shortlist_cap - len(shortlist),
            )
        )
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
        rank_score = score_by_siren.get(reference.siren)
        candidates.append(
            CascadeCandidateV12(
                siret=reference.siret,
                siren=reference.siren,
                master_rank=master_rank,
                hydration_rank=hydration_ranks.get(reference.siret),
                parent_rank=rank_score[0] if rank_score else None,
                site_rank=site_ranks[reference.siret],
                parent_score=(
                    rank_score[1] if rank_score else reference.parent_score_hint
                ),
                parent_feature_names=tuple(CASCADE_PARENT_FEATURES),
                parent_feature_values=feature_by_siren.get(
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
    by_id = {value.siret: value for value in candidates}
    hydration = tuple(by_id[value.siret] for value in shortlist)
    parent_rows = tuple(
        _parent_reference(parent, score_by_siren[parent.siren][0])
        for parent in refreshed_parents
    )
    if len(candidates) > config.master_reference_cap:
        raise AssertionError("v12 fold-fused master cap exceeded")
    if len(hydration) > config.hydration_shortlist_cap:
        raise AssertionError("v12 fold-fused hydration cap exceeded")
    if len(parent_rows) > config.parent_shortlist_cap:
        raise AssertionError("v12 fold-fused parent cap exceeded")
    return RetrievalTop100CascadeV12Result(
        query=scene.query,
        master_references=tuple(candidates),
        hydration_shortlist=hydration,
        parent_candidates=frozen_parent_candidates,
        parent_shortlist=parent_rows,
        rescue_triggered=scene.rescue_triggered,
        rescue_executed=False,
        strategy_counts={
            "input_references": scene.input_reference_count,
            "small_insee_sites": scene.small_insee_sites,
            "small_insee_pool": scene.small_insee_pool,
            "small_insee_strategy": scene.small_insee_strategy,
            "structured_relation_references": scene.structured_relation_references,
            "parent_candidates_before_rescue": scene.parent_count_before_rescue,
            "parent_candidates": len(frozen_parent_candidates),
            "rescue_triggered": scene.rescue_triggered,
            "rescue_executed": False,
            "rescue_parent_hits": 0,
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


def logical_result_projection_v12(
    value: RetrievalTop100CascadeV12Result,
    *,
    query_id: str,
) -> Mapping[str, Any]:
    """Canonical logical projection used for old-vs-fused differential tests."""

    return {
        "master": value.master_rows(query_id),
        "hydration": value.admission_rows(query_id),
        "parents": value.parent_rows(query_id),
        "parent_candidates": value.parent_candidate_rows(query_id),
        "rescue_triggered": value.rescue_triggered,
        "rescue_executed": value.rescue_executed,
        "strategy_counts": dict(value.strategy_counts),
    }


def assert_logical_result_parity_v12(
    reference: RetrievalTop100CascadeV12Result,
    fused: RetrievalTop100CascadeV12Result,
    *,
    query_id: str,
) -> None:
    if logical_result_projection_v12(
        reference, query_id=query_id
    ) != logical_result_projection_v12(fused, query_id=query_id):
        raise AssertionError("legacy and fold-fused v12 cascade results differ")


__all__ = [
    "ALLOWED_FOLDS_V12",
    "FoldFusedContextMaskV12",
    "FoldFusedSceneV12",
    "FrozenReferenceV12",
    "LinearParentMatrixScorerV12",
    "PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12",
    "PARENT_MODEL_KEYS_V12",
    "ParentMatrixScorerV12",
    "PreparedFoldFusedSceneV12",
    "XGBoostParentMatrixScorerV12",
    "assert_logical_result_parity_v12",
    "compile_label_blind_scene_v12",
    "logical_result_projection_v12",
    "materialize_outer_context_v12",
    "prepare_fold_fused_scene_v12",
    "score_three_outer_contexts_v12",
    "validate_parent_scorer_catalog_v12",
]
