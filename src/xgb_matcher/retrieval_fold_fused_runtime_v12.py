"""Production adapters for the fold-fused retrieval-v12 OOF executor.

The orchestration core in :mod:`retrieval_fold_fused_oof_v12` intentionally
knows nothing about the physical SceneIR, official sidecars or pandas v12
feature implementation.  This module binds those protocols to the sealed
project artifacts without changing the frozen scientific recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import json

import numpy as np
import pandas as pd

from .retrieval_fold_fused_oof_v12 import (
    ENGINE_ID,
    FeatureBatch,
    FoldFusedInvariantError,
    FullMasterStatistics,
    HumanLabel,
    MasterContext,
    ParityEvidence,
    REQUIRED_PARITY_CHECKS,
    SamplingPlan,
    SceneRecord,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_v12 import (
    PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12,
    PARENT_MODEL_KEYS_V12,
    PreparedFoldFusedSceneV12,
    XGBoostParentMatrixScorerV12,
    compile_label_blind_scene_v12,
    materialize_outer_context_v12,
    prepare_fold_fused_scene_v12,
)
from .retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    site_feature_names,
)
from .retrieval_sidecars_v11 import RetrievalSidecarReader, SiteQueryResult
from .retrieval_top100_cascade_v12 import RetrievalTop100CascadeV12Config
from .retrieval_v12_mmap_sidecar import MMapLocalSitesReader
from .retrieval_v12_lazy_site_features import (
    HumanSiteLabelV12,
    LazySiteStatisticsV12,
    _select_fit_indices,
)
from .retrieval_fold_fused_site_matrix_v12 import (
    compute_fold_fused_site_statistics_v12,
    materialize_fold_fused_site_matrix_v12,
)
from .retrieval_v12_scene_ir import SceneIR, SceneIRCache


EXECUTOR_CONFIG_SCHEMA = "sireto-retrieval-v12-fold-fused-executor-1"
PREFIT_PARITY_SCHEMA_VERSION = "sireto-retrieval-v12-fold-fused-prefit-parity-1"
PREFIT_PARITY_CHECKS = (
    "parent_scores_bit_identical",
    "master_candidates_and_ranks_byte_identical",
    "site_features_float32_bit_identical",
    "fit_row_ids_and_order_byte_identical",
)
EXPECTED_SITE_FEATURE_COUNT = 319
PARITY_QUERY_COUNT = 512
PARITY_CONTEXT_COUNT = PARITY_QUERY_COUNT * len(TRAIN_FOLDS)
PARITY_ROW_SCHEMA_VERSION = "sireto-retrieval-v12-fold-fused-parity-rows-1"


def _require_hash(value: Any, *, field_name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise FoldFusedInvariantError(f"{field_name} must be a lowercase SHA-256")
    return text


@dataclass(frozen=True)
class ValidatedExecutorConfigV12:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    scientific_config_path: Path
    scientific_config_sha256: str
    scientific_contract_path: Path
    scientific_contract_sha256: str
    execution_addendum_path: Path
    scratch_root: Path
    maximum_temporary_gib: int


def load_executor_config_v12(
    path: Path,
    *,
    scientific_config_path: Path,
    workspace_root: Path,
) -> ValidatedExecutorConfigV12:
    """Validate the preregistered physical plan against the frozen v12 files."""

    path = Path(path).resolve()
    scientific_config_path = Path(scientific_config_path).resolve()
    workspace_root = Path(workspace_root).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    required_claims = {
        "schema_version": EXECUTOR_CONFIG_SCHEMA,
        "status": "PREREGISTERED_NOT_EXECUTED",
        "engine_id": ENGINE_ID,
        "logical_recipe_changes": False,
        "source_folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
    }
    if any(raw.get(key) != value for key, value in required_claims.items()):
        raise FoldFusedInvariantError("fold-fused executor config claims drifted")
    scientific_sha = sha256_file(scientific_config_path)
    if raw.get("scientific_config_sha256") != scientific_sha:
        raise FoldFusedInvariantError("executor/scientific config hash mismatch")
    declared_scientific = Path(str(raw.get("scientific_config_path") or ""))
    if not declared_scientific.is_absolute():
        declared_scientific = workspace_root / declared_scientific
    if declared_scientific.resolve() != scientific_config_path:
        raise FoldFusedInvariantError("executor points to another scientific config")

    scientific_raw = json.loads(scientific_config_path.read_text(encoding="utf-8"))
    contract = Path(str(scientific_raw.get("contract_path") or ""))
    if not contract.is_absolute():
        contract = workspace_root / contract
    contract = contract.resolve()
    contract_sha = sha256_file(contract)
    declared_contract = Path(str(raw.get("scientific_contract_path") or ""))
    if not declared_contract.is_absolute():
        declared_contract = workspace_root / declared_contract
    if (
        declared_contract.resolve() != contract
        or raw.get("scientific_contract_sha256") != contract_sha
    ):
        raise FoldFusedInvariantError("executor/scientific contract hash mismatch")
    addendum = Path(str(raw.get("execution_addendum_path") or ""))
    if not addendum.is_absolute():
        addendum = workspace_root / addendum
    addendum = addendum.resolve()
    if not addendum.is_file():
        raise FileNotFoundError(addendum)

    scene = raw.get("scene_ir") or {}
    fit = raw.get("fit_materialization") or {}
    parent = raw.get("parent_scoring") or {}
    lookup = raw.get("official_site_lookup") or {}
    parity = raw.get("parity_gate") or {}
    if (
        scene.get("label_blind") is not True
        or scene.get("content_addressed") is not True
        or scene.get("system_disk_allowed") is not False
        or fit.get("maximum_site_rows_per_query_per_training_context") != 128
        or fit.get("full_master_statistics_before_sampling") is not True
        or fit.get("validation_master_sampling") is not False
        or parent.get("feature_dtype") != "float32"
        or parent.get("post_expansion_rescore") is not False
        or lookup.get("evaluation_mode")
        != "ONE_BATCHED_PARENT_LOOKUP_PER_QUERY_THEN_CONTEXT_MASK"
        or lookup.get("context_isolation_required") is not True
        or lookup.get("reader_parity_required") is not True
        or parity.get("query_count") != 512
        or parity.get("stop_on_any_divergence") is not True
    ):
        raise FoldFusedInvariantError("executor physical invariants drifted")
    resources = raw.get("resources") or {}
    scratch_root = Path(str(resources.get("scratch_root") or "")).resolve()
    maximum_temporary_gib = int(resources.get("maximum_temporary_gib", 0))
    if not scratch_root.is_absolute() or maximum_temporary_gib <= 0:
        raise FoldFusedInvariantError("executor scratch budget is invalid")
    return ValidatedExecutorConfigV12(
        path=path,
        sha256=sha256_file(path),
        raw=raw,
        scientific_config_path=scientific_config_path,
        scientific_config_sha256=scientific_sha,
        scientific_contract_path=contract,
        scientific_contract_sha256=contract_sha,
        execution_addendum_path=addendum,
        scratch_root=scratch_root,
        maximum_temporary_gib=maximum_temporary_gib,
    )


class SceneIRSourceAdapterV12:
    """Expose the validated SceneIR cache through the executor protocol."""

    def __init__(self, cache: SceneIRCache, *, batch_size: int = 262_144) -> None:
        if batch_size < 1:
            raise FoldFusedInvariantError("SceneIR batch size must be positive")
        self.cache = cache
        self.batch_size = int(batch_size)

    def iter_scenes(self, *, folds: tuple[int, ...]) -> Iterable[SceneRecord]:
        if tuple(folds) != TRAIN_FOLDS:
            raise FoldFusedInvariantError("SceneIR adapter may open only folds 2/3/4")
        for scene in self.cache.iter_scenes(batch_size=self.batch_size):
            if scene.fold not in TRAIN_FOLDS:
                raise FoldFusedInvariantError("SceneIR adapter observed fold 0/1")
            yield SceneRecord(
                query_id=scene.query_id,
                fold=scene.fold,
                payload=scene,
                metadata={
                    "scene_ir_row_start": scene.row_start,
                    "scene_ir_row_count": scene.row_count,
                },
            )


class StrictV12LabelLedger:
    """Read-only adapter around the existing strict all-human ledger loader."""

    def __init__(
        self,
        labels: pd.DataFrame,
        *,
        ledger_path: Path,
        manifest_path: Path,
        manifest: Mapping[str, Any],
    ) -> None:
        self.ledger_path = Path(ledger_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = dict(manifest)
        self.ledger_sha256 = sha256_file(self.ledger_path)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.operational_sidecar_sha256 = _require_hash(
            self.manifest.get("operational_sidecar_sha256"),
            field_name="operational_sidecar_sha256",
        )
        self._labels: dict[str, HumanLabel] = {}
        for row in labels.to_dict("records"):
            query_id = str(row["query_id"])
            try:
                operational_raw = json.loads(
                    str(row["acceptable_sirets_operational_json"])
                )
            except json.JSONDecodeError as error:
                raise FoldFusedInvariantError("invalid operational label JSON") from error
            operational = tuple(sorted(str(value) for value in operational_raw))
            audit = {
                "official_catalog_present": (
                    None
                    if pd.isna(row.get("official_catalog_present"))
                    else bool(row.get("official_catalog_present"))
                ),
                "active_closed": str(row.get("ground_truth_state") or "UNKNOWN"),
                "crm_origin": str(row.get("data_origin", row.get("source_kind")) or "UNKNOWN"),
                "seen_unseen_siren": str(row.get("seen_unseen_siren") or "UNKNOWN"),
                "known_new_site": str(row.get("known_new_site") or "UNKNOWN"),
                "megapolis": str(row.get("megapolis") or "UNKNOWN"),
                "label_kind": str(row.get("label_kind") or "UNKNOWN"),
                "qualification": str(row.get("qualification") or "UNKNOWN"),
                "qualification_v2": str(row.get("qualification_v2") or "UNKNOWN"),
                "qualification_v3": str(row.get("qualification_v3") or "UNKNOWN"),
            }
            label = HumanLabel(
                query_id=query_id,
                fold=int(row["fold"]),
                human_siret=str(row["human_siret"]),
                operational_sirets=operational,
                audit_metadata=audit,
            )
            if query_id in self._labels:
                raise FoldFusedInvariantError("strict ledger contains duplicate query IDs")
            self._labels[query_id] = label
        if not self._labels or {value.fold for value in self._labels.values()} != set(TRAIN_FOLDS):
            raise FoldFusedInvariantError("strict ledger must contain exactly folds 2/3/4")

    @classmethod
    def load(cls, path: Path, *, config_path: Path) -> "StrictV12LabelLedger":
        # Reuse the existing, audited manifest/sidecar/SIREN-disjoint validator.
        from scripts.run_retrieval_hierarchical_admission_v12 import (
            _load_sealed_query_labels,
        )

        path = Path(path).resolve()
        labels, manifest_path = _load_sealed_query_labels(
            path, config_path=Path(config_path).resolve()
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(
            labels,
            ledger_path=path,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    def expected_query_ids(self, *, folds: tuple[int, ...]) -> Sequence[str]:
        if tuple(folds) != TRAIN_FOLDS:
            raise FoldFusedInvariantError("label ledger may expose only folds 2/3/4")
        return tuple(self._labels)

    def get(self, query_id: str) -> HumanLabel:
        try:
            return self._labels[str(query_id)]
        except KeyError as error:
            raise FoldFusedInvariantError(
                f"query {query_id!r} is absent from the sealed all-human ledger"
            ) from error

    def fold_by_query(self) -> Mapping[str, int]:
        return {query_id: label.fold for query_id, label in self._labels.items()}


@dataclass(frozen=True)
class LoadedParentBundleV12:
    path: Path
    sha256: str
    scorers: Mapping[str, XGBoostParentMatrixScorerV12]
    model_sha256: Mapping[str, str]
    source_retrieval_lineage: Mapping[str, str]
    source_labels_sha256: str
    source_labels_manifest_sha256: str
    config_sha256: str


def load_parent_scorer_catalog_v12(
    path: Path,
    *,
    config_sha256: str,
    source_candidate_sha256: str,
    source_manifest_sha256: str,
    sidecar_manifest_sha256: str,
    label_ledger_sha256: str,
    label_ledger_manifest_sha256: str,
) -> LoadedParentBundleV12:
    """Validate all three nested recipes and load each of the six models once."""

    from scripts.run_retrieval_top100_cascade_v12 import _load_parent_model_recipe

    path = Path(path).resolve()
    recipes = {
        outer: _load_parent_model_recipe(
            path,
            recipe_name=f"outer{outer}",
            required_folds=set(TRAIN_FOLDS),
            scorer_loader=lambda model_path: Path(model_path).resolve(),
        )
        for outer in TRAIN_FOLDS
    }
    bundle_hashes = {recipe.bundle_sha256 for recipe in recipes.values()}
    if len(bundle_hashes) != 1 or next(iter(bundle_hashes)) != sha256_file(path):
        raise FoldFusedInvariantError("nested parent recipe bundle hash drift")
    expected_retrieval = {
        "source_candidate_sha256": source_candidate_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "sidecar_manifest_sha256": sidecar_manifest_sha256,
    }
    for outer, recipe in recipes.items():
        if recipe.outer_fold != outer:
            raise FoldFusedInvariantError("parent recipe outer-fold identity drift")
        if recipe.config_sha256 != config_sha256:
            raise FoldFusedInvariantError("parent bundle/scientific config hash mismatch")
        if dict(recipe.source_retrieval_lineage) != expected_retrieval:
            raise FoldFusedInvariantError("parent bundle retrieval/sidecar lineage mismatch")
        if (
            recipe.source_labels_sha256 != label_ledger_sha256
            or recipe.source_labels_manifest_sha256 != label_ledger_manifest_sha256
        ):
            raise FoldFusedInvariantError("parent bundle label-ledger lineage mismatch")
        for query_fold in TRAIN_FOLDS:
            expected_key = PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12[outer][query_fold]
            if recipe.model_provenance[query_fold]["model"] != expected_key:
                raise FoldFusedInvariantError("parent bundle nested route table drift")

    model_paths: dict[str, Path] = {}
    model_hashes: dict[str, str] = {}
    for recipe in recipes.values():
        for query_fold, provenance in recipe.model_provenance.items():
            model_key = str(provenance["model"])
            model_path = Path(recipe.scorers_by_fold[query_fold]).resolve()
            existing = model_paths.setdefault(model_key, model_path)
            if existing != model_path:
                raise FoldFusedInvariantError("one parent model key resolves to two files")
            model_hashes[model_key] = str(provenance["sha256"])
    if set(model_paths) != set(PARENT_MODEL_KEYS_V12):
        raise FoldFusedInvariantError("parent bundle does not resolve all six model files")
    scorers = {
        model_key: XGBoostParentMatrixScorerV12.from_model_path(model_paths[model_key])
        for model_key in sorted(model_paths)
    }
    return LoadedParentBundleV12(
        path=path,
        sha256=next(iter(bundle_hashes)),
        scorers=scorers,
        model_sha256=model_hashes,
        source_retrieval_lineage=expected_retrieval,
        source_labels_sha256=label_ledger_sha256,
        source_labels_manifest_sha256=label_ledger_manifest_sha256,
        config_sha256=config_sha256,
    )


class FoldFusedOfficialSidecarsV12:
    """Use v11 commune census plus the parity-checked mmap parent lookup."""

    def __init__(
        self,
        official: RetrievalSidecarReader,
        mmap_sites: MMapLocalSitesReader,
        *,
        sidecar_manifest_sha256: str,
        mmap_manifest_sha256: str,
    ) -> None:
        self.official = official
        self.mmap_sites = mmap_sites
        self.sidecar_manifest_sha256 = sidecar_manifest_sha256
        self.mmap_manifest_sha256 = mmap_manifest_sha256

    @classmethod
    def open(
        cls,
        sidecar_dir: Path,
        mmap_dir: Path,
        *,
        verify_hashes: bool = True,
    ) -> "FoldFusedOfficialSidecarsV12":
        sidecar_dir = Path(sidecar_dir).resolve()
        mmap_dir = Path(mmap_dir).resolve()
        official = RetrievalSidecarReader(
            sidecar_dir, verify_hashes=verify_hashes
        )
        try:
            mmap_sites = MMapLocalSitesReader(
                mmap_dir, verify_hashes=verify_hashes
            )
        except Exception:
            official.close()
            raise
        sidecar_manifest = sidecar_dir / "manifest.json"
        mmap_manifest = mmap_dir / "manifest.json"
        sidecar_sha = sha256_file(sidecar_manifest)
        mmap_sha = sha256_file(mmap_manifest)
        local_record = (official.manifest.get("files") or {}).get(
            "siren_local_sites.parquet"
        ) or {}
        if (
            mmap_sites.manifest.get("source_manifest_sha256") != sidecar_sha
            or mmap_sites.manifest.get("source_local_sites_sha256")
            != local_record.get("sha256")
        ):
            official.close()
            mmap_sites.close()
            raise FoldFusedInvariantError("mmap/v11 official sidecar lineage mismatch")
        return cls(
            official,
            mmap_sites,
            sidecar_manifest_sha256=sidecar_sha,
            mmap_manifest_sha256=mmap_sha,
        )

    def all_sites_by_insee_if_fits(
        self, insee: str, *, max_sites: int
    ) -> SiteQueryResult:
        return self.official.all_sites_by_insee_if_fits(
            insee, max_sites=max_sites
        )

    def sites_by_sirens_geo(self, **kwargs: Any) -> dict[str, SiteQueryResult]:
        return self.mmap_sites.sites_by_sirens_geo(**kwargs)

    def sites_by_siren_geo(self, **kwargs: Any) -> SiteQueryResult:
        return self.mmap_sites.sites_by_siren_geo(**kwargs)

    def close(self) -> None:
        self.mmap_sites.close()
        self.official.close()

    def __enter__(self) -> "FoldFusedOfficialSidecarsV12":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


@dataclass
class _LazyMasterState:
    feature_names: tuple[str, ...]
    raw_rows: tuple[Mapping[str, Any], ...]
    statistics_sha256: str
    exact_statistics: LazySiteStatisticsV12
    sampled_indices: tuple[int, ...] | None = None
    sampled_frame: pd.DataFrame | None = None
    full_canonical: pd.DataFrame | None = None
    full_matrix: np.ndarray | None = None


@dataclass
class _MasterPayload:
    raw_rows: tuple[Mapping[str, Any], ...]
    lazy_state: _LazyMasterState | None = None


class FoldFusedContextAdapterV12:
    """Bind fused parent generation to the exact current 319-site-feature path."""

    def __init__(
        self,
        *,
        sidecars: Any,
        parent_scorers: Mapping[str, Any],
        cascade_config: RetrievalTop100CascadeV12Config,
        admission_config: AdmissionConfigV12,
    ) -> None:
        if set(parent_scorers) != set(PARENT_MODEL_KEYS_V12):
            raise FoldFusedInvariantError("context adapter requires all six parent scorers")
        self.sidecars = sidecars
        self.parent_scorers = dict(parent_scorers)
        self.cascade_config = cascade_config
        self.admission_config = admission_config
        self.feature_names = tuple(site_feature_names(admission_config))
        if len(self.feature_names) != EXPECTED_SITE_FEATURE_COUNT:
            raise FoldFusedInvariantError(
                f"frozen v12 site schema must contain {EXPECTED_SITE_FEATURE_COUNT} features"
            )
        self._cached_query_id: str | None = None
        self._prepared: PreparedFoldFusedSceneV12 | None = None
        self._metadata: Mapping[str, Any] | None = None
        self._contexts: dict[int, MasterContext] = {}

    def _prepare_scene(self, scene: SceneRecord) -> None:
        if self._cached_query_id == scene.query_id:
            return
        if not isinstance(scene.payload, SceneIR):
            raise FoldFusedInvariantError("context adapter requires an actual SceneIR payload")
        source_scene = scene.payload
        if source_scene.query_id != scene.query_id or source_scene.fold != scene.fold:
            raise FoldFusedInvariantError("SceneIR payload identity differs from SceneRecord")
        rows = source_scene.as_mappings()
        compiled = compile_label_blind_scene_v12(
            rows,
            query_id=scene.query_id,
            sidecars=self.sidecars,
            config=self.cascade_config,
        )
        self._prepared = prepare_fold_fused_scene_v12(
            compiled,
            scorers=self.parent_scorers,
            sidecars=self.sidecars,
            config=self.cascade_config,
        )
        self._cached_query_id = scene.query_id
        self._metadata = dict(rows[0])
        self._contexts = {}

    def prepare(self, scene: SceneRecord, *, outer_fold: int) -> MasterContext:
        if outer_fold not in TRAIN_FOLDS:
            raise FoldFusedInvariantError("context adapter refuses fold 0/1")
        self._prepare_scene(scene)
        cached = self._contexts.get(outer_fold)
        if cached is not None:
            return cached
        assert self._prepared is not None and self._metadata is not None
        result = materialize_outer_context_v12(
            self._prepared,
            outer_fold=outer_fold,
            config=self.cascade_config,
        )
        rows = tuple(result.master_rows(scene.query_id, self._metadata))
        context = MasterContext(
            query_id=scene.query_id,
            scene_fold=scene.fold,
            outer_fold=outer_fold,
            candidate_sirets=tuple(value.siret for value in result.master_references),
            candidate_parents=tuple(value.siren for value in result.master_references),
            opaque=_MasterPayload(rows),
            metadata={
                "parent_model_key": self._prepared.context(outer_fold).model_key,
                "master_candidate_count": len(rows),
                "parent_expansion_emissions": int(
                    result.strategy_counts.get("parent_expansion_emissions", 0)
                ),
            },
        )
        self._contexts[outer_fold] = context
        return context

    def compute_full_master_statistics(
        self, context: MasterContext
    ) -> FullMasterStatistics:
        if not isinstance(context.opaque, _MasterPayload):
            raise FoldFusedInvariantError("context lacks the actual v12 Master payload")
        payload = context.opaque
        if payload.lazy_state is None:
            if not payload.raw_rows:
                raise FoldFusedInvariantError("actual v12 Master is unexpectedly empty")
            exact = compute_fold_fused_site_statistics_v12(
                payload.raw_rows,
                self.admission_config,
            )
            observed_ids = tuple(exact._base["candidate_siret"].astype(str))
            if observed_ids != context.candidate_sirets:
                raise FoldFusedInvariantError("lightweight Master sort/order drift")
            payload.lazy_state = _LazyMasterState(
                feature_names=self.feature_names,
                raw_rows=payload.raw_rows,
                statistics_sha256=exact.statistics_sha256,
                exact_statistics=exact,
            )
        state = payload.lazy_state
        return FullMasterStatistics(
            query_id=context.query_id,
            outer_fold=context.outer_fold,
            master_count=len(context.candidate_sirets),
            candidate_order_sha256=context.candidate_order_sha256,
            statistics_sha256=state.statistics_sha256,
            complete=True,
            opaque=state,
        )

    def materialize(
        self,
        context: MasterContext,
        statistics: FullMasterStatistics,
        *,
        candidate_indices: Sequence[int] | None,
    ) -> FeatureBatch:
        if not isinstance(statistics.opaque, _LazyMasterState):
            raise FoldFusedInvariantError("materialization requires sealed full-Master statistics")
        state = statistics.opaque
        if state.statistics_sha256 != statistics.statistics_sha256:
            raise FoldFusedInvariantError("full-Master statistics digest drift")
        indices = (
            tuple(range(len(context.candidate_sirets)))
            if candidate_indices is None
            else tuple(int(value) for value in candidate_indices)
        )
        if candidate_indices is None:
            if state.full_canonical is None or state.full_matrix is None:
                state.full_matrix = materialize_fold_fused_site_matrix_v12(
                    state.exact_statistics,
                    self.admission_config,
                )
                # Keep only the identity projection needed by the executor's
                # order guard.  The old path retained a fragmented 319-column
                # DataFrame in addition to the model matrix.
                state.full_canonical = state.exact_statistics._base[
                    ["candidate_siret", "candidate_parent"]
                ].copy()
            canonical = state.full_canonical
            matrix = state.full_matrix
        else:
            if state.sampled_indices != indices or state.sampled_frame is None:
                raise FoldFusedInvariantError(
                    "training features must be materialized by the sealed lazy sampler"
                )
            canonical = state.sampled_frame
            matrix = materialize_fold_fused_site_matrix_v12(
                state.exact_statistics,
                self.admission_config,
                selected_indices=indices,
            )
        expected_ids = tuple(context.candidate_sirets[index] for index in indices)
        if tuple(canonical["candidate_siret"].astype(str)) != expected_ids:
            raise FoldFusedInvariantError("lazy canonical feature row order drift")
        return FeatureBatch(
            query_id=context.query_id,
            scene_fold=context.scene_fold,
            outer_fold=context.outer_fold,
            candidate_indices=indices,
            candidate_sirets=tuple(context.candidate_sirets[index] for index in indices),
            candidate_parents=tuple(context.candidate_parents[index] for index in indices),
            feature_names=state.feature_names,
            values=matrix,
            statistics_sha256=statistics.statistics_sha256,
            row_payloads=tuple(state.raw_rows[index] for index in indices),
        )


class ExistingV12FitSamplerAdapter:
    """Apply the current v12 sampler after sealing its complete-pool inputs."""

    def __init__(self, config: AdmissionConfigV12) -> None:
        self.config = config
        self.maximum_rows = int(config.fit_candidate_sampling["maximum_candidates_per_query"])
        if self.maximum_rows > 128:
            raise FoldFusedInvariantError("frozen v12 fit sampler exceeds 128 rows")

    def choose(
        self,
        context: MasterContext,
        statistics: FullMasterStatistics,
        label: HumanLabel,
        *,
        maximum_rows: int,
    ) -> SamplingPlan:
        if maximum_rows != self.maximum_rows:
            raise FoldFusedInvariantError("executor/sampler fit cap mismatch")
        if not isinstance(statistics.opaque, _LazyMasterState):
            raise FoldFusedInvariantError("v12 sampler requires full-Master statistics")
        state = statistics.opaque
        site_label = HumanSiteLabelV12(
            query_id=label.query_id,
            human_siret=label.human_siret,
            operational_sirets=label.operational_sirets,
        )
        indices = _select_fit_indices(
            state.exact_statistics._base,
            MappingProxyType({label.query_id: site_label}),
            self.config,
        )
        if len(indices) > maximum_rows:
            raise FoldFusedInvariantError("historical sampler exceeded its frozen cap")
        state.sampled_indices = indices
        # Only identities are retained here.  The exact 319 features are
        # compiled in one batch after the frozen index-only sampler returns.
        state.sampled_frame = state.exact_statistics._base.loc[
            list(indices), ["candidate_siret", "candidate_parent"]
        ].reset_index(drop=True)
        return SamplingPlan(
            candidate_indices=indices,
            statistics_sha256=statistics.statistics_sha256,
            rule_id="V12_SAMPLE_PRECOMPUTED_CANDIDATES_FROZEN",
        )


def _validate_parity_lineage(
    raw: Mapping[str, Any], expected_lineage: Mapping[str, str] | None
) -> None:
    if expected_lineage is None:
        return
    observed = raw.get("lineage")
    if not isinstance(observed, Mapping):
        raise FoldFusedInvariantError("parity evidence has no sealed input lineage")
    for key, expected in expected_lineage.items():
        if observed.get(key) != expected:
            raise FoldFusedInvariantError(f"parity evidence lineage mismatch: {key}")


def _parity_artifact_path(
    manifest_path: Path,
    artifacts: Mapping[str, Any],
    *,
    key: str,
    expected_name: str,
) -> tuple[Path, Mapping[str, Any]]:
    """Resolve one parity artifact without permitting path indirection.

    The parity manifest is a gate, not merely a report.  Its claims therefore
    have to be bound to immutable bytes stored beside it; accepting an
    arbitrary absolute path (or ``../``) would make replacing those bytes
    after review too easy.
    """

    record = artifacts.get(key)
    if not isinstance(record, Mapping) or record.get("path") != expected_name:
        raise FoldFusedInvariantError(f"parity artifact record drifted: {key}")
    path = (manifest_path.parent / expected_name).resolve()
    if path.parent != manifest_path.parent or not path.is_file():
        raise FoldFusedInvariantError(f"parity artifact is missing: {key}")
    expected_sha = _require_hash(
        record.get("sha256"), field_name=f"parity artifact {key}"
    )
    if sha256_file(path) != expected_sha:
        raise FoldFusedInvariantError(f"parity artifact hash mismatch: {key}")
    return path, record


def _validate_physical_parity_evidence_v12(
    manifest_path: Path,
    raw: Mapping[str, Any],
    *,
    required_checks: Sequence[str],
    phase: str,
) -> None:
    """Recompute the sealed parity gate from its reference and row artifacts."""

    if raw.get("phase") != phase or raw.get("context_count") != PARITY_CONTEXT_COUNT:
        raise FoldFusedInvariantError("parity phase/context cardinality drifted")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "reference_manifest",
        "parity_rows",
    }:
        raise FoldFusedInvariantError("parity artifact set drifted")
    reference_path, reference_record = _parity_artifact_path(
        manifest_path,
        artifacts,
        key="reference_manifest",
        expected_name="reference_manifest.json",
    )
    rows_path, rows_record = _parity_artifact_path(
        manifest_path,
        artifacts,
        key="parity_rows",
        expected_name="parity_rows.parquet",
    )
    reference_sha = sha256_file(reference_path)
    if (
        raw.get("reference_manifest_sha256") != reference_sha
        or reference_record.get("sha256") != reference_sha
    ):
        raise FoldFusedInvariantError("parity reference manifest is not hash-bound")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected_reference = {
        "schema_version": "sireto-retrieval-v12-fold-fused-reference-1",
        "engine": "LEGACY_SINGLE_CONTEXT_V12",
        "query_count": PARITY_QUERY_COUNT,
        "context_count": PARITY_CONTEXT_COUNT,
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
    }
    if any(reference.get(key) != value for key, value in expected_reference.items()):
        raise FoldFusedInvariantError("parity reference manifest claims drifted")
    if reference.get("lineage") != raw.get("lineage"):
        raise FoldFusedInvariantError("parity reference/top-level lineage mismatch")

    rows = pd.read_parquet(rows_path)
    if (
        rows_record.get("rows") != PARITY_CONTEXT_COUNT
        or len(rows) != PARITY_CONTEXT_COUNT
    ):
        raise FoldFusedInvariantError("parity row artifact cardinality mismatch")
    identity_columns = {"schema_version", "query_id", "fold", "outer_fold"}
    if not identity_columns.issubset(rows.columns):
        raise FoldFusedInvariantError("parity row identity schema drifted")
    if (
        not rows["schema_version"].eq(PARITY_ROW_SCHEMA_VERSION).all()
        or rows["query_id"].astype(str).eq("").any()
        or not rows["fold"].isin(TRAIN_FOLDS).all()
        or not rows["outer_fold"].isin(TRAIN_FOLDS).all()
    ):
        raise FoldFusedInvariantError("parity row identity/fold values drifted")
    identities = rows[["query_id", "outer_fold"]].astype(
        {"query_id": str, "outer_fold": int}
    )
    if identities.duplicated().any():
        raise FoldFusedInvariantError("parity row query/context identities are duplicated")
    grouped = rows.groupby("query_id", sort=False, dropna=False)
    if (
        grouped.ngroups != PARITY_QUERY_COUNT
        or not grouped.size().eq(len(TRAIN_FOLDS)).all()
        or any(set(group["outer_fold"].astype(int)) != set(TRAIN_FOLDS) for _, group in grouped)
        or any(group["fold"].nunique(dropna=False) != 1 for _, group in grouped)
    ):
        raise FoldFusedInvariantError("parity rows do not cover sealed 512 x 3 contexts")

    checks = raw.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(required_checks):
        raise FoldFusedInvariantError("parity check schema drifted")
    digest_pairs = {
        "parent_scores_bit_identical": (
            "legacy_parent_sha256",
            "fused_parent_sha256",
        ),
        "master_candidates_and_ranks_byte_identical": (
            "legacy_master_sha256",
            "fused_master_sha256",
        ),
        "site_features_float32_bit_identical": (
            "legacy_features_sha256",
            "fused_features_sha256",
        ),
        "fit_row_ids_and_order_byte_identical": (
            "legacy_fit_rows_sha256",
            "fused_fit_rows_sha256",
        ),
        "prerank500_byte_identical": (
            "legacy_prerank500_sha256",
            "fused_prerank500_sha256",
        ),
        "top100_byte_identical": (
            "legacy_top100_sha256",
            "fused_top100_sha256",
        ),
    }
    if any(name not in digest_pairs for name in required_checks):
        raise FoldFusedInvariantError("parity check/digest mapping drifted")
    digest_columns = {
        column for name in required_checks for column in digest_pairs[name]
    }
    for name in sorted(digest_columns):
        if name not in rows.columns or not rows[name].map(
            lambda value: len(str(value)) == 64
            and all(character in "0123456789abcdef" for character in str(value))
        ).all():
            raise FoldFusedInvariantError(f"parity row digest schema drifted: {name}")
    for name in required_checks:
        if name not in rows.columns:
            raise FoldFusedInvariantError(f"parity row check is missing: {name}")
        legacy_column, fused_column = digest_pairs[name]
        observed = bool(rows[legacy_column].eq(rows[fused_column]).all())
        claimed = bool(rows[name].eq(True).all())  # noqa: E712 - exact gate value
        if checks.get(name) is not observed or claimed is not observed or observed is not True:
            raise FoldFusedInvariantError(f"parity row check failed: {name}")

    complete = phase == "FULL_POSTFIT"
    expected_site_model = reference.get("site_model_manifest_sha256")
    if complete:
        site_model_sha = _require_hash(
            raw.get("site_model_manifest_sha256"),
            field_name="parity site model manifest",
        )
        model_hashes = raw.get("site_model_sha256")
        reference_model_hashes = reference.get("site_model_sha256")
        recipe_hashes = raw.get("site_model_recipe_sha256")
        reference_recipe_hashes = reference.get("site_model_recipe_sha256")
        if (
            not isinstance(model_hashes, Mapping)
            or set(model_hashes) != {str(value) for value in TRAIN_FOLDS}
            or model_hashes != reference_model_hashes
            or not isinstance(recipe_hashes, Mapping)
            or set(recipe_hashes) != {str(value) for value in TRAIN_FOLDS}
            or recipe_hashes != reference_recipe_hashes
        ):
            raise FoldFusedInvariantError("postfit parity model hash mapping drifted")
        for outer in TRAIN_FOLDS:
            _require_hash(
                model_hashes[str(outer)],
                field_name=f"postfit site model outer {outer}",
            )
            _require_hash(
                recipe_hashes[str(outer)],
                field_name=f"postfit site recipe outer {outer}",
            )
        if len(set(recipe_hashes.values())) != 1 or (
            raw.get("postfit_checks_status") != "PASS"
            or raw.get("required_checks") != list(required_checks)
            or expected_site_model != site_model_sha
        ):
            raise FoldFusedInvariantError("postfit parity model/check lineage drifted")
    elif (
        raw.get("postfit_checks_status") != "PENDING_SITE_MODELS"
        or expected_site_model is not None
        or reference.get("site_model_sha256") is not None
        or reference.get("site_model_recipe_sha256") is not None
        or "site_model_manifest_sha256" in raw
        or "site_model_sha256" in raw
        or "site_model_recipe_sha256" in raw
        or "required_checks" in raw
    ):
        raise FoldFusedInvariantError("prefit parity contains postfit claims")


def load_prefit_parity_evidence_v12(
    path: Path, *, expected_lineage: Mapping[str, str] | None = None
) -> tuple[Mapping[str, Any], str]:
    """Load the non-circular gate required before fitting site models."""

    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": PREFIT_PARITY_SCHEMA_VERSION,
        "status": "PASS",
        "query_count": PARITY_QUERY_COUNT,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "stop_on_any_divergence": True,
        "postfit_checks_status": "PENDING_SITE_MODELS",
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise FoldFusedInvariantError("fold-fused prefit parity is not gate eligible")
    _validate_physical_parity_evidence_v12(
        path,
        raw,
        required_checks=PREFIT_PARITY_CHECKS,
        phase="PREFIT_NON_CIRCULAR",
    )
    _validate_parity_lineage(raw, expected_lineage)
    return raw, sha256_file(path)


def load_parity_evidence_v12(
    path: Path, *, expected_lineage: Mapping[str, str] | None = None
) -> tuple[ParityEvidence, str]:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "sireto-retrieval-v12-fold-fused-parity-1",
        "status": "PASS",
        "query_count": PARITY_QUERY_COUNT,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "stop_on_any_divergence": True,
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise FoldFusedInvariantError("fold-fused parity evidence is not gate eligible")
    _validate_physical_parity_evidence_v12(
        path,
        raw,
        required_checks=REQUIRED_PARITY_CHECKS,
        phase="FULL_POSTFIT",
    )
    reference = _require_hash(
        raw.get("reference_manifest_sha256"),
        field_name="parity reference manifest",
    )
    checks = raw.get("checks")
    if not isinstance(checks, Mapping):
        raise FoldFusedInvariantError("parity evidence has no check mapping")
    evidence = ParityEvidence(
        reference_manifest_sha256=reference,
        query_count=PARITY_QUERY_COUNT,
        checks={str(key): bool(value) for key, value in checks.items()},
    )
    # Force the core schema validator now rather than after a multi-hour run.
    if evidence.as_manifest()["status"] != "PASS":
        raise FoldFusedInvariantError("one or more parity checks failed")
    _validate_parity_lineage(raw, expected_lineage)
    return evidence, sha256_file(path)


def reconcile_scene_ir_and_labels_v12(
    cache: SceneIRCache, ledger: StrictV12LabelLedger
) -> None:
    index = cache.index.to_pandas()
    index["query_id"] = index["query_id"].astype(str)
    index["fold"] = index["fold"].astype(int)
    expected = ledger.fold_by_query()
    observed = dict(zip(index["query_id"], index["fold"], strict=True))
    if observed != expected:
        raise FoldFusedInvariantError(
            "SceneIR query/fold population differs from the sealed all-human ledger"
        )
    if set(index["fold"]) != set(TRAIN_FOLDS):
        raise FoldFusedInvariantError("SceneIR/ledger reconciliation opened fold 0/1")


__all__ = [
    "EXECUTOR_CONFIG_SCHEMA",
    "EXPECTED_SITE_FEATURE_COUNT",
    "ExistingV12FitSamplerAdapter",
    "FoldFusedContextAdapterV12",
    "FoldFusedOfficialSidecarsV12",
    "LoadedParentBundleV12",
    "SceneIRSourceAdapterV12",
    "StrictV12LabelLedger",
    "ValidatedExecutorConfigV12",
    "load_executor_config_v12",
    "load_parent_scorer_catalog_v12",
    "load_prefit_parity_evidence_v12",
    "load_parity_evidence_v12",
    "reconcile_scene_ir_and_labels_v12",
]
