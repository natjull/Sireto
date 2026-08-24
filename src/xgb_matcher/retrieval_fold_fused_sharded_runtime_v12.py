"""Real sealed worker adapters for the query-sharded v12 executor.

The physical core owns partitioning, merging, fitting and replay ordering.
This module only reconstructs process-local readers from JSON-safe paths.
Build workers open SceneIR, official sidecars and parent models.  Replay
workers intentionally open none of those heavy generation inputs: they read
only the label-blind held-out spool, the strict label ledger, the frozen
scientific config and the three fitted site models.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .retrieval_fold_fused_oof_v12 import (
    ExecutionCaps,
    ExistingV12SelectionAdapter,
    FeatureBatch,
    FoldFusedInvariantError,
    FittedSiteModel,
    TRAIN_FOLDS,
    XGBoostSiteLambdaMARTBackend,
    sha256_file,
)
from .retrieval_fold_fused_runtime_v12 import (
    ExistingV12FitSamplerAdapter,
    FoldFusedContextAdapterV12,
    FoldFusedOfficialSidecarsV12,
    StrictV12LabelLedger,
    load_executor_config_v12,
    load_parent_scorer_catalog_v12,
    reconcile_scene_ir_and_labels_v12,
)
from .retrieval_fold_fused_prefit_bridge_v12 import PrefitBridgeSinkV12
from .retrieval_fold_fused_sharded_oof_v12 import (
    SceneIRShardSourceV12,
    ShardSpecV12,
    build_shard_v12,
    load_fitted_site_model_records_v12,
    load_heldout_master_record_v12,
    replay_shard_v12,
)
from .retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    site_feature_names,
)
from .retrieval_top100_cascade_v12 import RetrievalTop100CascadeV12Config
from .retrieval_v12_reader_parity import load_mmap_reader_parity_evidence_v12
from .retrieval_v12_scene_ir import SceneIRCache


RUNTIME_PAYLOAD_SCHEMA_VERSION = "sireto-retrieval-v12-sharded-runtime-paths-2"
BUILD_ENTRYPOINT_V12 = (
    "src.xgb_matcher.retrieval_fold_fused_sharded_runtime_v12:"
    "build_worker_entrypoint_v12"
)
REPLAY_ENTRYPOINT_V12 = (
    "src.xgb_matcher.retrieval_fold_fused_sharded_runtime_v12:"
    "replay_worker_entrypoint_v12"
)
_SHA256 = frozenset("0123456789abcdef")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_SHA256)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _manifest_file(path: Path) -> Path:
    value = Path(path).resolve()
    if value.is_dir():
        value = value / "manifest.json"
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def _paths(
    common: Mapping[str, Any], *, resolve_generation_inputs: bool
) -> dict[str, Path]:
    raw = common.get("paths")
    required = {
        "workspace_root",
        "scene_ir",
        "labels",
        "sidecars",
        "mmap_sidecar",
        "reader_parity_manifest",
        "parent_model_bundle",
        "scientific_config",
        "executor_config",
        "sharded_execution_plan",
        "prefit_parity_manifest",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise FoldFusedInvariantError("sharded runtime path schema drifted")
    generation = {
        "scene_ir",
        "sidecars",
        "mmap_sidecar",
        "reader_parity_manifest",
        "parent_model_bundle",
        "prefit_parity_manifest",
    }
    output = {
        key: (
            Path(str(value)).resolve()
            if resolve_generation_inputs or key not in generation
            else Path(str(value))
        )
        for key, value in raw.items()
    }
    if any(not value.is_absolute() for value in output.values()):
        raise FoldFusedInvariantError("sharded runtime paths must be absolute")
    return output


def make_runtime_common_payload_v12(
    *,
    workspace_root: Path,
    scene_ir: Path,
    labels: Path,
    sidecars: Path,
    mmap_sidecar: Path,
    reader_parity_manifest: Path,
    parent_model_bundle: Path,
    scientific_config: Path,
    executor_config: Path,
    sharded_execution_plan: Path,
    sharded_execution_plan_sha256: str,
    prefit_parity_manifest: Path,
    prefit_parity_manifest_sha256: str,
    lineage: Mapping[str, str],
) -> Mapping[str, Any]:
    """Create the only JSON-safe heavy-input descriptor sent to workers."""

    values = {
        "workspace_root": Path(workspace_root).resolve(),
        "scene_ir": Path(scene_ir).resolve(),
        "labels": Path(labels).resolve(),
        "sidecars": Path(sidecars).resolve(),
        "mmap_sidecar": Path(mmap_sidecar).resolve(),
        "reader_parity_manifest": _manifest_file(reader_parity_manifest),
        "parent_model_bundle": _manifest_file(parent_model_bundle),
        "scientific_config": Path(scientific_config).resolve(),
        "executor_config": Path(executor_config).resolve(),
        "sharded_execution_plan": Path(sharded_execution_plan).resolve(),
        "prefit_parity_manifest": _manifest_file(prefit_parity_manifest),
    }
    if (
        not values["workspace_root"].is_dir()
        or not values["scene_ir"].is_dir()
        or not values["sidecars"].is_dir()
        or not values["mmap_sidecar"].is_dir()
        or not values["labels"].is_file()
        or not values["scientific_config"].is_file()
        or not values["executor_config"].is_file()
        or not values["sharded_execution_plan"].is_file()
        or not _valid_sha256(sharded_execution_plan_sha256)
        or not _valid_sha256(prefit_parity_manifest_sha256)
        or sha256_file(values["sharded_execution_plan"])
        != sharded_execution_plan_sha256
        or sha256_file(values["prefit_parity_manifest"])
        != prefit_parity_manifest_sha256
    ):
        raise FoldFusedInvariantError("sharded runtime authority is missing or unsealed")
    lineage_value = {str(key): str(value) for key, value in lineage.items()}
    if not lineage_value or any(not _valid_sha256(value) for value in lineage_value.values()):
        raise FoldFusedInvariantError("sharded runtime lineage is not SHA-256 sealed")
    payload = {
        "schema_version": RUNTIME_PAYLOAD_SCHEMA_VERSION,
        "paths": {key: str(value) for key, value in values.items()},
        "sharded_execution_plan_sha256": sharded_execution_plan_sha256,
        "prefit_parity_manifest_sha256": prefit_parity_manifest_sha256,
        "lineage": lineage_value,
        "replay_generation_inputs_opened": False,
        "fold0_opened": False,
        "fold1_opened": False,
    }
    # Ensure the descriptor is spawn/JSON safe before the core sees it.
    _canonical_json(payload)
    return payload


def _require_common(
    common: Mapping[str, Any],
    lineage: Mapping[str, str],
    *,
    open_generation_inputs: bool,
) -> tuple[dict[str, Path], Any, AdmissionConfigV12, RetrievalTop100CascadeV12Config, StrictV12LabelLedger]:
    if any(
        common.get(key) != value
        for key, value in {
            "schema_version": RUNTIME_PAYLOAD_SCHEMA_VERSION,
            "replay_generation_inputs_opened": False,
            "fold0_opened": False,
            "fold1_opened": False,
        }.items()
    ):
        raise FoldFusedInvariantError("unsafe sharded runtime descriptor")
    lineage_value = {str(key): str(value) for key, value in lineage.items()}
    if not lineage_value or any(not _valid_sha256(value) for value in lineage_value.values()):
        raise FoldFusedInvariantError("sharded runtime lineage is not SHA-256 sealed")
    common_lineage = {
        str(key): str(value)
        for key, value in dict(common.get("lineage") or {}).items()
    }
    if (
        not common_lineage
        or any(lineage_value.get(key) != value for key, value in common_lineage.items())
    ):
        raise FoldFusedInvariantError(
            "worker job lineage differs from sealed descriptor"
        )
    paths = _paths(
        common, resolve_generation_inputs=open_generation_inputs
    )
    plan_sha = str(common.get("sharded_execution_plan_sha256") or "")
    if (
        not _valid_sha256(plan_sha)
        or sha256_file(paths["sharded_execution_plan"]) != plan_sha
    ):
        raise FoldFusedInvariantError("sharded execution plan changed inside worker")
    prefit_sha = str(common.get("prefit_parity_manifest_sha256") or "")
    if open_generation_inputs and (
        not _valid_sha256(prefit_sha)
        or sha256_file(paths["prefit_parity_manifest"]) != prefit_sha
    ):
        raise FoldFusedInvariantError("PREFIT parity changed inside build worker")
    executor = load_executor_config_v12(
        paths["executor_config"],
        scientific_config_path=paths["scientific_config"],
        workspace_root=paths["workspace_root"],
    )
    admission = AdmissionConfigV12.load(paths["scientific_config"])
    cascade = RetrievalTop100CascadeV12Config.load(paths["scientific_config"])
    if (
        admission.train_folds != TRAIN_FOLDS
        or admission.internal_cap != 2_000
        or admission.prerank_cap != 500
        or admission.final_cap != 100
        or cascade.master_reference_cap != 2_000
        or cascade.hydration_shortlist_cap != 500
        or len(site_feature_names(admission)) != 319
    ):
        raise FoldFusedInvariantError("sharded worker scientific caps/schema drifted")
    ledger = StrictV12LabelLedger.load(
        paths["labels"], config_path=paths["scientific_config"]
    )
    observed = {
        "label_ledger_sha256": ledger.ledger_sha256,
        "label_ledger_manifest_sha256": ledger.manifest_sha256,
        "scientific_config_sha256": executor.scientific_config_sha256,
        "executor_config_sha256": executor.sha256,
        "query_sample_sha256": str(
            ledger.manifest.get("runtime_query_sample_sha256") or ""
        ),
    }
    if open_generation_inputs:
        scene_manifest = _manifest_file(paths["scene_ir"])
        scene_raw = json.loads(scene_manifest.read_text(encoding="utf-8"))
        observed.update(
            {
                "scene_ir_sha256": str(scene_raw.get("build_id") or ""),
                "scene_ir_manifest_sha256": sha256_file(scene_manifest),
                "source_candidate_sha256": str(
                    scene_raw.get("source_candidate_sha256") or ""
                ),
                "source_manifest_sha256": str(
                    scene_raw.get("source_manifest_sha256") or ""
                ),
                "official_sidecar_manifest_sha256": sha256_file(
                    _manifest_file(paths["sidecars"])
                ),
                "mmap_sidecar_manifest_sha256": sha256_file(
                    _manifest_file(paths["mmap_sidecar"])
                ),
                "reader_parity_evidence_sha256": sha256_file(
                    _manifest_file(paths["reader_parity_manifest"])
                ),
                "parent_model_bundle_sha256": sha256_file(
                    _manifest_file(paths["parent_model_bundle"])
                ),
            }
        )
    if any(lineage_value.get(key) != value for key, value in observed.items()):
        raise FoldFusedInvariantError("sharded worker input/PREFIT lineage drifted")
    return paths, executor, admission, cascade, ledger


def _caps(raw: Mapping[str, Any]) -> ExecutionCaps:
    try:
        value = ExecutionCaps(**{str(key): int(item) for key, item in raw.items()})
    except (TypeError, ValueError) as error:
        raise FoldFusedInvariantError("invalid sharded worker caps") from error
    if asdict(value) != {str(key): int(item) for key, item in raw.items()}:
        raise FoldFusedInvariantError("sharded worker caps contain unknown fields")
    return value


def _require_output(path: Any, *, scratch_root: Path) -> Path:
    output = Path(str(path or "")).resolve()
    scratch = Path(scratch_root).resolve()
    if output == scratch or scratch not in output.parents or output.exists():
        raise FoldFusedInvariantError("sharded worker output escaped external scratch")
    return output


def build_worker_entrypoint_v12(
    job: Mapping[str, Any],
    *,
    train_store_factory: Any | None = None,
    row_group_observer: Any | None = None,
) -> str:
    """Spawn entrypoint: build one exact train/spool shard from real inputs."""

    if job.get("phase") != "BUILD" or not isinstance(job.get("common"), Mapping):
        raise FoldFusedInvariantError("invalid sharded build job")
    lineage = dict(job.get("lineage") or {})
    paths, executor, admission, cascade, ledger = _require_common(
        job["common"], lineage, open_generation_inputs=True
    )
    spec = ShardSpecV12.from_manifest(dict(job.get("spec") or {}))
    caps = _caps(dict(job.get("caps") or {}))
    output = _require_output(job.get("output_dir"), scratch_root=executor.scratch_root)
    maximum_spool_bytes = int(job.get("maximum_spool_bytes", 0))
    if not 0 < maximum_spool_bytes <= executor.maximum_temporary_gib * 1024**3:
        raise FoldFusedInvariantError("sharded build spool budget drifted")
    miniature_fit = job.get("miniature_fit")
    if miniature_fit not in (True, False):
        raise FoldFusedInvariantError("sharded build phase kind drifted")
    bridge_sink = (
        PrefitBridgeSinkV12.from_prefit_manifest(
            paths["prefit_parity_manifest"],
            expected_prefit_lineage=dict(job["common"].get("lineage") or {}),
            bridge_lineage=lineage,
            expected_query_ids=spec.query_ids,
            expected_query_folds=spec.query_folds,
            expected_prefit_manifest_sha256=str(
                job["common"].get("prefit_parity_manifest_sha256") or ""
            ),
        )
        if miniature_fit
        else None
    )
    # PREFIT, the sharded plan and the content-addressed SceneIR manifest have
    # already sealed the 4.7 GiB source.  Rehashing it independently in W1/W3
    # would add worker-count-dependent I/O and invalidate the physical gate.
    cache = SceneIRCache(paths["scene_ir"], verify_source_hash=False)
    reconcile_scene_ir_and_labels_v12(cache, ledger)
    with FoldFusedOfficialSidecarsV12.open(
        paths["sidecars"], paths["mmap_sidecar"], verify_hashes=False
    ) as sidecars:
        local_record = (sidecars.official.manifest.get("files") or {}).get(
            "siren_local_sites.parquet"
        ) or {}
        source_rows = int(
            (sidecars.official.manifest.get("counts") or {}).get(
                "siren_local_sites", -1
            )
        )
        _reader, reader_sha = load_mmap_reader_parity_evidence_v12(
            paths["reader_parity_manifest"],
            expected_official_manifest_sha256=sidecars.sidecar_manifest_sha256,
            expected_mmap_manifest_sha256=sidecars.mmap_manifest_sha256,
            expected_source_local_sites_sha256=str(local_record.get("sha256") or ""),
            expected_source_rows=source_rows,
            expected_mmap_build_id=str(sidecars.mmap_sites.manifest.get("build_id") or ""),
        )
        if reader_sha != lineage["reader_parity_evidence_sha256"]:
            raise FoldFusedInvariantError("reader parity changed inside build worker")
        parent = load_parent_scorer_catalog_v12(
            paths["parent_model_bundle"],
            config_sha256=executor.scientific_config_sha256,
            source_candidate_sha256=str(cache.manifest["source_candidate_sha256"]),
            source_manifest_sha256=str(cache.manifest["source_manifest_sha256"]),
            sidecar_manifest_sha256=sidecars.sidecar_manifest_sha256,
            label_ledger_sha256=ledger.ledger_sha256,
            label_ledger_manifest_sha256=ledger.manifest_sha256,
        )
        build_kwargs = (
            {"train_store_factory": train_store_factory}
            if train_store_factory is not None
            else {}
        )
        artifact = build_shard_v12(
            spec,
            source=SceneIRShardSourceV12(
                cache,
                spec,
                row_group_observer=row_group_observer,
            ),
            labels=ledger,
            contexts=FoldFusedContextAdapterV12(
                sidecars=sidecars,
                parent_scorers=parent.scorers,
                cascade_config=cascade,
                admission_config=admission,
            ),
            sampler=ExistingV12FitSamplerAdapter(admission),
            admission_config=admission,
            output_dir=output,
            lineage=lineage,
            caps=caps,
            maximum_spool_bytes=maximum_spool_bytes,
            prefit_bridge_sink=bridge_sink,
            **build_kwargs,
        )
    return str(artifact.manifest_path)


class _ReplayFeatureContextV12(FoldFusedContextAdapterV12):
    """Feature-only view; construction cannot open generation authorities."""

    def __init__(self, admission: AdmissionConfigV12) -> None:
        self.admission_config = admission
        self.feature_names = tuple(site_feature_names(admission))

    def prepare(self, *_args: Any, **_kwargs: Any) -> Any:
        raise FoldFusedInvariantError("replay is forbidden from preparing SceneIR")


class _ReplaySiteBackendV12:
    @staticmethod
    def score(model: FittedSiteModel, features: FeatureBatch) -> np.ndarray:
        if features.feature_names != model.feature_names:
            raise FoldFusedInvariantError("replay model/feature schema drifted")
        return np.asarray(model.handle.predict(features.values), dtype=np.float64)


class LazyXGBoostSiteLambdaMARTBackendV12:
    """Delay directory creation until the sharded core owns its output root."""

    def __init__(self, *, params: Mapping[str, Any], output_dir: Path) -> None:
        self.params = dict(params)
        self.output_dir = Path(output_dir).resolve()
        self._delegate: XGBoostSiteLambdaMARTBackend | None = None

    def _backend(self) -> XGBoostSiteLambdaMARTBackend:
        if self._delegate is None:
            self._delegate = XGBoostSiteLambdaMARTBackend(
                params=self.params, output_dir=self.output_dir
            )
        return self._delegate

    def fit(self, artifact: Any) -> FittedSiteModel:
        return self._backend().fit(artifact)

    def score(self, model: FittedSiteModel, features: FeatureBatch) -> np.ndarray:
        return self._backend().score(model, features)


def _load_xgb_handle(path: Path, _outer: int) -> Any:
    import xgboost as xgb

    model = xgb.XGBRanker()
    model.load_model(path)
    return model


def replay_worker_entrypoint_v12(job: Mapping[str, Any]) -> str:
    """Spawn entrypoint: score one spool shard without generation inputs."""

    if job.get("phase") != "REPLAY" or not isinstance(job.get("common"), Mapping):
        raise FoldFusedInvariantError("invalid sharded replay job")
    lineage = dict(job.get("lineage") or {})
    _paths_value, executor, admission, _cascade, ledger = _require_common(
        job["common"], lineage, open_generation_inputs=False
    )
    spec = ShardSpecV12.from_manifest(dict(job.get("spec") or {}))
    caps = _caps(dict(job.get("caps") or {}))
    output = _require_output(job.get("output_dir"), scratch_root=executor.scratch_root)
    build_manifest_path = Path(str(job.get("build_manifest_path") or "")).resolve()
    build_manifest_sha = str(job.get("build_manifest_sha256") or "")
    if (
        not build_manifest_path.is_file()
        or not _valid_sha256(build_manifest_sha)
        or sha256_file(build_manifest_path) != build_manifest_sha
    ):
        raise FoldFusedInvariantError("replay build-manifest lineage drifted")
    heldout = load_heldout_master_record_v12(
        dict(job.get("heldout") or {}), verify_hashes=True
    )
    models = load_fitted_site_model_records_v12(
        dict(job.get("models") or {}),
        handle_loader=_load_xgb_handle,
        verify_hashes=True,
    )
    expected_features = tuple(site_feature_names(admission))
    expected_recipe = hashlib.sha256(
        _canonical_json(dict(admission.xgb_params("site"))).encode("utf-8")
    ).hexdigest()
    if any(
        model.feature_names != expected_features
        or model.recipe_sha256 != expected_recipe
        for model in models.values()
    ):
        raise FoldFusedInvariantError("replay model recipe/319-feature schema drifted")
    artifact = replay_shard_v12(
        spec,
        heldout=heldout,
        labels=ledger,
        contexts=_ReplayFeatureContextV12(admission),
        model_backend=_ReplaySiteBackendV12(),
        models=models,
        selector=ExistingV12SelectionAdapter(admission),
        output_dir=output,
        lineage=lineage,
        build_manifest_sha256=build_manifest_sha,
        caps=caps,
    )
    return str(artifact.manifest_path)


__all__ = [
    "BUILD_ENTRYPOINT_V12",
    "REPLAY_ENTRYPOINT_V12",
    "RUNTIME_PAYLOAD_SCHEMA_VERSION",
    "LazyXGBoostSiteLambdaMARTBackendV12",
    "build_worker_entrypoint_v12",
    "make_runtime_common_payload_v12",
    "replay_worker_entrypoint_v12",
]
