"""Shared fail-closed preflight for physical and national sharded v12 CLIs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import pyarrow.parquet as pq

from .retrieval_fold_fused_heldout_spool_v12 import validate_spool_calibration_v12
from .retrieval_fold_fused_oof_v12 import (
    ExecutionCaps,
    FoldFusedInvariantError,
    GateThresholds,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_runtime_v12 import (
    FoldFusedOfficialSidecarsV12,
    StrictV12LabelLedger,
    ValidatedExecutorConfigV12,
    load_executor_config_v12,
    load_parent_scorer_catalog_v12,
    load_prefit_parity_evidence_v12,
    reconcile_scene_ir_and_labels_v12,
)
from .retrieval_fold_fused_sharded_runtime_v12 import (
    make_runtime_common_payload_v12,
)
from .retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    site_feature_names,
)
from .retrieval_top100_cascade_v12 import RetrievalTop100CascadeV12Config
from .retrieval_v12_reader_parity import load_mmap_reader_parity_evidence_v12
from .retrieval_v12_scene_ir import SceneIRCache


@dataclass(frozen=True)
class PreparedShardedExecutionV12:
    scene_cache: SceneIRCache
    sample_query_ids: tuple[str, ...]
    national_query_ids: tuple[str, ...]
    lineage: Mapping[str, str]
    executor: ValidatedExecutorConfigV12
    admission: AdmissionConfigV12
    cascade: RetrievalTop100CascadeV12Config
    caps: ExecutionCaps
    gates: GateThresholds
    runtime_payload: Mapping[str, Any]
    prefit_parity_manifest_path: Path
    prefit_parity_manifest_sha256: str
    sharded_execution_plan_path: Path
    sharded_execution_plan_sha256: str


def _manifest_file(path: Path) -> Path:
    value = Path(path).resolve()
    if value.is_dir():
        value = value / "manifest.json"
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def _sealed_sample_ids_v12(
    path: Path, *, ledger: StrictV12LabelLedger
) -> tuple[tuple[str, ...], str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    declared_name = str(ledger.manifest.get("runtime_query_sample_path") or "")
    expected_path = (ledger.manifest_path.parent / declared_name).resolve()
    observed_sha = sha256_file(resolved)
    artifact = (ledger.manifest.get("artifacts") or {}).get(
        "runtime_query_sample"
    ) or {}
    if (
        resolved != expected_path
        or ledger.manifest.get("runtime_query_sample_sha256") != observed_sha
        or ledger.manifest.get("runtime_query_sample_rows") != 512
        or ledger.manifest.get("runtime_query_sample_rule")
        != "SHA256_QUERY_ID_ASC_FIRST_512"
        or artifact.get("path") != declared_name
        or artifact.get("sha256") != observed_sha
        or artifact.get("rows") != 512
    ):
        raise FoldFusedInvariantError("sharded benchmark sample is not ledger sealed")
    table = pq.read_table(resolved)
    if table.column_names != ["query_id", "fold"] or table.num_rows != 512:
        raise FoldFusedInvariantError("sharded benchmark sample schema/cardinality drifted")
    query_ids = tuple(str(value) for value in table["query_id"].to_pylist())
    folds = tuple(int(value.as_py()) for value in table["fold"])
    if (
        len(set(query_ids)) != 512
        or set(folds) != set(TRAIN_FOLDS)
        or any(ledger.get(query_id).fold != fold for query_id, fold in zip(query_ids, folds, strict=True))
    ):
        raise FoldFusedInvariantError("sharded benchmark sample opens wrong queries/folds")
    return query_ids, observed_sha


def prepare_sharded_execution_v12(
    *,
    workspace_root: Path,
    scene_ir: Path,
    query_sample: Path,
    labels: Path,
    sidecars: Path,
    mmap_sidecar: Path,
    reader_parity_manifest: Path,
    parent_model_bundle: Path,
    scientific_config: Path,
    executor_config: Path,
    prefit_parity_manifest: Path,
    sharded_execution_plan: Path,
) -> PreparedShardedExecutionV12:
    """Validate every sealed authority once, outside measured W1/W2/W3."""

    root = Path(workspace_root).resolve()
    executor = load_executor_config_v12(
        executor_config,
        scientific_config_path=scientific_config,
        workspace_root=root,
    )
    admission = AdmissionConfigV12.load(scientific_config)
    cascade = RetrievalTop100CascadeV12Config.load(scientific_config)
    sampling_cap = int(
        admission.fit_candidate_sampling["maximum_candidates_per_query"]
    )
    if (
        admission.train_folds != TRAIN_FOLDS
        or admission.internal_cap != 2_000
        or admission.prerank_cap != 500
        or admission.final_cap != 100
        or cascade.master_reference_cap != 2_000
        or cascade.hydration_shortlist_cap != 500
        or sampling_cap > 128
        or len(site_feature_names(admission)) != 319
    ):
        raise FoldFusedInvariantError("sharded preflight scientific recipe drifted")
    cache = SceneIRCache(scene_ir, verify_source_hash=True)
    ledger = StrictV12LabelLedger.load(labels, config_path=scientific_config)
    reconcile_scene_ir_and_labels_v12(cache, ledger)
    sample_ids, sample_sha = _sealed_sample_ids_v12(query_sample, ledger=ledger)
    scene_rows = cache.index.to_pylist()
    national_ids = tuple(str(row["query_id"]) for row in scene_rows)
    if (
        len(national_ids) != len(set(national_ids))
        or set(national_ids) != set(ledger.expected_query_ids(folds=TRAIN_FOLDS))
        or any(
            int(row["fold"]) != ledger.get(str(row["query_id"])).fold
            for row in scene_rows
        )
    ):
        raise FoldFusedInvariantError("sharded national population differs from ledger")
    scene_manifest = Path(scene_ir).resolve() / "manifest.json"
    parent_manifest = _manifest_file(parent_model_bundle)
    reader_manifest = _manifest_file(reader_parity_manifest)
    with FoldFusedOfficialSidecarsV12.open(
        sidecars, mmap_sidecar, verify_hashes=True
    ) as opened_sidecars:
        local_record = (opened_sidecars.official.manifest.get("files") or {}).get(
            "siren_local_sites.parquet"
        ) or {}
        source_rows = int(
            (opened_sidecars.official.manifest.get("counts") or {}).get(
                "siren_local_sites", -1
            )
        )
        _reader, reader_sha = load_mmap_reader_parity_evidence_v12(
            reader_manifest,
            expected_official_manifest_sha256=opened_sidecars.sidecar_manifest_sha256,
            expected_mmap_manifest_sha256=opened_sidecars.mmap_manifest_sha256,
            expected_source_local_sites_sha256=str(local_record.get("sha256") or ""),
            expected_source_rows=source_rows,
            expected_mmap_build_id=str(
                opened_sidecars.mmap_sites.manifest.get("build_id") or ""
            ),
        )
        parent = load_parent_scorer_catalog_v12(
            parent_manifest,
            config_sha256=executor.scientific_config_sha256,
            source_candidate_sha256=str(cache.manifest["source_candidate_sha256"]),
            source_manifest_sha256=str(cache.manifest["source_manifest_sha256"]),
            sidecar_manifest_sha256=opened_sidecars.sidecar_manifest_sha256,
            label_ledger_sha256=ledger.ledger_sha256,
            label_ledger_manifest_sha256=ledger.manifest_sha256,
        )
        lineage = {
            "scene_ir_sha256": str(cache.manifest["build_id"]),
            "scene_ir_manifest_sha256": sha256_file(scene_manifest),
            "source_candidate_sha256": str(cache.manifest["source_candidate_sha256"]),
            "source_manifest_sha256": str(cache.manifest["source_manifest_sha256"]),
            "label_ledger_sha256": ledger.ledger_sha256,
            "label_ledger_manifest_sha256": ledger.manifest_sha256,
            "scientific_config_sha256": executor.scientific_config_sha256,
            "executor_config_sha256": executor.sha256,
            "official_sidecar_manifest_sha256": opened_sidecars.sidecar_manifest_sha256,
            "mmap_sidecar_manifest_sha256": opened_sidecars.mmap_manifest_sha256,
            "reader_parity_evidence_sha256": reader_sha,
            "parent_model_bundle_sha256": parent.sha256,
            "query_sample_sha256": sample_sha,
        }
    prefit_path = _manifest_file(prefit_parity_manifest)
    prefit, prefit_sha = load_prefit_parity_evidence_v12(
        prefit_path, expected_lineage=lineage
    )
    maximum_bytes = executor.maximum_temporary_gib * 1024**3
    validate_spool_calibration_v12(
        prefit_path,
        prefit.get("heldout_spool_calibration") or {},
        expected_sample_queries=512,
        expected_national_queries=len(national_ids),
        expected_maximum_bytes=maximum_bytes,
    )
    plan_path = Path(sharded_execution_plan).resolve()
    plan_sha = sha256_file(plan_path)
    runtime_payload = make_runtime_common_payload_v12(
        workspace_root=root,
        scene_ir=scene_ir,
        labels=labels,
        sidecars=sidecars,
        mmap_sidecar=mmap_sidecar,
        reader_parity_manifest=reader_manifest,
        parent_model_bundle=parent_manifest,
        scientific_config=scientific_config,
        executor_config=executor_config,
        sharded_execution_plan=plan_path,
        sharded_execution_plan_sha256=plan_sha,
        prefit_parity_manifest=prefit_path,
        prefit_parity_manifest_sha256=prefit_sha,
        lineage=lineage,
    )
    return PreparedShardedExecutionV12(
        scene_cache=cache,
        sample_query_ids=sample_ids,
        national_query_ids=national_ids,
        lineage=lineage,
        executor=executor,
        admission=admission,
        cascade=cascade,
        caps=ExecutionCaps(fit_sample=sampling_cap),
        gates=GateThresholds(
            master_recall_at_2000=admission.gates["oracle_2000"],
            prerank_recall_at_500=admission.gates["prerank_500"],
            top100_recall=admission.gates["recall_100"],
            top100_wilson95_lower=admission.gates["wilson_100"],
            minimum_each_outer_fold_top100=admission.gates["worst_fold_100"],
        ),
        runtime_payload=runtime_payload,
        prefit_parity_manifest_path=prefit_path,
        prefit_parity_manifest_sha256=prefit_sha,
        sharded_execution_plan_path=plan_path,
        sharded_execution_plan_sha256=plan_sha,
    )


def require_external_sharded_output_v12(
    output_dir: Path, *, executor: ValidatedExecutorConfigV12
) -> Path:
    output = Path(output_dir).resolve()
    scratch = executor.scratch_root.resolve()
    if output == scratch or scratch not in output.parents or output.exists():
        raise FoldFusedInvariantError("sharded output must be a new scratch child")
    required = executor.maximum_temporary_gib * 1024**3
    if shutil.disk_usage(scratch).free < required:
        raise FoldFusedInvariantError("external scratch lacks frozen temporary budget")
    return output


__all__ = [
    "PreparedShardedExecutionV12",
    "prepare_sharded_execution_v12",
    "require_external_sharded_output_v12",
]
