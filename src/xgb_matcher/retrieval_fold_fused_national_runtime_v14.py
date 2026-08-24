"""Authorized, transactional national OOF runtime for retrieval v14 only.

The runtime accepts only a strictly reopened v14 physical authorization.  It
creates three fresh fold-owned build actors, releases them before sequential
fits, replays through the v14 logical-hash provider, seals a zero-copy tree in
a content-addressed stage, and atomically promotes it.  V13 authorization,
fragments, projections and failed stages are never accepted or salvaged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, MutableMapping, Sequence

from .retrieval_fold_fused_fragment_dataset_v13 import (
    XGBoostFragmentedSiteLambdaMARTBackendV13,
    fit_site_models_sequential_from_fragments_v13,
    load_build_fragment_dataset_v13,
    seal_build_fragment_dataset_v13,
)
from .retrieval_fold_fused_fragment_outputs_v13 import (
    aggregate_compact_replay_gates_v13,
    load_compact_replay_dataset_v13,
    load_compact_replay_fragment_v13,
    seal_compact_replay_dataset_v13,
)
from .retrieval_fold_fused_fragment_runtime_v13 import (
    _replay_jobs_v13,
    load_and_validate_build_row_group_ledgers_v13,
)
from .retrieval_fold_fused_logical_hash_provider_v14 import (
    V14_LOGICAL_HASH_PROVIDER,
    logical_hash_lineage_record,
)
from .retrieval_fold_fused_national_runtime_v13 import (
    _artifact_record,
    _fit_resource_guard_v13,
    _national_inventory_v13,
    _national_projection_totals_v13,
    _resolve_artifact,
    _validate_fit_resource_guard_v13,
    _validate_relocatable_manifest_tree_v13,
    _validate_runtime_resource_guard_v13,
)
from .retrieval_fold_fused_oof_v12 import (
    GateThresholds,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_physical_benchmark_v12 import (
    current_national_worker_guard_v12,
    sharded_execution_lineage_v12,
)
from .retrieval_fold_fused_physical_benchmark_v13 import (
    _build_jobs_v13,
    _inventory_index_v13,
    _sha256_json,
    v13_implementation_lineage,
)
from .retrieval_fold_fused_physical_benchmark_v14 import (
    BUILD_ENTRYPOINT_V14,
    CONFIG_RELATIVE_PATH,
    CONFIG_SHA256,
    NATIONAL_QUERY_COUNT,
    REPLAY_ENTRYPOINT_V14,
    PhysicalBenchmarkInvariantErrorV14,
    load_execution_authorization_v14,
    load_physical_config_v14,
    v14_implementation_lineage,
    _validate_prepared_authorities_v14,
)
from .retrieval_fold_fused_row_group_ledger_v13 import (
    ROW_GROUP_LEDGER_RECORD_KEYS,
    RowGroupLedgerInvariantErrorV13,
    actor_row_group_ledger_record_v13,
    load_actor_row_group_ledger_records_v13,
    validate_actual_row_group_ledger_set_v13,
)
from .retrieval_fold_fused_sharded_oof_v12 import (
    ShardSpecV12,
    load_shard_build_manifest_v12,
    plan_query_shards,
    spawn_manifest_workers_v12,
)
from .retrieval_fold_fused_sharded_preflight_v12 import (
    PreparedShardedExecutionV12,
)
from .retrieval_fold_fused_zero_copy_run_v13 import (
    CANONICAL_FAMILIES,
    load_zero_copy_canonical_run_v13,
    seal_zero_copy_canonical_run_v13,
)


GIB = 1024**3


class _PostOExclNationalMarkerErrorV14(RuntimeError):
    """The national process won O_EXCL but could not seal the marker."""


SCHEMA_VERSION = "sireto-retrieval-v14-national-fragment-oof-1"
ENGINE_ID = "FOLD_FUSED_FRAGMENT_NATIONAL_OOF_V14_1"
RUN_KIND = "NATIONAL_OOF_FOLDS_2_3_4_ZERO_COPY_V14"

_NATIONAL_MANIFEST_KEYS_V14 = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "identity",
        "run_kind",
        "config_sha256",
        "logical_hash_protocol",
        "terminal_v13_tombstone_preserved",
        "v13_artifact_salvaged",
        "authorization",
        "source_candidate",
        "sharded_execution_plan",
        "prefit_parity",
        "runtime_resource_guard",
        "fit_resource_guard",
        "national_attempt",
        "lineage",
        "query_count",
        "worker_count",
        "folds",
        "fold0_opened",
        "fold1_opened",
        "positive_injection",
        "source_specs",
        "source_row_group_ledgers",
        "build_fragment_dataset",
        "compact_replay_dataset",
        "zero_copy_canonical_run",
        "canonical_datasets",
        "gates",
        "no_merged_files",
        "top500_compact",
        "top100_full",
        "fold1_ready_claimed",
        "artifact_inventory",
        "materialized_output_bytes",
    }
)
_AUTHORIZATION_RECORD_KEYS_V14 = frozenset(
    {"path", "sha256", "benchmark_manifest_path", "benchmark_manifest_sha256"}
)
_SOURCE_RECORD_KEYS_V14 = frozenset({"path", "sha256"})
_ATTEMPT_RECORD_KEYS_V14 = frozenset(
    {"path", "sha256", "attempt_id", "stage_directory_name"}
)


def _validate_national_manifest_keysets_v14(raw: Mapping[str, Any]) -> None:
    """Reject extension fields on every top-level external authority record."""

    records = (
        ("authorization", _AUTHORIZATION_RECORD_KEYS_V14),
        ("source_candidate", _SOURCE_RECORD_KEYS_V14),
        ("sharded_execution_plan", _SOURCE_RECORD_KEYS_V14),
        ("prefit_parity", _SOURCE_RECORD_KEYS_V14),
        ("national_attempt", _ATTEMPT_RECORD_KEYS_V14),
    )
    if set(raw) != _NATIONAL_MANIFEST_KEYS_V14:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national root manifest keyset drifted"
        )
    for name, expected in records:
        value = raw.get(name)
        if not isinstance(value, Mapping) or set(value) != expected:
            raise PhysicalBenchmarkInvariantErrorV14(
                f"v14 national {name} keyset drifted"
            )


@dataclass(frozen=True)
class NationalFragmentOOFV14:
    manifest_path: Path
    manifest_sha256: str
    run_id: str
    query_count: int
    lineage: Mapping[str, str]
    gates: Mapping[str, Any]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path).resolve(), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _execution_paths(
    config: Mapping[str, Any], *, requested: Path, attempt_id: str
) -> tuple[Path, Path, Path]:
    protocol = config["national_execution_protocol"]
    canonical = Path(str(protocol["canonical_output_dir"])).resolve()
    marker = Path(str(protocol["attempt_consumed_marker_path"])).resolve()
    prefix = str(protocol["stage_directory_prefix"])
    stage = canonical.parent / f"{prefix}{attempt_id}"
    if (
        Path(requested).resolve() != canonical
        or marker.parent != canonical.parent
        or stage.parent != canonical.parent
        or not stage.name.startswith(prefix)
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national execution paths escaped the preregistration"
        )
    return canonical, stage, marker


def _attempt_base(
    *,
    authorization_sha256: str,
    benchmark_sha256: str,
    v14_implementation_sha256: str,
    v13_engine_sha256: str,
    source_sha256: str,
    plan_sha256: str,
    prefit_sha256: str,
    canonical: Path,
) -> Mapping[str, Any]:
    return {
        "schema_version": "sireto-retrieval-v14-national-attempt-base-1",
        "config_sha256": CONFIG_SHA256,
        "authorization_sha256": authorization_sha256,
        "physical_benchmark_manifest_sha256": benchmark_sha256,
        "v14_implementation_sha256": v14_implementation_sha256,
        "v13_fragment_engine_implementation_sha256": v13_engine_sha256,
        "logical_hash_algorithm_sha256": V14_LOGICAL_HASH_PROVIDER.algorithm_sha256,
        "source_candidate_sha256": source_sha256,
        "sharded_execution_plan_sha256": plan_sha256,
        "prefit_parity_manifest_sha256": prefit_sha256,
        "canonical_output_dir": str(Path(canonical).resolve()),
        "worker_count": 3,
        "query_count": NATIONAL_QUERY_COUNT,
    }


def _attempt_payload(
    base: Mapping[str, Any], *, attempt_id: str, stage: Path, marker: Path
) -> Mapping[str, Any]:
    return {
        "schema_version": "sireto-retrieval-v14-national-attempt-1",
        "status": "ATTEMPT_IRREVERSIBLY_CONSUMED_BEFORE_STAGE_OR_WORKER",
        "attempt_id": attempt_id,
        "attempt_identity": dict(base),
        "stage_output_dir": str(Path(stage).resolve()),
        "marker_path": str(Path(marker).resolve()),
        "v13_retry_or_salvage": False,
        "automatic_retry_allowed": False,
    }


def _consume_attempt(marker: Path, payload: Mapping[str, Any]) -> str:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national one-shot was already consumed"
        ) from error
    try:
        try:
            os.write(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(marker.parent)
        digest = sha256_file(marker)
        if marker.stat().st_nlink != 1:
            raise PhysicalBenchmarkInvariantErrorV14(
                "v14 national attempt marker is hardlinked"
            )
        return digest
    except BaseException as error:
        raise _PostOExclNationalMarkerErrorV14(
            "v14 national marker failed after O_EXCL"
        ) from error


def _observed_marker_sha256(marker: Path) -> str:
    try:
        path = Path(marker).resolve()
        if path.is_file() and not path.is_symlink():
            return sha256_file(path)
    except BaseException:
        pass
    return "UNREADABLE_POST_O_EXCL"


def _validate_attempt(
    marker: Path, *, expected_sha256: str, expected_payload: Mapping[str, Any]
) -> None:
    path = Path(marker).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or sha256_file(path) != expected_sha256
        or json.loads(path.read_text(encoding="utf-8")) != dict(expected_payload)
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national attempt marker drifted"
        )


def _publish_exception_stop(
    *,
    stage: Path,
    canonical: Path,
    marker: Path,
    marker_sha256: str,
    attempt_id: str,
    v14_implementation_sha256: str,
    implementation_files: Sequence[Mapping[str, Any]],
    authorization: Path,
    authorization_sha256: str,
    benchmark: Path,
    benchmark_sha256: str,
    phase: str,
    error: BaseException,
) -> Path:
    stage.mkdir(parents=True, exist_ok=True)
    document = {
            "schema_version": SCHEMA_VERSION,
            "status": "STOP_V14_NATIONAL_EXECUTION_EXCEPTION",
            "attempt_id": attempt_id,
            "attempt_marker": {"path": str(marker), "sha256": marker_sha256},
            "config_sha256": CONFIG_SHA256,
            "v14_implementation_sha256": v14_implementation_sha256,
            "implementation_files": list(implementation_files),
            "authorization": {
                "path": str(authorization),
                "sha256": authorization_sha256,
            },
            "physical_benchmark": {
                "path": str(benchmark),
                "sha256": benchmark_sha256,
            },
            "failed_phase": phase,
            "canonical_output_dir": str(Path(canonical).resolve()),
            "stage_output_dir": str(Path(stage).resolve()),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "v13_artifact_salvaged": False,
            "automatic_retry_allowed": False,
            "exception": {
                "type": type(error).__name__,
                "message_sanitized": " ".join(str(error).split())[:512],
            },
        }
    document["stop_id"] = _sha256_json(document)
    manifest = stage / "manifest.json"
    _atomic_json(manifest, document)
    descriptor = os.open(manifest, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(stage)
    _fsync_directory(stage.parent)
    return manifest


def _frozen_gates(config: Mapping[str, Any]) -> GateThresholds:
    value = config["scientific_invariants"]
    gates = GateThresholds(
        master_recall_at_2000=float(value["master_recall_at_2000"]),
        prerank_recall_at_500=float(value["prerank_recall_at_500"]),
        top100_recall=float(value["exact_recall_at_100"]),
        top100_wilson95_lower=float(value["exact_recall_at_100_wilson95_lower"]),
        minimum_each_outer_fold_top100=float(
            value["minimum_each_outer_fold_exact_recall_at_100"]
        ),
    )
    if gates != GateThresholds():
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national quality gates differ from frozen v12"
        )
    return gates


def _validated_authorized_base_lineage(
    *,
    prepared: PreparedShardedExecutionV12,
    workspace_root: Path,
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    benchmark_path: Path,
    v13_implementation_sha256: str,
    v14_implementation_sha256: str,
) -> Mapping[str, str]:
    """Bind Prepared to the exact measured W3 before and after O_EXCL."""

    root = Path(workspace_root).resolve()
    _validate_prepared_authorities_v14(
        prepared, config=config, workspace_root=root
    )
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    auth_identity = authorization["authorization_identity"]
    authorized_w3_lineage = {
        str(key): str(value)
        for key, value in dict(
            ((benchmark.get("w3") or {}).get("lineage") or {})
        ).items()
    }
    base_lineage = dict(
        sharded_execution_lineage_v12(dict(prepared.lineage), workspace_root=root)
    )
    expected_w3_lineage = {
        **base_lineage,
        "v13_implementation_sha256": v13_implementation_sha256,
    }
    trial_path = Path(
        str((benchmark.get("w3") or {}).get("trial_manifest_path") or "")
    ).resolve()
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    configured_source = Path(
        str(config["input_lineage"]["source_candidate_path"])
    ).resolve()
    configured_plan = (
        root / str(config["authorities"]["v12_sharded_plan"]["path"])
    ).resolve()
    plan_authority = config["authorities"]["v12_sharded_plan"]
    if sha256_file(configured_plan) != plan_authority["sha256"]:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 frozen sharded plan bytes drifted"
        )
    plan_raw = json.loads(configured_plan.read_text(encoding="utf-8"))
    prefit_authority = (
        (plan_raw.get("input_lineage") or {}).get("prefit_parity") or {}
    )
    expected_prefit = Path(
        str(prefit_authority.get("manifest_path") or "")
    ).resolve()
    expected_prefit_sha = str(prefit_authority.get("manifest_sha256") or "")
    if (
        authorized_w3_lineage != expected_w3_lineage
        or base_lineage["sharded_implementation_sha256"]
        != auth_identity["sharded_implementation_sha256"]
        or base_lineage["runtime_environment_sha256"]
        != auth_identity["runtime_environment_sha256"]
        or v14_implementation_sha256
        != auth_identity["v14_implementation_sha256"]
        or v13_implementation_sha256
        != auth_identity["v13_fragment_engine_implementation_sha256"]
        or prepared.sharded_execution_plan_sha256
        != auth_identity["sharded_execution_plan_sha256"]
        or prepared.prefit_parity_manifest_sha256
        != auth_identity["prefit_parity_manifest_sha256"]
        or prepared.gates != _frozen_gates(config)
        or prepared.scene_cache.source.resolve() != configured_source
        or prepared.sharded_execution_plan_path.resolve() != configured_plan
        or prepared.sharded_execution_plan_sha256 != plan_authority["sha256"]
        or prepared.prefit_parity_manifest_path.resolve() != expected_prefit
        or prepared.prefit_parity_manifest_sha256 != expected_prefit_sha
        or not expected_prefit.is_file()
        or sha256_file(expected_prefit) != expected_prefit_sha
        or str((trial.get("source_candidate") or {}).get("sha256") or "")
        != config["input_lineage"]["source_candidate_sha256"]
        or Path(
            str((trial.get("source_candidate") or {}).get("path") or "")
        ).resolve()
        != prepared.scene_cache.source.resolve()
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national Prepared differs from authorized measured W3"
        )
    return base_lineage


def _identity(
    *,
    authorization_sha256: str,
    benchmark_sha256: str,
    v14_implementation_sha256: str,
    v13_engine_sha256: str,
    lineage: Mapping[str, str],
    source_sha256: str,
    plan_sha256: str,
    prefit_sha256: str,
    runtime_guard_sha256: str,
    fit_guard_sha256: str,
    marker_sha256: str,
    attempt_id: str,
    canonical: Path,
    inventory_sha256: str,
    specs: Sequence[Any],
    row_group_manifest_sha256: Sequence[str],
    row_group_union_sha256: str,
    row_group_bytes: int,
    build_sha256: str,
    replay_sha256: str,
    canonical_sha256: str,
    gates: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "engine_id": ENGINE_ID,
        "config_sha256": CONFIG_SHA256,
        "authorization_sha256": authorization_sha256,
        "physical_benchmark_manifest_sha256": benchmark_sha256,
        "v14_implementation_sha256": v14_implementation_sha256,
        "v13_fragment_engine_implementation_sha256": v13_engine_sha256,
        "logical_hash_algorithm_sha256": V14_LOGICAL_HASH_PROVIDER.algorithm_sha256,
        "sharded_implementation_sha256": lineage["sharded_implementation_sha256"],
        "runtime_environment_sha256": lineage["runtime_environment_sha256"],
        "source_candidate_sha256": source_sha256,
        "sharded_execution_plan_sha256": plan_sha256,
        "prefit_parity_manifest_sha256": prefit_sha256,
        "runtime_resource_guard_sha256": runtime_guard_sha256,
        "fit_resource_guard_sha256": fit_guard_sha256,
        "attempt_marker_sha256": marker_sha256,
        "attempt_id": attempt_id,
        "canonical_output_dir": str(Path(canonical).resolve()),
        "artifact_inventory_sha256": inventory_sha256,
        "query_count": NATIONAL_QUERY_COUNT,
        "worker_count": 3,
        "source_spec_sha256": [value.digest for value in specs],
        "source_row_group_ledger_sha256": list(row_group_manifest_sha256),
        "source_row_group_union_sha256": row_group_union_sha256,
        "source_row_group_logical_compressed_bytes": row_group_bytes,
        "build_fragment_dataset_sha256": build_sha256,
        "compact_replay_dataset_sha256": replay_sha256,
        "zero_copy_canonical_run_sha256": canonical_sha256,
        "gates_sha256": _sha256_json(gates),
    }


def _execute_stage(
    *,
    prepared: PreparedShardedExecutionV12,
    workspace_root: Path,
    stage: Path,
    canonical: Path,
    benchmark: Path,
    benchmark_sha256: str,
    authorization: Path,
    authorization_sha256: str,
    marker: Path,
    marker_sha256: str,
    attempt_id: str,
    phase: MutableMapping[str, str],
) -> NationalFragmentOOFV14:
    root = Path(workspace_root).resolve()
    config = load_physical_config_v14(root / CONFIG_RELATIVE_PATH)
    auth = load_execution_authorization_v14(
        authorization,
        expected_authorization_sha256=authorization_sha256,
        benchmark_manifest_path=benchmark,
        expected_benchmark_manifest_sha256=benchmark_sha256,
        workspace_root=root,
    )
    v14_sha, _v14_files = v14_implementation_lineage(root)
    v13_sha, _v13_files = v13_implementation_lineage(root)
    base_lineage = dict(
        _validated_authorized_base_lineage(
            prepared=prepared,
            workspace_root=root,
            config=config,
            authorization=auth,
            benchmark_path=benchmark,
            v13_implementation_sha256=v13_sha,
            v14_implementation_sha256=v14_sha,
        )
    )
    lineage = dict(base_lineage)
    lineage["v13_implementation_sha256"] = v13_sha
    lineage["v14_implementation_sha256"] = v14_sha
    lineage["physical_benchmark_manifest_sha256"] = benchmark_sha256
    lineage["execution_authorization_sha256"] = authorization_sha256
    source_sha = str(config["input_lineage"]["source_candidate_sha256"])
    attempt_base = _attempt_base(
        authorization_sha256=authorization_sha256,
        benchmark_sha256=benchmark_sha256,
        v14_implementation_sha256=v14_sha,
        v13_engine_sha256=v13_sha,
        source_sha256=source_sha,
        plan_sha256=prepared.sharded_execution_plan_sha256,
        prefit_sha256=prepared.prefit_parity_manifest_sha256,
        canonical=canonical,
    )
    expected_payload = _attempt_payload(
        attempt_base, attempt_id=attempt_id, stage=stage, marker=marker
    )
    _validate_attempt(
        marker, expected_sha256=marker_sha256, expected_payload=expected_payload
    )
    specs = plan_query_shards(
        prepared.scene_cache.index, prepared.national_query_ids, 3
    )
    if (
        len(specs) != 3
        or tuple(value.folds for value in specs) != ((2,), (3,), (4,))
        or sum(value.query_count for value in specs) != NATIONAL_QUERY_COUNT
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national population is not three exhaustive fold shards"
        )
    guard_base = dict(
        current_national_worker_guard_v12(
            3, prepared.sharded_execution_plan_path
        )
    )
    base_guard_sha = str(guard_base.pop("record_sha256"))
    storage_free = shutil.disk_usage(prepared.executor.scratch_root).free
    storage_required = prepared.executor.maximum_temporary_gib * GIB
    runtime_guard = {
        **guard_base,
        "base_guard_sha256": base_guard_sha,
        "storage_free_bytes": storage_free,
        "storage_required_bytes": storage_required,
        "storage_free_pass": storage_free >= storage_required,
        "fallback_worker_count_allowed": False,
    }
    runtime_guard["record_sha256"] = _sha256_json(runtime_guard)
    _validate_runtime_resource_guard_v13(runtime_guard)
    phase["phase"] = "BUILD_ACTORS"
    stage.mkdir(parents=True)
    jobs = _build_jobs_v13(
        prepared=prepared,
        specs=specs,
        output_dir=stage,
        lineage=lineage,
        miniature_fit=False,
        execution_phase="NATIONAL_OOF_FOLDS_2_3_4_V14_BUILD",
        physical_config_sha256=CONFIG_SHA256,
    )
    build_manifests = spawn_manifest_workers_v12(
        BUILD_ENTRYPOINT_V14, jobs, max_workers=3
    )
    build_artifacts = tuple(
        load_shard_build_manifest_v12(path) for path in build_manifests
    )
    source_extent = config["cache_neutral_io_gate"]["source_extent"]
    projection = tuple(source_extent["production_projection"]["columns"])
    groups, logical_bytes = _national_projection_totals_v13(
        prepared.scene_cache.source, projection
    )
    if (
        groups != source_extent["national_expected_unique_row_groups"]
        or logical_bytes
        != source_extent["national_expected_logical_compressed_bytes"]
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national source footer differs from preregistration"
        )
    actual_reads = load_and_validate_build_row_group_ledgers_v13(
        prepared=prepared,
        specs=specs,
        build_artifacts=build_artifacts,
        lineage=lineage,
        expected_unique_row_groups=groups,
        expected_logical_compressed_bytes=logical_bytes,
        physical_config_loader=load_physical_config_v14,
        physical_config_relative_path=CONFIG_RELATIVE_PATH,
    )
    ledger_records = [
        actor_row_group_ledger_record_v13(value, root=stage)
        for value in actual_reads.artifacts
    ]
    build = seal_build_fragment_dataset_v13(
        build_artifacts,
        stage / "build_fragment_dataset",
        relocation_root=stage,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    build = load_build_fragment_dataset_v13(
        build.manifest_path,
        verify_hashes=True,
        verify_logical=True,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    phase["phase"] = "PREFIT_RESOURCE_GUARD"
    fit_guard = _fit_resource_guard_v13(
        sharded_execution_plan_path=prepared.sharded_execution_plan_path,
        memory_used_maximum_bytes=int(
            config["benchmark_512"]["maximum_w3_peak_machine_used_memory_bytes"]
        ),
    )
    _validate_fit_resource_guard_v13(fit_guard)
    if fit_guard["status"] != "PASS_PREFIT_RESOURCES":
        raise PhysicalBenchmarkInvariantErrorV14(
            "STOP_V14_NATIONAL_PREFIT_RESOURCE_GUARD"
        )
    phase["phase"] = "SEQUENTIAL_SITE_FITS"
    backend = XGBoostFragmentedSiteLambdaMARTBackendV13(
        params=prepared.admission.xgb_params("site"),
        output_dir=stage / "site_models",
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    models = fit_site_models_sequential_from_fragments_v13(
        backend,
        build,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    phase["phase"] = "REPLAY_ACTORS"
    replay_jobs = _replay_jobs_v13(
        prepared=prepared,
        specs=specs,
        builds=build_artifacts,
        models=models,
        heldout_merkle=build.heldout_logical_merkle_root,
        lineage=lineage,
        output=stage,
        execution_phase="NATIONAL_OOF_FOLDS_2_3_4_V14_REPLAY",
        physical_config_sha256=CONFIG_SHA256,
    )
    replay_manifests = spawn_manifest_workers_v12(
        REPLAY_ENTRYPOINT_V14, replay_jobs, max_workers=3
    )
    replay_fragments = tuple(
        load_compact_replay_fragment_v13(
            path.parent / "compact_fragment_manifest.json",
            verify_hashes=True,
            verify_logical=True,
            logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
        )
        for path in replay_manifests
    )
    replay = seal_compact_replay_dataset_v13(
        replay_fragments,
        stage / "compact_replay_dataset",
        relocation_root=stage,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    replay = load_compact_replay_dataset_v13(
        replay.manifest_path,
        verify_hashes=True,
        verify_logical=True,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    canonical_run = seal_zero_copy_canonical_run_v13(
        build,
        replay,
        models,
        stage / "canonical_datasets",
        relocation_root=stage,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    canonical_run = load_zero_copy_canonical_run_v13(
        canonical_run.manifest_path,
        verify_hashes=True,
        verify_logical=True,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    gates = dict(
        aggregate_compact_replay_gates_v13(
            replay, _frozen_gates(config), V14_LOGICAL_HASH_PROVIDER
        )
    )
    phase["phase"] = "SEAL_NATIONAL_MANIFEST"
    manifest_path = stage / "manifest.json"
    inventory = _national_inventory_v13(stage, manifest_path)
    identity = _identity(
        authorization_sha256=authorization_sha256,
        benchmark_sha256=benchmark_sha256,
        v14_implementation_sha256=v14_sha,
        v13_engine_sha256=v13_sha,
        lineage=lineage,
        source_sha256=source_sha,
        plan_sha256=prepared.sharded_execution_plan_sha256,
        prefit_sha256=prepared.prefit_parity_manifest_sha256,
        runtime_guard_sha256=runtime_guard["record_sha256"],
        fit_guard_sha256=fit_guard["record_sha256"],
        marker_sha256=marker_sha256,
        attempt_id=attempt_id,
        canonical=canonical,
        inventory_sha256=_sha256_json(inventory),
        specs=specs,
        row_group_manifest_sha256=[
            value.manifest_sha256 for value in actual_reads.artifacts
        ],
        row_group_union_sha256=actual_reads.union_sha256,
        row_group_bytes=actual_reads.logical_compressed_bytes,
        build_sha256=build.manifest_sha256,
        replay_sha256=replay.manifest_sha256,
        canonical_sha256=canonical_run.manifest_sha256,
        gates=gates,
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "EXECUTED_NO_READY_CLAIM",
        "run_id": _sha256_json(identity),
        "identity": identity,
        "run_kind": RUN_KIND,
        "config_sha256": CONFIG_SHA256,
        "logical_hash_protocol": logical_hash_lineage_record(
            V14_LOGICAL_HASH_PROVIDER
        ),
        "terminal_v13_tombstone_preserved": True,
        "v13_artifact_salvaged": False,
        "authorization": {
            "path": str(authorization),
            "sha256": authorization_sha256,
            "benchmark_manifest_path": str(benchmark),
            "benchmark_manifest_sha256": benchmark_sha256,
        },
        "source_candidate": {
            "path": str(prepared.scene_cache.source.resolve()),
            "sha256": source_sha,
        },
        "sharded_execution_plan": {
            "path": str(prepared.sharded_execution_plan_path.resolve()),
            "sha256": prepared.sharded_execution_plan_sha256,
        },
        "prefit_parity": {
            "path": str(prepared.prefit_parity_manifest_path.resolve()),
            "sha256": prepared.prefit_parity_manifest_sha256,
        },
        "runtime_resource_guard": runtime_guard,
        "fit_resource_guard": fit_guard,
        "national_attempt": {
            "path": str(marker),
            "sha256": marker_sha256,
            "attempt_id": attempt_id,
            "stage_directory_name": stage.name,
        },
        "lineage": lineage,
        "query_count": NATIONAL_QUERY_COUNT,
        "worker_count": 3,
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "source_specs": [value.as_manifest() for value in specs],
        "source_row_group_ledgers": ledger_records,
        "build_fragment_dataset": _artifact_record(build.manifest_path, stage),
        "compact_replay_dataset": _artifact_record(replay.manifest_path, stage),
        "zero_copy_canonical_run": _artifact_record(
            canonical_run.manifest_path, stage
        ),
        "canonical_datasets": list(CANONICAL_FAMILIES),
        "gates": gates,
        "no_merged_files": True,
        "top500_compact": True,
        "top100_full": True,
        "fold1_ready_claimed": False,
        "artifact_inventory": inventory,
        "materialized_output_bytes": 0,
    }
    child_bytes = sum(int(value["size_bytes"]) for value in inventory)
    for _ in range(16):
        _atomic_json(manifest_path, manifest)
        observed = child_bytes + manifest_path.stat().st_size
        if manifest["materialized_output_bytes"] == observed:
            break
        manifest["materialized_output_bytes"] = observed
    else:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national materialized-byte fixed point did not converge"
        )
    phase["phase"] = "STAGE_SEALED"
    return NationalFragmentOOFV14(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        run_id=str(manifest["run_id"]),
        query_count=NATIONAL_QUERY_COUNT,
        lineage=lineage,
        gates=gates,
    )


def load_national_fragment_oof_v14(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    workspace_root: Path,
) -> NationalFragmentOOFV14:
    """Strictly reopen the promoted v14 national tree and all authorities."""

    path = Path(manifest_path).resolve()
    root = path.parent
    if not path.is_file() or sha256_file(path) != expected_manifest_sha256:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national manifest bytes drifted"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    _validate_national_manifest_keysets_v14(raw)
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("status") != "EXECUTED_NO_READY_CLAIM"
        or raw.get("run_kind") != RUN_KIND
        or raw.get("config_sha256") != CONFIG_SHA256
        or raw.get("query_count") != NATIONAL_QUERY_COUNT
        or raw.get("worker_count") != 3
        or raw.get("folds") != list(TRAIN_FOLDS)
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or raw.get("no_merged_files") is not True
        or raw.get("top500_compact") is not True
        or raw.get("top100_full") is not True
        or raw.get("fold1_ready_claimed") is not False
        or raw.get("terminal_v13_tombstone_preserved") is not True
        or raw.get("v13_artifact_salvaged") is not False
        or raw.get("logical_hash_protocol")
        != logical_hash_lineage_record(V14_LOGICAL_HASH_PROVIDER)
        or raw.get("run_id") != _sha256_json(raw.get("identity"))
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national manifest contract drifted"
        )
    config = load_physical_config_v14(
        Path(workspace_root).resolve() / CONFIG_RELATIVE_PATH
    )
    protocol = config["national_execution_protocol"]
    if root != Path(str(protocol["canonical_output_dir"])).resolve():
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national manifest is not at canonical output"
        )
    _validate_relocatable_manifest_tree_v13(root)
    inventory, materialized_bytes = _inventory_index_v13(path, raw)
    authorization = raw["authorization"]
    source_record = raw["source_candidate"]
    plan_record = raw["sharded_execution_plan"]
    prefit_record = raw["prefit_parity"]
    marker = raw["national_attempt"]
    auth = load_execution_authorization_v14(
        Path(str(authorization["path"])),
        expected_authorization_sha256=str(authorization["sha256"]),
        benchmark_manifest_path=Path(str(authorization["benchmark_manifest_path"])),
        expected_benchmark_manifest_sha256=str(
            authorization["benchmark_manifest_sha256"]
        ),
        workspace_root=workspace_root,
    )
    current_v14, _files = v14_implementation_lineage(workspace_root)
    current_v13, _engine_files = v13_implementation_lineage(workspace_root)
    lineage = {str(key): str(value) for key, value in raw["lineage"].items()}
    benchmark_raw = json.loads(
        Path(str(authorization["benchmark_manifest_path"])).read_text(
            encoding="utf-8"
        )
    )
    authorized_w3_lineage = {
        str(key): str(value)
        for key, value in dict(
            ((benchmark_raw.get("w3") or {}).get("lineage") or {})
        ).items()
    }
    expected_lineage = {
        **authorized_w3_lineage,
        "v14_implementation_sha256": current_v14,
        "physical_benchmark_manifest_sha256": authorization[
            "benchmark_manifest_sha256"
        ],
        "execution_authorization_sha256": authorization["sha256"],
    }
    if (
        lineage != expected_lineage
        or
        lineage.get("v14_implementation_sha256") != current_v14
        or lineage.get("v13_implementation_sha256") != current_v13
        or lineage.get("physical_benchmark_manifest_sha256")
        != authorization["benchmark_manifest_sha256"]
        or lineage.get("execution_authorization_sha256")
        != authorization["sha256"]
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national implementation lineage drifted"
        )
    identity = raw["identity"]
    attempt_base = _attempt_base(
        authorization_sha256=str(authorization["sha256"]),
        benchmark_sha256=str(authorization["benchmark_manifest_sha256"]),
        v14_implementation_sha256=current_v14,
        v13_engine_sha256=current_v13,
        source_sha256=str(raw["source_candidate"]["sha256"]),
        plan_sha256=str(raw["sharded_execution_plan"]["sha256"]),
        prefit_sha256=str(raw["prefit_parity"]["sha256"]),
        canonical=root,
    )
    attempt_id = _sha256_json(attempt_base)
    canonical, stage, expected_marker = _execution_paths(
        config, requested=root, attempt_id=attempt_id
    )
    if (
        canonical != root
        or marker["attempt_id"] != attempt_id
        or marker["stage_directory_name"] != stage.name
        or Path(str(marker["path"])).resolve() != expected_marker
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national attempt identity drifted"
        )
    _validate_attempt(
        expected_marker,
        expected_sha256=str(marker["sha256"]),
        expected_payload=_attempt_payload(
            attempt_base,
            attempt_id=attempt_id,
            stage=stage,
            marker=expected_marker,
        ),
    )
    if raw.get("materialized_output_bytes") != materialized_bytes:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national materialized-byte declaration drifted"
        )
    build = load_build_fragment_dataset_v13(
        _resolve_artifact(
            root,
            raw["build_fragment_dataset"],
            label="build",
            inventory=inventory,
        ),
        verify_hashes=True,
        verify_logical=True,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    replay = load_compact_replay_dataset_v13(
        _resolve_artifact(
            root,
            raw["compact_replay_dataset"],
            label="replay",
            inventory=inventory,
        ),
        verify_hashes=True,
        verify_logical=True,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    canonical_run = load_zero_copy_canonical_run_v13(
        _resolve_artifact(
            root,
            raw["zero_copy_canonical_run"],
            label="canonical",
            inventory=inventory,
        ),
        verify_hashes=True,
        verify_logical=True,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    gates = dict(
        aggregate_compact_replay_gates_v13(
            replay, _frozen_gates(config), V14_LOGICAL_HASH_PROVIDER
        )
    )
    _validate_runtime_resource_guard_v13(raw["runtime_resource_guard"])
    _validate_fit_resource_guard_v13(raw["fit_resource_guard"])
    specs_raw = raw.get("source_specs")
    if not isinstance(specs_raw, list) or len(specs_raw) != 3:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national source specs drifted"
        )
    specs = tuple(ShardSpecV12.from_manifest(value) for value in specs_raw)
    if (
        specs != tuple(build.specs)
        or tuple(value.folds for value in specs) != ((2,), (3,), (4,))
        or tuple(raw.get("canonical_datasets") or ()) != CANONICAL_FAMILIES
        or set(canonical_run.canonical_datasets) != set(CANONICAL_FAMILIES)
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national shard/canonical family identity drifted"
        )
    workspace = Path(workspace_root).resolve()
    expected_source = Path(
        str(config["input_lineage"]["source_candidate_path"])
    ).resolve()
    expected_plan = (
        workspace / str(config["authorities"]["v12_sharded_plan"]["path"])
    ).resolve()
    plan_raw = json.loads(expected_plan.read_text(encoding="utf-8"))
    expected_prefit = Path(
        str(
            (((plan_raw.get("input_lineage") or {}).get("prefit_parity") or {}).get(
                "manifest_path"
            ))
            or ""
        )
    ).resolve()
    expected_prefit_sha = str(
        (((plan_raw.get("input_lineage") or {}).get("prefit_parity") or {}).get(
            "manifest_sha256"
        ))
        or ""
    )
    if (
        Path(str(source_record["path"])).resolve() != expected_source
        or source_record["sha256"]
        != config["input_lineage"]["source_candidate_sha256"]
        or Path(str(plan_record["path"])).resolve() != expected_plan
        or plan_record["sha256"]
        != config["authorities"]["v12_sharded_plan"]["sha256"]
        or sha256_file(expected_plan) != plan_record["sha256"]
        or Path(str(prefit_record["path"])).resolve() != expected_prefit
        or prefit_record["sha256"] != expected_prefit_sha
        or sha256_file(expected_prefit) != prefit_record["sha256"]
        or auth["authorization_identity"]["sharded_execution_plan_sha256"]
        != plan_record["sha256"]
        or auth["authorization_identity"]["prefit_parity_manifest_sha256"]
        != prefit_record["sha256"]
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national source/plan/PREFIT authority drifted"
        )
    try:
        row_group_artifacts = load_actor_row_group_ledger_records_v13(
            raw.get("source_row_group_ledgers"), root=root
        )
        for record in raw["source_row_group_ledgers"]:
            if set(record) != set(ROW_GROUP_LEDGER_RECORD_KEYS):
                raise RowGroupLedgerInvariantErrorV13(
                    "v14 national actual-read record fields drifted"
                )
            manifest_inventory = inventory.get(str(record["manifest_path"]))
            parquet_inventory = inventory.get(str(record["parquet_path"]))
            if (
                manifest_inventory is None
                or manifest_inventory.get("role")
                != "actual_row_group_ledger_manifest"
                or manifest_inventory.get("family") != "source_read_ledgers"
                or manifest_inventory.get("sha256")
                != record["manifest_sha256"]
                or manifest_inventory.get("size_bytes")
                != record["manifest_size_bytes"]
                or parquet_inventory is None
                or parquet_inventory.get("role")
                != "actual_row_group_ledger_rows"
                or parquet_inventory.get("family") != "source_read_ledgers"
                or parquet_inventory.get("sha256") != record["parquet_sha256"]
                or parquet_inventory.get("size_bytes")
                != record["parquet_size_bytes"]
            ):
                raise RowGroupLedgerInvariantErrorV13(
                    "v14 national actual-read ledger is not inventory-bound"
                )
        source_extent = config["cache_neutral_io_gate"]["source_extent"]
        projection = tuple(source_extent["production_projection"]["columns"])
        expected_groups, expected_bytes = _national_projection_totals_v13(
            expected_source, projection
        )
        if (
            expected_groups
            != int(source_extent["national_expected_unique_row_groups"])
            or expected_bytes
            != int(
                source_extent["national_expected_logical_compressed_bytes"]
            )
        ):
            raise RowGroupLedgerInvariantErrorV13(
                "v14 national source footer differs from extent pins"
            )
        actual_reads = validate_actual_row_group_ledger_set_v13(
            [
                (value.manifest_path, value.manifest_sha256)
                for value in row_group_artifacts
            ],
            expected_specs=specs,
            source_path=expected_source,
            expected_source_candidate_sha256=str(source_record["sha256"]),
            expected_projection_columns=projection,
            expected_projection_sha256=str(
                source_extent["production_projection"]["sha256"]
            ),
            expected_sharded_execution_plan_sha256=str(plan_record["sha256"]),
            expected_execution_lineage=lineage,
            expected_unique_row_groups=expected_groups,
            expected_logical_compressed_bytes=expected_bytes,
            verify_source_hash=True,
        )
    except (KeyError, TypeError, RowGroupLedgerInvariantErrorV13) as error:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national actual source row-group ledger drifted"
        ) from error
    if (
        raw.get("gates") != gates
        or dict(build.lineage) != lineage
        or dict(replay.lineage) != lineage
        or dict(canonical_run.lineage) != lineage
        or build.query_count != NATIONAL_QUERY_COUNT
        or replay.query_count != NATIONAL_QUERY_COUNT
        or canonical_run.query_count != NATIONAL_QUERY_COUNT
        or actual_reads.union_sha256
        != raw["identity"]["source_row_group_union_sha256"]
        or actual_reads.logical_compressed_bytes
        != raw["identity"]["source_row_group_logical_compressed_bytes"]
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national artifacts/gates drifted"
        )
    expected_identity = _identity(
        authorization_sha256=str(authorization["sha256"]),
        benchmark_sha256=str(authorization["benchmark_manifest_sha256"]),
        v14_implementation_sha256=current_v14,
        v13_engine_sha256=current_v13,
        lineage=lineage,
        source_sha256=str(raw["source_candidate"]["sha256"]),
        plan_sha256=str(raw["sharded_execution_plan"]["sha256"]),
        prefit_sha256=str(raw["prefit_parity"]["sha256"]),
        runtime_guard_sha256=str(raw["runtime_resource_guard"]["record_sha256"]),
        fit_guard_sha256=str(raw["fit_resource_guard"]["record_sha256"]),
        marker_sha256=str(marker["sha256"]),
        attempt_id=attempt_id,
        canonical=root,
        inventory_sha256=_sha256_json(raw["artifact_inventory"]),
        specs=specs,
        row_group_manifest_sha256=[
            str(value["manifest_sha256"])
            for value in raw["source_row_group_ledgers"]
        ],
        row_group_union_sha256=str(identity["source_row_group_union_sha256"]),
        row_group_bytes=int(identity["source_row_group_logical_compressed_bytes"]),
        build_sha256=build.manifest_sha256,
        replay_sha256=replay.manifest_sha256,
        canonical_sha256=canonical_run.manifest_sha256,
        gates=gates,
    )
    if dict(identity) != dict(expected_identity):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national content identity drifted"
        )
    return NationalFragmentOOFV14(
        manifest_path=path,
        manifest_sha256=expected_manifest_sha256,
        run_id=str(raw["run_id"]),
        query_count=NATIONAL_QUERY_COUNT,
        lineage=lineage,
        gates=gates,
    )


def execute_national_fragment_oof_v14(
    *,
    prepared: PreparedShardedExecutionV12,
    workspace_root: Path,
    output_dir: Path,
    physical_benchmark_manifest: Path,
    physical_benchmark_manifest_sha256: str,
    execution_authorization: Path,
    execution_authorization_sha256: str,
) -> NationalFragmentOOFV14:
    """Consume the sole v14 national attempt, stage, then atomically promote."""

    root = Path(workspace_root).resolve()
    config = load_physical_config_v14(root / CONFIG_RELATIVE_PATH)
    benchmark = Path(physical_benchmark_manifest).resolve()
    authorization = Path(execution_authorization).resolve()
    auth = load_execution_authorization_v14(
        authorization,
        expected_authorization_sha256=execution_authorization_sha256,
        benchmark_manifest_path=benchmark,
        expected_benchmark_manifest_sha256=physical_benchmark_manifest_sha256,
        workspace_root=root,
    )
    v14_sha, v14_files = v14_implementation_lineage(root)
    v13_sha, _engine_files = v13_implementation_lineage(root)
    # Fail before O_EXCL when Prepared does not exactly reproduce the W3
    # scientific/base lineage authorized by the physical benchmark.  The same
    # validation is repeated inside the consumed stage to close TOCTOU.
    _validated_authorized_base_lineage(
        prepared=prepared,
        workspace_root=root,
        config=config,
        authorization=auth,
        benchmark_path=benchmark,
        v13_implementation_sha256=v13_sha,
        v14_implementation_sha256=v14_sha,
    )
    source_sha = str(config["input_lineage"]["source_candidate_sha256"])
    plan_sha = prepared.sharded_execution_plan_sha256
    prefit_sha = prepared.prefit_parity_manifest_sha256
    requested = Path(output_dir).resolve()
    provisional_base = _attempt_base(
        authorization_sha256=execution_authorization_sha256,
        benchmark_sha256=physical_benchmark_manifest_sha256,
        v14_implementation_sha256=v14_sha,
        v13_engine_sha256=v13_sha,
        source_sha256=source_sha,
        plan_sha256=plan_sha,
        prefit_sha256=prefit_sha,
        canonical=requested,
    )
    attempt_id = _sha256_json(provisional_base)
    canonical, stage, marker = _execution_paths(
        config, requested=requested, attempt_id=attempt_id
    )
    base = _attempt_base(
        authorization_sha256=execution_authorization_sha256,
        benchmark_sha256=physical_benchmark_manifest_sha256,
        v14_implementation_sha256=v14_sha,
        v13_engine_sha256=v13_sha,
        source_sha256=source_sha,
        plan_sha256=plan_sha,
        prefit_sha256=prefit_sha,
        canonical=canonical,
    )
    if (
        attempt_id != _sha256_json(base)
        or canonical.exists()
        or marker.exists()
        or stage.exists()
        or any(canonical.parent.glob(f"{config['national_execution_protocol']['stage_directory_prefix']}*"))
        or len(prepared.national_query_ids) != NATIONAL_QUERY_COUNT
        or auth["authorization_identity"]["authorized_scope"]
        != "NATIONAL_OOF_FOLDS_2_3_4_V14_ONLY"
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 national canonical one-shot is unavailable"
        )
    payload = _attempt_payload(base, attempt_id=attempt_id, stage=stage, marker=marker)
    phase: dict[str, str] = {"phase": "ATOMIC_ATTEMPT_MARKER_CREATION"}
    marker_sha = "0" * 64
    promoted = False
    try:
        marker_sha = _consume_attempt(marker, payload)
        phase["phase"] = "AUTHORIZED_ATTEMPT_CONSUMED"
        product = _execute_stage(
            prepared=prepared,
            workspace_root=root,
            stage=stage,
            canonical=canonical,
            benchmark=benchmark,
            benchmark_sha256=physical_benchmark_manifest_sha256,
            authorization=authorization,
            authorization_sha256=execution_authorization_sha256,
            marker=marker,
            marker_sha256=marker_sha,
            attempt_id=attempt_id,
            phase=phase,
        )
        if product.manifest_path != stage / "manifest.json" or canonical.exists():
            raise PhysicalBenchmarkInvariantErrorV14(
                "v14 sealed national stage cannot be promoted"
            )
        phase["phase"] = "ATOMIC_PROMOTION"
        os.rename(stage, canonical)
        promoted = True
        _fsync_directory(canonical.parent)
        phase["phase"] = "STRICT_PROMOTED_RELOAD"
        try:
            return load_national_fragment_oof_v14(
                canonical / "manifest.json",
                expected_manifest_sha256=product.manifest_sha256,
                workspace_root=root,
            )
        except BaseException:
            os.rename(canonical, stage)
            promoted = False
            _fsync_directory(stage.parent)
            raise
    except BaseException as error:
        if promoted and canonical.exists():
            os.rename(canonical, stage)
            promoted = False
            _fsync_directory(stage.parent)
        if marker.exists():
            _publish_exception_stop(
                stage=stage,
                canonical=canonical,
                marker=marker,
                marker_sha256=_observed_marker_sha256(marker),
                attempt_id=attempt_id,
                v14_implementation_sha256=v14_sha,
                implementation_files=v14_files,
                authorization=authorization,
                authorization_sha256=execution_authorization_sha256,
                benchmark=benchmark,
                benchmark_sha256=physical_benchmark_manifest_sha256,
                phase=phase["phase"],
                error=error,
            )
        raise


__all__ = [
    "ENGINE_ID",
    "NationalFragmentOOFV14",
    "SCHEMA_VERSION",
    "execute_national_fragment_oof_v14",
    "load_national_fragment_oof_v14",
]
