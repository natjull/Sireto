"""Fresh one-shot physical benchmark outer protocol for retrieval v14.

V14 is not a retry or a salvage of v13.  It reuses the frozen v13 fragment
engine only through explicit providers, injects the null-canonical logical
hash, builds new projections from the immutable v12 W1 authority, and writes
only to v14 paths.  A private capability is created after cache-neutral
preflight and before the irreversible v14 marker; only that path can publish
an authorization consumed by the v14 national runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from .retrieval_fold_fused_fragment_runtime_v13 import (
    _compute_w1_functional_projection,
    build_worker_entrypoint_v13,
    finish_fragment_w3_trial_v13,
    replay_worker_entrypoint_v13,
)
from .retrieval_fold_fused_logical_hash_provider_v14 import (
    V14_LOGICAL_HASH_PROVIDER,
    logical_hash_lineage_record,
    require_logical_hash_provider_v14,
)
from .retrieval_fold_fused_logical_hash_v14 import (
    LOGICAL_TABLE_HASH_ALGORITHM_ID_V14,
    LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14,
)
from .retrieval_fold_fused_logical_io_v13 import (
    build_immutable_w1_io_projection_v13,
    compare_w3_to_immutable_w1_io_v13,
    load_immutable_w1_io_projection_v13,
)
from .retrieval_fold_fused_oof_v12 import (
    ExecutionCaps,
    GateThresholds,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_runtime_v12 import load_executor_config_v12
from .retrieval_hierarchical_admission_v12 import AdmissionConfigV12
from .retrieval_top100_cascade_v12 import RetrievalTop100CascadeV12Config
from .retrieval_v12_scene_ir import SceneIRCache
from .retrieval_fold_fused_physical_benchmark_v12 import (
    PhysicalResourceMetricsV12,
    measure_physical_trial_v12,
    sharded_execution_lineage_v12,
)
from .retrieval_fold_fused_physical_benchmark_v13 import (
    FragmentW3TrialProductV13,
    ImmutableW1AnchorV13,
    LogicalIOEvidenceV13,
    PhysicalBenchmarkInvariantErrorV13,
    _build_jobs_v13,
    _derive_physical_checks_v13,
    _logical_io_record,
    _read_json,
    _reconcile_product_v13,
    _resource_record,
    _sha256_json,
    _strict_sha_map,
    _valid_sha256,
    load_fragment_w3_trial_product_v13,
    load_immutable_w1_anchor_v13,
    logical_io_evidence_from_gate_v13,
    v13_implementation_lineage,
)
from .retrieval_fold_fused_sharded_oof_v12 import (
    ShardSpecV12,
    load_shard_build_manifest_v12,
    plan_query_shards,
    spawn_manifest_workers_v12,
)
from .retrieval_fold_fused_sharded_preflight_v12 import (
    PreparedShardedExecutionV12,
    prepare_sharded_execution_v12,
    require_external_sharded_output_v12,
)
from .retrieval_fold_fused_sharded_runtime_v12 import (
    RUNTIME_PAYLOAD_SCHEMA_VERSION,
)


SCHEMA_VERSION = "sireto-retrieval-v14-physical-benchmark-1"
TRIAL_WRAPPER_SCHEMA_VERSION = "sireto-retrieval-v14-sharded-run-1"
AUTHORIZATION_SCHEMA_VERSION = "sireto-retrieval-v14-national-authorization-1"
CONFIG_SCHEMA_VERSION = "sireto-retrieval-v14-physical-execution-1"
CONFIG_STATUS = "PREREGISTERED_V14_PHYSICAL_BENCHMARK_NOT_RUN"
FUNCTIONAL_PROJECTION_SCHEMA_VERSION = (
    "sireto-retrieval-v14-immutable-w1-functional-projection-1"
)
FUNCTIONAL_PROJECTION_STATUS = (
    "SEALED_IMMUTABLE_V12_W1_FUNCTIONAL_PROJECTION_FOR_V14"
)
ENGINE_ID = "FOLD_FUSED_FRAGMENT_W3_PHYSICAL_BENCHMARK_V14_1"
TRIAL_RUN_KIND = "PHYSICAL_BENCHMARK_W3_V14"
CONFIG_RELATIVE_PATH = "config/retrieval_all_human_v14_physical_execution.json"
CONFIG_SHA256 = "d39b347c971720a9e166c52bd2b5424544f56370108c7a76c52f5fe0df2d0b1c"
SAMPLE_QUERY_COUNT = 512
NATIONAL_QUERY_COUNT = 23_571
ALLOWED_FOLDS = tuple(TRAIN_FOLDS)
GIB = 1024**3

BUILD_ENTRYPOINT_V14 = (
    "src.xgb_matcher.retrieval_fold_fused_physical_benchmark_v14:"
    "build_worker_entrypoint_v14"
)
REPLAY_ENTRYPOINT_V14 = (
    "src.xgb_matcher.retrieval_fold_fused_physical_benchmark_v14:"
    "replay_worker_entrypoint_v14"
)

V14_IMPLEMENTATION_FILES = (
    CONFIG_RELATIVE_PATH,
    "docs/retrieval_all_human_v14_physical_execution_contract.md",
    "requirements.txt",
    "src/xgb_matcher/retrieval_fold_fused_logical_hash_v14.py",
    "src/xgb_matcher/retrieval_fold_fused_logical_hash_provider_v14.py",
    "src/xgb_matcher/retrieval_fold_fused_fragment_dataset_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_fragment_outputs_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_fragment_runtime_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_zero_copy_run_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_logical_io_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_physical_benchmark_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_national_runtime_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_physical_benchmark_v14.py",
    "src/xgb_matcher/retrieval_fold_fused_national_runtime_v14.py",
    "scripts/benchmark_retrieval_fold_fused_workers_v14.py",
    "scripts/run_retrieval_fold_fused_sharded_oof_v14.py",
)

_REQUIRED_LOGICAL_DIGESTS = (
    "heldout_spool",
    "train_outer2",
    "train_outer3",
    "train_outer4",
    "query_groups_outer2",
    "query_groups_outer3",
    "query_groups_outer4",
    "model_outer2",
    "model_outer3",
    "model_outer4",
    "prefit_bridge_logical",
    "site_feature_values",
    "prerank500_common",
    "prerank500_full_payload",
    "top100_full",
    "outcomes",
)


class PhysicalBenchmarkInvariantErrorV14(PhysicalBenchmarkInvariantErrorV13):
    """A v14 input, artifact or transition escaped the preregistration."""


class _PostOExclMarkerErrorV14(RuntimeError):
    """The process won O_EXCL, then failed while making the marker durable."""


@dataclass(frozen=True)
class ImmutableW1FunctionalProjectionV14:
    """V12 W1 content projected and rehashed under the sealed v14 protocol."""

    manifest_path: Path
    manifest_sha256: str
    projection_id: str
    anchor_trial_sha256: str
    prefit_parity_manifest_sha256: str
    v13_implementation_sha256: str
    v14_implementation_sha256: str
    logical_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            not self.manifest_path.is_file()
            or sha256_file(self.manifest_path) != self.manifest_sha256
            or any(
                not _valid_sha256(value)
                for value in (
                    self.manifest_sha256,
                    self.projection_id,
                    self.anchor_trial_sha256,
                    self.prefit_parity_manifest_sha256,
                    self.v13_implementation_sha256,
                    self.v14_implementation_sha256,
                )
            )
        ):
            raise PhysicalBenchmarkInvariantErrorV14(
                "v14 immutable W1 functional projection is invalid"
            )
        _strict_sha_map(
            self.logical_digests,
            expected_keys=_REQUIRED_LOGICAL_DIGESTS,
            label="v14 immutable W1 functional projection",
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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


def v14_implementation_lineage(
    workspace_root: Path,
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    """Seal every v14 outer/provider byte plus the reused v13 engine bytes."""

    root = Path(workspace_root).resolve()
    records: list[Mapping[str, Any]] = []
    for relative in V14_IMPLEMENTATION_FILES:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise PhysicalBenchmarkInvariantErrorV14(
                f"v14 implementation file is absent: {relative}"
            )
        digest = sha256_file(path)
        if relative == CONFIG_RELATIVE_PATH and digest != CONFIG_SHA256:
            raise PhysicalBenchmarkInvariantErrorV14(
                "v14 implementation embeds another physical config"
            )
        records.append(
            {"path": relative, "sha256": digest, "size_bytes": path.stat().st_size}
        )
    sealed = tuple(records)
    return _sha256_json(sealed), sealed


def _anchor_record_v14(anchor: ImmutableW1AnchorV13) -> Mapping[str, Any]:
    return {
        "benchmark_manifest_path": str(anchor.benchmark_manifest_path.resolve()),
        "benchmark_manifest_sha256": anchor.benchmark_manifest_sha256,
        "trial_manifest_path": str(anchor.trial_manifest_path.resolve()),
        "trial_manifest_sha256": anchor.trial_manifest_sha256,
        "wall_seconds": anchor.wall_seconds,
        "materialized_bytes": anchor.materialized_bytes,
        "query_count": anchor.query_count,
        "folds": list(anchor.folds),
    }


def _anchor_from_record_v14(value: Mapping[str, Any]) -> ImmutableW1AnchorV13:
    if set(value) != {
        "benchmark_manifest_path",
        "benchmark_manifest_sha256",
        "trial_manifest_path",
        "trial_manifest_sha256",
        "wall_seconds",
        "materialized_bytes",
        "query_count",
        "folds",
    }:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 immutable W1 anchor record drifted"
        )
    anchor = ImmutableW1AnchorV13(
        benchmark_manifest_path=Path(str(value["benchmark_manifest_path"])).resolve(),
        benchmark_manifest_sha256=str(value["benchmark_manifest_sha256"]),
        trial_manifest_path=Path(str(value["trial_manifest_path"])).resolve(),
        trial_manifest_sha256=str(value["trial_manifest_sha256"]),
        wall_seconds=float(value["wall_seconds"]),
        materialized_bytes=int(value["materialized_bytes"]),
        query_count=int(value["query_count"]),
        folds=tuple(int(item) for item in value["folds"]),
    )
    if (
        anchor.query_count != SAMPLE_QUERY_COUNT
        or anchor.folds != ALLOWED_FOLDS
        or not anchor.benchmark_manifest_path.is_file()
        or not anchor.trial_manifest_path.is_file()
        or sha256_file(anchor.benchmark_manifest_path)
        != anchor.benchmark_manifest_sha256
        or sha256_file(anchor.trial_manifest_path) != anchor.trial_manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 immutable W1 anchor bytes drifted"
        )
    return anchor


def _v14_projection_identity(
    *,
    anchor: ImmutableW1AnchorV13,
    prefit_parity_manifest_path: Path,
    prefit_parity_manifest_sha256: str,
    expected_prefit_lineage: Mapping[str, str],
    sample_query_ids: Sequence[str],
    v13_implementation_sha256: str,
    v13_implementation_files: Sequence[Mapping[str, Any]],
    v14_implementation_sha256: str,
    v14_implementation_files: Sequence[Mapping[str, Any]],
    logical_digests: Mapping[str, str],
    inputs: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "anchor": _anchor_record_v14(anchor),
        "prefit_parity_manifest_path": str(
            Path(prefit_parity_manifest_path).resolve()
        ),
        "prefit_parity_manifest_sha256": prefit_parity_manifest_sha256,
        "expected_prefit_lineage": dict(expected_prefit_lineage),
        "sample_query_ids": [str(value) for value in sample_query_ids],
        "v13_fragment_engine_implementation_sha256": (
            v13_implementation_sha256
        ),
        "v13_fragment_engine_implementation_files": list(
            v13_implementation_files
        ),
        "v14_implementation_sha256": v14_implementation_sha256,
        "v14_implementation_files": list(v14_implementation_files),
        "logical_hash_protocol": logical_hash_lineage_record(
            require_logical_hash_provider_v14(V14_LOGICAL_HASH_PROVIDER)
        ),
        "inputs": list(inputs),
        "logical_digests": dict(logical_digests),
    }


def build_immutable_w1_functional_projection_v14(
    *,
    anchor: ImmutableW1AnchorV13,
    workspace_root: Path,
    output_root: Path,
    prefit_parity_manifest_path: Path,
    expected_prefit_parity_manifest_sha256: str,
    expected_prefit_lineage: Mapping[str, str],
    sample_query_ids: Sequence[str],
) -> ImmutableW1FunctionalProjectionV14:
    """Build/reopen a v14-native W1 projection and recompute every root."""

    require_logical_hash_provider_v14(V14_LOGICAL_HASH_PROVIDER)
    root = Path(workspace_root).resolve()
    v13_sha, v13_files = v13_implementation_lineage(root)
    v14_sha, v14_files = v14_implementation_lineage(root)
    logical, inputs = _compute_w1_functional_projection(
        anchor,
        prefit_parity_manifest_path=Path(prefit_parity_manifest_path).resolve(),
        expected_prefit_parity_manifest_sha256=(
            expected_prefit_parity_manifest_sha256
        ),
        expected_prefit_lineage=expected_prefit_lineage,
        sample_query_ids=sample_query_ids,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )
    identity = _v14_projection_identity(
        anchor=anchor,
        prefit_parity_manifest_path=prefit_parity_manifest_path,
        prefit_parity_manifest_sha256=expected_prefit_parity_manifest_sha256,
        expected_prefit_lineage=expected_prefit_lineage,
        sample_query_ids=sample_query_ids,
        v13_implementation_sha256=v13_sha,
        v13_implementation_files=v13_files,
        v14_implementation_sha256=v14_sha,
        v14_implementation_files=v14_files,
        logical_digests=logical,
        inputs=inputs,
    )
    projection_id = _sha256_json(identity)
    projection_root = Path(output_root).resolve()
    # Use the full digest as the directory authority.  Prefix-addressed caches
    # leave a (tiny but avoidable) collision surface before the one-shot.
    target = projection_root / projection_id
    if target.exists():
        return load_immutable_w1_functional_projection_v14(
            target / "manifest.json",
            expected_manifest_sha256=sha256_file(target / "manifest.json"),
            expected_anchor_trial_sha256=anchor.trial_manifest_sha256,
            workspace_root=root,
            verify_inputs=True,
            recompute_logical=True,
        )
    projection_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".w1-functional-v14-", dir=projection_root))
    try:
        manifest = {
            "schema_version": FUNCTIONAL_PROJECTION_SCHEMA_VERSION,
            "status": FUNCTIONAL_PROJECTION_STATUS,
            "projection_id": projection_id,
            "identity": identity,
            "folds": list(ALLOWED_FOLDS),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            **identity,
        }
        _atomic_json(stage / "manifest.json", manifest)
        os.rename(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return load_immutable_w1_functional_projection_v14(
        target / "manifest.json",
        expected_manifest_sha256=sha256_file(target / "manifest.json"),
        expected_anchor_trial_sha256=anchor.trial_manifest_sha256,
        workspace_root=root,
        verify_inputs=True,
        recompute_logical=True,
    )


def load_immutable_w1_functional_projection_v14(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_anchor_trial_sha256: str | None,
    workspace_root: Path,
    verify_inputs: bool = True,
    recompute_logical: bool = True,
) -> ImmutableW1FunctionalProjectionV14:
    """Strict loader; cached manifests never substitute for recomputation."""

    path = Path(manifest_path).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or not _valid_sha256(expected_manifest_sha256)
        or sha256_file(path) != expected_manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 immutable W1 functional manifest bytes drifted"
        )
    raw = _read_json(path, label="v14 immutable W1 functional projection")
    identity = raw.get("identity")
    identity_keys = {
        "anchor",
        "prefit_parity_manifest_path",
        "prefit_parity_manifest_sha256",
        "expected_prefit_lineage",
        "sample_query_ids",
        "v13_fragment_engine_implementation_sha256",
        "v13_fragment_engine_implementation_files",
        "v14_implementation_sha256",
        "v14_implementation_files",
        "logical_hash_protocol",
        "inputs",
        "logical_digests",
    }
    expected_keys = {
        "schema_version",
        "status",
        "projection_id",
        "identity",
        "folds",
        "fold0_opened",
        "fold1_opened",
        "positive_injection",
        *identity_keys,
    }
    if (
        set(raw) != expected_keys
        or raw.get("schema_version") != FUNCTIONAL_PROJECTION_SCHEMA_VERSION
        or raw.get("status") != FUNCTIONAL_PROJECTION_STATUS
        or not isinstance(identity, Mapping)
        or set(identity) != identity_keys
        or any(raw.get(key) != identity.get(key) for key in identity_keys)
        or raw.get("projection_id") != _sha256_json(identity)
        or path.parent.name != str(raw.get("projection_id"))
        or raw.get("folds") != list(ALLOWED_FOLDS)
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or raw.get("logical_hash_protocol")
        != logical_hash_lineage_record(V14_LOGICAL_HASH_PROVIDER)
        or not isinstance(raw.get("inputs"), list)
        or not isinstance(raw.get("logical_digests"), Mapping)
        or not isinstance(raw.get("expected_prefit_lineage"), Mapping)
        or not isinstance(raw.get("sample_query_ids"), list)
        or len(raw["sample_query_ids"]) != SAMPLE_QUERY_COUNT
        or len(set(str(value) for value in raw["sample_query_ids"]))
        != SAMPLE_QUERY_COUNT
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 immutable W1 functional projection contract drifted"
        )
    anchor = _anchor_from_record_v14(raw["anchor"])
    if (
        expected_anchor_trial_sha256 is not None
        and anchor.trial_manifest_sha256 != expected_anchor_trial_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 immutable W1 trial authority drifted"
        )
    current_v13_sha, current_v13_files = v13_implementation_lineage(
        Path(workspace_root).resolve()
    )
    current_v14_sha, current_v14_files = v14_implementation_lineage(
        Path(workspace_root).resolve()
    )
    if (
        raw.get("v13_fragment_engine_implementation_sha256") != current_v13_sha
        or raw.get("v13_fragment_engine_implementation_files")
        != list(current_v13_files)
        or raw.get("v14_implementation_sha256") != current_v14_sha
        or raw.get("v14_implementation_files") != list(current_v14_files)
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 immutable W1 projection implementation lineage drifted"
        )
    inputs = raw["inputs"]
    for record in inputs:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"role", "path", "sha256", "size_bytes"}
            or not _valid_sha256(record.get("sha256"))
            or type(record.get("size_bytes")) is not int
        ):
            raise PhysicalBenchmarkInvariantErrorV14(
                "v14 immutable W1 projection input record drifted"
            )
        source = Path(str(record.get("path") or "")).resolve()
        if (
            not source.is_file()
            or source.stat().st_size != int(record["size_bytes"])
            or (verify_inputs and sha256_file(source) != record["sha256"])
        ):
            raise PhysicalBenchmarkInvariantErrorV14(
                "v14 immutable W1 projection input bytes drifted"
            )
    logical = _strict_sha_map(
        raw["logical_digests"],
        expected_keys=_REQUIRED_LOGICAL_DIGESTS,
        label="v14 immutable W1 functional projection",
    )
    if recompute_logical:
        recomputed, recomputed_inputs = _compute_w1_functional_projection(
            anchor,
            prefit_parity_manifest_path=Path(
                str(raw["prefit_parity_manifest_path"])
            ).resolve(),
            expected_prefit_parity_manifest_sha256=str(
                raw["prefit_parity_manifest_sha256"]
            ),
            expected_prefit_lineage={
                str(key): str(value)
                for key, value in raw["expected_prefit_lineage"].items()
            },
            sample_query_ids=tuple(str(value) for value in raw["sample_query_ids"]),
            logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
        )
        if recomputed != logical or recomputed_inputs != inputs:
            raise PhysicalBenchmarkInvariantErrorV14(
                "v14 cached W1 projection differs from exact recomputation"
            )
    return ImmutableW1FunctionalProjectionV14(
        manifest_path=path,
        manifest_sha256=expected_manifest_sha256,
        projection_id=str(raw["projection_id"]),
        anchor_trial_sha256=anchor.trial_manifest_sha256,
        prefit_parity_manifest_sha256=str(
            raw["prefit_parity_manifest_sha256"]
        ),
        v13_implementation_sha256=current_v13_sha,
        v14_implementation_sha256=current_v14_sha,
        logical_digests=logical,
    )


def _authority_path(root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    value = Path(str(record.get("path") or ""))
    path = value.resolve() if value.is_absolute() else (root / value).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or not _valid_sha256(record.get("sha256"))
        or sha256_file(path) != record["sha256"]
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            f"v14 authority bytes drifted: {label}"
        )
    return path


def load_physical_config_v14(path: Path) -> Mapping[str, Any]:
    """Load the exact v14 preregistration and its terminal v13 tombstone."""

    resolved = Path(path).resolve()
    if (
        not resolved.is_file()
        or CONFIG_SHA256 == "TO_BE_REPLACED_AFTER_PREREGISTRATION_FREEZE"
        or sha256_file(resolved) != CONFIG_SHA256
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 physical config differs from the preregistered bytes"
        )
    config = _read_json(resolved, label="v14 physical config")
    benchmark = config.get("benchmark_512")
    national = config.get("national_execution_protocol")
    tombstone = config.get("terminal_v13_tombstone")
    logical_hash = config.get("logical_hash_protocol")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != CONFIG_STATUS
        or config.get("revision_scope") != "PHYSICAL_HASH_PROTOCOL_ONLY"
        or config.get("scientific_contract_changed") is not False
        or config.get("quality_metric_tuning_allowed") is not False
        or config.get("national_oof_allowed") is not False
        or not isinstance(benchmark, Mapping)
        or benchmark.get("mode") != "ONE_FRESH_V14_W3_VS_IMMUTABLE_V12_W1"
        or benchmark.get("fresh_trial_count") != 1
        or benchmark.get("fresh_trials_allowed") != [3]
        or benchmark.get("sample_query_count") != SAMPLE_QUERY_COUNT
        or benchmark.get("canonical_output_dir")
        != "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_physical_benchmark_v14_w3"
        or benchmark.get("attempt_consumed_marker_path")
        != "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/.retrieval_v14_w3_attempt_consumed.json"
        or benchmark.get("automatic_retry_allowed") is not False
        or benchmark.get("v13_artifact_salvage_allowed") is not False
        or benchmark.get("rerun_w1_allowed") is not False
        or benchmark.get("w2_allowed") is not False
        or not isinstance(national, Mapping)
        or national.get("status") != "LOCKED_PENDING_V14_AUTHORIZATION"
        or national.get("canonical_output_dir")
        != "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_national_oof_v14"
        or national.get("attempt_consumed_marker_path")
        != "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/.retrieval_v14_national_attempt_consumed.json"
        or national.get("stage_directory_prefix")
        != ".retrieval_fold_fused_national_oof_v14.stage-"
        or national.get("v13_authorization_accepted") is not False
        or not isinstance(logical_hash, Mapping)
        or logical_hash.get("revision") != 14
        or logical_hash.get("algorithm_id")
        != LOGICAL_TABLE_HASH_ALGORITHM_ID_V14
        or logical_hash.get("algorithm_sha256")
        != V14_LOGICAL_HASH_PROVIDER.algorithm_sha256
        or logical_hash.get("record_schema_id")
        != LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14
        or logical_hash.get("callable_fingerprint_sha256")
        != logical_hash_lineage_record(V14_LOGICAL_HASH_PROVIDER)[
            "callable_fingerprint_sha256"
        ]
        or logical_hash.get("self_test_sha256")
        != logical_hash_lineage_record(V14_LOGICAL_HASH_PROVIDER)[
            "self_test_sha256"
        ]
        or logical_hash.get("ambient_global_switch_allowed") is not False
        or not isinstance(tombstone, Mapping)
        or tombstone.get("status") != "TERMINAL_STOP_V13_NO_RETRY"
        or tombstone.get("v13_salvage_allowed") is not False
        or tombstone.get("v13_retry_allowed") is not False
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 physical configuration semantic drift"
        )
    root = resolved.parent.parent
    authorities = config.get("authorities")
    if not isinstance(authorities, Mapping):
        raise PhysicalBenchmarkInvariantErrorV14("v14 authority set is absent")
    required = {
        "scientific_contract",
        "scientific_config",
        "fold_fused_addendum",
        "fold_fused_config",
        "v12_sharded_addendum",
        "v12_sharded_plan",
        "v12_stop_report",
        "v13_terminal_report",
        "v13_stop_receipt_builder",
        "v13_stop_receipt",
    }
    if set(authorities) != required:
        raise PhysicalBenchmarkInvariantErrorV14("v14 authority set drifted")
    for label, record in authorities.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise PhysicalBenchmarkInvariantErrorV14(
                f"v14 authority record drifted: {label}"
            )
        _authority_path(root, record, label=label)
    receipt = _read_json(
        _authority_path(root, authorities["v13_stop_receipt"], label="receipt"),
        label="v13 terminal stop receipt",
    )
    if (
        receipt.get("schema_version")
        != "sireto-retrieval-v13-terminal-stop-receipt-1"
        or receipt.get("status") != "SEALED_TERMINAL_STOP_V13_NO_RETRY"
        or receipt.get("receipt_id")
        != "bdcba90d29bd7da0e84fb805936fe0f99f8518ce9661a30751412c005f6d6799"
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v13 terminal receipt is not the final tombstone authority"
        )
    return config


def _frozen_gates_v14(config: Mapping[str, Any]) -> GateThresholds:
    scientific = config["scientific_invariants"]
    return GateThresholds(
        master_recall_at_2000=float(scientific["master_recall_at_2000"]),
        prerank_recall_at_500=float(scientific["prerank_recall_at_500"]),
        top100_recall=float(scientific["exact_recall_at_100"]),
        top100_wilson95_lower=float(
            scientific["exact_recall_at_100_wilson95_lower"]
        ),
        minimum_each_outer_fold_top100=float(
            scientific["minimum_each_outer_fold_exact_recall_at_100"]
        ),
    )


def _validate_prepared_authorities_v14(
    prepared: PreparedShardedExecutionV12,
    *,
    config: Mapping[str, Any],
    workspace_root: Path,
) -> None:
    """Bind the cache-neutral Prepared to every frozen v12 authority.

    This runs before projection work and O_EXCL, and is repeated after the
    measured callback.  It intentionally does not hash the 4.7-GiB source;
    the cache-neutral preflight validates its footer and the post-timer I/O
    gate performs the sole full-source verification.
    """

    root = Path(workspace_root).resolve()
    plan_authority = config["authorities"]["v12_sharded_plan"]
    plan_path = (root / str(plan_authority["path"])).resolve()
    if (
        not plan_path.is_file()
        or sha256_file(plan_path) != plan_authority["sha256"]
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 frozen sharded plan bytes drifted"
        )
    plan = _read_json(plan_path, label="v14 frozen sharded plan")
    inputs = plan.get("input_lineage") or {}
    scene = inputs.get("scene_ir") or {}
    labels = inputs.get("labels") or {}
    sidecars = inputs.get("official_sidecars") or {}
    parent = inputs.get("parent_model_bundle") or {}
    prefit = inputs.get("prefit_parity") or {}
    prefit_path = Path(str(prefit.get("manifest_path") or "")).resolve()
    prefit_sha = str(prefit.get("manifest_sha256") or "")
    expected_lineage = {
        "scene_ir_sha256": str(config["input_lineage"]["scene_ir_build_id"]),
        "scene_ir_manifest_sha256": str(
            config["input_lineage"]["scene_ir_manifest_sha256"]
        ),
        "source_candidate_sha256": str(
            config["input_lineage"]["source_candidate_sha256"]
        ),
        "source_manifest_sha256": str(
            config["input_lineage"]["source_manifest_sha256"]
        ),
        "label_ledger_sha256": str(
            config["input_lineage"]["label_ledger_sha256"]
        ),
        "label_ledger_manifest_sha256": str(
            config["input_lineage"]["label_ledger_manifest_sha256"]
        ),
        "scientific_config_sha256": str(
            config["authorities"]["scientific_config"]["sha256"]
        ),
        "executor_config_sha256": str(
            config["authorities"]["fold_fused_config"]["sha256"]
        ),
        "official_sidecar_manifest_sha256": str(
            config["input_lineage"]["official_sidecar_manifest_sha256"]
        ),
        "mmap_sidecar_manifest_sha256": str(
            config["input_lineage"]["mmap_sidecar_manifest_sha256"]
        ),
        "reader_parity_evidence_sha256": str(
            config["input_lineage"]["reader_parity_manifest_sha256"]
        ),
        "parent_model_bundle_sha256": str(
            config["input_lineage"]["parent_model_bundle_sha256"]
        ),
        "query_sample_sha256": str(
            config["input_lineage"]["runtime_query_sample_sha256"]
        ),
    }
    expected_paths = {
        "workspace_root": root,
        "scene_ir": Path(str(scene.get("directory") or "")).resolve(),
        "labels": Path(str(labels.get("ledger_path") or "")).resolve(),
        "sidecars": Path(
            str(sidecars.get("sidecar_manifest_path") or "")
        ).resolve().parent,
        "mmap_sidecar": Path(
            str(sidecars.get("mmap_manifest_path") or "")
        ).resolve().parent,
        "reader_parity_manifest": Path(
            str(sidecars.get("reader_parity_manifest_path") or "")
        ).resolve(),
        "parent_model_bundle": Path(
            str(parent.get("manifest_path") or "")
        ).resolve(),
        "scientific_config": (
            root / str(config["authorities"]["scientific_config"]["path"])
        ).resolve(),
        "executor_config": (
            root / str(config["authorities"]["fold_fused_config"]["path"])
        ).resolve(),
        "sharded_execution_plan": plan_path,
        "prefit_parity_manifest": prefit_path,
    }
    runtime = prepared.runtime_payload
    runtime_paths = runtime.get("paths") if isinstance(runtime, Mapping) else None
    runtime_keys = {
        "schema_version",
        "paths",
        "sharded_execution_plan_sha256",
        "prefit_parity_manifest_sha256",
        "lineage",
        "replay_generation_inputs_opened",
        "fold0_opened",
        "fold1_opened",
    }
    scientific_path = expected_paths["scientific_config"]
    executor_path = expected_paths["executor_config"]
    expected_admission = AdmissionConfigV12.load(scientific_path)
    expected_cascade = RetrievalTop100CascadeV12Config.load(scientific_path)
    expected_executor = load_executor_config_v12(
        executor_path,
        scientific_config_path=scientific_path,
        workspace_root=root,
    )
    expected_caps = ExecutionCaps(
        fit_sample=int(
            expected_admission.fit_candidate_sampling[
                "maximum_candidates_per_query"
            ]
        )
    )
    canonical_scene = SceneIRCache(
        expected_paths["scene_ir"], verify_source_hash=False
    )
    scene_query_ids = tuple(
        str(row["query_id"]) for row in canonical_scene.index.to_pylist()
    )
    sample_path = Path(
        str(config["input_lineage"]["runtime_query_sample_path"])
    ).resolve()
    if (
        not sample_path.is_file()
        or sha256_file(sample_path)
        != config["input_lineage"]["runtime_query_sample_sha256"]
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 sealed runtime sample bytes drifted"
        )
    sample_table = pq.read_table(sample_path)
    sealed_sample_ids = tuple(
        str(value) for value in sample_table["query_id"].to_pylist()
    )
    if (
        not prefit_path.is_file()
        or sha256_file(prefit_path) != prefit_sha
        or prepared.scene_cache.source.resolve()
        != Path(str(config["input_lineage"]["source_candidate_path"])).resolve()
        or prepared.scene_cache.cache_dir.resolve() != canonical_scene.cache_dir
        or prepared.scene_cache.source.resolve() != canonical_scene.source
        or dict(prepared.scene_cache.manifest) != dict(canonical_scene.manifest)
        or not prepared.scene_cache.index.equals(
            canonical_scene.index, check_metadata=True
        )
        or prepared.sharded_execution_plan_path.resolve() != plan_path
        or prepared.sharded_execution_plan_sha256 != plan_authority["sha256"]
        or prepared.prefit_parity_manifest_path.resolve() != prefit_path
        or prepared.prefit_parity_manifest_sha256 != prefit_sha
        or dict(prepared.lineage) != expected_lineage
        or prepared.gates != _frozen_gates_v14(config)
        or prepared.admission != expected_admission
        or prepared.cascade != expected_cascade
        or prepared.executor != expected_executor
        or prepared.caps != expected_caps
        or sample_table.column_names != ["query_id", "fold"]
        or sample_table.num_rows != SAMPLE_QUERY_COUNT
        or config["input_lineage"]["runtime_query_sample_rows"]
        != SAMPLE_QUERY_COUNT
        or tuple(prepared.sample_query_ids) != sealed_sample_ids
        or len(set(prepared.sample_query_ids)) != SAMPLE_QUERY_COUNT
        or len(prepared.national_query_ids) != NATIONAL_QUERY_COUNT
        or tuple(prepared.national_query_ids) != scene_query_ids
        or not isinstance(runtime_paths, Mapping)
        or set(runtime) != runtime_keys
        or runtime.get("schema_version") != RUNTIME_PAYLOAD_SCHEMA_VERSION
        or set(runtime_paths) != set(expected_paths)
        or any(
            Path(str(runtime_paths[name])).resolve() != expected
            for name, expected in expected_paths.items()
        )
        or dict(runtime.get("lineage") or {}) != expected_lineage
        or runtime.get("sharded_execution_plan_sha256")
        != plan_authority["sha256"]
        or runtime.get("prefit_parity_manifest_sha256") != prefit_sha
        or runtime.get("replay_generation_inputs_opened") is not False
        or runtime.get("fold0_opened") is not False
        or runtime.get("fold1_opened") is not False
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 Prepared differs from frozen source/plan/PREFIT authorities"
        )


def build_worker_entrypoint_v14(job: Mapping[str, Any]) -> str:
    """V14 build actor: v13 science/sink with the v14 config authority."""

    return build_worker_entrypoint_v13(
        job,
        physical_config_loader=load_physical_config_v14,
        physical_config_relative_path=CONFIG_RELATIVE_PATH,
    )


def replay_worker_entrypoint_v14(job: Mapping[str, Any]) -> str:
    """V14 replay actor with the explicit null-canonical hash provider."""

    require_logical_hash_provider_v14(V14_LOGICAL_HASH_PROVIDER)
    return replay_worker_entrypoint_v13(
        job,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
    )


@dataclass(frozen=True)
class _OfficialCapabilityV14:
    token: object
    config_sha256: str
    output_dir: Path
    marker_path: Path


_CAPABILITY_TOKEN_V14 = object()


@dataclass(frozen=True)
class _OfficialDAGV14:
    prepared: PreparedShardedExecutionV12
    workspace_root: Path
    w1_functional_projection: Any
    w1_io_projection: Any
    capability: _OfficialCapabilityV14


@dataclass(frozen=True)
class PhysicalBenchmarkAssessmentV14:
    config_path: Path
    output: Path
    marker: Path
    marker_sha256: str
    anchor: ImmutableW1AnchorV13
    product: FragmentW3TrialProductV13
    resources: PhysicalResourceMetricsV12
    logical: LogicalIOEvidenceV13
    checks: Mapping[str, bool]
    speedup: float
    v14_implementation_sha256: str
    v14_implementation_files: tuple[Mapping[str, Any], ...]
    official_capability: object | None = None


def _benchmark_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    benchmark = config["benchmark_512"]
    return (
        Path(str(benchmark["canonical_output_dir"])).resolve(),
        Path(str(benchmark["attempt_consumed_marker_path"])).resolve(),
    )


def _marker_payload(config: Mapping[str, Any]) -> Mapping[str, Any]:
    output, marker = _benchmark_paths(config)
    return {
        "schema_version": "sireto-retrieval-v14-one-shot-attempt-1",
        "status": "ATTEMPT_IRREVERSIBLY_CONSUMED_BEFORE_CALLBACK",
        "config_sha256": CONFIG_SHA256,
        "canonical_output_dir": str(output),
        "marker_path": str(marker),
        "v13_retry_or_salvage": False,
        "retry_allowed": False,
    }


def _consume_attempt(config: Mapping[str, Any], output_dir: Path) -> tuple[Path, str]:
    output, marker = _benchmark_paths(config)
    if (
        Path(output_dir).resolve() != output
        or output.exists()
        or marker.exists()
        or marker.parent != output.parent
        or not output.parent.is_dir()
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 canonical W3 output is unavailable"
        )
    payload = (
        json.dumps(_marker_payload(config), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 W3 one-shot attempt was already consumed"
        ) from error
    try:
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(marker.parent)
        digest = sha256_file(marker)
        if marker.stat().st_nlink != 1:
            raise PhysicalBenchmarkInvariantErrorV14(
                "v14 W3 attempt marker is hardlinked"
            )
        return marker, digest
    except BaseException as error:
        raise _PostOExclMarkerErrorV14(
            "v14 marker failed after O_EXCL"
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
    config: Mapping[str, Any], marker: Path, expected_sha256: str
) -> None:
    output, expected_marker = _benchmark_paths(config)
    path = Path(marker).resolve()
    if (
        path != expected_marker
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or sha256_file(path) != expected_sha256
        or _read_json(path, label="v14 attempt marker") != _marker_payload(config)
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 one-shot capability drifted"
        )


def _publish_exception_stop(
    *,
    output: Path,
    config_path: Path,
    marker: Path,
    frozen_sha256: str,
    frozen_files: Sequence[Mapping[str, Any]],
    phase: str,
    error: BaseException,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    marker_sha = _observed_marker_sha256(marker)
    value = {
        "schema_version": SCHEMA_VERSION,
        "status": "STOP_V14_EXECUTION_EXCEPTION",
        "run_kind": "PHYSICAL_ONLY_W3_VS_IMMUTABLE_V12_W1",
        "config": {"path": str(config_path), "sha256": CONFIG_SHA256},
        "terminal_v13_tombstone_preserved": True,
        "v13_artifact_salvaged": False,
        "v14_implementation_sha256": frozen_sha256,
        "implementation_files": list(frozen_files),
        "attempt_marker": {"path": str(marker), "sha256": marker_sha},
        "failed_phase": phase,
        "canonical_output_dir": str(output),
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "execution_authorization_published": False,
        "automatic_retry_allowed": False,
        "exception": {
            "type": type(error).__name__,
            "message_sanitized": re.sub(r"\s+", " ", str(error)).strip()[:512],
        },
    }
    value["stop_id"] = _sha256_json(value)
    path = output / "manifest.json"
    _atomic_json(path, value)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(output)
    _fsync_directory(output.parent)
    return path


def execute_fragment_w3_trial_v14(
    *,
    prepared: PreparedShardedExecutionV12,
    workspace_root: Path,
    output_dir: Path,
    anchor: ImmutableW1AnchorV13,
    w1_functional_projection: Any,
    capability: _OfficialCapabilityV14,
) -> FragmentW3TrialProductV13:
    """Execute exactly three fresh fold-owned actors under the v14 capability."""

    root = Path(workspace_root).resolve()
    config = load_physical_config_v14(root / CONFIG_RELATIVE_PATH)
    canonical, marker = _benchmark_paths(config)
    if (
        capability.token is not _CAPABILITY_TOKEN_V14
        or capability.config_sha256 != CONFIG_SHA256
        or capability.output_dir != canonical
        or capability.marker_path != marker
        or Path(output_dir).resolve() != canonical / "trial_w3"
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "direct v14 W3 execution lacks the private one-shot capability"
        )
    marker_sha = sha256_file(marker)
    _validate_attempt(config, marker, marker_sha)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    lineage = dict(
        sharded_execution_lineage_v12(dict(prepared.lineage), workspace_root=root)
    )
    engine_sha, _engine_files = v13_implementation_lineage(root)
    lineage["v13_implementation_sha256"] = engine_sha
    specs = plan_query_shards(prepared.scene_cache.index, prepared.sample_query_ids, 3)
    if (
        len(specs) != 3
        or tuple(value.folds for value in specs) != ((2,), (3,), (4,))
        or sum(value.query_count for value in specs) != SAMPLE_QUERY_COUNT
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 W3 sample is not exactly three fold-owned shards"
        )
    jobs = _build_jobs_v13(
        prepared=prepared,
        specs=specs,
        output_dir=output,
        lineage=lineage,
        execution_phase="PHYSICAL_V14_W3_ONLY_BENCHMARK_512",
        physical_config_sha256=CONFIG_SHA256,
    )
    manifests = spawn_manifest_workers_v12(
        BUILD_ENTRYPOINT_V14, jobs, max_workers=3
    )
    builds = tuple(load_shard_build_manifest_v12(path) for path in manifests)
    if tuple(value.spec.digest for value in builds) != tuple(
        value.digest for value in specs
    ):
        raise PhysicalBenchmarkInvariantErrorV14("v14 W3 build set drifted")
    return finish_fragment_w3_trial_v13(
        prepared=prepared,
        specs=specs,
        build_artifacts=builds,
        lineage=lineage,
        output_dir=output,
        anchor=anchor,
        workspace_root=root,
        w1_functional_projection=w1_functional_projection,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
        physical_config_loader=load_physical_config_v14,
        physical_config_relative_path=CONFIG_RELATIVE_PATH,
        physical_config_sha256=CONFIG_SHA256,
        replay_entrypoint=REPLAY_ENTRYPOINT_V14,
        wrapper_schema_version=TRIAL_WRAPPER_SCHEMA_VERSION,
        wrapper_run_kind=TRIAL_RUN_KIND,
        functional_projection_loader=partial(
            load_immutable_w1_functional_projection_v14,
            workspace_root=root,
            recompute_logical=False,
        ),
    )


def _decision_status(checks: Mapping[str, bool]) -> str:
    precedence = (
        ("logical_bridge_exact", "STOP_V14_PARITY"),
        ("no_physical_merge_files", "STOP_V14_ARTIFACT_PROTOCOL"),
        ("top500_compact_top100_full", "STOP_V14_PAYLOAD_PROTOCOL"),
        ("compact_rehydration_exact", "STOP_V14_PARITY"),
        ("logical_io_pass", "STOP_V14_IO"),
        ("ram_pass", "STOP_V14_RAM"),
        ("swap_pass", "STOP_V14_SWAP"),
        ("speed_pass", "STOP_V14_SPEED"),
        ("storage_pass", "STOP_V14_STORAGE"),
    )
    for key, status in precedence:
        if checks.get(key) is not True:
            return status
    return "PASS_V14_PHYSICAL_BENCHMARK"


def _benchmark_identity(assessment: PhysicalBenchmarkAssessmentV14) -> Mapping[str, Any]:
    product = assessment.product
    return {
        "engine_id": ENGINE_ID,
        "config_sha256": CONFIG_SHA256,
        "v14_implementation_sha256": assessment.v14_implementation_sha256,
        "v13_fragment_engine_implementation_sha256": product.v13_implementation_sha256,
        "logical_hash_algorithm_sha256": V14_LOGICAL_HASH_PROVIDER.algorithm_sha256,
        "v13_terminal_receipt_sha256": (
            load_physical_config_v14(assessment.config_path)["authorities"]
            ["v13_stop_receipt"]["sha256"]
        ),
        "immutable_w1_benchmark_sha256": assessment.anchor.benchmark_manifest_sha256,
        "immutable_w1_trial_sha256": assessment.anchor.trial_manifest_sha256,
        "w3_trial_manifest_sha256": product.manifest_sha256,
        "logical_io_manifest_sha256": assessment.logical.manifest_sha256,
        "logical_digests": dict(product.logical_digests),
        "artifact_inventory_sha256": product.artifact_inventory_sha256,
        "fragment_dataset_manifest_sha256": product.fragment_dataset_manifest_sha256,
        "replay_dataset_manifest_sha256": product.replay_dataset_manifest_sha256,
        "sharded_implementation_sha256": product.lineage[
            "sharded_implementation_sha256"
        ],
        "runtime_environment_sha256": product.lineage[
            "runtime_environment_sha256"
        ],
        "sharded_execution_plan_sha256": product.sharded_execution_plan_sha256,
        "prefit_parity_manifest_sha256": product.prefit_parity_manifest_sha256,
        "attempt_marker_path": str(assessment.marker),
        "attempt_marker_sha256": assessment.marker_sha256,
        "query_count": SAMPLE_QUERY_COUNT,
        "worker_count": 3,
        "folds": list(ALLOWED_FOLDS),
    }


def _benchmark_manifest(
    assessment: PhysicalBenchmarkAssessmentV14,
    *,
    status: str,
) -> Mapping[str, Any]:
    identity = _benchmark_identity(assessment)
    passed = status == "PASS_V14_PHYSICAL_BENCHMARK"
    product = assessment.product
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "benchmark_id": _sha256_json(identity),
        "benchmark_identity": identity,
        "run_kind": "PHYSICAL_ONLY_W3_VS_IMMUTABLE_V12_W1",
        "config": {"path": str(assessment.config_path), "sha256": CONFIG_SHA256},
        "terminal_v13_tombstone_preserved": True,
        "v13_artifact_salvaged": False,
        "immutable_w1": {
            "benchmark_manifest_path": str(assessment.anchor.benchmark_manifest_path),
            "benchmark_manifest_sha256": assessment.anchor.benchmark_manifest_sha256,
            "trial_manifest_path": str(assessment.anchor.trial_manifest_path),
            "trial_manifest_sha256": assessment.anchor.trial_manifest_sha256,
            "wall_seconds": assessment.anchor.wall_seconds,
            "materialized_bytes": assessment.anchor.materialized_bytes,
        },
        "w3": {
            "trial_manifest_path": str(product.manifest_path),
            "trial_manifest_sha256": product.manifest_sha256,
            "v13_fragment_engine_implementation_sha256": product.v13_implementation_sha256,
            "v14_implementation_sha256": assessment.v14_implementation_sha256,
            "logical_hash_protocol": logical_hash_lineage_record(
                V14_LOGICAL_HASH_PROVIDER
            ),
            "lineage": dict(product.lineage),
            "logical_digests": dict(product.logical_digests),
            "materialized_bytes": product.materialized_bytes,
            "projected_national_storage_bytes": product.projected_national_storage_bytes,
            "resources": _resource_record(assessment.resources),
            "speedup_over_immutable_w1": assessment.speedup,
            "compact_rehydration": {
                "status": "PASS_ONE_INDEPENDENT_POST_TIMING_RECONSTRUCTION",
                "exact": product.compact_rehydration_exact is True,
                "site_feature_values": product.logical_digests[
                    "site_feature_values"
                ],
                "prerank500_full_payload": product.logical_digests[
                    "prerank500_full_payload"
                ],
            },
        },
        "attempt_marker": {
            "path": str(assessment.marker),
            "sha256": assessment.marker_sha256,
        },
        "non_circular_w1_bridge": {
            "reference_logical_digests": dict(product.immutable_w1_logical_digests),
            "w3_logical_digests": dict(product.logical_digests),
            "exact": product.bridge_exact,
            "sole_reference": "IMMUTABLE_V12_W1",
        },
        "logical_io": _logical_io_record(assessment.logical),
        "checks": dict(assessment.checks),
        "gate_pass": passed,
        "national_oof_launched": False,
        "execution_authorization_published": passed,
    }


def _authorization_document(
    assessment: PhysicalBenchmarkAssessmentV14,
    benchmark_sha256: str,
) -> Mapping[str, Any]:
    if assessment.official_capability is not _CAPABILITY_TOKEN_V14:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 authorization lacks official DAG provenance"
        )
    identity = {
        "engine_id": ENGINE_ID,
        "benchmark_manifest_sha256": benchmark_sha256,
        "config_sha256": CONFIG_SHA256,
        "v14_implementation_sha256": assessment.v14_implementation_sha256,
        "v13_fragment_engine_implementation_sha256": (
            assessment.product.v13_implementation_sha256
        ),
        "logical_hash_algorithm_sha256": V14_LOGICAL_HASH_PROVIDER.algorithm_sha256,
        "w3_trial_manifest_sha256": assessment.product.manifest_sha256,
        "logical_io_manifest_sha256": assessment.logical.manifest_sha256,
        "attempt_marker_path": str(assessment.marker),
        "attempt_marker_sha256": assessment.marker_sha256,
        "sharded_implementation_sha256": assessment.product.lineage[
            "sharded_implementation_sha256"
        ],
        "runtime_environment_sha256": assessment.product.lineage[
            "runtime_environment_sha256"
        ],
        "sharded_execution_plan_sha256": (
            assessment.product.sharded_execution_plan_sha256
        ),
        "prefit_parity_manifest_sha256": (
            assessment.product.prefit_parity_manifest_sha256
        ),
        "authorized_worker_count": 3,
        "authorized_scope": "NATIONAL_OOF_FOLDS_2_3_4_V14_ONLY",
    }
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": "AUTHORIZED_V14_W3_FOR_NATIONAL_OOF",
        "authorization_id": _sha256_json(identity),
        "authorization_identity": identity,
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "v13_consumer_allowed": False,
        "automatic_national_launch": False,
    }


def _assess_official(
    *,
    config_path: Path,
    output: Path,
    marker: Path,
    marker_sha256: str,
    dag: _OfficialDAGV14,
    frozen_sha256: str,
    frozen_files: tuple[Mapping[str, Any], ...],
) -> PhysicalBenchmarkAssessmentV14:
    config = load_physical_config_v14(config_path)
    _validate_prepared_authorities_v14(
        dag.prepared, config=config, workspace_root=dag.workspace_root
    )
    _validate_attempt(config, marker, marker_sha256)
    anchor = load_immutable_w1_anchor_v13(config)
    output.mkdir()

    def callback() -> FragmentW3TrialProductV13:
        return execute_fragment_w3_trial_v14(
            prepared=dag.prepared,
            workspace_root=dag.workspace_root,
            output_dir=output / "trial_w3",
            anchor=anchor,
            w1_functional_projection=dag.w1_functional_projection,
            capability=dag.capability,
        )

    claimed, resources = measure_physical_trial_v12(callback)
    expected_trial = output / "trial_w3" / "manifest.json"
    if Path(claimed.manifest_path).resolve() != expected_trial:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 measured callback returned an external trial"
        )
    product = load_fragment_w3_trial_product_v13(
        expected_trial,
        workspace_root=dag.workspace_root,
        expected_manifest_sha256=claimed.manifest_sha256,
        expected_anchor_trial_sha256=anchor.trial_manifest_sha256,
        verify_logical=True,
        # The measured finisher seals the replay-native proof but deliberately
        # does not rebuild the compact top500.  This is the sole independent
        # reconstruction and therefore must remain after the W3 timer.
        verify_rehydration=True,
        expected_wrapper_schema_version=TRIAL_WRAPPER_SCHEMA_VERSION,
        expected_wrapper_run_kind=TRIAL_RUN_KIND,
        physical_config_loader=load_physical_config_v14,
        physical_config_relative_path=CONFIG_RELATIVE_PATH,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
        functional_projection_loader=partial(
            load_immutable_w1_functional_projection_v14,
            workspace_root=dag.workspace_root,
            recompute_logical=False,
        ),
    )
    _reconcile_product_v13(claimed, product)
    _validate_prepared_authorities_v14(
        dag.prepared, config=config, workspace_root=dag.workspace_root
    )
    plan_authority = config["authorities"]["v12_sharded_plan"]
    plan_raw = _read_json(
        (dag.workspace_root / str(plan_authority["path"])).resolve(),
        label="v14 post-timer frozen sharded plan",
    )
    frozen_prefit_sha = str(
        (((plan_raw.get("input_lineage") or {}).get("prefit_parity") or {}).get(
            "manifest_sha256"
        ))
        or ""
    )
    if (
        product.sharded_execution_plan_sha256 != plan_authority["sha256"]
        or product.prefit_parity_manifest_sha256 != frozen_prefit_sha
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 measured W3 plan/PREFIT differ from frozen authorities"
        )
    current_sha, current_files = v14_implementation_lineage(dag.workspace_root)
    if current_sha != frozen_sha256 or current_files != frozen_files:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 implementation changed during the one-shot"
        )
    io_artifact = compare_w3_to_immutable_w1_io_v13(
        w1_projection_manifest_path=dag.w1_io_projection.manifest_path,
        w1_projection_manifest_sha256=dag.w1_io_projection.manifest_sha256,
        w3_run_manifest_path=product.manifest_path,
        w3_run_manifest_sha256=product.manifest_sha256,
        output_root=output / "logical_io",
        expected_config_schema_version=CONFIG_SCHEMA_VERSION,
        expected_config_status=CONFIG_STATUS,
        expected_projection_drift_policy="STOP_V14_IO_BEFORE_W3",
        expected_revision_scope="PHYSICAL_HASH_PROTOCOL_ONLY",
        expected_recipe_guard_field="scientific_contract_changed",
        expected_wrapper_schema_version=TRIAL_WRAPPER_SCHEMA_VERSION,
        expected_wrapper_run_kind=TRIAL_RUN_KIND,
    )
    logical = logical_io_evidence_from_gate_v13(io_artifact)
    checks, speedup = _derive_physical_checks_v13(
        config=config,
        anchor=anchor,
        product=product,
        logical=logical,
        resources=resources,
        compact_rehydration_exact=product.compact_rehydration_exact is True,
    )
    return PhysicalBenchmarkAssessmentV14(
        config_path=config_path,
        output=output,
        marker=marker,
        marker_sha256=marker_sha256,
        anchor=anchor,
        product=product,
        resources=resources,
        logical=logical,
        checks=checks,
        speedup=speedup,
        v14_implementation_sha256=frozen_sha256,
        v14_implementation_files=frozen_files,
        official_capability=_CAPABILITY_TOKEN_V14,
    )


def run_physical_benchmark_v14(
    *,
    config_path: Path,
    output_dir: Path,
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
    immutable_projection_root: Path,
) -> Path:
    """Run the sole authorization-capable v14 W3 benchmark; never national."""

    config_resolved = Path(config_path).resolve()
    root = Path(workspace_root).resolve()
    config = load_physical_config_v14(config_resolved)
    if root != config_resolved.parent.parent:
        raise PhysicalBenchmarkInvariantErrorV14("v14 workspace drifted")
    prepared = prepare_sharded_execution_v12(
        workspace_root=root,
        scene_ir=Path(scene_ir).resolve(),
        query_sample=Path(query_sample).resolve(),
        labels=Path(labels).resolve(),
        sidecars=Path(sidecars).resolve(),
        mmap_sidecar=Path(mmap_sidecar).resolve(),
        reader_parity_manifest=Path(reader_parity_manifest).resolve(),
        parent_model_bundle=Path(parent_model_bundle).resolve(),
        scientific_config=Path(scientific_config).resolve(),
        executor_config=Path(executor_config).resolve(),
        prefit_parity_manifest=Path(prefit_parity_manifest).resolve(),
        sharded_execution_plan=Path(sharded_execution_plan).resolve(),
        verify_source_hash=False,
    )
    _validate_prepared_authorities_v14(
        prepared, config=config, workspace_root=root
    )
    output = require_external_sharded_output_v12(
        output_dir, executor=prepared.executor
    )
    canonical, marker_path = _benchmark_paths(config)
    if output != canonical:
        raise PhysicalBenchmarkInvariantErrorV14("v14 output is noncanonical")
    anchor = load_immutable_w1_anchor_v13(config)
    support_root = Path(immutable_projection_root).resolve()
    expected_projection_root = Path(
        str(config["immutable_w1_projection"]["canonical_root"])
    ).resolve()
    if support_root != expected_projection_root:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 immutable projection root is noncanonical"
        )
    functional = build_immutable_w1_functional_projection_v14(
        anchor=anchor,
        workspace_root=root,
        output_root=support_root / "functional_v14",
        prefit_parity_manifest_path=prepared.prefit_parity_manifest_path,
        expected_prefit_parity_manifest_sha256=prepared.prefit_parity_manifest_sha256,
        expected_prefit_lineage=prepared.lineage,
        sample_query_ids=prepared.sample_query_ids,
    )
    built_io = build_immutable_w1_io_projection_v13(
        physical_config_path=config_resolved,
        physical_config_sha256=CONFIG_SHA256,
        output_root=support_root / "logical_io_v14",
        expected_config_schema_version=CONFIG_SCHEMA_VERSION,
        expected_config_status=CONFIG_STATUS,
        expected_projection_drift_policy="STOP_V14_IO_BEFORE_W3",
        expected_revision_scope="PHYSICAL_HASH_PROTOCOL_ONLY",
        expected_recipe_guard_field="scientific_contract_changed",
    )
    io_projection, _raw, _rows, _baseline = load_immutable_w1_io_projection_v13(
        built_io.manifest_path,
        expected_manifest_sha256=built_io.manifest_sha256,
        verify_source_hash=False,
        expected_config_schema_version=CONFIG_SCHEMA_VERSION,
        expected_config_status=CONFIG_STATUS,
        expected_projection_drift_policy="STOP_V14_IO_BEFORE_W3",
        expected_revision_scope="PHYSICAL_HASH_PROTOCOL_ONLY",
        expected_recipe_guard_field="scientific_contract_changed",
    )
    frozen_sha, frozen_files = v14_implementation_lineage(root)
    try:
        marker, marker_sha = _consume_attempt(config, output)
    except _PostOExclMarkerErrorV14 as error:
        _publish_exception_stop(
            output=output,
            config_path=config_resolved,
            marker=marker_path,
            frozen_sha256=frozen_sha,
            frozen_files=frozen_files,
            phase="ATOMIC_ATTEMPT_MARKER_CREATION",
            error=error,
        )
        raise
    capability = _OfficialCapabilityV14(
        token=_CAPABILITY_TOKEN_V14,
        config_sha256=CONFIG_SHA256,
        output_dir=output,
        marker_path=marker,
    )
    dag = _OfficialDAGV14(
        prepared=prepared,
        workspace_root=root,
        w1_functional_projection=functional,
        w1_io_projection=io_projection,
        capability=capability,
    )
    try:
        assessment = _assess_official(
            config_path=config_resolved,
            output=output,
            marker=marker,
            marker_sha256=marker_sha,
            dag=dag,
            frozen_sha256=frozen_sha,
            frozen_files=frozen_files,
        )
        status = _decision_status(assessment.checks)
        manifest_path = output / "manifest.json"
        _atomic_json(manifest_path, _benchmark_manifest(assessment, status=status))
        if status == "PASS_V14_PHYSICAL_BENCHMARK":
            _atomic_json(
                output / "execution_authorization_v14.json",
                _authorization_document(
                    assessment, benchmark_sha256=sha256_file(manifest_path)
                ),
            )
        return manifest_path
    except BaseException as error:
        _publish_exception_stop(
            output=output,
            config_path=config_resolved,
            marker=marker,
            frozen_sha256=frozen_sha,
            frozen_files=frozen_files,
            phase="MEASURED_W3_CALLBACK_OR_POSTCHECK",
            error=error,
        )
        raise


def _resources_from_record(value: Mapping[str, Any]) -> PhysicalResourceMetricsV12:
    try:
        return PhysicalResourceMetricsV12(**dict(value))
    except (TypeError, ValueError) as error:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 resource record drifted"
        ) from error


def load_execution_authorization_v14(
    path: Path,
    *,
    expected_authorization_sha256: str,
    benchmark_manifest_path: Path,
    expected_benchmark_manifest_sha256: str,
    workspace_root: Path,
) -> Mapping[str, Any]:
    """Reopen authorization, PASS benchmark, marker and all measured gates."""

    root = Path(workspace_root).resolve()
    auth_path = Path(path).resolve()
    benchmark_path = Path(benchmark_manifest_path).resolve()
    if (
        not auth_path.is_file()
        or sha256_file(auth_path) != expected_authorization_sha256
        or not benchmark_path.is_file()
        or sha256_file(benchmark_path) != expected_benchmark_manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 authorization or benchmark bytes drifted"
        )
    auth = _read_json(auth_path, label="v14 national authorization")
    benchmark = _read_json(benchmark_path, label="v14 physical benchmark")
    if (
        auth.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION
        or auth.get("status") != "AUTHORIZED_V14_W3_FOR_NATIONAL_OOF"
        or auth.get("v13_consumer_allowed") is not False
        or auth.get("automatic_national_launch") is not False
        or auth.get("authorization_id")
        != _sha256_json(auth.get("authorization_identity"))
        or benchmark.get("schema_version") != SCHEMA_VERSION
        or benchmark.get("status") != "PASS_V14_PHYSICAL_BENCHMARK"
        or benchmark.get("gate_pass") is not True
        or benchmark.get("execution_authorization_published") is not True
        or benchmark.get("national_oof_launched") is not False
        or benchmark.get("v13_artifact_salvaged") is not False
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 authorization/benchmark contract drifted"
        )
    config_path = Path(str((benchmark.get("config") or {}).get("path") or ""))
    config = load_physical_config_v14(config_path)
    canonical_output, canonical_marker = _benchmark_paths(config)
    anchor = load_immutable_w1_anchor_v13(config)
    expected_immutable = {
        "benchmark_manifest_path": str(anchor.benchmark_manifest_path),
        "benchmark_manifest_sha256": anchor.benchmark_manifest_sha256,
        "trial_manifest_path": str(anchor.trial_manifest_path),
        "trial_manifest_sha256": anchor.trial_manifest_sha256,
        "wall_seconds": anchor.wall_seconds,
        "materialized_bytes": anchor.materialized_bytes,
    }
    plan_record = config["authorities"]["v12_sharded_plan"]
    plan_path = (root / str(plan_record["path"])).resolve()
    plan_raw = _read_json(plan_path, label="v14 authorized sharded plan")
    frozen_prefit_sha = str(
        (((plan_raw.get("input_lineage") or {}).get("prefit_parity") or {}).get(
            "manifest_sha256"
        ))
        or ""
    )
    immutable_record = benchmark.get("immutable_w1")
    marker_record = benchmark.get("attempt_marker")
    if (
        benchmark_path != canonical_output / "manifest.json"
        or auth_path != canonical_output / "execution_authorization_v14.json"
        or dict(immutable_record or {}) != expected_immutable
        or not isinstance(marker_record, Mapping)
        or Path(str(marker_record.get("path") or "")).resolve()
        != canonical_marker
        or (auth.get("authorization_identity") or {}).get(
            "sharded_execution_plan_sha256"
        )
        != plan_record["sha256"]
        or (auth.get("authorization_identity") or {}).get(
            "prefit_parity_manifest_sha256"
        )
        != frozen_prefit_sha
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 authorization no longer binds canonical paths/W1 authorities"
        )
    current_sha, current_files = v14_implementation_lineage(root)
    identity = auth.get("authorization_identity")
    benchmark_identity = benchmark.get("benchmark_identity")
    if not isinstance(identity, Mapping) or not isinstance(benchmark_identity, Mapping):
        raise PhysicalBenchmarkInvariantErrorV14("v14 identities are absent")
    _validate_attempt(
        config,
        Path(str(marker_record.get("path") or "")),
        str(marker_record.get("sha256") or ""),
    )
    w3 = benchmark.get("w3") or {}
    immutable = immutable_record or {}
    product = load_fragment_w3_trial_product_v13(
        Path(str(w3.get("trial_manifest_path") or "")),
        workspace_root=root,
        expected_manifest_sha256=str(w3.get("trial_manifest_sha256") or ""),
        expected_anchor_trial_sha256=str(immutable.get("trial_manifest_sha256") or ""),
        verify_logical=True,
        verify_rehydration=False,
        expected_wrapper_schema_version=TRIAL_WRAPPER_SCHEMA_VERSION,
        expected_wrapper_run_kind=TRIAL_RUN_KIND,
        physical_config_loader=load_physical_config_v14,
        physical_config_relative_path=CONFIG_RELATIVE_PATH,
        logical_hash_provider=V14_LOGICAL_HASH_PROVIDER,
        functional_projection_loader=partial(
            load_immutable_w1_functional_projection_v14,
            workspace_root=root,
            recompute_logical=False,
        ),
    )
    logical_record = benchmark.get("logical_io") or {}
    from .retrieval_fold_fused_logical_io_v13 import load_logical_io_gate_v13

    logical_gate = load_logical_io_gate_v13(
        Path(str(logical_record.get("manifest_path") or "")),
        expected_manifest_sha256=str(
            logical_record.get("manifest_sha256") or ""
        ),
        verify_source_hash=True,
        expected_config_schema_version=CONFIG_SCHEMA_VERSION,
        expected_config_status=CONFIG_STATUS,
        expected_projection_drift_policy="STOP_V14_IO_BEFORE_W3",
        expected_wrapper_schema_version=TRIAL_WRAPPER_SCHEMA_VERSION,
        expected_wrapper_run_kind=TRIAL_RUN_KIND,
        expected_revision_scope="PHYSICAL_HASH_PROTOCOL_ONLY",
        expected_recipe_guard_field="scientific_contract_changed",
    )
    logical = logical_io_evidence_from_gate_v13(logical_gate)
    resources = _resources_from_record(w3.get("resources") or {})
    rehydration = w3.get("compact_rehydration")
    rehydration_exact = (
        isinstance(rehydration, Mapping)
        and set(rehydration)
        == {
            "status",
            "exact",
            "site_feature_values",
            "prerank500_full_payload",
        }
        and rehydration.get("status")
        == "PASS_ONE_INDEPENDENT_POST_TIMING_RECONSTRUCTION"
        and rehydration.get("exact") is True
        and rehydration.get("site_feature_values")
        == product.logical_digests["site_feature_values"]
        and rehydration.get("prerank500_full_payload")
        == product.logical_digests["prerank500_full_payload"]
    )
    if not rehydration_exact:
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 authorization lost its one post-timing rehydration proof"
        )
    # Authorization reopening must not run the expensive reconstruction a
    # second time.  Rehydrate the *typed evidence state* only after the exact
    # sealed roots above have been validated against the strictly reopened
    # W3 wrapper.
    product = replace(product, compact_rehydration_exact=True)
    checks, speedup = _derive_physical_checks_v13(
        config=config,
        anchor=anchor,
        product=product,
        logical=logical,
        resources=resources,
        compact_rehydration_exact=rehydration_exact,
    )
    synthetic = PhysicalBenchmarkAssessmentV14(
        config_path=config_path.resolve(),
        output=benchmark_path.parent,
        marker=Path(str(marker_record["path"])).resolve(),
        marker_sha256=str(marker_record["sha256"]),
        anchor=anchor,
        product=product,
        resources=resources,
        logical=logical,
        checks=checks,
        speedup=speedup,
        v14_implementation_sha256=current_sha,
        v14_implementation_files=current_files,
        official_capability=_CAPABILITY_TOKEN_V14,
    )
    expected_benchmark_identity = _benchmark_identity(synthetic)
    expected_benchmark = _benchmark_manifest(
        synthetic, status="PASS_V14_PHYSICAL_BENCHMARK"
    )
    expected_auth = _authorization_document(
        synthetic, expected_benchmark_manifest_sha256
    )
    if (
        dict(benchmark_identity) != dict(expected_benchmark_identity)
        or dict(benchmark.get("checks") or {}) != checks
        or benchmark.get("status") != _decision_status(checks)
        or not math.isclose(float(w3.get("speedup_over_immutable_w1")), speedup)
        or auth != expected_auth
        or benchmark != expected_benchmark
        or identity.get("v14_implementation_sha256") != current_sha
    ):
        raise PhysicalBenchmarkInvariantErrorV14(
            "v14 authorization does not reopen from measured facts"
        )
    return auth


__all__ = [
    "AUTHORIZATION_SCHEMA_VERSION",
    "BUILD_ENTRYPOINT_V14",
    "CONFIG_RELATIVE_PATH",
    "CONFIG_SCHEMA_VERSION",
    "CONFIG_SHA256",
    "ENGINE_ID",
    "FUNCTIONAL_PROJECTION_SCHEMA_VERSION",
    "ImmutableW1FunctionalProjectionV14",
    "PhysicalBenchmarkInvariantErrorV14",
    "REPLAY_ENTRYPOINT_V14",
    "SCHEMA_VERSION",
    "TRIAL_WRAPPER_SCHEMA_VERSION",
    "build_worker_entrypoint_v14",
    "build_immutable_w1_functional_projection_v14",
    "execute_fragment_w3_trial_v14",
    "load_execution_authorization_v14",
    "load_immutable_w1_functional_projection_v14",
    "load_physical_config_v14",
    "replay_worker_entrypoint_v14",
    "run_physical_benchmark_v14",
    "v14_implementation_lineage",
]
