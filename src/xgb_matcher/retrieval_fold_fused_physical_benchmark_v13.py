"""Sealed W3-only physical benchmark for retrieval all-human v13.

This module is deliberately split in two layers:

* :func:`execute_fragment_w3_trial_v13` runs the frozen build/fit/replay DAG;
* :func:`run_physical_benchmark_v13` applies the preregistered physical gates,
  publishes a PASS/STOP manifest, and emits a separate national authorization
  only after every gate passed.

The v12 STOP remains immutable.  Its content-addressed W1 trial is a baseline
and a non-circular logical-parity authority; it is never rerun or relabelled.
No quality metric participates in this physical decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from itertools import zip_longest
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

from .retrieval_fold_fused_oof_v12 import (
    FoldFusedInvariantError,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_physical_benchmark_v12 import (
    PhysicalResourceMetricsV12,
    measure_physical_trial_v12,
    sharded_execution_lineage_v12,
)
from .retrieval_fold_fused_sharded_oof_v12 import (
    ShardSpecV12,
    ShardedPhysicalInputsV12,
    WorkerJobV12,
    load_shard_build_manifest_v12,
    plan_query_shards,
    spawn_manifest_workers_v12,
)
from .retrieval_fold_fused_sharded_preflight_v12 import (
    PreparedShardedExecutionV12,
    prepare_sharded_execution_v12,
    require_external_sharded_output_v12,
)
from .retrieval_fold_fused_logical_hash_provider_v14 import (
    LogicalTableHashProvider,
    V13_LOGICAL_HASH_PROVIDER,
    logical_hash_lineage_record,
)


SCHEMA_VERSION = "sireto-retrieval-v13-physical-benchmark-1"
TRIAL_SCHEMA_VERSION = "sireto-retrieval-v13-w3-fragment-trial-1"
AUTHORIZATION_SCHEMA_VERSION = "sireto-retrieval-v13-national-authorization-1"
ENGINE_ID = "FOLD_FUSED_FRAGMENT_W3_PHYSICAL_BENCHMARK_V13_1"
CONFIG_SHA256 = "40c75e669289147ecf3b3c89ad37f9ef0f8e289cc092868ec26de10d0e15bf0d"
ALLOWED_FOLDS = tuple(TRAIN_FOLDS)
SAMPLE_QUERY_COUNT = 512
NATIONAL_QUERY_COUNT = 23_571
GIB = 1024**3
_SHA256_CHARS = frozenset("0123456789abcdef")

REQUIRED_LOGICAL_BRIDGE_DIGESTS = (
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

V13_IMPLEMENTATION_FILES = (
    "config/retrieval_all_human_v13_physical_execution.json",
    "docs/retrieval_all_human_v13_physical_execution_addendum.md",
    "requirements.txt",
    "src/xgb_matcher/retrieval_fold_fused_fragment_dataset_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_fragment_outputs_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_relocatable_paths_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_zero_copy_run_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_fragment_runtime_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_row_group_ledger_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_logical_io_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_physical_benchmark_v13.py",
    "src/xgb_matcher/retrieval_fold_fused_national_runtime_v13.py",
    "scripts/audit_retrieval_fold_fused_logical_io_v13.py",
    "scripts/benchmark_retrieval_fold_fused_workers_v13.py",
    "scripts/run_retrieval_fold_fused_sharded_oof_v13.py",
)

TRIAL_EVIDENCE_SCHEMA_VERSION = "sireto-retrieval-v13-trial-evidence-1"
IMMUTABLE_FUNCTIONAL_PROJECTION_SCHEMA_VERSION = (
    "sireto-retrieval-v13-immutable-w1-functional-projection-1"
)
BUILD_ENTRYPOINT_V13 = (
    "src.xgb_matcher.retrieval_fold_fused_fragment_runtime_v13:"
    "build_worker_entrypoint_v13"
)

_W3_RUN_KEYS = {
    "schema_version",
    "run_kind",
    "query_count",
    "worker_count",
    "fold0_opened",
    "fold1_opened",
    "positive_injection",
    "source_candidate",
    "source_specs",
    "source_row_group_ledgers",
    "canonical_datasets",
    "artifact_inventory",
    "materialized_output_bytes",
    "logical_digests",
    "no_merged_files",
}
_CANONICAL_DATASET_KEYS = (
    "train_outer2",
    "train_outer3",
    "train_outer4",
    "heldout_spool",
    "prerank500_compact",
    "top100_full",
    "outcomes",
)
_DATASET_RECORD_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "merkle_root",
    "rows",
    "fragment_count",
}
_INVENTORY_RECORD_KEYS = {"role", "family", "path", "sha256", "size_bytes"}
_BENCHMARK_KEYS = {
    "schema_version",
    "status",
    "benchmark_id",
    "benchmark_identity",
    "run_kind",
    "quality_metrics_decision_blocking",
    "national_oof_launched",
    "config",
    "attempt_marker",
    "immutable_v12_stop_preserved",
    "immutable_w1",
    "w3",
    "non_circular_w1_bridge",
    "logical_io",
    "process_and_machine_io_decision_blocking",
    "checks",
    "gate_pass",
    "execution_authorization_published",
}
_BENCHMARK_IDENTITY_KEYS = {
    "engine_id",
    "config_sha256",
    "v13_implementation_sha256",
    "immutable_w1_benchmark_sha256",
    "immutable_w1_trial_sha256",
    "w3_trial_manifest_sha256",
    "logical_io_manifest_sha256",
    "artifact_inventory_sha256",
    "fragment_dataset_manifest_sha256",
    "replay_dataset_manifest_sha256",
    "logical_digests",
    "query_count",
    "folds",
    "worker_count",
    "sharded_implementation_sha256",
    "runtime_environment_sha256",
    "attempt_marker_path",
    "attempt_marker_sha256",
    "sharded_execution_plan_sha256",
    "prefit_parity_manifest_sha256",
}
_BENCHMARK_W3_KEYS = {
    "trial_manifest_path",
    "trial_manifest_sha256",
    "worker_count",
    "query_count",
    "folds",
    "fold0_opened",
    "fold1_opened",
    "positive_injection",
    "v13_implementation_sha256",
    "lineage",
    "logical_digests",
    "compact_rehydration",
    "materialized_bytes",
    "projected_national_storage_bytes",
    "resources",
    "speedup_over_immutable_w1",
}
_AUTHORIZATION_KEYS = {
    "schema_version",
    "status",
    "authorization_id",
    "authorization_identity",
    "folds",
    "fold0_opened",
    "fold1_opened",
    "positive_injection",
    "automatic_national_launch",
}
_AUTHORIZATION_IDENTITY_KEYS = {
    "engine_id",
    "benchmark_manifest_sha256",
    "config_sha256",
    "v13_implementation_sha256",
    "immutable_w1_benchmark_sha256",
    "w3_trial_manifest_sha256",
    "logical_io_manifest_sha256",
    "authorized_worker_count",
    "authorized_scope",
    "sharded_implementation_sha256",
    "runtime_environment_sha256",
    "attempt_marker_path",
    "attempt_marker_sha256",
    "sharded_execution_plan_sha256",
    "prefit_parity_manifest_sha256",
}


class PhysicalBenchmarkInvariantErrorV13(FoldFusedInvariantError):
    """A v13 benchmark input or result escaped the frozen physical contract."""


class _OneShotAttemptConsumptionErrorV13(RuntimeError):
    """An error raised only after this process won the marker O_EXCL."""


def _canonical_json(value: Any) -> str:
    def compatible(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): compatible(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [compatible(val) for val in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, float) and not math.isfinite(item):
            raise PhysicalBenchmarkInvariantErrorV13(
                "non-finite value cannot be sealed"
            )
        return item

    return json.dumps(
        compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            f"{label} is not readable JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise PhysicalBenchmarkInvariantErrorV13(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _strict_sha_map(
    value: Mapping[str, Any], *, expected_keys: Sequence[str], label: str
) -> dict[str, str]:
    output = {str(key): str(item) for key, item in value.items()}
    if set(output) != set(expected_keys) or any(
        not _valid_sha256(item) for item in output.values()
    ):
        raise PhysicalBenchmarkInvariantErrorV13(f"{label} digest set drifted")
    return {key: output[key] for key in expected_keys}


def v13_implementation_lineage(
    workspace_root: Path,
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    """Seal the exact v13 implementation used by benchmark and national run.

    The v12 implementation pin is intentionally left untouched.  This second,
    v13-only seal covers the amended physical config, dependency lock surface,
    every executable v13 module and both v13 CLIs.  A national loader recomputes
    it from current bytes, so an authorization cannot outlive a code change.
    """

    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 implementation workspace is absent"
        )
    records: list[Mapping[str, Any]] = []
    for relative in V13_IMPLEMENTATION_FILES:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise PhysicalBenchmarkInvariantErrorV13(
                f"v13 implementation file is absent: {relative}"
            )
        digest = sha256_file(path)
        if relative == V13_IMPLEMENTATION_FILES[0] and digest != CONFIG_SHA256:
            raise PhysicalBenchmarkInvariantErrorV13(
                "v13 implementation embeds another physical config"
            )
        records.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    sealed = tuple(records)
    return _sha256_json(sealed), sealed


_AUTHORITY_CONTRACT_V13: Mapping[str, tuple[str, str, str | None]] = {
    "scientific_contract": (
        "docs/retrieval_all_human_v12_contract.md",
        "markdown",
        "# Contrat v12 — retrieval all-human top100-first",
    ),
    "scientific_config": (
        "config/retrieval_all_human_v12.json",
        "json",
        "sireto-retrieval-all-human-v12-config-1",
    ),
    "fold_fused_addendum": (
        "docs/retrieval_all_human_v12_fold_fused_execution_addendum.md",
        "markdown",
        "# Addendum d’exécution v12 — compilateur OOF fold-fused",
    ),
    "fold_fused_config": (
        "config/retrieval_all_human_v12_fold_fused_executor.json",
        "json",
        "sireto-retrieval-v12-fold-fused-executor-1",
    ),
    "v12_sharded_addendum": (
        "docs/retrieval_all_human_v12_sharded_execution_addendum.md",
        "markdown",
        "# Addendum physique v12 — OOF fold-sharded exact",
    ),
    "v12_sharded_plan": (
        "config/retrieval_all_human_v12_sharded_execution_plan.json",
        "json",
        "sireto-retrieval-v12-fold-sharded-execution-plan-1",
    ),
    "v12_stop_report": (
        "docs/retrieval_all_human_v12_sharded_benchmark_results.md",
        "markdown",
        "Verdict : `STOP_NO_PARALLEL_WORKER_AUTHORIZED`",
    ),
    "v13_physical_addendum": (
        "docs/retrieval_all_human_v13_physical_execution_addendum.md",
        "markdown",
        "# Addendum physique v13 — acteurs persistants et artefacts sans fusion",
    ),
}


def _validate_physical_authorities_v13(
    config: Mapping[str, Any], *, workspace_root: Path
) -> None:
    authorities = config.get("authorities")
    root = Path(workspace_root).resolve()
    if not isinstance(authorities, Mapping) or set(authorities) != set(
        _AUTHORITY_CONTRACT_V13
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 physical authority set drifted"
        )
    for name, (expected_relative, kind, schema_or_marker) in (
        _AUTHORITY_CONTRACT_V13.items()
    ):
        record = authorities[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or record.get("path") != expected_relative
            or not _valid_sha256(record.get("sha256"))
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                f"v13 physical authority record drifted: {name}"
            )
        path = (root / expected_relative).resolve()
        if (
            root not in path.parents
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != record["sha256"]
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                f"v13 physical authority bytes drifted: {name}"
            )
        if kind == "json":
            raw = _read_json(path, label=f"v13 authority {name}")
            if raw.get("schema_version") != schema_or_marker:
                raise PhysicalBenchmarkInvariantErrorV13(
                    f"v13 physical authority schema drifted: {name}"
                )
        else:
            text_value = path.read_text(encoding="utf-8")
            if schema_or_marker not in text_value:
                raise PhysicalBenchmarkInvariantErrorV13(
                    f"v13 physical authority marker drifted: {name}"
                )


@dataclass(frozen=True)
class ImmutableW1AnchorV13:
    benchmark_manifest_path: Path
    benchmark_manifest_sha256: str
    trial_manifest_path: Path
    trial_manifest_sha256: str
    wall_seconds: float
    materialized_bytes: int
    query_count: int
    folds: tuple[int, ...]


@dataclass(frozen=True)
class FragmentW3TrialProductV13:
    manifest_path: Path
    manifest_sha256: str
    query_count: int
    folds: tuple[int, ...]
    v13_implementation_sha256: str
    sharded_execution_plan_sha256: str
    prefit_parity_manifest_sha256: str
    lineage: Mapping[str, str]
    logical_digests: Mapping[str, str]
    immutable_w1_logical_digests: Mapping[str, str]
    materialized_bytes: int
    projected_national_storage_bytes: int
    artifact_inventory_sha256: str
    fragment_dataset_manifest_sha256: str
    replay_dataset_manifest_sha256: str
    physical_merge_files_present: bool
    full_payload_outside_top100: bool
    compact_rehydration_exact: bool | None
    fold0_opened: bool = False
    fold1_opened: bool = False
    positive_injection: bool = False
    worker_count: int = 3

    def __post_init__(self) -> None:
        path = Path(self.manifest_path).resolve()
        if (
            self.worker_count != 3
            or self.query_count != SAMPLE_QUERY_COUNT
            or tuple(self.folds) != ALLOWED_FOLDS
            or self.fold0_opened
            or self.fold1_opened
            or self.positive_injection
            or not _valid_sha256(self.v13_implementation_sha256)
            or not _valid_sha256(self.sharded_execution_plan_sha256)
            or not _valid_sha256(self.prefit_parity_manifest_sha256)
            or not path.is_file()
            or sha256_file(path) != self.manifest_sha256
            or self.materialized_bytes <= 0
            or self.projected_national_storage_bytes <= 0
            or not _valid_sha256(self.artifact_inventory_sha256)
            or not _valid_sha256(self.fragment_dataset_manifest_sha256)
            or not _valid_sha256(self.replay_dataset_manifest_sha256)
            or type(self.compact_rehydration_exact) not in {bool, type(None)}
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "W3 trial identity/resources are invalid"
            )
        _strict_sha_map(
            self.logical_digests,
            expected_keys=REQUIRED_LOGICAL_BRIDGE_DIGESTS,
            label="W3 logical bridge",
        )
        _strict_sha_map(
            self.immutable_w1_logical_digests,
            expected_keys=REQUIRED_LOGICAL_BRIDGE_DIGESTS,
            label="immutable W1 logical bridge",
        )
        lineage = {str(key): str(value) for key, value in self.lineage.items()}
        if (
            not lineage
            or lineage.get("v13_implementation_sha256")
            != self.v13_implementation_sha256
            or any(not _valid_sha256(value) for value in lineage.values())
        ):
            raise PhysicalBenchmarkInvariantErrorV13("W3 lineage is incomplete")

    @property
    def bridge_exact(self) -> bool:
        return dict(self.logical_digests) == dict(
            self.immutable_w1_logical_digests
        )


@dataclass(frozen=True)
class LogicalIOEvidenceV13:
    manifest_path: Path
    manifest_sha256: str
    status: str
    gate_pass: bool
    projection_sha256: str
    unique_row_groups: int
    logical_compressed_bytes: int
    cross_actor_overlap_count: int
    w3_union_equals_w1: bool
    logical_amplification_ratio: float
    materialized_ratio: float

    def __post_init__(self) -> None:
        path = Path(self.manifest_path).resolve()
        if (
            not path.is_file()
            or sha256_file(path) != self.manifest_sha256
            or not _valid_sha256(self.projection_sha256)
            or self.unique_row_groups < 0
            or self.logical_compressed_bytes < 0
            or self.cross_actor_overlap_count < 0
            or not math.isfinite(self.logical_amplification_ratio)
            or self.logical_amplification_ratio < 0
            or not math.isfinite(self.materialized_ratio)
            or self.materialized_ratio < 0
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "logical-I/O evidence is incomplete"
            )


@dataclass(frozen=True)
class _PhysicalBenchmarkAssessmentV13:
    """Measured evidence before either publication surface decides status.

    The assessment layer can be exercised with test doubles, but it cannot
    publish a PASS or an execution authorization.  Only the callback-free
    official consumed runner below may hand an assessment to the official
    publisher.
    """

    config_resolved: Path
    anchor: ImmutableW1AnchorV13
    output: Path
    marker: Path
    marker_sha256: str
    product: FragmentW3TrialProductV13
    resources: PhysicalResourceMetricsV12
    logical: LogicalIOEvidenceV13
    speedup: float
    checks: Mapping[str, bool]
    identity: Mapping[str, Any]
    authorization_capability: object | None = None


class _OfficialDAGCapabilityV13:
    """Opaque in-process provenance produced only by the official consumed DAG."""


_OFFICIAL_DAG_CAPABILITY_V13 = _OfficialDAGCapabilityV13()


@dataclass(frozen=True)
class _OfficialPhysicalDAGV13:
    prepared: PreparedShardedExecutionV12
    workspace_root: Path
    w1_functional_projection: Any
    w1_io_projection: Any
    capability: object


def logical_io_evidence_from_gate_v13(artifact: Any) -> LogicalIOEvidenceV13:
    """Adapt the independently sealed logical-I/O gate without weakening it."""

    path = Path(getattr(artifact, "manifest_path", "")).resolve()
    digest = str(getattr(artifact, "manifest_sha256", ""))
    if not path.is_file() or not _valid_sha256(digest) or sha256_file(path) != digest:
        raise PhysicalBenchmarkInvariantErrorV13("logical-I/O gate bytes drifted")
    raw = _read_json(path, label="v13 logical-I/O gate")
    source = raw.get("source")
    w1 = raw.get("w1")
    w3 = raw.get("w3")
    checks = raw.get("checks")
    ratios = raw.get("ratios")
    if (
        raw.get("schema_version")
        != "sireto-retrieval-v13-cache-neutral-io-gate-1"
        or raw.get("status") not in {"PASS_V13_IO", "STOP_V13_IO"}
        or not isinstance(source, Mapping)
        or not isinstance(w1, Mapping)
        or not isinstance(w3, Mapping)
        or not isinstance(checks, Mapping)
        or not isinstance(ratios, Mapping)
        or raw.get("gate_pass") is not all(bool(value) for value in checks.values())
    ):
        raise PhysicalBenchmarkInvariantErrorV13("logical-I/O gate contract drifted")
    return LogicalIOEvidenceV13(
        manifest_path=path,
        manifest_sha256=digest,
        status=str(raw["status"]),
        gate_pass=bool(raw["gate_pass"]),
        projection_sha256=str(source.get("projection_sha256") or ""),
        unique_row_groups=int(w3.get("union_unique_extent_count", -1)),
        logical_compressed_bytes=int(w3.get("logical_compressed_bytes", -1)),
        cross_actor_overlap_count=int(
            w3.get("inter_actor_duplicate_extent_references", -1)
        ),
        w3_union_equals_w1=bool(checks.get("source_extent_union_identical")),
        logical_amplification_ratio=float(
            ratios.get("logical_compressed_bytes_w3_over_w1", math.nan)
        ),
        materialized_ratio=float(
            ratios.get("materialized_output_bytes_w3_over_w1", math.nan)
        ),
    )


class W3TrialRunnerV13(Protocol):
    def __call__(
        self, output_dir: Path, anchor: ImmutableW1AnchorV13
    ) -> FragmentW3TrialProductV13: ...


class LogicalIOBuilderV13(Protocol):
    def __call__(
        self,
        product: FragmentW3TrialProductV13,
        anchor: ImmutableW1AnchorV13,
        output_dir: Path,
    ) -> LogicalIOEvidenceV13: ...


_ProductT = TypeVar("_ProductT")


class TrialMeasurerV13(Protocol):
    def __call__(
        self, callback: Callable[[], _ProductT]
    ) -> tuple[_ProductT, PhysicalResourceMetricsV12]: ...


def load_physical_config_v13(path: Path) -> Mapping[str, Any]:
    """Load the exact currently preregistered physical configuration."""

    resolved = Path(path).resolve()
    if not resolved.is_file() or sha256_file(resolved) != CONFIG_SHA256:
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 physical config differs from the preregistered amendments"
        )
    config = _read_json(resolved, label="v13 physical config")
    benchmark = config.get("benchmark_512")
    io_gate = config.get("cache_neutral_io_gate")
    actors = config.get("actor_architecture")
    fragments = config.get("ordered_fragment_protocol")
    payload = config.get("payload_policy")
    national = config.get("national_execution_protocol")
    pre_w3 = io_gate.get("pre_w3_source_validation") if isinstance(io_gate, Mapping) else None
    source_extent = io_gate.get("source_extent") if isinstance(io_gate, Mapping) else None
    fragment_inventory = (
        fragments.get("artifact_inventory")
        if isinstance(fragments, Mapping)
        else None
    )
    if (
        config.get("schema_version")
        != "sireto-retrieval-v13-physical-execution-1"
        or config.get("status")
        != "PREREGISTERED_V13_PHYSICAL_BENCHMARK_NOT_RUN"
        or config.get("revision_scope") != "PHYSICAL_ONLY"
        or config.get("logical_recipe_changes") is not False
        or config.get("national_oof_allowed") is not False
        or not isinstance(benchmark, Mapping)
        or benchmark.get("mode") != "ONE_FRESH_W3_VS_IMMUTABLE_V12_W1"
        or benchmark.get("fresh_trials_allowed") != [3]
        or benchmark.get("fresh_trial_count") != 1
        or benchmark.get("canonical_output_dir")
        != "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_physical_benchmark_v13_w3"
        or benchmark.get("attempt_consumed_marker_path")
        != "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/.retrieval_v13_w3_attempt_consumed.json"
        or benchmark.get("attempt_marker_creation")
        != "ATOMIC_O_EXCL_BEFORE_MEASURED_CALLBACK"
        or benchmark.get(
            "direct_w3_execution_requires_validated_marker_capability"
        )
        is not True
        or benchmark.get(
            "authorization_capable_entrypoint_owns_preflight_and_w1_projections"
        )
        is not True
        or benchmark.get("authorization_capable_callbacks_allowed") is not False
        or benchmark.get("generic_prepared_or_projection_injection_allowed")
        is not False
        or benchmark.get("implementation_lineage_capture")
        != "BEFORE_ATTEMPT_MARKER_AND_CALLBACK"
        or benchmark.get("authorization_loader_recomputes_all_physical_gates")
        is not True
        or len(benchmark.get("required_negative_tests") or ()) != 37
        or benchmark.get("any_pass_stop_or_exception_consumes_attempt") is not True
        or benchmark.get("second_attempt_or_other_output_forbidden") is not True
        or benchmark.get("rerun_w1_allowed") is not False
        or benchmark.get("w2_allowed") is not False
        or not isinstance(io_gate, Mapping)
        or not isinstance(pre_w3, Mapping)
        or pre_w3.get("full_source_hash_allowed") is not False
        or pre_w3.get("owned_by_authorization_capable_runner") is not True
        or pre_w3.get("generic_prepared_input_allowed") is not False
        or pre_w3.get("verify_source_hash_argument") is not False
        or pre_w3.get(
            "immutable_w1_projections_built_and_strictly_reloaded_before_marker"
        )
        is not True
        or pre_w3.get("full_source_hash_timing")
        != "POST_W3_ONLY_IN_LOGICAL_IO_COMPARISON"
        or io_gate.get("diagnostic_metrics_can_authorize_or_reject") is not False
        or not isinstance(source_extent, Mapping)
        or source_extent.get("observation_source")
        != "ACTUAL_SCENE_IR_READ_ROW_GROUP_CACHE_MISS_CALLBACK_AFTER_SUCCESSFUL_DECODE"
        or source_extent.get(
            "theoretical_source_range_reconstruction_can_satisfy_gate"
        )
        is not False
        or source_extent.get("actual_read_ledger_required_per_build_actor")
        is not True
        or source_extent.get("actual_read_ledger_count") != 3
        or source_extent.get("actual_read_ledger_inventoried") is not True
        or source_extent.get("sealed_sample_expected_unique_row_groups") != 368
        or source_extent.get(
            "sealed_sample_expected_logical_compressed_bytes"
        )
        != 2_696_076_099
        or source_extent.get("national_expected_unique_row_groups") != 635
        or source_extent.get(
            "national_expected_logical_compressed_bytes"
        )
        != 4_646_932_145
        or not isinstance(actors, Mapping)
        or actors.get("start_method") != "spawn"
        or actors.get("maximum_concurrent_actors") != 3
        or (actors.get("replay_phase") or {}).get("model_scope_per_actor")
        != "EXACTLY_ONE_OUTER_MODEL_EQUAL_TO_OWNED_FOLD"
        or not isinstance(fragments, Mapping)
        or fragments.get("physical_merge_files_allowed") is not False
        or not isinstance(fragment_inventory, Mapping)
        or fragment_inventory.get(
            "actual_row_group_ledger_manifest_and_rows_required_per_build_actor"
        )
        is not True
        or fragment_inventory.get("actual_row_group_ledger_roles")
        != [
            "actual_row_group_ledger_manifest",
            "actual_row_group_ledger_rows",
        ]
        or not isinstance(payload, Mapping)
        or (payload.get("prerank500") or {}).get("full_payload_persisted")
        is not False
        or (payload.get("prerank500") or {}).get(
            "full_payload_logical_digest_sealed_before_compaction"
        )
        is not True
        or (payload.get("prerank500") or {}).get(
            "full_payload_must_equal_immutable_w1"
        )
        is not True
        or (payload.get("prerank500") or {}).get(
            "independent_rehydration_timing"
        )
        != "EXACTLY_ONCE_AFTER_W3_RESOURCE_TIMER"
        or (payload.get("top100") or {}).get("full_payload_persisted") is not True
        or not isinstance(national, Mapping)
        or national.get("status")
        != "LOCKED_PENDING_SEPARATE_AUTHORIZATION"
        or national.get("canonical_output_dir")
        != "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_national_oof_v13"
        or national.get("attempt_consumed_marker_path")
        != "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/.retrieval_v13_national_attempt_consumed.json"
        or national.get("stage_directory_prefix")
        != ".retrieval_fold_fused_national_oof_v13.stage-"
        or national.get("stage_identity")
        != "SHA256_CONFIG_AUTH_BENCHMARK_IMPLEMENTATION_SOURCE_PLAN_PREFIT_OUTPUT"
        or national.get("attempt_marker_creation")
        != "ATOMIC_O_EXCL_BEFORE_STAGE_OR_WORKER"
        or national.get("success_promotion")
        != "ATOMIC_RENAME_SAME_FILESYSTEM"
        or national.get("stage_retained_on_failure") is not True
        or national.get("automatic_retry_allowed") is not False
        or national.get("alternate_output_allowed") is not False
        or national.get("exception_status")
        != "STOP_NATIONAL_EXECUTION_EXCEPTION"
        or len(national.get("required_negative_tests") or ()) != 9
        or (national.get("fit_resource_guard") or {}).get("phase")
        != "AFTER_BUILD_ACTORS_JOIN_BEFORE_FIRST_SITE_FIT"
        or (national.get("fit_resource_guard") or {}).get(
            "live_build_actor_count_required"
        )
        != 0
        or (national.get("fit_resource_guard") or {}).get("fit_parallelism")
        != 1
        or (national.get("fit_resource_guard") or {}).get("outer_fit_order")
        != [2, 3, 4]
        or (national.get("fit_resource_guard") or {}).get(
            "gc_between_outer_fits"
        )
        is not True
        or (national.get("fit_resource_guard") or {}).get(
            "memory_used_maximum_source"
        )
        != "benchmark_512.maximum_w3_peak_machine_used_memory_bytes"
        or (national.get("fit_resource_guard") or {}).get(
            "memory_available_required_formula"
        )
        != "MACHINE_TOTAL_BYTES_MINUS_MEMORY_USED_MAXIMUM_BYTES"
        or (national.get("fit_resource_guard") or {}).get(
            "swapout_delta_bytes_required"
        )
        != 0
        or (national.get("fit_resource_guard") or {}).get(
            "measurement_seconds"
        )
        != 5.0
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 physical configuration semantic drift"
        )
    _validate_physical_authorities_v13(
        config, workspace_root=resolved.parent.parent
    )
    return config


def load_immutable_w1_anchor_v13(
    config: Mapping[str, Any],
) -> ImmutableW1AnchorV13:
    """Verify and load the immutable v12 STOP/W1 baseline without mutation."""

    stop = config.get("immutable_v12_stop")
    if not isinstance(stop, Mapping):
        raise PhysicalBenchmarkInvariantErrorV13("v12 STOP anchor is missing")
    benchmark_path = Path(str(stop.get("benchmark_manifest_path") or "")).resolve()
    benchmark_sha = str(stop.get("benchmark_manifest_sha256") or "")
    if (
        stop.get("verdict") != "STOP_NO_PARALLEL_WORKER_AUTHORIZED"
        or stop.get("reinterpretation_forbidden") is not True
        or stop.get("replacement_forbidden") is not True
        or stop.get("execution_authorization_published") is not False
        or not benchmark_path.is_file()
        or sha256_file(benchmark_path) != benchmark_sha
    ):
        raise PhysicalBenchmarkInvariantErrorV13("immutable v12 STOP anchor drifted")
    raw = _read_json(benchmark_path, label="immutable v12 benchmark")
    w1 = stop.get("w1_trial")
    if not isinstance(w1, Mapping):
        raise PhysicalBenchmarkInvariantErrorV13("immutable W1 record is missing")
    trial_path = (benchmark_path.parent / str(w1.get("relative_manifest_path") or "")).resolve()
    trial_sha = str(w1.get("manifest_sha256") or "")
    trials = raw.get("trials")
    raw_w1 = trials.get("1") if isinstance(trials, Mapping) else None
    artifact = raw_w1.get("artifact_manifest") if isinstance(raw_w1, Mapping) else None
    if (
        raw.get("schema_version")
        != "sireto-retrieval-v12-physical-worker-benchmark-1"
        or raw.get("status") != "STOP_NO_PARALLEL_WORKER_AUTHORIZED"
        or raw.get("selected_worker_count") != 0
        or not isinstance(artifact, Mapping)
        or artifact.get("path") != w1.get("relative_manifest_path")
        or artifact.get("sha256") != trial_sha
        or not trial_path.is_file()
        or sha256_file(trial_path) != trial_sha
        or int(w1.get("query_count", -1)) != SAMPLE_QUERY_COUNT
        or tuple(int(value) for value in w1.get("folds") or ()) != ALLOWED_FOLDS
        or bool(w1.get("fold0_opened"))
        or bool(w1.get("fold1_opened"))
        or bool(w1.get("positive_injection"))
    ):
        raise PhysicalBenchmarkInvariantErrorV13("immutable W1 trial drifted")
    wall = float(w1.get("wall_seconds", math.nan))
    materialized = int(w1.get("materialized_bytes", -1))
    if not math.isfinite(wall) or wall <= 0 or materialized <= 0:
        raise PhysicalBenchmarkInvariantErrorV13("immutable W1 resources drifted")
    return ImmutableW1AnchorV13(
        benchmark_manifest_path=benchmark_path,
        benchmark_manifest_sha256=benchmark_sha,
        trial_manifest_path=trial_path,
        trial_manifest_sha256=trial_sha,
        wall_seconds=wall,
        materialized_bytes=materialized,
        query_count=SAMPLE_QUERY_COUNT,
        folds=ALLOWED_FOLDS,
    )


def _safe_trial_child(root: Path, value: Any, *, label: str) -> tuple[str, Path]:
    raw = str(value or "")
    relative = Path(raw)
    if (
        not raw
        or relative.is_absolute()
        or ".." in relative.parts
        or raw.startswith("/")
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            f"{label} is not a safe trial-relative path"
        )
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise PhysicalBenchmarkInvariantErrorV13(
            f"{label} escaped, is absent, or is a symlink"
        )
    return relative.as_posix(), path


def _inventory_index_v13(
    manifest_path: Path, raw: Mapping[str, Any]
) -> tuple[Mapping[str, Mapping[str, Any]], int]:
    root = manifest_path.parent
    if manifest_path.stat().st_nlink != 1:
        raise PhysicalBenchmarkInvariantErrorV13(
            "W3 root manifest must not be hardlinked"
        )
    inventory = raw.get("artifact_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise PhysicalBenchmarkInvariantErrorV13("W3 artifact inventory is empty")
    records: dict[str, Mapping[str, Any]] = {}
    inodes: set[tuple[int, int]] = set()
    for ordinal, value in enumerate(inventory):
        if (
            not isinstance(value, Mapping)
            or set(value) != _INVENTORY_RECORD_KEYS
            or not isinstance(value.get("role"), str)
            or not value.get("role")
            or not isinstance(value.get("family"), str)
            or not value.get("family")
            or not _valid_sha256(value.get("sha256"))
            or type(value.get("size_bytes")) is not int
            or int(value["size_bytes"]) <= 0
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                f"W3 inventory record {ordinal} drifted"
            )
        relative, path = _safe_trial_child(
            root, value.get("path"), label=f"W3 inventory record {ordinal}"
        )
        if relative in records or path == manifest_path:
            raise PhysicalBenchmarkInvariantErrorV13(
                "W3 inventory duplicates a path or includes the root manifest"
            )
        if any(
            part.startswith(("merged_", "canonical_merged_"))
            for part in Path(relative).parts
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "W3 inventory contains a forbidden merged artifact"
            )
        stat = path.stat()
        inode = (int(stat.st_dev), int(stat.st_ino))
        if (
            inode in inodes
            or int(stat.st_nlink) != 1
            or int(stat.st_size) != int(value["size_bytes"])
            or sha256_file(path) != value["sha256"]
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "W3 inventory contains a hardlink or byte drift"
            )
        inodes.add(inode)
        records[relative] = value
    actual_children: set[str] = set()
    for child in root.rglob("*"):
        if child.is_symlink():
            raise PhysicalBenchmarkInvariantErrorV13(
                "W3 trial tree contains a symlink"
            )
        if child.is_file() and child.resolve() != manifest_path:
            relative = child.resolve().relative_to(root).as_posix()
            actual_children.add(relative)
            if any(
                part.startswith(("merged_", "canonical_merged_"))
                for part in Path(relative).parts
            ):
                raise PhysicalBenchmarkInvariantErrorV13(
                    "W3 trial tree contains a forbidden merged artifact"
                )
    if actual_children != set(records):
        raise PhysicalBenchmarkInvariantErrorV13(
            "W3 inventory is not exhaustive over the trial tree"
        )
    total = manifest_path.stat().st_size + sum(
        int(value["size_bytes"]) for value in records.values()
    )
    if raw.get("materialized_output_bytes") != total:
        raise PhysicalBenchmarkInvariantErrorV13(
            "W3 materialized byte declaration differs from the complete tree"
        )
    return records, total


def _inventory_role_path_v13(
    root: Path,
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    role: str,
) -> Path:
    matches = [relative for relative, value in inventory.items() if value["role"] == role]
    if len(matches) != 1:
        raise PhysicalBenchmarkInvariantErrorV13(
            f"W3 inventory requires exactly one {role}"
        )
    return (root / matches[0]).resolve()


def _resolve_trial_evidence_record_v13(
    *,
    trial_root: Path,
    evidence: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]],
    name: str,
    external_authority: bool = False,
) -> tuple[Path, str]:
    """Resolve one evidence record without letting internal files escape."""

    record = evidence.get(name)
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise PhysicalBenchmarkInvariantErrorV13(f"v13 {name} evidence drifted")
    child = Path(str(record.get("path") or ""))
    digest = str(record.get("sha256") or "")
    if external_authority:
        if not child.is_absolute():
            raise PhysicalBenchmarkInvariantErrorV13(
                f"v13 {name} external authority is not absolute"
            )
        resolved = child.resolve()
    else:
        relative, resolved = _safe_trial_child(
            Path(trial_root).resolve(), child, label=f"v13 {name} evidence"
        )
        inventory_record = inventory.get(relative)
        if (
            inventory_record is None
            or inventory_record.get("sha256") != digest
            or inventory_record.get("size_bytes") != resolved.stat().st_size
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                f"v13 {name} evidence is not exactly inventory-bound"
            )
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or resolved.stat().st_nlink != 1
        or not _valid_sha256(digest)
        or sha256_file(resolved) != digest
    ):
        raise PhysicalBenchmarkInvariantErrorV13(f"v13 {name} bytes drifted")
    return resolved, digest


def _validate_compact_rehydration_v13(
    build: Any,
    replay: Any,
    *,
    feature_context: Any,
    logical_hash_provider: LogicalTableHashProvider = V13_LOGICAL_HASH_PROVIDER,
) -> tuple[int, str]:
    """Differentially replay top500 features/proofs from the heldout source."""

    from .retrieval_fold_fused_fragment_dataset_v13 import (
        iter_heldout_contexts_from_fragments_v13,
    )
    from .retrieval_fold_fused_fragment_outputs_v13 import (
        TOP100_SCHEMA,
        _canonical_json as _output_canonical_json,
        feature_batch_logical_leaf_v13,
        iter_compact_output_query_tables_v13,
        prerank500_full_payload_leaf_v13,
    )
    from .retrieval_fold_fused_fragment_dataset_v13 import merkle_root_v13
    import pyarrow as pa

    digest = hashlib.sha256()
    rows = 0
    compact_iterator = iter_compact_output_query_tables_v13(
        replay, "top500", logical_hash_provider
    )
    top100_iterator = iter_compact_output_query_tables_v13(
        replay, "top100", logical_hash_provider
    )
    context_iterator = iter_heldout_contexts_from_fragments_v13(
        build, logical_hash_provider=logical_hash_provider
    )
    feature_leaves: list[Mapping[str, Any]] = []
    payload_leaves: list[Mapping[str, Any]] = []
    for compact, top100, context in zip_longest(
        compact_iterator, top100_iterator, context_iterator
    ):
        if compact is None or top100 is None or context is None:
            raise PhysicalBenchmarkInvariantErrorV13(
                "compact outputs and heldout query populations differ"
            )
        compact_rows = compact.to_pylist()
        top100_rows = top100.to_pylist()
        query_id = str(context.query_id)
        fold = int(context.scene_fold)
        if (
            not compact_rows
            or not top100_rows
            or str(compact_rows[0].get("query_id") or "") != query_id
            or int(compact_rows[0].get("fold", -1)) != fold
            or str(top100_rows[0].get("query_id") or "") != query_id
            or int(top100_rows[0].get("fold", -1)) != fold
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "compact top500 query identity differs from heldout"
            )
        statistics = feature_context.compute_full_master_statistics(context)
        features = feature_context.materialize(
            context, statistics, candidate_indices=None
        )
        if (
            features.query_id != query_id
            or features.scene_fold != fold
            or features.candidate_sirets != context.candidate_sirets
            or features.candidate_parents != context.candidate_parents
            or len(features.feature_names) != 319
            or features.values.shape
            != (len(context.candidate_sirets), len(features.feature_names))
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "rehydrated 319-feature Master drifted"
            )
        feature_leaves.append(feature_batch_logical_leaf_v13(context, features))
        by_siret = {
            siret: index for index, siret in enumerate(features.candidate_sirets)
        }
        if len(by_siret) != len(features.candidate_sirets):
            raise PhysicalBenchmarkInvariantErrorV13(
                "heldout natural key is empty or duplicated"
            )
        rehydrated: list[Mapping[str, Any]] = []
        payload_records: list[Mapping[str, Any]] = []
        for compact_row in compact_rows:
            siret = str(compact_row.get("candidate_siret") or "")
            index = by_siret.get(siret)
            if (
                index is None
                or features.candidate_parents[index]
                != str(compact_row.get("candidate_parent") or "")
            ):
                raise PhysicalBenchmarkInvariantErrorV13(
                    "compact top500 natural-key rehydration drifted"
                )
            feature_values = features.values[index]
            rehydrated.append(
                {
                    "candidate_siret": siret,
                    "candidate_parent": features.candidate_parents[index],
                    "rank": int(compact_row.get("rank", -1)),
                    "feature_names_sha256": _sha256_json(
                        list(features.feature_names)
                    ),
                    "feature_float32_sha256": hashlib.sha256(
                        feature_values.view("uint32").tobytes(order="C")
                    ).hexdigest(),
                    "payload_sha256": _sha256_json(
                        features.row_payloads[index]
                        if features.row_payloads
                        else {}
                    ),
                }
            )
            payload_records.append(
                {
                    "query_id": query_id,
                    "fold": fold,
                    "candidate_siret": siret,
                    "candidate_parent": features.candidate_parents[index],
                    "rank": int(compact_row.get("rank", -1)),
                    "site_score": float(compact_row.get("site_score")),
                    "selection_reason": str(
                        compact_row.get("selection_reason") or ""
                    ),
                    "payload_json": _output_canonical_json(
                        features.row_payloads[index]
                        if features.row_payloads
                        else {}
                    ),
                }
            )
        payload_leaves.append(
            prerank500_full_payload_leaf_v13(
                pa.Table.from_pylist(payload_records, schema=TOP100_SCHEMA),
                logical_hash_provider,
            )
        )
        for top100_row in top100_rows:
            siret = str(top100_row.get("candidate_siret") or "")
            index = by_siret.get(siret)
            try:
                observed_payload = json.loads(
                    str(top100_row.get("payload_json") or "")
                )
            except json.JSONDecodeError as error:
                raise PhysicalBenchmarkInvariantErrorV13(
                    "top100 full payload is not valid canonical JSON"
                ) from error
            expected_payload = (
                features.row_payloads[index]
                if index is not None and features.row_payloads
                else None
            )
            if index is None or observed_payload != expected_payload:
                raise PhysicalBenchmarkInvariantErrorV13(
                    "top100 full payload differs from heldout rehydration"
                )
        digest.update(
            _canonical_json(
                {
                    "query_id": query_id,
                    "fold": fold,
                    "full_feature_batch": feature_leaves[-1],
                    "rehydrated_top500": rehydrated,
                }
            ).encode("utf-8")
        )
        digest.update(b"\n")
        rows += len(compact_rows)
    if rows <= 0:
        raise PhysicalBenchmarkInvariantErrorV13(
            "compact top500 rehydration proof is empty"
        )
    if merkle_root_v13(feature_leaves) != replay.feature_values_logical_merkle_root:
        raise PhysicalBenchmarkInvariantErrorV13(
            "replayed 319-feature values differ from the sealed replay digest"
        )
    if (
        merkle_root_v13(payload_leaves)
        != replay.prerank500_full_payload_logical_merkle_root
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "rehydrated PreRank500 payload/proofs differ from the sealed replay digest"
        )
    return rows, digest.hexdigest()


def load_fragment_w3_trial_product_v13(
    manifest_path: Path,
    *,
    workspace_root: Path,
    expected_manifest_sha256: str | None = None,
    expected_anchor_trial_sha256: str | None = None,
    verify_logical: bool = True,
    verify_rehydration: bool = True,
    expected_wrapper_schema_version: str = "sireto-retrieval-v13-sharded-run-1",
    expected_wrapper_run_kind: str = "PHYSICAL_BENCHMARK_W3_V13",
    physical_config_loader: Callable[[Path], Mapping[str, Any]] = load_physical_config_v13,
    physical_config_relative_path: str = (
        "config/retrieval_all_human_v13_physical_execution.json"
    ),
    logical_hash_provider: LogicalTableHashProvider = V13_LOGICAL_HASH_PROVIDER,
    functional_projection_loader: Callable[..., Any] | None = None,
) -> FragmentW3TrialProductV13:
    """Strictly reopen a W3 wrapper and derive every benchmark-facing fact."""

    path = Path(manifest_path).resolve()
    expected = expected_manifest_sha256 or sha256_file(path)
    if not path.is_file() or not _valid_sha256(expected) or sha256_file(path) != expected:
        raise PhysicalBenchmarkInvariantErrorV13("W3 wrapper hash drifted")
    raw = _read_json(path, label="v13 W3 wrapper")
    source = raw.get("source_candidate")
    specs_raw = raw.get("source_specs")
    datasets_raw = raw.get("canonical_datasets")
    logical_raw = raw.get("logical_digests")
    if (
        set(raw) != _W3_RUN_KEYS
        or raw.get("schema_version") != expected_wrapper_schema_version
        or raw.get("run_kind") != expected_wrapper_run_kind
        or raw.get("query_count") != SAMPLE_QUERY_COUNT
        or raw.get("worker_count") != 3
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or raw.get("no_merged_files") is not True
        or not isinstance(source, Mapping)
        or set(source) != {"path", "sha256", "projection_sha256"}
        or not Path(str(source.get("path") or "")).resolve().is_file()
        or not _valid_sha256(source.get("sha256"))
        or not _valid_sha256(source.get("projection_sha256"))
        or not isinstance(specs_raw, list)
        or len(specs_raw) != 3
        or not isinstance(datasets_raw, Mapping)
        or set(datasets_raw) != set(_CANONICAL_DATASET_KEYS)
        or not isinstance(logical_raw, Mapping)
    ):
        raise PhysicalBenchmarkInvariantErrorV13("W3 wrapper contract drifted")
    logical = _strict_sha_map(
        logical_raw,
        expected_keys=REQUIRED_LOGICAL_BRIDGE_DIGESTS,
        label="W3 wrapper logical bridge",
    )
    specs: list[ShardSpecV12] = []
    previous_stop = 0
    seen_queries: set[str] = set()
    for ordinal, value in enumerate(specs_raw):
        if not isinstance(value, Mapping):
            raise PhysicalBenchmarkInvariantErrorV13("W3 source spec is not an object")
        spec = ShardSpecV12.from_manifest(value)
        if (
            dict(value) != spec.as_manifest()
            or spec.shard_ordinal != ordinal
            or spec.sequence_start != previous_stop
            or spec.folds != (ALLOWED_FOLDS[ordinal],)
            or seen_queries.intersection(spec.query_ids)
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "W3 source specs overlap, migrate queries, or mix folds"
            )
        specs.append(spec)
        previous_stop = spec.sequence_stop
        seen_queries.update(spec.query_ids)
    if previous_stop != SAMPLE_QUERY_COUNT:
        raise PhysicalBenchmarkInvariantErrorV13("W3 source specs do not cover 512 queries")

    inventory, materialized = _inventory_index_v13(path, raw)
    dataset_records: dict[str, Mapping[str, Any]] = {}
    for family in _CANONICAL_DATASET_KEYS:
        record = datasets_raw[family]
        if (
            not isinstance(record, Mapping)
            or set(record) != _DATASET_RECORD_KEYS
            or not _valid_sha256(record.get("sha256"))
            or not _valid_sha256(record.get("merkle_root"))
            or type(record.get("size_bytes")) is not int
            or int(record["size_bytes"]) <= 0
            or type(record.get("rows")) is not int
            or int(record["rows"]) < 0
            or type(record.get("fragment_count")) is not int
            or int(record["fragment_count"]) <= 0
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                f"W3 canonical dataset {family} record drifted"
            )
        relative, dataset_path = _safe_trial_child(
            path.parent, record.get("path"), label=f"W3 dataset {family}"
        )
        inventory_record = inventory.get(relative)
        if (
            inventory_record is None
            or inventory_record.get("family") != family
            or inventory_record.get("sha256") != record.get("sha256")
            or inventory_record.get("size_bytes") != record.get("size_bytes")
            or dataset_path.stat().st_size != int(record["size_bytes"])
            or sha256_file(dataset_path) != record["sha256"]
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                f"W3 dataset {family} is not exactly inventory-bound"
            )
        dataset_records[family] = record

    evidence_path = _inventory_role_path_v13(
        path.parent, inventory, role="trial_evidence"
    )
    evidence = _read_json(evidence_path, label="v13 trial evidence")
    evidence_keys = {
        "schema_version",
        "lineage",
        "implementation_files",
        "immutable_w1_functional_projection",
        "prefit_parity_manifest_sha256",
        "sharded_execution_plan_sha256",
        "build_fragment_dataset",
        "compact_replay_dataset",
        "zero_copy_canonical_run",
        "compact_rehydration",
    }
    if logical_hash_provider.revision != 13:
        evidence_keys.add("logical_hash_provider")
    if set(evidence) != evidence_keys or evidence.get("schema_version") != TRIAL_EVIDENCE_SCHEMA_VERSION:
        raise PhysicalBenchmarkInvariantErrorV13("v13 trial evidence contract drifted")
    lineage = {
        str(key): str(value) for key, value in dict(evidence.get("lineage") or {}).items()
    }
    if logical_hash_provider.revision != 13 and evidence.get(
        "logical_hash_provider"
    ) != logical_hash_lineage_record(logical_hash_provider):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v14 logical hash provider evidence drifted"
        )
    current_implementation, current_files = v13_implementation_lineage(workspace_root)
    scientific_lineage = {
        key: value
        for key, value in lineage.items()
        if key
        not in {
            "sharded_implementation_sha256",
            "runtime_environment_sha256",
            "v13_implementation_sha256",
        }
    }
    expected_lineage = sharded_execution_lineage_v12(
        scientific_lineage, workspace_root=Path(workspace_root).resolve()
    )
    expected_lineage["v13_implementation_sha256"] = current_implementation
    if (
        not lineage
        or lineage != expected_lineage
        or lineage.get("v13_implementation_sha256") != current_implementation
        or evidence.get("implementation_files") != list(current_files)
        or any(not _valid_sha256(value) for value in lineage.values())
        or lineage.get("source_candidate_sha256") != source.get("sha256")
    ):
        raise PhysicalBenchmarkInvariantErrorV13("v13 implementation/input lineage drifted")

    from .retrieval_fold_fused_row_group_ledger_v13 import (
        RowGroupLedgerInvariantErrorV13,
        load_actor_row_group_ledger_records_v13,
        validate_actual_row_group_ledger_set_v13,
    )

    try:
        row_group_artifacts = load_actor_row_group_ledger_records_v13(
            raw.get("source_row_group_ledgers"), root=path.parent
        )
        for record in raw["source_row_group_ledgers"]:
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
                    "actual-read ledger is not exactly inventory-bound"
                )
        physical_config = physical_config_loader(
            Path(workspace_root).resolve()
            / physical_config_relative_path
        )
        source_extent = physical_config["cache_neutral_io_gate"][
            "source_extent"
        ]
        source_projection = source_extent["production_projection"]
        actual_reads = validate_actual_row_group_ledger_set_v13(
            [
                (value.manifest_path, value.manifest_sha256)
                for value in row_group_artifacts
            ],
            expected_specs=specs,
            source_path=Path(str(source["path"])).resolve(),
            expected_source_candidate_sha256=str(source["sha256"]),
            expected_projection_columns=tuple(source_projection["columns"]),
            expected_projection_sha256=str(source_projection["sha256"]),
            expected_sharded_execution_plan_sha256=str(
                evidence["sharded_execution_plan_sha256"]
            ),
            expected_execution_lineage=lineage,
            expected_unique_row_groups=int(
                source_extent["sealed_sample_expected_unique_row_groups"]
            ),
            expected_logical_compressed_bytes=int(
                source_extent[
                    "sealed_sample_expected_logical_compressed_bytes"
                ]
            ),
            verify_source_hash=False,
            enforce_gate_totals=False,
        )
    except (KeyError, TypeError, RowGroupLedgerInvariantErrorV13) as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            "W3 actual source row-group ledger drifted"
        ) from error

    def internal_evidence_record(name: str) -> tuple[Path, str]:
        return _resolve_trial_evidence_record_v13(
            trial_root=path.parent,
            evidence=evidence,
            inventory=inventory,
            name=name,
        )

    build_path, build_sha = internal_evidence_record("build_fragment_dataset")
    replay_path, replay_sha = internal_evidence_record("compact_replay_dataset")
    canonical_path, canonical_sha = internal_evidence_record(
        "zero_copy_canonical_run"
    )
    projection_path, projection_sha = _resolve_trial_evidence_record_v13(
        trial_root=path.parent,
        evidence=evidence,
        inventory=inventory,
        name="immutable_w1_functional_projection",
        external_authority=True,
    )
    from .retrieval_fold_fused_fragment_dataset_v13 import (
        load_build_fragment_dataset_v13,
    )
    from .retrieval_fold_fused_fragment_outputs_v13 import (
        TOP100_SCHEMA,
        TOP500_SCHEMA,
        load_compact_replay_dataset_v13,
    )
    from .retrieval_fold_fused_zero_copy_run_v13 import (
        load_zero_copy_canonical_run_v13,
    )
    if functional_projection_loader is None:
        from .retrieval_fold_fused_fragment_runtime_v13 import (
            load_immutable_w1_functional_projection_v13,
        )

        functional_projection_loader = load_immutable_w1_functional_projection_v13

    build = load_build_fragment_dataset_v13(
        build_path,
        verify_hashes=True,
        verify_logical=verify_logical,
        logical_hash_provider=logical_hash_provider,
    )
    replay = load_compact_replay_dataset_v13(
        replay_path,
        verify_hashes=True,
        verify_logical=verify_logical,
        logical_hash_provider=logical_hash_provider,
    )
    canonical = load_zero_copy_canonical_run_v13(
        canonical_path,
        verify_hashes=True,
        verify_logical=verify_logical,
        logical_hash_provider=logical_hash_provider,
    )
    projection = functional_projection_loader(
        projection_path,
        expected_manifest_sha256=projection_sha,
        expected_anchor_trial_sha256=expected_anchor_trial_sha256,
        verify_inputs=False,
    )
    if (
        build.manifest_sha256 != build_sha
        or replay.manifest_sha256 != replay_sha
        or canonical.manifest_sha256 != canonical_sha
        or tuple(build.specs) != tuple(specs)
        or tuple(replay.specs) != tuple(specs)
        or evidence.get("prefit_parity_manifest_sha256")
        != projection.prefit_parity_manifest_sha256
        or not _valid_sha256(evidence.get("sharded_execution_plan_sha256"))
        or lineage.get("prefit_parity_evidence_sha256")
        not in {None, projection.prefit_parity_manifest_sha256}
        or canonical.query_count != SAMPLE_QUERY_COUNT
        or list(TOP500_SCHEMA.names)
        != [
            "query_id",
            "fold",
            "candidate_siret",
            "candidate_parent",
            "rank",
            "site_score",
            "selection_reason",
        ]
        or "payload_json" in TOP500_SCHEMA.names
        or TOP100_SCHEMA.names[-1] != "payload_json"
    ):
        raise PhysicalBenchmarkInvariantErrorV13("v13 system artifact lineage drifted")
    for family, dataset in canonical.canonical_datasets.items():
        record = dataset_records[family]
        relative = Path(str(record["path"])).as_posix()
        if (
            (path.parent / relative).resolve() != dataset.path.resolve()
            or record["sha256"] != dataset.sha256
            or record["size_bytes"] != dataset.size_bytes
            or record["merkle_root"] != dataset.merkle_root
            or record["rows"] != dataset.rows
            or record["fragment_count"] != dataset.fragment_count
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                f"W3 wrapper/canonical {family} drifted"
            )
    rehydration = evidence.get("compact_rehydration")
    if (
        not isinstance(rehydration, Mapping)
        or set(rehydration)
        != {
            "status",
            "top500_row_count",
            "feature_values_logical_merkle_root",
            "prerank500_full_payload_logical_merkle_root",
        }
        or rehydration.get("status")
        != "PENDING_ONE_INDEPENDENT_POST_TIMING_RECONSTRUCTION"
        or type(rehydration.get("top500_row_count")) is not int
        or int(rehydration["top500_row_count"]) <= 0
        or rehydration.get("feature_values_logical_merkle_root")
        != replay.feature_values_logical_merkle_root
        or rehydration.get("prerank500_full_payload_logical_merkle_root")
        != replay.prerank500_full_payload_logical_merkle_root
    ):
        raise PhysicalBenchmarkInvariantErrorV13("compact rehydration evidence drifted")
    declared_top500_rows = sum(
        value.row_count for value in replay.outputs["top500"]
    )
    if int(rehydration["top500_row_count"]) != declared_top500_rows:
        raise PhysicalBenchmarkInvariantErrorV13(
            "compact rehydration row population drifted"
        )
    compact_rehydration_exact: bool | None = None
    if verify_rehydration:
        from .retrieval_fold_fused_sharded_runtime_v12 import (
            _ReplayFeatureContextV12,
        )
        from .retrieval_hierarchical_admission_v12 import AdmissionConfigV12

        admission = AdmissionConfigV12.load(
            Path(workspace_root).resolve()
            / "config/retrieval_all_human_v12.json"
        )
        observed_rows, _observed_rehydration = _validate_compact_rehydration_v13(
            build,
            replay,
            feature_context=_ReplayFeatureContextV12(admission),
            logical_hash_provider=logical_hash_provider,
        )
        if observed_rows != declared_top500_rows:
            raise PhysicalBenchmarkInvariantErrorV13(
                "compact rehydration proof is not reproducible"
            )
        compact_rehydration_exact = True
    projected_storage = math.ceil(
        materialized * NATIONAL_QUERY_COUNT / SAMPLE_QUERY_COUNT
    )
    product = FragmentW3TrialProductV13(
        manifest_path=path,
        manifest_sha256=expected,
        query_count=SAMPLE_QUERY_COUNT,
        folds=ALLOWED_FOLDS,
        v13_implementation_sha256=current_implementation,
        sharded_execution_plan_sha256=str(
            evidence["sharded_execution_plan_sha256"]
        ),
        prefit_parity_manifest_sha256=str(
            evidence["prefit_parity_manifest_sha256"]
        ),
        lineage=lineage,
        logical_digests=logical,
        immutable_w1_logical_digests=projection.logical_digests,
        materialized_bytes=materialized,
        projected_national_storage_bytes=projected_storage,
        artifact_inventory_sha256=_sha256_json(raw["artifact_inventory"]),
        fragment_dataset_manifest_sha256=build_sha,
        replay_dataset_manifest_sha256=replay_sha,
        physical_merge_files_present=False,
        full_payload_outside_top100=False,
        compact_rehydration_exact=compact_rehydration_exact,
    )
    return product


def _reconcile_product_v13(
    claimed: FragmentW3TrialProductV13,
    observed: FragmentW3TrialProductV13,
) -> None:
    claimed_record = asdict(claimed)
    observed_record = asdict(observed)
    if claimed_record.get("compact_rehydration_exact") is None:
        claimed_record["compact_rehydration_exact"] = observed_record.get(
            "compact_rehydration_exact"
        )
    if claimed_record != observed_record:
        raise PhysicalBenchmarkInvariantErrorV13(
            "W3 runner product differs from its strictly reloaded wrapper"
        )


def _resource_record(resources: PhysicalResourceMetricsV12) -> Mapping[str, Any]:
    return asdict(resources)


def _logical_io_record(evidence: LogicalIOEvidenceV13) -> Mapping[str, Any]:
    return {
        "manifest_path": str(evidence.manifest_path),
        "manifest_sha256": evidence.manifest_sha256,
        "status": evidence.status,
        "gate_pass": evidence.gate_pass,
        "projection_sha256": evidence.projection_sha256,
        "unique_row_groups": evidence.unique_row_groups,
        "logical_compressed_bytes": evidence.logical_compressed_bytes,
        "cross_actor_overlap_count": evidence.cross_actor_overlap_count,
        "w3_union_equals_w1": evidence.w3_union_equals_w1,
        "logical_amplification_ratio": evidence.logical_amplification_ratio,
        "materialized_ratio": evidence.materialized_ratio,
    }


def _derive_physical_checks_v13(
    *,
    config: Mapping[str, Any],
    anchor: ImmutableW1AnchorV13,
    product: FragmentW3TrialProductV13,
    logical: LogicalIOEvidenceV13,
    resources: PhysicalResourceMetricsV12,
    compact_rehydration_exact: bool,
) -> tuple[dict[str, bool], float]:
    """Derive every physical gate from measured/reopened facts."""

    benchmark = config["benchmark_512"]
    io_config = config["cache_neutral_io_gate"]
    source_extent = io_config["source_extent"]
    materialized = io_config["materialized"]
    expected_projection = source_extent["production_projection"]["sha256"]
    speedup = anchor.wall_seconds / resources.wall_seconds
    materialized_crosscheck = (
        product.materialized_bytes * int(materialized["maximum_ratio_denominator"])
        <= anchor.materialized_bytes * int(materialized["maximum_ratio_numerator"])
    )
    io_exact = (
        logical.gate_pass
        and logical.projection_sha256 == expected_projection
        and logical.unique_row_groups
        == int(source_extent["sealed_sample_expected_unique_row_groups"])
        and logical.logical_compressed_bytes
        == int(source_extent["sealed_sample_expected_logical_compressed_bytes"])
        and logical.cross_actor_overlap_count == 0
        and logical.w3_union_equals_w1
        and logical.logical_amplification_ratio
        <= float(source_extent["logical_amplification_maximum"])
        and logical.materialized_ratio
        <= (
            int(materialized["maximum_ratio_numerator"])
            / int(materialized["maximum_ratio_denominator"])
        )
        and materialized_crosscheck
    )
    return (
        {
            "logical_bridge_exact": product.bridge_exact,
            "no_physical_merge_files": not product.physical_merge_files_present,
            "top500_compact_top100_full": not product.full_payload_outside_top100,
            "compact_rehydration_exact": compact_rehydration_exact,
            "logical_io_pass": io_exact,
            "ram_pass": (
                resources.peak_machine_used_memory_bytes
                <= int(benchmark["maximum_w3_peak_machine_used_memory_bytes"])
            ),
            "swap_pass": (
                resources.swap_out_delta_bytes
                == int(benchmark["required_w3_swapout_delta_bytes"])
            ),
            "speed_pass": (
                speedup >= float(benchmark["minimum_speedup"])
                and resources.wall_seconds
                <= float(benchmark["maximum_w3_wall_seconds"])
            ),
            "storage_pass": (
                product.projected_national_storage_bytes
                <= int(benchmark["maximum_projected_national_storage_bytes"])
            ),
        },
        speedup,
    )


def _decision_status(checks: Mapping[str, bool]) -> str:
    precedence = (
        ("logical_bridge_exact", "STOP_V13_PARITY"),
        ("no_physical_merge_files", "STOP_V13_ARTIFACT_PROTOCOL"),
        ("top500_compact_top100_full", "STOP_V13_PAYLOAD_PROTOCOL"),
        ("compact_rehydration_exact", "STOP_V13_PARITY"),
        ("logical_io_pass", "STOP_V13_IO"),
        ("ram_pass", "STOP_V13_RAM"),
        ("swap_pass", "STOP_V13_SWAP"),
        ("speed_pass", "STOP_V13_SPEED"),
        ("storage_pass", "STOP_V13_STORAGE"),
    )
    for key, status in precedence:
        if not checks.get(key, False):
            return status
    return "PASS_V13_PHYSICAL_BENCHMARK"


def _consume_one_shot_attempt_v13(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Atomically and irreversibly consume the sole preregistered W3 attempt."""

    benchmark = config.get("benchmark_512")
    if not isinstance(benchmark, Mapping):
        raise PhysicalBenchmarkInvariantErrorV13("v13 benchmark config is absent")
    canonical_output = Path(str(benchmark.get("canonical_output_dir") or "")).resolve()
    marker = Path(
        str(benchmark.get("attempt_consumed_marker_path") or "")
    ).resolve()
    output = Path(output_dir).resolve()
    if (
        output != canonical_output
        or marker.parent != canonical_output.parent
        or marker.name != ".retrieval_v13_w3_attempt_consumed.json"
        or output.exists()
        or not output.parent.is_dir()
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 W3 output is noncanonical or already exists"
        )
    payload = (
        json.dumps(
            {
                "schema_version": "sireto-retrieval-v13-one-shot-attempt-1",
                "status": "ATTEMPT_IRREVERSIBLY_CONSUMED_BEFORE_CALLBACK",
                "config_sha256": CONFIG_SHA256,
                "canonical_output_dir": str(canonical_output),
                "marker_path": str(marker),
                "retry_allowed": False,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 W3 one-shot attempt was already consumed"
        ) from error
    try:
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_descriptor = os.open(marker.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return marker
    except BaseException as error:
        # This process already won O_EXCL.  The caller must preserve the
        # consumed attempt as a durable STOP even when write/fsync/close fails.
        raise _OneShotAttemptConsumptionErrorV13(
            "v13 W3 marker failed after O_EXCL"
        ) from error


def _validate_one_shot_attempt_v13(
    *,
    config: Mapping[str, Any],
    benchmark_output_dir: Path,
    attempt_marker_path: Path,
    expected_attempt_marker_sha256: str | None = None,
) -> str:
    """Validate the exact consumed capability before any W3 callback/build."""

    benchmark = config.get("benchmark_512")
    if not isinstance(benchmark, Mapping):
        raise PhysicalBenchmarkInvariantErrorV13("v13 benchmark config is absent")
    canonical_output = Path(str(benchmark.get("canonical_output_dir") or "")).resolve()
    configured_marker = Path(
        str(benchmark.get("attempt_consumed_marker_path") or "")
    ).resolve()
    output = Path(benchmark_output_dir).resolve()
    marker = Path(attempt_marker_path).resolve()
    if (
        output != canonical_output
        or marker != configured_marker
        or not marker.is_file()
        or marker.is_symlink()
        or marker.stat().st_nlink != 1
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 W3 lacks the canonical consumed one-shot capability"
        )
    marker_sha = sha256_file(marker)
    if (
        expected_attempt_marker_sha256 is not None
        and marker_sha != expected_attempt_marker_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 W3 one-shot capability hash drifted"
        )
    raw = _read_json(marker, label="v13 W3 one-shot attempt marker")
    expected = {
        "schema_version": "sireto-retrieval-v13-one-shot-attempt-1",
        "status": "ATTEMPT_IRREVERSIBLY_CONSUMED_BEFORE_CALLBACK",
        "config_sha256": CONFIG_SHA256,
        "canonical_output_dir": str(canonical_output),
        "marker_path": str(configured_marker),
        "retry_allowed": False,
    }
    if dict(raw) != expected:
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 W3 one-shot capability payload drifted"
        )
    return marker_sha


def _sanitized_exception_message_v13(error: BaseException) -> str:
    value = " ".join(str(error).split())[:240]
    value = re.sub(r"(?:https?|ftp)://\S+", "<redacted-url>", value)
    value = re.sub(r"/[^ ]+", "<redacted-path>", value)
    return value or "no exception message"


def _observed_attempt_marker_sha256_v13(marker: Path) -> str:
    try:
        if marker.is_file() and not marker.is_symlink():
            return sha256_file(marker)
    except BaseException:
        pass
    return "UNREADABLE_POST_O_EXCL"


def _publish_execution_exception_v13(
    *,
    output: Path,
    config_path: Path,
    marker: Path,
    workspace_root: Path,
    error: BaseException,
    frozen_implementation_sha256: str,
    frozen_implementation_files: Sequence[str],
    phase: str = "MEASURED_W3_CALLBACK_OR_POSTCHECK",
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    try:
        current_sha, _current_files = v13_implementation_lineage(workspace_root)
    except BaseException:
        current_sha = None
    manifest = {
        "schema_version": "sireto-retrieval-v13-execution-exception-1",
        "status": "STOP_V13_EXECUTION_EXCEPTION",
        "phase": phase,
        "config": {"path": str(config_path), "sha256": CONFIG_SHA256},
        "v13_implementation_sha256": frozen_implementation_sha256,
        "implementation_files": list(frozen_implementation_files),
        "post_error_current_implementation_sha256": current_sha,
        "attempt_marker": {
            "path": str(marker),
            "sha256": _observed_attempt_marker_sha256_v13(marker),
        },
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "execution_authorization_published": False,
        "automatic_retry_allowed": False,
        "exception": {
            "type": type(error).__name__,
            "message_sanitized": _sanitized_exception_message_v13(error),
        },
    }
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    descriptor = os.open(manifest_path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _consume_one_shot_or_publish_stop_v13(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output: Path,
    workspace_root: Path,
    frozen_implementation_sha256: str,
    frozen_implementation_files: Sequence[str],
) -> Path:
    """Consume O_EXCL or publish a STOP for any post-creation failure."""

    benchmark = config.get("benchmark_512")
    marker = Path(
        str((benchmark or {}).get("attempt_consumed_marker_path") or "")
    ).resolve()
    try:
        return _consume_one_shot_attempt_v13(
            config=config,
            output_dir=output,
        )
    except _OneShotAttemptConsumptionErrorV13 as error:
        # The private error is emitted only after this process obtained O_EXCL.
        # Publishing the STOP at the canonical output makes the failed attempt
        # non-retryable even if the marker bytes themselves are incomplete.
        _publish_execution_exception_v13(
            output=output,
            config_path=config_path,
            marker=marker,
            workspace_root=workspace_root,
            error=error,
            frozen_implementation_sha256=frozen_implementation_sha256,
            frozen_implementation_files=frozen_implementation_files,
            phase="ATOMIC_ATTEMPT_MARKER_CREATION",
        )
        raise


def _assess_physical_benchmark_v13(
    *,
    config_path: Path,
    output_dir: Path,
    attempt_marker: Path,
    runner: W3TrialRunnerV13,
    logical_io_builder: LogicalIOBuilderV13,
    measurer: TrialMeasurerV13,
    frozen_implementation_sha256: str,
    frozen_implementation_files: Sequence[str],
) -> _PhysicalBenchmarkAssessmentV13:
    """Measure and validate evidence without publishing PASS or authorization.

    This deliberately callback-capable layer is publication-inert.  The test
    harness can exercise it, while the official consumed DAG owns the only
    path from a successful assessment to an authorization.
    """

    config_resolved = Path(config_path).resolve()
    config = load_physical_config_v13(config_resolved)
    anchor = load_immutable_w1_anchor_v13(config)
    output = Path(output_dir).resolve()
    marker = Path(attempt_marker).resolve()
    marker_sha = _validate_one_shot_attempt_v13(
        config=config,
        benchmark_output_dir=output,
        attempt_marker_path=marker,
    )
    if output.exists():
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 W3 canonical output appeared after attempt consumption"
        )
    output.mkdir()

    claimed_product, resources = measurer(
        lambda: runner(output / "trial_w3", anchor)
    )
    if not isinstance(claimed_product, FragmentW3TrialProductV13):
        raise PhysicalBenchmarkInvariantErrorV13(
            "W3 runner returned another product type"
        )
    expected_trial_manifest = output / "trial_w3" / "manifest.json"
    if Path(claimed_product.manifest_path).resolve() != expected_trial_manifest:
        raise PhysicalBenchmarkInvariantErrorV13(
            "W3 runner returned a noncanonical external trial product"
        )
    # The measured callback may only seal the replay-native feature digest.
    # Exactly one independent 319-feature/top500 reconstruction happens here,
    # after the resource timer has stopped.
    product = load_fragment_w3_trial_product_v13(
        claimed_product.manifest_path,
        workspace_root=config_resolved.parent.parent,
        expected_manifest_sha256=claimed_product.manifest_sha256,
        expected_anchor_trial_sha256=anchor.trial_manifest_sha256,
        verify_logical=True,
        verify_rehydration=True,
    )
    _reconcile_product_v13(claimed_product, product)
    current_implementation_sha, current_implementation_files = (
        v13_implementation_lineage(config_resolved.parent.parent)
    )
    if (
        product.v13_implementation_sha256 != frozen_implementation_sha256
        or current_implementation_sha != frozen_implementation_sha256
        or tuple(current_implementation_files)
        != tuple(frozen_implementation_files)
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 implementation changed after one-shot capability consumption"
        )
    plan_authority = config["authorities"]["v12_sharded_plan"]
    plan_path = (
        config_resolved.parent.parent / str(plan_authority["path"])
    ).resolve()
    plan_raw = _read_json(plan_path, label="v13 frozen sharded plan")
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
        raise PhysicalBenchmarkInvariantErrorV13(
            "W3 trial plan/PREFIT differ from frozen v12 authorities"
        )
    logical = logical_io_builder(product, anchor, output / "logical_io")
    if not isinstance(logical, LogicalIOEvidenceV13):
        raise PhysicalBenchmarkInvariantErrorV13(
            "logical-I/O builder returned another evidence type"
        )

    checks, speedup = _derive_physical_checks_v13(
        config=config,
        anchor=anchor,
        product=product,
        logical=logical,
        resources=resources,
        compact_rehydration_exact=product.compact_rehydration_exact is True,
    )
    identity = {
        "engine_id": ENGINE_ID,
        "config_sha256": CONFIG_SHA256,
        "v13_implementation_sha256": product.v13_implementation_sha256,
        "immutable_w1_benchmark_sha256": anchor.benchmark_manifest_sha256,
        "immutable_w1_trial_sha256": anchor.trial_manifest_sha256,
        "w3_trial_manifest_sha256": product.manifest_sha256,
        "logical_io_manifest_sha256": logical.manifest_sha256,
        "artifact_inventory_sha256": product.artifact_inventory_sha256,
        "fragment_dataset_manifest_sha256": (
            product.fragment_dataset_manifest_sha256
        ),
        "replay_dataset_manifest_sha256": product.replay_dataset_manifest_sha256,
        "logical_digests": dict(product.logical_digests),
        "query_count": SAMPLE_QUERY_COUNT,
        "folds": list(ALLOWED_FOLDS),
        "worker_count": 3,
        "sharded_implementation_sha256": product.lineage[
            "sharded_implementation_sha256"
        ],
        "runtime_environment_sha256": product.lineage[
            "runtime_environment_sha256"
        ],
        "attempt_marker_path": str(marker),
        "attempt_marker_sha256": marker_sha,
        "sharded_execution_plan_sha256": (
            product.sharded_execution_plan_sha256
        ),
        "prefit_parity_manifest_sha256": (
            product.prefit_parity_manifest_sha256
        ),
    }
    return _PhysicalBenchmarkAssessmentV13(
        config_resolved=config_resolved,
        anchor=anchor,
        output=output,
        marker=marker,
        marker_sha256=marker_sha,
        product=product,
        resources=resources,
        logical=logical,
        speedup=speedup,
        checks=checks,
        identity=identity,
    )


def _benchmark_manifest_v13(
    assessment: _PhysicalBenchmarkAssessmentV13,
    *,
    status: str,
    authorization_published: bool,
) -> Mapping[str, Any]:
    """Build the common sealed verdict; this helper never writes authorization."""

    config_resolved = assessment.config_resolved
    anchor = assessment.anchor
    marker = assessment.marker
    marker_sha = assessment.marker_sha256
    product = assessment.product
    resources = assessment.resources
    logical = assessment.logical
    speedup = assessment.speedup
    checks = assessment.checks
    identity = assessment.identity
    pass_gate = status == "PASS_V13_PHYSICAL_BENCHMARK"
    if authorization_published is not pass_gate:
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 verdict and authorization publication disagree"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "benchmark_id": _sha256_json(identity),
        "benchmark_identity": identity,
        "run_kind": "PHYSICAL_ONLY_W3_VS_IMMUTABLE_W1",
        "quality_metrics_decision_blocking": False,
        "national_oof_launched": False,
        "config": {
            "path": str(config_resolved),
            "sha256": CONFIG_SHA256,
            "preregistered_commits": ["77838a1", "ee30390"],
        },
        "attempt_marker": {"path": str(marker), "sha256": marker_sha},
        "immutable_v12_stop_preserved": True,
        "immutable_w1": {
            "benchmark_manifest_path": str(anchor.benchmark_manifest_path),
            "benchmark_manifest_sha256": anchor.benchmark_manifest_sha256,
            "trial_manifest_path": str(anchor.trial_manifest_path),
            "trial_manifest_sha256": anchor.trial_manifest_sha256,
            "wall_seconds": anchor.wall_seconds,
            "materialized_bytes": anchor.materialized_bytes,
        },
        "w3": {
            "trial_manifest_path": str(product.manifest_path),
            "trial_manifest_sha256": product.manifest_sha256,
            "worker_count": product.worker_count,
            "query_count": product.query_count,
            "folds": list(product.folds),
            "fold0_opened": product.fold0_opened,
            "fold1_opened": product.fold1_opened,
            "positive_injection": product.positive_injection,
            "v13_implementation_sha256": product.v13_implementation_sha256,
            "lineage": dict(product.lineage),
            "logical_digests": dict(product.logical_digests),
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
            "materialized_bytes": product.materialized_bytes,
            "projected_national_storage_bytes": (
                product.projected_national_storage_bytes
            ),
            "resources": _resource_record(resources),
            "speedup_over_immutable_w1": speedup,
        },
        "non_circular_w1_bridge": {
            "reference_logical_digests": dict(
                product.immutable_w1_logical_digests
            ),
            "w3_logical_digests": dict(product.logical_digests),
            "byte_or_bit_exact": product.bridge_exact,
            "new_format_self_parity_alone_sufficient": False,
        },
        "logical_io": _logical_io_record(logical),
        "process_and_machine_io_decision_blocking": False,
        "checks": checks,
        "gate_pass": pass_gate,
        "execution_authorization_published": authorization_published,
    }


def _authorization_document_v13(
    assessment: _PhysicalBenchmarkAssessmentV13,
    *,
    benchmark_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Build an authorization only for the callback-free official publisher."""

    if assessment.authorization_capability is not _OFFICIAL_DAG_CAPABILITY_V13:
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 authorization lacks official callback-free DAG provenance"
        )
    product = assessment.product
    anchor = assessment.anchor
    logical = assessment.logical
    authorization_identity = {
            "engine_id": ENGINE_ID,
            "benchmark_manifest_sha256": benchmark_manifest_sha256,
            "config_sha256": CONFIG_SHA256,
            "v13_implementation_sha256": product.v13_implementation_sha256,
            "immutable_w1_benchmark_sha256": anchor.benchmark_manifest_sha256,
            "w3_trial_manifest_sha256": product.manifest_sha256,
            "logical_io_manifest_sha256": logical.manifest_sha256,
            "authorized_worker_count": 3,
            "authorized_scope": "NATIONAL_OOF_FOLDS_2_3_4_PHYSICAL_EXECUTION",
            "sharded_implementation_sha256": product.lineage[
                "sharded_implementation_sha256"
            ],
            "runtime_environment_sha256": product.lineage[
                "runtime_environment_sha256"
            ],
            "attempt_marker_path": str(assessment.marker),
            "attempt_marker_sha256": assessment.marker_sha256,
            "sharded_execution_plan_sha256": (
                product.sharded_execution_plan_sha256
            ),
            "prefit_parity_manifest_sha256": (
                product.prefit_parity_manifest_sha256
            ),
    }
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": "AUTHORIZED_V13_W3_FOR_NATIONAL_OOF",
        "authorization_id": _sha256_json(authorization_identity),
        "authorization_identity": authorization_identity,
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "automatic_national_launch": False,
    }


def _publish_untrusted_assessment_v13(
    assessment: _PhysicalBenchmarkAssessmentV13,
) -> Path:
    """Publish test evidence, permanently ineligible for PASS or authorization."""

    status = _decision_status(assessment.checks)
    if status == "PASS_V13_PHYSICAL_BENCHMARK":
        status = "STOP_V13_UNTRUSTED_EXECUTION_HOOKS"
    manifest_path = assessment.output / "manifest.json"
    _atomic_json(
        manifest_path,
        _benchmark_manifest_v13(
            assessment,
            status=status,
            authorization_published=False,
        ),
    )
    if (assessment.output / "execution_authorization.json").exists():
        raise PhysicalBenchmarkInvariantErrorV13(
            "untrusted v13 assessment unexpectedly published authorization"
        )
    return manifest_path


def _publish_official_assessment_v13(
    assessment: _PhysicalBenchmarkAssessmentV13,
) -> Path:
    """Publish the sole official verdict and, only on PASS, authorization."""

    if assessment.authorization_capability is not _OFFICIAL_DAG_CAPABILITY_V13:
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 official publisher refuses an injectable assessment"
        )
    status = _decision_status(assessment.checks)
    pass_gate = status == "PASS_V13_PHYSICAL_BENCHMARK"
    manifest_path = assessment.output / "manifest.json"
    _atomic_json(
        manifest_path,
        _benchmark_manifest_v13(
            assessment,
            status=status,
            authorization_published=pass_gate,
        ),
    )
    if pass_gate:
        _atomic_json(
            assessment.output / "execution_authorization.json",
            _authorization_document_v13(
                assessment,
                benchmark_manifest_sha256=sha256_file(manifest_path),
            ),
        )
    return manifest_path


def _run_physical_benchmark_test_harness_consumed_v13(
    *,
    config_resolved: Path,
    output: Path,
    marker: Path,
    runner: W3TrialRunnerV13,
    logical_io_builder: LogicalIOBuilderV13,
    measurer: TrialMeasurerV13,
    frozen_implementation_sha256: str,
    frozen_implementation_files: Sequence[str],
) -> Path:
    assessment = _assess_physical_benchmark_v13(
        config_path=config_resolved,
        output_dir=output,
        attempt_marker=marker,
        runner=runner,
        logical_io_builder=logical_io_builder,
        measurer=measurer,
        frozen_implementation_sha256=frozen_implementation_sha256,
        frozen_implementation_files=frozen_implementation_files,
    )
    return _publish_untrusted_assessment_v13(assessment)


def _run_physical_benchmark_consumed_v13(
    *,
    config_resolved: Path,
    output: Path,
    marker: Path,
    official_dag: _OfficialPhysicalDAGV13,
    frozen_implementation_sha256: str,
    frozen_implementation_files: Sequence[str],
) -> Path:
    """Run the sole authorization-capable DAG with no injectable callbacks."""

    if official_dag.capability is not _OFFICIAL_DAG_CAPABILITY_V13:
        raise PhysicalBenchmarkInvariantErrorV13(
            "official consumed W3 DAG lacks private preflight provenance"
        )
    config = load_physical_config_v13(config_resolved)
    prepared = official_dag.prepared
    root = Path(official_dag.workspace_root).resolve()
    w1_functional_projection = official_dag.w1_functional_projection
    w1_io_projection = official_dag.w1_io_projection
    if root != config_resolved.parent.parent:
        raise PhysicalBenchmarkInvariantErrorV13(
            "official W3 workspace differs from the preregistered repository"
        )
    marker_sha = _validate_one_shot_attempt_v13(
        config=config,
        benchmark_output_dir=output,
        attempt_marker_path=marker,
    )

    def official_runner(trial_dir: Path, anchor: ImmutableW1AnchorV13):
        return execute_fragment_w3_trial_v13(
            prepared=prepared,
            workspace_root=root,
            output_dir=trial_dir,
            anchor=anchor,
            w1_functional_projection=w1_functional_projection,
            attempt_marker_path=marker,
            attempt_marker_sha256=marker_sha,
        )

    def official_logical_io_builder(
        product: FragmentW3TrialProductV13,
        _anchor: ImmutableW1AnchorV13,
        logical_output: Path,
    ) -> LogicalIOEvidenceV13:
        from .retrieval_fold_fused_logical_io_v13 import (
            compare_w3_to_immutable_w1_io_v13,
        )

        artifact = compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1_io_projection.manifest_path,
            w1_projection_manifest_sha256=w1_io_projection.manifest_sha256,
            w3_run_manifest_path=product.manifest_path,
            w3_run_manifest_sha256=product.manifest_sha256,
            output_root=logical_output,
        )
        return logical_io_evidence_from_gate_v13(artifact)

    assessment = _assess_physical_benchmark_v13(
        config_path=config_resolved,
        output_dir=output,
        attempt_marker=marker,
        runner=official_runner,
        logical_io_builder=official_logical_io_builder,
        measurer=measure_physical_trial_v12,
        frozen_implementation_sha256=frozen_implementation_sha256,
        frozen_implementation_files=frozen_implementation_files,
    )
    official_assessment = replace(
        assessment,
        authorization_capability=_OFFICIAL_DAG_CAPABILITY_V13,
    )
    return _publish_official_assessment_v13(official_assessment)


def run_physical_benchmark_test_harness_v13(
    *,
    config_path: Path,
    output_dir: Path,
    runner: W3TrialRunnerV13,
    logical_io_builder: LogicalIOBuilderV13,
    measurer: TrialMeasurerV13,
) -> Path:
    """Exercise gates with injected hooks, permanently ineligible for PASS/auth."""

    config_resolved = Path(config_path).resolve()
    config = load_physical_config_v13(config_resolved)
    frozen_sha, frozen_files = v13_implementation_lineage(
        config_resolved.parent.parent
    )
    output = Path(output_dir).resolve()
    marker = _consume_one_shot_or_publish_stop_v13(
        config=config,
        config_path=config_resolved,
        output=output,
        workspace_root=config_resolved.parent.parent,
        frozen_implementation_sha256=frozen_sha,
        frozen_implementation_files=frozen_files,
    )
    try:
        return _run_physical_benchmark_test_harness_consumed_v13(
            config_resolved=config_resolved,
            output=output,
            marker=marker,
            runner=runner,
            logical_io_builder=logical_io_builder,
            measurer=measurer,
            frozen_implementation_sha256=frozen_sha,
            frozen_implementation_files=frozen_files,
        )
    except BaseException as error:
        _publish_execution_exception_v13(
            output=output,
            config_path=config_resolved,
            marker=marker,
            workspace_root=config_resolved.parent.parent,
            error=error,
            frozen_implementation_sha256=frozen_sha,
            frozen_implementation_files=frozen_files,
        )
        raise


def run_physical_benchmark_v13(
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
    """Own cache-neutral preflight and run the non-injectable official W3 DAG."""

    config_resolved = Path(config_path).resolve()
    config = load_physical_config_v13(config_resolved)
    root = Path(workspace_root).resolve()
    if root != config_resolved.parent.parent:
        raise PhysicalBenchmarkInvariantErrorV13(
            "official W3 workspace differs from the preregistered repository"
        )
    # The authorization-capable API owns preparation and hardcodes the only
    # admissible pre-W3 source policy.  A generic Prepared object cannot be
    # supplied by callers after a cache-warming full source hash.
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
    output = require_external_sharded_output_v12(
        output_dir, executor=prepared.executor
    )
    anchor = load_immutable_w1_anchor_v13(config)
    support_root = Path(immutable_projection_root).resolve()
    from .retrieval_fold_fused_fragment_runtime_v13 import (
        build_immutable_w1_functional_projection_v13,
        load_immutable_w1_functional_projection_v13,
    )
    from .retrieval_fold_fused_logical_io_v13 import (
        build_immutable_w1_io_projection_v13,
        load_immutable_w1_io_projection_v13,
    )

    w1_functional_projection = build_immutable_w1_functional_projection_v13(
        anchor=anchor,
        workspace_root=root,
        output_root=support_root / "functional",
        prefit_parity_manifest_path=prepared.prefit_parity_manifest_path,
        expected_prefit_parity_manifest_sha256=(
            prepared.prefit_parity_manifest_sha256
        ),
        expected_prefit_lineage=prepared.lineage,
        sample_query_ids=prepared.sample_query_ids,
    )
    w1_functional_projection = load_immutable_w1_functional_projection_v13(
        w1_functional_projection.manifest_path,
        expected_manifest_sha256=w1_functional_projection.manifest_sha256,
        expected_anchor_trial_sha256=anchor.trial_manifest_sha256,
        verify_inputs=True,
    )
    built_w1_io_projection = build_immutable_w1_io_projection_v13(
        physical_config_path=config_resolved,
        physical_config_sha256=CONFIG_SHA256,
        output_root=support_root / "logical_io",
    )
    w1_io_projection, _raw_io, _rows_io, _baseline_io = (
        load_immutable_w1_io_projection_v13(
            built_w1_io_projection.manifest_path,
            expected_manifest_sha256=built_w1_io_projection.manifest_sha256,
            verify_source_hash=False,
        )
    )
    frozen_sha, frozen_files = v13_implementation_lineage(root)
    marker = _consume_one_shot_or_publish_stop_v13(
        config=config,
        config_path=config_resolved,
        output=output,
        workspace_root=root,
        frozen_implementation_sha256=frozen_sha,
        frozen_implementation_files=frozen_files,
    )
    official_dag = _OfficialPhysicalDAGV13(
        prepared=prepared,
        workspace_root=root,
        w1_functional_projection=w1_functional_projection,
        w1_io_projection=w1_io_projection,
        capability=_OFFICIAL_DAG_CAPABILITY_V13,
    )
    try:
        return _run_physical_benchmark_consumed_v13(
            config_resolved=config_resolved,
            output=output,
            marker=marker,
            official_dag=official_dag,
            frozen_implementation_sha256=frozen_sha,
            frozen_implementation_files=frozen_files,
        )
    except BaseException as error:
        _publish_execution_exception_v13(
            output=output,
            config_path=config_resolved,
            marker=marker,
            workspace_root=root,
            error=error,
            frozen_implementation_sha256=frozen_sha,
            frozen_implementation_files=frozen_files,
        )
        raise


def _build_jobs_v13(
    *,
    prepared: PreparedShardedExecutionV12,
    specs: Sequence[Any],
    output_dir: Path,
    lineage: Mapping[str, str],
    miniature_fit: bool = True,
    execution_phase: str = "PHYSICAL_V13_W3_ONLY_BENCHMARK_512",
    physical_config_sha256: str = CONFIG_SHA256,
) -> tuple[WorkerJobV12, ...]:
    return tuple(
        WorkerJobV12(
            job_id=f"build-{spec.shard_id}",
            payload={
                "phase": "BUILD",
                "spec": spec.as_manifest(),
                "common": dict(prepared.runtime_payload),
                "lineage": dict(lineage),
                "caps": asdict(prepared.caps),
                "maximum_spool_bytes": (
                    prepared.executor.maximum_temporary_gib * GIB
                ),
                "output_dir": str(output_dir / "worker_build" / spec.shard_id),
                "miniature_fit": bool(miniature_fit),
                "execution_evidence": {
                    "phase": execution_phase,
                    "physical_config_sha256": physical_config_sha256,
                    "sharded_execution_plan_sha256": (
                        prepared.sharded_execution_plan_sha256
                    ),
                },
            },
        )
        for spec in specs
    )


def execute_fragment_w3_trial_v13(
    *,
    prepared: PreparedShardedExecutionV12,
    workspace_root: Path,
    output_dir: Path,
    anchor: ImmutableW1AnchorV13,
    trial_finisher: Callable[..., FragmentW3TrialProductV13] | None = None,
    w1_functional_projection: Any | None = None,
    attempt_marker_path: Path | None = None,
    attempt_marker_sha256: str | None = None,
) -> FragmentW3TrialProductV13:
    """Execute W3 build actors and delegate fragment fit/replay publication.

    Build is implemented here because it is already a stable v12 worker
    boundary.  ``trial_finisher`` is the v13 fragment adapter: it receives the
    sealed build artifacts after every build process exited, and must perform
    zero-copy sealing, sequential fit, compact replay and the W1 bridge.  The
    indirection keeps this orchestrator import-safe while the adapter remains
    independently testable.
    """

    root = Path(workspace_root).resolve()
    config = load_physical_config_v13(
        root / "config/retrieval_all_human_v13_physical_execution.json"
    )
    canonical_output = Path(
        str(config["benchmark_512"]["canonical_output_dir"])
    ).resolve()
    if (
        attempt_marker_path is None
        or attempt_marker_sha256 is None
        or Path(output_dir).resolve() != canonical_output / "trial_w3"
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "direct W3 execution without the sealed one-shot capability is forbidden"
        )
    _validate_one_shot_attempt_v13(
        config=config,
        benchmark_output_dir=canonical_output,
        attempt_marker_path=attempt_marker_path,
        expected_attempt_marker_sha256=attempt_marker_sha256,
    )
    if trial_finisher is None and w1_functional_projection is None:
        raise PhysicalBenchmarkInvariantErrorV13(
            "immutable W1 functional projection must be built before measured W3"
        )
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    lineage = dict(sharded_execution_lineage_v12(
        dict(prepared.lineage),
        workspace_root=root,
    ))
    implementation_sha, _implementation_files = v13_implementation_lineage(
        workspace_root
    )
    lineage["v13_implementation_sha256"] = implementation_sha
    specs = plan_query_shards(
        prepared.scene_cache.index,
        prepared.sample_query_ids,
        3,
    )
    if (
        len(specs) != 3
        or tuple(spec.folds for spec in specs) != ((2,), (3,), (4,))
        or sum(spec.query_count for spec in specs) != SAMPLE_QUERY_COUNT
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "W3 sample did not produce exactly one fold-owned shard per actor"
        )
    jobs = _build_jobs_v13(
        prepared=prepared,
        specs=specs,
        output_dir=output,
        lineage=lineage,
    )
    manifests = spawn_manifest_workers_v12(
        BUILD_ENTRYPOINT_V13,
        jobs,
        max_workers=3,
    )
    builds = tuple(load_shard_build_manifest_v12(path) for path in manifests)
    if (
        len(builds) != 3
        or tuple(value.spec.digest for value in builds)
        != tuple(value.digest for value in specs)
    ):
        raise PhysicalBenchmarkInvariantErrorV13("W3 build artifact set drifted")
    if trial_finisher is None:
        try:
            from .retrieval_fold_fused_fragment_runtime_v13 import (
                finish_fragment_w3_trial_v13,
            )
        except ImportError as error:
            raise PhysicalBenchmarkInvariantErrorV13(
                "v13 compact fragment runtime is unavailable"
            ) from error
        trial_finisher = finish_fragment_w3_trial_v13
    return trial_finisher(
        prepared=prepared,
        specs=specs,
        build_artifacts=builds,
        lineage=lineage,
        output_dir=output,
        anchor=anchor,
        workspace_root=Path(workspace_root).resolve(),
        w1_functional_projection=w1_functional_projection,
    )


def load_execution_authorization_v13(
    path: Path,
    *,
    expected_authorization_sha256: str,
    benchmark_manifest_path: Path,
    expected_benchmark_manifest_sha256: str,
    workspace_root: Path,
    verify_artifacts: bool = True,
) -> Mapping[str, Any]:
    """Reopen authorization, PASS benchmark and measured implementation.

    This is the only admissible bridge to a national run.  Supplying a
    self-consistent authorization JSON is insufficient: the exact benchmark,
    W3 wrapper, logical-I/O evidence and current v13 bytes are all rebound.
    """

    if verify_artifacts is not True:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national authorization cannot skip measured artifact verification"
        )
    resolved = Path(path).resolve()
    benchmark_path = Path(benchmark_manifest_path).resolve()
    if (
        not resolved.is_file()
        or not _valid_sha256(expected_authorization_sha256)
        or sha256_file(resolved) != expected_authorization_sha256
        or not benchmark_path.is_file()
        or not _valid_sha256(expected_benchmark_manifest_sha256)
        or sha256_file(benchmark_path) != expected_benchmark_manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 authorization or benchmark bytes drifted"
        )
    raw = _read_json(resolved, label="v13 execution authorization")
    identity = raw.get("authorization_identity")
    if (
        set(raw) != _AUTHORIZATION_KEYS
        or
        raw.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION
        or raw.get("status") != "AUTHORIZED_V13_W3_FOR_NATIONAL_OOF"
        or not isinstance(identity, Mapping)
        or set(identity) != _AUTHORIZATION_IDENTITY_KEYS
        or raw.get("authorization_id") != _sha256_json(identity)
        or identity.get("benchmark_manifest_sha256")
        != expected_benchmark_manifest_sha256
        or identity.get("config_sha256") != CONFIG_SHA256
        or identity.get("authorized_worker_count") != 3
        or raw.get("folds") != list(ALLOWED_FOLDS)
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or raw.get("automatic_national_launch") is not False
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 execution authorization drifted"
        )
    benchmark = _read_json(benchmark_path, label="v13 physical benchmark")
    benchmark_identity = benchmark.get("benchmark_identity")
    checks = benchmark.get("checks")
    w3 = benchmark.get("w3")
    logical_io = benchmark.get("logical_io")
    immutable = benchmark.get("immutable_w1")
    attempt = benchmark.get("attempt_marker")
    if (
        set(benchmark) != _BENCHMARK_KEYS
        or benchmark.get("schema_version") != SCHEMA_VERSION
        or benchmark.get("status") != "PASS_V13_PHYSICAL_BENCHMARK"
        or benchmark.get("run_kind") != "PHYSICAL_ONLY_W3_VS_IMMUTABLE_W1"
        or benchmark.get("gate_pass") is not True
        or benchmark.get("execution_authorization_published") is not True
        or benchmark.get("national_oof_launched") is not False
        or benchmark.get("quality_metrics_decision_blocking") is not False
        or benchmark.get("immutable_v12_stop_preserved") is not True
        or not isinstance(benchmark_identity, Mapping)
        or set(benchmark_identity) != _BENCHMARK_IDENTITY_KEYS
        or benchmark.get("benchmark_id") != _sha256_json(benchmark_identity)
        or not isinstance(checks, Mapping)
        or set(checks)
        != {
            "logical_bridge_exact",
            "no_physical_merge_files",
            "top500_compact_top100_full",
            "compact_rehydration_exact",
            "logical_io_pass",
            "ram_pass",
            "swap_pass",
            "speed_pass",
            "storage_pass",
        }
        or not all(value is True for value in checks.values())
        or not isinstance(w3, Mapping)
        or set(w3) != _BENCHMARK_W3_KEYS
        or not isinstance(logical_io, Mapping)
        or not isinstance(immutable, Mapping)
        or not isinstance(attempt, Mapping)
        or set(attempt) != {"path", "sha256"}
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national authorization benchmark is not an exact all-gates PASS"
        )
    current_implementation, _files = v13_implementation_lineage(workspace_root)
    if (
        identity.get("engine_id") != ENGINE_ID
        or identity.get("config_sha256") != CONFIG_SHA256
        or identity.get("v13_implementation_sha256") != current_implementation
        or benchmark_identity.get("engine_id") != ENGINE_ID
        or benchmark_identity.get("config_sha256") != CONFIG_SHA256
        or benchmark_identity.get("v13_implementation_sha256")
        != current_implementation
        or identity.get("immutable_w1_benchmark_sha256")
        != benchmark_identity.get("immutable_w1_benchmark_sha256")
        or identity.get("w3_trial_manifest_sha256")
        != benchmark_identity.get("w3_trial_manifest_sha256")
        or identity.get("logical_io_manifest_sha256")
        != benchmark_identity.get("logical_io_manifest_sha256")
        or identity.get("sharded_implementation_sha256")
        != benchmark_identity.get("sharded_implementation_sha256")
        or identity.get("runtime_environment_sha256")
        != benchmark_identity.get("runtime_environment_sha256")
        or identity.get("attempt_marker_path")
        != benchmark_identity.get("attempt_marker_path")
        or identity.get("attempt_marker_sha256")
        != benchmark_identity.get("attempt_marker_sha256")
        or identity.get("sharded_execution_plan_sha256")
        != benchmark_identity.get("sharded_execution_plan_sha256")
        or identity.get("prefit_parity_manifest_sha256")
        != benchmark_identity.get("prefit_parity_manifest_sha256")
        or identity.get("authorized_scope")
        != "NATIONAL_OOF_FOLDS_2_3_4_PHYSICAL_EXECUTION"
        or benchmark_identity.get("worker_count") != 3
        or benchmark_identity.get("query_count") != SAMPLE_QUERY_COUNT
        or benchmark_identity.get("folds") != list(ALLOWED_FOLDS)
        or w3.get("v13_implementation_sha256") != current_implementation
        or w3.get("trial_manifest_sha256")
        != benchmark_identity.get("w3_trial_manifest_sha256")
        or (w3.get("lineage") or {}).get("sharded_implementation_sha256")
        != benchmark_identity.get("sharded_implementation_sha256")
        or (w3.get("lineage") or {}).get("runtime_environment_sha256")
        != benchmark_identity.get("runtime_environment_sha256")
        or logical_io.get("manifest_sha256")
        != benchmark_identity.get("logical_io_manifest_sha256")
        or logical_io.get("status") != "PASS_V13_IO"
        or logical_io.get("gate_pass") is not True
        or immutable.get("benchmark_manifest_sha256")
        != benchmark_identity.get("immutable_w1_benchmark_sha256")
        or immutable.get("trial_manifest_sha256")
        != benchmark_identity.get("immutable_w1_trial_sha256")
        or attempt.get("path") != benchmark_identity.get("attempt_marker_path")
        or attempt.get("sha256") != benchmark_identity.get("attempt_marker_sha256")
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 authorization, benchmark and implementation are not identical"
        )
    # Reopen the pinned config and immutable STOP, even if the caller already
    # did so during the benchmark process.
    config = load_physical_config_v13(
        Path(workspace_root).resolve()
        / "config/retrieval_all_human_v13_physical_execution.json"
    )
    anchor = load_immutable_w1_anchor_v13(config)
    expected_immutable_record = {
        "benchmark_manifest_path": str(anchor.benchmark_manifest_path),
        "benchmark_manifest_sha256": anchor.benchmark_manifest_sha256,
        "trial_manifest_path": str(anchor.trial_manifest_path),
        "trial_manifest_sha256": anchor.trial_manifest_sha256,
        "wall_seconds": anchor.wall_seconds,
        "materialized_bytes": anchor.materialized_bytes,
    }
    plan_authority = config["authorities"]["v12_sharded_plan"]
    plan_path = (
        Path(workspace_root).resolve() / str(plan_authority["path"])
    ).resolve()
    plan_raw = _read_json(plan_path, label="v13 authorized sharded plan")
    frozen_prefit_sha = str(
        (((plan_raw.get("input_lineage") or {}).get("prefit_parity") or {}).get(
            "manifest_sha256"
        ))
        or ""
    )
    benchmark_config = config["benchmark_512"]
    canonical_output = Path(str(benchmark_config["canonical_output_dir"])).resolve()
    marker_path = Path(str(benchmark_config["attempt_consumed_marker_path"])).resolve()
    if (
        dict(immutable) != expected_immutable_record
        or anchor.benchmark_manifest_sha256
        != benchmark_identity["immutable_w1_benchmark_sha256"]
        or anchor.trial_manifest_sha256
        != benchmark_identity["immutable_w1_trial_sha256"]
        or benchmark_path != canonical_output / "manifest.json"
        or resolved != canonical_output / "execution_authorization.json"
        or marker_path != Path(str(attempt.get("path") or "")).resolve()
        or not marker_path.is_file()
        or marker_path.stat().st_nlink != 1
        or sha256_file(marker_path) != attempt.get("sha256")
        or identity.get("sharded_execution_plan_sha256")
        != plan_authority["sha256"]
        or identity.get("prefit_parity_manifest_sha256")
        != frozen_prefit_sha
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 authorization no longer binds the immutable W1 anchor"
        )
    trial_path = Path(str(w3.get("trial_manifest_path") or "")).resolve()
    logical_path = Path(str(logical_io.get("manifest_path") or "")).resolve()
    if (
        not trial_path.is_file()
        or sha256_file(trial_path) != w3["trial_manifest_sha256"]
        or not logical_path.is_file()
        or sha256_file(logical_path) != logical_io["manifest_sha256"]
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 authorization measured artifacts changed"
        )
    product = load_fragment_w3_trial_product_v13(
        trial_path,
        workspace_root=workspace_root,
        expected_manifest_sha256=str(w3["trial_manifest_sha256"]),
        expected_anchor_trial_sha256=anchor.trial_manifest_sha256,
        verify_logical=True,
        verify_rehydration=False,
    )
    if (
        dict(product.lineage) != dict(w3.get("lineage") or {})
        or product.artifact_inventory_sha256
        != benchmark_identity["artifact_inventory_sha256"]
        or product.fragment_dataset_manifest_sha256
        != benchmark_identity["fragment_dataset_manifest_sha256"]
        or product.replay_dataset_manifest_sha256
        != benchmark_identity["replay_dataset_manifest_sha256"]
        or product.sharded_execution_plan_sha256
        != benchmark_identity["sharded_execution_plan_sha256"]
        or product.prefit_parity_manifest_sha256
        != benchmark_identity["prefit_parity_manifest_sha256"]
        or dict(product.logical_digests)
        != dict(benchmark_identity["logical_digests"])
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 authorization W3 wrapper differs from benchmark identity"
        )
    from .retrieval_fold_fused_logical_io_v13 import load_logical_io_gate_v13

    gate = load_logical_io_gate_v13(
        logical_path,
        expected_manifest_sha256=str(logical_io["manifest_sha256"]),
        verify_source_hash=True,
    )
    if not gate.gate_pass or gate.status != "PASS_V13_IO":
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 authorization logical-I/O gate is no longer reproducible"
        )
    logical_evidence = logical_io_evidence_from_gate_v13(gate)
    resources_raw = w3.get("resources")
    resource_keys = set(PhysicalResourceMetricsV12.__dataclass_fields__)
    rehydration = w3.get("compact_rehydration")
    if (
        not isinstance(resources_raw, Mapping)
        or set(resources_raw) != resource_keys
        or not isinstance(rehydration, Mapping)
        or set(rehydration)
        != {
            "status",
            "exact",
            "site_feature_values",
            "prerank500_full_payload",
        }
        or rehydration.get("status")
        != "PASS_ONE_INDEPENDENT_POST_TIMING_RECONSTRUCTION"
        or rehydration.get("exact") is not True
        or rehydration.get("site_feature_values")
        != product.logical_digests["site_feature_values"]
        or rehydration.get("prerank500_full_payload")
        != product.logical_digests["prerank500_full_payload"]
        or dict(logical_io) != dict(_logical_io_record(logical_evidence))
        or w3.get("materialized_bytes") != product.materialized_bytes
        or w3.get("projected_national_storage_bytes")
        != product.projected_national_storage_bytes
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 benchmark measured facts are not artifact-derived"
        )
    try:
        resources = PhysicalResourceMetricsV12(**dict(resources_raw))
    except (TypeError, ValueError) as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 benchmark resource facts drifted"
        ) from error
    derived_checks, derived_speedup = _derive_physical_checks_v13(
        config=config,
        anchor=anchor,
        product=product,
        logical=logical_evidence,
        resources=resources,
        compact_rehydration_exact=True,
    )
    if (
        dict(checks) != derived_checks
        or _decision_status(derived_checks) != "PASS_V13_PHYSICAL_BENCHMARK"
        or w3.get("speedup_over_immutable_w1") != derived_speedup
        or benchmark.get("status") != _decision_status(derived_checks)
        or benchmark.get("gate_pass") is not all(derived_checks.values())
        or benchmark.get("execution_authorization_published")
        is not all(derived_checks.values())
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 benchmark gates were not reproduced from sealed facts"
        )
    return raw


__all__ = [
    "ALLOWED_FOLDS",
    "AUTHORIZATION_SCHEMA_VERSION",
    "CONFIG_SHA256",
    "ENGINE_ID",
    "FragmentW3TrialProductV13",
    "ImmutableW1AnchorV13",
    "LogicalIOEvidenceV13",
    "PhysicalBenchmarkInvariantErrorV13",
    "REQUIRED_LOGICAL_BRIDGE_DIGESTS",
    "SCHEMA_VERSION",
    "TRIAL_SCHEMA_VERSION",
    "execute_fragment_w3_trial_v13",
    "load_execution_authorization_v13",
    "load_immutable_w1_anchor_v13",
    "load_physical_config_v13",
    "run_physical_benchmark_v13",
]
