"""Authorized national OOF runtime for the zero-copy retrieval-v13 layout.

The module owns no scientific implementation.  It reuses the frozen v12
build/replay code through the v13 buffered fragment sinks, keeps three
fold-owned actors per parallel phase, fits the three site models sequentially
after all build actors exited, and never creates merged Parquet copies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

import psutil
import pyarrow.parquet as pq

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
    REPLAY_ENTRYPOINT_V13,
    _replay_jobs_v13,
    load_and_validate_build_row_group_ledgers_v13,
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
    BUILD_ENTRYPOINT_V13,
    CONFIG_SHA256,
    NATIONAL_QUERY_COUNT,
    PhysicalBenchmarkInvariantErrorV13,
    _build_jobs_v13,
    _canonical_json,
    _inventory_index_v13,
    _read_json,
    _sha256_json,
    load_execution_authorization_v13,
    load_physical_config_v13,
    v13_implementation_lineage,
)
from .retrieval_fold_fused_sharded_oof_v12 import (
    load_shard_build_manifest_v12,
    plan_query_shards,
    spawn_manifest_workers_v12,
)
from .retrieval_fold_fused_sharded_preflight_v12 import (
    PreparedShardedExecutionV12,
)
from .retrieval_fold_fused_row_group_ledger_v13 import (
    ROW_GROUP_LEDGER_RECORD_KEYS,
    RowGroupLedgerInvariantErrorV13,
    actor_row_group_ledger_record_v13,
    load_actor_row_group_ledger_records_v13,
    validate_actual_row_group_ledger_set_v13,
)
from .retrieval_fold_fused_zero_copy_run_v13 import (
    CANONICAL_FAMILIES,
    load_zero_copy_canonical_run_v13,
    seal_zero_copy_canonical_run_v13,
)


SCHEMA_VERSION = "sireto-retrieval-v13-national-fragment-oof-1"
STOP_SCHEMA_VERSION = "sireto-retrieval-v13-national-execution-stop-1"
ATTEMPT_SCHEMA_VERSION = "sireto-retrieval-v13-national-attempt-1"
FIT_GUARD_SCHEMA_VERSION = "sireto-retrieval-v13-prefit-resource-guard-1"
ENGINE_ID = "FOLD_FUSED_FRAGMENT_NATIONAL_OOF_V13_1"
_V12_GUARD_KEYS = {
    "schema_version",
    "status",
    "selection_reason",
    "physical_authorized_maximum_concurrent_workers",
    "selected_maximum_concurrent_workers",
    "fixed_shard_count",
    "memory_metric",
    "memory_used_before_bytes",
    "memory_used_after_bytes",
    "memory_used_peak_bytes",
    "memory_threshold_bytes",
    "swapout_metric",
    "swapout_source",
    "swapout_page_size_bytes",
    "swapout_before_pages",
    "swapout_after_pages",
    "swapout_delta_pages",
    "swapout_delta_bytes",
    "observation_seconds",
    "observation_started_at_unix_seconds",
    "observation_finished_at_unix_seconds",
    "measurement_valid",
    "measured_before_first_shard_open",
    "worker_count_change_after_measurement",
    "automatic_retry_allowed",
}
_V13_GUARD_EXTRA_KEYS = {
    "base_guard_sha256",
    "storage_free_bytes",
    "storage_required_bytes",
    "storage_free_pass",
    "fallback_worker_count_allowed",
    "record_sha256",
}
_FIT_GUARD_KEYS = {
    "schema_version",
    "status",
    "phase",
    "build_actors_joined",
    "live_build_actor_count",
    "fit_parallelism",
    "outer_fit_order",
    "gc_between_outer_fits",
    "memory_metric",
    "memory_total_bytes",
    "memory_used_peak_bytes",
    "memory_used_maximum_bytes",
    "memory_available_before_bytes",
    "memory_available_after_bytes",
    "memory_available_minimum_bytes",
    "memory_available_required_bytes",
    "swapout_metric",
    "swapout_source",
    "swapout_page_size_bytes",
    "swapout_before_pages",
    "swapout_after_pages",
    "swapout_delta_pages",
    "swapout_delta_bytes",
    "observation_seconds",
    "observation_started_at_unix_seconds",
    "observation_finished_at_unix_seconds",
    "base_guard_sha256",
    "automatic_retry_allowed",
    "record_sha256",
}
_SAFE_EXCEPTION = re.compile(r"[^A-Za-z0-9_ .,:;()\[\]/=-]+")


class _NationalAttemptConsumptionErrorV13(RuntimeError):
    """An error raised only after this process won the marker O_EXCL."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _fsync_directory_v13(path: Path) -> None:
    descriptor = os.open(str(Path(path).resolve()), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _national_execution_paths_v13(
    config: Mapping[str, Any],
    *,
    requested_output: Path,
    attempt_id: str,
) -> tuple[Path, Path, Path]:
    protocol = config.get("national_execution_protocol")
    if not isinstance(protocol, Mapping):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 staging protocol is absent"
        )
    canonical = Path(str(protocol.get("canonical_output_dir") or "")).resolve()
    marker = Path(str(protocol.get("attempt_consumed_marker_path") or "")).resolve()
    prefix = str(protocol.get("stage_directory_prefix") or "")
    if (
        Path(requested_output).resolve() != canonical
        or canonical.parent != marker.parent
        or not prefix.startswith(".")
        or "/" in prefix
        or not re.fullmatch(r"[0-9a-f]{64}", attempt_id)
        or protocol.get("stage_identity")
        != "SHA256_CONFIG_AUTH_BENCHMARK_IMPLEMENTATION_SOURCE_PLAN_PREFIT_OUTPUT"
        or protocol.get("success_promotion")
        != "ATOMIC_RENAME_SAME_FILESYSTEM"
        or protocol.get("automatic_retry_allowed") is not False
        or protocol.get("stage_retained_on_failure") is not True
        or protocol.get("exception_status")
        != "STOP_NATIONAL_EXECUTION_EXCEPTION"
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 staging protocol drifted"
        )
    stage = canonical.parent / f"{prefix}{attempt_id}"
    if stage == canonical or stage.parent != canonical.parent:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 staging path escaped its canonical parent"
        )
    return canonical, stage, marker


def _national_attempt_base_v13(
    *,
    config_sha256: str,
    authorization_sha256: str,
    benchmark_sha256: str,
    implementation_sha256: str,
    source_candidate_sha256: str,
    sharded_execution_plan_sha256: str,
    prefit_parity_manifest_sha256: str,
    canonical_output_dir: Path,
) -> dict[str, Any]:
    return {
        "config_sha256": config_sha256,
        "authorization_sha256": authorization_sha256,
        "physical_benchmark_manifest_sha256": benchmark_sha256,
        "v13_implementation_sha256": implementation_sha256,
        "source_candidate_sha256": source_candidate_sha256,
        "sharded_execution_plan_sha256": sharded_execution_plan_sha256,
        "prefit_parity_manifest_sha256": prefit_parity_manifest_sha256,
        "canonical_output_dir": str(Path(canonical_output_dir).resolve()),
        "query_count": NATIONAL_QUERY_COUNT,
        "worker_count": 3,
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
    }


def _national_attempt_payload_v13(
    *,
    base: Mapping[str, Any],
    attempt_id: str,
    stage_output_dir: Path,
) -> dict[str, Any]:
    if attempt_id != _sha256_json(base):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 attempt id drifted"
        )
    return {
        **dict(base),
        "attempt_id": attempt_id,
        "stage_output_dir": str(Path(stage_output_dir).resolve()),
        "stage_is_content_addressed": True,
        "success_promotion": "ATOMIC_RENAME_SAME_FILESYSTEM",
        "stage_retained_on_failure": True,
    }


def _consume_national_attempt_v13(
    *,
    marker_path: Path,
    payload: Mapping[str, Any],
) -> str:
    marker = Path(marker_path).resolve()
    marker.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "status": "CONSUMED_BEFORE_NATIONAL_EXECUTION",
        "automatic_retry_allowed": False,
        **dict(payload),
    }
    encoded = (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            str(marker),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 one-shot attempt was already consumed"
        ) from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory_v13(marker.parent)
        digest = sha256_file(marker)
        if marker.stat().st_nlink != 1:
            raise PhysicalBenchmarkInvariantErrorV13(
                "national v13 attempt marker is hardlinked"
            )
        return digest
    except BaseException as error:
        # O_EXCL has consumed the one-shot attempt.  Preserve that fact for the
        # outer transaction so it can publish a durable STOP instead of
        # leaving an unexplained marker after a write/fsync/hash/stat fault.
        raise _NationalAttemptConsumptionErrorV13(
            "national v13 attempt marker failed after O_EXCL"
        ) from error


def _observed_marker_sha256_v13(marker_path: Path) -> str:
    """Hash the bytes left by a failed post-O_EXCL marker operation."""

    marker = Path(marker_path).resolve()
    try:
        if marker.is_file() and not marker.is_symlink():
            return sha256_file(marker)
    except BaseException:
        pass
    # The STOP remains structurally sealed even when the failed storage
    # operation made the marker unreadable.  This sentinel is explicitly not
    # presented as a digest of readable marker bytes.
    return "UNREADABLE_POST_O_EXCL"


def _load_national_attempt_v13(
    *,
    marker_path: Path,
    expected_sha256: str,
    expected_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    marker = Path(marker_path).resolve()
    if (
        not marker.is_file()
        or marker.is_symlink()
        or marker.stat().st_nlink != 1
        or sha256_file(marker) != expected_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 attempt marker bytes drifted"
        )
    raw = _read_json(marker, label="national v13 attempt marker")
    expected = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "status": "CONSUMED_BEFORE_NATIONAL_EXECUTION",
        "automatic_retry_allowed": False,
        **dict(expected_payload),
    }
    if raw != expected:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 attempt marker payload drifted"
        )
    return raw


def _sanitized_exception_v13(error: BaseException) -> str:
    value = str(error).replace("\n", " ").replace("\r", " ")[:500]
    return _SAFE_EXCEPTION.sub("?", value)


def _publish_national_exception_stop_v13(
    *,
    stage: Path,
    canonical: Path,
    marker_path: Path,
    marker_sha256: str,
    attempt_id: str,
    frozen_implementation_sha256: str,
    frozen_implementation_files: tuple[Mapping[str, Any], ...],
    config_sha256: str,
    authorization_path: Path,
    authorization_sha256: str,
    benchmark_path: Path,
    benchmark_sha256: str,
    phase: str,
    error: BaseException,
) -> Path:
    stage.mkdir(parents=True, exist_ok=True)
    manifest = stage / "manifest.json"
    document = {
        "schema_version": STOP_SCHEMA_VERSION,
        "status": "STOP_NATIONAL_EXECUTION_EXCEPTION",
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "automatic_retry_allowed": False,
        "canonical_output_dir": str(canonical),
        "stage_output_dir": str(stage),
        "attempt_id": attempt_id,
        "attempt_marker": {
            "path": str(marker_path),
            "sha256": marker_sha256,
        },
        "config_sha256": config_sha256,
        "frozen_implementation_sha256": frozen_implementation_sha256,
        "frozen_implementation_files": [
            dict(value) for value in frozen_implementation_files
        ],
        "authorization": {
            "path": str(Path(authorization_path).resolve()),
            "sha256": authorization_sha256,
        },
        "physical_benchmark": {
            "path": str(Path(benchmark_path).resolve()),
            "sha256": benchmark_sha256,
        },
        "failed_phase": str(phase),
        "exception_type": type(error).__name__,
        "exception_message_sanitized": _sanitized_exception_v13(error),
    }
    document["stop_id"] = _sha256_json(document)
    _atomic_json(manifest, document)
    descriptor = os.open(manifest, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory_v13(stage)
    return manifest


def _artifact_record(path: Path, root: Path) -> Mapping[str, Any]:
    resolved = Path(path).resolve()
    canonical_root = Path(root).resolve()
    try:
        location = resolved.relative_to(canonical_root).as_posix()
    except ValueError as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 internal artifact escaped the output tree"
        ) from error
    if (
        not location
        or ".." in Path(location).parts
        or not resolved.is_file()
        or resolved.is_symlink()
        or resolved.stat().st_nlink != 1
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 internal artifact is unsafe"
        )
    return {
        "path": location,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_artifact(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    inventory: Mapping[str, Mapping[str, Any]],
) -> Path:
    if set(record) != {"path", "sha256", "size_bytes"}:
        raise PhysicalBenchmarkInvariantErrorV13(
            f"national v13 {label} record drifted"
        )
    value = Path(str(record.get("path") or ""))
    canonical_root = Path(root).resolve()
    if value.is_absolute() or not str(value) or ".." in value.parts:
        raise PhysicalBenchmarkInvariantErrorV13(
            f"national v13 {label} is not output-relative"
        )
    path = (canonical_root / value).resolve()
    try:
        relative = path.relative_to(canonical_root).as_posix()
    except ValueError as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            f"national v13 {label} escaped the output tree"
        ) from error
    inventory_record = inventory.get(relative)
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != int(record.get("size_bytes", -1))
        or sha256_file(path) != record.get("sha256")
        or inventory_record is None
        or inventory_record.get("sha256") != record.get("sha256")
        or inventory_record.get("size_bytes") != record.get("size_bytes")
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            f"national v13 {label} bytes drifted"
        )
    return path


def _national_inventory_v13(output: Path, manifest_path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    inodes: set[tuple[int, int]] = set()
    for child in sorted(output.rglob("*")):
        if child.is_symlink():
            raise PhysicalBenchmarkInvariantErrorV13(
                "national v13 tree contains a symlink"
            )
        if not child.is_file() or child == manifest_path:
            continue
        relative = child.relative_to(output).as_posix()
        if any(
            part.startswith(("merged_", "canonical_merged_"))
            for part in Path(relative).parts
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "national v13 tree contains a merged artifact"
            )
        stat = child.stat()
        inode = (int(stat.st_dev), int(stat.st_ino))
        if inode in inodes or int(stat.st_nlink) != 1:
            raise PhysicalBenchmarkInvariantErrorV13(
                "national v13 tree contains a hardlink"
            )
        inodes.add(inode)
        is_row_group_ledger = "source_row_group_ledger" in Path(relative).parts
        family = (
            "source_read_ledgers"
            if is_row_group_ledger
            else next(
                (value for value in CANONICAL_FAMILIES if value in child.name),
                "support",
            )
        )
        role = (
            "actual_row_group_ledger_manifest"
            if is_row_group_ledger and child.name == "manifest.json"
            else (
                "actual_row_group_ledger_rows"
                if is_row_group_ledger
                else "national_artifact"
            )
        )
        records.append(
            {
                "role": role,
                "family": family,
                "path": relative,
                "sha256": sha256_file(child),
                "size_bytes": stat.st_size,
            }
        )
    if not records:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 inventory is empty"
        )
    return records


def _validate_relocatable_manifest_tree_v13(root: Path) -> None:
    """Reject absolute/escaping paths in every internal retrieval manifest."""

    canonical = Path(root).resolve()

    def visit(value: Any, *, manifest_root: Path) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"path", "manifest_path"} and isinstance(item, str):
                    candidate = Path(item)
                    if candidate.is_absolute() or not item:
                        raise PhysicalBenchmarkInvariantErrorV13(
                            "national v13 internal manifest retained an absolute path"
                        )
                    resolved = (manifest_root / candidate).resolve()
                    try:
                        resolved.relative_to(canonical)
                    except ValueError as error:
                        raise PhysicalBenchmarkInvariantErrorV13(
                            "national v13 internal manifest path escaped after promotion"
                        ) from error
                visit(item, manifest_root=manifest_root)
        elif isinstance(value, list):
            for item in value:
                visit(item, manifest_root=manifest_root)

    root_manifest = canonical / "manifest.json"
    for path in sorted(canonical.rglob("*.json")):
        if path == root_manifest or path.is_symlink() or not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # XGBoost model JSON is not an artifact-manifest authority here.
            continue
        if not isinstance(raw, Mapping) or not str(
            raw.get("schema_version") or ""
        ).startswith("sireto-retrieval-"):
            continue
        visit(raw, manifest_root=path.parent)


def _validate_runtime_resource_guard_v13(guard: Mapping[str, Any]) -> str:
    """Recompute W3-only RAM/swap/storage guard arithmetic exactly."""

    if set(guard) != _V12_GUARD_KEYS | _V13_GUARD_EXTRA_KEYS:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 runtime guard keyset drifted"
        )
    base = {key: guard[key] for key in _V12_GUARD_KEYS}
    before = guard.get("memory_used_before_bytes")
    after = guard.get("memory_used_after_bytes")
    peak = guard.get("memory_used_peak_bytes")
    before_pages = guard.get("swapout_before_pages")
    after_pages = guard.get("swapout_after_pages")
    page_size = guard.get("swapout_page_size_bytes")
    delta_pages = guard.get("swapout_delta_pages")
    free = guard.get("storage_free_bytes")
    required = guard.get("storage_required_bytes")
    numeric_ints = (
        before,
        after,
        peak,
        before_pages,
        after_pages,
        page_size,
        delta_pages,
        guard.get("swapout_delta_bytes"),
        free,
        required,
    )
    if (
        any(type(value) is not int or int(value) < 0 for value in numeric_ints)
        or guard.get("schema_version")
        != "sireto-retrieval-v12-national-worker-guard-1"
        or guard.get("status") != "PASS_W3"
        or guard.get("selection_reason") != "W3_CURRENT_STATE_PASS"
        or guard.get("physical_authorized_maximum_concurrent_workers") != 3
        or guard.get("selected_maximum_concurrent_workers") != 3
        or guard.get("fixed_shard_count") != 3
        or guard.get("memory_metric") != "psutil.virtual_memory().used"
        or peak != max(before, after)
        or guard.get("memory_threshold_bytes") != 14 * 1024**3
        or peak > guard["memory_threshold_bytes"]
        or guard.get("swapout_metric") != "vm_stat.Swapouts"
        or not isinstance(guard.get("swapout_source"), str)
        or not guard.get("swapout_source")
        or page_size <= 0
        or delta_pages != after_pages - before_pages
        or delta_pages != 0
        or guard.get("swapout_delta_bytes") != delta_pages * page_size
        or guard.get("observation_seconds") != 5.0
        or not isinstance(guard.get("observation_started_at_unix_seconds"), (int, float))
        or not isinstance(guard.get("observation_finished_at_unix_seconds"), (int, float))
        or guard["observation_finished_at_unix_seconds"]
        < guard["observation_started_at_unix_seconds"]
        or guard.get("measurement_valid") is not True
        or guard.get("measured_before_first_shard_open") is not True
        or guard.get("worker_count_change_after_measurement") is not False
        or guard.get("automatic_retry_allowed") is not False
        or guard.get("base_guard_sha256") != _sha256_json(base)
        or free < required
        or guard.get("storage_free_pass") is not (free >= required)
        or guard.get("storage_free_pass") is not True
        or guard.get("fallback_worker_count_allowed") is not False
        or guard.get("record_sha256")
        != _sha256_json(
            {key: value for key, value in guard.items() if key != "record_sha256"}
        )
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 runtime guard arithmetic drifted"
        )
    return str(guard["record_sha256"])


def _fit_resource_guard_v13(
    *,
    sharded_execution_plan_path: Path,
    memory_used_maximum_bytes: int,
    memory_reader: Any = psutil.virtual_memory,
    guard_reader: Any = current_national_worker_guard_v12,
) -> Mapping[str, Any]:
    """Measure the post-build/pre-fit state after all actors have joined."""

    before = memory_reader()
    base_with_sha = dict(guard_reader(3, sharded_execution_plan_path))
    after = memory_reader()
    base_sha = str(base_with_sha.pop("record_sha256", ""))
    total_before = int(before.total)
    total_after = int(after.total)
    available_before = int(before.available)
    available_after = int(after.available)
    used_peak = max(
        int(base_with_sha.get("memory_used_peak_bytes", -1)),
        int(before.used),
        int(after.used),
    )
    available_required = total_before - int(memory_used_maximum_bytes)
    record: dict[str, Any] = {
        "schema_version": FIT_GUARD_SCHEMA_VERSION,
        "status": "PASS_PREFIT_RESOURCES",
        "phase": "AFTER_BUILD_ACTORS_JOIN_BEFORE_FIRST_SITE_FIT",
        "build_actors_joined": True,
        "live_build_actor_count": 0,
        "fit_parallelism": 1,
        "outer_fit_order": list(TRAIN_FOLDS),
        "gc_between_outer_fits": True,
        "memory_metric": "psutil.virtual_memory().used_and_available",
        "memory_total_bytes": total_before,
        "memory_used_peak_bytes": used_peak,
        "memory_used_maximum_bytes": int(memory_used_maximum_bytes),
        "memory_available_before_bytes": available_before,
        "memory_available_after_bytes": available_after,
        "memory_available_minimum_bytes": min(
            available_before, available_after
        ),
        "memory_available_required_bytes": available_required,
        "swapout_metric": base_with_sha.get("swapout_metric"),
        "swapout_source": base_with_sha.get("swapout_source"),
        "swapout_page_size_bytes": base_with_sha.get(
            "swapout_page_size_bytes"
        ),
        "swapout_before_pages": base_with_sha.get("swapout_before_pages"),
        "swapout_after_pages": base_with_sha.get("swapout_after_pages"),
        "swapout_delta_pages": base_with_sha.get("swapout_delta_pages"),
        "swapout_delta_bytes": base_with_sha.get("swapout_delta_bytes"),
        "observation_seconds": base_with_sha.get("observation_seconds"),
        "observation_started_at_unix_seconds": base_with_sha.get(
            "observation_started_at_unix_seconds"
        ),
        "observation_finished_at_unix_seconds": base_with_sha.get(
            "observation_finished_at_unix_seconds"
        ),
        "base_guard_sha256": base_sha,
        "automatic_retry_allowed": False,
    }
    if (
        set(base_with_sha) != _V12_GUARD_KEYS
        or base_sha != _sha256_json(base_with_sha)
        or total_before != total_after
        or total_before <= memory_used_maximum_bytes
        or available_required <= 0
        or base_with_sha.get("status") != "PASS_W3"
        or base_with_sha.get("selected_maximum_concurrent_workers") != 3
        or base_with_sha.get("swapout_delta_pages") != 0
        or base_with_sha.get("swapout_delta_bytes") != 0
        or used_peak > memory_used_maximum_bytes
        or record["memory_available_minimum_bytes"] < available_required
    ):
        record["status"] = "STOP_PREFIT_RESOURCES"
    record["record_sha256"] = _sha256_json(record)
    _validate_fit_resource_guard_v13(record, require_pass=False)
    return record


def _validate_fit_resource_guard_v13(
    guard: Mapping[str, Any],
    *,
    require_pass: bool = True,
) -> str:
    if set(guard) != _FIT_GUARD_KEYS:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 fit guard keyset drifted"
        )
    numeric = (
        guard.get("memory_total_bytes"),
        guard.get("memory_used_peak_bytes"),
        guard.get("memory_used_maximum_bytes"),
        guard.get("memory_available_before_bytes"),
        guard.get("memory_available_after_bytes"),
        guard.get("memory_available_minimum_bytes"),
        guard.get("memory_available_required_bytes"),
        guard.get("swapout_page_size_bytes"),
        guard.get("swapout_before_pages"),
        guard.get("swapout_after_pages"),
        guard.get("swapout_delta_pages"),
        guard.get("swapout_delta_bytes"),
    )
    numeric_valid = not any(
        type(value) is not int or value < 0 for value in numeric
    )
    started = guard.get("observation_started_at_unix_seconds")
    finished = guard.get("observation_finished_at_unix_seconds")
    timestamps_valid = (
        isinstance(started, (int, float))
        and isinstance(finished, (int, float))
        and finished >= started
    )
    expected_status = bool(
        numeric_valid
        and guard.get("memory_used_peak_bytes")
        <= guard.get("memory_used_maximum_bytes")
        and guard.get("memory_available_minimum_bytes")
        >= guard.get("memory_available_required_bytes")
        and guard.get("swapout_delta_pages") == 0
        and guard.get("swapout_delta_bytes") == 0
    )
    if (
        not numeric_valid
        or guard.get("schema_version") != FIT_GUARD_SCHEMA_VERSION
        or guard.get("phase")
        != "AFTER_BUILD_ACTORS_JOIN_BEFORE_FIRST_SITE_FIT"
        or guard.get("build_actors_joined") is not True
        or guard.get("live_build_actor_count") != 0
        or guard.get("fit_parallelism") != 1
        or guard.get("outer_fit_order") != list(TRAIN_FOLDS)
        or guard.get("gc_between_outer_fits") is not True
        or guard.get("memory_metric")
        != "psutil.virtual_memory().used_and_available"
        or guard.get("memory_used_maximum_bytes") != 14 * 1024**3
        or guard.get("memory_total_bytes")
        <= guard.get("memory_used_maximum_bytes")
        or guard.get("memory_available_required_bytes")
        != guard.get("memory_total_bytes")
        - guard.get("memory_used_maximum_bytes")
        or guard.get("memory_available_minimum_bytes")
        != min(
            guard.get("memory_available_before_bytes"),
            guard.get("memory_available_after_bytes"),
        )
        or guard.get("swapout_metric") != "vm_stat.Swapouts"
        or not isinstance(guard.get("swapout_source"), str)
        or not guard.get("swapout_source")
        or guard.get("swapout_page_size_bytes") <= 0
        or guard.get("swapout_delta_pages")
        != guard.get("swapout_after_pages")
        - guard.get("swapout_before_pages")
        or guard.get("swapout_delta_bytes")
        != guard.get("swapout_delta_pages")
        * guard.get("swapout_page_size_bytes")
        or guard.get("observation_seconds") != 5.0
        or not timestamps_valid
        or guard.get("automatic_retry_allowed") is not False
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(guard.get("base_guard_sha256") or "")
        )
        or guard.get("status")
        != (
            "PASS_PREFIT_RESOURCES"
            if expected_status
            else "STOP_PREFIT_RESOURCES"
        )
        or (require_pass and not expected_status)
        or guard.get("record_sha256")
        != _sha256_json(
            {
                key: value
                for key, value in guard.items()
                if key != "record_sha256"
            }
        )
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 fit guard arithmetic drifted"
        )
    return str(guard["record_sha256"])


@dataclass(frozen=True)
class NationalFragmentOOFV13:
    manifest_path: Path
    manifest_sha256: str
    run_id: str
    query_count: int
    lineage: Mapping[str, str]
    gates: Mapping[str, Any]


def _national_identity_v13(
    *,
    authorization_sha256: str,
    physical_benchmark_manifest_sha256: str,
    implementation_sha256: str,
    lineage: Mapping[str, str],
    source_candidate_sha256: str,
    sharded_execution_plan_sha256: str,
    prefit_parity_manifest_sha256: str,
    runtime_resource_guard_sha256: str,
    fit_resource_guard_sha256: str,
    attempt_marker_sha256: str,
    attempt_id: str,
    canonical_output_dir: str,
    artifact_inventory_sha256: str,
    source_spec_sha256: list[str],
    source_row_group_ledger_sha256: list[str],
    source_row_group_union_sha256: str,
    source_row_group_logical_compressed_bytes: int,
    build_fragment_dataset_sha256: str,
    compact_replay_dataset_sha256: str,
    zero_copy_canonical_run_sha256: str,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the single canonical national identity used writer and loader."""

    return {
        "engine_id": ENGINE_ID,
        "config_sha256": CONFIG_SHA256,
        "authorization_sha256": authorization_sha256,
        "physical_benchmark_manifest_sha256": (
            physical_benchmark_manifest_sha256
        ),
        "v13_implementation_sha256": implementation_sha256,
        "sharded_implementation_sha256": lineage[
            "sharded_implementation_sha256"
        ],
        "runtime_environment_sha256": lineage[
            "runtime_environment_sha256"
        ],
        "source_candidate_sha256": source_candidate_sha256,
        "sharded_execution_plan_sha256": sharded_execution_plan_sha256,
        "prefit_parity_manifest_sha256": prefit_parity_manifest_sha256,
        "runtime_resource_guard_sha256": runtime_resource_guard_sha256,
        "fit_resource_guard_sha256": fit_resource_guard_sha256,
        "attempt_marker_sha256": attempt_marker_sha256,
        "attempt_id": attempt_id,
        "canonical_output_dir": canonical_output_dir,
        "artifact_inventory_sha256": artifact_inventory_sha256,
        "query_count": NATIONAL_QUERY_COUNT,
        "worker_count": 3,
        "source_spec_sha256": list(source_spec_sha256),
        "source_row_group_ledger_sha256": list(
            source_row_group_ledger_sha256
        ),
        "source_row_group_union_sha256": source_row_group_union_sha256,
        "source_row_group_logical_compressed_bytes": (
            source_row_group_logical_compressed_bytes
        ),
        "build_fragment_dataset_sha256": build_fragment_dataset_sha256,
        "compact_replay_dataset_sha256": compact_replay_dataset_sha256,
        "zero_copy_canonical_run_sha256": zero_copy_canonical_run_sha256,
        "gates_sha256": _sha256_json(gates),
    }


def _national_projection_totals_v13(
    source_path: Path, projection_columns: tuple[str, ...]
) -> tuple[int, int]:
    """Footer-only expected extent for the exhaustive national shards."""

    parquet = pq.ParquetFile(Path(source_path).resolve())
    indexes = {
        name: ordinal for ordinal, name in enumerate(parquet.schema_arrow.names)
    }
    if not projection_columns or any(
        name not in indexes for name in projection_columns
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 production projection drifted"
        )
    logical_bytes = sum(
        int(
            parquet.metadata.row_group(row_group)
            .column(indexes[name])
            .total_compressed_size
        )
        for row_group in range(parquet.num_row_groups)
        for name in projection_columns
    )
    if parquet.num_row_groups <= 0 or logical_bytes <= 0:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 source footer is empty"
        )
    return parquet.num_row_groups, logical_bytes


def _frozen_gate_thresholds_v13(config: Mapping[str, Any]) -> GateThresholds:
    scientific = config.get("scientific_invariants")
    if not isinstance(scientific, Mapping):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 scientific gate authorities are absent"
        )
    gates = GateThresholds(
        master_recall_at_2000=float(scientific.get("master_recall_at_2000", -1)),
        prerank_recall_at_500=float(scientific.get("prerank_recall_at_500", -1)),
        top100_recall=float(scientific.get("exact_recall_at_100", -1)),
        top100_wilson95_lower=float(
            scientific.get("exact_recall_at_100_wilson95_lower", -1)
        ),
        minimum_each_outer_fold_top100=float(
            scientific.get("minimum_each_outer_fold_exact_recall_at_100", -1)
        ),
    )
    if gates != GateThresholds():
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 scientific thresholds differ from frozen v12"
        )
    return gates


def load_national_fragment_oof_v13(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    workspace_root: Path,
) -> NationalFragmentOOFV13:
    """Strictly reopen the national zero-copy run and all canonical manifests."""

    path = Path(manifest_path).resolve()
    digest = str(expected_manifest_sha256)
    if not path.is_file() or sha256_file(path) != digest:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 manifest bytes drifted"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "status",
        "run_id",
        "identity",
        "run_kind",
        "config_sha256",
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
    identity = raw.get("identity")
    lineage = {str(key): str(value) for key, value in dict(raw.get("lineage") or {}).items()}
    if (
        set(raw) != expected_keys
        or raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("status") != "EXECUTED_NO_READY_CLAIM"
        or raw.get("run_kind") != "NATIONAL_OOF_FOLDS_2_3_4_ZERO_COPY_V13"
        or raw.get("config_sha256") != CONFIG_SHA256
        or not isinstance(identity, Mapping)
        or raw.get("run_id") != _sha256_json(identity)
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
        or not lineage
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 manifest contract drifted"
        )
    implementation_sha, _files = v13_implementation_lineage(workspace_root)
    if lineage.get("v13_implementation_sha256") != implementation_sha:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 implementation changed after execution"
        )
    root = path.parent
    inventory, materialized = _inventory_index_v13(path, raw)
    if materialized != raw.get("materialized_output_bytes"):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 materialized bytes drifted"
        )
    _validate_relocatable_manifest_tree_v13(root)
    authorization_record = raw.get("authorization")
    source_record = raw.get("source_candidate")
    plan_record = raw.get("sharded_execution_plan")
    prefit_record = raw.get("prefit_parity")
    guard = raw.get("runtime_resource_guard")
    fit_guard = raw.get("fit_resource_guard")
    attempt_record = raw.get("national_attempt")
    if (
        not isinstance(authorization_record, Mapping)
        or set(authorization_record)
        != {
            "path",
            "sha256",
            "benchmark_manifest_path",
            "benchmark_manifest_sha256",
        }
        or not isinstance(source_record, Mapping)
        or set(source_record) != {"path", "sha256"}
        or not isinstance(plan_record, Mapping)
        or set(plan_record) != {"path", "sha256"}
        or not isinstance(prefit_record, Mapping)
        or set(prefit_record) != {"path", "sha256"}
        or not isinstance(guard, Mapping)
        or not isinstance(fit_guard, Mapping)
        or not isinstance(attempt_record, Mapping)
        or set(attempt_record)
        != {"path", "sha256", "attempt_id", "stage_directory_name"}
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 authority/resource records drifted"
        )
    _validate_runtime_resource_guard_v13(guard)
    _validate_fit_resource_guard_v13(fit_guard)
    authorization = load_execution_authorization_v13(
        Path(str(authorization_record["path"])).resolve(),
        expected_authorization_sha256=str(authorization_record["sha256"]),
        benchmark_manifest_path=Path(
            str(authorization_record["benchmark_manifest_path"])
        ).resolve(),
        expected_benchmark_manifest_sha256=str(
            authorization_record["benchmark_manifest_sha256"]
        ),
        workspace_root=workspace_root,
    )
    authorization_identity = authorization["authorization_identity"]
    benchmark_path = Path(
        str(authorization_record["benchmark_manifest_path"])
    ).resolve()
    benchmark = _read_json(
        benchmark_path, label="national v13 authorized physical benchmark"
    )
    authorized_w3_lineage = dict((benchmark.get("w3") or {}).get("lineage") or {})
    physical_config = load_physical_config_v13(
        Path(workspace_root).resolve()
        / "config/retrieval_all_human_v13_physical_execution.json"
    )
    expected_plan = (
        Path(workspace_root).resolve()
        / str(physical_config["authorities"]["v12_sharded_plan"]["path"])
    ).resolve()
    plan_raw = _read_json(expected_plan, label="national v13 sealed plan")
    expected_prefit = Path(
        str(
            ((plan_raw.get("input_lineage") or {}).get("prefit_parity") or {}).get(
                "manifest_path"
            )
            or ""
        )
    ).resolve()
    expected_source = Path(
        str(physical_config["input_lineage"]["source_candidate_path"])
    ).resolve()
    attempt_base = _national_attempt_base_v13(
        config_sha256=CONFIG_SHA256,
        authorization_sha256=str(authorization_record["sha256"]),
        benchmark_sha256=str(
            authorization_record["benchmark_manifest_sha256"]
        ),
        implementation_sha256=implementation_sha,
        source_candidate_sha256=str(source_record["sha256"]),
        sharded_execution_plan_sha256=str(plan_record["sha256"]),
        prefit_parity_manifest_sha256=str(prefit_record["sha256"]),
        canonical_output_dir=root,
    )
    expected_attempt_id = _sha256_json(attempt_base)
    expected_canonical, expected_stage, expected_marker = (
        _national_execution_paths_v13(
            physical_config,
            requested_output=root,
            attempt_id=expected_attempt_id,
        )
    )
    expected_attempt_payload = _national_attempt_payload_v13(
        base=attempt_base,
        attempt_id=expected_attempt_id,
        stage_output_dir=expected_stage,
    )
    if (
        root != expected_canonical
        or Path(str(attempt_record["path"])).resolve() != expected_marker
        or attempt_record["attempt_id"] != expected_attempt_id
        or attempt_record["stage_directory_name"] != expected_stage.name
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 canonical attempt identity drifted"
        )
    _load_national_attempt_v13(
        marker_path=expected_marker,
        expected_sha256=str(attempt_record["sha256"]),
        expected_payload=expected_attempt_payload,
    )
    if (
        Path(str(plan_record["path"])).resolve() != expected_plan
        or sha256_file(expected_plan) != plan_record["sha256"]
        or Path(str(prefit_record["path"])).resolve() != expected_prefit
        or sha256_file(expected_prefit) != prefit_record["sha256"]
        or Path(str(source_record["path"])).resolve() != expected_source
        or source_record["sha256"]
        != physical_config["input_lineage"]["source_candidate_sha256"]
        or authorization_identity.get("sharded_execution_plan_sha256")
        != plan_record["sha256"]
        or authorization_identity.get("prefit_parity_manifest_sha256")
        != prefit_record["sha256"]
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 source/plan/PREFIT authority drifted"
        )
    scientific_lineage = {
        key: value
        for key, value in lineage.items()
        if key
        not in {
            "sharded_implementation_sha256",
            "runtime_environment_sha256",
            "v13_implementation_sha256",
            "physical_benchmark_manifest_sha256",
            "execution_authorization_sha256",
        }
    }
    expected_lineage = sharded_execution_lineage_v12(
        scientific_lineage, workspace_root=Path(workspace_root).resolve()
    )
    expected_lineage["v13_implementation_sha256"] = implementation_sha
    expected_lineage["physical_benchmark_manifest_sha256"] = str(
        authorization_record["benchmark_manifest_sha256"]
    )
    expected_lineage["execution_authorization_sha256"] = str(
        authorization_record["sha256"]
    )
    national_base_lineage = {
        key: value
        for key, value in lineage.items()
        if key
        not in {
            "physical_benchmark_manifest_sha256",
            "execution_authorization_sha256",
        }
    }
    if (
        lineage != expected_lineage
        or not authorized_w3_lineage
        or national_base_lineage != authorized_w3_lineage
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 base/v13 execution lineage drifted"
        )
    build_path = _resolve_artifact(
        root,
        raw["build_fragment_dataset"],
        label="build dataset",
        inventory=inventory,
    )
    replay_path = _resolve_artifact(
        root,
        raw["compact_replay_dataset"],
        label="replay dataset",
        inventory=inventory,
    )
    canonical_path = _resolve_artifact(
        root,
        raw["zero_copy_canonical_run"],
        label="canonical run",
        inventory=inventory,
    )
    build = load_build_fragment_dataset_v13(
        build_path, verify_hashes=True, verify_logical=True
    )
    replay = load_compact_replay_dataset_v13(
        replay_path, verify_hashes=True, verify_logical=True
    )
    canonical = load_zero_copy_canonical_run_v13(
        canonical_path, verify_hashes=True, verify_logical=True
    )
    try:
        row_group_artifacts = load_actor_row_group_ledger_records_v13(
            raw.get("source_row_group_ledgers"), root=root
        )
        for record in raw["source_row_group_ledgers"]:
            if set(record) != set(ROW_GROUP_LEDGER_RECORD_KEYS):
                raise RowGroupLedgerInvariantErrorV13(
                    "national actual-read record fields drifted"
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
                or parquet_inventory.get("sha256")
                != record["parquet_sha256"]
                or parquet_inventory.get("size_bytes")
                != record["parquet_size_bytes"]
            ):
                raise RowGroupLedgerInvariantErrorV13(
                    "national actual-read ledger is not inventory-bound"
                )
        source_extent = physical_config["cache_neutral_io_gate"][
            "source_extent"
        ]
        projection = tuple(
            source_extent["production_projection"]["columns"]
        )
        expected_groups, expected_bytes = _national_projection_totals_v13(
            expected_source, projection
        )
        if (
            expected_groups
            != int(source_extent["national_expected_unique_row_groups"])
            or expected_bytes
            != int(
                source_extent[
                    "national_expected_logical_compressed_bytes"
                ]
            )
        ):
            raise RowGroupLedgerInvariantErrorV13(
                "national source footer differs from its extent pins"
            )
        actual_reads = validate_actual_row_group_ledger_set_v13(
            [
                (value.manifest_path, value.manifest_sha256)
                for value in row_group_artifacts
            ],
            expected_specs=build.specs,
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
            verify_source_hash=False,
        )
    except (KeyError, TypeError, RowGroupLedgerInvariantErrorV13) as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 actual source row-group ledger drifted"
        ) from error
    if (
        build.query_count != NATIONAL_QUERY_COUNT
        or replay.query_count != NATIONAL_QUERY_COUNT
        or canonical.query_count != NATIONAL_QUERY_COUNT
        or dict(build.lineage) != lineage
        or dict(replay.lineage) != lineage
        or dict(canonical.lineage) != lineage
        or tuple(raw.get("source_specs") or ())
        != tuple(value.as_manifest() for value in build.specs)
        or tuple(raw.get("canonical_datasets") or ()) != CANONICAL_FAMILIES
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 system artifact lineage drifted"
        )
    gates = raw.get("gates")
    frozen_gates = _frozen_gate_thresholds_v13(physical_config)
    recomputed_gates = dict(
        aggregate_compact_replay_gates_v13(replay, frozen_gates)
    )
    if (
        not isinstance(gates, Mapping)
        or dict(gates) != recomputed_gates
        or gates.get("denominator")
        != "ALL_USABLE_HUMAN_SIRET_LABELS_FOLDS_2_3_4"
        or gates.get("gate_eligible") is not False
        or gates.get("decision") != "NO_READY_DECISION"
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 prefit gates drifted"
        )
    expected_identity = _national_identity_v13(
        authorization_sha256=str(authorization_record["sha256"]),
        physical_benchmark_manifest_sha256=str(
            authorization_record["benchmark_manifest_sha256"]
        ),
        implementation_sha256=implementation_sha,
        lineage=lineage,
        source_candidate_sha256=str(source_record["sha256"]),
        sharded_execution_plan_sha256=str(plan_record["sha256"]),
        prefit_parity_manifest_sha256=str(prefit_record["sha256"]),
        runtime_resource_guard_sha256=str(guard["record_sha256"]),
        fit_resource_guard_sha256=str(fit_guard["record_sha256"]),
        attempt_marker_sha256=str(attempt_record["sha256"]),
        attempt_id=expected_attempt_id,
        canonical_output_dir=str(root),
        artifact_inventory_sha256=_sha256_json(raw["artifact_inventory"]),
        source_spec_sha256=[value.digest for value in build.specs],
        source_row_group_ledger_sha256=[
            value.manifest_sha256 for value in actual_reads.artifacts
        ],
        source_row_group_union_sha256=actual_reads.union_sha256,
        source_row_group_logical_compressed_bytes=(
            actual_reads.logical_compressed_bytes
        ),
        build_fragment_dataset_sha256=build.manifest_sha256,
        compact_replay_dataset_sha256=replay.manifest_sha256,
        zero_copy_canonical_run_sha256=canonical.manifest_sha256,
        gates=gates,
    )
    if dict(identity) != expected_identity:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 content identity drifted"
        )
    return NationalFragmentOOFV13(
        manifest_path=path,
        manifest_sha256=digest,
        run_id=str(raw["run_id"]),
        query_count=NATIONAL_QUERY_COUNT,
        lineage=lineage,
        gates=dict(gates),
    )


def _execute_national_fragment_oof_stage_v13(
    *,
    prepared: PreparedShardedExecutionV12,
    workspace_root: Path,
    output_dir: Path,
    physical_benchmark_manifest: Path,
    physical_benchmark_manifest_sha256: str,
    execution_authorization: Path,
    execution_authorization_sha256: str,
    canonical_output_dir: Path,
    attempt_marker_path: Path,
    attempt_marker_sha256: str,
    attempt_id: str,
    phase_state: dict[str, str],
) -> NationalFragmentOOFV13:
    """Build one already-consumed national attempt inside its stage root."""

    root = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists() or len(prepared.national_query_ids) != NATIONAL_QUERY_COUNT:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 output/population drifted"
        )
    authorization = load_execution_authorization_v13(
        execution_authorization,
        expected_authorization_sha256=execution_authorization_sha256,
        benchmark_manifest_path=physical_benchmark_manifest,
        expected_benchmark_manifest_sha256=physical_benchmark_manifest_sha256,
        workspace_root=root,
    )
    authorization_identity = authorization["authorization_identity"]
    config = load_physical_config_v13(
        root / "config/retrieval_all_human_v13_physical_execution.json"
    )
    frozen_gates = _frozen_gate_thresholds_v13(config)
    expected_canonical, expected_stage, expected_marker = (
        _national_execution_paths_v13(
            config,
            requested_output=canonical_output_dir,
            attempt_id=attempt_id,
        )
    )
    if (
        output != expected_stage
        or Path(canonical_output_dir).resolve() != expected_canonical
        or Path(attempt_marker_path).resolve() != expected_marker
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 stage capability drifted"
        )
    if prepared.gates != frozen_gates:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 prepared gates differ from frozen scientific thresholds"
        )
    plan_authority = config["authorities"]["v12_sharded_plan"]
    expected_plan = (root / str(plan_authority["path"])).resolve()
    plan_raw = _read_json(expected_plan, label="national v13 sharded plan")
    prefit_record = (plan_raw.get("input_lineage") or {}).get("prefit_parity") or {}
    expected_prefit = Path(str(prefit_record.get("manifest_path") or "")).resolve()
    expected_source = Path(
        str(config["input_lineage"]["source_candidate_path"])
    ).resolve()
    benchmark_raw = _read_json(
        Path(physical_benchmark_manifest).resolve(),
        label="national v13 physical benchmark",
    )
    w3_record = benchmark_raw.get("w3") or {}
    trial_raw = _read_json(
        Path(str(w3_record.get("trial_manifest_path") or "")).resolve(),
        label="national v13 measured W3 wrapper",
    )
    if (
        prepared.sharded_execution_plan_path.resolve() != expected_plan
        or prepared.sharded_execution_plan_sha256 != plan_authority["sha256"]
        or prepared.prefit_parity_manifest_path.resolve() != expected_prefit
        or prepared.prefit_parity_manifest_sha256
        != prefit_record.get("manifest_sha256")
        or prepared.scene_cache.source.resolve() != expected_source
        or trial_raw.get("source_candidate", {}).get("path")
        != str(expected_source)
        or trial_raw.get("source_candidate", {}).get("sha256")
        != config["input_lineage"]["source_candidate_sha256"]
        or authorization_identity.get("sharded_execution_plan_sha256")
        != prepared.sharded_execution_plan_sha256
        or authorization_identity.get("prefit_parity_manifest_sha256")
        != prepared.prefit_parity_manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 prepared plan/PREFIT/source differs from authorization"
        )
    base_lineage = sharded_execution_lineage_v12(
        dict(prepared.lineage), workspace_root=root
    )
    implementation_sha, _implementation_files = v13_implementation_lineage(root)
    base_lineage["v13_implementation_sha256"] = implementation_sha
    if dict(w3_record.get("lineage") or {}) != base_lineage:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 prepared lineage differs from measured W3"
        )
    attempt_base = _national_attempt_base_v13(
        config_sha256=CONFIG_SHA256,
        authorization_sha256=execution_authorization_sha256,
        benchmark_sha256=physical_benchmark_manifest_sha256,
        implementation_sha256=implementation_sha,
        source_candidate_sha256=config["input_lineage"][
            "source_candidate_sha256"
        ],
        sharded_execution_plan_sha256=(
            prepared.sharded_execution_plan_sha256
        ),
        prefit_parity_manifest_sha256=(
            prepared.prefit_parity_manifest_sha256
        ),
        canonical_output_dir=expected_canonical,
    )
    if attempt_id != _sha256_json(attempt_base):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 consumed attempt differs from its authorities"
        )
    _load_national_attempt_v13(
        marker_path=expected_marker,
        expected_sha256=attempt_marker_sha256,
        expected_payload=_national_attempt_payload_v13(
            base=attempt_base,
            attempt_id=attempt_id,
            stage_output_dir=expected_stage,
        ),
    )
    lineage = dict(base_lineage)
    lineage["physical_benchmark_manifest_sha256"] = (
        physical_benchmark_manifest_sha256
    )
    lineage["execution_authorization_sha256"] = execution_authorization_sha256
    specs = plan_query_shards(
        prepared.scene_cache.index, prepared.national_query_ids, 3
    )
    if (
        len(specs) != 3
        or tuple(value.folds for value in specs) != ((2,), (3,), (4,))
        or sum(value.query_count for value in specs) != NATIONAL_QUERY_COUNT
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 requires three exhaustive fold-owned shards"
        )
    guard_base = dict(
        current_national_worker_guard_v12(
            3, prepared.sharded_execution_plan_path
        )
    )
    guard_base_sha = str(guard_base.pop("record_sha256"))
    storage_free = shutil.disk_usage(prepared.executor.scratch_root).free
    storage_required = prepared.executor.maximum_temporary_gib * 1024**3
    runtime_guard = {
        **guard_base,
        "base_guard_sha256": guard_base_sha,
        "storage_free_bytes": storage_free,
        "storage_required_bytes": storage_required,
        "storage_free_pass": storage_free >= storage_required,
        "fallback_worker_count_allowed": False,
    }
    runtime_guard["record_sha256"] = _sha256_json(runtime_guard)
    try:
        _validate_runtime_resource_guard_v13(runtime_guard)
    except PhysicalBenchmarkInvariantErrorV13 as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            "STOP_V13_NATIONAL_RESOURCE_GUARD"
        ) from error
    phase_state["phase"] = "BUILD_ACTORS"
    output.mkdir(parents=True)
    jobs = _build_jobs_v13(
        prepared=prepared,
        specs=specs,
        output_dir=output,
        lineage=lineage,
        miniature_fit=False,
        execution_phase="NATIONAL_OOF_FOLDS_2_3_4_V13_BUILD",
    )
    build_manifests = spawn_manifest_workers_v12(
        BUILD_ENTRYPOINT_V13, jobs, max_workers=3
    )
    build_artifacts = tuple(
        load_shard_build_manifest_v12(path) for path in build_manifests
    )
    source_extent = config["cache_neutral_io_gate"]["source_extent"]
    source_projection = tuple(
        source_extent["production_projection"]["columns"]
    )
    national_row_groups, national_logical_bytes = (
        _national_projection_totals_v13(
            Path(prepared.scene_cache.source).resolve(), source_projection
        )
    )
    if (
        national_row_groups
        != int(source_extent["national_expected_unique_row_groups"])
        or national_logical_bytes
        != int(
            source_extent[
                "national_expected_logical_compressed_bytes"
            ]
        )
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 source footer differs from preregistered extent pins"
        )
    actual_reads = load_and_validate_build_row_group_ledgers_v13(
        prepared=prepared,
        specs=specs,
        build_artifacts=build_artifacts,
        lineage=lineage,
        expected_unique_row_groups=national_row_groups,
        expected_logical_compressed_bytes=national_logical_bytes,
    )
    row_group_ledger_records = [
        actor_row_group_ledger_record_v13(value, root=output)
        for value in actual_reads.artifacts
    ]
    build = seal_build_fragment_dataset_v13(
        build_artifacts,
        output / "build_fragment_dataset",
        relocation_root=output,
    )
    build = load_build_fragment_dataset_v13(
        build.manifest_path, verify_hashes=True, verify_logical=True
    )
    phase_state["phase"] = "PREFIT_RESOURCE_GUARD"
    try:
        fit_guard = _fit_resource_guard_v13(
            sharded_execution_plan_path=prepared.sharded_execution_plan_path,
            memory_used_maximum_bytes=int(
                config["benchmark_512"][
                    "maximum_w3_peak_machine_used_memory_bytes"
                ]
            ),
        )
    except Exception as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            "STOP_V13_NATIONAL_PREFIT_RESOURCE_GUARD"
        ) from error
    if fit_guard["status"] != "PASS_PREFIT_RESOURCES":
        raise PhysicalBenchmarkInvariantErrorV13(
            "STOP_V13_NATIONAL_PREFIT_RESOURCE_GUARD"
        )
    phase_state["phase"] = "SEQUENTIAL_SITE_FITS"
    backend = XGBoostFragmentedSiteLambdaMARTBackendV13(
        params=prepared.admission.xgb_params("site"),
        output_dir=output / "site_models",
    )
    models = fit_site_models_sequential_from_fragments_v13(backend, build)
    phase_state["phase"] = "REPLAY_ACTORS"
    replay_jobs = _replay_jobs_v13(
        prepared=prepared,
        specs=specs,
        builds=build_artifacts,
        models=models,
        heldout_merkle=build.heldout_logical_merkle_root,
        lineage=lineage,
        output=output,
        execution_phase="NATIONAL_OOF_FOLDS_2_3_4_V13_REPLAY",
    )
    scientific_manifests = spawn_manifest_workers_v12(
        REPLAY_ENTRYPOINT_V13, replay_jobs, max_workers=3
    )
    replay_fragments = tuple(
        load_compact_replay_fragment_v13(
            path.parent / "compact_fragment_manifest.json",
            verify_hashes=True,
            verify_logical=True,
        )
        for path in scientific_manifests
    )
    replay = seal_compact_replay_dataset_v13(
        replay_fragments,
        output / "compact_replay_dataset",
        relocation_root=output,
    )
    replay = load_compact_replay_dataset_v13(
        replay.manifest_path, verify_hashes=True, verify_logical=True
    )
    canonical = seal_zero_copy_canonical_run_v13(
        build,
        replay,
        models,
        output / "canonical_datasets",
        relocation_root=output,
    )
    canonical = load_zero_copy_canonical_run_v13(
        canonical.manifest_path, verify_hashes=True, verify_logical=True
    )
    gates = dict(aggregate_compact_replay_gates_v13(replay, frozen_gates))
    phase_state["phase"] = "SEAL_NATIONAL_MANIFEST"
    manifest_path = output / "manifest.json"
    inventory = _national_inventory_v13(output, manifest_path)
    identity = _national_identity_v13(
        authorization_sha256=execution_authorization_sha256,
        physical_benchmark_manifest_sha256=(
            physical_benchmark_manifest_sha256
        ),
        implementation_sha256=implementation_sha,
        lineage=lineage,
        source_candidate_sha256=config["input_lineage"][
            "source_candidate_sha256"
        ],
        sharded_execution_plan_sha256=(
            prepared.sharded_execution_plan_sha256
        ),
        prefit_parity_manifest_sha256=(
            prepared.prefit_parity_manifest_sha256
        ),
        runtime_resource_guard_sha256=runtime_guard["record_sha256"],
        fit_resource_guard_sha256=fit_guard["record_sha256"],
        attempt_marker_sha256=attempt_marker_sha256,
        attempt_id=attempt_id,
        canonical_output_dir=str(Path(canonical_output_dir).resolve()),
        artifact_inventory_sha256=_sha256_json(inventory),
        source_spec_sha256=[value.digest for value in specs],
        source_row_group_ledger_sha256=[
            value.manifest_sha256 for value in actual_reads.artifacts
        ],
        source_row_group_union_sha256=actual_reads.union_sha256,
        source_row_group_logical_compressed_bytes=(
            actual_reads.logical_compressed_bytes
        ),
        build_fragment_dataset_sha256=build.manifest_sha256,
        compact_replay_dataset_sha256=replay.manifest_sha256,
        zero_copy_canonical_run_sha256=canonical.manifest_sha256,
        gates=gates,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "EXECUTED_NO_READY_CLAIM",
        "run_id": _sha256_json(identity),
        "identity": identity,
        "run_kind": "NATIONAL_OOF_FOLDS_2_3_4_ZERO_COPY_V13",
        "config_sha256": CONFIG_SHA256,
        "authorization": {
            "path": str(Path(execution_authorization).resolve()),
            "sha256": execution_authorization_sha256,
            "benchmark_manifest_path": str(
                Path(physical_benchmark_manifest).resolve()
            ),
            "benchmark_manifest_sha256": physical_benchmark_manifest_sha256,
        },
        "source_candidate": {
            "path": str(prepared.scene_cache.source.resolve()),
            "sha256": config["input_lineage"]["source_candidate_sha256"],
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
            "path": str(Path(attempt_marker_path).resolve()),
            "sha256": attempt_marker_sha256,
            "attempt_id": attempt_id,
            "stage_directory_name": output.name,
        },
        "lineage": lineage,
        "query_count": NATIONAL_QUERY_COUNT,
        "worker_count": 3,
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "source_specs": [value.as_manifest() for value in specs],
        "source_row_group_ledgers": row_group_ledger_records,
        "build_fragment_dataset": _artifact_record(build.manifest_path, output),
        "compact_replay_dataset": _artifact_record(replay.manifest_path, output),
        "zero_copy_canonical_run": _artifact_record(
            canonical.manifest_path, output
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
    for _iteration in range(16):
        _atomic_json(manifest_path, manifest)
        if manifest_path.stat().st_nlink != 1:
            raise PhysicalBenchmarkInvariantErrorV13(
                "national v13 root manifest is hardlinked"
            )
        observed_bytes = child_bytes + manifest_path.stat().st_size
        if manifest["materialized_output_bytes"] == observed_bytes:
            break
        manifest["materialized_output_bytes"] = observed_bytes
    else:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 materialized-byte fixed point did not converge"
        )
    phase_state["phase"] = "STAGE_SEALED"
    return NationalFragmentOOFV13(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        run_id=str(manifest["run_id"]),
        query_count=NATIONAL_QUERY_COUNT,
        lineage=lineage,
        gates=gates,
    )


def execute_national_fragment_oof_v13(
    *,
    prepared: PreparedShardedExecutionV12,
    workspace_root: Path,
    output_dir: Path,
    physical_benchmark_manifest: Path,
    physical_benchmark_manifest_sha256: str,
    execution_authorization: Path,
    execution_authorization_sha256: str,
) -> NationalFragmentOOFV13:
    """Consume one authorized attempt, stage it, then atomically promote it."""

    root = Path(workspace_root).resolve()
    requested = Path(output_dir).resolve()
    benchmark_path = Path(physical_benchmark_manifest).resolve()
    authorization_path = Path(execution_authorization).resolve()
    config = load_physical_config_v13(
        root / "config/retrieval_all_human_v13_physical_execution.json"
    )
    protocol = config.get("national_execution_protocol")
    if not isinstance(protocol, Mapping):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 staging protocol is absent"
        )
    canonical = Path(str(protocol.get("canonical_output_dir") or "")).resolve()
    marker = Path(
        str(protocol.get("attempt_consumed_marker_path") or "")
    ).resolve()
    if (
        requested != canonical
        or canonical.exists()
        or marker.exists()
        or len(prepared.national_query_ids) != NATIONAL_QUERY_COUNT
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 canonical output/attempt is unavailable"
        )
    authorization = load_execution_authorization_v13(
        authorization_path,
        expected_authorization_sha256=execution_authorization_sha256,
        benchmark_manifest_path=benchmark_path,
        expected_benchmark_manifest_sha256=(
            physical_benchmark_manifest_sha256
        ),
        workspace_root=root,
    )
    authorization_identity = authorization.get("authorization_identity") or {}
    implementation_sha, implementation_files = v13_implementation_lineage(root)
    source_sha = str(config["input_lineage"]["source_candidate_sha256"])
    plan_sha = str(
        config["authorities"]["v12_sharded_plan"]["sha256"]
    )
    prefit_sha = str(prepared.prefit_parity_manifest_sha256)
    if (
        prepared.sharded_execution_plan_sha256 != plan_sha
        or authorization_identity.get("sharded_execution_plan_sha256")
        != plan_sha
        or authorization_identity.get("prefit_parity_manifest_sha256")
        != prefit_sha
        or prepared.scene_cache.source.resolve()
        != Path(str(config["input_lineage"]["source_candidate_path"])).resolve()
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 prepared authorities differ before attempt consumption"
        )
    attempt_base = _national_attempt_base_v13(
        config_sha256=CONFIG_SHA256,
        authorization_sha256=execution_authorization_sha256,
        benchmark_sha256=physical_benchmark_manifest_sha256,
        implementation_sha256=implementation_sha,
        source_candidate_sha256=source_sha,
        sharded_execution_plan_sha256=plan_sha,
        prefit_parity_manifest_sha256=prefit_sha,
        canonical_output_dir=canonical,
    )
    attempt_id = _sha256_json(attempt_base)
    expected_canonical, stage, expected_marker = _national_execution_paths_v13(
        config,
        requested_output=requested,
        attempt_id=attempt_id,
    )
    if expected_canonical != canonical or expected_marker != marker:
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 canonical paths differ from preregistration"
        )
    prefix = str(protocol["stage_directory_prefix"])
    if stage.exists() or any(canonical.parent.glob(f"{prefix}*")):
        raise PhysicalBenchmarkInvariantErrorV13(
            "national v13 prior stage forbids an automatic retry"
        )
    marker_payload = _national_attempt_payload_v13(
        base=attempt_base,
        attempt_id=attempt_id,
        stage_output_dir=stage,
    )
    phase_state = {"phase": "ATOMIC_ATTEMPT_MARKER_CREATION"}
    try:
        marker_sha = _consume_national_attempt_v13(
            marker_path=marker,
            payload=marker_payload,
        )
    except _NationalAttemptConsumptionErrorV13 as error:
        # The private exception proves that this process obtained O_EXCL.  A
        # marker failure before O_EXCL (including a competing existing marker)
        # never reaches this branch and must not publish a competing STOP.
        if marker.exists():
            _publish_national_exception_stop_v13(
                stage=stage,
                canonical=canonical,
                marker_path=marker,
                marker_sha256=_observed_marker_sha256_v13(marker),
                attempt_id=attempt_id,
                frozen_implementation_sha256=implementation_sha,
                frozen_implementation_files=implementation_files,
                config_sha256=CONFIG_SHA256,
                authorization_path=authorization_path,
                authorization_sha256=execution_authorization_sha256,
                benchmark_path=benchmark_path,
                benchmark_sha256=physical_benchmark_manifest_sha256,
                phase=phase_state["phase"],
                error=error,
            )
        raise
    phase_state["phase"] = "AUTHORIZED_ATTEMPT_CONSUMED"
    promoted = False
    try:
        product = _execute_national_fragment_oof_stage_v13(
            prepared=prepared,
            workspace_root=root,
            output_dir=stage,
            physical_benchmark_manifest=benchmark_path,
            physical_benchmark_manifest_sha256=(
                physical_benchmark_manifest_sha256
            ),
            execution_authorization=authorization_path,
            execution_authorization_sha256=execution_authorization_sha256,
            canonical_output_dir=canonical,
            attempt_marker_path=marker,
            attempt_marker_sha256=marker_sha,
            attempt_id=attempt_id,
            phase_state=phase_state,
        )
        if (
            product.manifest_path != stage / "manifest.json"
            or not stage.is_dir()
            or canonical.exists()
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "national v13 sealed stage cannot be promoted"
            )
        phase_state["phase"] = "ATOMIC_PROMOTION"
        os.rename(stage, canonical)
        promoted = True
        _fsync_directory_v13(canonical.parent)
        phase_state["phase"] = "STRICT_PROMOTED_RELOAD"
        return load_national_fragment_oof_v13(
            canonical / "manifest.json",
            expected_manifest_sha256=product.manifest_sha256,
            workspace_root=root,
        )
    except BaseException as error:
        if promoted and canonical.exists() and not stage.exists():
            os.rename(canonical, stage)
            _fsync_directory_v13(stage.parent)
        _publish_national_exception_stop_v13(
            stage=stage,
            canonical=canonical,
            marker_path=marker,
            marker_sha256=marker_sha,
            attempt_id=attempt_id,
            frozen_implementation_sha256=implementation_sha,
            frozen_implementation_files=implementation_files,
            config_sha256=CONFIG_SHA256,
            authorization_path=authorization_path,
            authorization_sha256=execution_authorization_sha256,
            benchmark_path=benchmark_path,
            benchmark_sha256=physical_benchmark_manifest_sha256,
            phase=phase_state.get("phase", "UNKNOWN"),
            error=error,
        )
        raise


__all__ = [
    "ENGINE_ID",
    "NationalFragmentOOFV13",
    "SCHEMA_VERSION",
    "execute_national_fragment_oof_v13",
    "load_national_fragment_oof_v13",
]
