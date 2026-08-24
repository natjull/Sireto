"""Fail-closed physical benchmark policy for exact sharded retrieval v12.

This module deliberately does not implement query workers, cascade execution,
or shard merging.  Those operations belong to the single sharded OOF core in
``retrieval_fold_fused_sharded_oof_v12``.  Here we only:

* measure one invocation of that core with 1, 2, or 3 workers;
* require exact logical digests against the one-worker reference;
* apply the preregistered RAM/swap/speed/I/O decision; and
* publish an authorization that a future national OOF CLI can load.

No fold 0/1 value is accepted at any boundary.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import psutil
import pyarrow.parquet as pq

from .retrieval_fold_fused_oof_v12 import FoldFusedInvariantError, sha256_file
from .retrieval_fold_fused_prefit_bridge_v12 import (
    load_prefit_bridge_artifact_v12,
    validate_prefit_bridge_against_reference_v12,
)
from .retrieval_fold_fused_runtime_v12 import load_prefit_parity_evidence_v12
from .retrieval_fold_fused_sharded_oof_v12 import (
    ENGINE_ID as SHARDED_ENGINE_ID,
    SHARDED_RUN_SCHEMA_VERSION,
    ShardSpecV12,
    load_fitted_site_model_records_v12,
    load_heldout_master_record_v12,
    load_shard_build_manifest_v12,
    load_shard_replay_manifest_v12,
)


SCHEMA_VERSION = "sireto-retrieval-v12-physical-worker-benchmark-1"
AUTHORIZATION_SCHEMA_VERSION = (
    "sireto-retrieval-v12-sharded-oof-authorization-1"
)
SAMPLE_QUERY_COUNT = 512
ALLOWED_FOLDS = (2, 3, 4)
GIB = 1024**3
REQUIRED_DIGESTS = (
    "prefit_bridge",
    "train_matrices",
    "query_groups",
    "heldout_spool",
    "miniature_models",
    "prerank500",
    "top100",
    "outcomes",
)
MATERIALIZED_FAMILIES = (
    "worker_train_matrices",
    "worker_heldout_spools",
    "merged_train_matrices",
    "merged_heldout_spool",
    "site_models",
    "worker_replay_outputs",
    "merged_replay_outputs",
)
EXECUTION_LINEAGE_KEYS = (
    "sharded_implementation_sha256",
    "runtime_environment_sha256",
)
SHARDED_IMPLEMENTATION_FILES = (
    "requirements.txt",
    "config/retrieval_all_human_v12.json",
    "config/retrieval_all_human_v12_fold_fused_executor.json",
    "config/retrieval_all_human_v12_sharded_execution_plan.json",
    "scripts/benchmark_retrieval_fold_fused_workers_v12.py",
    "scripts/run_retrieval_fold_fused_sharded_oof_v12.py",
    "src/xgb_matcher/__init__.py",
    "src/xgb_matcher/retrieval_fold_fused_v12.py",
    "src/xgb_matcher/retrieval_fold_fused_oof_v12.py",
    "src/xgb_matcher/retrieval_fold_fused_runtime_v12.py",
    "src/xgb_matcher/retrieval_fold_fused_parity_v12.py",
    "src/xgb_matcher/retrieval_fold_fused_prefit_bridge_v12.py",
    "src/xgb_matcher/retrieval_fold_fused_heldout_spool_v12.py",
    "src/xgb_matcher/retrieval_fold_fused_site_matrix_v12.py",
    "src/xgb_matcher/retrieval_v12_lazy_site_features.py",
    "src/xgb_matcher/hierarchical_retrieval.py",
    "src/xgb_matcher/official_evidence_retrieval.py",
    "src/xgb_matcher/official_evidence_tantivy.py",
    "src/xgb_matcher/official_sector_evidence.py",
    "src/xgb_matcher/official_source_sync.py",
    "src/xgb_matcher/retrieval_hierarchical_admission_v12.py",
    "src/xgb_matcher/rne_legal_entity_roles.py",
    "src/xgb_matcher/retrieval_sidecars_v11.py",
    "src/xgb_matcher/retrieval_top100_cascade_v12.py",
    "src/xgb_matcher/retrieval_v12_mmap_sidecar.py",
    "src/xgb_matcher/retrieval_v12_reader_parity.py",
    "src/xgb_matcher/retrieval_v12_scene_ir.py",
    "src/xgb_matcher/siren_dossier.py",
    "src/xgb_matcher/retrieval_fold_fused_sharded_oof_v12.py",
    "src/xgb_matcher/retrieval_fold_fused_sharded_runtime_v12.py",
    "src/xgb_matcher/retrieval_fold_fused_sharded_preflight_v12.py",
    "src/xgb_matcher/retrieval_fold_fused_physical_benchmark_v12.py",
)


class PhysicalBenchmarkInvariantError(FoldFusedInvariantError):
    """The physical benchmark cannot issue a safe worker authorization."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _require_lineage(lineage: Mapping[str, str]) -> dict[str, str]:
    output = {str(key): str(value) for key, value in lineage.items()}
    if not output or any(not key or not _is_sha256(value) for key, value in output.items()):
        raise PhysicalBenchmarkInvariantError("physical benchmark lineage is not SHA-256 sealed")
    return output


def _validate_digests(digests: Mapping[str, str]) -> dict[str, str]:
    output = {str(key): str(value) for key, value in digests.items()}
    if set(output) != set(REQUIRED_DIGESTS) or any(
        not _is_sha256(value) for value in output.values()
    ):
        raise PhysicalBenchmarkInvariantError(
            "physical trial digest schema is incomplete or invalid"
        )
    return {name: output[name] for name in REQUIRED_DIGESTS}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sharded_execution_lineage_v12(
    scientific_lineage: Mapping[str, str],
    *,
    workspace_root: Path | None = None,
) -> dict[str, str]:
    """Bind one execution to exact code bytes and runtime dependencies."""

    base = _require_lineage(scientific_lineage)
    if any(key in base for key in EXECUTION_LINEAGE_KEYS):
        raise PhysicalBenchmarkInvariantError(
            "scientific lineage already contains execution fingerprints"
        )
    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    files: dict[str, str] = {}
    for relative in SHARDED_IMPLEMENTATION_FILES:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise PhysicalBenchmarkInvariantError(
                f"sharded implementation file is missing: {relative}"
            )
        files[relative] = sha256_file(path)
    packages = {}
    for name in ("duckdb", "numpy", "pandas", "psutil", "pyarrow", "rapidfuzz", "xgboost"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as error:
            raise PhysicalBenchmarkInvariantError(
                f"runtime dependency is missing: {name}"
            ) from error
    environment = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "packages": packages,
    }
    return {
        **base,
        "sharded_implementation_sha256": _sha256_json(files),
        "runtime_environment_sha256": _sha256_json(environment),
    }


def current_national_worker_guard_v12(
    authorized_worker_count: int,
    sharded_execution_plan_path: Path,
    *,
    memory_reader: Callable[[], Any] = psutil.virtual_memory,
    swap_reader: Callable[[], Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> Mapping[str, Any]:
    """Choose W3 or W2 once from the preregistered current-state guard.

    The physical benchmark authorizes at most three concurrent workers.  The
    national process still measures the *current* Mac state immediately before
    the first spawn.  Any unavailable/invalid measurement or any W3 resource
    failure deterministically falls back to W2, as preregistered; it never
    retries or changes concurrency after opening a shard.
    """

    if swap_reader is None:
        swap_reader = _vm_stat_swap_observation
    plan_path = Path(sharded_execution_plan_path).resolve()
    if not plan_path.is_file():
        raise PhysicalBenchmarkInvariantError("runtime guard plan is missing")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    process = plan.get("process_policy") or {}
    guard = process.get("preferred_w3_requires_all") or {}
    observation_seconds = float(guard.get("swapout_observation_seconds", -1))
    threshold = int(guard.get("machine_used_ram_maximum_bytes", -1))
    if (
        authorized_worker_count not in (2, 3)
        or process.get("decision_phase") != "ONCE_BEFORE_FIRST_SHARD_OPEN"
        or process.get("shard_count") != 3
        or process.get("preferred_maximum_concurrent_workers") != 3
        or process.get("fallback_maximum_concurrent_workers") != 2
        or process.get("measurement_failure_policy") != "FALLBACK_W2"
        or process.get("runtime_change_after_first_shard_open") != "FORBIDDEN"
        or process.get("automatic_retry_with_different_worker_count") is not False
        or guard.get("machine_used_ram_metric") != "psutil.virtual_memory().used"
        or guard.get("swapout_metric") != "vm_stat.Swapouts"
        or observation_seconds != 5.0
        or threshold != 14 * GIB
        or guard.get("swapout_delta_pages_required") != 0
    ):
        raise PhysicalBenchmarkInvariantError("runtime worker guard policy drifted")

    def swap_value(value: Any) -> tuple[int, int, str]:
        if isinstance(value, Mapping):
            pages = int(value.get("swapouts_pages", -1))
            page_size = int(value.get("page_size_bytes", -1))
            source = str(value.get("source") or "")
        else:
            # Kept for deterministic tests/injected readers only.  Production
            # uses ``_vm_stat_swap_observation`` and therefore publishes the
            # real page size and raw page counter.
            pages = int(value)
            page_size = 1
            source = "INJECTED_COUNTER"
        if pages < 0 or page_size <= 0 or not source:
            raise ValueError("invalid swap observation")
        return pages, page_size, source

    before_timestamp = float(clock())
    measurement_valid = True
    before_memory: int | None = None
    after_memory: int | None = None
    before_pages: int | None = None
    after_pages: int | None = None
    page_size: int | None = None
    swap_source: str | None = None
    try:
        before_memory = int(memory_reader().used)
        before_pages, page_size, swap_source = swap_value(swap_reader())
        if before_memory < 0:
            raise ValueError("invalid memory observation")
    except Exception:  # measurement failure is a preregistered W2 fallback
        measurement_valid = False
    sleeper(observation_seconds)
    after_timestamp = float(clock())
    try:
        after_memory = int(memory_reader().used)
        after_pages, after_page_size, after_source = swap_value(swap_reader())
        if (
            after_memory < 0
            or page_size != after_page_size
            or swap_source != after_source
            or before_pages is None
            or after_pages < before_pages
        ):
            raise ValueError("inconsistent resource observation")
    except Exception:  # measurement failure is a preregistered W2 fallback
        measurement_valid = False
    peak_used = (
        max(before_memory, after_memory)
        if before_memory is not None and after_memory is not None
        else None
    )
    swap_delta_pages = (
        after_pages - before_pages
        if before_pages is not None and after_pages is not None
        else None
    )
    w3_resources_pass = bool(
        measurement_valid
        and peak_used is not None
        and peak_used <= threshold
        and swap_delta_pages == 0
    )
    selected = 3 if authorized_worker_count == 3 and w3_resources_pass else 2
    if authorized_worker_count == 2:
        reason = "PHYSICAL_BENCHMARK_AUTHORIZED_W2_ONLY"
    elif not measurement_valid:
        reason = "W3_MEASUREMENT_UNAVAILABLE_FALLBACK_W2"
    elif peak_used is not None and peak_used > threshold:
        reason = "W3_MEMORY_LIMIT_EXCEEDED_FALLBACK_W2"
    elif swap_delta_pages != 0:
        reason = "W3_SWAPOUT_ACTIVITY_FALLBACK_W2"
    else:
        reason = "W3_CURRENT_STATE_PASS"
    record = {
        "schema_version": "sireto-retrieval-v12-national-worker-guard-1",
        "status": "PASS_W3" if selected == 3 else "PASS_W2_FALLBACK",
        "selection_reason": reason,
        "physical_authorized_maximum_concurrent_workers": authorized_worker_count,
        "selected_maximum_concurrent_workers": selected,
        "fixed_shard_count": 3,
        "memory_metric": "psutil.virtual_memory().used",
        "memory_used_before_bytes": before_memory,
        "memory_used_after_bytes": after_memory,
        "memory_used_peak_bytes": peak_used,
        "memory_threshold_bytes": threshold,
        "swapout_metric": "vm_stat.Swapouts",
        "swapout_source": swap_source,
        "swapout_page_size_bytes": page_size,
        "swapout_before_pages": before_pages,
        "swapout_after_pages": after_pages,
        "swapout_delta_pages": swap_delta_pages,
        "swapout_delta_bytes": (
            swap_delta_pages * page_size
            if swap_delta_pages is not None and page_size is not None
            else None
        ),
        "observation_seconds": observation_seconds,
        "observation_started_at_unix_seconds": before_timestamp,
        "observation_finished_at_unix_seconds": after_timestamp,
        "measurement_valid": measurement_valid,
        "measured_before_first_shard_open": True,
        "worker_count_change_after_measurement": False,
        "automatic_retry_allowed": False,
    }
    return {**record, "record_sha256": _sha256_json(record)}


@dataclass(frozen=True)
class PhysicalBenchmarkThresholdsV12:
    maximum_machine_used_memory_bytes: int = 14 * GIB
    maximum_swap_out_bytes: int = 0
    maximum_io_ratio: float = 1.15
    minimum_speedup_w3: float = 2.2
    minimum_speedup_w2: float = 1.7

    def __post_init__(self) -> None:
        if (
            self.maximum_machine_used_memory_bytes != 14 * GIB
            or self.maximum_swap_out_bytes != 0
            or not math.isclose(self.maximum_io_ratio, 1.15)
            or not math.isclose(self.minimum_speedup_w3, 2.2)
            or not math.isclose(self.minimum_speedup_w2, 1.7)
        ):
            raise PhysicalBenchmarkInvariantError(
                "physical worker thresholds differ from the frozen decision"
            )


@dataclass(frozen=True)
class PhysicalResourceMetricsV12:
    wall_seconds: float
    coordinator_cpu_seconds: float
    children_cpu_seconds: float
    machine_busy_cpu_seconds: float
    peak_process_tree_rss_bytes: int
    peak_machine_used_memory_bytes: int
    minimum_machine_available_memory_bytes: int
    swap_out_delta_bytes: int
    process_read_bytes: int
    process_write_bytes: int
    machine_read_bytes: int
    machine_write_bytes: int
    sample_interval_seconds: float

    def __post_init__(self) -> None:
        numeric = asdict(self)
        if any(
            (isinstance(value, float) and (not math.isfinite(value) or value < 0))
            or (isinstance(value, int) and value < 0)
            for value in numeric.values()
        ) or self.wall_seconds <= 0:
            raise PhysicalBenchmarkInvariantError("physical resource metrics are invalid")

    @property
    def process_io_bytes(self) -> int:
        return int(self.process_read_bytes + self.process_write_bytes)

    @property
    def machine_io_bytes(self) -> int:
        return int(self.machine_read_bytes + self.machine_write_bytes)

    @property
    def decision_io_bytes(self) -> int:
        # Process-tree counters isolate the benchmark from unrelated disk
        # traffic.  System counters remain published as a machine audit.
        return self.process_io_bytes


@dataclass(frozen=True)
class PhysicalTrialProductV12:
    worker_count: int
    manifest_path: Path
    query_count: int
    folds: tuple[int, ...]
    lineage: Mapping[str, str]
    digests: Mapping[str, str]
    materialized_io_bytes: int
    site_model_bytes: int = 0
    materialized_families: Mapping[str, int] | None = None
    fold0_opened: bool = False
    fold1_opened: bool = False
    positive_injection: bool = False

    def __post_init__(self) -> None:
        if self.worker_count not in (1, 2, 3):
            raise PhysicalBenchmarkInvariantError("physical trial worker count is invalid")
        if self.query_count != SAMPLE_QUERY_COUNT or tuple(self.folds) != ALLOWED_FOLDS:
            raise PhysicalBenchmarkInvariantError("physical trial population/folds drifted")
        if self.fold0_opened or self.fold1_opened or self.positive_injection:
            raise PhysicalBenchmarkInvariantError("physical trial opened forbidden truth/folds")
        families = dict(self.materialized_families or {})
        if (
            self.materialized_io_bytes <= 0
            or self.site_model_bytes < 0
            or self.site_model_bytes > self.materialized_io_bytes
            or set(families) != set(MATERIALIZED_FAMILIES)
            or any(int(value) < 0 for value in families.values())
            or sum(int(value) for value in families.values())
            > self.materialized_io_bytes
            or int(families.get("site_models", -1)) != self.site_model_bytes
        ):
            raise PhysicalBenchmarkInvariantError("physical trial materialized I/O is missing")
        if not Path(self.manifest_path).is_file():
            raise FileNotFoundError(self.manifest_path)
        _require_lineage(self.lineage)
        _validate_digests(self.digests)


def physical_trial_product_from_sharded_v12(artifact: Any) -> PhysicalTrialProductV12:
    """Bind the benchmark policy to the unique sharded execution core.

    The adapter deliberately derives every comparison digest from a complete
    sealed core artifact.  In particular, held-out parity covers the merged
    Parquet payload SHA-256, not just query counts or sequence metadata.
    """

    manifest_path = Path(artifact.manifest_path).resolve()
    if (
        not manifest_path.is_file()
        or not _is_sha256(getattr(artifact, "manifest_sha256", None))
        or sha256_file(manifest_path) != artifact.manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantError("sharded trial manifest hash drifted")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    lineage = _require_lineage(getattr(artifact, "lineage", {}))
    specs = tuple(getattr(artifact, "specs", ()))
    shard_count = len(specs)
    workers = int(
        raw.get("maximum_concurrent_workers")
        or raw.get("worker_count")
        or 0
    )
    required = {
        "status": "EXECUTED_NO_READY_CLAIM",
        "run_kind": "PHYSICAL_BENCHMARK_MINIATURE",
        "miniature_fit": True,
        "logical_recipe_changes": False,
        "query_count": SAMPLE_QUERY_COUNT,
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "worker_count": workers,
        "shard_count": shard_count,
        "maximum_concurrent_workers": workers,
        "lineage": lineage,
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise PhysicalBenchmarkInvariantError("sharded trial safety claims drifted")
    if (
        int(getattr(artifact, "query_count", -1)) != SAMPLE_QUERY_COUNT
        or tuple(getattr(artifact, "folds", ())) != ALLOWED_FOLDS
        or workers not in (1, 2, 3)
        or shard_count != (1 if workers == 1 else 3)
    ):
        raise PhysicalBenchmarkInvariantError("sharded trial population drifted")

    train = getattr(artifact, "train", None)
    heldout = getattr(artifact, "heldout", None)
    models = getattr(artifact, "models", None)
    replay = getattr(artifact, "replay", None)
    prefit_bridge = getattr(artifact, "prefit_bridge", None)
    if (
        train is None
        or heldout is None
        or replay is None
        or prefit_bridge is None
        or not isinstance(models, Mapping)
        or set(models) != set(ALLOWED_FOLDS)
    ):
        raise PhysicalBenchmarkInvariantError("sharded trial products are incomplete")
    train_sha = {
        str(fold): str(train.matrices[fold].sha256) for fold in ALLOWED_FOLDS
    }
    group_sha = {
        str(fold): str(train.group_sequence_sha256[fold]) for fold in ALLOWED_FOLDS
    }
    model_sha = {str(fold): str(models[fold].sha256) for fold in ALLOWED_FOLDS}
    replay_sha = {
        name: str(replay.outputs[name].sha256)
        for name in ("top500", "top100", "outcomes")
    }
    heldout_sha = str(heldout.artifact.sha256)
    if any(
        not _is_sha256(value)
        for value in (
            *train_sha.values(),
            *group_sha.values(),
            *model_sha.values(),
            *replay_sha.values(),
            heldout_sha,
            str(prefit_bridge.sha256),
        )
    ):
        raise PhysicalBenchmarkInvariantError("sharded trial product digest is invalid")

    core_digests = dict(getattr(artifact, "digests", {}) or {})
    raw_digests = raw.get("digests")
    if not isinstance(raw_digests, Mapping) or dict(raw_digests) != core_digests:
        raise PhysicalBenchmarkInvariantError("sharded trial digest manifest drifted")
    expected_core = {
        "train_matrix": train_sha,
        "group_sequence": group_sha,
        "heldout_spool": heldout_sha,
        "model_by_outer": model_sha,
        "prerank500": replay_sha["top500"],
        "top100": replay_sha["top100"],
        "outcomes": replay_sha["outcomes"],
        "prefit_bridge": str(prefit_bridge.sha256),
    }
    if core_digests != expected_core:
        raise PhysicalBenchmarkInvariantError(
            "sharded core did not seal the exact functional-plus-PREFIT parity surface"
        )
    core_io = dict(getattr(artifact, "io_bytes", {}) or {})
    if (
        not isinstance(raw.get("io_bytes"), Mapping)
        or dict(raw["io_bytes"]) != core_io
        or int(core_io.get("total_materialized", 0)) <= 0
    ):
        raise PhysicalBenchmarkInvariantError("sharded trial materialized I/O drifted")
    digests = {
        "prefit_bridge": str(prefit_bridge.sha256),
        "train_matrices": _sha256_json(train_sha),
        "query_groups": _sha256_json(group_sha),
        "heldout_spool": heldout_sha,
        "miniature_models": _sha256_json(model_sha),
        "prerank500": replay_sha["top500"],
        "top100": replay_sha["top100"],
        "outcomes": replay_sha["outcomes"],
    }
    families = {
        "worker_train_matrices": sum(
            matrix.path.stat().st_size
            for shard in artifact.build_artifacts
            for matrix in shard.train_matrices.values()
        ),
        "worker_heldout_spools": sum(
            shard.heldout_master.size_bytes for shard in artifact.build_artifacts
        ),
        "merged_train_matrices": sum(
            train.matrices[fold].path.stat().st_size for fold in ALLOWED_FOLDS
        ),
        "merged_heldout_spool": heldout.artifact.size_bytes,
        "site_models": sum(
            Path(models[fold].path).stat().st_size for fold in ALLOWED_FOLDS
        ),
        "worker_replay_outputs": sum(
            record.path.stat().st_size
            for shard in artifact.replay_artifacts
            for record in shard.outputs.values()
        ),
        "merged_replay_outputs": sum(
            replay.outputs[name].path.stat().st_size
            for name in ("top500", "top100", "outcomes")
        ),
    }
    return PhysicalTrialProductV12(
        worker_count=workers,
        manifest_path=manifest_path,
        query_count=SAMPLE_QUERY_COUNT,
        folds=ALLOWED_FOLDS,
        lineage=lineage,
        digests=digests,
        materialized_io_bytes=int(
            dict(getattr(artifact, "io_bytes", {}) or {}).get(
                "total_materialized", 0
            )
        ),
        site_model_bytes=families["site_models"],
        materialized_families=families,
    )


def _resolve_child_artifact_v12(root: Path, value: Any, *, label: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if root != path and root not in path.parents:
        raise PhysicalBenchmarkInvariantError(
            f"physical trial {label} escaped its sealed root"
        )
    if not path.is_file():
        raise PhysicalBenchmarkInvariantError(
            f"physical trial {label} is missing"
        )
    return path


def _group_sequence_from_train_v12(path: Path) -> tuple[str, int, str]:
    """Return rows, groups and their immutable (query, fold) digest."""

    parquet = pq.ParquetFile(path)
    groups: list[list[Any]] = []
    rows = 0
    seen: set[str] = set()
    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(
            row_group, columns=["query_id", "scene_fold"]
        )
        query_ids = {str(value) for value in table["query_id"].to_pylist()}
        folds = {int(value.as_py()) for value in table["scene_fold"]}
        if len(query_ids) != 1 or len(folds) != 1:
            raise PhysicalBenchmarkInvariantError(
                "physical trial train group identity drifted"
            )
        query_id = next(iter(query_ids))
        fold = next(iter(folds))
        if query_id in seen or fold not in ALLOWED_FOLDS:
            raise PhysicalBenchmarkInvariantError(
                "physical trial train groups overlap or open fold 0/1"
            )
        seen.add(query_id)
        groups.append([query_id, fold])
        rows += table.num_rows
    return rows, len(groups), _sha256_json(groups)


def _validate_ranked_output_v12(
    path: Path,
    *,
    expected_sequence: tuple[tuple[str, int], ...],
    cap: int,
) -> int:
    parquet = pq.ParquetFile(path)
    observed: list[tuple[str, int]] = []
    rows = 0
    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(
            row_group, columns=["query_id", "fold", "rank"]
        )
        query_ids = {str(value) for value in table["query_id"].to_pylist()}
        folds = {int(value.as_py()) for value in table["fold"]}
        ranks = tuple(int(value.as_py()) for value in table["rank"])
        if (
            len(query_ids) != 1
            or len(folds) != 1
            or not 1 <= table.num_rows <= cap
            or ranks != tuple(range(1, table.num_rows + 1))
        ):
            raise PhysicalBenchmarkInvariantError(
                "physical trial ranked output cap/order drifted"
            )
        observed.append((next(iter(query_ids)), next(iter(folds))))
        rows += table.num_rows
    if tuple(observed) != expected_sequence:
        raise PhysicalBenchmarkInvariantError(
            "physical trial ranked output population drifted"
        )
    return rows


def physical_trial_product_from_sharded_manifest_v12(
    manifest_path: Path,
    *,
    expected_lineage: Mapping[str, str],
    expected_worker_count: int,
    expected_prefit_parity_manifest_path: Path | None = None,
    expected_prefit_parity_manifest_sha256: str | None = None,
    expected_sharded_execution_plan_sha256: str | None = None,
) -> PhysicalTrialProductV12:
    """Reconstruct benchmark evidence from the physical files themselves.

    The outer benchmark manifest is not an authority for a child run.  This
    loader parses the sharded core manifest, reloads every worker manifest,
    verifies every materialized file hash/cardinality, and only then derives
    the functional digests plus the direct historical-PREFIT bridge used by
    the worker decision.
    """

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise PhysicalBenchmarkInvariantError("physical trial manifest is missing")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhysicalBenchmarkInvariantError(
            "physical trial manifest is not valid JSON"
        ) from error
    if not isinstance(raw, Mapping):
        raise PhysicalBenchmarkInvariantError("physical trial manifest is invalid")
    lineage = _require_lineage(expected_lineage)
    shard_count = 1 if expected_worker_count == 1 else 3
    required = {
        "schema_version": SHARDED_RUN_SCHEMA_VERSION,
        "engine_id": SHARDED_ENGINE_ID,
        "run_kind": "PHYSICAL_BENCHMARK_MINIATURE",
        "status": "EXECUTED_NO_READY_CLAIM",
        "query_count": SAMPLE_QUERY_COUNT,
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "logical_recipe_changes": False,
        "miniature_fit": True,
        "lineage": lineage,
        "worker_count": expected_worker_count,
        "shard_count": shard_count,
        "maximum_concurrent_workers": expected_worker_count,
    }
    if expected_worker_count not in (1, 2, 3) or any(
        raw.get(key) != value for key, value in required.items()
    ):
        raise PhysicalBenchmarkInvariantError(
            "physical trial core claims do not match the frozen benchmark"
        )

    specs_raw = raw.get("specs")
    if not isinstance(specs_raw, list) or len(specs_raw) != shard_count:
        raise PhysicalBenchmarkInvariantError("physical trial shard plan is incomplete")
    specs = tuple(ShardSpecV12.from_manifest(value) for value in specs_raw)
    if (
        tuple(spec.shard_ordinal for spec in specs) != tuple(range(shard_count))
        or specs[0].sequence_start != 0
        or any(
            specs[index].sequence_stop != specs[index + 1].sequence_start
            for index in range(len(specs) - 1)
        )
        or specs[-1].sequence_stop != SAMPLE_QUERY_COUNT
        or sum(spec.query_count for spec in specs) != SAMPLE_QUERY_COUNT
        or tuple(sorted({fold for spec in specs for fold in spec.query_folds}))
        != ALLOWED_FOLDS
        or (shard_count == 3 and tuple(spec.folds for spec in specs) != ((2,), (3,), (4,)))
    ):
        raise PhysicalBenchmarkInvariantError("physical trial shard topology drifted")
    expected_sequence = tuple(
        (query_id, fold)
        for spec in specs
        for query_id, fold in zip(spec.query_ids, spec.query_folds, strict=True)
    )
    if len({query_id for query_id, _fold in expected_sequence}) != SAMPLE_QUERY_COUNT:
        raise PhysicalBenchmarkInvariantError("physical trial query population overlaps")

    digests_raw = raw.get("digests")
    execution_evidence = raw.get("execution_evidence")
    if not isinstance(digests_raw, Mapping) or not isinstance(
        execution_evidence, Mapping
    ):
        raise PhysicalBenchmarkInvariantError("physical trial core digests are missing")
    if expected_prefit_parity_manifest_sha256 is not None:
        expected_execution_evidence = {
            "phase": "PHYSICAL_PARITY_BENCHMARK_512",
            "sharded_implementation_sha256": lineage.get(
                "sharded_implementation_sha256"
            ),
            "runtime_environment_sha256": lineage.get(
                "runtime_environment_sha256"
            ),
            "prefit_parity_manifest_sha256": (
                expected_prefit_parity_manifest_sha256
            ),
            "sharded_execution_plan_sha256": (
                expected_sharded_execution_plan_sha256
            ),
        }
        if dict(execution_evidence) != expected_execution_evidence:
            raise PhysicalBenchmarkInvariantError(
                "physical trial execution authority drifted"
            )
        if expected_prefit_parity_manifest_path is None:
            raise PhysicalBenchmarkInvariantError(
                "physical trial lacks the direct PREFIT bridge authority"
            )
    run_identity = {
        "engine_id": SHARDED_ENGINE_ID,
        "run_kind": "PHYSICAL_BENCHMARK_MINIATURE",
        "query_count": SAMPLE_QUERY_COUNT,
        "query_sequence_sha256": _sha256_json(
            [[query_id, fold] for query_id, fold in expected_sequence]
        ),
        "lineage": lineage,
        "spec_sha256": [spec.digest for spec in specs],
        "shard_count": shard_count,
        "maximum_concurrent_workers": expected_worker_count,
        "execution_evidence": dict(execution_evidence),
        "digests": dict(digests_raw),
        "miniature_fit": True,
    }
    if raw.get("run_identity") != run_identity or raw.get("run_id") != _sha256_json(
        run_identity
    ):
        raise PhysicalBenchmarkInvariantError("physical trial run identity drifted")

    root = path.parent
    build_hashes = raw.get("worker_build_manifest_sha256")
    replay_hashes = raw.get("worker_replay_manifest_sha256")
    if (
        not isinstance(build_hashes, list)
        or not isinstance(replay_hashes, list)
        or len(build_hashes) != shard_count
        or len(replay_hashes) != shard_count
    ):
        raise PhysicalBenchmarkInvariantError("physical trial worker lineage is incomplete")
    builds = []
    replays = []
    for index, spec in enumerate(specs):
        build_path = root / "worker_build" / spec.shard_id / "manifest.json"
        replay_path = root / "worker_replay" / spec.shard_id / "manifest.json"
        build = load_shard_build_manifest_v12(build_path)
        replay = load_shard_replay_manifest_v12(replay_path)
        if (
            build.manifest_sha256 != build_hashes[index]
            or replay.manifest_sha256 != replay_hashes[index]
            or build.spec.as_manifest() != spec.as_manifest()
            or replay.spec.as_manifest() != spec.as_manifest()
            or dict(build.lineage) != lineage
            or dict(replay.lineage) != lineage
            or replay.build_manifest_sha256 != build.manifest_sha256
        ):
            raise PhysicalBenchmarkInvariantError(
                "physical trial worker artifact lineage drifted"
            )
        builds.append(build)
        replays.append(replay)

    train_records = raw.get("train_matrices")
    if not isinstance(train_records, Mapping) or set(train_records) != {
        str(fold) for fold in ALLOWED_FOLDS
    }:
        raise PhysicalBenchmarkInvariantError("physical trial train matrices are incomplete")
    train_sha: dict[str, str] = {}
    group_sha: dict[str, str] = {}
    train_paths: list[Path] = []
    for outer in ALLOWED_FOLDS:
        record = train_records[str(outer)]
        if not isinstance(record, Mapping):
            raise PhysicalBenchmarkInvariantError("physical trial train record is invalid")
        artifact_path = _resolve_child_artifact_v12(
            root, record.get("path"), label=f"outer{outer} train matrix"
        )
        digest = str(record.get("sha256") or "")
        names = tuple(str(value) for value in record.get("feature_names") or ())
        rows, groups, group_digest = _group_sequence_from_train_v12(artifact_path)
        if (
            not _is_sha256(digest)
            or sha256_file(artifact_path) != digest
            or int(record.get("rows", -1)) != rows
            or int(record.get("queries", -1)) != groups
            or tuple(int(value) for value in record.get("fit_folds") or ())
            != tuple(fold for fold in ALLOWED_FOLDS if fold != outer)
            or not names
            or record.get("feature_schema_sha256") != _sha256_json(list(names))
            or record.get("group_sequence_sha256") != group_digest
        ):
            raise PhysicalBenchmarkInvariantError(
                "physical trial train matrix content/schema drifted"
            )
        train_sha[str(outer)] = digest
        group_sha[str(outer)] = group_digest
        train_paths.append(artifact_path)

    held_record = raw.get("heldout_master")
    if not isinstance(held_record, Mapping):
        raise PhysicalBenchmarkInvariantError("physical trial heldout spool is missing")
    heldout = load_heldout_master_record_v12(held_record, root=root)
    if (
        root not in heldout.path.parents
        or root not in heldout.manifest_path.parents
        or heldout.query_count != SAMPLE_QUERY_COUNT
        or heldout.query_sequence_sha256 != run_identity["query_sequence_sha256"]
    ):
        raise PhysicalBenchmarkInvariantError("physical trial heldout population drifted")

    bridge_record = raw.get("prefit_bridge")
    if not isinstance(bridge_record, Mapping):
        raise PhysicalBenchmarkInvariantError("physical trial PREFIT bridge is missing")
    prefit_bridge = load_prefit_bridge_artifact_v12(
        bridge_record,
        root=root,
        verify_hashes=True,
    )
    if (
        root not in prefit_bridge.path.parents
        or root not in prefit_bridge.manifest_path.parents
        or prefit_bridge.query_count != SAMPLE_QUERY_COUNT
        or dict(prefit_bridge.lineage) != lineage
        or prefit_bridge.prefit_manifest_sha256
        != expected_prefit_parity_manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantError(
            "physical trial PREFIT bridge lineage/population drifted"
        )
    base_lineage = {
        key: value
        for key, value in lineage.items()
        if key not in EXECUTION_LINEAGE_KEYS
    }
    validate_prefit_bridge_against_reference_v12(
        prefit_bridge,
        prefit_manifest_path=Path(
            expected_prefit_parity_manifest_path
        ).resolve(),
        expected_lineage=base_lineage,
    )

    model_records = raw.get("site_models")
    if not isinstance(model_records, Mapping):
        raise PhysicalBenchmarkInvariantError("physical trial site models are missing")
    models = load_fitted_site_model_records_v12(
        model_records,
        handle_loader=lambda _path, _outer: None,
        root=root,
    )
    model_sha = {str(outer): models[outer].sha256 for outer in ALLOWED_FOLDS}
    model_paths = [Path(models[outer].path).resolve() for outer in ALLOWED_FOLDS]
    if any(root not in model_path.parents for model_path in model_paths):
        raise PhysicalBenchmarkInvariantError("physical trial model escaped its root")

    output_records = raw.get("outputs")
    if not isinstance(output_records, Mapping) or set(output_records) != {
        "top500",
        "top100",
        "outcomes",
    }:
        raise PhysicalBenchmarkInvariantError("physical trial outputs are incomplete")
    output_sha: dict[str, str] = {}
    output_paths: dict[str, Path] = {}
    for name, record in output_records.items():
        if not isinstance(record, Mapping):
            raise PhysicalBenchmarkInvariantError("physical trial output record is invalid")
        artifact_path = _resolve_child_artifact_v12(
            root, record.get("path"), label=f"output/{name}"
        )
        digest = str(record.get("sha256") or "")
        if not _is_sha256(digest) or sha256_file(artifact_path) != digest:
            raise PhysicalBenchmarkInvariantError("physical trial output hash drifted")
        output_sha[name] = digest
        output_paths[name] = artifact_path
    top500_rows = _validate_ranked_output_v12(
        output_paths["top500"], expected_sequence=expected_sequence, cap=500
    )
    top100_rows = _validate_ranked_output_v12(
        output_paths["top100"], expected_sequence=expected_sequence, cap=100
    )
    outcomes_table = pq.read_table(
        output_paths["outcomes"], columns=["query_id", "fold"]
    )
    outcome_sequence = tuple(
        zip(
            (str(value) for value in outcomes_table["query_id"].to_pylist()),
            (int(value.as_py()) for value in outcomes_table["fold"]),
            strict=True,
        )
    )
    if (
        outcome_sequence != expected_sequence
        or int(output_records["top500"].get("rows", -1)) != top500_rows
        or int(output_records["top100"].get("rows", -1)) != top100_rows
        or int(output_records["outcomes"].get("rows", -1)) != SAMPLE_QUERY_COUNT
    ):
        raise PhysicalBenchmarkInvariantError("physical trial output cardinality drifted")

    expected_core_digests = {
        "train_matrix": train_sha,
        "group_sequence": group_sha,
        "heldout_spool": heldout.sha256,
        "model_by_outer": model_sha,
        "prerank500": output_sha["top500"],
        "top100": output_sha["top100"],
        "outcomes": output_sha["outcomes"],
        "prefit_bridge": prefit_bridge.sha256,
    }
    if dict(digests_raw) != expected_core_digests:
        raise PhysicalBenchmarkInvariantError(
            "physical trial functional-plus-PREFIT digest surface drifted"
        )
    gates = raw.get("gates")
    if (
        not isinstance(gates, Mapping)
        or gates.get("denominator")
        != "ALL_USABLE_HUMAN_SIRET_LABELS_FOLDS_2_3_4"
        or gates.get("decision") != "NO_READY_DECISION"
        or gates.get("gate_eligible") is not False
        or gates.get("postfit_parity_gate") != "PENDING"
        or gates.get("latency_gate") != "NOT_EVALUATED"
        or (gates.get("exact_recall_at_100") or {}).get("total")
        != SAMPLE_QUERY_COUNT
    ):
        raise PhysicalBenchmarkInvariantError("physical trial safety gates drifted")

    io_bytes = raw.get("io_bytes")
    temporary = raw.get("temporary_budget")
    actual_materialized = sum(
        child.stat().st_size
        for child in root.rglob("*")
        if child.is_file() and child.resolve() != path
    )
    if (
        not isinstance(io_bytes, Mapping)
        or int(io_bytes.get("total_materialized", -1)) != actual_materialized
        or not isinstance(temporary, Mapping)
        or temporary.get("pass") is not True
        or int(temporary.get("observed_materialized_bytes", -1))
        != actual_materialized
        or int(temporary.get("maximum_bytes", 0)) < actual_materialized
    ):
        raise PhysicalBenchmarkInvariantError("physical trial materialized I/O drifted")

    families = {
        "worker_train_matrices": sum(
            matrix.path.stat().st_size
            for build in builds
            for matrix in build.train_matrices.values()
        ),
        "worker_heldout_spools": sum(
            build.heldout_master.size_bytes for build in builds
        ),
        "merged_train_matrices": sum(value.stat().st_size for value in train_paths),
        "merged_heldout_spool": heldout.size_bytes,
        "site_models": sum(value.stat().st_size for value in model_paths),
        "worker_replay_outputs": sum(
            record.path.stat().st_size
            for replay in replays
            for record in replay.outputs.values()
        ),
        "merged_replay_outputs": sum(value.stat().st_size for value in output_paths.values()),
    }
    return PhysicalTrialProductV12(
        worker_count=expected_worker_count,
        manifest_path=path,
        query_count=SAMPLE_QUERY_COUNT,
        folds=ALLOWED_FOLDS,
        lineage=lineage,
        digests={
            "prefit_bridge": prefit_bridge.sha256,
            "train_matrices": _sha256_json(train_sha),
            "query_groups": _sha256_json(group_sha),
            "heldout_spool": heldout.sha256,
            "miniature_models": _sha256_json(model_sha),
            "prerank500": output_sha["top500"],
            "top100": output_sha["top100"],
            "outcomes": output_sha["outcomes"],
        },
        materialized_io_bytes=actual_materialized,
        site_model_bytes=families["site_models"],
        materialized_families=families,
    )


@dataclass(frozen=True)
class PhysicalTrialEvidenceV12:
    product: PhysicalTrialProductV12
    resources: PhysicalResourceMetricsV12

    def manifest_record(self, root: Path) -> Mapping[str, Any]:
        manifest_path = Path(self.product.manifest_path).resolve()
        root = Path(root).resolve()
        if root not in manifest_path.parents:
            raise PhysicalBenchmarkInvariantError("trial manifest escaped benchmark root")
        return {
            "worker_count": self.product.worker_count,
            "query_count": self.product.query_count,
            "folds": list(self.product.folds),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "lineage": dict(self.product.lineage),
            "digests": _validate_digests(self.product.digests),
            "materialized_io_bytes": self.product.materialized_io_bytes,
            "site_model_bytes": self.product.site_model_bytes,
            "materialized_families": {
                name: int(self.product.materialized_families[name])
                for name in MATERIALIZED_FAMILIES
            },
            "resources": asdict(self.resources),
            "artifact_manifest": {
                "path": str(manifest_path.relative_to(root)),
                "sha256": sha256_file(manifest_path),
            },
        }


@dataclass(frozen=True)
class CandidateDecisionV12:
    worker_count: int
    parity_exact: bool
    speedup: float
    io_ratio: float
    io_measurement_source: str
    ram_pass: bool
    swap_pass: bool
    speed_pass: bool
    io_pass: bool
    eligible: bool


@dataclass(frozen=True)
class ShardedOOFAuthorizationV12:
    worker_count: int
    manifest_path: Path
    manifest_sha256: str
    authorization_path: Path
    authorization_sha256: str
    prefit_parity_manifest_sha256: str
    sharded_execution_plan_sha256: str
    lineage: Mapping[str, str]


class TrialRunnerV12(Protocol):
    def __call__(self, worker_count: int, output_dir: Path) -> PhysicalTrialProductV12: ...


class TrialMeasurerV12(Protocol):
    def __call__(
        self, callback: Callable[[], PhysicalTrialProductV12]
    ) -> tuple[PhysicalTrialProductV12, PhysicalResourceMetricsV12]: ...


def _busy_cpu_seconds(value: Any) -> float:
    ignored = {"idle", "iowait"}
    return float(
        sum(
            float(field)
            for name, field in value._asdict().items()
            if name not in ignored
        )
    )


def _disk_bytes(value: Any) -> tuple[int, int]:
    if value is None:
        return 0, 0
    return int(getattr(value, "read_bytes", 0)), int(
        getattr(value, "write_bytes", 0)
    )


def _vm_stat_swap_observation() -> Mapping[str, Any]:
    """Read the raw macOS Swapouts page counter and its actual page size."""

    try:
        completed = subprocess.run(
            ["/usr/bin/vm_stat"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PhysicalBenchmarkInvariantError(
            "macOS swap-out counter is unavailable"
        ) from error
    page_size_match = re.search(
        r"page size of\s+(\d+)\s+bytes", completed.stdout, re.IGNORECASE
    )
    swapouts_match = re.search(
        r"^Swapouts:\s+(\d+)\.\s*$", completed.stdout, re.MULTILINE
    )
    if page_size_match is None or swapouts_match is None:
        raise PhysicalBenchmarkInvariantError(
            "macOS swap-out counter could not be parsed"
        )
    return {
        "source": "vm_stat.Swapouts",
        "page_size_bytes": int(page_size_match.group(1)),
        "swapouts_pages": int(swapouts_match.group(1)),
    }


def _swap_out_bytes() -> int:
    """Return cumulative swap-out bytes, never an assumed zero."""

    try:
        observation = _vm_stat_swap_observation()
        return int(observation["page_size_bytes"]) * int(
            observation["swapouts_pages"]
        )
    except PhysicalBenchmarkInvariantError:
        try:
            value = psutil.swap_memory()
        except OSError as error:
            raise PhysicalBenchmarkInvariantError(
                "swap-out counter is unavailable"
            ) from error
        if not hasattr(value, "sout"):
            raise PhysicalBenchmarkInvariantError("psutil lacks a swap-out counter")
        return int(value.sout)


def _rusage_cpu(value: resource.struct_rusage) -> float:
    return float(value.ru_utime + value.ru_stime)


class _DarwinRUsageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *(
            (name, ctypes.c_uint64)
            for name in (
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
                "ri_child_user_time",
                "ri_child_system_time",
                "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups",
                "ri_child_pageins",
                "ri_child_elapsed_abstime",
                "ri_diskio_bytesread",
                "ri_diskio_byteswritten",
            )
        ),
    ]


_DARWIN_PROC_PID_RUSAGE: Any = None


def _darwin_process_io(pid: int) -> tuple[int, int] | None:
    global _DARWIN_PROC_PID_RUSAGE
    if sys.platform != "darwin":
        return None
    try:
        if _DARWIN_PROC_PID_RUSAGE is None:
            library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            function = library.proc_pid_rusage
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(_DarwinRUsageInfoV2),
            ]
            function.restype = ctypes.c_int
            # Keep the library reachable through the bound function object.
            function._sireto_library = library
            _DARWIN_PROC_PID_RUSAGE = function
        value = _DarwinRUsageInfoV2()
        if _DARWIN_PROC_PID_RUSAGE(int(pid), 2, ctypes.byref(value)) != 0:
            return None
        return int(value.ri_diskio_bytesread), int(value.ri_diskio_byteswritten)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


class _ProcessTreeSampler:
    """Sample aggregate job RSS and per-process I/O without payload IPC."""

    def __init__(self, *, interval_seconds: float) -> None:
        self.interval_seconds = float(interval_seconds)
        if not 0.01 <= self.interval_seconds <= 1.0:
            raise PhysicalBenchmarkInvariantError("resource sample interval is unsafe")
        self.root = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.peak_rss = 0
        self.peak_machine_used = 0
        self.minimum_machine_available = psutil.virtual_memory().available
        self._io_baseline: dict[tuple[int, float], tuple[int, int]] = {}
        self._io_maximum: dict[tuple[int, float], tuple[int, int]] = {}
        self._root_key = (self.root.pid, self.root.create_time())

    @staticmethod
    def _process_io(process: psutil.Process) -> tuple[int, int]:
        try:
            counters = process.io_counters()
            return int(getattr(counters, "read_bytes", 0)), int(
                getattr(counters, "write_bytes", 0)
            )
        except (AttributeError, psutil.Error, PermissionError):
            return _darwin_process_io(process.pid) or (0, 0)

    def _sample(self) -> None:
        try:
            processes = [self.root, *self.root.children(recursive=True)]
        except psutil.Error:
            processes = [self.root]
        rss = 0
        for process in processes:
            try:
                key = (process.pid, process.create_time())
                rss += int(process.memory_info().rss)
            except psutil.Error:
                continue
            current = self._process_io(process)
            if key not in self._io_baseline:
                # The coordinator predates the measurement; children were
                # created by the trial and their counters start at zero.
                self._io_baseline[key] = current if key == self._root_key else (0, 0)
            previous = self._io_maximum.get(key, (0, 0))
            self._io_maximum[key] = (
                max(previous[0], current[0]),
                max(previous[1], current[1]),
            )
        memory = psutil.virtual_memory()
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_machine_used = max(
            self.peak_machine_used, int(memory.used)
        )
        self.minimum_machine_available = min(
            self.minimum_machine_available, int(memory.available)
        )

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self.thread.start()

    def stop(self) -> tuple[int, int]:
        self._sample()
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        read_bytes = write_bytes = 0
        for key, maximum in self._io_maximum.items():
            baseline = self._io_baseline.get(key, (0, 0))
            read_bytes += max(0, maximum[0] - baseline[0])
            write_bytes += max(0, maximum[1] - baseline[1])
        return read_bytes, write_bytes


def measure_physical_trial_v12(
    callback: Callable[[], PhysicalTrialProductV12],
    *,
    sample_interval_seconds: float = 0.05,
) -> tuple[PhysicalTrialProductV12, PhysicalResourceMetricsV12]:
    """Measure one complete invocation of the unique sharded OOF core."""

    sampler = _ProcessTreeSampler(interval_seconds=sample_interval_seconds)
    # Probe before CPU/wall baselines so the macOS ``vm_stat`` fallback is
    # instrumentation rather than benchmark work.
    swap_before = _swap_out_bytes()
    self_before = resource.getrusage(resource.RUSAGE_SELF)
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    machine_cpu_before = psutil.cpu_times()
    machine_disk_before = psutil.disk_io_counters()
    wall_start = time.perf_counter()
    sampler.start()
    try:
        product = callback()
    finally:
        process_read, process_write = sampler.stop()
    wall_seconds = time.perf_counter() - wall_start
    swap_after = _swap_out_bytes()
    self_after = resource.getrusage(resource.RUSAGE_SELF)
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    machine_cpu_after = psutil.cpu_times()
    machine_disk_after = psutil.disk_io_counters()
    disk_before = _disk_bytes(machine_disk_before)
    disk_after = _disk_bytes(machine_disk_after)
    metrics = PhysicalResourceMetricsV12(
        wall_seconds=wall_seconds,
        coordinator_cpu_seconds=max(
            0.0, _rusage_cpu(self_after) - _rusage_cpu(self_before)
        ),
        children_cpu_seconds=max(
            0.0, _rusage_cpu(children_after) - _rusage_cpu(children_before)
        ),
        machine_busy_cpu_seconds=max(
            0.0, _busy_cpu_seconds(machine_cpu_after) - _busy_cpu_seconds(machine_cpu_before)
        ),
        peak_process_tree_rss_bytes=sampler.peak_rss,
        peak_machine_used_memory_bytes=sampler.peak_machine_used,
        minimum_machine_available_memory_bytes=sampler.minimum_machine_available,
        swap_out_delta_bytes=max(0, swap_after - swap_before),
        process_read_bytes=process_read,
        process_write_bytes=process_write,
        machine_read_bytes=max(0, disk_after[0] - disk_before[0]),
        machine_write_bytes=max(0, disk_after[1] - disk_before[1]),
        sample_interval_seconds=sample_interval_seconds,
    )
    return product, metrics


def compare_candidate_v12(
    reference: PhysicalTrialEvidenceV12,
    candidate: PhysicalTrialEvidenceV12,
    *,
    thresholds: PhysicalBenchmarkThresholdsV12,
) -> CandidateDecisionV12:
    if reference.product.worker_count != 1 or candidate.product.worker_count not in (2, 3):
        raise PhysicalBenchmarkInvariantError("invalid reference/candidate worker counts")
    if dict(reference.product.lineage) != dict(candidate.product.lineage):
        raise PhysicalBenchmarkInvariantError("physical trial lineage differs by worker count")
    parity_exact = _validate_digests(reference.product.digests) == _validate_digests(
        candidate.product.digests
    )
    speedup = reference.resources.wall_seconds / candidate.resources.wall_seconds
    reference_process_io = reference.resources.decision_io_bytes
    candidate_process_io = candidate.resources.decision_io_bytes
    if reference_process_io > 0 and candidate_process_io > 0:
        reference_io = reference_process_io
        candidate_io = candidate_process_io
        io_source = "PROCESS_TREE_DISK_BYTES"
    else:
        # Exact core materialization is a conservative, reproducible fallback
        # on platforms that expose no per-process byte counters.  OS machine
        # disk bytes remain published separately for audit.
        reference_io = reference.product.materialized_io_bytes
        candidate_io = candidate.product.materialized_io_bytes
        io_source = "CORE_MATERIALIZED_ARTIFACT_BYTES"
    if reference_io <= 0 or candidate_io <= 0:
        raise PhysicalBenchmarkInvariantError("physical I/O counters are unavailable")
    io_ratio = candidate_io / reference_io
    minimum_speedup = (
        thresholds.minimum_speedup_w3
        if candidate.product.worker_count == 3
        else thresholds.minimum_speedup_w2
    )
    ram_pass = (
        candidate.resources.peak_machine_used_memory_bytes
        <= thresholds.maximum_machine_used_memory_bytes
    )
    swap_pass = (
        candidate.resources.swap_out_delta_bytes
        <= thresholds.maximum_swap_out_bytes
    )
    speed_pass = speedup >= minimum_speedup
    io_pass = io_ratio <= thresholds.maximum_io_ratio
    return CandidateDecisionV12(
        worker_count=candidate.product.worker_count,
        parity_exact=parity_exact,
        speedup=speedup,
        io_ratio=io_ratio,
        io_measurement_source=io_source,
        ram_pass=ram_pass,
        swap_pass=swap_pass,
        speed_pass=speed_pass,
        io_pass=io_pass,
        eligible=bool(parity_exact and ram_pass and swap_pass and speed_pass and io_pass),
    )


def _trial(
    runner: TrialRunnerV12,
    measurer: TrialMeasurerV12,
    *,
    worker_count: int,
    output_dir: Path,
    expected_lineage: Mapping[str, str],
) -> PhysicalTrialEvidenceV12:
    product, resources = measurer(
        lambda: runner(worker_count, output_dir)
    )
    if product.worker_count != worker_count:
        raise PhysicalBenchmarkInvariantError("trial runner returned the wrong worker count")
    if dict(product.lineage) != dict(expected_lineage):
        raise PhysicalBenchmarkInvariantError("trial runner lineage differs from PREFIT")
    return PhysicalTrialEvidenceV12(product, resources)


_FROZEN_SHARDED_AUTHORITY_PATHS_V12 = {
    "scientific_config": "config/retrieval_all_human_v12.json",
    "scientific_contract": "docs/retrieval_all_human_v12_contract.md",
    "fold_fused_executor_config": (
        "config/retrieval_all_human_v12_fold_fused_executor.json"
    ),
    "fold_fused_execution_addendum": (
        "docs/retrieval_all_human_v12_fold_fused_execution_addendum.md"
    ),
    "sharded_execution_addendum": (
        "docs/retrieval_all_human_v12_sharded_execution_addendum.md"
    ),
}

_FROZEN_SHARDS_V12 = [
    {
        "shard_id": "fold2",
        "fold": 2,
        "query_ordinal_start_inclusive": 0,
        "query_ordinal_stop_exclusive": 7767,
        "query_count": 7767,
        "source_row_group_start_inclusive": 0,
        "source_row_group_stop_exclusive": 210,
        "source_row_group_count": 210,
        "candidate_row_start_inclusive": 0,
        "candidate_row_stop_exclusive": 13750106,
        "candidate_row_count": 13750106,
    },
    {
        "shard_id": "fold3",
        "fold": 3,
        "query_ordinal_start_inclusive": 7767,
        "query_ordinal_stop_exclusive": 15734,
        "query_count": 7967,
        "source_row_group_start_inclusive": 210,
        "source_row_group_stop_exclusive": 423,
        "source_row_group_count": 213,
        "candidate_row_start_inclusive": 13750106,
        "candidate_row_stop_exclusive": 27705638,
        "candidate_row_count": 13955532,
    },
    {
        "shard_id": "fold4",
        "fold": 4,
        "query_ordinal_start_inclusive": 15734,
        "query_ordinal_stop_exclusive": 23571,
        "query_count": 7837,
        "source_row_group_start_inclusive": 423,
        "source_row_group_stop_exclusive": 635,
        "source_row_group_count": 212,
        "candidate_row_start_inclusive": 27705638,
        "candidate_row_stop_exclusive": 41590480,
        "candidate_row_count": 13884842,
    },
]

_FROZEN_PROCESS_POLICY_V12 = {
    "start_method": "spawn",
    "shard_count": 3,
    "preferred_maximum_concurrent_workers": 3,
    "fallback_maximum_concurrent_workers": 2,
    "decision_phase": "ONCE_BEFORE_FIRST_SHARD_OPEN",
    "preferred_w3_requires_all": {
        "machine_used_ram_metric": "psutil.virtual_memory().used",
        "machine_used_ram_maximum_bytes": 14 * GIB,
        "machine_used_ram_maximum_gib": 14,
        "swapout_metric": "vm_stat.Swapouts",
        "swapout_observation_seconds": 5,
        "swapout_delta_pages_required": 0,
    },
    "measurement_failure_policy": "FALLBACK_W2",
    "runtime_change_after_first_shard_open": "FORBIDDEN",
    "automatic_retry_with_different_worker_count": False,
    "worker_assignment_may_change_artifact_order": False,
    "parent_prediction_threads_per_worker": 4,
    "site_prediction_threads_per_worker": 4,
    "prediction_thread_cap_activation": "ONLY_AFTER_W1_W3_FLOAT32_BIT_PARITY",
    "site_fit_parallelism": 1,
    "site_fit_order": [2, 3, 4],
    "site_fit_uses_frozen_n_jobs_zero": True,
}

_FROZEN_QUERY_COMPILATION_V12 = {
    "scene_compilations_per_query": 1,
    "outer_contexts_per_query": 3,
    "training_contexts_per_query": 2,
    "heldout_contexts_per_query": 1,
    "official_lookup_mode": "ONE_BATCHED_PARENT_LOOKUP_THEN_THREE_CONTEXT_MASKS",
    "heldout_spooled_before_label_access": True,
    "full_master_statistics_before_fit_sampling": True,
    "fit_sampling_maximum_rows_per_training_context": 128,
    "positive_injection": False,
    "label_blind_worker_cache_only": True,
}

_FROZEN_TRAIN_MERGE_V12 = {
    "operator": "PHYSICAL_GROUP_CONCATENATION_NO_SORT",
    "outer2": [3, 4],
    "outer3": [2, 4],
    "outer4": [2, 3],
    "preserve_scene_ir_query_order": True,
    "preserve_intra_query_candidate_order": True,
    "preserve_float32_feature_bits": True,
    "query_groups_must_remain_contiguous": True,
    "distributed_or_incremental_fit": False,
}

_FROZEN_HELDOUT_REPLAY_V12 = {
    "parallel_by_scene_fold": True,
    "fold_to_site_model_outer": {"2": 2, "3": 3, "4": 4},
    "scene_ir_reopened_after_fit": False,
    "official_sidecars_reopened_after_fit": False,
    "parent_models_reopened_after_fit": False,
    "validation_candidate_sampling": False,
    "full_master_scored": True,
}

_FROZEN_OUTPUT_MERGE_V12 = {
    "fold_order": [2, 3, 4],
    "preserve_shard_query_order": True,
    "global_query_count": 23571,
    "global_gate_reduction": "RECOMPUTE_FROM_ALL_ORDERED_QUERY_OUTCOMES",
    "per_shard_metric_average_forbidden": True,
    "tie_break": "SCORE_DESC_THEN_SIRET_ASC",
}

_FROZEN_PARITY_CHECKS_V12 = [
    "query_population_and_global_order_byte_identical",
    "parent_scores_float32_bit_identical",
    "master_rows_and_ranks_byte_identical",
    "full_master_statistics_byte_identical",
    "fit_row_ids_and_order_byte_identical",
    "train_feature_values_float32_bit_identical",
    "train_group_order_byte_identical",
    "heldout_spool_and_replay_order_byte_identical",
    "sample_site_model_sha256_identical",
    "site_scores_bit_identical",
    "prerank500_byte_identical",
    "top100_and_reasons_byte_identical",
    "outcomes_and_global_aggregates_byte_identical",
]

_FROZEN_PARITY_NEGATIVE_TESTS_V12 = [
    "missing_shard_rejected",
    "duplicate_shard_rejected",
    "overlapping_shard_rejected",
    "train_merge_inversion_rejected",
    "wrong_outer_model_rejected",
    "fold0_or_fold1_rejected",
    "input_or_model_hash_drift_rejected",
    "label_access_before_heldout_seal_rejected",
    "positive_injection_rejected",
    "output_order_drift_rejected",
    "per_shard_metric_average_rejected",
]

_FROZEN_SCIENTIFIC_GATES_V12 = {
    "master_recall_at_2000": 0.995,
    "master_decision_blocking": False,
    "prerank_recall_at_500": 0.994,
    "exact_recall_at_100": 0.992,
    "exact_recall_at_100_wilson95_lower": 0.99,
    "minimum_each_outer_fold_exact_recall_at_100": 0.988,
    "maximum_master_candidates": 2000,
    "maximum_fit_candidates": 128,
    "maximum_prerank_candidates": 500,
    "maximum_product_candidates": 100,
    "warm_latency_p95_seconds": 1.0,
    "warm_latency_p99_seconds": 2.0,
}

_FROZEN_RESOURCES_V12 = {
    "scratch_root": "/Volumes/CATNAT_DATA",
    "system_disk_bulk_artifacts_allowed": False,
    "maximum_temporary_bytes": 20 * GIB,
    "maximum_temporary_gib": 20,
    "projected_national_heldout_spool_bytes": 5481806143,
    "shard_artifacts_content_hashed": True,
    "partial_manifest_can_claim_gate": False,
}

_FROZEN_FINAL_DECISION_POLICY_V12 = {
    "oof_manifest_before_postfit": "EXECUTED_NO_READY_CLAIM",
    "postfit_must_bind_exact_three_site_model_hashes": True,
    "postfit_refit_allowed": False,
    "runtime_profile_required_after_postfit": True,
    "fold1_open_requires_explicit_one_shot_authorization": True,
}


def _validate_sharded_execution_plan_v12(
    path: Path,
    sha256: str,
    *,
    lineage: Mapping[str, str],
    prefit_parity_manifest_sha256: str,
    thresholds: PhysicalBenchmarkThresholdsV12,
) -> tuple[Path, Mapping[str, Any]]:
    resolved = Path(path).resolve()
    if (
        not resolved.is_file()
        or not _is_sha256(sha256)
        or sha256_file(resolved) != sha256
    ):
        raise PhysicalBenchmarkInvariantError("sharded execution plan hash drifted")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    expected_top_level_keys = {
        "schema_version",
        "status",
        "engine_id",
        "physical_plan_id",
        "logical_recipe_changes",
        "national_oof_allowed",
        "required_next_gate",
        "authorities",
        "input_lineage",
        "fold_policy",
        "shards",
        "process_policy",
        "query_compilation",
        "train_merge",
        "heldout_replay",
        "output_merge",
        "sharded_parity_gate",
        "scientific_gates_unchanged",
        "resources",
        "final_decision_policy",
    }
    if set(raw) != expected_top_level_keys or any(
        raw.get(key) != value
        for key, value in {
            "schema_version": "sireto-retrieval-v12-fold-sharded-execution-plan-1",
            "status": "PREREGISTERED_SHARDED_PARITY_PENDING",
            "engine_id": "FOLD_FUSED_CROSSFIT_COMPILER_V12_1",
            "physical_plan_id": "FOLD_CONTIGUOUS_SPAWN_W3_OR_W2_V12_1",
            "logical_recipe_changes": False,
            "national_oof_allowed": False,
            "required_next_gate": "SEALED_W1_W3_PARITY_512_PASS",
        }.items()
    ):
        raise PhysicalBenchmarkInvariantError("sharded execution plan is not pending parity")

    authorities = raw.get("authorities") or {}
    if set(authorities) != set(_FROZEN_SHARDED_AUTHORITY_PATHS_V12) or any(
        set(authorities.get(name) or {}) != {"path", "sha256"}
        or (authorities.get(name) or {}).get("path") != expected_path
        or not _is_sha256((authorities.get(name) or {}).get("sha256"))
        for name, expected_path in _FROZEN_SHARDED_AUTHORITY_PATHS_V12.items()
    ):
        raise PhysicalBenchmarkInvariantError("sharded execution authorities drifted")

    fold_policy = raw.get("fold_policy") or {}
    gate = raw.get("sharded_parity_gate") or {}
    if (
        fold_policy
        != {
            "opened_folds": list(ALLOWED_FOLDS),
            "fold0_opened": False,
            "fold1_opened": False,
            "fold0_diagnostic_only": True,
            "fold1_one_shot_authorization_required": True,
        }
        or raw.get("shards") != _FROZEN_SHARDS_V12
        or raw.get("process_policy") != _FROZEN_PROCESS_POLICY_V12
        or raw.get("query_compilation") != _FROZEN_QUERY_COMPILATION_V12
        or raw.get("train_merge") != _FROZEN_TRAIN_MERGE_V12
        or raw.get("heldout_replay") != _FROZEN_HELDOUT_REPLAY_V12
        or raw.get("output_merge") != _FROZEN_OUTPUT_MERGE_V12
        or raw.get("scientific_gates_unchanged")
        != _FROZEN_SCIENTIFIC_GATES_V12
        or raw.get("resources") != _FROZEN_RESOURCES_V12
        or raw.get("final_decision_policy")
        != _FROZEN_FINAL_DECISION_POLICY_V12
        or thresholds.maximum_machine_used_memory_bytes
        != _FROZEN_PROCESS_POLICY_V12["preferred_w3_requires_all"][
            "machine_used_ram_maximum_bytes"
        ]
    ):
        raise PhysicalBenchmarkInvariantError("sharded execution policy drifted")

    input_lineage = raw.get("input_lineage") or {}
    if set(input_lineage) != {
        "scene_ir",
        "labels",
        "official_sidecars",
        "parent_model_bundle",
        "prefit_parity",
    }:
        raise PhysicalBenchmarkInvariantError("sharded input lineage schema drifted")
    scene = input_lineage.get("scene_ir") or {}
    labels = input_lineage.get("labels") or {}
    parent_bundle = input_lineage.get("parent_model_bundle") or {}
    if (
        scene.get("query_count") != 23571
        or scene.get("candidate_rows") != 41590480
        or scene.get("source_row_groups") != 635
        or labels.get("runtime_query_sample_rows") != SAMPLE_QUERY_COUNT
        or parent_bundle.get("model_count") != 6
    ):
        raise PhysicalBenchmarkInvariantError("sharded national resource plan drifted")

    expected_gate = {
        "status": "NOT_RUN",
        "sample_query_count": SAMPLE_QUERY_COUNT,
        "sample_sha256": labels.get("runtime_query_sample_sha256"),
        "reference_worker_count": 1,
        "candidate_worker_count": 3,
        "real_required_comparison": "W1_EQUALS_W3_BYTE_OR_BIT_EXACT",
        "fixture_required_worker_counts": [1, 2, 3],
        "quality_metrics_may_be_used_for_tuning": False,
        "required_checks": _FROZEN_PARITY_CHECKS_V12,
        "required_negative_tests": _FROZEN_PARITY_NEGATIVE_TESTS_V12,
        "failure_decision": "STOP_SHARDED_PARITY",
        "national_oof_on_failure": False,
    }
    if gate != expected_gate:
        raise PhysicalBenchmarkInvariantError("sharded parity gate drifted")

    expected = {
        "scene_ir_sha256": scene.get("build_id"),
        "scene_ir_manifest_sha256": scene.get("manifest_sha256"),
        "source_candidate_sha256": scene.get("source_candidate_sha256"),
        "source_manifest_sha256": scene.get("source_manifest_sha256"),
        "label_ledger_sha256": labels.get("ledger_sha256"),
        "label_ledger_manifest_sha256": labels.get("manifest_sha256"),
        "official_sidecar_manifest_sha256": (
            input_lineage.get("official_sidecars") or {}
        ).get("sidecar_manifest_sha256"),
        "mmap_sidecar_manifest_sha256": (
            input_lineage.get("official_sidecars") or {}
        ).get("mmap_manifest_sha256"),
        "reader_parity_evidence_sha256": (
            input_lineage.get("official_sidecars") or {}
        ).get("reader_parity_manifest_sha256"),
        "parent_model_bundle_sha256": (
            input_lineage.get("parent_model_bundle") or {}
        ).get("manifest_sha256"),
        "query_sample_sha256": labels.get("runtime_query_sample_sha256"),
    }
    expected.update(
        {
            "scientific_config_sha256": (
                authorities.get("scientific_config") or {}
            ).get("sha256"),
            "executor_config_sha256": (
                authorities.get("fold_fused_executor_config") or {}
            ).get("sha256"),
        }
    )
    if any(lineage.get(key) != value for key, value in expected.items()):
        raise PhysicalBenchmarkInvariantError("sharded plan/input lineage drifted")
    prefit = input_lineage.get("prefit_parity") or {}
    if (
        prefit.get("manifest_sha256") != prefit_parity_manifest_sha256
        or prefit.get("phase") != "PREFIT_NON_CIRCULAR"
        or prefit.get("status") != "PASS"
        or prefit.get("query_count") != SAMPLE_QUERY_COUNT
        or prefit.get("fold0_opened") is not False
        or prefit.get("fold1_opened") is not False
        or prefit.get("positive_injection") is not False
    ):
        raise PhysicalBenchmarkInvariantError("sharded plan/PREFIT binding drifted")
    return resolved, raw


def run_physical_worker_benchmark_v12(
    *,
    runner: TrialRunnerV12,
    output_dir: Path,
    expected_lineage: Mapping[str, str],
    prefit_parity_manifest_path: Path,
    prefit_parity_manifest_sha256: str,
    sharded_execution_plan_path: Path,
    sharded_execution_plan_sha256: str,
    measurer: TrialMeasurerV12 = measure_physical_trial_v12,
    thresholds: PhysicalBenchmarkThresholdsV12 = PhysicalBenchmarkThresholdsV12(),
) -> Path:
    """Run W3 then conservative warm-cache W1, with conditional W2 fallback."""

    lineage = _require_lineage(expected_lineage)
    prefit_path = Path(prefit_parity_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not prefit_path.is_file() or sha256_file(prefit_path) != prefit_parity_manifest_sha256:
        raise PhysicalBenchmarkInvariantError("PREFIT parity manifest hash drifted")
    if not _is_sha256(prefit_parity_manifest_sha256):
        raise PhysicalBenchmarkInvariantError("PREFIT parity SHA-256 is invalid")
    prefit = json.loads(prefit_path.read_text(encoding="utf-8"))
    prefit_lineage = dict(prefit.get("lineage") or {})
    execution_only = set(lineage) - set(prefit_lineage)
    if any(
        prefit.get(key) != value
        for key, value in {
            "status": "PASS",
            "phase": "PREFIT_NON_CIRCULAR",
            "query_count": SAMPLE_QUERY_COUNT,
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
        }.items()
    ) or any(lineage.get(key) != value for key, value in prefit_lineage.items()) or (
        execution_only not in (set(), set(EXECUTION_LINEAGE_KEYS))
    ):
        raise PhysicalBenchmarkInvariantError("PREFIT parity is not benchmark eligible")
    plan_path, plan = _validate_sharded_execution_plan_v12(
        sharded_execution_plan_path,
        sharded_execution_plan_sha256,
        lineage=lineage,
        prefit_parity_manifest_sha256=prefit_parity_manifest_sha256,
        thresholds=thresholds,
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    # W3 runs first; W1 therefore receives any warm-cache advantage.  This
    # makes the speedup threshold conservative for the parallel candidate.
    trials: dict[int, PhysicalTrialEvidenceV12] = {}
    trials[3] = _trial(
        runner,
        measurer,
        worker_count=3,
        output_dir=output_dir / "trial_w3",
        expected_lineage=lineage,
    )
    trials[1] = _trial(
        runner,
        measurer,
        worker_count=1,
        output_dir=output_dir / "trial_w1",
        expected_lineage=lineage,
    )
    w3 = compare_candidate_v12(trials[1], trials[3], thresholds=thresholds)
    if not w3.parity_exact:
        raise PhysicalBenchmarkInvariantError(
            "W3 diverged from W1; worker fallback is forbidden after parity failure"
        )

    decisions: dict[int, CandidateDecisionV12] = {3: w3}
    selected = 3 if w3.eligible else 0
    if selected == 0:
        trials[2] = _trial(
            runner,
            measurer,
            worker_count=2,
            output_dir=output_dir / "trial_w2",
            expected_lineage=lineage,
        )
        w2 = compare_candidate_v12(trials[1], trials[2], thresholds=thresholds)
        decisions[2] = w2
        if not w2.parity_exact:
            raise PhysicalBenchmarkInvariantError(
                "W2 diverged from W1; no parallel worker count is authorized"
            )
        if w2.eligible:
            selected = 2

    national_query_count = int(plan["input_lineage"]["scene_ir"]["query_count"])
    maximum_temporary_bytes = int(plan["resources"]["maximum_temporary_bytes"])
    reference_product = trials[1].product
    projected_families = {
        name: (
            int(reference_product.materialized_families[name])
            if name == "site_models"
            else int(
                math.ceil(
                    int(reference_product.materialized_families[name])
                    * national_query_count
                    / SAMPLE_QUERY_COUNT
                )
            )
        )
        for name in MATERIALIZED_FAMILIES
    }
    sample_overhead = reference_product.materialized_io_bytes - sum(
        int(value) for value in reference_product.materialized_families.values()
    )
    projected_overhead = int(
        math.ceil(sample_overhead * national_query_count / SAMPLE_QUERY_COUNT)
    )
    projected_national_bytes = sum(projected_families.values()) + projected_overhead
    storage_pass = projected_national_bytes <= maximum_temporary_bytes

    records = {
        str(worker_count): evidence.manifest_record(output_dir)
        for worker_count, evidence in sorted(trials.items())
    }
    decision_records = {
        str(worker_count): asdict(decision)
        for worker_count, decision in sorted(decisions.items())
    }
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": "AUTHORIZED",
        "selected_worker_count": selected,
        "benchmark_manifest_sha256": None,
        "prefit_parity_manifest_sha256": prefit_parity_manifest_sha256,
        "sharded_execution_plan_sha256": sharded_execution_plan_sha256,
        "lineage": lineage,
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "PASS"
            if storage_pass and selected in (2, 3)
            else (
                "STOP_RESOURCE_POLICY"
                if not storage_pass
                else "STOP_NO_PARALLEL_WORKER_AUTHORIZED"
            )
        ),
        "run_kind": "PHYSICAL_W3_THEN_W1_WITH_CONDITIONAL_W2",
        "query_count": SAMPLE_QUERY_COUNT,
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "lineage": lineage,
        "prefit_parity": {
            "path": str(prefit_path),
            "sha256": prefit_parity_manifest_sha256,
            "status": "PASS",
        },
        "sharded_execution_plan": {
            "path": str(plan_path),
            "sha256": sharded_execution_plan_sha256,
            "input_status": "PREREGISTERED_SHARDED_PARITY_PENDING",
        },
        "thresholds": asdict(thresholds),
        "execution_order": [3, 1, *([2] if 2 in trials else [])],
        "trials": records,
        "candidate_decisions": decision_records,
        "selected_worker_count": selected,
        "selection": f"W{selected}" if selected in (2, 3) else "NONE",
        "all_selected_digests_exact_to_w1": True,
        "national_storage_projection": {
            "method": "W1_SAMPLE_LINEAR_EXCEPT_SITE_MODELS_CONSTANT",
            "sample_query_count": SAMPLE_QUERY_COUNT,
            "national_query_count": national_query_count,
            "sample_total_materialized_bytes": reference_product.materialized_io_bytes,
            "sample_site_model_bytes": reference_product.site_model_bytes,
            "sample_families": dict(reference_product.materialized_families),
            "sample_other_bytes": sample_overhead,
            "projected_families": projected_families,
            "projected_other_bytes": projected_overhead,
            "projected_national_bytes": projected_national_bytes,
            "maximum_temporary_bytes": maximum_temporary_bytes,
            "pass": storage_pass,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if storage_pass and selected in (2, 3):
        benchmark_sha = sha256_file(manifest_path)
        authorization["benchmark_manifest_sha256"] = benchmark_sha
        authorization_path = output_dir / "execution_authorization.json"
        authorization_path.write_text(
            json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest_path


def load_sharded_oof_authorization_v12(
    benchmark_manifest_path: Path,
    authorization_path: Path,
    *,
    expected_lineage: Mapping[str, str],
    expected_prefit_parity_manifest_sha256: str,
    expected_sharded_execution_plan_sha256: str,
) -> ShardedOOFAuthorizationV12:
    """Future OOF wiring: recompute the worker decision from sealed evidence."""

    benchmark_path = Path(benchmark_manifest_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    if (
        not benchmark_path.is_file()
        or not authorization_path.is_file()
        or benchmark_path.parent != authorization_path.parent
    ):
        raise PhysicalBenchmarkInvariantError("physical authorization files are incomplete")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    scientific_lineage = _require_lineage(expected_lineage)
    lineage = sharded_execution_lineage_v12(scientific_lineage)
    benchmark_sha = sha256_file(benchmark_path)
    authorization_sha = sha256_file(authorization_path)
    required_benchmark = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "query_count": SAMPLE_QUERY_COUNT,
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "lineage": lineage,
    }
    required_authorization = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": "AUTHORIZED",
        "benchmark_manifest_sha256": benchmark_sha,
        "prefit_parity_manifest_sha256": expected_prefit_parity_manifest_sha256,
        "sharded_execution_plan_sha256": expected_sharded_execution_plan_sha256,
        "lineage": lineage,
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
    }
    if any(benchmark.get(key) != value for key, value in required_benchmark.items()) or any(
        authorization.get(key) != value for key, value in required_authorization.items()
    ):
        raise PhysicalBenchmarkInvariantError("physical authorization lineage/status drifted")
    if (benchmark.get("prefit_parity") or {}).get("sha256") != (
        expected_prefit_parity_manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantError("physical authorization PREFIT lineage drifted")
    prefit_record = benchmark.get("prefit_parity") or {}
    prefit_path = Path(str(prefit_record.get("path") or "")).resolve()
    if (
        not prefit_path.is_file()
        or sha256_file(prefit_path) != expected_prefit_parity_manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantError("physical authorization PREFIT bytes drifted")
    _prefit, strict_prefit_sha = load_prefit_parity_evidence_v12(
        prefit_path,
        expected_lineage=scientific_lineage,
    )
    if strict_prefit_sha != expected_prefit_parity_manifest_sha256:
        raise PhysicalBenchmarkInvariantError("physical authorization PREFIT gate drifted")
    plan_record = benchmark.get("sharded_execution_plan") or {}
    plan_path = Path(str(plan_record.get("path") or "")).resolve()
    if (
        not _is_sha256(expected_sharded_execution_plan_sha256)
        or plan_record.get("sha256") != expected_sharded_execution_plan_sha256
        or not plan_path.is_file()
        or sha256_file(plan_path) != expected_sharded_execution_plan_sha256
    ):
        raise PhysicalBenchmarkInvariantError("physical authorization plan drifted")

    trials_raw = benchmark.get("trials")
    decisions_raw = benchmark.get("candidate_decisions")
    if not isinstance(trials_raw, Mapping) or not isinstance(decisions_raw, Mapping):
        raise PhysicalBenchmarkInvariantError("physical authorization lacks trial evidence")
    parsed_trials: dict[int, PhysicalTrialEvidenceV12] = {}
    for worker_text, record in trials_raw.items():
        worker = int(worker_text)
        if not isinstance(record, Mapping):
            raise PhysicalBenchmarkInvariantError("invalid physical trial record")
        artifact_record = record.get("artifact_manifest") or {}
        artifact_path = (benchmark_path.parent / str(artifact_record.get("path") or "")).resolve()
        if (
            benchmark_path.parent not in artifact_path.parents
            or not artifact_path.is_file()
            or artifact_record.get("sha256") != sha256_file(artifact_path)
        ):
            raise PhysicalBenchmarkInvariantError("physical trial artifact hash drifted")
        product = physical_trial_product_from_sharded_manifest_v12(
            artifact_path,
            expected_lineage=lineage,
            expected_worker_count=worker,
            expected_prefit_parity_manifest_path=prefit_path,
            expected_prefit_parity_manifest_sha256=(
                expected_prefit_parity_manifest_sha256
            ),
            expected_sharded_execution_plan_sha256=(
                expected_sharded_execution_plan_sha256
            ),
        )
        expected_product_record = {
            "worker_count": product.worker_count,
            "query_count": product.query_count,
            "folds": list(product.folds),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "lineage": dict(product.lineage),
            "digests": _validate_digests(product.digests),
            "materialized_io_bytes": product.materialized_io_bytes,
            "site_model_bytes": product.site_model_bytes,
            "materialized_families": {
                name: int(product.materialized_families[name])
                for name in MATERIALIZED_FAMILIES
            },
            "artifact_manifest": {
                "path": str(artifact_path.relative_to(benchmark_path.parent)),
                "sha256": sha256_file(artifact_path),
            },
        }
        observed_product_record = {
            key: record.get(key) for key in expected_product_record
        }
        if observed_product_record != expected_product_record:
            raise PhysicalBenchmarkInvariantError(
                "physical trial benchmark record differs from core artifacts"
            )
        metrics = PhysicalResourceMetricsV12(**dict(record.get("resources") or {}))
        parsed_trials[worker] = PhysicalTrialEvidenceV12(product, metrics)
    if 1 not in parsed_trials or 3 not in parsed_trials:
        raise PhysicalBenchmarkInvariantError("physical authorization lacks W1/W3")
    thresholds = PhysicalBenchmarkThresholdsV12(**dict(benchmark.get("thresholds") or {}))
    recomputed: dict[int, CandidateDecisionV12] = {
        3: compare_candidate_v12(parsed_trials[1], parsed_trials[3], thresholds=thresholds)
    }
    if 2 in parsed_trials:
        recomputed[2] = compare_candidate_v12(
            parsed_trials[1], parsed_trials[2], thresholds=thresholds
        )
    expected_decisions = {
        str(worker): asdict(value) for worker, value in sorted(recomputed.items())
    }
    if dict(decisions_raw) != expected_decisions:
        raise PhysicalBenchmarkInvariantError("physical worker decision was not reproducible")
    selected = int(benchmark.get("selected_worker_count") or 0)
    expected_selected = 3 if recomputed[3].eligible else (
        2 if recomputed.get(2) is not None and recomputed[2].eligible else 0
    )
    plan_raw = json.loads(plan_path.read_text(encoding="utf-8"))
    national_query_count = int(
        ((plan_raw.get("input_lineage") or {}).get("scene_ir") or {}).get(
            "query_count", 0
        )
    )
    maximum_temporary_bytes = int(
        (plan_raw.get("resources") or {}).get("maximum_temporary_bytes", 0)
    )
    reference_product = parsed_trials[1].product
    projected_families = {
        name: (
            int(reference_product.materialized_families[name])
            if name == "site_models"
            else int(
                math.ceil(
                    int(reference_product.materialized_families[name])
                    * national_query_count
                    / SAMPLE_QUERY_COUNT
                )
            )
        )
        for name in MATERIALIZED_FAMILIES
    }
    sample_overhead = reference_product.materialized_io_bytes - sum(
        int(value) for value in reference_product.materialized_families.values()
    )
    projected_overhead = int(
        math.ceil(sample_overhead * national_query_count / SAMPLE_QUERY_COUNT)
    )
    projected_national_bytes = sum(projected_families.values()) + projected_overhead
    expected_storage = {
        "method": "W1_SAMPLE_LINEAR_EXCEPT_SITE_MODELS_CONSTANT",
        "sample_query_count": SAMPLE_QUERY_COUNT,
        "national_query_count": national_query_count,
        "sample_total_materialized_bytes": reference_product.materialized_io_bytes,
        "sample_site_model_bytes": reference_product.site_model_bytes,
        "sample_families": dict(reference_product.materialized_families),
        "sample_other_bytes": sample_overhead,
        "projected_families": projected_families,
        "projected_other_bytes": projected_overhead,
        "projected_national_bytes": projected_national_bytes,
        "maximum_temporary_bytes": maximum_temporary_bytes,
        "pass": projected_national_bytes <= maximum_temporary_bytes,
    }
    if (
        selected != expected_selected
        or authorization.get("selected_worker_count") != selected
        or selected not in (2, 3)
        or benchmark.get("selection") != f"W{selected}"
        or benchmark.get("all_selected_digests_exact_to_w1") is not True
        or benchmark.get("national_storage_projection") != expected_storage
        or expected_storage["pass"] is not True
    ):
        raise PhysicalBenchmarkInvariantError("physical worker selection was altered")
    return ShardedOOFAuthorizationV12(
        worker_count=selected,
        manifest_path=benchmark_path,
        manifest_sha256=benchmark_sha,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha,
        prefit_parity_manifest_sha256=expected_prefit_parity_manifest_sha256,
        sharded_execution_plan_sha256=expected_sharded_execution_plan_sha256,
        lineage=lineage,
    )


__all__ = [
    "ALLOWED_FOLDS",
    "AUTHORIZATION_SCHEMA_VERSION",
    "CandidateDecisionV12",
    "MATERIALIZED_FAMILIES",
    "PhysicalBenchmarkInvariantError",
    "PhysicalBenchmarkThresholdsV12",
    "PhysicalResourceMetricsV12",
    "PhysicalTrialEvidenceV12",
    "PhysicalTrialProductV12",
    "REQUIRED_DIGESTS",
    "SAMPLE_QUERY_COUNT",
    "SCHEMA_VERSION",
    "ShardedOOFAuthorizationV12",
    "compare_candidate_v12",
    "load_sharded_oof_authorization_v12",
    "measure_physical_trial_v12",
    "physical_trial_product_from_sharded_manifest_v12",
    "physical_trial_product_from_sharded_v12",
    "run_physical_worker_benchmark_v12",
]
