#!/usr/bin/env python3
"""Run and publish the sealed V4.12 per-query retrieval worker.

The parent is intentionally independent from matching code.  It verifies the
frozen contract inputs, creates a private two-module package, launches it in a
deny-by-default macOS sandbox, validates its three outputs, then publishes the
audit before the immutable runtime artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import joblib
import numpy
import pandas
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn


STOP = "STOP_V412_UNIT_RETRIEVAL"
SEALED = "SEALED_V412_UNIT_RETRIEVAL"
PARITY_GO = "GO_V412_UNIT_RETRIEVAL_PARITY"
PARITY_RUN_SPEC_SCHEMA = "sireto-v4.12-unit-retrieval-parity-run-spec-1"
PARITY_BUILD_SCHEMA = "sireto-v4.12-unit-retrieval-parity-build-1"
PARITY_SOURCE_PATH = Path("scripts/audit_v412_unit_retrieval_parity.py")
PARITY_PROFILE_PATH = Path("config/v4_12_unit_retrieval_parity.sb")
PRIVATE_PYTHON_RELATIVE = Path(
    "runtime/Python.framework/Versions/3.14/"
    "Resources/Python.app/Contents/MacOS/Python"
)
PRIVATE_LIBRARY_RELATIVE = Path(
    "runtime/Python.framework/Versions/3.14/Python"
)
PLAN_PATH = Path("config/v4_12_unit_retrieval_engine_plan.json")
LOCK_PATH = Path("config/v4_12_unit_retrieval_execution_lock.json")
PLAN_SCHEMA = "sireto-v4.12-unit-retrieval-engine-plan-1"
LOCK_SCHEMA = "sireto-v4.12-unit-retrieval-execution-lock-1"
LOCK_PURPOSE = "V4.12_UNIT_RETRIEVAL_PARITY"
LOCK_VERDICT = "GO_CODE_V412_UNIT_RETRIEVAL"
LOCK_PROJECTION_SCHEMA = "sireto-v4.12-unit-retrieval-worker-lock-projection-1"
WORKER_INTEGRITY_SCHEMA = "sireto-v4.12-unit-retrieval-integrity-1"
WORKER_MANIFEST_SCHEMA = "sireto-v4.12-unit-retrieval-manifest-1"
WORKER_AUDIT_SCHEMA = "sireto-v4.12-unit-retrieval-audit-manifest-1"
WORKER_PROVENANCE_SCHEMA = "sireto-v4.12-unit-retrieval-provenance-1"
RUN_SPEC_SCHEMA = "sireto-v4.12-unit-retrieval-worker-run-spec-1"
LOOKUP_DESCRIPTOR_SCHEMA = "sireto-v4.12-strict-lookup-descriptor-1"
PROFILE_MARKERS = {
    "@@SYSTEM_READ_RULES@@",
    "@@DEVICE_READ_RULES@@",
    "@@ALLOWED_READ_RULES@@",
    "@@ANCESTOR_METADATA_RULES@@",
    "@@EXPLICIT_DENY_RULES@@",
}
QUERY_COLUMNS = [
    "query_id",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
]
STATUS_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("candidate_count", pa.uint8(), nullable=False),
    ]
)
CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("candidate_rank", pa.uint8(), nullable=False),
        pa.field("candidate_siret", pa.string(), nullable=False),
    ]
)
LEDGER_SCHEMA = pa.schema(
    [
        pa.field("role", pa.string(), nullable=False),
        pa.field("absolute_path", pa.string(), nullable=False),
        pa.field("projection", pa.string(), nullable=False),
        pa.field("size_before", pa.uint64(), nullable=False),
        pa.field("sha256_before", pa.string(), nullable=False),
        pa.field("size_after", pa.uint64(), nullable=False),
        pa.field("sha256_after", pa.string(), nullable=False),
    ]
)
DECLARATIONS = {
    "labels_opened": False,
    "oracle_opened": False,
    "historical_candidates_opened": False,
    "models_opened": False,
    "network_used": False,
    "writes_outside_staging": False,
    "cache_rebuild_attempted": False,
    "positive_injection": False,
}
INTEGRITY_KEYS = {
    "schema_version",
    "worker_build_id",
    "query_count",
    "candidate_count",
    "minimum_pool_size",
    "maximum_pool_size",
    "under_ceiling_query_count",
    "empty_query_count",
    "lookup_missing_count",
    "candidate_payload_bytes",
    "candidate_payload_sha256",
    "status_payload_bytes",
    "status_payload_sha256",
    "sandbox_checks",
    "peak_rss_bytes",
    "durations_ns",
    "declarations",
}
SANDBOX_CHECK_KEYS = {
    "allowed_read",
    "oracle_denied",
    "oracle_audit_denied",
    "historical_denied",
    "model_denied",
    "network_denied",
    "write_denied",
}
DURATION_KEYS = {"retrieval", "lookup", "serialization", "total"}
PLAN_KEYS = {
    "schema_version",
    "purpose",
    "prerequisite",
    "safe_input",
    "retrieval",
    "tfidf_cache",
    "expected_routing",
    "historical_parity",
    "forbidden_worker_roots",
    "forbidden_worker_files",
    "parity_controller",
    "outputs",
    "identity_projections",
    "runtime",
    "max_rss_bytes",
    "parent_sources",
    "worker_sources",
    "parity_sources",
    "test_sources",
    "independent_audit_sources",
}
LOCK_KEYS = {
    "schema_version",
    "purpose",
    "audit_verdict",
    "git_commit",
    "source_hashes",
    "input_paths",
    "input_hashes",
    "worker_policy_sha256",
    "worker_lock_projection_sha256",
    "runtime",
    "outputs",
    "max_rss_bytes",
    "sandbox",
}
SANDBOX_KEYS = {
    "executable",
    "executable_sha256",
    "python_framework_app",
    "python_framework_app_sha256",
    "python_framework_library",
    "python_framework_library_sha256",
    "git_executable",
    "git_executable_sha256",
    "system_read_roots",
    "device_read_paths",
    "network_allowed",
    "fork_allowed",
    "write_scope",
}
RUNTIME_MANIFEST_KEYS = {
    "schema_version",
    "worker_build_id",
    "safe_input_build_id",
    "strict_stores_build_id",
    "files",
    "runtime",
    "declarations",
    "verdict",
}
AUDIT_MANIFEST_KEYS = {"schema_version", "worker_build_id", "files"}
PROVENANCE_KEYS = {
    "schema_version",
    "worker_build_id",
    "git_commit",
    "parent_source_hashes",
    "worker_source_hashes",
    "lock_sha256",
    "plan_sha256",
    "runtime",
    "data_input_count",
    "runtime_manifest_sha256",
    "declarations",
}
PARITY_WORKER_FILES = {
    "query_status.parquet",
    "candidates_top100.parquet",
    "integrity.json",
}
PARITY_EXPECTED_KEYS = {
    "query_count",
    "candidate_count",
    "minimum_pool_size",
    "maximum_pool_size",
    "under_ceiling_query_count",
    "empty_query_count",
    "candidate_payload_bytes",
    "candidate_payload_sha256",
    "status_payload_bytes",
    "status_payload_sha256",
}
PARITY_DECLARATIONS = {
    "oracle_opened": False,
    "oracle_audit_opened": False,
    "historical_candidates_opened": False,
    "models_opened": False,
    "stores_opened": False,
    "network_used": False,
    "writes_outside_staging": False,
}
PARITY_RESULT_KEYS = {"verdict", "parity_build_id", "audit"}
_ACTIVE_PRIVATE_ROOTS: dict[Path, tuple[int, int]] = {}


class RetrievalRunStopped(RuntimeError):
    pass


def _stop(message: str) -> None:
    raise RetrievalRunStopped(f"{STOP}: {message}")


def canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _stop(f"non-canonical JSON value: {exc}")


def _parse_json(payload: bytes, label: str) -> dict[str, Any]:
    duplicates: list[str] = []

    def hook(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _stop(f"invalid {label}: {exc}")
    if duplicates or type(value) is not dict:
        _stop(f"invalid or duplicate-key {label}")
    return value


def _runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pa.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
        "duckdb": duckdb.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def _rss(limit: int) -> None:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if peak > limit:
        _stop("parent RSS limit exceeded")


def _components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except OSError as exc:
            _stop(f"cannot inspect path component {current}: {exc}")
        if stat.S_ISLNK(info.st_mode):
            _stop(f"symlink component forbidden: {current}")


def _openat_anchored(
    path: Path,
    *,
    directory: bool,
) -> tuple[Path, int]:
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts or absolute == Path("/"):
        _stop(f"unsafe anchored path: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.open("/", directory_flags)
    try:
        components = absolute.parts[1:]
        for index, component in enumerate(components):
            final = index == len(components) - 1
            flags = (
                directory_flags
                if not final or directory
                else os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                child = os.open(component, flags, dir_fd=current)
            except OSError as exc:
                _stop(f"anchored open failed at {absolute}: {exc}")
            os.close(current)
            current = child
        info = os.fstat(current)
        expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if not expected:
            _stop(f"anchored path has wrong type: {absolute}")
        return absolute, current
    except BaseException:
        os.close(current)
        raise


def _regular(path: Path) -> os.stat_result:
    _absolute, descriptor = _openat_anchored(path, directory=False)
    try:
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def _directory(path: Path) -> os.stat_result:
    _absolute, descriptor = _openat_anchored(path, directory=True)
    try:
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def _canonical_file(path: Path) -> Path:
    absolute, descriptor = _openat_anchored(path, directory=False)
    os.close(descriptor)
    return absolute


def _canonical_dir(path: Path) -> Path:
    absolute, descriptor = _openat_anchored(path, directory=True)
    os.close(descriptor)
    return absolute


def _snapshot_fd(descriptor: int, limit: int) -> dict[str, Any]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        _stop("anchored descriptor is not regular")
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(descriptor, 8 * 1024 * 1024, offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
        _rss(limit)
    after = os.fstat(descriptor)
    identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in identity):
        _stop("anchored descriptor changed while hashing")
    return {
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode": stat.S_IMODE(after.st_mode),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _snapshot(path: Path, limit: int) -> dict[str, Any]:
    _canonical, descriptor = _openat_anchored(path, directory=False)
    try:
        return _snapshot_fd(descriptor, limit)
    finally:
        os.close(descriptor)


def _read(path: Path, limit: int, maximum: int = 512 * 1024 * 1024) -> bytes:
    _canonical, descriptor = _openat_anchored(path, directory=False)
    try:
        snap = _snapshot_fd(descriptor, limit)
        if snap["size"] > maximum:
            _stop(f"control file too large: {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        if _snapshot_fd(descriptor, limit) != snap:
            _stop(f"control file changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_anchored(path: Path, expected: Mapping[str, Any], limit: int) -> int:
    _canonical, descriptor = _openat_anchored(path, directory=False)
    if _snapshot_fd(descriptor, limit) != dict(expected):
        os.close(descriptor)
        _stop(f"anchored descriptor mismatch: {path}")
    return descriptor


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                _stop(f"short write: {path}")
            view = view[count:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Any, mode: int = 0o444) -> None:
    _write_exclusive(path, canonical_json(value), mode)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _canonical_dir(path)


def _fsync_dir(path: Path) -> None:
    _canonical, descriptor = _openat_anchored(path, directory=True)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        _stop(f"Git verification failed: {args}")
    return result.stdout


def validate_plan(plan: Mapping[str, Any]) -> None:
    if set(plan) != PLAN_KEYS or plan.get("schema_version") != PLAN_SCHEMA:
        _stop("plan schema/keyset mismatch")
    if plan.get("purpose") != LOCK_PURPOSE:
        _stop("plan purpose mismatch")
    if plan.get("runtime") != _runtime():
        _stop("plan runtime mismatch")
    if plan.get("max_rss_bytes") != 8 * 1024**3:
        _stop("plan RSS limit mismatch")
    if plan["safe_input"].get("query_count") != 1456:
        _stop("safe query count mismatch")
    if plan["safe_input"].get("columns") != QUERY_COLUMNS:
        _stop("safe query projection mismatch")
    if plan["retrieval"].get("candidate_ceiling") != 100:
        _stop("candidate ceiling mismatch")
    if plan["historical_parity"].get("controller_opens_reference") is not False:
        _stop("historical reference boundary changed")
    identity = plan.get("identity_projections")
    expected_policy = [
        "prerequisite",
        "safe_input",
        "retrieval",
        "tfidf_cache",
        "expected_routing",
        "outputs",
        "runtime",
        "max_rss_bytes",
    ]
    if (
        type(identity) is not dict
        or identity.get("worker_policy_keys") != expected_policy
        or identity.get("worker_receives_full_plan") is not False
        or identity.get("worker_receives_full_lock") is not False
    ):
        _stop("worker identity projection mismatch")
    all_sources: list[str] = []
    for key in (
        "parent_sources",
        "worker_sources",
        "parity_sources",
        "test_sources",
        "independent_audit_sources",
    ):
        values = plan.get(key)
        if type(values) is not list or any(type(item) is not str for item in values):
            _stop(f"invalid source list: {key}")
        all_sources.extend(values)
    if len(all_sources) != len(set(all_sources)):
        _stop("duplicate planned source")


def _worker_policy(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: plan[key]
        for key in plan["identity_projections"]["worker_policy_keys"]
    }


def _expected_lock_inputs(
    plan: Mapping[str, Any],
    sandbox: Mapping[str, Any],
    limit: int,
) -> tuple[dict[str, str], dict[str, str]]:
    safe_root = Path(plan["safe_input"]["root"])
    safe_manifest_payload = _read(safe_root / "runtime_manifest.json", limit)
    if hashlib.sha256(safe_manifest_payload).hexdigest() != plan["safe_input"][
        "runtime_manifest_sha256"
    ]:
        _stop("safe manifest changed before lock validation")
    safe_manifest = _parse_json(safe_manifest_payload, "safe runtime manifest")
    safe_files = safe_manifest.get("files")
    expected_safe_names = {
        "runtime_manifest.json",
        "queries_all.parquet",
        "queries_dev.parquet",
        "partition_inventory.parquet",
        "tfidf_inventory.parquet",
        "integrity.json",
    }
    if type(safe_files) is not dict or set(safe_files) != expected_safe_names - {
        "runtime_manifest.json"
    }:
        _stop("safe package file inventory mismatch")
    gate_root = Path(plan["prerequisite"]["certification_root"])
    gate_audit = Path(plan["prerequisite"]["audit_root"])
    paths = {
        "safe_runtime_manifest": str(safe_root / "runtime_manifest.json"),
        "safe_queries_all": str(safe_root / "queries_all.parquet"),
        "safe_queries_dev": str(safe_root / "queries_dev.parquet"),
        "safe_partition_inventory": str(safe_root / "partition_inventory.parquet"),
        "safe_tfidf_inventory": str(safe_root / "tfidf_inventory.parquet"),
        "safe_input_integrity": str(safe_root / "integrity.json"),
        "strict_stores_manifest": str(gate_root / "manifest.json"),
        "strict_stores_run_spec": str(gate_root / "run_spec.json"),
        "strict_stores_lookup_descriptor": str(
            gate_root / "lookup_descriptor.json"
        ),
        "strict_stores_store_probe": str(gate_root / "store_probe.json"),
        "strict_stores_audit_manifest": str(gate_audit / "manifest.json"),
        "strict_stores_data_ledger": str(gate_audit / "open_ledger.parquet"),
        "sandbox_executable": sandbox["executable"],
        "python_framework_app": sandbox["python_framework_app"],
        "python_framework_library": sandbox["python_framework_library"],
        "git_executable": sandbox["git_executable"],
    }
    hashes = {
        "safe_runtime_manifest": plan["safe_input"]["runtime_manifest_sha256"],
        **{
            role: safe_files[name]["sha256"]
            for role, name in (
                ("safe_queries_all", "queries_all.parquet"),
                ("safe_queries_dev", "queries_dev.parquet"),
                ("safe_partition_inventory", "partition_inventory.parquet"),
                ("safe_tfidf_inventory", "tfidf_inventory.parquet"),
                ("safe_input_integrity", "integrity.json"),
            )
        },
        "strict_stores_manifest": plan["prerequisite"][
            "certification_manifest_sha256"
        ],
        "strict_stores_run_spec": plan["prerequisite"]["run_spec_sha256"],
        "strict_stores_lookup_descriptor": plan["prerequisite"][
            "lookup_descriptor_sha256"
        ],
        "strict_stores_store_probe": plan["prerequisite"]["store_probe_sha256"],
        "strict_stores_audit_manifest": plan["prerequisite"][
            "audit_manifest_sha256"
        ],
        "strict_stores_data_ledger": plan["prerequisite"]["data_ledger_sha256"],
        "sandbox_executable": sandbox["executable_sha256"],
        "python_framework_app": sandbox["python_framework_app_sha256"],
        "python_framework_library": sandbox["python_framework_library_sha256"],
        "git_executable": sandbox["git_executable_sha256"],
    }
    return paths, hashes


def _worker_lock_projection(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    worker_sources = set(plan["parent_sources"] + plan["worker_sources"])
    return {
        "schema_version": LOCK_PROJECTION_SCHEMA,
        "git_commit": lock["git_commit"],
        "source_hashes": {
            name: lock["source_hashes"][name]
            for name in sorted(worker_sources)
        },
        "input_paths": lock["input_paths"],
        "input_hashes": lock["input_hashes"],
        "sandbox": lock["sandbox"],
        "outputs": {
            key: plan["outputs"][key]
            for key in ("runtime_root", "worker_audit_root", "temp_root")
        },
        "runtime": lock["runtime"],
        "max_rss_bytes": lock["max_rss_bytes"],
    }


def validate_lock(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    repo: Path,
    plan_sha: str,
    limit: int,
) -> dict[Path, dict[str, Any]]:
    if set(lock) != LOCK_KEYS or lock.get("schema_version") != LOCK_SCHEMA:
        _stop("lock schema/keyset mismatch")
    if (
        lock.get("purpose") != LOCK_PURPOSE
        or lock.get("audit_verdict") != LOCK_VERDICT
        or lock.get("runtime") != plan["runtime"]
        or lock.get("outputs") != plan["outputs"]
        or lock.get("max_rss_bytes") != limit
    ):
        _stop("lock values mismatch")
    commit = lock.get("git_commit")
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        _stop("invalid locked commit")
    if str(_git(repo, "cat-file", "-t", commit)).strip() != "commit":
        _stop("locked object is not a commit")
    policy_sha = hashlib.sha256(canonical_json(_worker_policy(plan))).hexdigest()
    if lock.get("worker_policy_sha256") != policy_sha:
        _stop("worker policy projection hash mismatch")
    source_hashes = lock.get("source_hashes")
    if type(source_hashes) is not dict:
        _stop("lock source hashes missing")
    planned = set(
        plan["parent_sources"]
        + plan["worker_sources"]
        + plan["parity_sources"]
        + plan["test_sources"]
        + plan["independent_audit_sources"]
    )
    if set(source_hashes) != planned:
        _stop("lock source closure mismatch")
    snapshots: dict[Path, dict[str, Any]] = {}
    for relative in sorted(planned):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            _stop("unsafe source path")
        worktree = _canonical_file(repo / rel)
        snap = _snapshot(worktree, limit)
        if snap["sha256"] != source_hashes[relative]:
            _stop(f"source worktree differs from lock: {relative}")
        blob = _git(repo, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(blob, bytes)
        if hashlib.sha256(blob).hexdigest() != source_hashes[relative]:
            _stop(f"source blob differs from lock: {relative}")
        snapshots[worktree] = snap
    if source_hashes[str(PLAN_PATH)] != plan_sha:
        _stop("plan is not sealed by lock")
    sandbox = lock.get("sandbox")
    if type(sandbox) is not dict or set(sandbox) != SANDBOX_KEYS:
        _stop("sandbox lock keyset mismatch")
    if (
        sandbox["system_read_roots"] != ["/System", "/usr", "/opt/homebrew"]
        or sandbox["device_read_paths"] != ["/dev/null", "/dev/urandom", "/dev/fd"]
        or sandbox["network_allowed"] is not False
        or sandbox["fork_allowed"] is not False
        or sandbox["write_scope"] != "PRIVATE_WORKER_STAGING_ONLY"
        or sandbox["executable"] != "/usr/bin/sandbox-exec"
        or sandbox["git_executable"] != "/usr/bin/git"
    ):
        _stop("sandbox lock values mismatch")
    input_paths = lock.get("input_paths")
    input_hashes = lock.get("input_hashes")
    expected_paths, expected_hashes = _expected_lock_inputs(plan, sandbox, limit)
    if input_paths != expected_paths or input_hashes != expected_hashes:
        _stop("lock input paths/hashes mismatch")
    for role, raw in input_paths.items():
        path = _canonical_file(Path(raw))
        snap = _snapshot(path, limit)
        if snap["sha256"] != input_hashes[role]:
            _stop(f"locked input changed: {role}")
        snapshots[path] = snap
    projection_sha = hashlib.sha256(
        canonical_json(_worker_lock_projection(plan, lock))
    ).hexdigest()
    if lock.get("worker_lock_projection_sha256") != projection_sha:
        _stop("worker lock projection hash mismatch")
    return snapshots


def _verify_gate_a(plan: Mapping[str, Any], limit: int) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = plan["prerequisite"]
    root = _canonical_dir(Path(gate["certification_root"]))
    expected = {
        "manifest.json": gate["certification_manifest_sha256"],
        "run_spec.json": gate["run_spec_sha256"],
        "lookup_descriptor.json": gate["lookup_descriptor_sha256"],
        "store_probe.json": gate["store_probe_sha256"],
    }
    documents: dict[str, Any] = {}
    for name, digest in expected.items():
        payload = _read(root / name, limit)
        if hashlib.sha256(payload).hexdigest() != digest:
            _stop(f"Gate A {name} hash mismatch")
        documents[name] = _parse_json(payload, f"Gate A {name}")
    manifest = documents["manifest.json"]
    if (
        manifest.get("build_id") != gate["build_id"]
        or manifest.get("verdict") != gate["verdict"]
    ):
        _stop("Gate A prerequisite mismatch")
    run_spec = documents["run_spec.json"]
    descriptor = documents["lookup_descriptor.json"]
    if descriptor.get("schema_version") != LOOKUP_DESCRIPTOR_SCHEMA:
        _stop("lookup descriptor schema mismatch")
    allowed = run_spec.get("allowed_read_files")
    if type(allowed) is not list or len(allowed) != 1945:
        _stop("Gate A allow-list count mismatch")
    counts = Counter(row.get("role") for row in allowed if type(row) is dict)
    if counts != Counter(
        {"partition": 648, "cache_pickle": 648, "cache_sidecar": 648, "lookup_database": 1}
    ):
        _stop("Gate A allow-list roles mismatch")
    audit_root = _canonical_dir(Path(gate["audit_root"]))
    audit_manifest_payload = _read(audit_root / "manifest.json", limit)
    if hashlib.sha256(audit_manifest_payload).hexdigest() != gate[
        "audit_manifest_sha256"
    ]:
        _stop("Gate A audit manifest hash mismatch")
    audit_manifest = _parse_json(audit_manifest_payload, "Gate A audit manifest")
    audit_files = audit_manifest.get("files")
    if type(audit_files) is not list:
        _stop("Gate A audit file records missing")
    ledger_records = [
        row
        for row in audit_files
        if type(row) is dict and row.get("path") == "open_ledger.parquet"
    ]
    if len(ledger_records) != 1:
        _stop("Gate A data ledger record missing")
    ledger_payload = _read(audit_root / "open_ledger.parquet", limit)
    ledger_sha = hashlib.sha256(ledger_payload).hexdigest()
    if (
        ledger_sha != gate["data_ledger_sha256"]
        or ledger_records[0].get("sha256") != ledger_sha
        or ledger_records[0].get("size_bytes") != len(ledger_payload)
    ):
        _stop("Gate A data ledger mismatch")
    return run_spec, descriptor


def _verify_safe_input(plan: Mapping[str, Any], limit: int) -> tuple[Path, dict[str, Any], pa.Table]:
    safe = plan["safe_input"]
    root = _canonical_dir(Path(safe["root"]))
    manifest_path = root / "runtime_manifest.json"
    manifest_payload = _read(manifest_path, limit)
    if hashlib.sha256(manifest_payload).hexdigest() != safe["runtime_manifest_sha256"]:
        _stop("safe runtime manifest hash mismatch")
    manifest = _parse_json(manifest_payload, "safe runtime manifest")
    if manifest.get("build_id") != safe["build_id"]:
        _stop("safe input build mismatch")
    queries_path = root / "queries_dev.parquet"
    payload = _read(queries_path, limit)
    if hashlib.sha256(payload).hexdigest() != safe["queries_dev_sha256"]:
        _stop("safe queries hash mismatch")
    parquet = pq.ParquetFile(pa.BufferReader(payload))
    expected = pa.schema(
        [pa.field(column, pa.string(), nullable=False) for column in QUERY_COLUMNS]
    )
    if parquet.schema_arrow != expected or parquet.schema_arrow.metadata is not None:
        _stop("safe query schema/metadata mismatch")
    table = parquet.read(columns=QUERY_COLUMNS, use_threads=False)
    if table.num_rows != safe["query_count"]:
        _stop("safe query row count mismatch")
    ids = table["query_id"].to_pylist()
    if len(ids) != len(set(ids)):
        _stop("duplicate safe query ID")
    if not all(type(value) is str for value in ids):
        _stop("invalid safe query ID")
    payload_ids = b"".join(f"{value}\n".encode() for value in ids)
    if hashlib.sha256(payload_ids).hexdigest() != safe["query_id_payload_sha256"]:
        _stop("safe query ID payload mismatch")
    return queries_path, manifest, table


def _partition_key(relative: str) -> str:
    match = re.fullmatch(r"insee/insee=([0-9]{5})/[^/]+\.parquet", relative)
    if match:
        return match.group(1) + "_"
    match = re.fullmatch(r"cp/postcode=([0-9]{5})/[^/]+\.parquet", relative)
    if match:
        return "_" + match.group(1)
    _stop("invalid Gate A partition path")


def _derive_routes(
    queries: pa.Table, run_spec: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[dict[str, str]]:
    partition_keys = {
        _partition_key(row["relative_path"]) for row in run_spec["partition_records"]
    }
    cache_keys = {row["partition_key"] for row in run_spec["cache_records"]}
    routes = []
    insee_count = cp_count = 0

    def geo(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        if not text.strip() or text.lower() == "nan":
            return None
        text = text.strip()
        return text.split(".")[0] if re.fullmatch(r"\d+\.0+", text) else text

    for row in queries.to_pylist():
        insee = geo(row["crm_insee"])
        postcode = geo(row["crm_postcode"])
        insee_key = None if insee is None else insee + "_"
        cp_key = None if postcode is None else "_" + postcode
        if insee_key in partition_keys:
            key = insee_key
            insee_count += 1
        elif cp_key in partition_keys:
            key = cp_key
            cp_count += 1
        else:
            _stop(f"missing geographic route: {row['query_id']}")
        if key not in cache_keys:
            _stop(f"missing routed cache: {key}")
        routes.append({"query_id": row["query_id"], "partition_key": key})
    payload = b"".join(
        row["query_id"].encode() + b"\0" + row["partition_key"].encode() + b"\n"
        for row in routes
    )
    observed = {
        "query_count": len(routes),
        "insee_query_count": insee_count,
        "cp_query_count": cp_count,
        "distinct_key_count": len({row["partition_key"] for row in routes}),
        "missing_key_count": 0,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if observed != dict(expected):
        _stop("routing differs from frozen plan")
    return routes


def _sb(value: str) -> str:
    return json.dumps(value)


def _ancestors(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        current = path.absolute()
        while current != current.parent:
            result.add(str(current.parent))
            current = current.parent
    return result


def render_profile(
    template: str,
    *,
    allowed_files: Sequence[Path],
    forbidden_roots: Sequence[Path],
    system_roots: Sequence[Path],
    devices: Sequence[Path],
    metadata_extra: Sequence[Path],
) -> str:
    if {marker for marker in PROFILE_MARKERS if marker not in template}:
        _stop("sandbox profile marker missing")
    system_parents = sorted(
        {
            str(parent)
            for root in system_roots
            for parent in root.absolute().parents
        },
        key=lambda value: value.encode(),
    )
    system = "\n".join(
        [*(f"  (literal {_sb(path)})" for path in system_parents)]
        + [
        f"  (literal {_sb(str(path))})\n  (subpath {_sb(str(path))})"
        for path in system_roots
        ]
    )
    device = "\n".join(
        f"  (literal {_sb(str(path))})\n  (subpath {_sb(str(path))})"
        for path in devices
    )
    allowed = "\n".join(
        f"  (literal {_sb(str(path))})"
        for path in sorted(set(allowed_files), key=lambda item: str(item).encode())
    )
    metadata_paths = _ancestors([*allowed_files, *metadata_extra, *system_roots])
    metadata = "\n".join(
        f"  (literal {_sb(path)})"
        for path in sorted(metadata_paths, key=lambda item: item.encode())
    )
    denies = "\n".join(
        f"(deny file-read* file-write* (subpath {_sb(str(path))}))"
        for path in sorted(set(forbidden_roots), key=lambda item: str(item).encode())
    )
    replacements = {
        "@@SYSTEM_READ_RULES@@": system,
        "@@DEVICE_READ_RULES@@": device,
        "@@ALLOWED_READ_RULES@@": allowed,
        "@@ANCESTOR_METADATA_RULES@@": metadata,
        "@@EXPLICIT_DENY_RULES@@": denies,
    }
    effective = template
    for marker, value in replacements.items():
        effective = effective.replace(marker, value)
    if "@@" in effective:
        _stop("unresolved sandbox marker")
    return effective.rstrip() + "\n"


def _copy_private_python(run_root: Path, sandbox: Mapping[str, Any], limit: int) -> tuple[Path, Path]:
    runtime_root = run_root / "runtime"
    framework = runtime_root / "Python.framework/Versions/3.14"
    executable = run_root / PRIVATE_PYTHON_RELATIVE
    library = run_root / PRIVATE_LIBRARY_RELATIVE
    executable.parent.mkdir(parents=True, mode=0o700)
    app_source = _canonical_file(Path(sandbox["python_framework_app"]))
    lib_source = _canonical_file(Path(sandbox["python_framework_library"]))
    for source, destination, mode, expected in (
        (app_source, executable, 0o555, sandbox["python_framework_app_sha256"]),
        (lib_source, library, 0o444, sandbox["python_framework_library_sha256"]),
    ):
        payload = _read(source, limit)
        if hashlib.sha256(payload).hexdigest() != expected:
            _stop("private Python source hash mismatch")
        _write_exclusive(destination, payload, mode)
    directories = [runtime_root, *(
        item for item in runtime_root.rglob("*") if item.is_dir()
    )]
    for path in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        os.chmod(path, 0o555)
    if (
        stat.S_IMODE(os.lstat(executable).st_mode) != 0o555
        or stat.S_IMODE(os.lstat(library).st_mode) != 0o444
        or any(
            stat.S_ISLNK(os.lstat(path).st_mode)
            or stat.S_IMODE(os.lstat(path).st_mode) != 0o555
            for path in directories
        )
    ):
        _stop("private Python boundary sealing failed")
    return executable, runtime_root


def _record(path: Path, limit: int) -> dict[str, Any]:
    snap = _snapshot(path, limit)
    return {"path": path.name, "size_bytes": snap["size"], "sha256": snap["sha256"]}


def _runtime_file_records(root: Path, limit: int) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in ("query_status.parquet", "candidates_top100.parquet"):
        path = root / name
        payload = _read(path, limit)
        parquet = pq.ParquetFile(pa.BufferReader(payload))
        records[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "row_count": parquet.metadata.num_rows,
            "schema": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in parquet.schema_arrow
            ],
            "metadata": parquet.schema_arrow.metadata,
        }
    integrity_payload = _read(root / "integrity.json", limit)
    records["integrity.json"] = {
        "sha256": hashlib.sha256(integrity_payload).hexdigest(),
        "size_bytes": len(integrity_payload),
    }
    return records


def _validate_outputs(
    output: Path,
    worker_build_id: str,
    query_ids: Sequence[str],
    limit: int,
) -> dict[str, Any]:
    expected_names = {"query_status.parquet", "candidates_top100.parquet", "integrity.json"}
    if {path.name for path in output.iterdir()} != expected_names:
        _stop("worker output file-set mismatch")
    status_path = output / "query_status.parquet"
    candidate_path = output / "candidates_top100.parquet"
    parquet_payloads = {
        status_path: _read(status_path, limit),
        candidate_path: _read(candidate_path, limit),
    }
    for path, schema in ((status_path, STATUS_SCHEMA), (candidate_path, CANDIDATE_SCHEMA)):
        parquet = pq.ParquetFile(pa.BufferReader(parquet_payloads[path]))
        if parquet.schema_arrow != schema or parquet.schema_arrow.metadata is not None:
            _stop(f"worker Parquet schema/metadata mismatch: {path.name}")
    status = pq.read_table(pa.BufferReader(parquet_payloads[status_path]))
    candidates = pq.read_table(pa.BufferReader(parquet_payloads[candidate_path]))
    if status.num_rows != len(query_ids) or status["query_id"].to_pylist() != list(query_ids):
        _stop("worker status query population/order mismatch")
    counts = status["candidate_count"].to_pylist()
    if any(type(value) is not int or not 0 <= value <= 100 for value in counts):
        _stop("worker candidate count invalid")
    rows = candidates.to_pylist()
    position = {query_id: index for index, query_id in enumerate(query_ids)}
    previous: tuple[int, int] | None = None
    observed: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for row in rows:
        query_id = row["query_id"]
        rank = row["candidate_rank"]
        siret = row["candidate_siret"]
        if query_id not in position or re.fullmatch(r"[0-9]{14}", siret) is None:
            _stop("worker candidate value invalid")
        current = (position[query_id], rank)
        if previous is not None and current <= previous:
            _stop("worker candidate order invalid")
        previous = current
        observed[query_id] += 1
        if rank != observed[query_id] or (query_id, siret) in seen:
            _stop("worker ranks or SIRET uniqueness invalid")
        seen.add((query_id, siret))
    if [observed[query_id] for query_id in query_ids] != counts:
        _stop("worker status/candidate counts disagree")
    integrity = _parse_json(_read(output / "integrity.json", limit), "worker integrity")
    if set(integrity) != INTEGRITY_KEYS:
        _stop("worker integrity keyset mismatch")
    if (
        integrity["schema_version"] != WORKER_INTEGRITY_SCHEMA
        or integrity["worker_build_id"] != worker_build_id
        or integrity["query_count"] != len(query_ids)
        or integrity["candidate_count"] != len(rows)
        or integrity["declarations"] != DECLARATIONS
        or set(integrity["sandbox_checks"]) != SANDBOX_CHECK_KEYS
        or any(value is not True for value in integrity["sandbox_checks"].values())
        or set(integrity["durations_ns"]) != DURATION_KEYS
        or any(type(value) is not int or value < 0 for value in integrity["durations_ns"].values())
        or type(integrity["peak_rss_bytes"]) is not int
        or not 0 <= integrity["peak_rss_bytes"] <= limit
    ):
        _stop("worker integrity values mismatch")
    status_payload = b"".join(
        query_id.encode() + b"\0" + str(count).encode() + b"\n"
        for query_id, count in zip(query_ids, counts, strict=True)
    )
    candidate_payload = b"".join(
        row["query_id"].encode()
        + b"\0"
        + row["candidate_siret"].encode()
        + b"\0"
        + str(row["candidate_rank"]).encode()
        + b"\n"
        for row in rows
    )
    expected_scalars = {
        "minimum_pool_size": min(counts) if counts else 0,
        "maximum_pool_size": max(counts) if counts else 0,
        "under_ceiling_query_count": sum(value < 100 for value in counts),
        "empty_query_count": sum(value == 0 for value in counts),
        "candidate_payload_bytes": len(candidate_payload),
        "candidate_payload_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "status_payload_bytes": len(status_payload),
        "status_payload_sha256": hashlib.sha256(status_payload).hexdigest(),
    }
    if any(integrity[key] != value for key, value in expected_scalars.items()):
        _stop("worker integrity payload/count mismatch")
    return integrity


def _seal_tree(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        _path, current_fd = _openat_anchored(current_path, directory=True)
        os.close(current_fd)
        for name in directories:
            _path, child_fd = _openat_anchored(
                current_path / name, directory=True
            )
            os.close(child_fd)
        for name in files:
            child = current_path / name
            _path, descriptor = _openat_anchored(child, directory=False)
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _path, descriptor = _openat_anchored(directory, directory=True)
        try:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _path, root_fd = _openat_anchored(root, directory=True)
    try:
        os.fchmod(root_fd, 0o700)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


def _freeze_root(root: Path) -> None:
    _canonical, descriptor = _openat_anchored(root, directory=True)
    info = os.fstat(descriptor)
    if stat.S_IMODE(info.st_mode) not in {0o700, 0o555}:
        os.close(descriptor)
        _stop("published root mode invalid")
    identity = (info.st_dev, info.st_ino)
    try:
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino) != identity:
            _stop("published root identity changed")
    finally:
        os.close(descriptor)
    _fsync_dir(root.parent)


def _promote(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        _stop(f"immutable destination exists: {destination}")
    _seal_tree(source)
    _source_parent_path, source_parent_fd = _openat_anchored(
        source.parent, directory=True
    )
    _destination_parent_path, destination_parent_fd = _openat_anchored(
        destination.parent, directory=True
    )
    if os.fstat(source_parent_fd).st_dev != os.fstat(destination_parent_fd).st_dev:
        os.close(source_parent_fd)
        os.close(destination_parent_fd)
        _stop("publication must remain on one filesystem")
    try:
        _source_path, descriptor = _openat_anchored(source, directory=True)
    except BaseException:
        os.close(source_parent_fd)
        os.close(destination_parent_fd)
        raise
    identity = os.fstat(descriptor)
    primary: BaseException | None = None
    try:
        os.rename(
            source.name,
            destination.name,
            src_dir_fd=source_parent_fd,
            dst_dir_fd=destination_parent_fd,
        )
        _destination_path, destination_fd = _openat_anchored(
            destination, directory=True
        )
        after = os.fstat(destination_fd)
        os.close(destination_fd)
        if (after.st_dev, after.st_ino) != (identity.st_dev, identity.st_ino):
            _stop("publication rename identity mismatch")
    except BaseException as exc:
        primary = exc
    finally:
        cleanup: BaseException | None = None
        for action in (
            lambda: os.fchmod(descriptor, 0o555),
            lambda: os.fsync(descriptor),
            lambda: os.close(descriptor),
            lambda: os.close(source_parent_fd),
            lambda: os.close(destination_parent_fd),
        ):
            try:
                action()
            except BaseException as exc:
                cleanup = cleanup or exc
        for parent in dict.fromkeys((source.parent, destination.parent)):
            try:
                _fsync_dir(parent)
            except BaseException as exc:
                cleanup = cleanup or exc
        if primary is None and cleanup is not None:
            _stop(f"publication cleanup failed: {cleanup}")
    if primary is not None:
        raise primary
    _freeze_root(destination)


def _validate_runtime_publication(
    runtime: Path,
    build_id: str,
    plan: Mapping[str, Any],
    limit: int,
) -> dict[str, Any]:
    _freeze_root(runtime)
    if {p.name for p in runtime.iterdir()} != {
        "query_status.parquet",
        "candidates_top100.parquet",
        "integrity.json",
        "manifest.json",
    }:
        _stop("published runtime file-set mismatch")
    manifest = _parse_json(_read(runtime / "manifest.json", limit), "runtime manifest")
    if set(manifest) != RUNTIME_MANIFEST_KEYS:
        _stop("published runtime manifest keyset mismatch")
    if (
        manifest["schema_version"] != WORKER_MANIFEST_SCHEMA
        or manifest["worker_build_id"] != build_id
        or manifest["safe_input_build_id"] != plan["safe_input"]["build_id"]
        or manifest["strict_stores_build_id"] != plan["prerequisite"]["build_id"]
        or manifest["runtime"] != plan["runtime"]
        or manifest["declarations"] != DECLARATIONS
        or manifest["verdict"] != SEALED
    ):
        _stop("published runtime manifest values mismatch")
    if manifest.get("files") != _runtime_file_records(runtime, limit):
        _stop("published runtime manifest records mismatch")
    return manifest


def _validate_published(
    runtime: Path,
    audit: Path,
    build_id: str,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    plan_sha: str,
    lock_sha: str,
    limit: int,
) -> None:
    runtime_manifest = _validate_runtime_publication(runtime, build_id, plan, limit)
    _freeze_root(audit)
    if {p.name for p in audit.iterdir()} != {
        "open_ledger.parquet",
        "provenance.json",
        "manifest.json",
    }:
        _stop("published audit file-set mismatch")
    audit_manifest = _parse_json(_read(audit / "manifest.json", limit), "audit manifest")
    if (
        set(audit_manifest) != AUDIT_MANIFEST_KEYS
        or audit_manifest.get("schema_version") != WORKER_AUDIT_SCHEMA
        or audit_manifest.get("worker_build_id") != build_id
    ):
        _stop("published audit manifest mismatch")
    expected_audit = sorted(
        [_record(path, limit) for path in audit.iterdir() if path.name != "manifest.json"],
        key=lambda row: row["path"].encode(),
    )
    if audit_manifest.get("files") != expected_audit:
        _stop("published audit manifest records mismatch")
    provenance = _parse_json(
        _read(audit / "provenance.json", limit),
        "worker provenance",
    )
    ledger_payload = _read(audit / "open_ledger.parquet", limit)
    ledger = pq.read_table(pa.BufferReader(ledger_payload))
    if ledger.schema != LEDGER_SCHEMA or ledger.schema.metadata is not None:
        _stop("published audit ledger schema mismatch")
    ledger_rows = ledger.to_pylist()
    if any(
        row["size_before"] != row["size_after"]
        or row["sha256_before"] != row["sha256_after"]
        for row in ledger_rows
    ):
        _stop("published audit ledger records a mutation")
    if len({row["absolute_path"] for row in ledger_rows}) != len(ledger_rows):
        _stop("published audit ledger contains duplicate paths")
    expected_provenance = {
        "schema_version": WORKER_PROVENANCE_SCHEMA,
        "worker_build_id": build_id,
        "git_commit": lock["git_commit"],
        "parent_source_hashes": {
            key: lock["source_hashes"][key] for key in plan["parent_sources"]
        },
        "worker_source_hashes": {
            key: lock["source_hashes"][key] for key in plan["worker_sources"]
        },
        "lock_sha256": lock_sha,
        "plan_sha256": plan_sha,
        "runtime": plan["runtime"],
        "data_input_count": len(ledger_rows),
        "runtime_manifest_sha256": hashlib.sha256(
            canonical_json(runtime_manifest)
        ).hexdigest(),
        "declarations": DECLARATIONS,
    }
    if set(provenance) != PROVENANCE_KEYS or provenance != expected_provenance:
        _stop("published worker provenance mismatch")


def _path_commitment(path: Path) -> str:
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts:
        _stop("unsafe committed path")
    return hashlib.sha256((str(absolute) + "\n").encode("utf-8")).hexdigest()


def _parity_build_id(
    spec: Mapping[str, Any],
    run_spec_sha256: str,
) -> str:
    payload = {
        "schema_version": PARITY_BUILD_SCHEMA,
        "worker_build_id": spec["worker_build_id"],
        "worker_manifest_sha256": spec["worker_manifest_sha256"],
        "worker_file_hashes": spec["worker_file_hashes"],
        "parity_run_spec_sha256": run_spec_sha256,
        "parity_source_hashes": spec["parity_source_hashes"],
        "parity_profile_sha256": spec["parity_profile_sha256"],
        "lock_sha256": spec["lock_sha256"],
        "runtime": spec["runtime"],
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


def _build_parity_run_spec(
    *,
    repo: Path,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    runtime: Path,
    lock_sha256: str,
    limit: int,
) -> dict[str, Any]:
    """Build the controller's sanitized input from sealed parent commitments."""
    build_id = runtime.name
    manifest_path = runtime / "manifest.json"
    manifest = _validate_runtime_publication(runtime, build_id, plan, limit)
    manifest_snapshot = _snapshot(manifest_path, limit)
    file_paths = {
        name: str((runtime / name).absolute())
        for name in sorted(PARITY_WORKER_FILES)
    }
    file_hashes = {
        name: manifest["files"][name]["sha256"]
        for name in sorted(PARITY_WORKER_FILES)
    }
    historical = plan["historical_parity"]
    expected = {key: historical[key] for key in PARITY_EXPECTED_KEYS}
    safe_queries_path = Path(lock["input_paths"]["safe_queries_dev"]).absolute()
    safe_manifest_path = Path(
        lock["input_paths"]["safe_runtime_manifest"]
    ).absolute()
    parity_source = str(PARITY_SOURCE_PATH)
    parity_profile = str(PARITY_PROFILE_PATH)
    if (
        plan["parity_sources"] != [parity_source]
        or parity_source not in lock["source_hashes"]
        or parity_profile not in lock["source_hashes"]
    ):
        _stop("parity source/profile closure mismatch")
    sandbox = lock["sandbox"]
    spec = {
        "schema_version": PARITY_RUN_SPEC_SCHEMA,
        "worker_build_id": build_id,
        "worker_manifest_path": str(manifest_path.absolute()),
        "worker_manifest_sha256": manifest_snapshot["sha256"],
        "worker_file_paths": file_paths,
        "worker_file_hashes": file_hashes,
        "safe_input_build_id": plan["safe_input"]["build_id"],
        "safe_queries_path": str(safe_queries_path),
        "safe_queries_sha256": lock["input_hashes"]["safe_queries_dev"],
        "safe_manifest_path": str(safe_manifest_path),
        "safe_manifest_sha256": lock["input_hashes"]["safe_runtime_manifest"],
        "safe_query_id_payload_sha256": plan["safe_input"][
            "query_id_payload_sha256"
        ],
        "expected": expected,
        "git_commit": lock["git_commit"],
        "lock_sha256": lock_sha256,
        "parity_source_hashes": {
            parity_source: lock["source_hashes"][parity_source]
        },
        "parity_profile_sha256": lock["source_hashes"][parity_profile],
        "sandbox_executable_path": sandbox["executable"],
        "sandbox_executable_sha256": sandbox["executable_sha256"],
        "python_executable_path": sandbox["python_framework_app"],
        "python_executable_sha256": sandbox["python_framework_app_sha256"],
        "audit_root_path_sha256": _path_commitment(
            Path(plan["outputs"]["parity_audit_root"])
        ),
        "runtime": plan["runtime"],
        "temp_root": str(Path(plan["outputs"]["temp_root"]).absolute()),
        "max_rss_bytes": limit,
        "declarations": dict(PARITY_DECLARATIONS),
    }
    if (
        set(expected) != PARITY_EXPECTED_KEYS
        or set(file_paths) != PARITY_WORKER_FILES
        or set(file_hashes) != PARITY_WORKER_FILES
    ):
        _stop("parity run-spec closure mismatch")
    if any(
        _snapshot(Path(file_paths[name]), limit)["sha256"] != file_hashes[name]
        for name in PARITY_WORKER_FILES
    ):
        _stop("worker file changed while building parity run-spec")
    if _snapshot(manifest_path, limit) != manifest_snapshot:
        _stop("worker manifest changed while building parity run-spec")
    return spec


def _parity_sentinels(
    plan: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
    worker_build_id: str,
) -> dict[str, str]:
    sentinels = _forbidden_sentinels(
        plan["parity_controller"]["forbidden_files"]
    )
    stores = [
        row["absolute_path"]
        for row in gate_spec["allowed_read_files"]
        if type(row) is dict and row.get("role") == "lookup_database"
    ]
    if len(stores) != 1:
        _stop("strict store sentinel is not unique")
    sentinels["store"] = stores[0]
    sentinels["write"] = str(
        (
            Path(plan["outputs"]["temp_root"])
            / f".parity-write-denied-{worker_build_id}"
        ).absolute()
    )
    if (
        set(sentinels)
        != {"oracle", "oracle_audit", "historical", "model", "store", "write"}
        or len(set(sentinels.values())) != len(sentinels)
        or any(
            not Path(value).is_absolute() or ".." in Path(value).parts
            for value in sentinels.values()
        )
    ):
        _stop("invalid parity sentinel closure")
    if os.path.lexists(sentinels["write"]):
        _stop("parity write sentinel already exists")
    return sentinels


_ANCHORED_SOURCE_WRAPPER = (
    "import os,sys;"
    "source=sys.argv[1];fd=int(sys.argv[2]);"
    "payload=os.fdopen(os.dup(fd),'rb').read();"
    "sys.argv=[source,*sys.argv[3:]];"
    "scope={'__name__':'__main__','__file__':source,"
    "'__package__':None,'__cached__':None};"
    "exec(compile(payload,source,'exec'),scope,scope)"
)


def _anchor_private_runtime_directories(
    *,
    staging_root: Path,
    python_path: Path,
    library_path: Path,
    framework_root: Path,
) -> dict[Path, tuple[int, tuple[int, int, int]]]:
    expected_python = (staging_root / PRIVATE_PYTHON_RELATIVE).absolute()
    expected_library = (staging_root / PRIVATE_LIBRARY_RELATIVE).absolute()
    expected_runtime = (staging_root / "runtime").absolute()
    if (
        python_path != expected_python
        or library_path.absolute() != expected_library
        or framework_root.absolute() != expected_runtime
    ):
        _stop("private Python runtime layout mismatch")
    # macOS cannot execute this Mach-O launcher through /dev/fd.  The explicit
    # exec boundary is therefore the registered staging inode plus this fully
    # sealed 0555 ancestor chain; every inode remains open until child return.
    directories: list[Path] = [staging_root]
    cursor = expected_python.parent
    while True:
        directories.append(cursor)
        if cursor == expected_runtime:
            break
        if expected_runtime not in cursor.parents:
            _stop("private Python ancestor escaped runtime root")
        cursor = cursor.parent
    directories = list(dict.fromkeys(directories))
    anchors: dict[Path, tuple[int, tuple[int, int, int]]] = {}
    try:
        for path in reversed(directories):
            info = os.lstat(path)
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISDIR(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o555
            ):
                _stop("private Python ancestor is mutable or invalid")
            canonical, descriptor = _openat_anchored(path, directory=True)
            anchored = os.fstat(descriptor)
            identity = (
                anchored.st_dev,
                anchored.st_ino,
                stat.S_IMODE(anchored.st_mode),
            )
            if (
                canonical != path
                or identity
                != (info.st_dev, info.st_ino, 0o555)
            ):
                os.close(descriptor)
                _stop("private Python ancestor identity mismatch")
            anchors[path] = (descriptor, identity)
        return anchors
    except BaseException:
        for descriptor, _ in anchors.values():
            os.close(descriptor)
        raise


def _verify_private_runtime_directories(
    anchors: Mapping[Path, tuple[int, tuple[int, int, int]]],
) -> None:
    for path, (descriptor, expected) in anchors.items():
        anchored = os.fstat(descriptor)
        if (
            anchored.st_dev,
            anchored.st_ino,
            stat.S_IMODE(anchored.st_mode),
        ) != expected:
            _stop("anchored private Python ancestor changed")
        info = os.lstat(path)
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or (info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode))
            != expected
        ):
            _stop("private Python ancestor path was substituted")


def _invoke_parity_controller(
    *,
    repo: Path,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    spec: Mapping[str, Any],
    run_spec_path: Path,
    sentinels: Mapping[str, str],
    source_snapshot: Mapping[str, Any],
    python_framework_root: Path,
    python_library_path: Path,
    limit: int,
) -> tuple[Path, dict[str, Any]]:
    """Execute the pinned controller bytes and verify its published identity."""
    source_path = (repo / PARITY_SOURCE_PATH).absolute()
    source_fd = _open_anchored(source_path, source_snapshot, limit)
    anchored_before = _snapshot_fd(source_fd, limit)
    expected_run_spec = canonical_json(spec)
    run_spec_snapshot = _snapshot(run_spec_path, limit)
    run_spec_fd = _open_anchored(run_spec_path, run_spec_snapshot, limit)
    actual_run_spec = os.pread(
        run_spec_fd,
        run_spec_snapshot["size"],
        0,
    )
    if (
        actual_run_spec != expected_run_spec
        or run_spec_snapshot["sha256"]
        != hashlib.sha256(expected_run_spec).hexdigest()
    ):
        os.close(source_fd)
        os.close(run_spec_fd)
        _stop("sealed parity run-spec differs from canonical parent spec")
    expected_parity_id = _parity_build_id(
        spec,
        run_spec_snapshot["sha256"],
    )
    python_path = Path(spec["python_executable_path"]).absolute()
    python_snapshot = _snapshot(python_path, limit)
    library_snapshot = _snapshot(python_library_path, limit)
    directory_anchors = _anchor_private_runtime_directories(
        staging_root=run_spec_path.parent.absolute(),
        python_path=python_path,
        library_path=python_library_path,
        framework_root=python_framework_root,
    )
    if (
        python_snapshot["sha256"]
        != lock["sandbox"]["python_framework_app_sha256"]
        or library_snapshot["sha256"]
        != lock["sandbox"]["python_framework_library_sha256"]
        or python_snapshot["mode"] != 0o555
        or library_snapshot["mode"] != 0o444
    ):
        for descriptor, _ in directory_anchors.values():
            os.close(descriptor)
        os.close(source_fd)
        os.close(run_spec_fd)
        _stop("private Python runtime differs from locked originals")
    python_fd = _open_anchored(python_path, python_snapshot, limit)
    library_fd = _open_anchored(
        python_library_path,
        library_snapshot,
        limit,
    )
    command = [
        str(python_path),
        "-B",
        "-c",
        _ANCHORED_SOURCE_WRAPPER,
        str(source_path),
        str(source_fd),
        "--run-spec",
        f"/dev/fd/{run_spec_fd}",
        "--profile",
        str((repo / PARITY_PROFILE_PATH).absolute()),
        "--audit-root",
        str(Path(plan["outputs"]["parity_audit_root"]).absolute()),
        "--forbidden-oracle",
        sentinels["oracle"],
        "--forbidden-oracle-audit",
        sentinels["oracle_audit"],
        "--forbidden-historical",
        sentinels["historical"],
        "--forbidden-model",
        sentinels["model"],
        "--forbidden-store",
        sentinels["store"],
        "--write-sentinel",
        sentinels["write"],
        "--sandbox-executable",
        lock["sandbox"]["executable"],
        "--python-executable",
        str(python_path),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=run_spec_path.parent,
            env={
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "JOBLIB_MULTIPROCESSING": "0",
                "TMPDIR": str(run_spec_path.parent),
                "DYLD_FRAMEWORK_PATH": str(python_framework_root),
            },
            pass_fds=(source_fd, run_spec_fd),
            capture_output=True,
            text=True,
            check=False,
        )
        if _snapshot_fd(source_fd, limit) != anchored_before:
            _stop("anchored parity controller changed during execution")
        if _snapshot_fd(run_spec_fd, limit) != run_spec_snapshot:
            _stop("anchored parity run-spec changed during execution")
        if os.pread(run_spec_fd, run_spec_snapshot["size"], 0) != expected_run_spec:
            _stop("anchored parity run-spec bytes changed during execution")
        if _snapshot_fd(python_fd, limit) != python_snapshot:
            _stop("private Python launcher changed during execution")
        if _snapshot_fd(library_fd, limit) != library_snapshot:
            _stop("private Python library changed during execution")
        if (
            _snapshot(python_path, limit) != python_snapshot
            or _snapshot(python_library_path, limit) != library_snapshot
        ):
            _stop("private Python runtime path was substituted")
        _verify_private_runtime_directories(directory_anchors)
        validator_source_payload = os.pread(
            source_fd,
            anchored_before["size"],
            0,
        )
        if (
            len(validator_source_payload) != anchored_before["size"]
            or hashlib.sha256(validator_source_payload).hexdigest()
            != anchored_before["sha256"]
        ):
            _stop("anchored parity validator bytes mismatch")
    finally:
        for descriptor in (source_fd, run_spec_fd, python_fd, library_fd):
            os.close(descriptor)
        for descriptor, _ in directory_anchors.values():
            os.close(descriptor)
    if result.returncode != 0:
        _stop(
            f"parity controller failed rc={result.returncode}: "
            f"{result.stderr[-1000:]}"
        )
    if result.stderr:
        _stop("parity controller emitted unexpected stderr")
    response = _parse_json(result.stdout.encode("utf-8"), "parity controller result")
    if (
        set(response) != PARITY_RESULT_KEYS
        or response.get("verdict") != PARITY_GO
        or response.get("parity_build_id") != expected_parity_id
    ):
        _stop("parity controller returned a non-GO or malformed verdict")
    parity_root = (
        Path(plan["outputs"]["parity_audit_root"])
        / response["parity_build_id"]
    ).absolute()
    if Path(response["audit"]).absolute() != parity_root:
        _stop("parity controller audit identity mismatch")
    try:
        namespace: dict[str, Any] = {
            "__name__": "_v412_anchored_parity_validator",
            "__file__": str(source_path),
            "__package__": None,
            "__cached__": None,
        }
        exec(
            compile(validator_source_payload, str(source_path), "exec"),
            namespace,
            namespace,
        )
        parsed_spec = namespace["parse_json"](
            expected_run_spec,
            "parent parity run-spec",
        )
        if parsed_spec != spec:
            _stop("parent parity run-spec value mismatch")
        namespace["validate_run_spec"](
            parsed_spec,
            active_python_path=python_path,
        )
        report = namespace["validate_published_audit"](
            parity_root,
            parity_id=expected_parity_id,
            spec=parsed_spec,
            run_spec_sha256=run_spec_snapshot["sha256"],
            limit=limit,
        )
    except RetrievalRunStopped:
        raise
    except BaseException as exc:
        _stop(f"parent parity publication validation failed: {exc}")
    if (
        report.get("parity_build_id") != response["parity_build_id"]
        or report.get("worker_build_id") != spec["worker_build_id"]
        or report.get("verdict") != PARITY_GO
    ):
        _stop("fully validated parity report identity mismatch")
    if stat.S_IMODE(os.lstat(parity_root).st_mode) != 0o555:
        _stop("published parity root is not immutable")
    return parity_root, report


def _register_private(path: Path) -> None:
    if path in _ACTIVE_PRIVATE_ROOTS:
        _stop("private staging already registered")
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        _stop("private staging must be a real directory")
    descriptor_path, descriptor = _openat_anchored(path, directory=True)
    try:
        anchored = os.fstat(descriptor)
        if descriptor_path != path.absolute() or (
            anchored.st_dev,
            anchored.st_ino,
        ) != (info.st_dev, info.st_ino):
            _stop("private staging identity mismatch")
        _ACTIVE_PRIVATE_ROOTS[path] = (anchored.st_dev, anchored.st_ino)
    finally:
        os.close(descriptor)


def _clear_private_directory(descriptor: int) -> None:
    os.fchmod(descriptor, 0o700)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for name in os.listdir(descriptor):
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            _stop("symlink in private cleanup")
        if stat.S_ISDIR(info.st_mode):
            child_fd = os.open(name, flags, dir_fd=descriptor)
            try:
                child = os.fstat(child_fd)
                if (child.st_dev, child.st_ino) != (info.st_dev, info.st_ino):
                    _stop("private cleanup child identity mismatch")
                _clear_private_directory(child_fd)
                os.fchmod(child_fd, 0o700)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=descriptor)
        elif stat.S_ISREG(info.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                child = os.fstat(child_fd)
                if (child.st_dev, child.st_ino) != (info.st_dev, info.st_ino):
                    _stop("private cleanup file identity mismatch")
                os.fchmod(child_fd, 0o600)
            finally:
                os.close(child_fd)
            os.unlink(name, dir_fd=descriptor)
        else:
            _stop("unsupported entry in private cleanup")


def _remove_private(path: Path) -> None:
    expected = _ACTIVE_PRIVATE_ROOTS.get(path)
    if expected is None:
        _stop("refusing private cleanup")
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        _stop("registered private staging disappeared")
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or (info.st_dev, info.st_ino) != expected
    ):
        _stop("registered private staging was substituted")
    parent_path, parent_fd = _openat_anchored(path.parent, directory=True)
    descriptor_path, descriptor = _openat_anchored(path, directory=True)
    try:
        anchored = os.fstat(descriptor)
        if (
            parent_path != path.parent.absolute()
            or descriptor_path != path.absolute()
            or (anchored.st_dev, anchored.st_ino) != expected
        ):
            _stop("anchored private staging identity mismatch")
        _clear_private_directory(descriptor)
        os.fchmod(descriptor, 0o700)
        current = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != expected
        ):
            _stop("private staging changed before removal")
        os.rmdir(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(descriptor)
        os.close(parent_fd)
    del _ACTIVE_PRIVATE_ROOTS[path]


def _recover(
    output_root: Path,
    audit_root: Path,
    build_id: str,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    plan_sha: str,
    lock_sha: str,
    limit: int,
) -> tuple[Path, Path] | None:
    runtime = output_root / build_id
    pending = output_root / f".pending-{build_id}"
    audit = audit_root / build_id
    state = (os.path.lexists(runtime), os.path.lexists(pending), os.path.lexists(audit))
    if state == (False, False, False):
        return None
    if state == (True, False, True):
        _validate_published(
            runtime, audit, build_id, plan, lock, plan_sha, lock_sha, limit
        )
        return runtime, audit
    if state == (False, True, True):
        _validate_published(
            pending, audit, build_id, plan, lock, plan_sha, lock_sha, limit
        )
        _promote(pending, runtime)
        _validate_published(
            runtime, audit, build_id, plan, lock, plan_sha, lock_sha, limit
        )
        return runtime, audit
    if state == (False, True, False):
        _validate_runtime_publication(pending, build_id, plan, limit)
        if pending.parent != output_root or pending.name != f".pending-{build_id}":
            _stop("pending cleanup target mismatch")
        for child in pending.rglob("*"):
            if child.is_symlink():
                _stop("symlink in pending cleanup")
            if child.is_file():
                os.chmod(child, 0o600)
        for child in sorted(
            (item for item in pending.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            os.chmod(child, 0o700)
        os.chmod(pending, 0o700)
        shutil.rmtree(pending)
        _fsync_dir(output_root)
        return None
    _stop("inconsistent publication recovery state")


def _worker_identity(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    safe_manifest_sha: str,
) -> tuple[str, str]:
    policy_sha = hashlib.sha256(canonical_json(_worker_policy(plan))).hexdigest()
    identity = {
        "schema_version": "sireto-v4.12-unit-retrieval-worker-build-identity-1",
        "worker_policy_sha256": policy_sha,
        "worker_lock_projection_sha256": lock["worker_lock_projection_sha256"],
        "parent_runner_sha256": lock["source_hashes"][
            "scripts/run_v412_unit_retrieval.py"
        ],
        "worker_source_hashes": {
            key: lock["source_hashes"][key] for key in plan["worker_sources"]
        },
        "safe_input_build_id": plan["safe_input"]["build_id"],
        "safe_runtime_manifest_sha256": safe_manifest_sha,
        "safe_queries_dev_sha256": plan["safe_input"]["queries_dev_sha256"],
        "strict_stores_build_id": plan["prerequisite"]["build_id"],
        "strict_stores_manifest_sha256": plan["prerequisite"][
            "certification_manifest_sha256"
        ],
        "retrieval": plan["retrieval"],
        "tfidf_cache": plan["tfidf_cache"],
        "runtime": plan["runtime"],
    }
    return hashlib.sha256(canonical_json(identity)).hexdigest(), policy_sha


def _worker_run_spec(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
    *,
    policy_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SPEC_SCHEMA,
        "safe_input_build_id": plan["safe_input"]["build_id"],
        "safe_runtime_manifest_sha256": plan["safe_input"][
            "runtime_manifest_sha256"
        ],
        "safe_queries_dev_sha256": plan["safe_input"]["queries_dev_sha256"],
        "query_count": plan["safe_input"]["query_count"],
        "query_id_payload_sha256": plan["safe_input"]["query_id_payload_sha256"],
        "routing_payload_sha256": plan["expected_routing"]["payload_sha256"],
        "worker_policy_sha256": policy_sha,
        "worker_lock_projection_sha256": lock["worker_lock_projection_sha256"],
        "parent_runner_sha256": lock["source_hashes"][
            "scripts/run_v412_unit_retrieval.py"
        ],
        "worker_source_hashes": {
            key: lock["source_hashes"][key] for key in plan["worker_sources"]
        },
        "strict_stores_build_id": plan["prerequisite"]["build_id"],
        "strict_stores_manifest_sha256": plan["prerequisite"][
            "certification_manifest_sha256"
        ],
        "retrieval": plan["retrieval"],
        "tfidf_cache": plan["tfidf_cache"],
        "runtime": plan["runtime"],
        "max_rss_bytes": plan["max_rss_bytes"],
        "gate_a_run_spec": dict(gate_spec),
        "declarations": dict(DECLARATIONS),
    }


def _forbidden_sentinels(paths: Sequence[str]) -> dict[str, str]:
    predicates = {
        "oracle": lambda value: "/oracles/" in value,
        "oracle_audit": lambda value: "/audits/" in value,
        "historical": lambda value: "/datasets/" in value,
        "model": lambda value: value.endswith(".pkl"),
    }
    result: dict[str, str] = {}
    for role, predicate in predicates.items():
        matches = [path for path in paths if type(path) is str and predicate(path)]
        if len(matches) != 1:
            _stop(f"forbidden {role} sentinel is not unique")
        result[role] = matches[0]
    if len(set(result.values())) != len(result):
        _stop("forbidden sentinels overlap")
    return result


def _validate_worker_stdout(payload: str, worker_build_id: str) -> dict[str, Any]:
    try:
        status = _parse_json(payload.encode(), "worker stdout")
    except RetrievalRunStopped:
        _stop("worker stdout is not one JSON status")
    if (
        set(status)
        != {"verdict", "worker_build_id", "query_count", "candidate_count"}
        or status.get("verdict") != SEALED
        or status.get("worker_build_id") != worker_build_id
        or type(status.get("query_count")) is not int
        or status["query_count"] < 0
        or type(status.get("candidate_count")) is not int
        or status["candidate_count"] < 0
    ):
        _stop("worker stdout contract mismatch")
    return status


def run(plan_path: Path, lock_path: Path) -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    if plan_path.absolute() != (repo / PLAN_PATH).absolute():
        _stop("only canonical plan may execute")
    if lock_path.absolute() != (repo / LOCK_PATH).absolute():
        _stop("only canonical lock may execute")
    if not sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        _stop("runner requires python -B and PYTHONDONTWRITEBYTECODE=1")
    plan_bytes = _read(plan_path, 8 * 1024**3)
    plan = _parse_json(plan_bytes, "plan")
    validate_plan(plan)
    limit = plan["max_rss_bytes"]
    lock_bytes = _read(lock_path, limit)
    lock = _parse_json(lock_bytes, "lock")
    source_snaps = validate_lock(
        plan, lock, repo, hashlib.sha256(plan_bytes).hexdigest(), limit
    )
    lock_snap = _snapshot(lock_path, limit)
    gate_spec, descriptor = _verify_gate_a(plan, limit)
    queries_path, safe_manifest, queries = _verify_safe_input(plan, limit)
    routes = _derive_routes(queries, gate_spec, plan["expected_routing"])
    safe_manifest_sha = plan["safe_input"]["runtime_manifest_sha256"]
    worker_build_id, policy_sha = _worker_identity(plan, lock, safe_manifest_sha)

    output_root = Path(plan["outputs"]["runtime_root"])
    audit_root = Path(plan["outputs"]["worker_audit_root"])
    temp_root = Path(plan["outputs"]["temp_root"])
    for root in (output_root, audit_root, temp_root):
        _ensure_dir(root)
    recovered = _recover(
        output_root,
        audit_root,
        worker_build_id,
        plan,
        lock,
        hashlib.sha256(plan_bytes).hexdigest(),
        hashlib.sha256(lock_bytes).hexdigest(),
        limit,
    )
    if recovered is not None:
        return recovered

    run_root = (temp_root / f".run-{uuid.uuid4().hex}").absolute()
    run_root.mkdir(mode=0o700)
    _register_private(run_root)
    output = run_root / "output"
    tmp = run_root / "tmp"
    package = run_root / "xgb_matcher"
    output.mkdir(mode=0o700)
    tmp.mkdir(mode=0o700)
    package.mkdir(mode=0o700)
    private_init = package / "__init__.py"
    _write_exclusive(private_init, b"", 0o444)

    strict_source = repo / "src/xgb_matcher/v412_strict_stores.py"
    engine_source = repo / "src/xgb_matcher/v412_unit_retrieval.py"
    strict_payload = _read(strict_source, limit)
    engine_payload = _read(engine_source, limit)
    for source, payload in (
        (strict_source, strict_payload),
        (engine_source, engine_payload),
    ):
        relative = str(source.relative_to(repo))
        if hashlib.sha256(payload).hexdigest() != lock["source_hashes"][relative]:
            _stop("private worker source differs from lock")
    private_strict = package / "v412_strict_stores.py"
    private_engine = package / "v412_unit_retrieval.py"
    _write_exclusive(private_strict, strict_payload)
    _write_exclusive(private_engine, engine_payload)
    os.chmod(package, 0o555)

    descriptor_path = run_root / "lookup_descriptor.json"
    run_spec_path = run_root / "run_spec.json"
    profile_path = run_root / "sandbox_profile_effective.sb"
    _write_json(descriptor_path, descriptor)
    allowed = gate_spec["allowed_read_files"]
    run_spec = _worker_run_spec(
        plan,
        lock,
        gate_spec,
        policy_sha=policy_sha,
    )
    if run_spec["query_count"] != len(routes):
        _stop("worker run-spec query count differs from routed queries")
    _write_json(run_spec_path, run_spec)

    sandbox = lock["sandbox"]
    private_python, framework_root = _copy_private_python(run_root, sandbox, limit)
    template_path = repo / "config/v4_12_unit_retrieval.sb"
    template_payload = _read(template_path, limit)
    template_relative = str(template_path.relative_to(repo))
    if hashlib.sha256(template_payload).hexdigest() != lock["source_hashes"][
        template_relative
    ]:
        _stop("sandbox profile template changed after lock validation")
    template = template_payload.decode("utf-8")
    allowed_paths = [Path(row["absolute_path"]) for row in allowed]
    allowed_paths.append(private_init)
    profile = render_profile(
        template,
        allowed_files=allowed_paths,
        forbidden_roots=[Path(path) for path in plan["forbidden_worker_roots"]],
        system_roots=[Path(path) for path in sandbox["system_read_roots"]],
        devices=[Path(path) for path in sandbox["device_read_paths"]],
        metadata_extra=[run_root, output, tmp],
    )
    _write_exclusive(profile_path, profile.encode("utf-8"))

    controlled = {
        "run_spec": run_spec_path,
        "descriptor": descriptor_path,
        "queries": queries_path,
        "strict": private_strict,
        "engine": private_engine,
        "profile": profile_path,
    }
    snapshots = {name: _snapshot(path, limit) for name, path in controlled.items()}
    ledger_before: dict[Path, tuple[str, str, dict[str, Any]]] = {}

    def register_input(
        path: Path,
        role: str,
        projection: str,
        before: Mapping[str, Any],
    ) -> None:
        canonical = _canonical_file(path)
        value = dict(before)
        previous = ledger_before.get(canonical)
        if previous is not None:
            if previous[2] != value:
                _stop(f"conflicting input snapshots: {canonical}")
            return
        ledger_before[canonical] = (role, projection, value)

    for relative in sorted(lock["source_hashes"]):
        path = _canonical_file(repo / relative)
        register_input(
            path,
            f"source:{relative}",
            "GIT_BLOB_EXACT",
            source_snaps[path],
        )
    register_input(lock_path, "execution_lock", "JSON_EXACT", lock_snap)
    for role, raw in lock["input_paths"].items():
        path = _canonical_file(Path(raw))
        register_input(path, role, "LOCKED_INPUT_EXACT", source_snaps[path])
    for row in allowed:
        path = _canonical_file(Path(row["absolute_path"]))
        before = _snapshot(path, limit)
        if (
            before["sha256"] != row["sha256"]
            or before["size"] != row["size_bytes"]
        ):
            _stop(f"Gate A allowed input changed before worker: {path}")
        register_input(path, row["role"], "STRICT_WORKER_READ", before)
    for name, path in controlled.items():
        register_input(
            path,
            f"worker_control:{name}",
            "PRIVATE_ANCHORED_FD",
            snapshots[name],
        )
    anchored: dict[str, int] = {}
    try:
        for name, path in controlled.items():
            anchored[name] = _open_anchored(path, snapshots[name], limit)
        fd = {name: f"/dev/fd/{value}" for name, value in anchored.items()}
        sentinels = _forbidden_sentinels(plan["forbidden_worker_files"])
        command = [
            sandbox["executable"],
            "-D",
            f"RUN_ROOT={run_root}",
            "-D",
            f"RUN_SPEC={fd['run_spec']}",
            "-D",
            f"LOOKUP_DESCRIPTOR={fd['descriptor']}",
            "-D",
            f"QUERIES={fd['queries']}",
            "-D",
            f"STRICT_SOURCE={private_strict}",
            "-D",
            f"ENGINE_SOURCE={private_engine}",
            "-D",
            f"RUN_OUTPUT={output}",
            "-D",
            f"RUN_TMP={tmp}",
            "-D",
            f"PYTHON_EXECUTABLE={private_python}",
            "-D",
            f"PYTHON_FRAMEWORK_ROOT={framework_root}",
            "-p",
            profile,
            str(private_python),
            "-B",
            "-m",
            "xgb_matcher.v412_unit_retrieval",
            "--run-spec",
            fd["run_spec"],
            "--lookup-descriptor",
            fd["descriptor"],
            "--queries",
            fd["queries"],
            "--output-dir",
            "output",
            "--worker-build-id",
            worker_build_id,
            "--forbidden-oracle",
            sentinels["oracle"],
            "--forbidden-oracle-audit",
            sentinels["oracle_audit"],
            "--forbidden-historical",
            sentinels["historical"],
            "--forbidden-model",
            sentinels["model"],
        ]
        result = subprocess.run(
            command,
            cwd=run_root,
            env={
                "PYTHONDONTWRITEBYTECODE": "1",
                "JOBLIB_MULTIPROCESSING": "0",
                "TMPDIR": str(tmp),
                "DYLD_FRAMEWORK_PATH": str(framework_root),
            },
            pass_fds=tuple(anchored.values()),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        anchored_error: BaseException | None = None
        for name, descriptor in anchored.items():
            try:
                if _snapshot_fd(descriptor, limit) != snapshots[name]:
                    _stop(f"anchored worker input changed during execution: {name}")
            except BaseException as exc:
                anchored_error = anchored_error or exc
            finally:
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    anchored_error = anchored_error or exc
        if anchored_error is not None:
            raise anchored_error
    ledger_rows = []
    for path, (role, projection, before) in ledger_before.items():
        after = _snapshot(path, limit)
        if after != before:
            _stop(f"input mutated during worker: {path}")
        ledger_rows.append(
            {
                "role": role,
                "absolute_path": str(path),
                "projection": projection,
                "size_before": before["size"],
                "sha256_before": before["sha256"],
                "size_after": after["size"],
                "sha256_after": after["sha256"],
            }
        )
    ledger_rows.sort(
        key=lambda row: (row["role"].encode(), row["absolute_path"].encode())
    )
    if result.returncode != 0:
        _stop(f"worker failed rc={result.returncode}: {result.stderr[-1000:]}")
    status = _validate_worker_stdout(result.stdout, worker_build_id)

    integrity = _validate_outputs(
        output, worker_build_id, queries["query_id"].to_pylist(), limit
    )
    if status != {
        "verdict": SEALED,
        "worker_build_id": worker_build_id,
        "query_count": integrity["query_count"],
        "candidate_count": integrity["candidate_count"],
    }:
        _stop("worker stdout/output mismatch")
    manifest = {
        "schema_version": WORKER_MANIFEST_SCHEMA,
        "worker_build_id": worker_build_id,
        "safe_input_build_id": plan["safe_input"]["build_id"],
        "strict_stores_build_id": plan["prerequisite"]["build_id"],
        "files": _runtime_file_records(output, limit),
        "runtime": plan["runtime"],
        "declarations": DECLARATIONS,
        "verdict": SEALED,
    }
    _write_json(output / "manifest.json", manifest)

    ledger_table = pa.Table.from_pylist(ledger_rows, schema=LEDGER_SCHEMA)

    audit_stage = run_root / "audit"
    audit_stage.mkdir(mode=0o700)
    pq.write_table(ledger_table, audit_stage / "open_ledger.parquet", compression="zstd")
    provenance = {
        "schema_version": WORKER_PROVENANCE_SCHEMA,
        "worker_build_id": worker_build_id,
        "git_commit": lock["git_commit"],
        "parent_source_hashes": {
            key: lock["source_hashes"][key] for key in plan["parent_sources"]
        },
        "worker_source_hashes": {
            key: lock["source_hashes"][key] for key in plan["worker_sources"]
        },
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "runtime": plan["runtime"],
        "data_input_count": len(ledger_rows),
        "runtime_manifest_sha256": hashlib.sha256(
            canonical_json(manifest)
        ).hexdigest(),
        "declarations": DECLARATIONS,
    }
    _write_json(audit_stage / "provenance.json", provenance)
    audit_manifest = {
        "schema_version": WORKER_AUDIT_SCHEMA,
        "worker_build_id": worker_build_id,
        "files": sorted(
            [_record(audit_stage / name, limit) for name in (
                "open_ledger.parquet", "provenance.json"
            )],
            key=lambda row: row["path"].encode(),
        ),
    }
    _write_json(audit_stage / "manifest.json", audit_manifest)

    pending = output_root / f".pending-{worker_build_id}"
    final_runtime = output_root / worker_build_id
    final_audit = audit_root / worker_build_id
    _promote(output, pending)
    _promote(audit_stage, final_audit)
    _promote(pending, final_runtime)
    _validate_published(
        final_runtime,
        final_audit,
        worker_build_id,
        plan,
        lock,
        hashlib.sha256(plan_bytes).hexdigest(),
        hashlib.sha256(lock_bytes).hexdigest(),
        limit,
    )
    _remove_private(run_root)
    return final_runtime, final_audit


def run_end_to_end(
    plan_path: Path,
    lock_path: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Run/recover the worker, then require sealed aggregate parity."""
    repo = Path(__file__).resolve().parents[1]
    if plan_path.absolute() != (repo / PLAN_PATH).absolute():
        _stop("only canonical plan may execute")
    if lock_path.absolute() != (repo / LOCK_PATH).absolute():
        _stop("only canonical lock may execute")
    plan_snapshot = _snapshot(plan_path, 8 * 1024**3)
    lock_snapshot = _snapshot(lock_path, 8 * 1024**3)
    plan_bytes = _read(plan_path, 8 * 1024**3)
    lock_bytes = _read(lock_path, 8 * 1024**3)
    plan = _parse_json(plan_bytes, "plan")
    lock = _parse_json(lock_bytes, "lock")
    validate_plan(plan)
    limit = plan["max_rss_bytes"]
    source_snapshots = validate_lock(
        plan,
        lock,
        repo,
        hashlib.sha256(plan_bytes).hexdigest(),
        limit,
    )
    runtime, worker_audit = run(plan_path, lock_path)
    if (
        _snapshot(plan_path, limit) != plan_snapshot
        or _snapshot(lock_path, limit) != lock_snapshot
    ):
        _stop("plan or lock changed during worker execution")
    build_id = runtime.name
    expected_build_id, _ = _worker_identity(
        plan,
        lock,
        plan["safe_input"]["runtime_manifest_sha256"],
    )
    if build_id != expected_build_id:
        _stop("worker returned an unexpected build identity")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    _validate_published(
        runtime,
        worker_audit,
        build_id,
        plan,
        lock,
        plan_sha,
        lock_sha,
        limit,
    )
    gate_spec, _ = _verify_gate_a(plan, limit)
    staging_parent = Path(plan["outputs"]["temp_root"])
    _ensure_dir(staging_parent)
    staging = Path(
        tempfile.mkdtemp(prefix=".run-parity-parent-", dir=staging_parent)
    ).absolute()
    _register_private(staging)
    os.chmod(staging, 0o700)
    try:
        private_python, framework_root = _copy_private_python(
            staging,
            lock["sandbox"],
            limit,
        )
        private_library = (
            staging
            / "runtime/Python.framework/Versions/3.14/Python"
        )
        spec = _build_parity_run_spec(
            repo=repo,
            plan=plan,
            lock=lock,
            runtime=runtime,
            lock_sha256=lock_sha,
            limit=limit,
        )
        spec["python_executable_path"] = str(private_python.absolute())
        run_spec_path = staging / "parity_run_spec.json"
        _write_json(run_spec_path, spec)
        _fsync_dir(staging)
        os.chmod(staging, 0o555)
        _fsync_dir(staging.parent)
        sentinels = _parity_sentinels(plan, gate_spec, build_id)
        parity_source = (repo / PARITY_SOURCE_PATH).absolute()
        expected_source = source_snapshots.get(parity_source)
        if expected_source is None:
            _stop("locked parity controller snapshot missing")
        parity_audit, report = _invoke_parity_controller(
            repo=repo,
            plan=plan,
            lock=lock,
            spec=spec,
            run_spec_path=run_spec_path,
            sentinels=sentinels,
            source_snapshot=expected_source,
            python_framework_root=framework_root,
            python_library_path=private_library,
            limit=limit,
        )
        _validate_published(
            runtime,
            worker_audit,
            build_id,
            plan,
            lock,
            plan_sha,
            lock_sha,
            limit,
        )
        if (
            _snapshot(plan_path, limit) != plan_snapshot
            or _snapshot(lock_path, limit) != lock_snapshot
        ):
            _stop("plan or lock changed during parity execution")
        for path, before in source_snapshots.items():
            if _snapshot(path, limit) != before:
                _stop(f"locked source/input changed during parity: {path}")
        return runtime, worker_audit, parity_audit, report
    finally:
        _remove_private(staging)


def smoke() -> None:
    """Synthetic-only smoke: profile closure and immutable publication."""
    repo = Path(__file__).resolve().parents[1]
    template = (repo / "config/v4_12_unit_retrieval.sb").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="v412-unit-retrieval-smoke-") as raw:
        root = Path(raw).resolve()
        allowed = root / "allowed"
        allowed.write_bytes(b"ok")
        forbidden = root / "oracles"
        forbidden.mkdir()
        rendered = render_profile(
            template,
            allowed_files=[allowed],
            forbidden_roots=[forbidden],
            system_roots=[Path("/System"), Path("/usr"), Path("/opt/homebrew")],
            devices=[Path("/dev/null"), Path("/dev/urandom"), Path("/dev/fd")],
            metadata_extra=[root],
        )
        if str(allowed) not in rendered or str(forbidden) not in rendered or "@@" in rendered:
            _stop("synthetic sandbox profile smoke failed")
        source = root / "source"
        destination_parent = root / "published"
        source.mkdir()
        destination_parent.mkdir()
        (source / "value").write_bytes(b"value")
        destination = destination_parent / "build"
        _promote(source, destination)
        if stat.S_IMODE(os.lstat(destination).st_mode) != 0o555:
            _stop("synthetic publication smoke failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.smoke:
            smoke()
            print(json.dumps({"verdict": "SMOKE_OK"}, sort_keys=True))
            return 0
        if args.plan is None or args.lock is None:
            _stop("--plan and --lock are required")
        runtime, audit, parity_audit, parity_report = run_end_to_end(
            args.plan,
            args.lock,
        )
        print(
            json.dumps(
                {
                    "verdict": PARITY_GO,
                    "worker_build_id": runtime.name,
                    "runtime": str(runtime),
                    "worker_audit": str(audit),
                    "parity_build_id": parity_report["parity_build_id"],
                    "parity_audit": str(parity_audit),
                },
                sort_keys=True,
            )
        )
        return 0
    except RetrievalRunStopped as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
