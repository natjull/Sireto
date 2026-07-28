#!/usr/bin/env python3
"""Certify the V4.12 strict data stores inside the mandatory macOS sandbox.

The trusted parent inventories and seals inputs, creates the minimal child
specification, launches the standalone probe, and publishes only aggregate
proofs.  It never imports matching code and never opens an oracle, label,
historical candidate, challenge, holdout, final result, scene, or model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import functools
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


STOP = "STOP_V412_STRICT_STORES"
GO = "GO_V412_STRICT_STORES_SANDBOX"
CANONICAL_PLAN = Path("config/v4_12_strict_stores_plan.json")
CANONICAL_LOCK = Path("config/v4_12_strict_stores_execution_lock.json")
LOCK_SCHEMA = "sireto-v4.12-strict-stores-execution-lock-1"
LOCK_PURPOSE = "V4.12_STRICT_STORES_SANDBOX"
LOCK_VERDICT = "GO_CODE_V412_STRICT_STORES"
PROFILE_MARKERS = {
    "@@SYSTEM_READ_RULES@@",
    "@@DEVICE_READ_LITERALS@@",
    "@@CODE_DATA_READ_LITERALS@@",
    "@@ANCESTOR_METADATA_LITERALS@@",
    "@@EXPLICIT_DENY_RULES@@",
}
PYTHON_FRAMEWORK_LIBRARY = Path(
    "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/"
    "Python.framework/Versions/3.14/Python"
)
PYTHON_FRAMEWORK_LIBRARY_SHA256 = (
    "e5728c35bdc26dee85e45b3fb94780afc1c9f97ced6b0af64d54e4eab3422e0a"
)
DECLARATIONS = {
    "labels_opened": False,
    "oracle_opened": False,
    "historical_candidates_opened": False,
    "models_opened": False,
    "network_used": False,
    "writes_outside_staging": False,
    "cache_rebuild_attempted": False,
}
LOCK_KEYS = {
    "schema_version",
    "purpose",
    "audit_verdict",
    "git_commit",
    "source_hashes",
    "input_paths",
    "input_hashes",
    "expected_routing",
    "expected_partition_subset_sha256",
    "expected_cache_subset_sha256",
    "runtime",
    "output_root",
    "audit_output_root",
    "temp_root",
    "max_rss_bytes",
}
PLAN_KEYS = {
    "schema_version",
    "sources",
    "execution_lock_path",
    "safe_input",
    "partitions",
    "cache",
    "lookup",
    "routing",
    "sandbox",
    "runtime",
    "output_root",
    "audit_output_root",
    "temp_root",
    "max_rss_bytes",
}
PLAN_NESTED_KEYS = {
    "safe_input": {
        "build_id", "root", "runtime_manifest_path", "runtime_manifest_sha256",
        "queries_all_path", "queries_all_sha256", "queries_dev_path",
        "queries_dev_sha256", "partition_inventory_path",
        "partition_inventory_sha256", "tfidf_inventory_path",
        "tfidf_inventory_sha256", "integrity_path", "integrity_sha256",
    },
    "partitions": {
        "root", "expected_full_inventory_sha256", "expected_subset_file_count",
        "expected_subset_size_bytes", "expected_subset_row_count",
        "expected_subset_logical_sha256",
    },
    "cache": {
        "root", "namespace", "sidecar_schema_version",
        "expected_full_inventory_sha256", "expected_subset_key_count",
        "expected_subset_file_count", "expected_subset_size_bytes",
        "expected_subset_logical_sha256", "expected_aligned_row_count",
    },
    "lookup": {
        "database_path", "database_sha256", "database_size_bytes",
        "manifest_path", "manifest_sha256", "integrity_path",
        "integrity_sha256", "timing_path", "timing_sha256",
        "sample_max_sirets",
    },
    "routing": {
        "query_count", "insee_query_count", "cp_query_count",
        "distinct_key_count", "missing_key_count", "payload_bytes",
        "payload_sha256",
    },
    "sandbox": {
        "executable", "executable_sha256", "profile_path",
        "python_framework_app", "python_framework_app_sha256",
        "python_framework_library", "python_framework_library_sha256",
        "git_executable", "git_executable_sha256",
        "system_read_roots", "device_read_literals", "device_read_subpaths",
        "network_allowed",
        "forbidden_oracle_manifest", "forbidden_audit_manifest",
    },
    "runtime": {
        "python", "numpy", "pandas", "pyarrow", "scikit_learn", "scipy",
        "joblib", "duckdb", "machine", "platform",
    },
}
SAFE_FILES = {
    "runtime_manifest.json",
    "queries_all.parquet",
    "queries_dev.parquet",
    "partition_inventory.parquet",
    "tfidf_inventory.parquet",
    "integrity.json",
}
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
QUERY_COLUMNS = [
    "query_id",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
]
PARTITION_COLUMNS = ["relative_path", "size_bytes", "sha256"]
CACHE_COLUMNS = [
    "partition_key",
    "pickle_relative_path",
    "pickle_size_bytes",
    "pickle_sha256",
    "sidecar_relative_path",
    "sidecar_size_bytes",
    "sidecar_sha256",
]
PROBE_KEYS = {
    "schema_version",
    "build_id",
    "query_count",
    "distinct_key_count",
    "partition_verified_count",
    "partition_raw_row_count",
    "cache_verified_count",
    "aligned_pool_row_count",
    "cache_miss_count",
    "rebuild_count",
    "write_count",
    "lookup_sample_count",
    "lookup_missing_count",
    "lookup_extra_count",
    "sandbox_checks",
    "peak_rss_bytes",
    "durations_ns",
    "declarations",
}
_ACTIVE_RUN_ROOTS: set[Path] = set()


class CertificationStopped(RuntimeError):
    pass


def _stop(message: str) -> None:
    raise CertificationStopped(f"{STOP}: {message}")


def _declarations_exact(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(DECLARATIONS)
        and all(value[key] is False for key in DECLARATIONS)
    )


def _remove_private_run_root(path: Path) -> None:
    if not path.name.startswith(".run-") or path not in _ACTIVE_RUN_ROOTS:
        _stop(f"refusing cleanup outside private run naming: {path}")
    if not os.path.lexists(path):
        return
    if stat.S_ISDIR(os.lstat(path).st_mode):
        for current, directories, files in os.walk(
            path, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            if stat.S_ISLNK(os.lstat(current_path).st_mode):
                _stop(f"symlink substituted into private run: {current_path}")
            os.chmod(current_path, 0o700)
            for name in directories:
                child = current_path / name
                if stat.S_ISLNK(os.lstat(child).st_mode):
                    _stop(f"symlink substituted into private run: {child}")
                os.chmod(child, 0o700)
            for name in files:
                child = current_path / name
                if stat.S_ISLNK(os.lstat(child).st_mode):
                    _stop(f"symlink substituted into private run: {child}")
                os.chmod(child, 0o600)
        shutil.rmtree(path)
    else:
        os.unlink(path)


def _cleanup_private_runs(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        inherited = set(_ACTIVE_RUN_ROOTS)
        try:
            return function(*args, **kwargs)
        finally:
            for path in sorted(_ACTIVE_RUN_ROOTS - inherited):
                _remove_private_run_root(path)
                _ACTIVE_RUN_ROOTS.discard(path)
    return wrapped


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _stop(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _parse_json_bytes(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: _stop(f"non-finite JSON value: {token}"),
        )
    except CertificationStopped:
        raise
    except Exception as exc:
        _stop(f"invalid JSON {name}: {exc}")
    if not isinstance(value, dict):
        _stop(f"{name} must contain an object")
    return value


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _rss(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        _stop("invalid RSS limit")
    if _rss_bytes() > limit:
        _stop(f"RSS limit exceeded: {_rss_bytes()} > {limit}")


def _components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            _stop(f"path component absent: {current}")
        if stat.S_ISLNK(mode):
            _stop(f"symlink component forbidden: {current}")


def _regular(path: Path) -> os.stat_result:
    _components(path)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        _stop(f"regular file absent: {path}")
    if not stat.S_ISREG(info.st_mode):
        _stop(f"regular file required: {path}")
    return info


def _directory(path: Path) -> os.stat_result:
    _components(path)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode):
        _stop(f"directory required: {path}")
    return info


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _directory(path)


def _canonical_existing_file(path: Path) -> Path:
    absolute = path.absolute()
    _regular(absolute)
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        _stop(f"non-canonical or symlinked file path: {path}")
    return resolved


def _canonical_existing_dir(path: Path) -> Path:
    absolute = path.absolute()
    _directory(absolute)
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        _stop(f"non-canonical or symlinked directory path: {path}")
    return resolved


def _secure_file_read(
    path: Path, limit: int | None = None, capture: bool = False
) -> tuple[dict[str, Any], bytes | None]:
    before = _regular(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    chunks = [] if capture else None
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _stop(f"secure open failed for {path}: {exc}")
    try:
        opened = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
        if any(getattr(before, name) != getattr(opened, name) for name in identity):
            _stop(f"file changed between lstat and open: {path}")
        if not stat.S_ISREG(opened.st_mode):
            _stop(f"opened object is not regular: {path}")
        if capture and opened.st_size > 512 * 1024 * 1024:
            _stop(f"captured control/projection file is unexpectedly large: {path}")
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
            if limit is not None:
                _rss(limit)
        after = os.fstat(descriptor)
        if any(getattr(opened, name) != getattr(after, name) for name in identity):
            _stop(f"file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    snapshot = {
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode": stat.S_IMODE(after.st_mode),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }
    return snapshot, None if chunks is None else b"".join(chunks)


def _secure_file_snapshot(path: Path, limit: int | None = None) -> dict[str, Any]:
    return _secure_file_read(path, limit)[0]


def _secure_read_bytes(path: Path, limit: int | None = None) -> bytes:
    payload = _secure_file_read(path, limit, capture=True)[1]
    assert payload is not None
    return payload


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


def _open_anchored_fd(path: Path, expected: Mapping[str, Any], limit: int) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    if _snapshot_fd(descriptor, limit) != dict(expected):
        os.close(descriptor)
        _stop(f"anchored descriptor differs from sealed file: {path}")
    return descriptor


def load_json(path: Path) -> dict[str, Any]:
    return _parse_json_bytes(_secure_read_bytes(path), path.name)


def sha256_file(path: Path, limit: int | None = None) -> str:
    return _secure_file_snapshot(path, limit)["sha256"]


def _snapshot(path: Path, limit: int) -> dict[str, Any]:
    return _secure_file_snapshot(path, limit)


def _unchanged(path: Path, before: Mapping[str, Any], limit: int) -> None:
    if _snapshot(path, limit) != dict(before):
        _stop(f"TOCTOU mutation: {path}")


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


def validate_plan(plan: Mapping[str, Any]) -> None:
    if set(plan) != PLAN_KEYS:
        _stop("plan keyset mismatch")
    if plan["schema_version"] != "sireto-v4.12-strict-stores-plan-1":
        _stop("plan schema mismatch")
    for section, expected in PLAN_NESTED_KEYS.items():
        value = plan.get(section)
        if not isinstance(value, dict) or set(value) != expected:
            _stop(f"plan {section} keyset mismatch")
    if plan["sandbox"]["network_allowed"] is not False:
        _stop("sandbox network must be denied")
    if plan["sandbox"]["system_read_roots"] != ["/System", "/usr", "/opt/homebrew"]:
        _stop("sandbox runtime roots mismatch")
    if plan["sandbox"]["device_read_literals"] != ["/dev/null", "/dev/urandom"]:
        _stop("sandbox device literals mismatch")
    if plan["sandbox"]["device_read_subpaths"] != ["/dev/fd"]:
        _stop("sandbox device subpaths mismatch")
    if (
        plan["sandbox"]["python_framework_library"] != str(PYTHON_FRAMEWORK_LIBRARY)
        or plan["sandbox"]["python_framework_library_sha256"]
        != PYTHON_FRAMEWORK_LIBRARY_SHA256
        or plan["sandbox"]["git_executable"] != "/usr/bin/git"
    ):
        _stop("sandbox pinned runtime/tool paths mismatch")
    if plan["lookup"]["sample_max_sirets"] != 10_000:
        _stop("lookup sample cap mismatch")
    if not isinstance(plan["sources"], list) or len(set(plan["sources"])) != len(plan["sources"]):
        _stop("plan sources must be a unique list")


def _git_text(args: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["/usr/bin/git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _stop(f"Git verification failed: {exc}")


def _git_bytes(args: Sequence[str]) -> bytes:
    try:
        return subprocess.run(
            ["/usr/bin/git", *args], check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        _stop(f"Git blob verification failed: {exc}")


def _verify_sources(
    repo: Path,
    sources: Sequence[str],
    expected: Mapping[str, str],
    commit: str,
    limit: int,
) -> dict[Path, dict[str, Any]]:
    if set(sources) != set(expected):
        _stop("source_hashes keyset mismatch")
    result = {}
    for relative in sources:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            _stop(f"unsafe source path: {relative}")
        if _git_text(["-C", str(repo), "ls-files", "--error-unmatch", "--", relative]) != relative:
            _stop(f"source is not tracked: {relative}")
        path = repo / relative
        snap = _snapshot(path, limit)
        if snap["sha256"] != expected[relative]:
            _stop(f"source worktree hash mismatch: {relative}")
        blob = _git_bytes(["-C", str(repo), "show", f"{commit}:{relative}"])
        if hashlib.sha256(blob).hexdigest() != expected[relative]:
            _stop(f"source Git blob mismatch: {relative}")
        result[path] = snap
    return result


def _plan_input_paths(plan: Mapping[str, Any]) -> dict[str, Any]:
    safe = plan["safe_input"]
    lookup = plan["lookup"]
    sandbox = plan["sandbox"]
    return {
        "safe_input_root": safe["root"],
        "safe_runtime_manifest": safe["runtime_manifest_path"],
        "safe_queries_all": safe["queries_all_path"],
        "safe_queries_dev": safe["queries_dev_path"],
        "safe_partition_inventory": safe["partition_inventory_path"],
        "safe_tfidf_inventory": safe["tfidf_inventory_path"],
        "safe_input_integrity": safe["integrity_path"],
        "partition_root": plan["partitions"]["root"],
        "cache_root": plan["cache"]["root"],
        "lookup_database": lookup["database_path"],
        "lookup_manifest": lookup["manifest_path"],
        "lookup_integrity": lookup["integrity_path"],
        "lookup_timing": lookup["timing_path"],
        "sandbox_executable": sandbox["executable"],
        "python_framework_app": sandbox["python_framework_app"],
        "python_framework_library": sandbox["python_framework_library"],
        "git_executable": sandbox["git_executable"],
        "system_read_roots": sandbox["system_read_roots"],
        "device_read_literals": sandbox["device_read_literals"],
        "device_read_subpaths": sandbox["device_read_subpaths"],
    }


def _plan_input_hashes(plan: Mapping[str, Any]) -> dict[str, str]:
    safe = plan["safe_input"]
    lookup = plan["lookup"]
    sandbox = plan["sandbox"]
    return {
        "safe_runtime_manifest": safe["runtime_manifest_sha256"],
        "safe_queries_all": safe["queries_all_sha256"],
        "safe_queries_dev": safe["queries_dev_sha256"],
        "safe_partition_inventory": safe["partition_inventory_sha256"],
        "safe_tfidf_inventory": safe["tfidf_inventory_sha256"],
        "safe_input_integrity": safe["integrity_sha256"],
        "partition_full_inventory_logical": plan["partitions"]["expected_full_inventory_sha256"],
        "tfidf_full_inventory_logical": plan["cache"]["expected_full_inventory_sha256"],
        "lookup_database": lookup["database_sha256"],
        "lookup_manifest": lookup["manifest_sha256"],
        "lookup_integrity": lookup["integrity_sha256"],
        "lookup_timing": lookup["timing_sha256"],
        "sandbox_executable": sandbox["executable_sha256"],
        "python_framework_app": sandbox["python_framework_app_sha256"],
        "python_framework_library": sandbox["python_framework_library_sha256"],
        "git_executable": sandbox["git_executable_sha256"],
    }


def validate_lock(
    plan: Mapping[str, Any], lock: Mapping[str, Any], repo: Path, limit: int
) -> dict[Path, dict[str, Any]]:
    if set(lock) != LOCK_KEYS:
        _stop("lock keyset mismatch")
    commit = lock.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        _stop("lock git_commit must be a full lowercase SHA-1")
    if _git_text(["-C", str(repo), "cat-file", "-t", commit]) != "commit":
        _stop("lock git_commit is not a commit object")
    if _git_text(["-C", str(repo), "rev-parse", f"{commit}^{{commit}}"]) != commit:
        _stop("lock git_commit does not resolve exactly to itself")
    checks = (
        (lock["schema_version"] == LOCK_SCHEMA, "schema"),
        (lock["purpose"] == LOCK_PURPOSE, "purpose"),
        (lock["audit_verdict"] == LOCK_VERDICT, "audit verdict"),
        (lock["input_paths"] == _plan_input_paths(plan), "input paths"),
        (lock["input_hashes"] == _plan_input_hashes(plan), "input hashes"),
        (lock["expected_routing"] == plan["routing"], "routing"),
        (
            lock["expected_partition_subset_sha256"]
            == plan["partitions"]["expected_subset_logical_sha256"],
            "partition subset",
        ),
        (
            lock["expected_cache_subset_sha256"]
            == plan["cache"]["expected_subset_logical_sha256"],
            "cache subset",
        ),
        (lock["runtime"] == plan["runtime"] == _runtime(), "runtime"),
        (lock["output_root"] == plan["output_root"], "output root"),
        (lock["audit_output_root"] == plan["audit_output_root"], "audit root"),
        (lock["temp_root"] == plan["temp_root"], "temp root"),
        (lock["max_rss_bytes"] == plan["max_rss_bytes"] == limit, "RSS"),
    )
    for valid, label in checks:
        if not valid:
            _stop(f"lock {label} mismatch")
    return _verify_sources(
        repo, plan["sources"], lock["source_hashes"], commit, limit
    )


def _verify_sandbox_runtime(
    plan: Mapping[str, Any], limit: int
) -> dict[Path, dict[str, Any]]:
    sandbox = plan["sandbox"]
    expected = {
        Path(sandbox["executable"]): sandbox["executable_sha256"],
        Path(sandbox["python_framework_app"]): sandbox["python_framework_app_sha256"],
        Path(sandbox["python_framework_library"]):
            sandbox["python_framework_library_sha256"],
        Path(sandbox["git_executable"]): sandbox["git_executable_sha256"],
    }
    snapshots = {}
    manifest_payload: bytes | None = None
    for path, digest in expected.items():
        canonical = _canonical_existing_file(path)
        snap = _snapshot(canonical, limit)
        if snap["sha256"] != digest:
            _stop(f"sandbox executable hash mismatch: {path}")
        snapshots[canonical] = snap
    for path in map(Path, sandbox["system_read_roots"]):
        _directory(path)
        if path.resolve(strict=True) != path:
            _stop(f"non-canonical runtime root: {path}")
    for path in map(Path, sandbox["device_read_literals"]):
        _components(path)
        if not stat.S_ISCHR(os.lstat(path).st_mode):
            _stop(f"character device required: {path}")
    for path in map(Path, sandbox["device_read_subpaths"]):
        _directory(path)
    return snapshots


def _project(
    path: Path, columns: Sequence[str], limit: int, expected_sha256: str
) -> pa.Table:
    payload = _secure_read_bytes(path, limit)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        _stop(f"projected parquet hash mismatch: {path.name}")
    parquet = pq.ParquetFile(pa.BufferReader(payload))
    if not set(columns).issubset(parquet.schema_arrow.names):
        _stop(f"projection absent from {path.name}")
    return pa.Table.from_batches(
        list(parquet.iter_batches(columns=list(columns), use_threads=False))
    ).select(list(columns))


def _logical_partition(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            row["relative_path"].encode()
            + b"\0"
            + str(row["size_bytes"]).encode()
            + b"\0"
            + row["sha256"].encode()
            + b"\n"
        )
    return digest.hexdigest()


def _logical_cache(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            b"\0".join(str(row[name]).encode() for name in CACHE_COLUMNS) + b"\n"
        )
    return digest.hexdigest()


def _normalise_geo(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.strip() == "" or text.lower() == "nan":
        return None
    result = text.strip()
    if re.fullmatch(r"\d+\.0+", result):
        result = result.split(".")[0]
    return result


def _partition_key(relative: str) -> str | None:
    match = re.fullmatch(r"insee/insee=([^/]+)/[^/]+\.parquet", relative)
    if match:
        return match.group(1) + "_"
    match = re.fullmatch(r"cp/postcode=([^/]+)/[^/]+\.parquet", relative)
    if match:
        return "_" + match.group(1)
    return None


def derive_routing(
    queries: pa.Table,
    partition_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    partition_by_key: dict[str, Mapping[str, Any]] = {}
    for row in partition_rows:
        key = _partition_key(row["relative_path"])
        if key is None:
            continue
        if key in partition_by_key:
            _stop(f"multiple partitions for key {key}")
        partition_by_key[key] = row
    cache_by_key = {}
    for row in cache_rows:
        key = row["partition_key"]
        if key in cache_by_key:
            _stop(f"duplicate cache key {key}")
        cache_by_key[key] = row
    routed = []
    insee_count = 0
    cp_count = 0
    for row in queries.to_pylist():
        insee = _normalise_geo(row["crm_insee"])
        postcode = _normalise_geo(row["crm_postcode"])
        insee_key = None if insee is None else insee + "_"
        cp_key = None if postcode is None else "_" + postcode
        if insee_key in partition_by_key:
            key = insee_key
            insee_count += 1
        elif cp_key in partition_by_key:
            key = cp_key
            cp_count += 1
        else:
            _stop(f"missing geographic partition for {row['query_id']}")
        if key not in cache_by_key:
            _stop(f"missing TF-IDF cache for {key}")
        routed.append({"query_id": row["query_id"], "partition_key": key})
    payload = b"".join(
        row["query_id"].encode() + b"\0" + row["partition_key"].encode() + b"\n"
        for row in routed
    )
    keys = sorted({row["partition_key"] for row in routed}, key=lambda x: x.encode())
    expected = plan["routing"]
    observed = {
        "query_count": len(routed),
        "insee_query_count": insee_count,
        "cp_query_count": cp_count,
        "distinct_key_count": len(keys),
        "missing_key_count": 0,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if observed != expected:
        _stop(f"routing differs from frozen plan: {observed}")
    selected_partitions = [dict(partition_by_key[key]) for key in keys]
    selected_partitions.sort(key=lambda row: row["relative_path"].encode())
    selected_cache = [dict(cache_by_key[key]) for key in keys]
    selected_cache.sort(
        key=lambda row: (
            row["partition_key"].encode(),
            row["pickle_relative_path"].encode(),
            row["sidecar_relative_path"].encode(),
        )
    )
    return routed, selected_partitions, selected_cache


def _check_subset(
    partitions: Sequence[Mapping[str, Any]],
    caches: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> None:
    partition_size = sum(row["size_bytes"] for row in partitions)
    cache_size = sum(
        row["pickle_size_bytes"] + row["sidecar_size_bytes"] for row in caches
    )
    checks = (
        (len(partitions) == plan["partitions"]["expected_subset_file_count"], "partition count"),
        (partition_size == plan["partitions"]["expected_subset_size_bytes"], "partition size"),
        (
            _logical_partition(partitions)
            == plan["partitions"]["expected_subset_logical_sha256"],
            "partition logical hash",
        ),
        (len(caches) == plan["cache"]["expected_subset_key_count"], "cache key count"),
        (len(caches) * 2 == plan["cache"]["expected_subset_file_count"], "cache file count"),
        (cache_size == plan["cache"]["expected_subset_size_bytes"], "cache size"),
        (
            _logical_cache(caches) == plan["cache"]["expected_subset_logical_sha256"],
            "cache logical hash",
        ),
    )
    for valid, label in checks:
        if not valid:
            _stop(f"{label} mismatch")


def _safe_manifest(plan: Mapping[str, Any], limit: int) -> dict[Path, dict[str, Any]]:
    safe = plan["safe_input"]
    root = Path(safe["root"])
    _directory(root)
    if {path.name for path in root.iterdir()} != SAFE_FILES:
        _stop("safe runtime file-set mismatch")
    expected = {
        Path(safe["runtime_manifest_path"]): safe["runtime_manifest_sha256"],
        Path(safe["queries_all_path"]): safe["queries_all_sha256"],
        Path(safe["queries_dev_path"]): safe["queries_dev_sha256"],
        Path(safe["partition_inventory_path"]): safe["partition_inventory_sha256"],
        Path(safe["tfidf_inventory_path"]): safe["tfidf_inventory_sha256"],
        Path(safe["integrity_path"]): safe["integrity_sha256"],
    }
    snapshots = {}
    for path, sha in expected.items():
        if path == Path(safe["runtime_manifest_path"]):
            snap, manifest_payload = _secure_file_read(path, limit, capture=True)
        else:
            snap = _snapshot(path, limit)
        if snap["sha256"] != sha:
            _stop(f"safe runtime hash mismatch: {path.name}")
        snapshots[path] = snap
    assert manifest_payload is not None
    manifest = _parse_json_bytes(manifest_payload, "runtime_manifest.json")
    if manifest.get("build_id") != safe["build_id"]:
        _stop("safe build ID mismatch")
    for name in SAFE_FILES - {"runtime_manifest.json"}:
        record = manifest.get("files", {}).get(name)
        path = root / name
        if (
            not isinstance(record, dict)
            or record.get("sha256") != snapshots[path]["sha256"]
            or record.get("size_bytes") != snapshots[path]["size"]
        ):
            _stop(f"safe manifest record mismatch: {name}")
    if manifest.get("partition_inventory_sha256") != plan["partitions"]["expected_full_inventory_sha256"]:
        _stop("full partition logical hash mismatch")
    if manifest.get("tfidf_inventory_sha256") != plan["cache"]["expected_full_inventory_sha256"]:
        _stop("full cache logical hash mismatch")
    return snapshots


def _lookup_descriptor(plan: Mapping[str, Any], snapshots: Mapping[Path, Mapping[str, Any]]) -> dict[str, Any]:
    lookup = plan["lookup"]
    for key, sha_key in (
        ("database_path", "database_sha256"),
        ("manifest_path", "manifest_sha256"),
        ("integrity_path", "integrity_sha256"),
        ("timing_path", "timing_sha256"),
    ):
        path = Path(lookup[key])
        if snapshots[path]["sha256"] != lookup[sha_key]:
            _stop(f"lookup hash mismatch: {key}")
    if snapshots[Path(lookup["database_path"])]["size"] != lookup["database_size_bytes"]:
        _stop("lookup database size mismatch")
    database = Path(lookup["database_path"])
    for suffix in (".wal", ".tmp"):
        if os.path.lexists(str(database) + suffix):
            _stop(f"lookup writable sibling exists: {suffix}")
    parsed = {}
    for key in ("manifest_path", "integrity_path", "timing_path"):
        path = Path(lookup[key])
        reopened, payload = _secure_file_read(path, plan["max_rss_bytes"], capture=True)
        if reopened != dict(snapshots[path]) or payload is None:
            _stop(f"lookup control changed before parsing: {path.name}")
        parsed[key] = _parse_json_bytes(payload, path.name)
    manifest = parsed["manifest_path"]
    integrity = parsed["integrity_path"]
    timing = parsed["timing_path"]
    if (
        manifest.get("schema_version") != "sireto-v4.12-snapshot-lookup-1"
        or manifest.get("verdict") != "GO_V412_SNAPSHOT_LOOKUP"
    ):
        _stop("lookup manifest is not GO")
    outputs = manifest.get("outputs")
    expected_outputs = {
        database.name: (
            lookup["database_sha256"], lookup["database_size_bytes"]
        ),
        Path(lookup["integrity_path"]).name: (
            lookup["integrity_sha256"], snapshots[Path(lookup["integrity_path"])]["size"]
        ),
        Path(lookup["timing_path"]).name: (
            lookup["timing_sha256"], snapshots[Path(lookup["timing_path"])]["size"]
        ),
    }
    if not isinstance(outputs, dict) or set(outputs) != set(expected_outputs):
        _stop("lookup manifest output keyset mismatch")
    if any(
        not isinstance(outputs[name], dict)
        or set(outputs[name]) != {"sha256", "size_bytes"}
        or outputs[name]["sha256"] != digest
        or outputs[name]["size_bytes"] != size
        for name, (digest, size) in expected_outputs.items()
    ):
        _stop("lookup manifest database mismatch")
    if (
        integrity.get("verdict") != "GO_V412_SNAPSHOT_LOOKUP"
        or integrity.get("row_count") != 42_322_035
        or integrity.get("unique_siret_count") != 42_322_035
        or integrity.get("unique_index") != "candidate_details_siret_uidx"
        or integrity.get("invalid_siret_count") != 0
        or integrity.get("labels_opened") is not False
        or integrity.get("challenge_opened") is not False
        or integrity.get("lookup_opened_read_only") is not True
    ):
        _stop("lookup integrity mismatch")
    parity = integrity.get("parity")
    if (
        not isinstance(parity, dict)
        or parity.get("mismatch_count") != 0
        or parity.get("lookup_batch_max") != 100
        or parity.get("snapshot_sample_count") != 10_000
        or parity.get("reference_snapshot_scan_count") != 1
    ):
        _stop("lookup parity mismatch")
    if (
        set(timing) != {
            "elapsed_seconds", "peak_rss_bytes", "peak_rss_limit_bytes",
            "serve_latency_gate_evaluated",
        }
        or not isinstance(timing["elapsed_seconds"], (int, float))
        or isinstance(timing["elapsed_seconds"], bool)
        or timing["elapsed_seconds"] <= 0
        or not isinstance(timing["peak_rss_bytes"], int)
        or isinstance(timing["peak_rss_bytes"], bool)
        or timing["peak_rss_bytes"] > timing["peak_rss_limit_bytes"]
        or timing["peak_rss_limit_bytes"] != plan["max_rss_bytes"]
        or timing["serve_latency_gate_evaluated"] is not False
    ):
        _stop("lookup timing mismatch")
    return {
        "schema_version": "sireto-v4.12-strict-lookup-descriptor-1",
        "database_sha256": lookup["database_sha256"],
        "database_size_bytes": lookup["database_size_bytes"],
        "table_name": "candidate_details",
        "columns": [
            "siret",
            "candidate_state",
            "enseigne1",
            "enseigne2",
            "enseigne3",
            "denomination_usuelle",
            "activity_code",
        ],
        "column_types": ["VARCHAR"] * 7,
        "index_name": "candidate_details_siret_uidx",
        "index_unique": True,
        "row_count": 42322035,
        "max_sirets_per_call": 100,
        "read_only": True,
    }


def _sbpl_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _ancestors(paths: Iterable[Path]) -> list[str]:
    values = set()
    for path in paths:
        resolved = _canonical_existing_file(path)
        values.update(str(parent) for parent in resolved.parents)
    return sorted(values, key=lambda value: value.encode())


def render_profile(
    template: str,
    plan: Mapping[str, Any],
    readable_files: Sequence[Path],
    code_files: Sequence[Path],
) -> str:
    if any(template.count(marker) != 1 for marker in PROFILE_MARKERS):
        _stop("sandbox template marker mismatch")
    sandbox = plan["sandbox"]
    system_rules = "\n".join(
        f"  (literal {_sbpl_string(path)})\n  (subpath {_sbpl_string(path)})"
        for path in sandbox["system_read_roots"]
    )
    system_parents = sorted(
        {
            str(parent)
            for root in sandbox["system_read_roots"]
            for parent in Path(root).parents
        },
        key=lambda value: value.encode(),
    )
    # sandbox-exec requires content-level traversal of the runtime ancestors.
    # These are literal rules only; no ancestor receives subtree access.
    system_rules = "\n".join(
        f"  (literal {_sbpl_string(path)})" for path in system_parents
    ) + "\n" + system_rules
    devices = "\n".join(
        f"  (literal {_sbpl_string(path)})"
        for path in sandbox["device_read_literals"]
    )
    devices += "\n" + "\n".join(
        f"  (literal {_sbpl_string(path)})\n  (subpath {_sbpl_string(path)})"
        for path in sandbox["device_read_subpaths"]
    )
    readable = sorted(
        {str(_canonical_existing_file(path)) for path in [*readable_files, *code_files]},
        key=lambda value: value.encode(),
    )
    read_rules = "\n".join(
        f"  (literal {_sbpl_string(path)})" for path in readable
    )
    fixed_temp = _canonical_existing_dir(Path(plan["temp_root"]))
    metadata_paths = set(_ancestors([Path(path) for path in readable]))
    metadata_paths.update(system_parents)
    metadata_paths.update(str(parent) for parent in fixed_temp.parents)
    metadata_paths.add(str(fixed_temp))
    database = Path(plan["lookup"]["database_path"])
    metadata_paths.update((str(database) + ".wal", str(database) + ".tmp"))
    metadata = "\n".join(
        f"  (literal {_sbpl_string(path)})"
        for path in sorted(metadata_paths, key=lambda value: value.encode())
    )
    recall_root = Path(plan["temp_root"]).parents[1]
    denies = [
        recall_root / "oracles",
        recall_root / "audits",
        recall_root / "final",
        recall_root / "challenges",
        recall_root / "holdouts",
        recall_root / "tests",
    ]
    deny_rules = "\n".join(
        f"(deny file-read* file-write* (subpath {_sbpl_string(str(path))}))"
        for path in sorted(denies, key=lambda path: str(path).encode())
    )
    replacements = {
        "@@SYSTEM_READ_RULES@@": system_rules,
        "@@DEVICE_READ_LITERALS@@": devices,
        "@@CODE_DATA_READ_LITERALS@@": read_rules,
        "@@ANCESTOR_METADATA_LITERALS@@": metadata,
        "@@EXPLICIT_DENY_RULES@@": deny_rules,
    }
    effective = template
    for marker, replacement in replacements.items():
        effective = effective.replace(marker, replacement)
    if "@@" in effective:
        _stop("unresolved sandbox template marker")
    forbidden_exact = (
        plan["sandbox"]["forbidden_oracle_manifest"],
        plan["sandbox"]["forbidden_audit_manifest"],
    )
    if any(path in effective for path in forbidden_exact):
        _stop("effective profile leaks a forbidden sentinel")
    return effective.rstrip() + "\n"


def _allowed_files(
    partitions: Sequence[Mapping[str, Any]],
    caches: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[Path]]:
    rows = []
    root = Path(plan["partitions"]["root"])
    for row in partitions:
        path = _canonical_existing_file(root / row["relative_path"])
        rows.append(
            {
                "role": "partition",
                "partition_key": _partition_key(row["relative_path"]),
                "absolute_path": str(path),
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
        )
    root = Path(plan["cache"]["root"])
    for row in caches:
        for role, path_name, size_name, sha_name in (
            ("cache_pickle", "pickle_relative_path", "pickle_size_bytes", "pickle_sha256"),
            ("cache_sidecar", "sidecar_relative_path", "sidecar_size_bytes", "sidecar_sha256"),
        ):
            path = _canonical_existing_file(root / row[path_name])
            rows.append(
                {
                    "role": role,
                    "partition_key": row["partition_key"],
                    "absolute_path": str(path),
                    "size_bytes": row[size_name],
                    "sha256": row[sha_name],
                }
            )
    database = _canonical_existing_file(Path(plan["lookup"]["database_path"]))
    rows.append(
        {
            "role": "lookup_database",
            "partition_key": "",
            "absolute_path": str(database),
            "size_bytes": plan["lookup"]["database_size_bytes"],
            "sha256": plan["lookup"]["database_sha256"],
        }
    )
    rows.sort(key=lambda row: (row["role"].encode(), row["absolute_path"].encode()))
    if len(rows) != 1945 or len({row["absolute_path"] for row in rows}) != 1945:
        _stop("allowed child data closure is not exactly 1,945 files")
    return rows, [Path(row["absolute_path"]) for row in rows]


def _write_json(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())


def _write_exclusive_bytes(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _stop(f"short write while sealing {path}")
            view = view[written:]
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != len(payload):
            _stop(f"sealed private source mismatch: {path}")
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _seal_control_file(path: Path, expected: bytes, limit: int) -> dict[str, Any]:
    os.chmod(path, 0o444)
    snapshot, payload = _secure_file_read(path, limit, capture=True)
    if payload != expected or snapshot["mode"] != 0o444:
        _stop(f"private control file seal mismatch: {path}")
    return snapshot


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record(path: Path) -> dict[str, Any]:
    snapshot = _secure_file_snapshot(path)
    return {
        "path": path.name,
        "size_bytes": snapshot["size"],
        "sha256": snapshot["sha256"],
    }


def _table(rows: Sequence[Sequence[Any]]) -> pa.Table:
    columns = list(zip(*rows, strict=True))
    return pa.Table.from_arrays(
        [
            pa.array(values, type=field.type)
            for values, field in zip(columns, LEDGER_SCHEMA, strict=True)
        ],
        schema=LEDGER_SCHEMA,
    )


def _open_directory_fd(path: Path) -> tuple[int, tuple[int, int]]:
    before = _directory(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _stop(f"cannot anchor publication directory {path}: {exc}")
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
    ):
        os.close(descriptor)
        _stop(f"publication directory identity changed: {path}")
    return descriptor, (opened.st_dev, opened.st_ino)


def _seal_transfer_tree(root: Path) -> None:
    _directory(root)
    directories: list[Path] = []
    for current, child_directories, files in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        current_mode = os.lstat(current_path).st_mode
        if stat.S_ISLNK(current_mode) or not stat.S_ISDIR(current_mode):
            _stop(f"invalid directory in publication staging: {current_path}")
        if current_path != root:
            directories.append(current_path)
        for name in child_directories:
            child = current_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                _stop(f"invalid directory in publication staging: {child}")
        for name in files:
            child = current_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                _stop(f"invalid file in publication staging: {child}")
            os.chmod(child, 0o444)
            _fsync_file(child)
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        os.chmod(directory, 0o555)
        _fsync_dir(directory)
    # APFS volumes mounted with noowners reject rename of a 0555 directory.
    # Only the transferred root remains private 0700; descendants are sealed.
    os.chmod(root, 0o700)
    _fsync_dir(root)


def _freeze_published_root(root: Path) -> None:
    mode = os.lstat(root).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        _stop(f"published root is not a real directory: {root}")
    permissions = stat.S_IMODE(mode)
    if permissions not in {0o555, 0o700}:
        _stop(f"published root mode is not recoverable: {root}")
    descriptor, identity = _open_directory_fd(root)
    try:
        if permissions == 0o700:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(root)
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or (after.st_dev, after.st_ino) != identity
        or stat.S_IMODE(after.st_mode) != 0o555
    ):
        _stop(f"published root could not be frozen safely: {root}")
    _fsync_dir(root.parent)


def _promote(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        _stop(f"immutable destination exists: {destination}")
    source_parent = source.parent
    destination_parent = destination.parent
    source_parent_stat = _directory(source_parent)
    destination_parent_stat = _directory(destination_parent)
    if source_parent_stat.st_dev != destination_parent_stat.st_dev:
        _stop("staging and publication filesystems differ")
    _seal_transfer_tree(source)
    descriptor, identity = _open_directory_fd(source)
    renamed = False
    primary_error: BaseException | None = None
    try:
        try:
            os.rename(source, destination)
        except OSError as exc:
            _stop(f"publication rename failed: {exc}")
        renamed = True
        try:
            after = os.lstat(destination)
        except OSError as exc:
            _stop(f"cannot inspect renamed publication: {exc}")
        if (
            stat.S_ISLNK(after.st_mode)
            or not stat.S_ISDIR(after.st_mode)
            or (after.st_dev, after.st_ino) != identity
        ):
            _stop("renamed publication directory identity mismatch")
    except BaseException as exc:
        primary_error = exc
    finally:
        cleanup_error: BaseException | None = None
        for operation in (
            lambda: os.fchmod(descriptor, 0o555),
            lambda: os.fsync(descriptor),
            lambda: os.close(descriptor),
        ):
            try:
                operation()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        for parent in dict.fromkeys((source_parent, destination_parent)):
            try:
                _fsync_dir(parent)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if primary_error is not None:
            if cleanup_error is not None:
                primary_error.add_note(
                    f"secondary publication cleanup failure: {cleanup_error}"
                )
        elif cleanup_error is not None:
            _stop(f"publication cleanup/fsync failed: {cleanup_error}")
    if primary_error is not None:
        raise primary_error
    if not renamed or os.path.lexists(source):
        _stop("publication rename did not complete atomically")
    _freeze_published_root(destination)


def _validate_immutable_modes(root: Path) -> None:
    if stat.S_IMODE(os.lstat(root).st_mode) != 0o555:
        _stop(f"immutable directory mode mismatch: {root}")
    for path in root.rglob("*"):
        mode = os.lstat(path).st_mode
        expected = 0o555 if stat.S_ISDIR(mode) else 0o444
        if stat.S_ISLNK(mode) or stat.S_IMODE(mode) != expected:
            _stop(f"immutable artifact mode mismatch: {path}")


def _validate_certification_only(cert: Path, build_id: str) -> None:
    _directory(cert)
    names = {
        "store_probe.json", "lookup_descriptor.json", "run_spec.json",
        "sandbox_profile_effective.sb", "integrity.json", "manifest.json",
    }
    if {path.name for path in cert.iterdir()} != names:
        _stop("pending certification file-set mismatch")
    manifest = load_json(cert / "manifest.json")
    integrity = load_json(cert / "integrity.json")
    if (
        set(manifest) != {
            "schema_version", "build_id", "files", "runtime",
            "declarations", "verdict",
        }
        or manifest["schema_version"]
        != "sireto-v4.12-strict-stores-certification-1"
        or manifest["build_id"] != build_id
        or manifest["verdict"] != GO
        or not _declarations_exact(manifest["declarations"])
    ):
        _stop("pending certification manifest mismatch")
    expected = sorted(
        [_record(cert / name) for name in names - {"manifest.json"}],
        key=lambda row: row["path"].encode(),
    )
    if manifest["files"] != expected:
        _stop("pending certification records mismatch")
    if (
        set(integrity) != {
            "schema_version", "build_id", "run_spec_sha256",
            "lookup_descriptor_sha256", "sandbox_profile_effective_sha256",
            "store_probe_sha256", "data_input_count", "data_ledger_sha256",
            "declarations",
        }
        or integrity["schema_version"]
        != "sireto-v4.12-strict-stores-integrity-1"
        or integrity["build_id"] != build_id
        or integrity["data_input_count"] != 1954
        or not _declarations_exact(integrity["declarations"])
    ):
        _stop("pending certification integrity mismatch")
    sealed = {
        "run_spec_sha256": cert / "run_spec.json",
        "lookup_descriptor_sha256": cert / "lookup_descriptor.json",
        "sandbox_profile_effective_sha256": cert / "sandbox_profile_effective.sb",
        "store_probe_sha256": cert / "store_probe.json",
    }
    if any(integrity[key] != sha256_file(path) for key, path in sealed.items()):
        _stop("pending certification internal hash mismatch")


def _validate_publication(cert: Path, audit: Path, build_id: str) -> None:
    _validate_certification_only(cert, build_id)
    cert_names = {
        "store_probe.json", "lookup_descriptor.json", "run_spec.json",
        "sandbox_profile_effective.sb", "integrity.json", "manifest.json",
    }
    audit_names = {"open_ledger.parquet", "provenance.json", "manifest.json"}
    if {path.name for path in cert.iterdir()} != cert_names:
        _stop("certification publication file-set mismatch")
    if {path.name for path in audit.iterdir()} != audit_names:
        _stop("audit publication file-set mismatch")
    cert_manifest = load_json(cert / "manifest.json")
    audit_manifest = load_json(audit / "manifest.json")
    integrity = load_json(cert / "integrity.json")
    provenance = load_json(audit / "provenance.json")
    if set(cert_manifest) != {
        "schema_version", "build_id", "files", "runtime", "declarations", "verdict"
    } or (
        cert_manifest["schema_version"]
        != "sireto-v4.12-strict-stores-certification-1"
        or cert_manifest["build_id"] != build_id
        or cert_manifest["verdict"] != GO
    ):
        _stop("certification manifest mismatch")
    if (
        set(audit_manifest) != {"schema_version", "build_id", "files"}
        or audit_manifest["schema_version"]
        != "sireto-v4.12-strict-stores-audit-manifest-1"
    ):
        _stop("audit manifest keyset mismatch")
    if audit_manifest["build_id"] != build_id:
        _stop("audit build ID mismatch")
    expected_cert_records = sorted(
        [_record(cert / name) for name in cert_names - {"manifest.json"}],
        key=lambda row: row["path"].encode(),
    )
    expected_audit_records = sorted(
        [_record(audit / name) for name in audit_names - {"manifest.json"}],
        key=lambda row: row["path"].encode(),
    )
    if cert_manifest["files"] != expected_cert_records:
        _stop("certification manifest records mismatch")
    if audit_manifest["files"] != expected_audit_records:
        _stop("audit manifest records mismatch")
    if set(integrity) != {
        "schema_version", "build_id", "run_spec_sha256",
        "lookup_descriptor_sha256", "sandbox_profile_effective_sha256",
        "store_probe_sha256", "data_input_count", "data_ledger_sha256",
        "declarations",
    } or (
        integrity["schema_version"] != "sireto-v4.12-strict-stores-integrity-1"
        or integrity["build_id"] != build_id
    ):
        _stop("certification integrity mismatch")
    if integrity["data_input_count"] != 1954:
        _stop("certification data count mismatch")
    sealed = {
        "run_spec_sha256": cert / "run_spec.json",
        "lookup_descriptor_sha256": cert / "lookup_descriptor.json",
        "sandbox_profile_effective_sha256": cert / "sandbox_profile_effective.sb",
        "store_probe_sha256": cert / "store_probe.json",
        "data_ledger_sha256": audit / "open_ledger.parquet",
    }
    if any(integrity[key] != sha256_file(path) for key, path in sealed.items()):
        _stop("cross-artifact integrity mismatch")
    if set(provenance) != {
        "schema_version", "build_id", "git_commit", "source_hashes",
        "lock_sha256", "plan_sha256", "sandbox_profile_effective_sha256",
        "runtime", "data_input_count", "certification_manifest_sha256",
        "declarations",
    } or (
        provenance["schema_version"] != "sireto-v4.12-strict-stores-provenance-1"
        or provenance["build_id"] != build_id
    ):
        _stop("provenance mismatch")
    if provenance["certification_manifest_sha256"] != sha256_file(cert / "manifest.json"):
        _stop("provenance certification hash mismatch")
    if provenance["sandbox_profile_effective_sha256"] != integrity["sandbox_profile_effective_sha256"]:
        _stop("provenance profile hash mismatch")
    if provenance["data_input_count"] != 1954:
        _stop("provenance data count mismatch")
    if any(
        not _declarations_exact(document.get("declarations"))
        for document in (cert_manifest, integrity, provenance)
    ):
        _stop("publication declarations mismatch")
    ledger = pq.read_table(audit / "open_ledger.parquet")
    if ledger.schema != LEDGER_SCHEMA or ledger.num_rows != 1954:
        _stop("published ledger schema/count mismatch")
    order = list(zip(ledger["role"].to_pylist(), ledger["absolute_path"].to_pylist()))
    if order != sorted(order, key=lambda row: (row[0].encode(), row[1].encode())):
        _stop("published ledger order mismatch")
    combinations = Counter(zip(
        ledger["role"].to_pylist(), ledger["projection"].to_pylist()
    ))
    expected_combinations = Counter({
        ("safe_runtime_manifest", "JSON_EXACT"): 1,
        ("safe_queries_all", "HASH_ONLY"): 1,
        ("safe_queries_dev", ",".join(QUERY_COLUMNS)): 1,
        ("safe_partition_inventory", ",".join(PARTITION_COLUMNS)): 1,
        ("safe_tfidf_inventory", ",".join(CACHE_COLUMNS)): 1,
        ("safe_input_integrity", "HASH_ONLY"): 1,
        ("partition", "STRICT_PARTITION_SCHEMA"): 648,
        ("cache_pickle", "PICKLE_TUPLE_7"): 648,
        (
            "cache_sidecar",
            "config_hash,partition_key,schema_version,sha256,size_bytes",
        ): 648,
        (
            "lookup_database",
            "siret,candidate_state,enseigne1,enseigne2,enseigne3,"
            "denomination_usuelle,activity_code",
        ): 1,
        ("lookup_manifest", "PARENT_VALIDATION_ONLY"): 1,
        ("lookup_integrity", "PARENT_VALIDATION_ONLY"): 1,
        ("lookup_timing", "PARENT_VALIDATION_ONLY"): 1,
    })
    if combinations != expected_combinations:
        _stop("published ledger role/projection mismatch")


def _remove_validated_pending(path: Path) -> None:
    for child in path.rglob("*"):
        if child.is_symlink():
            _stop("symlink forbidden in pending certification")
    for child in path.rglob("*"):
        if child.is_file():
            os.chmod(child, 0o600)
    for child in sorted(
        (child for child in path.rglob("*") if child.is_dir()), reverse=True
    ):
        os.chmod(child, 0o700)
    os.chmod(path, 0o700)
    shutil.rmtree(path)
    _fsync_dir(path.parent)


def _validate_against_current_inputs(
    cert: Path,
    audit: Path,
    build_id: str,
    *,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    plan_bytes: bytes,
    lock_bytes: bytes,
    descriptor: Mapping[str, Any],
    descriptor_sha: str,
    run_spec: Mapping[str, Any],
    run_spec_sha: str,
    profile_sha: str,
    ledger_sources: Sequence[tuple[str, Path, str, Mapping[str, Any]]],
) -> None:
    _validate_publication(cert, audit, build_id)
    if _build_identity(
        plan_bytes, lock_bytes, lock, plan, profile_sha, descriptor_sha,
        run_spec_sha,
    ) != build_id:
        _stop("published build ID does not match current sealed identity")
    published_descriptor = load_json(cert / "lookup_descriptor.json")
    published_run_spec = load_json(cert / "run_spec.json")
    published_probe = load_json(cert / "store_probe.json")
    cert_manifest = load_json(cert / "manifest.json")
    provenance = load_json(audit / "provenance.json")
    integrity = load_json(cert / "integrity.json")
    if published_descriptor != dict(descriptor):
        _stop("published lookup descriptor differs from current descriptor")
    if published_run_spec != dict(run_spec):
        _stop("published run spec differs from current run spec")
    if cert_manifest["runtime"] != plan["runtime"]:
        _stop("published runtime differs from current frozen runtime")
    if sha256_file(cert / "sandbox_profile_effective.sb") != profile_sha:
        _stop("published profile differs from current sealed profile")
    _validate_probe(published_probe, build_id, plan)
    expected_provenance = {
        "git_commit": lock["git_commit"],
        "source_hashes": lock["source_hashes"],
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "sandbox_profile_effective_sha256": profile_sha,
        "runtime": plan["runtime"],
        "data_input_count": 1954,
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        _stop("published provenance differs from current sealed inputs")
    if integrity["run_spec_sha256"] != run_spec_sha:
        _stop("published run-spec seal differs from current run spec")
    if integrity["lookup_descriptor_sha256"] != descriptor_sha:
        _stop("published descriptor seal differs from current descriptor")
    expected_rows = sorted(
        [
            {
                "role": role,
                "absolute_path": str(path.absolute()),
                "projection": projection,
                "size_before": before["size"],
                "sha256_before": before["sha256"],
                "size_after": before["size"],
                "sha256_after": before["sha256"],
            }
            for role, path, projection, before in ledger_sources
        ],
        key=lambda row: (row["role"].encode(), row["absolute_path"].encode()),
    )
    published_rows = pq.read_table(audit / "open_ledger.parquet").to_pylist()
    if published_rows != expected_rows:
        _stop("published ledger differs from current sealed data inputs")


def _publication_recovery(
    output_root: Path,
    audit_root: Path,
    build_id: str,
    *,
    current: Mapping[str, Any],
) -> tuple[Path, Path] | None:
    final_cert = output_root / build_id
    pending = output_root / f".pending-{build_id}"
    final_audit = audit_root / build_id
    cert_exists = os.path.lexists(final_cert)
    pending_exists = os.path.lexists(pending)
    audit_exists = os.path.lexists(final_audit)
    if cert_exists and audit_exists and not pending_exists:
        _freeze_published_root(final_cert)
        _freeze_published_root(final_audit)
        state = "complete"
    elif pending_exists and audit_exists and not cert_exists:
        _freeze_published_root(pending)
        _freeze_published_root(final_audit)
        state = "pending_with_audit"
    elif pending_exists and not audit_exists and not cert_exists:
        _freeze_published_root(pending)
        state = "pending_only"
    elif cert_exists or pending_exists or audit_exists:
        _stop("inconsistent durable publication state")
    else:
        return None
    for _, path, _, snapshot in current["ledger_sources"]:
        _unchanged(path, snapshot, current["plan"]["max_rss_bytes"])
    if state == "complete":
        _validate_immutable_modes(final_cert)
        _validate_immutable_modes(final_audit)
        _validate_against_current_inputs(
            final_cert, final_audit, build_id, **current
        )
        return final_cert, final_audit
    if state == "pending_with_audit":
        _validate_immutable_modes(pending)
        _validate_immutable_modes(final_audit)
        _validate_against_current_inputs(
            pending, final_audit, build_id, **current
        )
        _promote(pending, final_cert)
        _validate_against_current_inputs(
            final_cert, final_audit, build_id, **current
        )
        return final_cert, final_audit
    if state == "pending_only":
        _validate_immutable_modes(pending)
        _validate_certification_only(pending, build_id)
        _remove_validated_pending(pending)
        return None
    _stop("unreachable publication recovery state")


def _build_identity(
    plan_bytes: bytes,
    lock_bytes: bytes,
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    profile_sha: str,
    descriptor_sha: str,
    run_spec_sha: str,
) -> str:
    lookup = plan["lookup"]
    identity = {
        "schema_version": "sireto-v4.12-strict-stores-build-identity-1",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "source_hashes": lock["source_hashes"],
        "safe_runtime_manifest_sha256": plan["safe_input"]["runtime_manifest_sha256"],
        "partition_inventory_sha256": plan["safe_input"]["partition_inventory_sha256"],
        "tfidf_inventory_sha256": plan["safe_input"]["tfidf_inventory_sha256"],
        "partition_subset_logical_sha256": plan["partitions"]["expected_subset_logical_sha256"],
        "cache_subset_logical_sha256": plan["cache"]["expected_subset_logical_sha256"],
        "lookup_input_hashes": {
            "database": lookup["database_sha256"],
            "manifest": lookup["manifest_sha256"],
            "integrity": lookup["integrity_sha256"],
            "timing": lookup["timing_sha256"],
        },
        "sandbox_profile_sha256": profile_sha,
        "lookup_descriptor_sha256": descriptor_sha,
        "run_spec_sha256": run_spec_sha,
        "runtime": plan["runtime"],
    }
    return hashlib.sha256(canonical_json(identity)).hexdigest()


def _validate_probe(probe: Mapping[str, Any], build_id: str, plan: Mapping[str, Any]) -> None:
    if set(probe) != PROBE_KEYS or probe["schema_version"] != "sireto-v4.12-strict-stores-probe-1":
        _stop("probe keyset/schema mismatch")
    expected_scalars = {
        "build_id": build_id,
        "query_count": plan["routing"]["query_count"],
        "distinct_key_count": plan["routing"]["distinct_key_count"],
        "partition_verified_count": plan["partitions"]["expected_subset_file_count"],
        "partition_raw_row_count": plan["partitions"]["expected_subset_row_count"],
        "cache_verified_count": plan["cache"]["expected_subset_key_count"],
        "aligned_pool_row_count": plan["cache"]["expected_aligned_row_count"],
        "cache_miss_count": 0,
        "rebuild_count": 0,
        "write_count": 0,
        "lookup_missing_count": 0,
        "lookup_extra_count": 0,
        "declarations": DECLARATIONS,
    }
    for key, value in expected_scalars.items():
        if probe[key] != value:
            _stop(f"probe {key} mismatch")
    numeric_keys = set(expected_scalars) - {"build_id", "declarations"}
    if any(
        not isinstance(probe[key], int) or isinstance(probe[key], bool)
        for key in numeric_keys
    ):
        _stop("probe counter type mismatch")
    if not _declarations_exact(probe["declarations"]):
        _stop("probe declarations mismatch")
    if set(probe["sandbox_checks"]) != {
        "allowed_read",
        "oracle_denied",
        "oracle_audit_denied",
        "network_denied",
        "write_denied",
    } or any(value is not True for value in probe["sandbox_checks"].values()):
        _stop("sandbox checks did not all pass")
    if set(probe["durations_ns"]) != {"partitions", "cache", "lookup", "total"}:
        _stop("probe durations keyset mismatch")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in probe["durations_ns"].values()):
        _stop("invalid probe duration")
    if (
        not isinstance(probe["peak_rss_bytes"], int)
        or isinstance(probe["peak_rss_bytes"], bool)
        or probe["peak_rss_bytes"] < 0
        or probe["peak_rss_bytes"] > plan["max_rss_bytes"]
    ):
        _stop("child RSS limit exceeded")
    if (
        not isinstance(probe["lookup_sample_count"], int)
        or isinstance(probe["lookup_sample_count"], bool)
        or not 0 <= probe["lookup_sample_count"] <= plan["lookup"]["sample_max_sirets"]
    ):
        _stop("lookup sample count mismatch")


@_cleanup_private_runs
def certify(plan_path: Path, lock_path: Path) -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    plan_path = plan_path.absolute()
    lock_path = lock_path.absolute()
    if plan_path != (repo / CANONICAL_PLAN).absolute():
        _stop("only canonical strict-stores plan may execute")
    plan_bytes = _secure_read_bytes(plan_path)
    plan = _parse_json_bytes(plan_bytes, plan_path.name)
    validate_plan(plan)
    if lock_path != (repo / CANONICAL_LOCK).absolute():
        _stop("execution lock path mismatch")
    lock_bytes = _secure_read_bytes(lock_path)
    lock = _parse_json_bytes(lock_bytes, lock_path.name)
    sealed_sources = lock.get("source_hashes")
    if (
        plan["execution_lock_path"] != str(CANONICAL_LOCK)
        or not isinstance(sealed_sources, dict)
        or sealed_sources.get(str(CANONICAL_PLAN))
        != hashlib.sha256(plan_bytes).hexdigest()
    ):
        _stop("raw plan is not the source sealed by the lock")
    limit = plan["max_rss_bytes"]
    _rss(limit)
    if not sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        _stop("certificate parent requires python -B and bytecode disabled")
    git_path = _canonical_existing_file(Path(plan["sandbox"]["git_executable"]))
    git_snap = _snapshot(git_path, limit)
    if git_snap["sha256"] != plan["sandbox"]["git_executable_sha256"]:
        _stop("pinned Git executable hash mismatch")
    sources = validate_lock(plan, lock, repo, limit)
    runtime_snapshots = _verify_sandbox_runtime(plan, limit)
    plan_snap = _snapshot(plan_path, limit)
    lock_snap = _snapshot(lock_path, limit)
    safe_snaps = _safe_manifest(plan, limit)
    queries = _project(
        Path(plan["safe_input"]["queries_dev_path"]),
        QUERY_COLUMNS,
        limit,
        plan["safe_input"]["queries_dev_sha256"],
    )
    partition_table = _project(
        Path(plan["safe_input"]["partition_inventory_path"]),
        PARTITION_COLUMNS,
        limit,
        plan["safe_input"]["partition_inventory_sha256"],
    )
    cache_table = _project(
        Path(plan["safe_input"]["tfidf_inventory_path"]),
        CACHE_COLUMNS,
        limit,
        plan["safe_input"]["tfidf_inventory_sha256"],
    )
    partition_rows = partition_table.to_pylist()
    cache_rows = cache_table.to_pylist()
    if _logical_partition(partition_rows) != plan["partitions"]["expected_full_inventory_sha256"]:
        _stop("full partition inventory content mismatch")
    if _logical_cache(cache_rows) != plan["cache"]["expected_full_inventory_sha256"]:
        _stop("full cache inventory content mismatch")
    _, partitions, caches = derive_routing(queries, partition_rows, cache_rows, plan)
    _check_subset(partitions, caches, plan)
    allowed_rows, child_data_paths = _allowed_files(partitions, caches, plan)

    data_snapshots: dict[Path, dict[str, Any]] = {}
    for row in allowed_rows:
        path = Path(row["absolute_path"])
        snap = _snapshot(path, limit)
        if snap["size"] != row["size_bytes"] or snap["sha256"] != row["sha256"]:
            _stop(f"child data differs from inventory: {path}")
        data_snapshots[path] = snap
    lookup_paths = [
        Path(plan["lookup"][key])
        for key in ("database_path", "manifest_path", "integrity_path", "timing_path")
    ]
    lookup_snaps = {
        path: data_snapshots[path] if path in data_snapshots else _snapshot(path, limit)
        for path in lookup_paths
    }
    descriptor = _lookup_descriptor(plan, lookup_snaps)

    temp_root = Path(plan["temp_root"])
    _ensure_dir(temp_root)
    run_root = (temp_root / f".run-{uuid.uuid4().hex}").absolute()
    run_root.mkdir(mode=0o700)
    _ACTIVE_RUN_ROOTS.add(run_root)
    output = run_root / "output"
    tmp = run_root / "tmp"
    output.mkdir(mode=0o700)
    tmp.mkdir(mode=0o700)
    private_framework_root = run_root / "runtime"
    private_version_root = (
        private_framework_root / "Python.framework" / "Versions" / "3.14"
    )
    private_python = (
        private_version_root
        / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    )
    private_library = private_version_root / "Python"
    private_python.parent.mkdir(parents=True, mode=0o700)
    app_source = _canonical_existing_file(
        Path(plan["sandbox"]["python_framework_app"])
    )
    app_bytes = _secure_read_bytes(app_source, limit)
    if hashlib.sha256(app_bytes).hexdigest() != plan["sandbox"]["python_framework_app_sha256"]:
        _stop("Python app changed before private copy")
    library_bytes = _secure_read_bytes(PYTHON_FRAMEWORK_LIBRARY, limit)
    if hashlib.sha256(library_bytes).hexdigest() != PYTHON_FRAMEWORK_LIBRARY_SHA256:
        _stop("Python framework library changed before private copy")
    _write_exclusive_bytes(private_python, app_bytes, mode=0o555)
    _write_exclusive_bytes(private_library, library_bytes)
    for directory in sorted(
        [path for path in private_framework_root.rglob("*") if path.is_dir()],
        reverse=True,
    ):
        os.chmod(directory, 0o555)
        _fsync_dir(directory)
    os.chmod(private_framework_root, 0o555)
    _fsync_dir(private_framework_root)
    private_python_snap = _snapshot(private_python, limit)
    private_library_snap = _snapshot(private_library, limit)
    descriptor_path = run_root / "lookup_descriptor.json"
    run_spec_path = run_root / "run_spec.json"
    profile_path = run_root / "sandbox_profile_effective.sb"
    probe_source_path = run_root / "v412_strict_stores.py"
    _write_json(descriptor_path, descriptor)
    descriptor_bytes = canonical_json(descriptor)
    descriptor_snap = _seal_control_file(descriptor_path, descriptor_bytes, limit)
    descriptor_sha = sha256_file(descriptor_path)
    run_spec = {
        "schema_version": "sireto-v4.12-strict-stores-run-spec-1",
        "safe_input_build_id": plan["safe_input"]["build_id"],
        "query_count": plan["routing"]["query_count"],
        "routing_payload_sha256": plan["routing"]["payload_sha256"],
        "partition_records": partitions,
        "cache_records": caches,
        "lookup_descriptor_sha256": descriptor_sha,
        "allowed_read_files": allowed_rows,
        "staging_dir": "output",
        "tmp_dir": "tmp",
        "max_rss_bytes": limit,
        "declarations": DECLARATIONS,
    }
    _write_json(run_spec_path, run_spec)
    run_spec_bytes = canonical_json(run_spec)
    run_spec_snap = _seal_control_file(run_spec_path, run_spec_bytes, limit)
    run_spec_sha = sha256_file(run_spec_path)
    core_path = _canonical_existing_file(repo / "src/xgb_matcher/v412_strict_stores.py")
    core_bytes = _secure_read_bytes(core_path, limit)
    core_sha = hashlib.sha256(core_bytes).hexdigest()
    if core_sha != lock["source_hashes"]["src/xgb_matcher/v412_strict_stores.py"]:
        _stop("probe source differs from sealed source")
    _write_exclusive_bytes(probe_source_path, core_bytes)
    probe_source_snap = _snapshot(probe_source_path, limit)
    if probe_source_snap["sha256"] != core_sha:
        _stop("private probe source copy mismatch")
    template_path = _canonical_existing_file(repo / plan["sandbox"]["profile_path"])
    template = _secure_read_bytes(template_path, limit).decode("utf-8")
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != lock["source_hashes"][
        plan["sandbox"]["profile_path"]
    ]:
        _stop("sandbox template differs from sealed source")
    effective = render_profile(template, plan, child_data_paths, [])
    profile_path.write_text(effective, encoding="utf-8", newline="\n")
    _fsync_file(profile_path)
    profile_bytes = effective.encode("utf-8")
    profile_snap = _seal_control_file(profile_path, profile_bytes, limit)
    profile_sha = sha256_file(profile_path)
    build_id = _build_identity(
        plan_bytes,
        lock_bytes,
        lock,
        plan,
        profile_sha,
        descriptor_sha,
        run_spec_sha,
    )
    output_root = Path(plan["output_root"])
    audit_root = Path(plan["audit_output_root"])
    _ensure_dir(output_root)
    _ensure_dir(audit_root)

    ledger_sources: list[tuple[str, Path, str, Mapping[str, Any]]] = []
    safe = plan["safe_input"]
    for role, key, projection in (
        ("safe_runtime_manifest", "runtime_manifest_path", "JSON_EXACT"),
        ("safe_queries_all", "queries_all_path", "HASH_ONLY"),
        ("safe_queries_dev", "queries_dev_path", ",".join(QUERY_COLUMNS)),
        ("safe_partition_inventory", "partition_inventory_path", ",".join(PARTITION_COLUMNS)),
        ("safe_tfidf_inventory", "tfidf_inventory_path", ",".join(CACHE_COLUMNS)),
        ("safe_input_integrity", "integrity_path", "HASH_ONLY"),
    ):
        path = Path(safe[key])
        ledger_sources.append((role, path, projection, safe_snaps[path]))
    for row in allowed_rows:
        path = Path(row["absolute_path"])
        projection = {
            "partition": "STRICT_PARTITION_SCHEMA",
            "cache_pickle": "PICKLE_TUPLE_7",
            "cache_sidecar": "config_hash,partition_key,schema_version,sha256,size_bytes",
            "lookup_database": "siret,candidate_state,enseigne1,enseigne2,enseigne3,denomination_usuelle,activity_code",
        }[row["role"]]
        ledger_sources.append((row["role"], path, projection, data_snapshots[path]))
    for role, key in (
        ("lookup_manifest", "manifest_path"),
        ("lookup_integrity", "integrity_path"),
        ("lookup_timing", "timing_path"),
    ):
        path = Path(plan["lookup"][key])
        ledger_sources.append((role, path, "PARENT_VALIDATION_ONLY", lookup_snaps[path]))
    if len(ledger_sources) != 1954:
        _stop("data ledger source count mismatch")
    recovery_context = {
        "plan": plan,
        "lock": lock,
        "plan_bytes": plan_bytes,
        "lock_bytes": lock_bytes,
        "descriptor": descriptor,
        "descriptor_sha": descriptor_sha,
        "run_spec": run_spec,
        "run_spec_sha": run_spec_sha,
        "profile_sha": profile_sha,
        "ledger_sources": ledger_sources,
    }
    recovered = _publication_recovery(
        output_root, audit_root, build_id, current=recovery_context
    )
    if recovered is not None:
        _remove_private_run_root(run_root)
        _ACTIVE_RUN_ROOTS.discard(run_root)
        return recovered

    run_root_canonical = _canonical_existing_dir(run_root)
    output_canonical = _canonical_existing_dir(output)
    tmp_canonical = _canonical_existing_dir(tmp)
    descriptor_canonical = _canonical_existing_file(descriptor_path)
    run_spec_canonical = _canonical_existing_file(run_spec_path)
    profile_canonical = _canonical_existing_file(profile_path)
    probe_source_canonical = _canonical_existing_file(probe_source_path)
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "JOBLIB_MULTIPROCESSING": "0",
        "TMPDIR": str(tmp_canonical),
        "DYLD_FRAMEWORK_PATH": str(private_framework_root),
    }
    sandbox = plan["sandbox"]
    anchored: dict[str, int] = {}
    try:
        for name, path, snapshot in (
            ("profile", profile_canonical, profile_snap),
            ("source", probe_source_canonical, probe_source_snap),
            ("run_spec", run_spec_canonical, run_spec_snap),
            ("descriptor", descriptor_canonical, descriptor_snap),
            ("runtime_app", private_python, private_python_snap),
            ("runtime_library", private_library, private_library_snap),
        ):
            anchored[name] = _open_anchored_fd(path, snapshot, limit)
    except BaseException:
        for descriptor_fd in anchored.values():
            os.close(descriptor_fd)
        raise
    fd_path = {
        name: f"/dev/fd/{descriptor_fd}"
        for name, descriptor_fd in anchored.items()
    }
    command = [
        sandbox["executable"],
        "-D",
        f"RUN_ROOT={run_root_canonical}",
        "-D",
        f"RUN_SPEC={fd_path['run_spec']}",
        "-D",
        f"LOOKUP_DESCRIPTOR={fd_path['descriptor']}",
        "-D",
        f"RUN_OUTPUT={output_canonical}",
        "-D",
        f"RUN_TMP={tmp_canonical}",
        "-D",
        f"PROBE_SOURCE={fd_path['source']}",
        "-D",
        f"PYTHON_EXECUTABLE={private_python}",
        "-D",
        f"PYTHON_FRAMEWORK_ROOT={private_framework_root}",
        "-p",
        effective,
        str(private_python),
        "-B",
        fd_path["source"],
        "--run-spec",
        fd_path["run_spec"],
        "--lookup-descriptor",
        fd_path["descriptor"],
        "--run-root",
        str(run_root_canonical),
        "--forbidden-oracle",
        sandbox["forbidden_oracle_manifest"],
        "--forbidden-audit",
        sandbox["forbidden_audit_manifest"],
        "--build-id",
        build_id,
    ]
    try:
        _unchanged(core_path, sources[core_path], limit)
        _unchanged(probe_source_canonical, probe_source_snap, limit)
        _unchanged(descriptor_canonical, descriptor_snap, limit)
        _unchanged(run_spec_canonical, run_spec_snap, limit)
        _unchanged(profile_canonical, profile_snap, limit)
        _unchanged(private_python, private_python_snap, limit)
        _unchanged(private_library, private_library_snap, limit)
        for path, snap in runtime_snapshots.items():
            _unchanged(path, snap, limit)
        result = subprocess.run(
            command,
            cwd=run_root,
            env=environment,
            pass_fds=tuple(
                anchored[name]
                for name in ("source", "run_spec", "descriptor")
            ),
            capture_output=True,
            text=True,
        )
        for name, expected in (
            ("profile", profile_snap),
            ("source", probe_source_snap),
            ("run_spec", run_spec_snap),
            ("descriptor", descriptor_snap),
            ("runtime_app", private_python_snap),
            ("runtime_library", private_library_snap),
        ):
            if _snapshot_fd(anchored[name], limit) != expected:
                _stop(f"anchored {name} changed during child execution")
        if result.returncode != 0:
            _stop(f"sandbox probe failed rc={result.returncode}: {result.stderr[-1000:]}")
        _unchanged(core_path, sources[core_path], limit)
        _unchanged(probe_source_canonical, probe_source_snap, limit)
        _unchanged(descriptor_canonical, descriptor_snap, limit)
        _unchanged(run_spec_canonical, run_spec_snap, limit)
        _unchanged(profile_canonical, profile_snap, limit)
        _unchanged(private_python, private_python_snap, limit)
        _unchanged(private_library, private_library_snap, limit)
        for path, snap in runtime_snapshots.items():
            _unchanged(path, snap, limit)
        probe_path = output / "store_probe.json"
        if {path.name for path in output.iterdir()} != {"store_probe.json"}:
            _stop("sandbox output file-set mismatch")
        probe_snapshot, probe_payload = _secure_file_read(
            probe_path, limit, capture=True
        )
        assert probe_payload is not None
        probe = _parse_json_bytes(probe_payload, probe_path.name)
        _validate_probe(probe, build_id, plan)
        os.chmod(probe_path, 0o444)
        probe_snapshot = _seal_control_file(probe_path, probe_payload, limit)

        ledger_rows = []
        for role, path, projection, before in ledger_sources:
            after = _snapshot(path, limit)
            if after != dict(before):
                _stop(f"data input mutated during probe: {path}")
            ledger_rows.append(
                (
                    role,
                    str(path.absolute()),
                    projection,
                    before["size"],
                    before["sha256"],
                    after["size"],
                    after["sha256"],
                )
            )
        ledger_rows.sort(key=lambda row: (row[0].encode(), row[1].encode()))
        ledger = _table(ledger_rows)
        audit_stage = run_root / "audit"
        cert_stage = run_root / "certification"
        audit_stage.mkdir()
        cert_stage.mkdir()
        ledger_path = audit_stage / "open_ledger.parquet"
        pq.write_table(ledger, ledger_path, compression="zstd")
        _fsync_file(ledger_path)
        ledger_sha = sha256_file(ledger_path)
        for payload, name in (
            (probe_payload, "store_probe.json"),
            (descriptor_bytes, "lookup_descriptor.json"),
            (run_spec_bytes, "run_spec.json"),
            (profile_bytes, "sandbox_profile_effective.sb"),
        ):
            _write_exclusive_bytes(cert_stage / name, payload)
        integrity = {
            "schema_version": "sireto-v4.12-strict-stores-integrity-1",
            "build_id": build_id,
            "run_spec_sha256": run_spec_sha,
            "lookup_descriptor_sha256": descriptor_sha,
            "sandbox_profile_effective_sha256": profile_sha,
            "store_probe_sha256": sha256_file(probe_path),
            "data_input_count": 1954,
            "data_ledger_sha256": ledger_sha,
            "declarations": DECLARATIONS,
        }
        _write_json(cert_stage / "integrity.json", integrity)
        cert_manifest = {
            "schema_version": "sireto-v4.12-strict-stores-certification-1",
            "build_id": build_id,
            "files": sorted(
                [_record(cert_stage / name) for name in (
                    "store_probe.json",
                    "lookup_descriptor.json",
                    "run_spec.json",
                    "sandbox_profile_effective.sb",
                    "integrity.json",
                )],
                key=lambda row: row["path"].encode(),
            ),
            "runtime": plan["runtime"],
            "declarations": DECLARATIONS,
            "verdict": GO,
        }
        _write_json(cert_stage / "manifest.json", cert_manifest)
        provenance = {
            "schema_version": "sireto-v4.12-strict-stores-provenance-1",
            "build_id": build_id,
            "git_commit": lock["git_commit"],
            "source_hashes": lock["source_hashes"],
            "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "sandbox_profile_effective_sha256": profile_sha,
            "runtime": plan["runtime"],
            "data_input_count": 1954,
            "certification_manifest_sha256": sha256_file(cert_stage / "manifest.json"),
            "declarations": DECLARATIONS,
        }
        _write_json(audit_stage / "provenance.json", provenance)
        audit_manifest = {
            "schema_version": "sireto-v4.12-strict-stores-audit-manifest-1",
            "build_id": build_id,
            "files": sorted(
                [_record(audit_stage / name) for name in ("open_ledger.parquet", "provenance.json")],
                key=lambda row: row["path"].encode(),
            ),
        }
        _write_json(audit_stage / "manifest.json", audit_manifest)
        _validate_against_current_inputs(
            cert_stage, audit_stage, build_id, **recovery_context
        )
        for path, snap in sources.items():
            _unchanged(path, snap, limit)
        _unchanged(plan_path, plan_snap, limit)
        _unchanged(lock_path, lock_snap, limit)
        for _, path, _, snap in ledger_sources:
            _unchanged(path, snap, limit)
        _unchanged(probe_source_canonical, probe_source_snap, limit)
        for path, snap in runtime_snapshots.items():
            _unchanged(path, snap, limit)
        _verify_sources(repo, plan["sources"], lock["source_hashes"], lock["git_commit"], limit)
        final_cert = output_root / build_id
        pending_cert = output_root / f".pending-{build_id}"
        final_audit = audit_root / build_id
        if any(os.path.lexists(path) for path in (
            final_cert, pending_cert, final_audit
        )):
            _stop("immutable publication already exists")
        _promote(cert_stage, pending_cert)
        _validate_immutable_modes(pending_cert)
        _validate_against_current_inputs(
            pending_cert, audit_stage, build_id, **recovery_context
        )
        _promote(audit_stage, final_audit)
        _validate_immutable_modes(final_audit)
        _validate_against_current_inputs(
            pending_cert, final_audit, build_id, **recovery_context
        )
        for path, snap in sources.items():
            _unchanged(path, snap, limit)
        _unchanged(plan_path, plan_snap, limit)
        _unchanged(lock_path, lock_snap, limit)
        for _, path, _, snap in ledger_sources:
            _unchanged(path, snap, limit)
        _unchanged(probe_source_canonical, probe_source_snap, limit)
        for path, snap in runtime_snapshots.items():
            _unchanged(path, snap, limit)
        _verify_sources(repo, plan["sources"], lock["source_hashes"], lock["git_commit"], limit)
        _promote(pending_cert, final_cert)
        _validate_immutable_modes(final_cert)
        _validate_against_current_inputs(
            final_cert, final_audit, build_id, **recovery_context
        )
        for path, snap in runtime_snapshots.items():
            _unchanged(path, snap, limit)
        return final_cert, final_audit
    finally:
        for descriptor_fd in anchored.values():
            os.close(descriptor_fd)
        _remove_private_run_root(run_root)
        _ACTIVE_RUN_ROOTS.discard(run_root)


def smoke() -> None:
    template = (Path(__file__).resolve().parents[1] / "config/v4_12_strict_stores.sb").read_text()
    with tempfile.TemporaryDirectory(prefix="v412-strict-smoke-") as temporary:
        root = Path(temporary).resolve()
        data = root / "input.bin"
        code = root / "probe.py"
        lookup = root / "lookup.duckdb"
        data.write_bytes(b"data")
        code.write_text(
            "import duckdb, joblib, numpy, pandas, pyarrow, scipy, sklearn\n"
            "import errno, os, pathlib, socket, sys\n"
            "def denied(call):\n"
            "    try: call()\n"
            "    except OSError as exc: assert exc.errno == errno.EPERM; return\n"
            "    raise AssertionError('operation unexpectedly allowed')\n"
            "assert pathlib.Path(sys.argv[1]).read_bytes() == b'data'\n"
            "cwd_stat = os.lstat('.')\n"
            "root_stat = os.lstat(sys.argv[4])\n"
            "assert (cwd_stat.st_dev, cwd_stat.st_ino) == (root_stat.st_dev, root_stat.st_ino)\n"
            "assert os.environ['JOBLIB_MULTIPROCESSING'] == '0'\n"
            "pathlib.Path(os.environ['TMPDIR'], 'ok').write_text('ok')\n"
            "denied(lambda: open(sys.argv[2], 'rb'))\n"
            "denied(lambda: open(sys.argv[3], 'rb'))\n"
            "denied(lambda: pathlib.Path('blocked').write_text('blocked'))\n"
            "def network():\n"
            "    sock = socket.socket(); sock.connect(('127.0.0.1', 9))\n"
            "denied(network)\n"
        )
        lookup.write_bytes(b"lookup")
        temp_root = root / "work" / "temp"
        temp_root.mkdir(parents=True)
        oracle = root / "oracles" / "manifest.json"
        audit = root / "audits" / "manifest.json"
        oracle.parent.mkdir()
        audit.parent.mkdir()
        oracle.write_text("{}")
        audit.write_text("{}")
        output = root / "output"
        tmp = root / "tmp"
        output.mkdir()
        tmp.mkdir()
        python_bin = Path(os.path.realpath(sys.executable))
        python_app = (
            python_bin.parent.parent
            / "Resources/Python.app/Contents/MacOS/Python"
        )
        smoke_framework_root = root / "runtime"
        smoke_version = (
            smoke_framework_root / "Python.framework" / "Versions" / "3.14"
        )
        smoke_python = (
            smoke_version
            / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
        )
        smoke_python.parent.mkdir(parents=True)
        _write_exclusive_bytes(
            smoke_python, _secure_read_bytes(python_app), mode=0o555
        )
        _write_exclusive_bytes(
            smoke_version / "Python",
            _secure_read_bytes(PYTHON_FRAMEWORK_LIBRARY),
        )
        plan = {
            "temp_root": str(temp_root),
            "lookup": {"database_path": str(lookup)},
            "sandbox": {
                "python_framework_app": str(python_app),
                "system_read_roots": ["/System", "/usr", "/opt/homebrew"],
                "device_read_literals": ["/dev/null", "/dev/urandom"],
                "device_read_subpaths": ["/dev/fd"],
                "forbidden_oracle_manifest": str(oracle),
                "forbidden_audit_manifest": str(audit),
            },
        }
        rendered = render_profile(template, plan, [data, lookup], [code])
        if str(data) not in rendered or str(code) not in rendered:
            _stop("smoke literal closure failed")
        if plan["sandbox"]["forbidden_oracle_manifest"] in rendered:
            _stop("smoke profile leaked forbidden path")
        profile = root / "effective.sb"
        profile.write_text(rendered, encoding="utf-8", newline="\n")
        smoke_fds = {
            name: os.open(path, os.O_RDONLY)
            for name, path in (
                ("profile", profile), ("source", code),
                ("spec", data), ("descriptor", lookup),
            )
        }
        smoke_fd_paths = {
            name: f"/dev/fd/{descriptor_fd}"
            for name, descriptor_fd in smoke_fds.items()
        }
        try:
            sandbox_result = subprocess.run([
                "/usr/bin/sandbox-exec",
                "-D", f"RUN_ROOT={root}",
                "-D", f"RUN_SPEC={smoke_fd_paths['spec']}",
                "-D", f"LOOKUP_DESCRIPTOR={smoke_fd_paths['descriptor']}",
                "-D", f"RUN_OUTPUT={output}",
                "-D", f"RUN_TMP={tmp}",
                "-D", f"PROBE_SOURCE={smoke_fd_paths['source']}",
                "-D", f"PYTHON_EXECUTABLE={smoke_python}",
                "-D", f"PYTHON_FRAMEWORK_ROOT={smoke_framework_root}",
                "-p", rendered,
                str(smoke_python), "-B", smoke_fd_paths["source"],
                str(data), str(oracle), str(audit), str(root),
            ], cwd=root, env={
                "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp),
                "JOBLIB_MULTIPROCESSING": "0",
                "DYLD_FRAMEWORK_PATH": str(smoke_framework_root),
            }, pass_fds=tuple(
                smoke_fds[name] for name in ("source", "spec", "descriptor")
            ), capture_output=True, text=True)
        finally:
            for descriptor_fd in smoke_fds.values():
                os.close(descriptor_fd)
        if sandbox_result.returncode != 0 or not (tmp / "ok").is_file():
            _stop(f"synthetic sandbox smoke failed: {sandbox_result.stderr[-500:]}")
        queries = pa.table(
            {
                "query_id": ["q"],
                "crm_name": ["n"],
                "crm_address": ["a"],
                "crm_postcode": ["75001"],
                "crm_city": ["Paris"],
                "crm_insee": ["75056"],
            }
        )
        partitions = [{"relative_path": "insee/insee=75056/x.parquet", "size_bytes": 1, "sha256": "a" * 64}]
        caches = [{
            "partition_key": "75056_",
            "pickle_relative_path": "75056_.pkl",
            "pickle_size_bytes": 1,
            "pickle_sha256": "b" * 64,
            "sidecar_relative_path": "75056_.pkl.sha256.json",
            "sidecar_size_bytes": 1,
            "sidecar_sha256": "c" * 64,
        }]
        payload = b"q\0" + b"75056_\n"
        route_plan = {
            "routing": {
                "query_count": 1,
                "insee_query_count": 1,
                "cp_query_count": 0,
                "distinct_key_count": 1,
                "missing_key_count": 0,
                "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
        _, selected_p, selected_c = derive_routing(queries, partitions, caches, route_plan)
        if len(selected_p) != 1 or len(selected_c) != 1:
            _stop("smoke routing closure failed")


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
        cert, audit = certify(args.plan, args.lock)
        print(
            json.dumps(
                {"verdict": GO, "build_id": cert.name, "certification": str(cert), "audit": str(audit)},
                sort_keys=True,
            )
        )
        return 0
    except CertificationStopped as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
