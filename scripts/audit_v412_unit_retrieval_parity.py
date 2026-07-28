#!/usr/bin/env python3
"""Audit a sealed V4.12 retrieval output without opening historical truth.

The trusted parent opens the sealed worker files and safe query IDs through
anchored descriptors, launches this same source as a distinct sandboxed
process, and publishes only the aggregate cryptographic parity proof.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


GO = "GO_V412_UNIT_RETRIEVAL_PARITY"
STOP = "STOP_V412_UNIT_RETRIEVAL"
RUN_SPEC_SCHEMA = "sireto-v4.12-unit-retrieval-parity-run-spec-1"
PARITY_SCHEMA = "sireto-v4.12-unit-retrieval-parity-1"
PROVENANCE_SCHEMA = "sireto-v4.12-unit-retrieval-parity-provenance-1"
MANIFEST_SCHEMA = "sireto-v4.12-unit-retrieval-parity-manifest-1"
BUILD_SCHEMA = "sireto-v4.12-unit-retrieval-parity-build-1"
WORKER_MANIFEST_SCHEMA = "sireto-v4.12-unit-retrieval-manifest-1"
WORKER_INTEGRITY_SCHEMA = "sireto-v4.12-unit-retrieval-integrity-1"
WORKER_VERDICT = "SEALED_V412_UNIT_RETRIEVAL"
SOURCE_RELATIVE_PATH = "scripts/audit_v412_unit_retrieval_parity.py"
PROFILE_RELATIVE_PATH = "config/v4_12_unit_retrieval_parity.sb"
SANDBOX_EXECUTABLE_PATH = Path("/usr/bin/sandbox-exec")
HASH_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SIRET_RE = re.compile(r"[0-9]{14}")

SAFE_QUERY_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("crm_name", pa.string(), nullable=False),
        pa.field("crm_address", pa.string(), nullable=False),
        pa.field("crm_postcode", pa.string(), nullable=False),
        pa.field("crm_city", pa.string(), nullable=False),
        pa.field("crm_insee", pa.string(), nullable=False),
    ]
)
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
RUNTIME_KEYS = {
    "python",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit_learn",
    "scipy",
    "joblib",
    "duckdb",
    "machine",
    "platform",
}
DECLARATIONS = {
    "oracle_opened": False,
    "oracle_audit_opened": False,
    "historical_candidates_opened": False,
    "models_opened": False,
    "stores_opened": False,
    "network_used": False,
    "writes_outside_staging": False,
}
WORKER_DECLARATIONS = {
    "labels_opened": False,
    "oracle_opened": False,
    "historical_candidates_opened": False,
    "models_opened": False,
    "network_used": False,
    "writes_outside_staging": False,
    "cache_rebuild_attempted": False,
    "positive_injection": False,
}
SANDBOX_CHECK_KEYS = {
    "allowed_read",
    "oracle_denied",
    "oracle_audit_denied",
    "historical_denied",
    "model_denied",
    "stores_denied",
    "network_denied",
    "write_denied",
}
CHECK_KEYS = {
    "schemas",
    "metadata",
    "query_population",
    "query_order",
    "counts",
    "ranks",
    "sirets",
    "candidate_payload",
    "status_payload",
}
PARITY_KEYS = {
    "schema_version",
    "parity_build_id",
    "worker_build_id",
    "query_count",
    "candidate_count",
    "minimum_pool_size",
    "maximum_pool_size",
    "under_ceiling_query_count",
    "empty_query_count",
    "candidate_payload_bytes",
    "candidate_payload_sha256",
    "expected_candidate_payload_bytes",
    "expected_candidate_payload_sha256",
    "status_payload_bytes",
    "status_payload_sha256",
    "expected_status_payload_bytes",
    "expected_status_payload_sha256",
    "checks",
    "sandbox_checks",
    "declarations",
    "verdict",
}
EXPECTED_KEYS = {
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
WORKER_FILE_NAMES = {
    "query_status.parquet",
    "candidates_top100.parquet",
    "integrity.json",
}
RUN_SPEC_KEYS = {
    "schema_version",
    "worker_build_id",
    "worker_manifest_path",
    "worker_manifest_sha256",
    "worker_file_paths",
    "worker_file_hashes",
    "safe_input_build_id",
    "safe_queries_path",
    "safe_queries_sha256",
    "safe_manifest_path",
    "safe_manifest_sha256",
    "safe_query_id_payload_sha256",
    "expected",
    "git_commit",
    "lock_sha256",
    "parity_source_hashes",
    "parity_profile_sha256",
    "sandbox_executable_path",
    "sandbox_executable_sha256",
    "python_executable_path",
    "python_executable_sha256",
    "audit_root_path_sha256",
    "runtime",
    "temp_root",
    "max_rss_bytes",
    "declarations",
}
WORKER_INTEGRITY_KEYS = {
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
WORKER_MANIFEST_KEYS = {
    "schema_version",
    "worker_build_id",
    "safe_input_build_id",
    "strict_stores_build_id",
    "files",
    "runtime",
    "declarations",
    "verdict",
}
FORBIDDEN_SPEC_FRAGMENTS = (
    "/oracles/",
    "/datasets/",
    "/models/",
    "/challenges/",
    "/final/",
    "/evaluations/",
    "is_ground_truth",
    "ground_truth_siret",
    "ground_truth_siren",
)
FORBIDDEN_SPEC_PATH_COMPONENTS = {"audits"}
_ACTIVE_PRIVATE_ROOTS: set[Path] = set()


class ParityStopped(RuntimeError):
    """Raised whenever a fail-closed contract check fails."""


def _stop(message: str) -> None:
    raise ParityStopped(f"{STOP}: {message}")


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _stop(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda token: _stop(
                f"non-finite JSON token in {label}: {token}"
            ),
        )
    except ParityStopped:
        raise
    except Exception as exc:
        _stop(f"invalid JSON {label}: {exc}")
    if not isinstance(value, dict):
        _stop(f"{label} must be a JSON object")
    return value


def runtime_identity() -> dict[str, str]:
    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "scikit_learn": "scikit-learn",
        "scipy": "scipy",
        "joblib": "joblib",
        "duckdb": "duckdb",
    }
    values = {
        key: importlib.metadata.version(distribution)
        for key, distribution in distributions.items()
    }
    values.update(
        {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        }
    )
    return values


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _check_rss(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        _stop("max_rss_bytes must be a positive integer")
    if _rss_bytes() > limit:
        _stop(f"RSS limit exceeded: {_rss_bytes()} > {limit}")


def _exact_false_declarations(value: Any, expected: Mapping[str, bool]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(expected)
        and all(value[key] is False for key in expected)
    )


def _hash_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        _stop(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        _stop(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        _stop(f"{label} must be an absolute path without '..'")
    return path


def path_commitment(path: Path) -> str:
    absolute = _absolute_path(str(path), "committed path")
    return sha256_bytes(str(absolute).encode("utf-8") + b"\n")


def _scan_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            text
            for key, child in value.items()
            for text in [str(key), *_scan_strings(child)]
        ]
    if isinstance(value, list):
        return [text for child in value for text in _scan_strings(child)]
    return []


def validate_run_spec(spec: Mapping[str, Any]) -> None:
    if set(spec) != RUN_SPEC_KEYS:
        _stop("parity run-spec keyset mismatch")
    if spec["schema_version"] != RUN_SPEC_SCHEMA:
        _stop("parity run-spec schema mismatch")
    if not isinstance(spec["worker_build_id"], str) or not spec["worker_build_id"]:
        _stop("worker_build_id must be a non-empty string")
    for key in (
        "worker_manifest_sha256",
        "safe_queries_sha256",
        "safe_manifest_sha256",
        "safe_query_id_payload_sha256",
        "lock_sha256",
        "parity_profile_sha256",
        "sandbox_executable_sha256",
        "python_executable_sha256",
        "audit_root_path_sha256",
    ):
        _hash_string(spec[key], key)
    if not isinstance(spec["safe_input_build_id"], str) or not spec["safe_input_build_id"]:
        _stop("safe_input_build_id must be a non-empty string")
    if not isinstance(spec["git_commit"], str) or COMMIT_RE.fullmatch(
        spec["git_commit"]
    ) is None:
        _stop("git_commit must be a full lowercase SHA-1")
    for key in (
        "worker_manifest_path",
        "safe_queries_path",
        "safe_manifest_path",
        "temp_root",
        "sandbox_executable_path",
        "python_executable_path",
    ):
        _absolute_path(spec[key], key)
    if Path(spec["sandbox_executable_path"]) != SANDBOX_EXECUTABLE_PATH:
        _stop("sandbox executable path is not the contractual executable")
    if Path(spec["python_executable_path"]) != _default_python_executable():
        _stop("python executable path differs from the active frozen runtime")
    file_paths = spec["worker_file_paths"]
    file_hashes = spec["worker_file_hashes"]
    if (
        not isinstance(file_paths, dict)
        or set(file_paths) != WORKER_FILE_NAMES
        or not isinstance(file_hashes, dict)
        or set(file_hashes) != WORKER_FILE_NAMES
    ):
        _stop("worker file closure mismatch")
    for name in WORKER_FILE_NAMES:
        _absolute_path(file_paths[name], f"worker_file_paths.{name}")
        _hash_string(file_hashes[name], f"worker_file_hashes.{name}")
    expected = spec["expected"]
    if not isinstance(expected, dict) or set(expected) != EXPECTED_KEYS:
        _stop("expected parity keyset mismatch")
    integer_keys = EXPECTED_KEYS - {
        "candidate_payload_sha256",
        "status_payload_sha256",
    }
    if any(
        not isinstance(expected[key], int)
        or isinstance(expected[key], bool)
        or expected[key] < 0
        for key in integer_keys
    ):
        _stop("expected counters and payload sizes must be non-negative integers")
    _hash_string(expected["candidate_payload_sha256"], "expected candidate hash")
    _hash_string(expected["status_payload_sha256"], "expected status hash")
    if expected["query_count"] <= 0 or expected["maximum_pool_size"] > 100:
        _stop("invalid expected query count or candidate ceiling")
    source_hashes = spec["parity_source_hashes"]
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != {SOURCE_RELATIVE_PATH}
    ):
        _stop("parity source hash closure mismatch")
    _hash_string(source_hashes[SOURCE_RELATIVE_PATH], "parity source hash")
    runtime = spec["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_KEYS:
        _stop("runtime keyset mismatch")
    if any(not isinstance(runtime[key], str) or not runtime[key] for key in runtime):
        _stop("runtime values must be non-empty strings")
    _check_rss(spec["max_rss_bytes"])
    if not _exact_false_declarations(spec["declarations"], DECLARATIONS):
        _stop("parity run-spec declarations mismatch")
    lowered = [text.lower() for text in _scan_strings(spec)]
    for fragment in FORBIDDEN_SPEC_FRAGMENTS:
        if any(fragment in text for text in lowered):
            _stop(f"forbidden content in sanitized run-spec: {fragment}")
    for text in lowered:
        path = Path(text)
        if path.is_absolute() and any(
            component in FORBIDDEN_SPEC_PATH_COMPONENTS
            for component in path.parts
        ):
            _stop("forbidden path component in sanitized run-spec: audits")


def _snapshot_fd(descriptor: int, limit: int) -> dict[str, Any]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        _stop("anchored descriptor is not a regular file")
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(descriptor, 8 * 1024 * 1024, offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
        _check_rss(limit)
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


def _read_fd(descriptor: int, limit: int, *, max_bytes: int = 1 << 30) -> bytes:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
        _stop("anchored input is not regular or exceeds the byte ceiling")
    chunks: list[bytes] = []
    offset = 0
    while offset < info.st_size:
        block = os.pread(
            descriptor,
            min(8 * 1024 * 1024, info.st_size - offset),
            offset,
        )
        if not block:
            _stop("short read from anchored descriptor")
        chunks.append(block)
        offset += len(block)
        _check_rss(limit)
    return b"".join(chunks)


def _open_anchored(
    path: Path,
    expected_sha256: str | None,
    limit: int,
) -> tuple[int, dict[str, Any]]:
    path = _absolute_path(str(path), "anchored input")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_fd = os.open(path.anchor, directory_flags)
    except OSError as exc:
        _stop(f"cannot anchor input root for {path}: {exc}")
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        descriptor = os.open(path.name, file_flags, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        _stop(f"anchored open failed for {path}: {exc}")
    os.close(parent_fd)
    try:
        snapshot = _snapshot_fd(descriptor, limit)
        if (
            expected_sha256 is not None
            and snapshot["sha256"] != expected_sha256
        ):
            _stop(f"anchored input hash mismatch: {path.name}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, snapshot


def _open_directory_anchored(path: Path) -> int:
    path = _absolute_path(str(path), "anchored directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        if not stat.S_ISDIR(os.fstat(parent_fd).st_mode):
            _stop("anchored directory has wrong type")
        return parent_fd
    except BaseException:
        os.close(parent_fd)
        raise


def _read_directory_json_snapshot(
    directory_fd: int,
    name: str,
    limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if "/" in name or name in {"", ".", ".."}:
        _stop("unsafe directory child name")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = _snapshot_fd(descriptor, limit)
        value = parse_json(_read_fd(descriptor, limit), name)
        after = _snapshot_fd(descriptor, limit)
        if after != before:
            _stop(f"JSON changed while consumed: {name}")
        return value, after
    finally:
        os.close(descriptor)


def _verify_anchored_snapshots(
    descriptors: Mapping[str, int],
    snapshots: Mapping[str, Mapping[str, Any]],
    limit: int,
    *,
    phase: str,
) -> None:
    if set(descriptors) != set(snapshots):
        _stop(f"anchored input closure changed {phase}")
    for name, descriptor in descriptors.items():
        if _snapshot_fd(descriptor, limit) != snapshots[name]:
            _stop(f"anchored input changed {phase}: {name}")


def _parquet_from_fd(descriptor: int, limit: int) -> pa.Table:
    payload = _read_fd(descriptor, limit)
    try:
        parquet = pq.ParquetFile(pa.BufferReader(payload))
        return pa.Table.from_batches(
            list(parquet.iter_batches(use_threads=False)),
            schema=parquet.schema_arrow,
        )
    except Exception as exc:
        _stop(f"invalid anchored Parquet: {exc}")


def _schema_has_no_metadata(schema: pa.Schema) -> bool:
    return schema.metadata is None and all(field.metadata is None for field in schema)


def _schema_description(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "nullable": field.nullable,
            "type": str(field.type),
        }
        for field in schema
    ]


def _parquet_record(
    name: str,
    snapshot: Mapping[str, Any],
    table: pa.Table,
) -> dict[str, Any]:
    return {
        "sha256": snapshot["sha256"],
        "size_bytes": snapshot["size"],
        "row_count": table.num_rows,
        "schema": _schema_description(table.schema),
        "metadata": None,
    }


def _validate_worker_manifest(
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
    tables: Mapping[str, pa.Table],
) -> None:
    if set(manifest) != WORKER_MANIFEST_KEYS:
        _stop("worker manifest keyset mismatch")
    if (
        manifest["schema_version"] != WORKER_MANIFEST_SCHEMA
        or manifest["worker_build_id"] != spec["worker_build_id"]
        or manifest["safe_input_build_id"] != spec["safe_input_build_id"]
        or manifest["verdict"] != WORKER_VERDICT
        or manifest["runtime"] != spec["runtime"]
        or not _exact_false_declarations(
            manifest["declarations"], WORKER_DECLARATIONS
        )
    ):
        _stop("worker manifest identity mismatch")
    if not isinstance(manifest["strict_stores_build_id"], str) or not manifest[
        "strict_stores_build_id"
    ]:
        _stop("worker strict stores build identity missing")
    expected_files = {
        "query_status.parquet": _parquet_record(
            "query_status.parquet",
            snapshots["query_status.parquet"],
            tables["query_status.parquet"],
        ),
        "candidates_top100.parquet": _parquet_record(
            "candidates_top100.parquet",
            snapshots["candidates_top100.parquet"],
            tables["candidates_top100.parquet"],
        ),
        "integrity.json": {
            "sha256": snapshots["integrity.json"]["sha256"],
            "size_bytes": snapshots["integrity.json"]["size"],
        },
    }
    if manifest["files"] != expected_files:
        _stop("worker manifest file records mismatch")


def _validate_worker_integrity(
    integrity: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    if set(integrity) != WORKER_INTEGRITY_KEYS:
        _stop("worker integrity keyset mismatch")
    if (
        integrity["schema_version"] != WORKER_INTEGRITY_SCHEMA
        or integrity["worker_build_id"] != spec["worker_build_id"]
        or not _exact_false_declarations(
            integrity["declarations"], WORKER_DECLARATIONS
        )
    ):
        _stop("worker integrity identity mismatch")
    sandbox_checks = integrity["sandbox_checks"]
    if (
        not isinstance(sandbox_checks, dict)
        or set(sandbox_checks)
        != {
            "allowed_read",
            "oracle_denied",
            "oracle_audit_denied",
            "historical_denied",
            "model_denied",
            "network_denied",
            "write_denied",
        }
        or any(value is not True for value in sandbox_checks.values())
    ):
        _stop("worker sandbox proof mismatch")
    durations = integrity["durations_ns"]
    if (
        not isinstance(durations, dict)
        or set(durations) != {"retrieval", "lookup", "serialization", "total"}
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in durations.values()
        )
    ):
        _stop("worker duration keyset mismatch")
    for key in (
        "query_count",
        "candidate_count",
        "minimum_pool_size",
        "maximum_pool_size",
        "under_ceiling_query_count",
        "empty_query_count",
        "lookup_missing_count",
        "candidate_payload_bytes",
        "status_payload_bytes",
        "peak_rss_bytes",
    ):
        value = integrity[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            _stop(f"worker integrity {key} must be non-negative")
    if integrity["peak_rss_bytes"] > spec["max_rss_bytes"]:
        _stop("worker peak RSS exceeds the locked limit")
    _hash_string(integrity["candidate_payload_sha256"], "worker candidate payload")
    _hash_string(integrity["status_payload_sha256"], "worker status payload")


def _safe_query_ids(table: pa.Table, expected_count: int) -> list[str]:
    if table.schema != SAFE_QUERY_SCHEMA or not _schema_has_no_metadata(table.schema):
        _stop("safe query schema or metadata mismatch")
    ids = table.column("query_id").to_pylist()
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        _stop("safe query ID population mismatch")
    if any(not isinstance(query_id, str) or not query_id for query_id in ids):
        _stop("safe query IDs must be non-empty strings")
    return ids


def _query_id_payload(ids: Sequence[str]) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for query_id in ids:
        payload = query_id.encode("utf-8") + b"\n"
        digest.update(payload)
        size += len(payload)
    return size, digest.hexdigest()


def _candidate_payloads(
    query_ids: Sequence[str],
    status: pa.Table,
    candidates: pa.Table,
) -> dict[str, Any]:
    if status.schema != STATUS_SCHEMA or candidates.schema != CANDIDATE_SCHEMA:
        _stop("worker Parquet schema mismatch")
    if not _schema_has_no_metadata(status.schema) or not _schema_has_no_metadata(
        candidates.schema
    ):
        _stop("worker Arrow metadata forbidden")
    status_rows = status.to_pylist()
    candidate_rows = candidates.to_pylist()
    if len(status_rows) != len(query_ids):
        _stop("query_status population mismatch")
    counts: list[int] = []
    for index, query_id in enumerate(query_ids):
        row = status_rows[index]
        if row["query_id"] != query_id:
            _stop("query_status order mismatch")
        count = row["candidate_count"]
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 100:
            _stop("candidate_count exceeds the strict per-query ceiling")
        counts.append(count)
    candidate_digest = hashlib.sha256()
    candidate_bytes = 0
    cursor = 0
    for query_id, count in zip(query_ids, counts, strict=True):
        seen: set[str] = set()
        for expected_rank in range(1, count + 1):
            if cursor >= len(candidate_rows):
                _stop("candidate table shorter than query_status")
            row = candidate_rows[cursor]
            cursor += 1
            siret = row["candidate_siret"]
            if row["query_id"] != query_id or row["candidate_rank"] != expected_rank:
                _stop("candidate query order or contiguous rank mismatch")
            if not isinstance(siret, str) or SIRET_RE.fullmatch(siret) is None:
                _stop("candidate SIRET is not canonical")
            if siret in seen:
                _stop("duplicate candidate SIRET within query")
            seen.add(siret)
            payload = (
                query_id.encode("utf-8")
                + b"\0"
                + siret.encode("ascii")
                + b"\0"
                + str(expected_rank).encode("ascii")
                + b"\n"
            )
            candidate_digest.update(payload)
            candidate_bytes += len(payload)
    if cursor != len(candidate_rows):
        _stop("candidate table contains rows beyond query_status")
    status_digest = hashlib.sha256()
    status_bytes = 0
    for query_id, count in zip(query_ids, counts, strict=True):
        payload = (
            query_id.encode("utf-8")
            + b"\0"
            + str(count).encode("ascii")
            + b"\n"
        )
        status_digest.update(payload)
        status_bytes += len(payload)
    return {
        "query_count": len(query_ids),
        "candidate_count": len(candidate_rows),
        "minimum_pool_size": min(counts),
        "maximum_pool_size": max(counts),
        "under_ceiling_query_count": sum(count < 100 for count in counts),
        "empty_query_count": sum(count == 0 for count in counts),
        "candidate_payload_bytes": candidate_bytes,
        "candidate_payload_sha256": candidate_digest.hexdigest(),
        "status_payload_bytes": status_bytes,
        "status_payload_sha256": status_digest.hexdigest(),
    }


def parity_build_id(spec: Mapping[str, Any], run_spec_sha256: str) -> str:
    payload = {
        "schema_version": BUILD_SCHEMA,
        "worker_build_id": spec["worker_build_id"],
        "worker_manifest_sha256": spec["worker_manifest_sha256"],
        "worker_file_hashes": spec["worker_file_hashes"],
        "parity_run_spec_sha256": run_spec_sha256,
        "parity_source_hashes": spec["parity_source_hashes"],
        "parity_profile_sha256": spec["parity_profile_sha256"],
        "lock_sha256": spec["lock_sha256"],
        "runtime": spec["runtime"],
    }
    return sha256_bytes(canonical_json(payload))


def evaluate_from_fds(
    spec: Mapping[str, Any],
    *,
    parity_id: str,
    safe_queries_fd: int,
    safe_manifest_fd: int,
    worker_manifest_fd: int,
    worker_integrity_fd: int,
    status_fd: int,
    candidates_fd: int,
    sandbox_checks: Mapping[str, bool],
) -> dict[str, Any]:
    validate_run_spec(spec)
    limit = spec["max_rss_bytes"]
    descriptors = {
        "safe_queries": (safe_queries_fd, spec["safe_queries_sha256"]),
        "safe_manifest": (safe_manifest_fd, spec["safe_manifest_sha256"]),
        "worker_manifest": (
            worker_manifest_fd,
            spec["worker_manifest_sha256"],
        ),
        "integrity.json": (
            worker_integrity_fd,
            spec["worker_file_hashes"]["integrity.json"],
        ),
        "query_status.parquet": (
            status_fd,
            spec["worker_file_hashes"]["query_status.parquet"],
        ),
        "candidates_top100.parquet": (
            candidates_fd,
            spec["worker_file_hashes"]["candidates_top100.parquet"],
        ),
    }
    before = {
        name: _snapshot_fd(descriptor, limit)
        for name, (descriptor, _) in descriptors.items()
    }
    for name, (_, expected_hash) in descriptors.items():
        if before[name]["sha256"] != expected_hash:
            _stop(f"sealed input hash mismatch: {name}")
    safe_manifest = parse_json(
        _read_fd(safe_manifest_fd, limit), "safe runtime manifest"
    )
    if safe_manifest.get("build_id") != spec["safe_input_build_id"]:
        _stop("safe runtime manifest build mismatch")
    safe_table = _parquet_from_fd(safe_queries_fd, limit)
    status_table = _parquet_from_fd(status_fd, limit)
    candidate_table = _parquet_from_fd(candidates_fd, limit)
    query_ids = _safe_query_ids(safe_table, spec["expected"]["query_count"])
    _, query_id_hash = _query_id_payload(query_ids)
    if query_id_hash != spec["safe_query_id_payload_sha256"]:
        _stop("safe query ID payload mismatch")
    worker_manifest = parse_json(
        _read_fd(worker_manifest_fd, limit), "worker manifest"
    )
    worker_integrity = parse_json(
        _read_fd(worker_integrity_fd, limit), "worker integrity"
    )
    _validate_worker_manifest(
        worker_manifest,
        spec,
        {
            "query_status.parquet": before["query_status.parquet"],
            "candidates_top100.parquet": before["candidates_top100.parquet"],
            "integrity.json": before["integrity.json"],
        },
        {
            "query_status.parquet": status_table,
            "candidates_top100.parquet": candidate_table,
        },
    )
    _validate_worker_integrity(worker_integrity, spec)
    actual = _candidate_payloads(query_ids, status_table, candidate_table)
    for key in (
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
    ):
        if worker_integrity[key] != actual[key]:
            _stop(f"worker integrity does not seal actual {key}")
    if worker_integrity["maximum_pool_size"] > 100:
        _stop("worker integrity exceeds candidate ceiling")
    if (
        not isinstance(sandbox_checks, dict)
        or set(sandbox_checks) != SANDBOX_CHECK_KEYS
        or any(value is not True for value in sandbox_checks.values())
    ):
        _stop("parity sandbox checks incomplete")
    expected = spec["expected"]
    checks = {
        "schemas": True,
        "metadata": True,
        "query_population": actual["query_count"] == expected["query_count"],
        "query_order": True,
        "counts": all(
            actual[key] == expected[key]
            for key in (
                "candidate_count",
                "minimum_pool_size",
                "maximum_pool_size",
                "under_ceiling_query_count",
                "empty_query_count",
            )
        ),
        "ranks": True,
        "sirets": True,
        "candidate_payload": (
            actual["candidate_payload_bytes"]
            == expected["candidate_payload_bytes"]
            and actual["candidate_payload_sha256"]
            == expected["candidate_payload_sha256"]
        ),
        "status_payload": (
            actual["status_payload_bytes"] == expected["status_payload_bytes"]
            and actual["status_payload_sha256"]
            == expected["status_payload_sha256"]
        ),
    }
    verdict = GO if all(checks.values()) else STOP
    report = {
        "schema_version": PARITY_SCHEMA,
        "parity_build_id": parity_id,
        "worker_build_id": spec["worker_build_id"],
        **actual,
        "expected_candidate_payload_bytes": expected["candidate_payload_bytes"],
        "expected_candidate_payload_sha256": expected[
            "candidate_payload_sha256"
        ],
        "expected_status_payload_bytes": expected["status_payload_bytes"],
        "expected_status_payload_sha256": expected["status_payload_sha256"],
        "checks": checks,
        "sandbox_checks": dict(sandbox_checks),
        "declarations": dict(DECLARATIONS),
        "verdict": verdict,
    }
    if set(report) != PARITY_KEYS:
        _stop("internal parity report keyset mismatch")
    after = {
        name: _snapshot_fd(descriptor, limit)
        for name, (descriptor, _) in descriptors.items()
    }
    if after != before:
        _stop("sealed input changed during parity evaluation")
    return report


def _expect_eperm(path: str, flags: int) -> bool:
    try:
        descriptor = os.open(
            path,
            flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        if exc.errno != 1:
            _stop(f"sandbox sentinel returned errno={exc.errno}: {path}")
        return True
    else:
        os.close(descriptor)
        _stop(f"sandbox sentinel unexpectedly opened: {path}")


def _network_denied() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 9), timeout=0.1):
            pass
    except OSError as exc:
        if exc.errno != 1:
            _stop(f"network sentinel returned errno={exc.errno}")
        return True
    _stop("network sentinel unexpectedly connected")


def sandbox_probes(args: argparse.Namespace) -> dict[str, bool]:
    checks = {
        "allowed_read": True,
        "oracle_denied": _expect_eperm(args.forbidden_oracle, os.O_RDONLY),
        "oracle_audit_denied": _expect_eperm(
            args.forbidden_oracle_audit, os.O_RDONLY
        ),
        "historical_denied": _expect_eperm(
            args.forbidden_historical, os.O_RDONLY
        ),
        "model_denied": _expect_eperm(args.forbidden_model, os.O_RDONLY),
        "stores_denied": _expect_eperm(args.forbidden_store, os.O_RDONLY),
        "network_denied": _network_denied(),
        "write_denied": _expect_eperm(
            args.write_sentinel,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        ),
    }
    if set(checks) != SANDBOX_CHECK_KEYS or any(
        value is not True for value in checks.values()
    ):
        _stop("sandbox proof incomplete")
    return checks


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                _stop(f"short write: {path.name}")
            view = view[count:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record_from_snapshot(
    name: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": name,
        "size_bytes": snapshot["size"],
        "sha256": snapshot["sha256"],
    }


def _seal_tree(root: Path) -> None:
    if stat.S_ISLNK(os.lstat(root).st_mode) or not root.is_dir():
        _stop("publication stage must be a real directory")
    for child in root.iterdir():
        mode = os.lstat(child).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            _stop("parity audit may contain regular files only")
        os.chmod(child, 0o444)
        descriptor = os.open(child, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.chmod(root, 0o700)
    _fsync_dir(root)


def _renameat_exclusive(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = function(
            source_parent_fd,
            os.fsencode(source_name),
            destination_parent_fd,
            os.fsencode(destination_name),
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = function(
            source_parent_fd,
            os.fsencode(source_name),
            destination_parent_fd,
            os.fsencode(destination_name),
            0x00000001,  # RENAME_NOREPLACE
        )
    else:
        _stop("atomic exclusive directory rename is unavailable")
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            _stop(
                f"immutable publication destination exists: {destination_name}"
            )
        _stop(f"exclusive publication rename failed errno={error}")


def _promote(
    source: Path,
    destination: Path,
    *,
    source_fd: int | None = None,
) -> None:
    if source.name in {"", ".", ".."} or destination.name in {"", ".", ".."}:
        _stop("unsafe publication name")
    owned_source_fd = source_fd is None
    if source_fd is None:
        source_fd = _open_directory_anchored(source)
    source_parent_fd: int | None = None
    destination_parent_fd: int | None = None
    try:
        if owned_source_fd:
            _seal_tree(source)
        source_parent_fd = _open_directory_anchored(source.parent)
        destination_parent_fd = _open_directory_anchored(destination.parent)
        identity = os.fstat(source_fd)
        source_entry = os.stat(
            source.name,
            dir_fd=source_parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(source_entry.st_mode)
            or (source_entry.st_dev, source_entry.st_ino)
            != (identity.st_dev, identity.st_ino)
        ):
            _stop("validated publication source was substituted")
        if os.fstat(source_parent_fd).st_dev != os.fstat(
            destination_parent_fd
        ).st_dev:
            _stop("publication crosses filesystems")
        _renameat_exclusive(
            source_parent_fd,
            source.name,
            destination_parent_fd,
            destination.name,
        )
        destination_entry = os.stat(
            destination.name,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(destination_entry.st_mode)
            or (destination_entry.st_dev, destination_entry.st_ino)
            != (identity.st_dev, identity.st_ino)
        ):
            _stop("publication directory identity changed during rename")
        os.fchmod(source_fd, 0o555)
        os.fsync(source_fd)
        os.fsync(source_parent_fd)
        if destination_parent_fd != source_parent_fd:
            os.fsync(destination_parent_fd)
    finally:
        if source_parent_fd is not None:
            os.close(source_parent_fd)
        if destination_parent_fd is not None:
            os.close(destination_parent_fd)
        if owned_source_fd:
            os.close(source_fd)


def _read_path_json_snapshot(
    path: Path,
    limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor, _ = _open_anchored(path, None, limit)
    try:
        before = _snapshot_fd(descriptor, limit)
        value = parse_json(_read_fd(descriptor, limit), path.name)
        after = _snapshot_fd(descriptor, limit)
        if after != before:
            _stop(f"JSON changed while consumed: {path.name}")
        return value, after
    finally:
        os.close(descriptor)


def _load_path_json(path: Path, limit: int) -> dict[str, Any]:
    return _read_path_json_snapshot(path, limit)[0]


def _validate_parity_report(
    report: Mapping[str, Any],
    *,
    parity_id: str,
    spec: Mapping[str, Any],
) -> None:
    if (
        set(report) != PARITY_KEYS
        or report["schema_version"] != PARITY_SCHEMA
        or report["parity_build_id"] != parity_id
        or report["worker_build_id"] != spec["worker_build_id"]
        or report["declarations"] != DECLARATIONS
    ):
        _stop("parity report identity or keyset mismatch")
    for key in (
        "query_count",
        "candidate_count",
        "minimum_pool_size",
        "maximum_pool_size",
        "under_ceiling_query_count",
        "empty_query_count",
        "candidate_payload_bytes",
        "expected_candidate_payload_bytes",
        "status_payload_bytes",
        "expected_status_payload_bytes",
    ):
        value = report[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            _stop(f"parity report {key} must be a non-negative integer")
    if report["maximum_pool_size"] > 100:
        _stop("published parity report exceeds candidate ceiling")
    for key in (
        "candidate_payload_sha256",
        "expected_candidate_payload_sha256",
        "status_payload_sha256",
        "expected_status_payload_sha256",
    ):
        _hash_string(report[key], f"parity report {key}")
    expected = spec["expected"]
    if (
        report["expected_candidate_payload_bytes"]
        != expected["candidate_payload_bytes"]
        or report["expected_candidate_payload_sha256"]
        != expected["candidate_payload_sha256"]
        or report["expected_status_payload_bytes"]
        != expected["status_payload_bytes"]
        or report["expected_status_payload_sha256"]
        != expected["status_payload_sha256"]
    ):
        _stop("parity report expected commitments mismatch")
    checks = report["checks"]
    sandbox_checks = report["sandbox_checks"]
    if (
        not isinstance(checks, dict)
        or set(checks) != CHECK_KEYS
        or any(type(value) is not bool for value in checks.values())
        or not isinstance(sandbox_checks, dict)
        or set(sandbox_checks) != SANDBOX_CHECK_KEYS
        or any(value is not True for value in sandbox_checks.values())
    ):
        _stop("parity report checks mismatch")
    derived_checks = {
        "schemas": True,
        "metadata": True,
        "query_population": report["query_count"] == expected["query_count"],
        "query_order": True,
        "counts": all(
            report[key] == expected[key]
            for key in (
                "candidate_count",
                "minimum_pool_size",
                "maximum_pool_size",
                "under_ceiling_query_count",
                "empty_query_count",
            )
        ),
        "ranks": True,
        "sirets": True,
        "candidate_payload": (
            report["candidate_payload_bytes"]
            == expected["candidate_payload_bytes"]
            and report["candidate_payload_sha256"]
            == expected["candidate_payload_sha256"]
        ),
        "status_payload": (
            report["status_payload_bytes"] == expected["status_payload_bytes"]
            and report["status_payload_sha256"]
            == expected["status_payload_sha256"]
        ),
    }
    if checks != derived_checks:
        _stop("parity report checks are not derived from observed commitments")
    expected_verdict = GO if all(derived_checks.values()) else STOP
    if report["verdict"] != expected_verdict:
        _stop("parity report verdict does not follow its checks")


def validate_published_audit(
    root: Path,
    *,
    parity_id: str,
    spec: Mapping[str, Any],
    run_spec_sha256: str,
    limit: int,
    directory_fd: int | None = None,
) -> dict[str, Any]:
    owned = directory_fd is None
    if directory_fd is None:
        directory_fd = _open_directory_anchored(root)
    try:
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            _stop("published parity root is not a directory")
        names = set(os.listdir(directory_fd))
        if names != {"parity.json", "provenance.json", "manifest.json"}:
            _stop("published parity file-set mismatch")
        parity, parity_snapshot = _read_directory_json_snapshot(
            directory_fd, "parity.json", limit
        )
        provenance, provenance_snapshot = _read_directory_json_snapshot(
            directory_fd, "provenance.json", limit
        )
        manifest, _ = _read_directory_json_snapshot(
            directory_fd, "manifest.json", limit
        )
        root_mode = stat.S_IMODE(os.fstat(directory_fd).st_mode)
        file_modes = {
            name: stat.S_IMODE(
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
            )
            for name in names
        }
    finally:
        if owned:
            os.close(directory_fd)
    _validate_parity_report(parity, parity_id=parity_id, spec=spec)
    if (
        set(provenance)
        != {
            "schema_version",
            "parity_build_id",
            "worker_build_id",
            "git_commit",
            "parity_source_hashes",
            "lock_sha256",
            "parity_run_spec_sha256",
            "worker_manifest_sha256",
            "runtime",
            "declarations",
        }
        or provenance["schema_version"] != PROVENANCE_SCHEMA
        or provenance["parity_build_id"] != parity_id
        or provenance["worker_build_id"] != spec["worker_build_id"]
        or provenance["git_commit"] != spec["git_commit"]
        or provenance["parity_source_hashes"] != spec["parity_source_hashes"]
        or provenance["lock_sha256"] != spec["lock_sha256"]
        or provenance["parity_run_spec_sha256"] != run_spec_sha256
        or provenance["worker_manifest_sha256"]
        != spec["worker_manifest_sha256"]
        or provenance["runtime"] != spec["runtime"]
        or not _exact_false_declarations(
            provenance["declarations"], DECLARATIONS
        )
    ):
        _stop("published parity provenance mismatch")
    expected_records = sorted(
        [
            _record_from_snapshot("parity.json", parity_snapshot),
            _record_from_snapshot("provenance.json", provenance_snapshot),
        ],
        key=lambda row: row["path"].encode(),
    )
    if (
        set(manifest)
        != {
            "schema_version",
            "parity_build_id",
            "worker_build_id",
            "files",
            "runtime",
            "declarations",
            "verdict",
        }
        or manifest["schema_version"] != MANIFEST_SCHEMA
        or manifest["parity_build_id"] != parity_id
        or manifest["worker_build_id"] != spec["worker_build_id"]
        or manifest["files"] != expected_records
        or manifest["runtime"] != spec["runtime"]
        or manifest["declarations"] != DECLARATIONS
        or manifest["verdict"] != parity["verdict"]
    ):
        _stop("published parity manifest mismatch")
    if root_mode not in {0o555, 0o700}:
        _stop("published parity root mode mismatch")
    if any(mode != 0o444 for mode in file_modes.values()):
        _stop("published parity file mode mismatch")
    return parity


def _safe_cleanup_private(root: Path) -> None:
    if root not in _ACTIVE_PRIVATE_ROOTS or not root.name.startswith(".run-"):
        _stop(f"refusing cleanup outside registered private root: {root}")
    if not os.path.lexists(root):
        _ACTIVE_PRIVATE_ROOTS.discard(root)
        return
    if stat.S_ISLNK(os.lstat(root).st_mode) or not stat.S_ISDIR(
        os.lstat(root).st_mode
    ):
        _stop("private run root substituted")
    for current, directories, files in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        os.chmod(current_path, 0o700)
        for name in directories:
            path = current_path / name
            if stat.S_ISLNK(os.lstat(path).st_mode):
                _stop("symlink in private run root")
            os.chmod(path, 0o700)
        for name in files:
            path = current_path / name
            if stat.S_ISLNK(os.lstat(path).st_mode):
                _stop("symlink in private run root")
            os.chmod(path, 0o600)
    shutil.rmtree(root)
    _ACTIVE_PRIVATE_ROOTS.discard(root)


def _profile_bytes(path: Path, expected_sha256: str, limit: int) -> bytes:
    descriptor, before = _open_anchored(path, expected_sha256, limit)
    try:
        payload = _read_fd(descriptor, limit, max_bytes=1024 * 1024)
        if _snapshot_fd(descriptor, limit) != before:
            _stop("sandbox profile changed while consumed")
    finally:
        os.close(descriptor)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        _stop("sandbox profile is not UTF-8")
    required = (
        "(deny default)",
        "(deny network*)",
        "(deny process-fork)",
        'param "RUN_OUTPUT"',
        'param "RUN_TMP"',
        'param "FORBIDDEN_HISTORICAL"',
    )
    if any(marker not in text for marker in required):
        _stop("sandbox profile mandatory rule missing")
    return payload


def _sandbox_command(
    *,
    sandbox_executable: Path,
    python_executable: Path,
    profile: str,
    run_root: Path,
    output: Path,
    tmp: Path,
    source_fd: int,
    input_fds: Mapping[str, int],
    parity_id: str,
    forbidden: Mapping[str, str],
) -> list[str]:
    source_path = f"/dev/fd/{source_fd}"
    parameters = {
        "RUN_ROOT": str(run_root),
        "RUN_OUTPUT": str(output),
        "RUN_TMP": str(tmp),
        "PYTHON_EXECUTABLE": str(python_executable),
        "FORBIDDEN_ORACLE": forbidden["oracle"],
        "FORBIDDEN_ORACLE_AUDIT": forbidden["oracle_audit"],
        "FORBIDDEN_HISTORICAL": forbidden["historical"],
        "FORBIDDEN_MODEL": forbidden["model"],
        "FORBIDDEN_STORE": forbidden["store"],
        "WRITE_SENTINEL": forbidden["write"],
    }
    command = [str(sandbox_executable)]
    for key, value in parameters.items():
        command.extend(["-D", f"{key}={value}"])
    command.extend(
        [
            "-p",
            profile,
            str(python_executable),
            "-B",
            source_path,
            "--sandbox-child",
            "--run-spec-fd",
            str(input_fds["run_spec"]),
            "--safe-queries-fd",
            str(input_fds["safe_queries"]),
            "--safe-manifest-fd",
            str(input_fds["safe_manifest"]),
            "--worker-manifest-fd",
            str(input_fds["worker_manifest"]),
            "--worker-integrity-fd",
            str(input_fds["integrity.json"]),
            "--status-fd",
            str(input_fds["query_status.parquet"]),
            "--candidates-fd",
            str(input_fds["candidates_top100.parquet"]),
            "--output-dir",
            str(output),
            "--parity-build-id",
            parity_id,
            "--forbidden-oracle",
            forbidden["oracle"],
            "--forbidden-oracle-audit",
            forbidden["oracle_audit"],
            "--forbidden-historical",
            forbidden["historical"],
            "--forbidden-model",
            forbidden["model"],
            "--forbidden-store",
            forbidden["store"],
            "--write-sentinel",
            forbidden["write"],
        ]
    )
    return command


def _default_python_executable() -> Path:
    launcher = Path(os.path.realpath(sys.executable))
    framework_app = (
        launcher.parent.parent
        / "Resources"
        / "Python.app"
        / "Contents"
        / "MacOS"
        / "Python"
    )
    return framework_app if framework_app.is_file() else launcher


def _child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sandbox-child", action="store_true")
    parser.add_argument("--run-spec-fd", type=int, required=True)
    parser.add_argument("--safe-queries-fd", type=int, required=True)
    parser.add_argument("--safe-manifest-fd", type=int, required=True)
    parser.add_argument("--worker-manifest-fd", type=int, required=True)
    parser.add_argument("--worker-integrity-fd", type=int, required=True)
    parser.add_argument("--status-fd", type=int, required=True)
    parser.add_argument("--candidates-fd", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parity-build-id", required=True)
    parser.add_argument("--forbidden-oracle", required=True)
    parser.add_argument("--forbidden-oracle-audit", required=True)
    parser.add_argument("--forbidden-historical", required=True)
    parser.add_argument("--forbidden-model", required=True)
    parser.add_argument("--forbidden-store", required=True)
    parser.add_argument("--write-sentinel", required=True)
    return parser


def sandbox_child(argv: Sequence[str]) -> int:
    if os.environ.get("V412_PARITY_SANDBOX_CHILD") != "1":
        _stop("sandbox child marker missing")
    args = _child_parser().parse_args(list(argv))
    if not args.sandbox_child:
        _stop("sandbox child mode missing")
    run_spec_payload = _read_fd(args.run_spec_fd, 1 << 30)
    spec = parse_json(run_spec_payload, "parity run-spec")
    validate_run_spec(spec)
    child_runtime = runtime_identity()
    # platform.platform() may invoke sw_vers through a subprocess on macOS.
    # Fork is deliberately denied in the child, so the trusted parent pins the
    # full platform string while the child rechecks every non-derived value.
    if any(
        child_runtime[key] != spec["runtime"][key]
        for key in RUNTIME_KEYS - {"platform"}
    ):
        _stop("sandbox child runtime mismatch")
    expected_id = parity_build_id(spec, sha256_bytes(run_spec_payload))
    if args.parity_build_id != expected_id:
        _stop("sandbox child build identity mismatch")
    output = args.output_dir
    if stat.S_ISLNK(os.lstat(output).st_mode) or not output.is_dir():
        _stop("sandbox output directory invalid")
    probes = sandbox_probes(args)
    report = evaluate_from_fds(
        spec,
        parity_id=expected_id,
        safe_queries_fd=args.safe_queries_fd,
        safe_manifest_fd=args.safe_manifest_fd,
        worker_manifest_fd=args.worker_manifest_fd,
        worker_integrity_fd=args.worker_integrity_fd,
        status_fd=args.status_fd,
        candidates_fd=args.candidates_fd,
        sandbox_checks=probes,
    )
    _write_exclusive(output / "parity.json", canonical_json(report))
    return 0


def audit_and_publish(
    run_spec_path: Path,
    profile_path: Path,
    audit_root: Path,
    *,
    forbidden_oracle: str,
    forbidden_oracle_audit: str,
    forbidden_historical: str,
    forbidden_model: str,
    forbidden_store: str,
    write_sentinel: str,
    sandbox_executable: Path = Path("/usr/bin/sandbox-exec"),
    python_executable: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    bootstrap_limit = 8 * 1024 * 1024 * 1024
    run_spec_fd, run_spec_snapshot = _open_anchored(
        run_spec_path, None, bootstrap_limit
    )
    descriptors: dict[str, int] = {"run_spec": run_spec_fd}
    snapshots: dict[str, dict[str, Any]] = {"run_spec": run_spec_snapshot}
    private_root: Path | None = None
    try:
        run_spec_payload = _read_fd(run_spec_fd, bootstrap_limit)
        spec = parse_json(run_spec_payload, "parity run-spec")
        validate_run_spec(spec)
        limit = spec["max_rss_bytes"]
        if spec["runtime"] != runtime_identity():
            _stop("parent runtime mismatch")
        requested_sandbox = _absolute_path(
            str(sandbox_executable), "sandbox executable"
        )
        requested_python = _absolute_path(
            str(python_executable or _default_python_executable()),
            "python executable",
        )
        if requested_sandbox != Path(spec["sandbox_executable_path"]):
            _stop("sandbox executable differs from locked run-spec")
        if requested_python != Path(spec["python_executable_path"]):
            _stop("python executable differs from locked run-spec")
        for name, path, digest in (
            (
                "sandbox_executable",
                requested_sandbox,
                spec["sandbox_executable_sha256"],
            ),
            (
                "python_executable",
                requested_python,
                spec["python_executable_sha256"],
            ),
        ):
            descriptors[name], snapshots[name] = _open_anchored(
                path, digest, limit
            )
            if not os.fstat(descriptors[name]).st_mode & 0o111:
                _stop(f"{name} is not executable")
        requested_audit_root = _absolute_path(str(audit_root), "audit root")
        if path_commitment(requested_audit_root) != spec[
            "audit_root_path_sha256"
        ]:
            _stop("audit root differs from locked run-spec")
        audit_root = requested_audit_root
        source_path = Path(__file__).resolve(strict=True)
        profile_path = profile_path.absolute()
        expected_source_hash = spec["parity_source_hashes"][SOURCE_RELATIVE_PATH]
        descriptors["source"], snapshots["source"] = _open_anchored(
            source_path, expected_source_hash, limit
        )
        profile_payload = _profile_bytes(
            profile_path, spec["parity_profile_sha256"], limit
        )
        inputs = {
            "safe_queries": (
                Path(spec["safe_queries_path"]),
                spec["safe_queries_sha256"],
            ),
            "safe_manifest": (
                Path(spec["safe_manifest_path"]),
                spec["safe_manifest_sha256"],
            ),
            "worker_manifest": (
                Path(spec["worker_manifest_path"]),
                spec["worker_manifest_sha256"],
            ),
            **{
                name: (
                    Path(spec["worker_file_paths"][name]),
                    spec["worker_file_hashes"][name],
                )
                for name in WORKER_FILE_NAMES
            },
        }
        for name, (path, digest) in inputs.items():
            descriptors[name], snapshots[name] = _open_anchored(
                path, digest, limit
            )
        parity_id = parity_build_id(spec, run_spec_snapshot["sha256"])
        audit_root.mkdir(parents=True, exist_ok=True)
        if stat.S_ISLNK(os.lstat(audit_root).st_mode) or not audit_root.is_dir():
            _stop("audit root must be a real directory")
        pending = audit_root / f".pending-{parity_id}"
        final = audit_root / parity_id
        pending_exists = os.path.lexists(pending)
        final_exists = os.path.lexists(final)
        if pending_exists and final_exists:
            _stop("conflicting pending and final parity publications")
        if final_exists:
            final_fd = _open_directory_anchored(final)
            try:
                report = validate_published_audit(
                    final,
                    parity_id=parity_id,
                    spec=spec,
                    run_spec_sha256=run_spec_snapshot["sha256"],
                    limit=limit,
                    directory_fd=final_fd,
                )
                _verify_anchored_snapshots(
                    descriptors,
                    snapshots,
                    limit,
                    phase="during final recovery validation",
                )
            finally:
                os.close(final_fd)
            return final, report
        if pending_exists:
            pending_fd = _open_directory_anchored(pending)
            try:
                report = validate_published_audit(
                    pending,
                    parity_id=parity_id,
                    spec=spec,
                    run_spec_sha256=run_spec_snapshot["sha256"],
                    limit=limit,
                    directory_fd=pending_fd,
                )
                _verify_anchored_snapshots(
                    descriptors,
                    snapshots,
                    limit,
                    phase="before pending recovery promotion",
                )
                _promote(pending, final, source_fd=pending_fd)
                _verify_anchored_snapshots(
                    descriptors,
                    snapshots,
                    limit,
                    phase="during pending recovery promotion",
                )
            finally:
                os.close(pending_fd)
            return final, report
        temp_root = Path(spec["temp_root"])
        temp_root.mkdir(parents=True, exist_ok=True)
        if stat.S_ISLNK(os.lstat(temp_root).st_mode) or not temp_root.is_dir():
            _stop("temp root must be a real directory")
        private_root = Path(
            tempfile.mkdtemp(prefix=".run-", dir=temp_root)
        ).absolute()
        _ACTIVE_PRIVATE_ROOTS.add(private_root)
        os.chmod(private_root, 0o700)
        output = private_root / "output"
        tmp = private_root / "tmp"
        output.mkdir(mode=0o700)
        tmp.mkdir(mode=0o700)
        python_path = requested_python
        forbidden = {
            "oracle": forbidden_oracle,
            "oracle_audit": forbidden_oracle_audit,
            "historical": forbidden_historical,
            "model": forbidden_model,
            "store": forbidden_store,
            "write": write_sentinel,
        }
        if any(
            not Path(value).is_absolute() or ".." in Path(value).parts
            for value in forbidden.values()
        ):
            _stop("sandbox sentinel paths must be absolute")
        if os.path.lexists(write_sentinel):
            _stop("write sentinel must be absent before sandbox execution")
        command = _sandbox_command(
            sandbox_executable=requested_sandbox,
            python_executable=python_path,
            profile=profile_payload.decode("utf-8"),
            run_root=private_root,
            output=output,
            tmp=tmp,
            source_fd=descriptors["source"],
            input_fds=descriptors,
            parity_id=parity_id,
            forbidden=forbidden,
        )
        pass_fds = tuple(
            descriptors[name]
            for name in (
                "source",
                "run_spec",
                "safe_queries",
                "safe_manifest",
                "worker_manifest",
                "integrity.json",
                "query_status.parquet",
                "candidates_top100.parquet",
            )
        )
        result = subprocess.run(
            command,
            cwd=private_root,
            env={
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "JOBLIB_MULTIPROCESSING": "0",
                "TMPDIR": str(tmp),
                "V412_PARITY_SANDBOX_CHILD": "1",
            },
            pass_fds=pass_fds,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            _stop(
                f"sandbox parity child failed rc={result.returncode}: "
                f"{result.stderr[-1000:]}"
            )
        if result.stdout or result.stderr:
            _stop("sandbox parity child emitted unexpected output")
        if {path.name for path in output.iterdir()} != {"parity.json"}:
            _stop("sandbox output file-set mismatch")
        report, report_snapshot = _read_path_json_snapshot(
            output / "parity.json", limit
        )
        if (
            set(report) != PARITY_KEYS
            or report["parity_build_id"] != parity_id
            or report["worker_build_id"] != spec["worker_build_id"]
        ):
            _stop("sandbox parity report mismatch")
        _verify_anchored_snapshots(
            descriptors,
            snapshots,
            limit,
            phase="during child execution",
        )
        provenance = {
            "schema_version": PROVENANCE_SCHEMA,
            "parity_build_id": parity_id,
            "worker_build_id": spec["worker_build_id"],
            "git_commit": spec["git_commit"],
            "parity_source_hashes": spec["parity_source_hashes"],
            "lock_sha256": spec["lock_sha256"],
            "parity_run_spec_sha256": run_spec_snapshot["sha256"],
            "worker_manifest_sha256": spec["worker_manifest_sha256"],
            "runtime": spec["runtime"],
            "declarations": dict(DECLARATIONS),
        }
        _write_exclusive(
            output / "provenance.json", canonical_json(provenance)
        )
        _, provenance_snapshot = _read_path_json_snapshot(
            output / "provenance.json", limit
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "parity_build_id": parity_id,
            "worker_build_id": spec["worker_build_id"],
            "files": sorted(
                [
                    _record_from_snapshot("parity.json", report_snapshot),
                    _record_from_snapshot(
                        "provenance.json", provenance_snapshot
                    ),
                ],
                key=lambda row: row["path"].encode(),
            ),
            "runtime": spec["runtime"],
            "declarations": dict(DECLARATIONS),
            "verdict": report["verdict"],
        }
        _write_exclusive(output / "manifest.json", canonical_json(manifest))
        validate_published_audit(
            output,
            parity_id=parity_id,
            spec=spec,
            run_spec_sha256=run_spec_snapshot["sha256"],
            limit=limit,
        )
        _promote(output, pending)
        pending_fd = _open_directory_anchored(pending)
        try:
            validate_published_audit(
                pending,
                parity_id=parity_id,
                spec=spec,
                run_spec_sha256=run_spec_snapshot["sha256"],
                limit=limit,
                directory_fd=pending_fd,
            )
            _verify_anchored_snapshots(
                descriptors,
                snapshots,
                limit,
                phase="before publication",
            )
            _promote(pending, final, source_fd=pending_fd)
        finally:
            os.close(pending_fd)
        validate_published_audit(
            final,
            parity_id=parity_id,
            spec=spec,
            run_spec_sha256=run_spec_snapshot["sha256"],
            limit=limit,
        )
        return final, report
    finally:
        for descriptor in descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        if private_root is not None and private_root in _ACTIVE_PRIVATE_ROOTS:
            _safe_cleanup_private(private_root)


def _parent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).resolve().parents[1] / PROFILE_RELATIVE_PATH,
    )
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--forbidden-oracle", required=True)
    parser.add_argument("--forbidden-oracle-audit", required=True)
    parser.add_argument("--forbidden-historical", required=True)
    parser.add_argument("--forbidden-model", required=True)
    parser.add_argument("--forbidden-store", required=True)
    parser.add_argument("--write-sentinel", required=True)
    parser.add_argument(
        "--sandbox-executable",
        type=Path,
        default=Path("/usr/bin/sandbox-exec"),
    )
    parser.add_argument("--python-executable", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--sandbox-child" in arguments:
            return sandbox_child(arguments)
        args = _parent_parser().parse_args(arguments)
        final, report = audit_and_publish(
            args.run_spec,
            args.profile,
            args.audit_root,
            forbidden_oracle=args.forbidden_oracle,
            forbidden_oracle_audit=args.forbidden_oracle_audit,
            forbidden_historical=args.forbidden_historical,
            forbidden_model=args.forbidden_model,
            forbidden_store=args.forbidden_store,
            write_sentinel=args.write_sentinel,
            sandbox_executable=args.sandbox_executable,
            python_executable=args.python_executable,
        )
        print(
            json.dumps(
                {
                    "verdict": report["verdict"],
                    "parity_build_id": report["parity_build_id"],
                    "audit": str(final),
                },
                sort_keys=True,
            )
        )
        return 0 if report["verdict"] == GO else 3
    except ParityStopped as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
