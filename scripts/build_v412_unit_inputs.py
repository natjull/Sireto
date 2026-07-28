#!/usr/bin/env python3
"""Build the physically blind V4.12 unit-engine input package.

This module deliberately does not import any SIRETO matching code.  It only
projects the six authorised CRM columns, inventories immutable byte streams,
and publishes a runtime package plus a physically separate provenance proof.
"""

from __future__ import annotations

import argparse
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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


STOP = "STOP_V412_UNIT_INPUTS"
LOCK_KEYS = {
    "schema_version",
    "purpose",
    "audit_verdict",
    "git_commit",
    "source_hashes",
    "input_paths",
    "input_hashes",
    "partition_inventory_sha256",
    "tfidf_inventory_sha256",
    "runtime",
    "output_root",
    "audit_output_root",
    "temp_root",
    "max_rss_bytes",
}
LOCK_SCHEMA = "sireto-v4.12-unit-input-execution-lock-1"
LOCK_PURPOSE = "V4.12_UNIT_INPUTS"
LOCK_VERDICT = "GO_CODE_V412_UNIT_INPUTS"
CANONICAL_PLAN_PATH = Path("config/v4_12_unit_input_plan.json")
RUNTIME_FILES = {
    "queries_all.parquet",
    "queries_dev.parquet",
    "partition_inventory.parquet",
    "tfidf_inventory.parquet",
    "integrity.json",
    "runtime_manifest.json",
}
AUDIT_FILES = {"data_inputs.parquet", "provenance.json", "manifest.json"}
DECLARATIONS = {
    "labels_opened": False,
    "oracle_opened": False,
    "models_opened": False,
    "candidate_results_opened": False,
}


class BuildStopped(RuntimeError):
    """Fail-closed V4.12 input build."""


def _stop(message: str) -> None:
    raise BuildStopped(f"{STOP}: {message}")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _stop(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> dict[str, Any]:
    _assert_regular_path(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
        )
    except BuildStopped:
        raise
    except Exception as exc:
        _stop(f"invalid JSON {path.name}: {exc}")
    if not isinstance(value, dict):
        _stop(f"{path.name} must contain a JSON object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
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


def sha256_file(path: Path, max_rss_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
            if max_rss_bytes is not None:
                _check_rss(max_rss_bytes)
    return digest.hexdigest()


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _check_rss(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        _stop("max_rss_bytes must be a positive integer")
    current = _rss_bytes()
    if current > limit:
        _stop(f"RSS limit exceeded: {current} > {limit}")


def _assert_no_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            _stop(f"path component does not exist: {current}")
        if stat.S_ISLNK(mode):
            _stop(f"symlink path component forbidden: {current}")


def _assert_regular_path(path: Path) -> os.stat_result:
    _assert_no_symlink_components(path)
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        _stop(f"regular file required: {path}")
    return info


def _assert_directory_path(path: Path) -> os.stat_result:
    _assert_no_symlink_components(path)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode):
        _stop(f"directory required: {path}")
    return info


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _assert_directory_path(path)


def _snapshot_file(path: Path, max_rss_bytes: int) -> dict[str, Any]:
    info = _assert_regular_path(path)
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "sha256": sha256_file(path, max_rss_bytes),
    }


def _same_snapshot(path: Path, before: Mapping[str, Any], max_rss_bytes: int) -> str:
    after = _snapshot_file(path, max_rss_bytes)
    if dict(before) != after:
        _stop(f"TOCTOU mutation detected: {path}")
    return str(after["sha256"])


def _schema(fields: Sequence[tuple[str, pa.DataType]]) -> pa.Schema:
    return pa.schema([pa.field(name, kind, nullable=False) for name, kind in fields])


QUERY_SCHEMA = _schema(
    [
        ("query_id", pa.string()),
        ("crm_name", pa.string()),
        ("crm_address", pa.string()),
        ("crm_postcode", pa.string()),
        ("crm_city", pa.string()),
        ("crm_insee", pa.string()),
    ]
)
PARTITION_SCHEMA = _schema(
    [("relative_path", pa.string()), ("size_bytes", pa.uint64()), ("sha256", pa.string())]
)
TFIDF_SCHEMA = _schema(
    [
        ("partition_key", pa.string()),
        ("pickle_relative_path", pa.string()),
        ("pickle_size_bytes", pa.uint64()),
        ("pickle_sha256", pa.string()),
        ("sidecar_relative_path", pa.string()),
        ("sidecar_size_bytes", pa.uint64()),
        ("sidecar_sha256", pa.string()),
    ]
)
LEDGER_SCHEMA = _schema(
    [
        ("role", pa.string()),
        ("absolute_path", pa.string()),
        ("projection", pa.string()),
        ("size_bytes_before", pa.uint64()),
        ("sha256_before", pa.string()),
        ("size_bytes_after", pa.uint64()),
        ("sha256_after", pa.string()),
    ]
)


def _table_from_rows(schema: pa.Schema, rows: Iterable[Sequence[Any]]) -> pa.Table:
    columns = list(zip(*rows, strict=True)) if schema.names else []
    if not columns:
        arrays = [pa.array([], type=field.type) for field in schema]
    else:
        arrays = [pa.array(column, type=field.type) for column, field in zip(columns, schema, strict=True)]
    return pa.Table.from_arrays(arrays, schema=schema.remove_metadata())


def _read_projection(path: Path, columns: Sequence[str]) -> pa.Table:
    parquet = pq.ParquetFile(path)
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        _stop(f"missing projected columns in {path.name}: {sorted(missing)}")
    batches = parquet.iter_batches(columns=list(columns), use_threads=False)
    table = pa.Table.from_batches(list(batches))
    table = table.select(list(columns))
    arrays = [table.column(name).combine_chunks() for name in columns]
    schema = _schema([(name, pa.string()) for name in columns])
    try:
        return pa.Table.from_arrays(arrays, schema=schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        _stop(f"projected schema is not exact VARCHAR: {exc}")


def _validate_no_nulls(table: pa.Table, label: str) -> None:
    if any(table.column(name).null_count for name in table.column_names):
        _stop(f"{label} contains null values")
    for field in table.schema:
        if field.nullable or field.type != pa.string():
            _stop(f"{label} schema must be non-nullable VARCHAR")


def _query_order(table: pa.Table, namespace: str) -> list[int]:
    ids = table.column("query_id").to_pylist()
    return sorted(
        range(len(ids)),
        key=lambda index: (
            hashlib.sha256((namespace + ids[index]).encode("utf-8")).hexdigest(),
            ids[index].encode("utf-8"),
        ),
    )


def _take_and_validate_order(
    table: pa.Table, spec: Mapping[str, Any], label: str
) -> pa.Table:
    order = _query_order(table, str(spec["namespace"]))
    result = table.take(pa.array(order, type=pa.int64()))
    ids = result.column("query_id").to_pylist()
    if len(ids) != spec["query_count"] or len(set(ids)) != len(ids):
        _stop(f"{label} count or uniqueness mismatch")
    payload = b"".join(value.encode("utf-8") + b"\n" for value in ids)
    if len(payload) != spec["payload_lf_bytes"]:
        _stop(f"{label} LF payload size mismatch")
    if hashlib.sha256(payload).hexdigest() != spec["payload_lf_sha256"]:
        _stop(f"{label} LF payload hash mismatch")
    if ids[:5] != spec["first_query_ids"] or ids[-5:] != spec["last_query_ids"]:
        _stop(f"{label} boundary IDs mismatch")
    return result


def build_query_tables(
    queries_path: Path, split_path: Path, plan: Mapping[str, Any]
) -> tuple[pa.Table, pa.Table]:
    query_projection = plan["queries"]["projection"]
    split_projection = plan["split_assignments"]["projection"]
    if query_projection != QUERY_SCHEMA.names or split_projection != ["query_id", "split"]:
        _stop("physical projection differs from the six/two authorised columns")
    queries = _read_projection(queries_path, query_projection)
    split = _read_projection(split_path, split_projection)
    _validate_no_nulls(queries, "queries")
    _validate_no_nulls(split, "split assignments")
    if queries.num_rows != plan["queries"]["row_count"]:
        _stop("queries row count mismatch")
    if split.num_rows != plan["split_assignments"]["row_count"]:
        _stop("split row count mismatch")
    query_ids = queries.column("query_id").to_pylist()
    split_ids = split.column("query_id").to_pylist()
    if len(set(query_ids)) != len(query_ids) or len(set(split_ids)) != len(split_ids):
        _stop("query_id is not unique")
    if set(query_ids) != set(split_ids):
        _stop("queries and split query_id sets differ")
    split_values = split.column("split").to_pylist()
    observed: dict[str, int] = {}
    for value in split_values:
        observed[value] = observed.get(value, 0) + 1
    if observed != plan["split_assignments"]["counts"]:
        _stop("fit/dev split counts mismatch")
    split_by_id = dict(zip(split_ids, split_values, strict=True))
    all_table = _take_and_validate_order(queries, plan["orders"]["all"], "queries_all")
    dev_mask = pa.array([split_by_id[value] == "dev" for value in query_ids])
    dev_source = queries.filter(dev_mask)
    dev_table = _take_and_validate_order(dev_source, plan["orders"]["dev"], "queries_dev")
    return all_table, dev_table


def _walk_regular_files(root: Path) -> list[Path]:
    _assert_directory_path(root)
    result: list[Path] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for dirname in list(dirnames):
            child = current_path / dirname
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                _stop(f"non-directory or symlink inside source tree: {child}")
        for filename in filenames:
            child = current_path / filename
            _assert_regular_path(child)
            result.append(child)
    return sorted(result, key=lambda p: p.relative_to(root).as_posix().encode("utf-8"))


@dataclass(frozen=True)
class InventoryResult:
    table: pa.Table
    logical_sha256: str
    total_size: int
    row_count: int
    snapshots: dict[Path, dict[str, Any]]
    roles: dict[Path, str]


def inventory_partitions(root: Path, spec: Mapping[str, Any], max_rss: int) -> InventoryResult:
    files = _walk_regular_files(root)
    rows: list[tuple[Any, ...]] = []
    snapshots: dict[Path, dict[str, Any]] = {}
    roles: dict[Path, str] = {}
    digest = hashlib.sha256()
    historical_digest = hashlib.sha256()
    total_size = 0
    footer_rows = 0
    candidate_count = 0
    manifest_paths = {
        spec["insee_manifest"]["relative_path"],
        spec["postcode_manifest"]["relative_path"],
    }
    for path in files:
        relative = path.relative_to(root).as_posix()
        snapshot = _snapshot_file(path, max_rss)
        snapshots[path] = snapshot
        size = int(snapshot["size"])
        sha = str(snapshot["sha256"])
        total_size += size
        rows.append((relative, size, sha))
        digest.update(relative.encode("utf-8") + b"\0" + str(size).encode("ascii") + b"\0" + sha.encode("ascii") + b"\n")
        historical_digest.update(relative.encode("utf-8") + sha.encode("ascii"))
        if relative in manifest_paths:
            roles[path] = "partition_manifest"
        elif (relative.startswith("insee/") or relative.startswith("cp/")) and relative.endswith(".parquet"):
            roles[path] = "candidate_partition"
            footer_rows += pq.ParquetFile(path).metadata.num_rows
            candidate_count += 1
        else:
            _stop(f"unexpected partition-tree file: {relative}")
    expected_candidate_count = spec["expected_file_count"] - len(manifest_paths)
    checks = (
        (len(files) == spec["expected_file_count"], "partition file count"),
        (candidate_count == expected_candidate_count, "candidate Parquet count"),
        (total_size == spec["expected_size_bytes"], "partition byte size"),
        (footer_rows == spec["expected_row_count"], "partition footer row count"),
        (digest.hexdigest() == spec["expected_inventory_sha256"], "partition inventory hash"),
        (
            historical_digest.hexdigest() == spec["expected_runtime_signature"],
            "partition historical runtime signature",
        ),
    )
    for valid, label in checks:
        if not valid:
            _stop(f"{label} mismatch")
    for name in ("insee_manifest", "postcode_manifest"):
        expected = spec[name]
        path = root / expected["relative_path"]
        snapshot = snapshots.get(path)
        if snapshot is None or snapshot["size"] != expected["size_bytes"] or snapshot["sha256"] != expected["sha256"]:
            _stop(f"{name} mismatch")
    return InventoryResult(
        _table_from_rows(PARTITION_SCHEMA, rows),
        digest.hexdigest(),
        total_size,
        footer_rows,
        snapshots,
        roles,
    )


def _safe_key(partition_key: str) -> str:
    return partition_key.replace("|", "_").replace("/", "_").replace("\\", "_")


def inventory_cache(root: Path, spec: Mapping[str, Any], max_rss: int) -> InventoryResult:
    if root.name != spec["namespace"]:
        _stop("cache namespace/path mismatch")
    files = _walk_regular_files(root)
    if any(path.parent != root for path in files):
        _stop("cache files must be direct children of the cache root")
    pickle_by_name = {p.name[:-4]: p for p in files if p.name.endswith(".pkl")}
    sidecar_by_name = {
        p.name[: -len(".pkl.sha256.json")]: p
        for p in files
        if p.name.endswith(".pkl.sha256.json")
    }
    expected_names = {p.name for p in pickle_by_name.values()} | {p.name for p in sidecar_by_name.values()}
    if expected_names != {p.name for p in files}:
        _stop("extra cache file")
    counts = (
        len(files),
        len(pickle_by_name),
        len(sidecar_by_name),
    )
    if counts != (
        spec["expected_file_count"],
        spec["expected_pickle_count"],
        spec["expected_sidecar_count"],
    ):
        _stop("cache file count mismatch")
    if set(pickle_by_name) != set(sidecar_by_name):
        _stop("missing pickle or sidecar")
    regex = re.compile(spec["safe_key_regex"])
    records: list[tuple[Any, ...]] = []
    snapshots: dict[Path, dict[str, Any]] = {}
    roles: dict[Path, str] = {}
    seen_keys: set[str] = set()
    seen_safe: set[str] = set()
    seen_paths: set[str] = set()
    for filename_key in pickle_by_name:
        pickle_path = pickle_by_name[filename_key]
        sidecar_path = sidecar_by_name[filename_key]
        pickle_snapshot = _snapshot_file(pickle_path, max_rss)
        sidecar_snapshot = _snapshot_file(sidecar_path, max_rss)
        snapshots[pickle_path] = pickle_snapshot
        snapshots[sidecar_path] = sidecar_snapshot
        roles[pickle_path] = "tfidf_pickle"
        roles[sidecar_path] = "tfidf_sidecar"
        metadata = load_json_strict(sidecar_path)
        exact_keys = {"schema_version", "config_hash", "partition_key", "size_bytes", "sha256"}
        if set(metadata) != exact_keys:
            _stop(f"sidecar key set mismatch: {sidecar_path.name}")
        partition_key = metadata["partition_key"]
        if not isinstance(partition_key, str):
            _stop("partition_key must be a string")
        safe = _safe_key(partition_key)
        if safe != filename_key or regex.fullmatch(safe) is None:
            _stop(f"non-canonical safe_key: {filename_key}")
        pickle_rel = pickle_path.relative_to(root).as_posix()
        sidecar_rel = sidecar_path.relative_to(root).as_posix()
        if partition_key in seen_keys or safe in seen_safe or pickle_rel in seen_paths or sidecar_rel in seen_paths:
            _stop("cache key/safe_key/path collision")
        seen_keys.add(partition_key)
        seen_safe.add(safe)
        seen_paths.update((pickle_rel, sidecar_rel))
        if metadata["schema_version"] != spec["sidecar_schema_version"]:
            _stop("sidecar schema version mismatch")
        if metadata["config_hash"] != spec["namespace"]:
            _stop("sidecar config hash mismatch")
        if metadata["size_bytes"] != pickle_snapshot["size"] or metadata["sha256"] != pickle_snapshot["sha256"]:
            _stop("sidecar does not seal the real pickle")
        records.append(
            (
                partition_key,
                pickle_rel,
                pickle_snapshot["size"],
                pickle_snapshot["sha256"],
                sidecar_rel,
                sidecar_snapshot["size"],
                sidecar_snapshot["sha256"],
            )
        )
    records.sort(key=lambda row: tuple(str(value).encode("utf-8") for value in (row[0], row[1], row[4])))
    digest = hashlib.sha256()
    for row in records:
        digest.update(b"\0".join(str(value).encode("utf-8") for value in row) + b"\n")
    total_size = sum(int(value["size"]) for value in snapshots.values())
    checks = (
        (len(records) == spec["expected_key_count"], "cache key count"),
        (total_size == spec["expected_size_bytes"], "cache byte size"),
        (digest.hexdigest() == spec["expected_inventory_sha256"], "cache inventory hash"),
    )
    for valid, label in checks:
        if not valid:
            _stop(f"{label} mismatch")
    return InventoryResult(
        _table_from_rows(TFIDF_SCHEMA, records),
        digest.hexdigest(),
        total_size,
        len(records),
        snapshots,
        roles,
    )


def _runtime_values() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "pyarrow": pa.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def _git(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _stop(f"Git verification failed: {exc}")
    return result.stdout.strip()


def _git_bytes(args: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _stop(f"Git blob verification failed: {exc}")
    return result.stdout


def _verify_git_sources(
    repo: Path, sources: Sequence[str], expected: Mapping[str, str], commit: str, max_rss: int
) -> dict[Path, dict[str, Any]]:
    if _git(["-C", str(repo), "rev-parse", "HEAD"]) != commit:
        _stop("Git HEAD differs from execution lock")
    if set(expected) != set(sources):
        _stop("source_hashes key set mismatch")
    snapshots: dict[Path, dict[str, Any]] = {}
    for relative in sources:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            _stop(f"unsafe source path: {relative}")
        if _git(["-C", str(repo), "ls-files", "--error-unmatch", "--", relative]) != relative:
            _stop(f"source is not tracked by Git: {relative}")
        path = repo / relative
        snapshot = _snapshot_file(path, max_rss)
        if snapshot["sha256"] != expected[relative]:
            _stop(f"source hash mismatch: {relative}")
        blob = _git_bytes(["-C", str(repo), "show", f"{commit}:{relative}"])
        if hashlib.sha256(blob).hexdigest() != expected[relative]:
            _stop(f"committed Git blob differs from worktree and lock: {relative}")
        snapshots[path] = snapshot
    return snapshots


def validate_lock(
    lock: Mapping[str, Any], plan: Mapping[str, Any], repo: Path, max_rss: int
) -> dict[Path, dict[str, Any]]:
    if set(lock) != LOCK_KEYS:
        _stop("execution lock key set mismatch")
    if lock["schema_version"] != LOCK_SCHEMA or lock["purpose"] != LOCK_PURPOSE:
        _stop("execution lock identity mismatch")
    if lock["audit_verdict"] != LOCK_VERDICT:
        _stop(f"execution lock audit verdict is not {LOCK_VERDICT}")
    expected_paths = {
        "queries": plan["queries"]["path"],
        "split_assignments": plan["split_assignments"]["path"],
        "partitions": plan["partitions"]["path"],
        "cache": plan["cache"]["path"],
    }
    expected_hashes = {
        "queries": plan["queries"]["sha256"],
        "split_assignments": plan["split_assignments"]["sha256"],
    }
    comparisons = (
        (lock["input_paths"] == expected_paths, "input paths"),
        (lock["input_hashes"] == expected_hashes, "input hashes"),
        (lock["partition_inventory_sha256"] == plan["partitions"]["expected_inventory_sha256"], "partition hash"),
        (lock["tfidf_inventory_sha256"] == plan["cache"]["expected_inventory_sha256"], "cache hash"),
        (lock["runtime"] == plan["runtime"], "runtime"),
        (lock["output_root"] == plan["output_root"], "output root"),
        (lock["audit_output_root"] == plan["audit_output_root"], "audit output root"),
        (lock["temp_root"] == plan["temp_root"], "temp root"),
        (lock["max_rss_bytes"] == plan["max_rss_bytes"] == max_rss, "RSS limit"),
    )
    for valid, label in comparisons:
        if not valid:
            _stop(f"execution lock {label} mismatch")
    if lock["runtime"] != _runtime_values():
        _stop("actual runtime differs from frozen runtime")
    return _verify_git_sources(
        repo,
        plan["sources"],
        lock["source_hashes"],
        lock["git_commit"],
        max_rss,
    )


def _write_parquet(path: Path, table: pa.Table) -> None:
    pq.write_table(table.replace_schema_metadata(None), path, compression="zstd")
    _fsync_file(path)


def _write_json(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parquet_record(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": parquet.metadata.num_rows,
        "schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
        "metadata": None if schema.metadata is None else dict(schema.metadata),
    }


def _validate_parquet_exact(path: Path, expected: pa.Table) -> None:
    actual = pq.read_table(path)
    if actual.schema != expected.schema.remove_metadata():
        _stop(f"output schema mismatch: {path.name}")
    if not actual.equals(expected):
        _stop(f"output values mismatch: {path.name}")


def _contains_sensitive_runtime_value(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return value.startswith("/") or any(
            token in lowered
            for token in ("labels.parquet", "source_hashes", "input_paths", "provenance.json")
        )
    if isinstance(value, dict):
        return any(_contains_sensitive_runtime_value(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_sensitive_runtime_value(item) for item in value)
    return False


def _build_id(
    plan_bytes: bytes,
    lock_bytes: bytes,
    source_hashes: Mapping[str, str],
    input_hashes: Mapping[str, str],
    partition_hash: str,
    cache_hash: str,
    runtime: Mapping[str, str],
) -> str:
    identity = {
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "partition_inventory_sha256": partition_hash,
        "tfidf_inventory_sha256": cache_hash,
        "runtime": runtime,
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _ledger_table(
    entries: Sequence[tuple[str, Path, str, Mapping[str, Any]]],
    max_rss: int,
) -> pa.Table:
    rows = []
    for role, path, projection, before in entries:
        after = _snapshot_file(path, max_rss)
        if dict(before) != after:
            _stop(f"TOCTOU mutation detected before publication: {path}")
        rows.append(
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
    rows.sort(key=lambda row: (row[0].encode("utf-8"), row[1].encode("utf-8")))
    return _table_from_rows(LEDGER_SCHEMA, rows)


def _validate_ledger_exact(
    table: pa.Table,
    entries: Sequence[tuple[str, Path, str, Mapping[str, Any]]],
) -> None:
    if table.schema != LEDGER_SCHEMA or table.num_rows != len(entries):
        _stop("data input ledger schema or row count mismatch")
    expected = {
        (
            role,
            str(path.absolute()),
            projection,
            int(before["size"]),
            str(before["sha256"]),
            int(before["size"]),
            str(before["sha256"]),
        )
        for role, path, projection, before in entries
    }
    actual = set(zip(*(table.column(name).to_pylist() for name in table.column_names), strict=True))
    if len(actual) != table.num_rows or actual != expected:
        _stop("data input ledger is not exact and exhaustive")


def _chmod_immutable_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        os.chmod(path, 0o555)
    os.chmod(root, 0o555)


def _validate_runtime_package(runtime_dir: Path) -> dict[str, Any]:
    _assert_directory_path(runtime_dir)
    runtime_names = {p.name for p in runtime_dir.iterdir()}
    if runtime_names != RUNTIME_FILES:
        _stop("runtime file set mismatch")
    for path in runtime_dir.iterdir():
        _assert_regular_path(path)
    runtime_manifest = load_json_strict(runtime_dir / "runtime_manifest.json")
    try:
        expected_files = {
            name: _parquet_record(runtime_dir / name)
            for name in sorted(RUNTIME_FILES)
            if name.endswith(".parquet")
        }
    except Exception as exc:
        _stop(f"invalid runtime Parquet: {exc}")
    expected_files["integrity.json"] = {
        "sha256": sha256_file(runtime_dir / "integrity.json"),
        "size_bytes": (runtime_dir / "integrity.json").stat().st_size,
    }
    if set(runtime_manifest.get("files", {})) != set(expected_files):
        _stop("runtime manifest file records are not exact")
    if runtime_manifest["files"] != expected_files:
        _stop("runtime file mutation or manifest mismatch")
    if _contains_sensitive_runtime_value(runtime_manifest):
        _stop("runtime manifest leaks sensitive provenance")
    return runtime_manifest


def validate_concordance(runtime_dir: Path, audit_dir: Path) -> None:
    runtime_manifest = _validate_runtime_package(runtime_dir)
    _assert_directory_path(audit_dir)
    if runtime_dir.name != audit_dir.name:
        _stop("runtime/audit build IDs differ")
    audit_names = {p.name for p in audit_dir.iterdir()}
    if audit_names != AUDIT_FILES:
        _stop("audit file set mismatch")
    for path in audit_dir.iterdir():
        _assert_regular_path(path)
    provenance = load_json_strict(audit_dir / "provenance.json")
    audit_manifest = load_json_strict(audit_dir / "manifest.json")
    if runtime_manifest.get("build_id") != runtime_dir.name or provenance.get("build_id") != runtime_dir.name:
        _stop("published build_id mismatch")
    if provenance.get("runtime_manifest_sha256") != sha256_file(runtime_dir / "runtime_manifest.json"):
        _stop("audit/runtime manifest mismatch")
    expected_audit = {
        "data_inputs.parquet": {
            "sha256": sha256_file(audit_dir / "data_inputs.parquet"),
            "size_bytes": (audit_dir / "data_inputs.parquet").stat().st_size,
        },
        "provenance.json": {
            "sha256": sha256_file(audit_dir / "provenance.json"),
            "size_bytes": (audit_dir / "provenance.json").stat().st_size,
        },
    }
    if audit_manifest.get("files") != expected_audit or audit_manifest.get("build_id") != runtime_dir.name:
        _stop("audit manifest mismatch")
    try:
        ledger = pq.read_table(audit_dir / "data_inputs.parquet")
    except Exception as exc:
        _stop(f"invalid audit ledger Parquet: {exc}")
    if ledger.schema != LEDGER_SCHEMA:
        _stop("audit ledger schema mismatch")
    if ledger.num_rows != provenance.get("data_input_count"):
        _stop("audit ledger cardinality mismatch")
    ledger_paths = ledger.column("absolute_path").to_pylist()
    if len(set(ledger_paths)) != ledger.num_rows:
        _stop("audit ledger contains duplicate data paths")


def _promote(source: Path, destination: Path) -> None:
    if destination.exists():
        _stop(f"immutable destination already exists: {destination}")
    if os.lstat(source).st_dev != os.lstat(destination.parent).st_dev:
        _stop("staging and destination are on different devices")
    _fsync_dir(source)
    os.rename(source, destination)
    _fsync_dir(destination.parent)
    _chmod_immutable_tree(destination)
    _fsync_dir(destination.parent)


def _process_context(plan: Mapping[str, Any]) -> Path:
    if not sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        _stop("builder must run with python -B and PYTHONDONTWRITEBYTECODE=1")
    staging_text = os.environ.get("SIRETO_V412_STAGING")
    tmp_text = os.environ.get("TMPDIR")
    if not staging_text or not tmp_text:
        _stop("secure staging environment is absent")
    staging = Path(staging_text)
    temp_root = Path(plan["temp_root"])
    _assert_directory_path(staging)
    _assert_directory_path(Path(tmp_text))
    try:
        staging.relative_to(temp_root)
        Path(tmp_text).relative_to(staging)
    except ValueError:
        _stop("TMPDIR must be inside this run staging under temp_root")
    if Path(tmp_text).resolve() != (staging / "tmp").resolve():
        _stop("TMPDIR is not the dedicated staging tmp directory")
    return staging


def build_inputs(plan_path: Path, lock_path: Path) -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    plan_path = plan_path.absolute()
    lock_path = lock_path.absolute()
    if plan_path != (repo / CANONICAL_PLAN_PATH).absolute():
        _stop("only the committed canonical V4.12 input plan may be executed")
    plan = load_json_strict(plan_path)
    lock = load_json_strict(lock_path)
    max_rss = plan.get("max_rss_bytes")
    _check_rss(max_rss)
    lock_snapshot = _snapshot_file(lock_path, max_rss)
    if load_json_strict(lock_path) != lock:
        _stop("execution lock changed while being read")
    staging = _process_context(plan)
    source_snapshots = validate_lock(lock, plan, repo, max_rss)
    if lock_path != (repo / plan["execution_lock_path"]).absolute():
        _stop("execution lock path differs from plan")
    queries_path = Path(plan["queries"]["path"])
    split_path = Path(plan["split_assignments"]["path"])
    partition_root = Path(plan["partitions"]["path"])
    cache_root = Path(plan["cache"]["path"])
    input_paths = (queries_path, split_path)
    input_snapshots = {path: _snapshot_file(path, max_rss) for path in input_paths}
    if (
        input_snapshots[queries_path]["sha256"] != plan["queries"]["sha256"]
        or input_snapshots[queries_path]["size"] != plan["queries"]["size_bytes"]
    ):
        _stop("queries source hash mismatch")
    if (
        input_snapshots[split_path]["sha256"] != plan["split_assignments"]["sha256"]
        or input_snapshots[split_path]["size"] != plan["split_assignments"]["size_bytes"]
    ):
        _stop("split source hash mismatch")
    all_table, dev_table = build_query_tables(queries_path, split_path, plan)
    partition_inventory = inventory_partitions(partition_root, plan["partitions"], max_rss)
    cache_inventory = inventory_cache(cache_root, plan["cache"], max_rss)
    build_id = _build_id(
        plan_path.read_bytes(),
        lock_path.read_bytes(),
        lock["source_hashes"],
        lock["input_hashes"],
        partition_inventory.logical_sha256,
        cache_inventory.logical_sha256,
        plan["runtime"],
    )
    output_root = Path(plan["output_root"])
    audit_root = Path(plan["audit_output_root"])
    _ensure_directory(output_root)
    _ensure_directory(audit_root)
    final_runtime = output_root / build_id
    final_audit = audit_root / build_id
    if final_runtime.exists() or final_audit.exists():
        _stop("runtime or audit destination already exists")
    run_runtime = staging / "runtime"
    run_audit = staging / "audit"
    run_runtime.mkdir()
    run_audit.mkdir()
    _write_parquet(run_runtime / "queries_all.parquet", all_table)
    _write_parquet(run_runtime / "queries_dev.parquet", dev_table)
    _write_parquet(run_runtime / "partition_inventory.parquet", partition_inventory.table)
    _write_parquet(run_runtime / "tfidf_inventory.parquet", cache_inventory.table)
    for name, table in (
        ("queries_all.parquet", all_table),
        ("queries_dev.parquet", dev_table),
        ("partition_inventory.parquet", partition_inventory.table),
        ("tfidf_inventory.parquet", cache_inventory.table),
    ):
        _validate_parquet_exact(run_runtime / name, table)
    integrity = {
        "schema_version": "sireto-v4.12-unit-input-integrity-1",
        "build_id": build_id,
        "partition_inventory_sha256": partition_inventory.logical_sha256,
        "tfidf_inventory_sha256": cache_inventory.logical_sha256,
        "declarations": DECLARATIONS,
    }
    _write_json(run_runtime / "integrity.json", integrity)
    parquet_records = {
        name: _parquet_record(run_runtime / name)
        for name in sorted(RUNTIME_FILES)
        if name.endswith(".parquet")
    }
    runtime_manifest = {
        "schema_version": "sireto-v4.12-unit-runtime-manifest-1",
        "build_id": build_id,
        "files": {
            **parquet_records,
            "integrity.json": {
                "sha256": sha256_file(run_runtime / "integrity.json"),
                "size_bytes": (run_runtime / "integrity.json").stat().st_size,
            },
        },
        "partition_inventory_sha256": partition_inventory.logical_sha256,
        "tfidf_inventory_sha256": cache_inventory.logical_sha256,
        "partition_runtime_signature": plan["partitions"]["expected_runtime_signature"],
        "tfidf_config_artifact_hash": plan["cache"]["tfidf_config_artifact_hash"],
        "runtime": plan["runtime"],
        "declarations": DECLARATIONS,
    }
    if _contains_sensitive_runtime_value(runtime_manifest):
        _stop("runtime manifest would leak sensitive provenance")
    _write_json(run_runtime / "runtime_manifest.json", runtime_manifest)
    staged_manifest = _validate_runtime_package(run_runtime)
    if staged_manifest != runtime_manifest:
        _stop("staged runtime manifest is not bit-exact after reopening")
    ledger_entries: list[tuple[str, Path, str, Mapping[str, Any]]] = [
        ("queries", queries_path, ",".join(plan["queries"]["projection"]), input_snapshots[queries_path]),
        ("split", split_path, ",".join(plan["split_assignments"]["projection"]), input_snapshots[split_path]),
    ]
    ledger_entries.extend(
        (partition_inventory.roles[path], path, "", snapshot)
        for path, snapshot in partition_inventory.snapshots.items()
    )
    ledger_entries.extend(
        (cache_inventory.roles[path], path, "", snapshot)
        for path, snapshot in cache_inventory.snapshots.items()
    )
    ledger = _ledger_table(ledger_entries, max_rss)
    expected_ledger_rows = 2 + plan["partitions"]["expected_file_count"] + plan["cache"]["expected_file_count"]
    if ledger.num_rows != expected_ledger_rows:
        _stop("data input ledger is not exhaustive")
    _validate_ledger_exact(ledger, ledger_entries)
    _write_parquet(run_audit / "data_inputs.parquet", ledger)
    reopened_ledger = pq.read_table(run_audit / "data_inputs.parquet")
    _validate_ledger_exact(reopened_ledger, ledger_entries)
    provenance = {
        "schema_version": "sireto-v4.12-unit-provenance-1",
        "build_id": build_id,
        "git_commit": lock["git_commit"],
        "sources": lock["source_hashes"],
        "runtime": plan["runtime"],
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "data_input_count": ledger.num_rows,
        "declarations": DECLARATIONS,
        "runtime_manifest_sha256": sha256_file(run_runtime / "runtime_manifest.json"),
    }
    _write_json(run_audit / "provenance.json", provenance)
    audit_manifest = {
        "schema_version": "sireto-v4.12-unit-audit-manifest-1",
        "build_id": build_id,
        "files": {
            "data_inputs.parquet": {
                "sha256": sha256_file(run_audit / "data_inputs.parquet"),
                "size_bytes": (run_audit / "data_inputs.parquet").stat().st_size,
            },
            "provenance.json": {
                "sha256": sha256_file(run_audit / "provenance.json"),
                "size_bytes": (run_audit / "provenance.json").stat().st_size,
            },
        },
    }
    _write_json(run_audit / "manifest.json", audit_manifest)
    def revalidate_before_promotion() -> None:
        projected_all, projected_dev = build_query_tables(queries_path, split_path, plan)
        if not projected_all.equals(all_table) or not projected_dev.equals(dev_table):
            _stop("CRM/split projection changed before promotion")
        for path, before in source_snapshots.items():
            _same_snapshot(path, before, max_rss)
        _same_snapshot(lock_path, lock_snapshot, max_rss)
        for path, before in input_snapshots.items():
            _same_snapshot(path, before, max_rss)
        for path, before in partition_inventory.snapshots.items():
            _same_snapshot(path, before, max_rss)
        for path, before in cache_inventory.snapshots.items():
            _same_snapshot(path, before, max_rss)
        _verify_git_sources(
            repo, plan["sources"], lock["source_hashes"], lock["git_commit"], max_rss
        )
        _check_rss(max_rss)

    revalidate_before_promotion()
    _promote(run_audit, final_audit)
    try:
        revalidate_before_promotion()
        _promote(run_runtime, final_runtime)
    except Exception:
        # The immutable orphan audit is intentional fail-closed evidence.
        raise
    validate_concordance(final_runtime, final_audit)
    return final_runtime, final_audit


def _bootstrap(plan_path: Path, lock_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    if plan_path != (repo / CANONICAL_PLAN_PATH).absolute():
        _stop("only the committed canonical V4.12 input plan may be executed")
    plan = load_json_strict(plan_path)
    lock = load_json_strict(lock_path)
    if lock_path != (repo / plan["execution_lock_path"]).absolute():
        _stop("execution lock path differs from plan")
    validate_lock(lock, plan, repo, plan["max_rss_bytes"])
    temp_root = Path(plan["temp_root"])
    _ensure_directory(temp_root)
    staging = temp_root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    tmp = staging / "tmp"
    tmp.mkdir(mode=0o700)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TMPDIR"] = str(tmp)
    environment["SIRETO_V412_STAGING"] = str(staging)
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--internal-run",
        "--plan",
        str(plan_path),
        "--lock",
        str(lock_path),
    ]
    try:
        result = subprocess.run(command, env=environment)
        if result.returncode:
            raise SystemExit(result.returncode)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--internal-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.internal_run:
            runtime_dir, audit_dir = build_inputs(args.plan, args.lock)
            print(
                json.dumps(
                    {
                        "verdict": "GO_V412_UNIT_INPUTS",
                        "build_id": runtime_dir.name,
                        "runtime_dir": str(runtime_dir),
                        "audit_dir": str(audit_dir),
                    },
                    sort_keys=True,
                )
            )
        else:
            _bootstrap(args.plan.absolute(), args.lock.absolute())
    except BuildStopped as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
