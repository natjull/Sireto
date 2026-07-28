#!/usr/bin/env python3
"""Build the physically separate V4.12 development oracle."""

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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


STOP = "STOP_V412_UNIT_ORACLE"
CANONICAL_PLAN_PATH = Path("config/v4_12_unit_oracle_plan.json")
LOCK_SCHEMA = "sireto-v4.12-unit-oracle-execution-lock-1"
LOCK_PURPOSE = "V4.12_UNIT_ORACLE"
LOCK_VERDICT = "GO_CODE_V412_UNIT_ORACLE"
LOCK_KEYS = {
    "schema_version",
    "purpose",
    "audit_verdict",
    "git_commit",
    "source_hashes",
    "input_paths",
    "input_hashes",
    "safe_input_build_id",
    "expected_population",
    "expected_id_payload_sha256",
    "expected_truth_logical_sha256",
    "runtime",
    "output_root",
    "audit_output_root",
    "temp_root",
    "max_rss_bytes",
}
ORACLE_FILES = {"oracle_dev.parquet", "integrity.json", "manifest.json"}
AUDIT_FILES = {"data_inputs.parquet", "provenance.json", "manifest.json"}
SAFE_RUNTIME_FILES = {
    "queries_all.parquet",
    "queries_dev.parquet",
    "partition_inventory.parquet",
    "tfidf_inventory.parquet",
    "integrity.json",
    "runtime_manifest.json",
}
SAFE_MANIFEST_KEYS = {
    "schema_version",
    "build_id",
    "files",
    "partition_inventory_sha256",
    "tfidf_inventory_sha256",
    "partition_runtime_signature",
    "tfidf_config_artifact_hash",
    "runtime",
    "declarations",
}
SAFE_DECLARATIONS = {
    "labels_opened": False,
    "oracle_opened": False,
    "models_opened": False,
    "candidate_results_opened": False,
}
DECLARATIONS = {
    "safe_runtime_files_opened_for_integrity": True,
    "retrieval_results_opened": False,
    "candidate_results_opened": False,
    "direct_evidence_opened": False,
    "guard_decisions_opened": False,
    "models_opened": False,
    "challenge_or_final_opened": False,
}
SAFE_LEDGER_ROLES = {
    "integrity.json": "safe_integrity",
    "partition_inventory.parquet": "safe_partition_inventory",
    "queries_all.parquet": "safe_queries_all",
    "queries_dev.parquet": "safe_queries_dev",
    "runtime_manifest.json": "safe_runtime_manifest",
    "tfidf_inventory.parquet": "safe_tfidf_inventory",
}
LEDGER_ROLES = set(SAFE_LEDGER_ROLES.values()) | {"labels", "split"}
INTEGRITY_KEYS = {
    "schema_version",
    "build_id",
    "query_count",
    "population_counts",
    "query_id_payload_sha256",
    "truth_logical_sha256",
    "declarations",
}
ORACLE_MANIFEST_KEYS = {
    "schema_version",
    "build_id",
    "safe_input_build_id",
    "files",
    "population_counts",
    "query_id_payload_sha256",
    "truth_logical_sha256",
    "runtime",
    "declarations",
    "historical_development_only",
    "independent_truth",
    "production_certified",
}
PROVENANCE_KEYS = {
    "schema_version",
    "build_id",
    "git_commit",
    "sources",
    "lock_sha256",
    "runtime",
    "data_input_count",
    "declarations",
    "oracle_manifest_sha256",
}
PLAN_KEYS = {
    "audit_output_root",
    "execution_lock_path",
    "expected",
    "labels",
    "forbidden_artifacts",
    "max_rss_bytes",
    "output_root",
    "runtime",
    "safe_input",
    "schema_version",
    "sources",
    "split_assignments",
    "temp_root",
}


class BuildStopped(RuntimeError):
    pass


def _stop(message: str) -> None:
    raise BuildStopped(f"{STOP}: {message}")


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _stop(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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


def _assert_no_symlink_components(path: Path) -> None:
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


def _assert_regular(path: Path) -> os.stat_result:
    _assert_no_symlink_components(path)
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        _stop(f"regular file required: {path}")
    return info


def _assert_directory(path: Path) -> os.stat_result:
    _assert_no_symlink_components(path)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode):
        _stop(f"directory required: {path}")
    return info


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _assert_directory(path)


def load_json_strict(path: Path) -> dict[str, Any]:
    _assert_regular(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_pairs,
        )
    except BuildStopped:
        raise
    except Exception as exc:
        _stop(f"invalid JSON {path.name}: {exc}")
    if not isinstance(value, dict):
        _stop(f"{path.name} must contain an object")
    return value


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _check_rss(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        _stop("max_rss_bytes must be a positive integer")
    if _rss_bytes() > limit:
        _stop(f"RSS limit exceeded: {_rss_bytes()} > {limit}")


def sha256_file(path: Path, max_rss: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
            if max_rss is not None:
                _check_rss(max_rss)
    return digest.hexdigest()


def _snapshot(path: Path, max_rss: int) -> dict[str, Any]:
    info = _assert_regular(path)
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "sha256": sha256_file(path, max_rss),
    }


def _require_unchanged(path: Path, before: Mapping[str, Any], max_rss: int) -> None:
    if _snapshot(path, max_rss) != dict(before):
        _stop(f"TOCTOU mutation detected: {path}")


def _runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "pyarrow": pa.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def _schema(fields: Sequence[tuple[str, pa.DataType, bool]]) -> pa.Schema:
    return pa.schema([pa.field(name, kind, nullable=nullable) for name, kind, nullable in fields])


ORACLE_SCHEMA = _schema(
    [
        ("query_id", pa.string(), False),
        ("dev_partition", pa.string(), False),
        ("label_kind", pa.string(), False),
        ("ground_truth_siret", pa.string(), True),
        ("ground_truth_siren", pa.string(), True),
    ]
)
LEDGER_SCHEMA = _schema(
    [
        ("role", pa.string(), False),
        ("absolute_path", pa.string(), False),
        ("projection", pa.string(), False),
        ("size_bytes_before", pa.uint64(), False),
        ("sha256_before", pa.string(), False),
        ("size_bytes_after", pa.uint64(), False),
        ("sha256_after", pa.string(), False),
    ]
)


def _table_from_rows(schema: pa.Schema, rows: Iterable[Sequence[Any]]) -> pa.Table:
    materialized = list(rows)
    if materialized:
        columns = list(zip(*materialized, strict=True))
        arrays = [
            pa.array(values, type=field.type)
            for values, field in zip(columns, schema, strict=True)
        ]
    else:
        arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema.remove_metadata())


def _read_projection(path: Path, projection: Sequence[str], nullable: Mapping[str, bool]) -> pa.Table:
    parquet = pq.ParquetFile(path)
    if not set(projection).issubset(parquet.schema_arrow.names):
        _stop(f"projected columns absent from {path.name}")
    table = pa.Table.from_batches(
        list(parquet.iter_batches(columns=list(projection), use_threads=False))
    ).select(list(projection))
    fields = [(name, pa.string(), nullable.get(name, False)) for name in projection]
    try:
        return pa.Table.from_arrays(
            [table.column(name).combine_chunks() for name in projection],
            schema=_schema(fields),
        )
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        _stop(f"projection schema mismatch: {exc}")


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _stop(f"Git verification failed: {exc}")


def _git_bytes(args: Sequence[str]) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        _stop(f"Git blob verification failed: {exc}")


def _verify_git_sources(
    repo: Path,
    sources: Sequence[str],
    expected: Mapping[str, str],
    commit: str,
    max_rss: int,
) -> dict[Path, dict[str, Any]]:
    if set(sources) != set(expected):
        _stop("source_hashes keyset mismatch")
    snapshots: dict[Path, dict[str, Any]] = {}
    for relative in sources:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            _stop(f"unsafe Git source path: {relative}")
        if _git(["-C", str(repo), "ls-files", "--error-unmatch", "--", relative]) != relative:
            _stop(f"source not tracked: {relative}")
        path = repo / relative
        snapshot = _snapshot(path, max_rss)
        if snapshot["sha256"] != expected[relative]:
            _stop(f"source worktree hash mismatch: {relative}")
        blob = _git_bytes(["-C", str(repo), "show", f"{commit}:{relative}"])
        if hashlib.sha256(blob).hexdigest() != expected[relative]:
            _stop(f"source Git blob hash mismatch: {relative}")
        snapshots[path] = snapshot
    return snapshots


def _expected_input_paths(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        "safe_queries_dev": plan["safe_input"]["queries_dev_path"],
        "safe_runtime_manifest": plan["safe_input"]["manifest_path"],
        "labels": plan["labels"]["path"],
        "split": plan["split_assignments"]["path"],
    }


def _expected_input_hashes(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        "safe_queries_dev": plan["safe_input"]["queries_dev_sha256"],
        "safe_runtime_manifest": plan["safe_input"]["manifest_sha256"],
        "labels": plan["labels"]["sha256"],
        "split": plan["split_assignments"]["sha256"],
    }


def validate_plan(plan: Mapping[str, Any]) -> None:
    if set(plan) != PLAN_KEYS or plan["schema_version"] != "sireto-v4.12-unit-oracle-plan-1":
        _stop("oracle plan keyset or schema mismatch")
    if set(plan["expected"]) != {
        "comparison_dev",
        "first_query_ids",
        "last_query_ids",
        "logical_payload_bytes",
        "logical_payload_sha256",
        "query_id_payload_bytes",
        "query_id_payload_sha256",
        "query_count",
        "threshold_dev",
    }:
        _stop("oracle plan expected keyset mismatch")
    if set(plan["labels"]) != {"path", "projection", "row_count", "sha256", "size_bytes"}:
        _stop("oracle plan labels keyset mismatch")
    if set(plan["split_assignments"]) != {
        "path",
        "projection",
        "row_count",
        "sha256",
        "size_bytes",
    }:
        _stop("oracle plan split keyset mismatch")
    if set(plan["safe_input"]) != {
        "build_id",
        "manifest_path",
        "manifest_sha256",
        "queries_dev_path",
        "queries_dev_projection",
        "queries_dev_sha256",
        "queries_dev_size_bytes",
    }:
        _stop("oracle plan safe_input keyset mismatch")
    if any(set(item) != {"path", "sha256"} for item in plan["forbidden_artifacts"]):
        _stop("oracle plan forbidden_artifacts keyset mismatch")
    for name in ("threshold_dev", "comparison_dev"):
        if set(plan["expected"][name]) != {"total", "MATCH_EXACT", "AMBIGUOUS"}:
            _stop("oracle plan population keyset mismatch")
    if set(plan["runtime"]) != {"python", "duckdb", "pyarrow", "machine", "platform"}:
        _stop("oracle plan runtime keyset mismatch")


def _expected_population(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_count": plan["expected"]["query_count"],
        "threshold_dev": plan["expected"]["threshold_dev"],
        "comparison_dev": plan["expected"]["comparison_dev"],
    }


def _validate_forbidden_boundary(plan: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    allowed = {str(Path(value).absolute()) for value in _expected_input_paths(plan).values()}
    forbidden = {
        str(Path(item["path"]).absolute())
        for item in plan.get("forbidden_artifacts", [])
    }
    if allowed & forbidden:
        _stop("an allowed input is a forbidden artifact")
    lock_paths = {str(Path(value).absolute()) for value in lock["input_paths"].values()}
    if lock_paths != allowed or lock_paths & forbidden:
        _stop("lock attempts to open a forbidden artifact")
    forbidden_tokens = (
        "candidates_sparse_top100.parquet",
        "query_audit.parquet",
        "query_evidence.parquet",
        "decisions.parquet",
        "challenge",
        "holdout",
        "random",
        "ranker",
        "acceptor",
        "model",
        "scene",
    )
    for role, path in lock["input_paths"].items():
        if role not in _expected_input_paths(plan):
            _stop("unknown input role")
        if any(token in path.lower() for token in forbidden_tokens):
            _stop(f"forbidden artifact path: {path}")


def validate_lock(
    lock: Mapping[str, Any], plan: Mapping[str, Any], repo: Path, max_rss: int
) -> dict[Path, dict[str, Any]]:
    if set(lock) != LOCK_KEYS:
        _stop("execution lock keyset mismatch")
    checks = (
        (lock["schema_version"] == LOCK_SCHEMA, "schema"),
        (lock["purpose"] == LOCK_PURPOSE, "purpose"),
        (lock["audit_verdict"] == LOCK_VERDICT, "audit verdict"),
        (lock["input_paths"] == _expected_input_paths(plan), "input paths"),
        (lock["input_hashes"] == _expected_input_hashes(plan), "input hashes"),
        (lock["safe_input_build_id"] == plan["safe_input"]["build_id"], "safe input build"),
        (lock["expected_population"] == _expected_population(plan), "population"),
        (
            lock["expected_id_payload_sha256"]
            == plan["expected"]["query_id_payload_sha256"],
            "ID payload",
        ),
        (
            lock["expected_truth_logical_sha256"]
            == plan["expected"]["logical_payload_sha256"],
            "truth payload",
        ),
        (lock["runtime"] == plan["runtime"] == _runtime(), "runtime"),
        (lock["output_root"] == plan["output_root"], "output root"),
        (lock["audit_output_root"] == plan["audit_output_root"], "audit root"),
        (lock["temp_root"] == plan["temp_root"], "temp root"),
        (lock["max_rss_bytes"] == plan["max_rss_bytes"] == max_rss, "RSS"),
    )
    for valid, label in checks:
        if not valid:
            _stop(f"execution lock {label} mismatch")
    _validate_forbidden_boundary(plan, lock)
    return _verify_git_sources(
        repo, plan["sources"], lock["source_hashes"], lock["git_commit"], max_rss
    )


def _safe_runtime_snapshots(
    root: Path, manifest: Mapping[str, Any], max_rss: int
) -> dict[Path, dict[str, Any]]:
    _assert_directory(root)
    names = {path.name for path in root.iterdir()}
    if names != SAFE_RUNTIME_FILES:
        _stop("safe runtime file-set mismatch")
    result: dict[Path, dict[str, Any]] = {}
    for name in sorted(names):
        path = root / name
        snapshot = _snapshot(path, max_rss)
        result[path] = snapshot
        if name == "runtime_manifest.json":
            continue
        record = manifest["files"].get(name)
        if name == "integrity.json":
            expected = {
                "sha256": snapshot["sha256"],
                "size_bytes": snapshot["size"],
            }
        else:
            try:
                expected = _parquet_record(path)
            except Exception as exc:
                _stop(f"invalid safe runtime Parquet {name}: {exc}")
        if record != expected:
            _stop(f"safe runtime record mismatch: {name}")
    return result


def validate_safe_runtime(
    manifest_path: Path,
    queries_path: Path,
    spec: Mapping[str, Any],
    max_rss: int,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    if manifest_path.parent != queries_path.parent:
        _stop("safe manifest and queries are not siblings")
    manifest_snapshot = _snapshot(manifest_path, max_rss)
    if manifest_snapshot["sha256"] != spec["manifest_sha256"]:
        _stop("safe input hash or size mismatch")
    manifest = load_json_strict(manifest_path)
    if set(manifest) != SAFE_MANIFEST_KEYS:
        _stop("safe runtime manifest keyset mismatch")
    if manifest["schema_version"] != "sireto-v4.12-unit-runtime-manifest-1":
        _stop("safe runtime manifest schema mismatch")
    if manifest["build_id"] != spec["build_id"] or manifest["declarations"] != SAFE_DECLARATIONS:
        _stop("safe runtime identity or declarations mismatch")
    if set(manifest["files"]) != SAFE_RUNTIME_FILES - {"runtime_manifest.json"}:
        _stop("safe runtime manifest file records mismatch")
    sibling_snapshots = _safe_runtime_snapshots(
        manifest_path.parent, manifest, max_rss
    )
    if sibling_snapshots[manifest_path] != manifest_snapshot:
        _stop("safe runtime manifest changed during validation")
    queries_snapshot = sibling_snapshots[queries_path]
    if (
        queries_snapshot["sha256"] != spec["queries_dev_sha256"]
        or queries_snapshot["size"] != spec["queries_dev_size_bytes"]
    ):
        _stop("safe input hash or size mismatch")
    record = manifest["files"]["queries_dev.parquet"]
    expected_record = {
        "sha256": spec["queries_dev_sha256"],
        "size_bytes": spec["queries_dev_size_bytes"],
        "row_count": 1456 if spec["queries_dev_size_bytes"] == 62365 else record.get("row_count"),
        "schema": record.get("schema"),
        "metadata": None,
    }
    if set(record) != {"sha256", "size_bytes", "row_count", "schema", "metadata"}:
        _stop("safe queries manifest record keyset mismatch")
    if (
        record["sha256"] != expected_record["sha256"]
        or record["size_bytes"] != expected_record["size_bytes"]
        or record["metadata"] is not None
    ):
        _stop("safe queries manifest record mismatch")
    return manifest, sibling_snapshots


def _clean_string(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or value == ""
        or value != value.strip()
        or "\0" in value
        or "\n" in value
        or "\r" in value
    ):
        _stop(f"invalid string in {label}")


def build_oracle_table(
    safe_queries_path: Path,
    labels_path: Path,
    split_path: Path,
    plan: Mapping[str, Any],
) -> tuple[pa.Table, dict[str, Any], str, str]:
    if plan["safe_input"]["queries_dev_projection"] != ["query_id"]:
        _stop("safe query projection is not exact")
    if plan["labels"]["projection"] != [
        "query_id",
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
    ]:
        _stop("labels projection is not exact")
    if plan["split_assignments"]["projection"] != [
        "query_id",
        "siren_component_id",
        "split",
    ]:
        _stop("split projection is not exact")
    safe = _read_projection(safe_queries_path, ["query_id"], {})
    labels = _read_projection(
        labels_path,
        plan["labels"]["projection"],
        {"ground_truth_siret": True, "ground_truth_siren": True},
    )
    split = _read_projection(split_path, plan["split_assignments"]["projection"], {})
    if labels.num_rows != plan["labels"]["row_count"] or split.num_rows != plan["split_assignments"]["row_count"]:
        _stop("historical source row count mismatch")
    safe_ids = safe.column("query_id").to_pylist()
    if len(safe_ids) != plan["expected"]["query_count"] or len(set(safe_ids)) != len(safe_ids):
        _stop("safe query count or uniqueness mismatch")
    label_rows = labels.to_pylist()
    split_rows = split.to_pylist()
    if len({row["query_id"] for row in label_rows}) != len(label_rows):
        _stop("labels query_id is not unique")
    if len({row["query_id"] for row in split_rows}) != len(split_rows):
        _stop("split query_id is not unique")
    labels_by_id = {row["query_id"]: row for row in label_rows}
    split_by_id = {row["query_id"]: row for row in split_rows}
    rows = []
    counts = {
        "threshold_dev": {"total": 0, "MATCH_EXACT": 0, "AMBIGUOUS": 0},
        "comparison_dev": {"total": 0, "MATCH_EXACT": 0, "AMBIGUOUS": 0},
    }
    for query_id in safe_ids:
        _clean_string(query_id, "query_id")
        label = labels_by_id.get(query_id)
        split_row = split_by_id.get(query_id)
        if label is None or split_row is None or split_row["split"] != "dev":
            _stop("safe query lacks an exact dev label/split row")
        component = split_row["siren_component_id"]
        _clean_string(component, "siren_component_id")
        partition = (
            "threshold_dev"
            if hashlib.sha256(("v411-threshold:" + component).encode("utf-8")).digest()[0] < 128
            else "comparison_dev"
        )
        kind = label["label_kind"]
        siret = label["ground_truth_siret"]
        siren = label["ground_truth_siren"]
        if kind not in {"MATCH_EXACT", "AMBIGUOUS"}:
            _stop("invalid dev label_kind")
        if kind == "MATCH_EXACT":
            if (
                not isinstance(siret, str)
                or re.fullmatch(r"[0-9]{14}", siret) is None
                or not isinstance(siren, str)
                or re.fullmatch(r"[0-9]{9}", siren) is None
                or siren != siret[:9]
            ):
                _stop("invalid exact SIRET/SIREN truth")
        elif siret is not None or siren is not None:
            _stop("AMBIGUOUS truth must be null")
        for value, label_name in ((kind, "label_kind"), (siret, "SIRET"), (siren, "SIREN")):
            if value is not None:
                _clean_string(value, label_name)
        rows.append((query_id, partition, kind, siret, siren))
        counts[partition]["total"] += 1
        counts[partition][kind] += 1
    table = _table_from_rows(ORACLE_SCHEMA, rows)
    expected = plan["expected"]
    if counts["threshold_dev"] != expected["threshold_dev"] or counts["comparison_dev"] != expected["comparison_dev"]:
        _stop("oracle population counts mismatch")
    if safe_ids[:5] != expected["first_query_ids"] or safe_ids[-5:] != expected["last_query_ids"]:
        _stop("safe query order boundaries mismatch")
    ordered = sorted(
        safe_ids,
        key=lambda value: (
            hashlib.sha256(("v412-unit-engine:" + value).encode()).hexdigest(),
            value.encode(),
        ),
    )
    if safe_ids != ordered:
        _stop("safe query order mismatch")
    id_payload = b"".join(value.encode() + b"\n" for value in safe_ids)
    if (
        len(id_payload) != expected["query_id_payload_bytes"]
        or hashlib.sha256(id_payload).hexdigest() != expected["query_id_payload_sha256"]
    ):
        _stop("query ID payload mismatch")
    truth_payload = bytearray()
    for query_id, partition, kind, siret, siren in rows:
        values = (query_id, partition, kind, siret, siren)
        truth_payload.extend(
            b"\0".join(b"\\N" if value is None else value.encode() for value in values)
            + b"\n"
        )
    truth_sha = hashlib.sha256(truth_payload).hexdigest()
    if len(truth_payload) != expected["logical_payload_bytes"] or truth_sha != expected["logical_payload_sha256"]:
        _stop("truth logical payload mismatch")
    return table, counts, hashlib.sha256(id_payload).hexdigest(), truth_sha


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


def _write_json(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _write_parquet(path: Path, table: pa.Table) -> None:
    pq.write_table(table.replace_schema_metadata(None), path, compression="zstd")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _build_id(
    plan_bytes: bytes,
    lock_bytes: bytes,
    source_hashes: Mapping[str, str],
    input_hashes: Mapping[str, str],
    safe_input_build_id: str,
    id_sha: str,
    truth_sha: str,
    runtime: Mapping[str, str],
) -> str:
    identity = {
        "schema_version": "sireto-v4.12-unit-oracle-build-identity-1",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "safe_input_build_id": safe_input_build_id,
        "query_id_payload_sha256": id_sha,
        "truth_logical_sha256": truth_sha,
        "runtime": runtime,
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _ledger(
    entries: Sequence[tuple[str, Path, str, Mapping[str, Any]]], max_rss: int
) -> pa.Table:
    rows = []
    for role, path, projection, before in entries:
        after = _snapshot(path, max_rss)
        if after != dict(before):
            _stop(f"TOCTOU mutation while creating ledger: {path}")
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
    rows.sort(key=lambda row: (row[0].encode(), row[1].encode()))
    table = _table_from_rows(LEDGER_SCHEMA, rows)
    if table.num_rows != 8 or set(table.column("role").to_pylist()) != LEDGER_ROLES:
        _stop("oracle data ledger is not exactly exhaustive")
    if len(set(table.column("absolute_path").to_pylist())) != 8:
        _stop("oracle ledger paths are not unique")
    return table


def _validate_oracle_package(
    root: Path, expected_build_id: str | None = None
) -> dict[str, Any]:
    _assert_directory(root)
    if {path.name for path in root.iterdir()} != ORACLE_FILES:
        _stop("oracle file-set mismatch")
    for path in root.iterdir():
        _assert_regular(path)
    manifest = load_json_strict(root / "manifest.json")
    integrity = load_json_strict(root / "integrity.json")
    if set(manifest) != ORACLE_MANIFEST_KEYS or set(integrity) != INTEGRITY_KEYS:
        _stop("oracle manifest or integrity keyset mismatch")
    try:
        table = pq.read_table(root / "oracle_dev.parquet")
        record = _parquet_record(root / "oracle_dev.parquet")
    except Exception as exc:
        _stop(f"invalid oracle Parquet: {exc}")
    if table.schema != ORACLE_SCHEMA:
        _stop("oracle Parquet schema mismatch")
    if set(manifest["files"]) != {"oracle_dev.parquet", "integrity.json"}:
        _stop("oracle manifest files keyset mismatch")
    expected_files = {
        "oracle_dev.parquet": record,
        "integrity.json": {
            "sha256": sha256_file(root / "integrity.json"),
            "size_bytes": (root / "integrity.json").stat().st_size,
        },
    }
    if manifest["files"] != expected_files:
        _stop("oracle file mutation or resealed manifest mismatch")
    rows = table.to_pylist()
    id_payload = b"".join(row["query_id"].encode() + b"\n" for row in rows)
    truth_payload = b"".join(
        b"\0".join(
            b"\\N" if row[name] is None else row[name].encode()
            for name in ORACLE_SCHEMA.names
        )
        + b"\n"
        for row in rows
    )
    counts = {
        part: {
            "total": sum(row["dev_partition"] == part for row in rows),
            "MATCH_EXACT": sum(row["dev_partition"] == part and row["label_kind"] == "MATCH_EXACT" for row in rows),
            "AMBIGUOUS": sum(row["dev_partition"] == part and row["label_kind"] == "AMBIGUOUS" for row in rows),
        }
        for part in ("threshold_dev", "comparison_dev")
    }
    identity = root.name if expected_build_id is None else expected_build_id
    common = (
        manifest["build_id"] == integrity["build_id"] == identity
        and manifest["population_counts"] == integrity["population_counts"] == counts
        and integrity["query_count"] == len(rows)
        and manifest["query_id_payload_sha256"]
        == integrity["query_id_payload_sha256"]
        == hashlib.sha256(id_payload).hexdigest()
        and manifest["truth_logical_sha256"]
        == integrity["truth_logical_sha256"]
        == hashlib.sha256(truth_payload).hexdigest()
        and manifest["declarations"] == integrity["declarations"] == DECLARATIONS
    )
    if not common:
        _stop("oracle logical content or identity mismatch")
    if (
        manifest["historical_development_only"] is not True
        or manifest["independent_truth"] is not False
        or manifest["production_certified"] is not False
    ):
        _stop("oracle qualification flags mismatch")
    return manifest


def validate_concordance(
    oracle_dir: Path,
    audit_dir: Path,
    plan_path: Path,
    lock_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    plan_path = plan_path.absolute()
    lock_path = lock_path.absolute()
    if plan_path != (repo / CANONICAL_PLAN_PATH).absolute():
        _stop("certification requires the canonical oracle plan")
    plan = load_json_strict(plan_path)
    validate_plan(plan)
    if lock_path != (repo / plan["execution_lock_path"]).absolute():
        _stop("certification lock path mismatch")
    lock = load_json_strict(lock_path)
    max_rss = plan["max_rss_bytes"]
    _check_rss(max_rss)
    plan_snapshot = _snapshot(plan_path, max_rss)
    lock_snapshot = _snapshot(lock_path, max_rss)
    if load_json_strict(plan_path) != plan or load_json_strict(lock_path) != lock:
        _stop("plan or lock changed while certification began")
    source_snapshots = validate_lock(lock, plan, repo, max_rss)
    paths = {role: Path(path) for role, path in _expected_input_paths(plan).items()}
    _, safe_runtime_snapshots = validate_safe_runtime(
        paths["safe_runtime_manifest"],
        paths["safe_queries_dev"],
        plan["safe_input"],
        max_rss,
    )
    input_snapshots = {role: _snapshot(path, max_rss) for role, path in paths.items()}
    for role, expected_hash in _expected_input_hashes(plan).items():
        if input_snapshots[role]["sha256"] != expected_hash:
            _stop(f"certification input hash mismatch: {role}")
    expected_table, counts, id_sha, truth_sha = build_oracle_table(
        paths["safe_queries_dev"], paths["labels"], paths["split"], plan
    )
    expected_build_id = _build_id(
        plan_path.read_bytes(),
        lock_path.read_bytes(),
        lock["source_hashes"],
        lock["input_hashes"],
        plan["safe_input"]["build_id"],
        id_sha,
        truth_sha,
        plan["runtime"],
    )
    if oracle_dir.name != expected_build_id or audit_dir.name != expected_build_id:
        _stop("certified build_id does not match external identity")
    oracle_manifest = _validate_oracle_package(oracle_dir, expected_build_id)
    actual_table = pq.read_table(oracle_dir / "oracle_dev.parquet")
    if not actual_table.equals(expected_table):
        _stop("published oracle differs from reprojected external truth")
    expected_integrity = {
        "schema_version": "sireto-v4.12-unit-oracle-integrity-1",
        "build_id": expected_build_id,
        "query_count": expected_table.num_rows,
        "population_counts": counts,
        "query_id_payload_sha256": id_sha,
        "truth_logical_sha256": truth_sha,
        "declarations": DECLARATIONS,
    }
    if load_json_strict(oracle_dir / "integrity.json") != expected_integrity:
        _stop("certified integrity.json differs from external truth")
    expected_oracle_manifest = {
        "schema_version": "sireto-v4.12-unit-oracle-manifest-1",
        "build_id": expected_build_id,
        "safe_input_build_id": plan["safe_input"]["build_id"],
        "files": {
            "oracle_dev.parquet": _parquet_record(oracle_dir / "oracle_dev.parquet"),
            "integrity.json": {
                "sha256": sha256_file(oracle_dir / "integrity.json"),
                "size_bytes": (oracle_dir / "integrity.json").stat().st_size,
            },
        },
        "population_counts": counts,
        "query_id_payload_sha256": id_sha,
        "truth_logical_sha256": truth_sha,
        "runtime": plan["runtime"],
        "declarations": DECLARATIONS,
        "historical_development_only": True,
        "independent_truth": False,
        "production_certified": False,
    }
    if oracle_manifest != expected_oracle_manifest:
        _stop("certified oracle manifest differs from external identity")
    _assert_directory(audit_dir)
    if oracle_dir.name != audit_dir.name or {p.name for p in audit_dir.iterdir()} != AUDIT_FILES:
        _stop("oracle/audit identity or file-set mismatch")
    for path in audit_dir.iterdir():
        _assert_regular(path)
    provenance = load_json_strict(audit_dir / "provenance.json")
    audit_manifest = load_json_strict(audit_dir / "manifest.json")
    if set(provenance) != PROVENANCE_KEYS or set(audit_manifest) != {"schema_version", "build_id", "files"}:
        _stop("audit JSON keyset mismatch")
    expected_files = {
        name: {
            "sha256": sha256_file(audit_dir / name),
            "size_bytes": (audit_dir / name).stat().st_size,
        }
        for name in ("data_inputs.parquet", "provenance.json")
    }
    if audit_manifest != {
        "schema_version": "sireto-v4.12-unit-oracle-audit-manifest-1",
        "build_id": oracle_dir.name,
        "files": expected_files,
    }:
        _stop("audit manifest mismatch")
    expected_provenance = {
        "schema_version": "sireto-v4.12-unit-oracle-provenance-1",
        "build_id": expected_build_id,
        "git_commit": lock["git_commit"],
        "sources": lock["source_hashes"],
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "runtime": plan["runtime"],
        "data_input_count": 8,
        "declarations": DECLARATIONS,
        "oracle_manifest_sha256": sha256_file(oracle_dir / "manifest.json"),
    }
    if provenance != expected_provenance:
        _stop("oracle/audit provenance mismatch")
    try:
        ledger = pq.read_table(audit_dir / "data_inputs.parquet")
    except Exception as exc:
        _stop(f"invalid oracle ledger: {exc}")
    if ledger.schema != LEDGER_SCHEMA or ledger.num_rows != 8:
        _stop("oracle ledger schema/cardinality mismatch")
    if set(ledger.column("role").to_pylist()) != LEDGER_ROLES:
        _stop("oracle ledger roles mismatch")
    expected_ledger_rows = [
        (
            role,
            str(paths[role].absolute()),
            projection,
            input_snapshots[role]["size"],
            input_snapshots[role]["sha256"],
            input_snapshots[role]["size"],
            input_snapshots[role]["sha256"],
        )
        for role, projection in (
            ("labels", ",".join(plan["labels"]["projection"])),
            ("split", ",".join(plan["split_assignments"]["projection"])),
        )
    ]
    safe_root = paths["safe_runtime_manifest"].parent
    expected_ledger_rows.extend(
        (
            role,
            str((safe_root / filename).absolute()),
            "query_id" if filename == "queries_dev.parquet" else "",
            safe_runtime_snapshots[safe_root / filename]["size"],
            safe_runtime_snapshots[safe_root / filename]["sha256"],
            safe_runtime_snapshots[safe_root / filename]["size"],
            safe_runtime_snapshots[safe_root / filename]["sha256"],
        )
        for filename, role in SAFE_LEDGER_ROLES.items()
    )
    expected_ledger_rows.sort(key=lambda row: (row[0].encode(), row[1].encode()))
    actual_ledger_rows = list(
        zip(*(ledger.column(name).to_pylist() for name in ledger.column_names), strict=True)
    )
    if actual_ledger_rows != expected_ledger_rows:
        _stop("oracle ledger differs from exact external inputs")
    if oracle_manifest["declarations"] != provenance["declarations"]:
        _stop("oracle/audit declarations mismatch")
    for path, snapshot in source_snapshots.items():
        _require_unchanged(path, snapshot, max_rss)
    _require_unchanged(plan_path, plan_snapshot, max_rss)
    _require_unchanged(lock_path, lock_snapshot, max_rss)
    for role, path in paths.items():
        _require_unchanged(path, input_snapshots[role], max_rss)


def _chmod_immutable(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        os.chmod(path, 0o555)
    os.chmod(root, 0o555)


def _promote(source: Path, destination: Path) -> None:
    if destination.exists():
        _stop(f"immutable destination exists: {destination}")
    if os.lstat(source).st_dev != os.lstat(destination.parent).st_dev:
        _stop("staging and output are on different filesystems")
    _fsync_dir(source)
    os.rename(source, destination)
    _fsync_dir(destination.parent)
    _chmod_immutable(destination)
    _fsync_dir(destination.parent)


def _process_context(plan: Mapping[str, Any]) -> Path:
    if not sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        _stop("python -B and PYTHONDONTWRITEBYTECODE=1 are mandatory")
    staging_value = os.environ.get("SIRETO_V412_ORACLE_STAGING")
    tmp_value = os.environ.get("TMPDIR")
    if not staging_value or not tmp_value:
        _stop("private oracle staging environment absent")
    staging = Path(staging_value)
    tmp = Path(tmp_value)
    _assert_directory(staging)
    _assert_directory(tmp)
    try:
        staging.relative_to(Path(plan["temp_root"]))
        tmp.relative_to(staging)
    except ValueError:
        _stop("oracle staging/TMPDIR boundary mismatch")
    if tmp.resolve() != (staging / "tmp").resolve():
        _stop("TMPDIR is not the private oracle staging tmp")
    return staging


def build_oracle(plan_path: Path, lock_path: Path) -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    plan_path = plan_path.absolute()
    lock_path = lock_path.absolute()
    if plan_path != (repo / CANONICAL_PLAN_PATH).absolute():
        _stop("only the canonical committed oracle plan may execute")
    plan = load_json_strict(plan_path)
    validate_plan(plan)
    if lock_path != (repo / plan["execution_lock_path"]).absolute():
        _stop("oracle execution lock path mismatch")
    lock = load_json_strict(lock_path)
    max_rss = plan["max_rss_bytes"]
    _check_rss(max_rss)
    staging = _process_context(plan)
    source_snapshots = validate_lock(lock, plan, repo, max_rss)
    plan_snapshot = _snapshot(plan_path, max_rss)
    lock_snapshot = _snapshot(lock_path, max_rss)
    if load_json_strict(plan_path) != plan or load_json_strict(lock_path) != lock:
        _stop("oracle plan or execution lock changed while being read")
    paths = {role: Path(path) for role, path in _expected_input_paths(plan).items()}
    safe_manifest, sibling_snapshots = validate_safe_runtime(
        paths["safe_runtime_manifest"],
        paths["safe_queries_dev"],
        plan["safe_input"],
        max_rss,
    )
    input_snapshots = {role: _snapshot(path, max_rss) for role, path in paths.items()}
    for role, expected_hash in _expected_input_hashes(plan).items():
        if input_snapshots[role]["sha256"] != expected_hash:
            _stop(f"{role} input hash mismatch")
    if (
        input_snapshots["labels"]["size"] != plan["labels"]["size_bytes"]
        or input_snapshots["split"]["size"] != plan["split_assignments"]["size_bytes"]
    ):
        _stop("labels/split size mismatch")
    oracle, counts, id_sha, truth_sha = build_oracle_table(
        paths["safe_queries_dev"], paths["labels"], paths["split"], plan
    )
    build_id = _build_id(
        plan_path.read_bytes(),
        lock_path.read_bytes(),
        lock["source_hashes"],
        lock["input_hashes"],
        plan["safe_input"]["build_id"],
        id_sha,
        truth_sha,
        plan["runtime"],
    )
    output_root = Path(plan["output_root"])
    audit_root = Path(plan["audit_output_root"])
    _ensure_directory(output_root)
    _ensure_directory(audit_root)
    final_oracle = output_root / build_id
    final_audit = audit_root / build_id
    if final_oracle.exists() or final_audit.exists():
        _stop("oracle or audit destination already exists")
    staged_oracle = staging / "oracle"
    staged_audit = staging / "audit"
    staged_oracle.mkdir()
    staged_audit.mkdir()
    _write_parquet(staged_oracle / "oracle_dev.parquet", oracle)
    reopened = pq.read_table(staged_oracle / "oracle_dev.parquet")
    if reopened.schema != ORACLE_SCHEMA or not reopened.equals(oracle):
        _stop("oracle Parquet changed after reopening")
    integrity = {
        "schema_version": "sireto-v4.12-unit-oracle-integrity-1",
        "build_id": build_id,
        "query_count": oracle.num_rows,
        "population_counts": counts,
        "query_id_payload_sha256": id_sha,
        "truth_logical_sha256": truth_sha,
        "declarations": DECLARATIONS,
    }
    if set(integrity) != INTEGRITY_KEYS:
        _stop("integrity keyset construction failure")
    _write_json(staged_oracle / "integrity.json", integrity)
    oracle_manifest = {
        "schema_version": "sireto-v4.12-unit-oracle-manifest-1",
        "build_id": build_id,
        "safe_input_build_id": plan["safe_input"]["build_id"],
        "files": {
            "oracle_dev.parquet": _parquet_record(staged_oracle / "oracle_dev.parquet"),
            "integrity.json": {
                "sha256": sha256_file(staged_oracle / "integrity.json"),
                "size_bytes": (staged_oracle / "integrity.json").stat().st_size,
            },
        },
        "population_counts": counts,
        "query_id_payload_sha256": id_sha,
        "truth_logical_sha256": truth_sha,
        "runtime": plan["runtime"],
        "declarations": DECLARATIONS,
        "historical_development_only": True,
        "independent_truth": False,
        "production_certified": False,
    }
    if set(oracle_manifest) != ORACLE_MANIFEST_KEYS:
        _stop("oracle manifest keyset construction failure")
    _write_json(staged_oracle / "manifest.json", oracle_manifest)
    if _validate_oracle_package(staged_oracle, build_id) != oracle_manifest:
        _stop("staged oracle revalidation mismatch")
    safe_root = paths["safe_runtime_manifest"].parent
    ledger_entries = [
        (
            role,
            safe_root / filename,
            "query_id" if filename == "queries_dev.parquet" else "",
            sibling_snapshots[safe_root / filename],
        )
        for filename, role in SAFE_LEDGER_ROLES.items()
    ]
    ledger_entries.extend(
        [
            (
                "labels",
                paths["labels"],
                ",".join(plan["labels"]["projection"]),
                input_snapshots["labels"],
            ),
            (
                "split",
                paths["split"],
                ",".join(plan["split_assignments"]["projection"]),
                input_snapshots["split"],
            ),
        ]
    )
    ledger = _ledger(ledger_entries, max_rss)
    _write_parquet(staged_audit / "data_inputs.parquet", ledger)
    if pq.read_table(staged_audit / "data_inputs.parquet").schema != LEDGER_SCHEMA:
        _stop("reopened oracle ledger schema mismatch")
    provenance = {
        "schema_version": "sireto-v4.12-unit-oracle-provenance-1",
        "build_id": build_id,
        "git_commit": lock["git_commit"],
        "sources": lock["source_hashes"],
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "runtime": plan["runtime"],
        "data_input_count": 8,
        "declarations": DECLARATIONS,
        "oracle_manifest_sha256": sha256_file(staged_oracle / "manifest.json"),
    }
    if set(provenance) != PROVENANCE_KEYS:
        _stop("provenance keyset construction failure")
    _write_json(staged_audit / "provenance.json", provenance)
    audit_manifest = {
        "schema_version": "sireto-v4.12-unit-oracle-audit-manifest-1",
        "build_id": build_id,
        "files": {
            name: {
                "sha256": sha256_file(staged_audit / name),
                "size_bytes": (staged_audit / name).stat().st_size,
            }
            for name in ("data_inputs.parquet", "provenance.json")
        },
    }
    _write_json(staged_audit / "manifest.json", audit_manifest)

    def revalidate_before_promotion() -> None:
        current_manifest, current_siblings = validate_safe_runtime(
            paths["safe_runtime_manifest"],
            paths["safe_queries_dev"],
            plan["safe_input"],
            max_rss,
        )
        if current_manifest != safe_manifest or current_siblings != sibling_snapshots:
            _stop("safe runtime sibling changed")
        projected, current_counts, current_id, current_truth = build_oracle_table(
            paths["safe_queries_dev"], paths["labels"], paths["split"], plan
        )
        if (
            not projected.equals(oracle)
            or current_counts != counts
            or current_id != id_sha
            or current_truth != truth_sha
        ):
            _stop("oracle source projection changed before promotion")
        for path, before in source_snapshots.items():
            _require_unchanged(path, before, max_rss)
        _require_unchanged(plan_path, plan_snapshot, max_rss)
        _require_unchanged(lock_path, lock_snapshot, max_rss)
        for role, path in paths.items():
            _require_unchanged(path, input_snapshots[role], max_rss)
        _verify_git_sources(
            repo, plan["sources"], lock["source_hashes"], lock["git_commit"], max_rss
        )
        _check_rss(max_rss)

    revalidate_before_promotion()
    _promote(staged_audit, final_audit)
    revalidate_before_promotion()
    _promote(staged_oracle, final_oracle)
    validate_concordance(
        final_oracle, final_audit, plan_path, lock_path
    )
    return final_oracle, final_audit


def _bootstrap(plan_path: Path, lock_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    if plan_path != (repo / CANONICAL_PLAN_PATH).absolute():
        _stop("only the canonical committed oracle plan may execute")
    plan = load_json_strict(plan_path)
    validate_plan(plan)
    lock = load_json_strict(lock_path)
    if lock_path != (repo / plan["execution_lock_path"]).absolute():
        _stop("oracle execution lock path mismatch")
    validate_lock(lock, plan, repo, plan["max_rss_bytes"])
    temp_root = Path(plan["temp_root"])
    _ensure_directory(temp_root)
    staging = temp_root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    tmp = staging / "tmp"
    tmp.mkdir(mode=0o700)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SIRETO_V412_ORACLE_STAGING"] = str(staging)
    environment["TMPDIR"] = str(tmp)
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
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--internal-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.internal_run:
            oracle, audit = build_oracle(args.plan, args.lock)
            print(
                json.dumps(
                    {
                        "verdict": "GO_V412_UNIT_ORACLE",
                        "build_id": oracle.name,
                        "oracle_dir": str(oracle),
                        "audit_dir": str(audit),
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
