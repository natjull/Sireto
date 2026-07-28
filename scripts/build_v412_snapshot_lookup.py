#!/usr/bin/env python3
"""Build and validate the immutable V4.12 SIRENE snapshot lookup.

Real publication requires an independently issued execution lock.  The
unlocked helpers are intentionally limited to unit-test fixtures.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v412_snapshot_lookup import (  # noqa: E402
    DETAIL_COLUMNS,
    INDEX_NAME,
    LOOKUP_COLUMNS,
    TABLE_NAME,
    V412SnapshotLookup,
    inspect_lookup_schema,
)


SCHEMA_VERSION = "sireto-v4.12-snapshot-lookup-1"
LOCK_SCHEMA_VERSION = "sireto-v4.12-snapshot-lookup-execution-lock-1"
PLAN_SCHEMA_VERSION = "sireto-v4.12-snapshot-lookup-plan-1"
DENYLIST_SCHEMA_VERSION = "sireto-v4.12-forbidden-artifacts-1"
PURPOSE = "BUILD_V412_SNAPSHOT_LOOKUP"
AUDIT_VERDICT = "GO_BUILD_V412_SNAPSHOT_LOOKUP"
GO_VERDICT = "GO_V412_SNAPSHOT_LOOKUP"
BUILD_STOP = "STOP_V412_LOOKUP_BUILD"
PARITY_STOP = "STOP_V412_LOOKUP_PARITY"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = REPO_ROOT / "config/v4_12_snapshot_lookup_plan.json"
DEFAULT_DENYLIST = REPO_ROOT / "config/v4_12_forbidden_artifacts.json"
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/v4_12_snapshot_lookup"
)
OUTPUT_FILES = {
    "candidate_details.duckdb",
    "integrity.json",
    "timing.json",
}
SOURCE_PATHS = [
    "src/xgb_matcher/__init__.py",
    "src/xgb_matcher/v412_snapshot_lookup.py",
    "scripts/__init__.py",
    "scripts/build_v412_snapshot_lookup.py",
    "docs/v4_12_snapshot_lookup_contract.md",
    "config/v4_12_snapshot_lookup_plan.json",
    "config/v4_12_forbidden_artifacts.json",
]
_HEX64 = set("0123456789abcdef")
EXPECTED_BUILD_KEYS = {
    "database_filename",
    "disk_free_min_bytes",
    "duckdb",
    "max_rss_bytes",
    "output_root",
    "publication",
    "table",
}
EXPECTED_DUCKDB_POLICY = {
    "checkpoint_before_close": True,
    "memory_limit": "6GB",
    "reject_wal": True,
    "temp_directory": "<staging>/duckdb_tmp",
    "threads": 4,
    "unique_index_name": INDEX_NAME,
    "unique_index_sql": (
        "CREATE UNIQUE INDEX candidate_details_siret_uidx "
        "ON candidate_details(siret)"
    ),
    "version": "1.4.3",
}
EXPECTED_COLUMNS = [
    {
        "lookup": "siret",
        "snapshot": "siret",
        "sql": "CAST(siret AS VARCHAR)",
    },
    {
        "lookup": "candidate_state",
        "snapshot": "etatAdministratifEtablissement",
        "sql": (
            "upper(trim(CAST(etatAdministratifEtablissement AS VARCHAR)))"
        ),
    },
    {
        "lookup": "enseigne1",
        "snapshot": "enseigne1Etablissement",
        "sql": "CAST(enseigne1Etablissement AS VARCHAR)",
    },
    {
        "lookup": "enseigne2",
        "snapshot": "enseigne2Etablissement",
        "sql": "CAST(enseigne2Etablissement AS VARCHAR)",
    },
    {
        "lookup": "enseigne3",
        "snapshot": "enseigne3Etablissement",
        "sql": "CAST(enseigne3Etablissement AS VARCHAR)",
    },
    {
        "lookup": "denomination_usuelle",
        "snapshot": "denominationUsuelleEtablissement",
        "sql": "CAST(denominationUsuelleEtablissement AS VARCHAR)",
    },
    {
        "lookup": "activity_code",
        "snapshot": "activitePrincipaleEtablissement",
        "sql": "CAST(activitePrincipaleEtablissement AS VARCHAR)",
    },
]
EXPECTED_SNAPSHOT = {
    "invalid_siret_count": 0,
    "path": str(
        Path(
            "/Users/nathanjullia/Documents/Projets/SIRETO/"
            "data/StockEtablissement_utf8.parquet"
        )
    ),
    "row_count": 42_322_035,
    "sha256": (
        "c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845"
    ),
    "unique_siret_count": 42_322_035,
}
EXPECTED_PARITY = {
    "candidate_path": (
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
        "v4_11_input_blind/ec4326ec57e4411d/"
        "candidates_sparse_top100.parquet"
    ),
    "candidate_projection": ["candidate_siret"],
    "candidate_row_count": 698_892,
    "candidate_sha256": (
        "78b2f78ddeac863ac39ca64301d42312c7fb766ac51e2b5d19dde5c5910aedac"
    ),
    "candidate_unique_siret_count": 508_081,
    "reference_snapshot_scan_count": 1,
    "snapshot_sample_count": 10_000,
    "snapshot_sample_namespace": "v412-lookup-parity:",
    "snapshot_sample_ordered_newline_sha256": (
        "58c9700d2a1ed2bb433e4f7a25a845ba236d63cfe633dcd64f9156469777f945"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value).issubset(_HEX64)
    )


def _runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "duckdb": importlib.metadata.version("duckdb"),
    }


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {
        relative: file_sha256(repo_root / relative)
        for relative in SOURCE_PATHS
    }


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = Path(path).stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path),
    }


def _assert_unchanged(path: Path, expected: Mapping[str, Any]) -> None:
    if _fingerprint(path) != dict(expected):
        raise ValueError(f"{BUILD_STOP}: input changed during execution")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(plan) != {
        "build",
        "columns",
        "future_latency_gate",
        "inference_lookup",
        "parity",
        "schema_version",
        "snapshot",
        "verdicts",
    } or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"{BUILD_STOP}: plan schema changed")
    columns = plan.get("columns")
    if (
        not isinstance(columns, list)
        or len(columns) != len(LOOKUP_COLUMNS)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"lookup", "snapshot", "sql"}
            for item in columns
        )
        or columns != EXPECTED_COLUMNS
    ):
        raise ValueError(f"{BUILD_STOP}: lookup columns changed")
    build = plan.get("build")
    if not isinstance(build, Mapping) or set(build) != EXPECTED_BUILD_KEYS:
        raise ValueError(f"{BUILD_STOP}: build policy changed")
    expected_build = {
        "database_filename": "candidate_details.duckdb",
        "disk_free_min_bytes": 50 * 1024**3,
        "duckdb": EXPECTED_DUCKDB_POLICY,
        "max_rss_bytes": 8 * 1024**3,
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "publication": "FSYNC_AND_ATOMIC_RENAME",
        "table": TABLE_NAME,
    }
    if dict(build) != expected_build:
        raise ValueError(f"{BUILD_STOP}: frozen build parameters changed")
    if plan.get("inference_lookup") != {
        "max_sirets_per_call": 100,
        "read_only": True,
        "returns_only_requested_sirets": True,
    }:
        raise ValueError(f"{BUILD_STOP}: inference lookup policy changed")
    parity = plan.get("parity")
    if parity != EXPECTED_PARITY:
        raise ValueError(f"{BUILD_STOP}: parity projection changed")
    if plan.get("snapshot") != EXPECTED_SNAPSHOT:
        raise ValueError(f"{BUILD_STOP}: frozen snapshot changed")
    if plan.get("verdicts") != [
        GO_VERDICT,
        BUILD_STOP,
        PARITY_STOP,
    ]:
        raise ValueError(f"{BUILD_STOP}: verdict set changed")
    if importlib.metadata.version("duckdb") != EXPECTED_DUCKDB_POLICY["version"]:
        raise ValueError(f"{BUILD_STOP}: DuckDB version changed")
    return plan


def _load_denylist(path: Path) -> tuple[dict[str, Any], set[str], list[Path]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        set(value) != {"schema_version", "artifacts"}
        or value.get("schema_version") != DENYLIST_SCHEMA_VERSION
        or not isinstance(value.get("artifacts"), list)
    ):
        raise ValueError(f"{BUILD_STOP}: denylist changed")
    hashes: set[str] = set()
    roots: list[Path] = []
    for artifact in value["artifacts"]:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "build_id",
            "files",
            "manifest_sha256",
            "path",
            "reason",
        }:
            raise ValueError(f"{BUILD_STOP}: denylist artifact changed")
        roots.append(Path(str(artifact["path"])).resolve())
        hashes.add(str(artifact["manifest_sha256"]))
        hashes.update(str(item) for item in artifact["files"].values())
    if (
        len(value["artifacts"]) != 3
        or len(set(roots)) != 3
        or not hashes
        or not all(_valid_sha256(item) for item in hashes)
    ):
        raise ValueError(f"{BUILD_STOP}: invalid denylist hash")
    return value, hashes, roots


def _assert_inputs_allowed(
    paths: Iterable[Path],
    *,
    forbidden_hashes: set[str],
    forbidden_roots: Iterable[Path],
) -> None:
    roots = list(forbidden_roots)
    for candidate in paths:
        resolved = Path(candidate).resolve()
        if any(resolved == root or root in resolved.parents for root in roots):
            raise ValueError(f"{BUILD_STOP}: forbidden challenge path")
        if resolved.is_file() and file_sha256(resolved) in forbidden_hashes:
            raise ValueError(f"{BUILD_STOP}: forbidden challenge hash")


def validate_execution_lock(
    lock_path: Path,
    *,
    plan_path: Path,
    denylist_path: Path,
    verify_git: bool = True,
) -> tuple[dict[str, Any], str]:
    """Validate the external authorization before any data file is opened."""

    repo_root = REPO_ROOT
    raw = Path(lock_path).read_bytes()
    lock = json.loads(raw)
    expected_fields = {
        "schema_version",
        "purpose",
        "audit_verdict",
        "git_commit",
        "source_hashes",
        "input_paths",
        "input_hashes",
        "runtime",
        "output_root",
    }
    if set(lock) != expected_fields:
        raise ValueError(f"{BUILD_STOP}: execution lock fields changed")
    if (
        lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("purpose") != PURPOSE
        or lock.get("audit_verdict") != AUDIT_VERDICT
    ):
        raise ValueError(f"{BUILD_STOP}: build is not independently authorized")
    sources = _source_hashes(repo_root)
    if lock.get("source_hashes") != sources:
        raise ValueError(f"{BUILD_STOP}: source hashes changed")
    plan = _load_plan(plan_path)
    expected_paths = {
        "plan": str(Path(plan_path).resolve()),
        "denylist": str(Path(denylist_path).resolve()),
        "snapshot": str(Path(plan["snapshot"]["path"]).resolve()),
        "candidates": str(Path(plan["parity"]["candidate_path"]).resolve()),
    }
    expected_hashes = {
        "plan": file_sha256(plan_path),
        "denylist": file_sha256(denylist_path),
        "snapshot": plan["snapshot"]["sha256"],
        "candidates": plan["parity"]["candidate_sha256"],
    }
    if (
        lock.get("input_paths") != expected_paths
        or lock.get("input_hashes") != expected_hashes
        or lock.get("runtime") != _runtime()
        or Path(str(lock.get("output_root"))).resolve()
        != Path(plan["build"]["output_root"]).resolve()
    ):
        raise ValueError(f"{BUILD_STOP}: locked environment changed")
    commit = str(lock.get("git_commit") or "")
    if not commit:
        raise ValueError(f"{BUILD_STOP}: locked commit missing")
    if verify_git:
        for relative, expected_hash in sources.items():
            result = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            )
            if hashlib.sha256(result.stdout).hexdigest() != expected_hash:
                raise ValueError(f"{BUILD_STOP}: commit does not pin {relative}")
    return lock, hashlib.sha256(raw).hexdigest()


def _sql_path(path: Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


def _projection_sql(plan: Mapping[str, Any], alias: str = "snapshot") -> str:
    expressions = []
    for column in plan["columns"]:
        expression = str(column["sql"])
        source = str(column["snapshot"])
        expression = expression.replace(source, f"{alias}.{source}")
        expressions.append(f"{expression} AS {column['lookup']}")
    return ",\n".join(expressions)


def _configure_duckdb(
    connection: duckdb.DuckDBPyConnection,
    *,
    plan: Mapping[str, Any],
    temp_directory: Path,
) -> None:
    duck = plan["build"]["duckdb"]
    temp_directory.mkdir(exist_ok=True)
    connection.execute(f"SET memory_limit = '{duck['memory_limit']}'")
    connection.execute(f"SET threads = {int(duck['threads'])}")
    connection.execute(f"SET temp_directory = '{_sql_path(temp_directory)}'")


def _build_database(
    *,
    database_path: Path,
    snapshot_path: Path,
    staging: Path,
    plan: Mapping[str, Any],
) -> dict[str, int]:
    """Build an unpublished lookup. Used by the locked publisher and fixtures."""

    temp_directory = staging / "duckdb_tmp"
    connection = duckdb.connect(str(database_path))
    try:
        _configure_duckdb(
            connection,
            plan=plan,
            temp_directory=temp_directory,
        )
        connection.execute(
            f"""
            CREATE TABLE {TABLE_NAME} AS
            SELECT {_projection_sql(plan)}
            FROM read_parquet('{_sql_path(snapshot_path)}') AS snapshot
            """
        )
        row_count, unique_count, invalid_count = connection.execute(
            f"""
            SELECT
                count(*),
                count(DISTINCT siret),
                count(*) FILTER (
                    WHERE siret IS NULL
                       OR NOT regexp_full_match(siret, '[0-9]{{14}}')
                )
            FROM {TABLE_NAME}
            """
        ).fetchone()
        expected = plan["snapshot"]
        if (
            int(row_count) != int(expected["row_count"])
            or int(unique_count) != int(expected["unique_siret_count"])
            or int(invalid_count) != int(expected["invalid_siret_count"])
            or int(row_count) != int(unique_count)
            or int(invalid_count) != 0
        ):
            raise ValueError(f"{BUILD_STOP}: snapshot cardinality or SIRET drift")
        connection.execute(
            plan["build"]["duckdb"]["unique_index_sql"]
        )
        inspect_lookup_schema(connection)
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    if database_path.with_suffix(database_path.suffix + ".wal").exists():
        raise ValueError(f"{BUILD_STOP}: DuckDB WAL remains after close")
    with V412SnapshotLookup(database_path):
        pass
    return {
        "row_count": int(row_count),
        "unique_siret_count": int(unique_count),
        "invalid_siret_count": int(invalid_count),
    }


def _build_reference_once(
    *,
    snapshot_path: Path,
    candidates_path: Path,
    reference_path: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce both parity populations with one snapshot scan."""

    parity = plan["parity"]
    namespace = str(parity["snapshot_sample_namespace"]).replace("'", "''")
    sample_count = int(parity["snapshot_sample_count"])
    connection = duckdb.connect()
    try:
        _configure_duckdb(
            connection,
            plan=plan,
            temp_directory=reference_path.parent / "duckdb_tmp",
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE requested_sirets AS
            SELECT DISTINCT CAST(candidate_siret AS VARCHAR) AS siret
            FROM read_parquet('{_sql_path(candidates_path)}')
            """
        )
        candidate_rows, unique_candidates = connection.execute(
            f"""
            SELECT count(*), count(DISTINCT CAST(candidate_siret AS VARCHAR))
            FROM read_parquet('{_sql_path(candidates_path)}')
            """
        ).fetchone()
        if (
            int(candidate_rows) != int(parity["candidate_row_count"])
            or int(unique_candidates)
            != int(parity["candidate_unique_siret_count"])
        ):
            raise ValueError(f"{PARITY_STOP}: candidate projection drift")
        reference_sql = f"""
            COPY (
                WITH projected AS MATERIALIZED (
                    SELECT DISTINCT {_projection_sql(plan)}
                    FROM read_parquet('{_sql_path(snapshot_path)}') AS snapshot
                ),
                ranked AS (
                    SELECT
                        projected.*,
                        row_number() OVER (
                            ORDER BY sha256('{namespace}' || siret), siret
                        ) AS snapshot_sample_rank
                    FROM projected
                )
                SELECT
                    ranked.*,
                    requested.siret IS NOT NULL AS in_candidate_pool,
                    ranked.snapshot_sample_rank <= {sample_count}
                        AS in_snapshot_sample
                FROM ranked
                LEFT JOIN requested_sirets AS requested USING (siret)
                WHERE requested.siret IS NOT NULL
                   OR ranked.snapshot_sample_rank <= {sample_count}
                ORDER BY siret
            ) TO '{_sql_path(reference_path)}'
              (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        snapshot_scan_count = reference_sql.count(
            f"read_parquet('{_sql_path(snapshot_path)}')"
        )
        if snapshot_scan_count != 1:
            raise ValueError(
                f"{PARITY_STOP}: reference does not contain one snapshot scan"
            )
        connection.execute(reference_sql)
    finally:
        connection.close()
    return {
        "candidate_row_count": int(candidate_rows),
        "candidate_unique_siret_count": int(unique_candidates),
        "reference_snapshot_scan_count": snapshot_scan_count,
    }


def _parity_check(
    *,
    database_path: Path,
    reference_path: Path,
    plan: Mapping[str, Any],
    reference_snapshot_scan_count: int,
) -> dict[str, Any]:
    parity = plan["parity"]
    if reference_snapshot_scan_count != 1:
        raise ValueError(f"{PARITY_STOP}: reference snapshot scan count changed")
    connection = duckdb.connect()
    try:
        reference_rows = connection.execute(
            f"""
            SELECT {", ".join(LOOKUP_COLUMNS)},
                   in_candidate_pool, in_snapshot_sample, snapshot_sample_rank
            FROM read_parquet('{_sql_path(reference_path)}')
            ORDER BY siret
            """
        ).fetchall()
    finally:
        connection.close()
    sample_rows = sorted(
        (row for row in reference_rows if row[-2] is True),
        key=lambda row: int(row[-1]),
    )
    sample_hash = hashlib.sha256(
        "".join(f"{row[0]}\n" for row in sample_rows).encode()
    ).hexdigest()
    if (
        len(sample_rows) != int(parity["snapshot_sample_count"])
        or sample_hash != parity["snapshot_sample_ordered_newline_sha256"]
    ):
        raise ValueError(f"{PARITY_STOP}: deterministic sample changed")
    mismatch_count = 0
    returned_count = 0
    with V412SnapshotLookup(database_path) as store:
        for offset in range(0, len(reference_rows), 100):
            batch = reference_rows[offset : offset + 100]
            expected = {row[0]: row[1:7] for row in batch}
            observed = store.get_candidate_scene_details(list(expected))
            returned_count += len(observed)
            if set(observed) != set(expected):
                mismatch_count += len(set(observed) ^ set(expected))
                continue
            for siret, values in expected.items():
                observed_values = tuple(
                    observed[siret][column] for column in DETAIL_COLUMNS
                )
                if observed_values != tuple(values):
                    mismatch_count += 1
    if mismatch_count or returned_count != len(reference_rows):
        raise ValueError(f"{PARITY_STOP}: lookup differs from bulk reference")
    pool_count = sum(row[-3] is True for row in reference_rows)
    if pool_count != int(parity["candidate_unique_siret_count"]):
        raise ValueError(f"{PARITY_STOP}: candidate SIRET missing from snapshot")
    return {
        "reference_row_count": len(reference_rows),
        "candidate_unique_siret_count": pool_count,
        "snapshot_sample_count": len(sample_rows),
        "snapshot_sample_ordered_newline_sha256": sample_hash,
        "mismatch_count": 0,
        "lookup_batch_max": 100,
        "reference_snapshot_scan_count": reference_snapshot_scan_count,
    }


def _external_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    required = DEFAULT_OUTPUT_ROOT.parent.parent.parent
    if resolved != DEFAULT_OUTPUT_ROOT or required not in resolved.parents:
        raise ValueError(f"{BUILD_STOP}: output root is not the frozen SSD path")
    return resolved


def _identity(
    *,
    plan_sha256: str,
    lock_sha256: str,
    source_hashes: Mapping[str, str],
    runtime: Mapping[str, str],
    snapshot_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "execution_lock_sha256": lock_sha256,
        "source_hashes": dict(source_hashes),
        "runtime": dict(runtime),
        "snapshot_sha256": snapshot_sha256,
        "candidate_sha256": candidate_sha256,
    }


def build_artifact(
    *,
    execution_lock_path: Path,
    plan_path: Path = DEFAULT_PLAN,
    denylist_path: Path = DEFAULT_DENYLIST,
) -> Path:
    """Execute the real locked build and publish atomically on the external SSD."""

    lock, lock_sha = validate_execution_lock(
        execution_lock_path,
        plan_path=plan_path,
        denylist_path=denylist_path,
    )
    plan = _load_plan(plan_path)
    snapshot = Path(plan["snapshot"]["path"]).resolve()
    candidates = Path(plan["parity"]["candidate_path"]).resolve()
    output_root = _external_output(Path(lock["output_root"]))
    _, forbidden_hashes, forbidden_roots = _load_denylist(denylist_path)
    _assert_inputs_allowed(
        [
            plan_path,
            denylist_path,
            snapshot,
            candidates,
            execution_lock_path,
            *[
                REPO_ROOT / relative
                for relative in SOURCE_PATHS
            ],
        ],
        forbidden_hashes=forbidden_hashes,
        forbidden_roots=forbidden_roots,
    )
    fingerprints = {
        "plan": _fingerprint(plan_path),
        "denylist": _fingerprint(denylist_path),
        "snapshot": _fingerprint(snapshot),
        "candidates": _fingerprint(candidates),
        "lock": _fingerprint(execution_lock_path),
        **{
            f"source:{relative}": _fingerprint(
                REPO_ROOT / relative
            )
            for relative in SOURCE_PATHS
        },
    }
    if (
        fingerprints["snapshot"]["sha256"] != plan["snapshot"]["sha256"]
        or fingerprints["candidates"]["sha256"]
        != plan["parity"]["candidate_sha256"]
    ):
        raise ValueError(f"{BUILD_STOP}: frozen input hash changed")
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < int(
        plan["build"]["disk_free_min_bytes"]
    ):
        raise ValueError(f"{BUILD_STOP}: less than 50 GiB free")
    identity = _identity(
        plan_sha256=fingerprints["plan"]["sha256"],
        lock_sha256=lock_sha,
        source_hashes=lock["source_hashes"],
        runtime=lock["runtime"],
        snapshot_sha256=fingerprints["snapshot"]["sha256"],
        candidate_sha256=fingerprints["candidates"]["sha256"],
    )
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = output_root / build_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Immutable V4.12 lookup already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    started = time.perf_counter()
    try:
        database_path = staging / plan["build"]["database_filename"]
        integrity = _build_database(
            database_path=database_path,
            snapshot_path=snapshot,
            staging=staging,
            plan=plan,
        )
        _assert_unchanged(snapshot, fingerprints["snapshot"])
        reference_path = staging / ".parity_reference.parquet"
        reference_stats = _build_reference_once(
            snapshot_path=snapshot,
            candidates_path=candidates,
            reference_path=reference_path,
            plan=plan,
        )
        _assert_unchanged(snapshot, fingerprints["snapshot"])
        _assert_unchanged(candidates, fingerprints["candidates"])
        parity = _parity_check(
            database_path=database_path,
            reference_path=reference_path,
            plan=plan,
            reference_snapshot_scan_count=reference_stats[
                "reference_snapshot_scan_count"
            ],
        )
        reference_path.unlink()
        temp_directory = staging / "duckdb_tmp"
        shutil.rmtree(temp_directory, ignore_errors=True)
        rss = peak_rss_bytes()
        if rss > int(plan["build"]["max_rss_bytes"]):
            raise ValueError(f"{BUILD_STOP}: RSS exceeds 8 GiB")
        integrity.update(
            {
                "unique_index": INDEX_NAME,
                "lookup_opened_read_only": True,
                "labels_opened": False,
                "challenge_opened": False,
                "candidate_projection": ["candidate_siret"],
                "parity": parity,
                "reference": reference_stats,
                "verdict": GO_VERDICT,
            }
        )
        timing = {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": rss,
            "peak_rss_limit_bytes": int(plan["build"]["max_rss_bytes"]),
            "serve_latency_gate_evaluated": False,
        }
        _write_json(staging / "integrity.json", integrity)
        _write_json(staging / "timing.json", timing)
        for path, expected in (
            (plan_path, fingerprints["plan"]),
            (denylist_path, fingerprints["denylist"]),
            (snapshot, fingerprints["snapshot"]),
            (candidates, fingerprints["candidates"]),
            (execution_lock_path, fingerprints["lock"]),
            *[
                (
                    REPO_ROOT / relative,
                    fingerprints[f"source:{relative}"],
                )
                for relative in SOURCE_PATHS
            ],
        ):
            _assert_unchanged(path, expected)
        outputs = {
            name: {
                "sha256": file_sha256(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in sorted(OUTPUT_FILES)
        }
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "outputs": outputs,
            "inputs": {
                "snapshot": {
                    "path": str(snapshot),
                    "sha256": fingerprints["snapshot"]["sha256"],
                },
                "candidates_parity_only": {
                    "path": str(candidates),
                    "sha256": fingerprints["candidates"]["sha256"],
                    "projection": ["candidate_siret"],
                    "opened_after_lookup_build": True,
                },
                "execution_lock": {
                    "path": str(Path(execution_lock_path).resolve()),
                    "sha256": lock_sha,
                },
            },
            "verdict": GO_VERDICT,
        }
        _write_json(staging / "manifest.json", manifest)
        for item in staging.iterdir():
            if item.is_file():
                _fsync_file(item)
        _fsync_directory(staging)
        validate_artifact(staging, expected_build_id=build_id)
        for path, expected in (
            (plan_path, fingerprints["plan"]),
            (denylist_path, fingerprints["denylist"]),
            (snapshot, fingerprints["snapshot"]),
            (candidates, fingerprints["candidates"]),
            (execution_lock_path, fingerprints["lock"]),
            *[
                (
                    REPO_ROOT / relative,
                    fingerprints[f"source:{relative}"],
                )
                for relative in SOURCE_PATHS
            ],
        ):
            _assert_unchanged(path, expected)
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Immutable V4.12 lookup already exists: {target}"
            )
        os.replace(staging, target)
        _fsync_directory(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    validate_artifact(target)
    return target


def validate_artifact(
    path: Path,
    *,
    expected_build_id: str | None = None,
) -> None:
    root = Path(path).resolve()
    entries = list(root.iterdir())
    expected_files = OUTPUT_FILES | {"manifest.json"}
    if (
        {item.name for item in entries} != expected_files
        or any(not item.is_file() or item.is_symlink() for item in entries)
        or any(item.name.endswith(".wal") for item in entries)
    ):
        raise ValueError(f"{BUILD_STOP}: artifact file set changed")
    manifest = json.loads((root / "manifest.json").read_text())
    if set(manifest) != {
        "build_id",
        "candidate_sha256",
        "created_at",
        "execution_lock_sha256",
        "inputs",
        "outputs",
        "plan_sha256",
        "runtime",
        "schema_version",
        "snapshot_sha256",
        "source_hashes",
        "verdict",
    } or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{BUILD_STOP}: artifact schema changed")
    if (
        not _valid_sha256(manifest.get("execution_lock_sha256"))
        or not isinstance(manifest.get("created_at"), str)
        or manifest.get("source_hashes") != _source_hashes(REPO_ROOT)
        or manifest.get("runtime") != _runtime()
    ):
        raise ValueError(f"{BUILD_STOP}: artifact provenance changed")
    if (
        manifest.get("plan_sha256") != file_sha256(DEFAULT_PLAN)
        or manifest.get("snapshot_sha256") != EXPECTED_SNAPSHOT["sha256"]
        or manifest.get("candidate_sha256")
        != EXPECTED_PARITY["candidate_sha256"]
        or (manifest.get("runtime") or {}).get("duckdb")
        != EXPECTED_DUCKDB_POLICY["version"]
    ):
        raise ValueError(f"{BUILD_STOP}: frozen artifact identity changed")
    identity = {
        key: manifest[key]
        for key in (
            "schema_version",
            "plan_sha256",
            "execution_lock_sha256",
            "source_hashes",
            "runtime",
            "snapshot_sha256",
            "candidate_sha256",
        )
    }
    expected_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    required_directory_name = expected_build_id or expected_id
    if (
        manifest.get("build_id") != expected_id
        or required_directory_name != expected_id
        or (
            expected_build_id is None
            and root.name != required_directory_name
        )
    ):
        raise ValueError(f"{BUILD_STOP}: artifact identity changed")
    if manifest.get("verdict") != GO_VERDICT:
        raise ValueError(f"{BUILD_STOP}: artifact verdict changed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != OUTPUT_FILES:
        raise ValueError(f"{BUILD_STOP}: artifact outputs changed")
    for filename, record in outputs.items():
        item = root / filename
        if (
            not isinstance(record, Mapping)
            or set(record) != {"sha256", "size_bytes"}
            or not _valid_sha256(record.get("sha256"))
            or type(record.get("size_bytes")) is not int
            or record.get("size_bytes", -1) < 0
            or file_sha256(item) != record.get("sha256")
            or item.stat().st_size != record.get("size_bytes")
        ):
            raise ValueError(f"{BUILD_STOP}: output drift {filename}")
    integrity = json.loads((root / "integrity.json").read_text())
    timing = json.loads((root / "timing.json").read_text())
    if set(integrity) != {
        "candidate_projection",
        "challenge_opened",
        "invalid_siret_count",
        "labels_opened",
        "lookup_opened_read_only",
        "parity",
        "reference",
        "row_count",
        "unique_index",
        "unique_siret_count",
        "verdict",
    } or set(timing) != {
        "elapsed_seconds",
        "peak_rss_bytes",
        "peak_rss_limit_bytes",
        "serve_latency_gate_evaluated",
    }:
        raise ValueError(f"{BUILD_STOP}: artifact declarations changed")
    parity = integrity.get("parity") or {}
    reference = integrity.get("reference") or {}
    inputs = manifest.get("inputs") or {}
    lock_record = inputs.get("execution_lock") or {}
    lock_path = Path(str(lock_record.get("path") or ""))
    if (
        integrity.get("verdict") != GO_VERDICT
        or integrity.get("row_count") != EXPECTED_SNAPSHOT["row_count"]
        or integrity.get("unique_siret_count")
        != EXPECTED_SNAPSHOT["unique_siret_count"]
        or integrity.get("invalid_siret_count") != 0
        or integrity.get("unique_index") != INDEX_NAME
        or integrity.get("lookup_opened_read_only") is not True
        or integrity.get("labels_opened") is not False
        or integrity.get("challenge_opened") is not False
        or integrity.get("candidate_projection") != ["candidate_siret"]
        or set(reference) != {
            "candidate_row_count",
            "candidate_unique_siret_count",
            "reference_snapshot_scan_count",
        }
        or reference.get("candidate_row_count")
        != EXPECTED_PARITY["candidate_row_count"]
        or reference.get("candidate_unique_siret_count")
        != EXPECTED_PARITY["candidate_unique_siret_count"]
        or reference.get("reference_snapshot_scan_count") != 1
        or parity.get("mismatch_count") != 0
        or set(parity) != {
            "candidate_unique_siret_count",
            "lookup_batch_max",
            "mismatch_count",
            "reference_row_count",
            "reference_snapshot_scan_count",
            "snapshot_sample_count",
            "snapshot_sample_ordered_newline_sha256",
        }
        or parity.get("lookup_batch_max") != 100
        or parity.get("reference_snapshot_scan_count") != 1
        or parity.get("candidate_unique_siret_count")
        != EXPECTED_PARITY["candidate_unique_siret_count"]
        or parity.get("snapshot_sample_count")
        != EXPECTED_PARITY["snapshot_sample_count"]
        or parity.get("snapshot_sample_ordered_newline_sha256")
        != EXPECTED_PARITY["snapshot_sample_ordered_newline_sha256"]
        or set(inputs) != {
            "snapshot",
            "candidates_parity_only",
            "execution_lock",
        }
        or set(inputs.get("snapshot") or {}) != {"path", "sha256"}
        or (inputs.get("snapshot") or {}).get("path")
        != EXPECTED_SNAPSHOT["path"]
        or (inputs.get("snapshot") or {}).get("sha256")
        != EXPECTED_SNAPSHOT["sha256"]
        or set(inputs.get("candidates_parity_only") or {}) != {
            "opened_after_lookup_build",
            "path",
            "projection",
            "sha256",
        }
        or (inputs.get("candidates_parity_only") or {}).get("path")
        != EXPECTED_PARITY["candidate_path"]
        or (inputs.get("candidates_parity_only") or {}).get("sha256")
        != EXPECTED_PARITY["candidate_sha256"]
        or (inputs.get("candidates_parity_only") or {}).get("projection")
        != ["candidate_siret"]
        or (inputs.get("candidates_parity_only") or {}).get(
            "opened_after_lookup_build"
        )
        is not True
        or set(lock_record) != {"path", "sha256"}
        or not lock_path.is_file()
        or lock_path.is_symlink()
        or file_sha256(lock_path) != manifest.get("execution_lock_sha256")
        or lock_record.get("sha256")
        != manifest.get("execution_lock_sha256")
        or timing.get("serve_latency_gate_evaluated") is not False
        or not isinstance(timing.get("peak_rss_bytes"), int)
        or timing.get("peak_rss_limit_bytes") != 8 * 1024**3
        or timing.get("peak_rss_bytes")
        > timing.get("peak_rss_limit_bytes", -1)
        or any(
            isinstance(timing.get(key), bool)
            or not isinstance(timing.get(key), (int, float))
            or not math.isfinite(float(timing[key]))
            or float(timing[key]) < 0
            for key in ("elapsed_seconds", "peak_rss_bytes")
        )
    ):
        raise ValueError(f"{BUILD_STOP}: integrity declaration changed")
    validate_execution_lock(
        lock_path,
        plan_path=DEFAULT_PLAN,
        denylist_path=DEFAULT_DENYLIST,
        verify_git=True,
    )
    with V412SnapshotLookup(root / "candidate_details.duckdb") as store:
        count = int(
            store._connection.execute(
                f"SELECT count(*) FROM {TABLE_NAME}"
            ).fetchone()[0]
        )
    if (
        count != int(integrity.get("row_count", -1))
        or count != int(integrity.get("unique_siret_count", -2))
        or int(integrity.get("invalid_siret_count", -1)) != 0
    ):
        raise ValueError(f"{BUILD_STOP}: database cardinality changed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST)
    parser.add_argument("--validate-artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact is not None:
        validate_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    if args.execution_lock is None:
        raise SystemExit("--execution-lock is required for a real build")
    print(
        build_artifact(
            execution_lock_path=args.execution_lock,
            plan_path=args.plan,
            denylist_path=args.denylist,
        )
    )


if __name__ == "__main__":
    main()
