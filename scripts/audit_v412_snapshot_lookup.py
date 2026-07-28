#!/usr/bin/env python3
"""Independently audit the frozen V4.12 snapshot lookup."""

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
from typing import Any, Mapping, Sequence

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v412_snapshot_lookup import V412SnapshotLookup


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = (
    REPO_ROOT
    / "config/v4_12_snapshot_lookup_independent_audit_plan.json"
)
DEFAULT_LOCK = (
    REPO_ROOT
    / "config/v4_12_snapshot_lookup_independent_audit_execution_lock.json"
)
OFFICIAL_VALIDATOR = REPO_ROOT / "scripts/build_v412_snapshot_lookup.py"
SCHEMA_VERSION = "sireto-v4.12-lookup-independent-audit-1"
LOCK_SCHEMA_VERSION = "sireto-v4.12-lookup-independent-audit-lock-1"
PURPOSE = "AUDIT_V412_SNAPSHOT_LOOKUP"
AUDIT_VERDICT = "GO_AUDIT_V412_SNAPSHOT_LOOKUP"
GO = "GO_V412_LOOKUP_INDEPENDENT_AUDIT"
STOP = "STOP_V412_LOOKUP_AUDIT"
STOP_SAMPLE = "STOP_V412_LOOKUP_SAMPLE"
OUTPUT_FILES = {"audit.json"}
SSD_ROOT = Path("/Volumes/CATNAT_DATA")
LOCK_FIELDS = {
    "schema_version",
    "purpose",
    "audit_verdict",
    "git_commit",
    "source_hashes",
    "input_paths",
    "input_hashes",
    "runtime",
    "output_root",
    "temp_root",
}


def file_sha256(path: Path) -> str:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise ValueError(f"{STOP}: expected a regular file")
    digest = hashlib.sha256()
    with raw.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "duckdb": importlib.metadata.version("duckdb"),
    }


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    raw = Path(path)
    file_sha256(raw)
    value = json.loads(raw.read_text("utf-8"))
    if (
        value.get("schema_version")
        != "sireto-v4.12-lookup-independent-audit-plan-1"
        or value.get("duckdb", {}).get("version") != "1.4.3"
        or value.get("duckdb", {}).get("threads") != 1
        or value.get("duckdb", {}).get("memory_limit") != "6GB"
        or value.get("expected", {}).get("batch_max") != 100
    ):
        raise ValueError(f"{STOP}: independent audit plan changed")
    return value


def source_hashes(plan: Mapping[str, Any]) -> dict[str, str]:
    expected = list(plan["audit_sources"])
    if expected.count("scripts/audit_v412_snapshot_lookup.py") != 1:
        raise ValueError(f"{STOP}: audit source set changed")
    return {
        relative: file_sha256(REPO_ROOT / relative)
        for relative in expected
    }


def fingerprint(path: Path) -> dict[str, Any]:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise ValueError(f"{STOP}: watched input is not a regular file")
    stat = raw.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(raw),
    }


def assert_unchanged(path: Path, expected: Mapping[str, Any]) -> None:
    if fingerprint(path) != dict(expected):
        raise ValueError(f"{STOP}: input changed during audit")


def validate_execution_lock(
    path: Path,
    *,
    plan_path: Path = DEFAULT_PLAN,
) -> tuple[dict[str, Any], str]:
    file_sha256(path)
    raw = Path(path).read_bytes()
    lock = json.loads(raw)
    if set(lock) != LOCK_FIELDS:
        raise ValueError(f"{STOP}: execution lock fields changed")
    if (
        lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("purpose") != PURPOSE
        or lock.get("audit_verdict") != AUDIT_VERDICT
    ):
        raise ValueError(f"{STOP}: independent audit is not authorized")
    plan = load_plan(plan_path)
    sources = source_hashes(plan)
    artifact = Path(plan["artifact"]["path"]).resolve()
    snapshot = Path(plan["snapshot"]["path"]).resolve()
    expected_paths = {
        "plan": str(Path(plan_path).resolve()),
        "snapshot": str(snapshot),
        "artifact": str(artifact),
        "official_validator": str(OFFICIAL_VALIDATOR.resolve()),
    }
    expected_hashes = {
        "plan": file_sha256(plan_path),
        "snapshot": plan["snapshot"]["sha256"],
        "artifact_manifest": plan["artifact"]["files"]["manifest.json"][
            "sha256"
        ],
        "official_validator": file_sha256(OFFICIAL_VALIDATOR),
    }
    if (
        lock.get("source_hashes") != sources
        or lock.get("input_paths") != expected_paths
        or lock.get("input_hashes") != expected_hashes
        or lock.get("runtime") != runtime()
        or Path(lock.get("output_root", "")).resolve()
        != Path(plan["output_root"]).resolve()
        or Path(lock.get("temp_root", "")).resolve()
        != Path(plan["duckdb"]["temp_root"]).resolve()
    ):
        raise ValueError(f"{STOP}: locked environment changed")
    commit = str(lock.get("git_commit") or "")
    if not commit:
        raise ValueError(f"{STOP}: locked commit missing")
    for relative, digest in sources.items():
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        if hashlib.sha256(result.stdout).hexdigest() != digest:
            raise ValueError(f"{STOP}: commit does not pin {relative}")
    return lock, hashlib.sha256(raw).hexdigest()


def run_official_validator(artifact: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(OFFICIAL_VALIDATOR),
            "--validate-artifact",
            str(Path(artifact).resolve()),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"{STOP}: official validator rejected artifact")


def validate_artifact_files(
    artifact: Path,
    plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = Path(artifact)
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError(f"{STOP}: artifact root is invalid")
    expected = plan["artifact"]["files"]
    entries = list(raw.iterdir())
    if (
        {item.name for item in entries} != set(expected)
        or any(not item.is_file() or item.is_symlink() for item in entries)
        or any(item.name.endswith(".wal") for item in entries)
    ):
        raise ValueError(f"{STOP}: artifact file set changed")
    observed = {}
    for filename, record in expected.items():
        path = raw / filename
        observed[filename] = fingerprint(path)
        if (
            observed[filename]["sha256"] != record["sha256"]
            or observed[filename]["size_bytes"] != record["size_bytes"]
        ):
            raise ValueError(f"{STOP}: artifact file changed")
    return observed


def validate_lookup_integrity(
    database: Path,
    plan: Mapping[str, Any],
) -> None:
    expected = plan["lookup_schema"]
    connection = duckdb.connect(str(database), read_only=True)
    try:
        tables = connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_type='BASE TABLE' "
            "ORDER BY table_name"
        ).fetchall()
        columns = connection.execute(
            "SELECT column_name,data_type FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name=? "
            "ORDER BY ordinal_position",
            [expected["table_name"]],
        ).fetchall()
        cardinality = connection.execute(
            "SELECT count(*),count(DISTINCT siret),"
            "count(*) FILTER(WHERE siret IS NULL OR "
            "NOT regexp_full_match(siret,'[0-9]{14}')) "
            "FROM candidate_details"
        ).fetchone()
        indexes = connection.execute(
            "SELECT index_name,is_unique,expressions FROM duckdb_indexes() "
            "WHERE schema_name='main' AND table_name='candidate_details'"
        ).fetchall()
    finally:
        connection.close()
    if (
        tables != [(expected["table_name"],)]
        or [row[0] for row in columns] != expected["columns"]
        or any(row[1] != expected["column_type"] for row in columns)
        or tuple(map(int, cardinality))
        != (
            expected["row_count"],
            expected["unique_siret_count"],
            expected["invalid_siret_count"],
        )
        or len(indexes) != 1
        or indexes[0][0] != expected["index_name"]
        or indexes[0][1] is not True
        or str(indexes[0][2]).replace('"', "")
        != expected["index_expressions"]
    ):
        raise ValueError(f"{STOP}: lookup integrity changed")


def _configure(
    connection: duckdb.DuckDBPyConnection,
    plan: Mapping[str, Any],
    temp_directory: Path,
) -> None:
    if temp_directory.exists():
        if (
            temp_directory.is_symlink()
            or not temp_directory.is_dir()
            or any(temp_directory.iterdir())
        ):
            raise ValueError(f"{STOP}: DuckDB temp directory is unsafe")
    else:
        temp_directory.mkdir(parents=True, exist_ok=False)
    connection.execute("SET threads=1")
    connection.execute("SET memory_limit='6GB'")
    escaped = str(temp_directory.resolve()).replace("'", "''")
    connection.execute(f"SET temp_directory='{escaped}'")


def select_sirets_phase_a(
    snapshot: Path,
    plan: Mapping[str, Any],
    temp_directory: Path,
) -> list[str]:
    selection = plan["selection"]
    connection = duckdb.connect()
    try:
        _configure(connection, plan, temp_directory)
        rows = connection.execute(
            """
            SELECT siret
            FROM (
                SELECT DISTINCT CAST(siret AS VARCHAR) AS siret
                FROM read_parquet(?)
                WHERE regexp_full_match(
                    CAST(siret AS VARCHAR), ?
                )
            )
            ORDER BY sha256(? || siret), siret
            LIMIT ?
            """,
            [
                str(Path(snapshot).resolve()),
                selection["canonical_regex"],
                selection["namespace"],
                int(plan["expected"]["sample_count"]),
            ],
        ).fetchall()
    finally:
        connection.close()
        shutil.rmtree(temp_directory, ignore_errors=True)
    values = [str(row[0]) for row in rows]
    if (
        len(values) != plan["expected"]["sample_count"]
        or values[:3] != plan["expected"]["first_sirets"]
        or values[-3:] != plan["expected"]["last_sirets"]
    ):
        raise ValueError(f"{STOP_SAMPLE}: SIRET selection changed")
    return values


def sample_payloads(
    sirets: Sequence[str],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    lf = bytearray()
    counterexample = bytearray()
    for siret in sirets:
        encoded = siret.encode("ascii")
        if len(encoded) != 14 or not encoded.isdigit():
            raise ValueError(f"{STOP_SAMPLE}: invalid selected SIRET")
        lf.extend(encoded)
        lf.extend(bytes([10]))
        counterexample.extend(encoded)
        counterexample.extend(bytes([92, 110]))
    result = {
        "lf_payload_bytes": len(lf),
        "lf_payload_sha256": hashlib.sha256(lf).hexdigest(),
        "counterexample_bytes": len(counterexample),
        "counterexample_sha256": hashlib.sha256(counterexample).hexdigest(),
    }
    if (
        result["lf_payload_bytes"] != expected["lf_payload_bytes"]
        or result["lf_payload_sha256"] != expected["lf_payload_sha256"]
        or result["counterexample_bytes"] != expected["counterexample_bytes"]
        or result["counterexample_sha256"]
        != expected["counterexample_sha256"]
        or result["lf_payload_sha256"]
        == result["counterexample_sha256"]
    ):
        raise ValueError(f"{STOP_SAMPLE}: payload encoding changed")
    return result


def project_values_phase_b(
    snapshot: Path,
    sirets: Sequence[str],
    plan: Mapping[str, Any],
    temp_directory: Path,
) -> dict[str, tuple[Any, ...]]:
    connection = duckdb.connect()
    try:
        _configure(connection, plan, temp_directory)
        connection.execute("CREATE TEMP TABLE selected(siret VARCHAR)")
        connection.executemany(
            "INSERT INTO selected VALUES (?)", [(value,) for value in sirets]
        )
        expressions = ", ".join(
            f"{item['sql'].replace(item['snapshot'], 'snapshot.' + item['snapshot'])} "
            f"AS {item['lookup']}"
            for item in plan["evidence_projection"]
        )
        snapshot_sql = str(Path(snapshot).resolve()).replace("'", "''")
        rows = connection.execute(
            f"SELECT {expressions} "
            f"FROM read_parquet('{snapshot_sql}') snapshot "
            "INNER JOIN selected USING(siret) ORDER BY siret"
        ).fetchall()
    finally:
        connection.close()
        shutil.rmtree(temp_directory, ignore_errors=True)
    result = {str(row[0]): tuple(row[1:]) for row in rows}
    if len(result) != len(sirets) or set(result) != set(sirets):
        raise ValueError(f"{STOP_SAMPLE}: snapshot projection incomplete")
    return result


def compare_lookup(
    database: Path,
    sirets: Sequence[str],
    expected_values: Mapping[str, tuple[Any, ...]],
    *,
    batch_max: int,
) -> int:
    if not 1 <= batch_max <= 100:
        raise ValueError(f"{STOP}: invalid lookup batch maximum")
    detail_names = [
        "candidate_state",
        "enseigne1",
        "enseigne2",
        "enseigne3",
        "denomination_usuelle",
        "activity_code",
    ]
    mismatch = 0
    with V412SnapshotLookup(database) as store:
        for offset in range(0, len(sirets), batch_max):
            batch = list(sirets[offset : offset + batch_max])
            observed = store.get_candidate_scene_details(batch)
            if set(observed) != set(batch):
                mismatch += len(set(observed) ^ set(batch))
                continue
            for siret in batch:
                values = tuple(observed[siret][name] for name in detail_names)
                if values != expected_values[siret]:
                    mismatch += 1
    if mismatch:
        raise ValueError(f"{STOP_SAMPLE}: lookup values differ")
    return mismatch


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    data = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_publication(
    root: Path,
    identity: Mapping[str, Any],
    expected_audit: Mapping[str, Any],
    *,
    build_id: str,
    max_rss_bytes: int,
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{STOP}: audit publication root changed")
    entries = list(root.iterdir())
    if (
        {item.name for item in entries} != {"audit.json", "manifest.json"}
        or any(not item.is_file() or item.is_symlink() for item in entries)
    ):
        raise ValueError(f"{STOP}: audit publication file set changed")
    audit_record = json.loads((root / "audit.json").read_text("utf-8"))
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    audit_fields = {
        "schema_version",
        "sample_count",
        "first_sirets",
        "last_sirets",
        "lf_payload_bytes",
        "lf_payload_sha256",
        "counterexample_bytes",
        "counterexample_sha256",
        "mismatch_count",
        "peak_rss_bytes",
        "verdict",
    }
    manifest_fields = set(identity) | {
        "build_id",
        "created_at",
        "outputs",
        "verdict",
    }
    if (
        set(audit_record) != audit_fields
        or audit_record != dict(expected_audit)
        or type(audit_record.get("sample_count")) is not int
        or type(audit_record.get("lf_payload_bytes")) is not int
        or type(audit_record.get("counterexample_bytes")) is not int
        or type(audit_record.get("mismatch_count")) is not int
        or type(audit_record.get("peak_rss_bytes")) is not int
        or not isinstance(audit_record.get("first_sirets"), list)
        or not isinstance(audit_record.get("last_sirets"), list)
        or any(
            not isinstance(value, str)
            for value in (
                audit_record.get("first_sirets", [])
                + audit_record.get("last_sirets", [])
            )
        )
        or audit_record.get("mismatch_count") != 0
        or audit_record.get("peak_rss_bytes", -1) < 0
        or audit_record.get("peak_rss_bytes", max_rss_bytes + 1)
        > max_rss_bytes
        or set(manifest) != manifest_fields
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("verdict") != GO
        or any(manifest.get(key) != value for key, value in identity.items())
        or manifest.get("build_id") != build_id
        or not isinstance(manifest.get("created_at"), str)
        or set(manifest.get("outputs") or {}) != OUTPUT_FILES
    ):
        raise ValueError(f"{STOP}: audit publication declarations changed")
    try:
        created_at = datetime.fromisoformat(manifest["created_at"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{STOP}: audit publication timestamp changed"
        ) from error
    if created_at.tzinfo is None:
        raise ValueError(f"{STOP}: audit publication timestamp changed")
    record = manifest["outputs"]["audit.json"]
    if (
        set(record) != {"sha256", "size_bytes"}
        or type(record.get("size_bytes")) is not int
        or record["sha256"] != file_sha256(root / "audit.json")
        or record["size_bytes"] != (root / "audit.json").stat().st_size
    ):
        raise ValueError(f"{STOP}: audit publication hash changed")


def validate_inputs_unchanged(
    watched: Sequence[Path],
    initial: Mapping[str, Mapping[str, Any]],
    artifact: Path,
    artifact_initial: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> None:
    for path in watched:
        assert_unchanged(path, initial[str(path)])
    if validate_artifact_files(artifact, plan) != dict(artifact_initial):
        raise ValueError(f"{STOP}: artifact changed during audit")


def audit(
    *,
    execution_lock: Path,
    plan_path: Path = DEFAULT_PLAN,
) -> Path:
    lock, lock_hash = validate_execution_lock(
        execution_lock, plan_path=plan_path
    )
    plan = load_plan(plan_path)
    artifact = Path(plan["artifact"]["path"])
    snapshot = Path(plan["snapshot"]["path"])
    source_files = [REPO_ROOT / item for item in plan["audit_sources"]]
    watched = [plan_path, execution_lock, snapshot, *source_files]
    initial = {str(path): fingerprint(path) for path in watched}
    artifact_initial = validate_artifact_files(artifact, plan)
    if file_sha256(snapshot) != plan["snapshot"]["sha256"]:
        raise ValueError(f"{STOP}: snapshot changed")
    run_official_validator(artifact)
    validate_lookup_integrity(artifact / "candidate_details.duckdb", plan)
    raw_temp_root = Path(lock["temp_root"])
    raw_output_root = Path(lock["output_root"])
    if raw_temp_root.is_symlink() or raw_output_root.is_symlink():
        raise ValueError(f"{STOP}: output root is symbolic")
    temp_root = raw_temp_root.resolve()
    output_root = raw_output_root.resolve()
    if (
        not temp_root.is_relative_to(SSD_ROOT.resolve())
        or not output_root.is_relative_to(SSD_ROOT.resolve())
    ):
        raise ValueError(f"{STOP}: output escaped SSD")
    temp_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    phase_a_tmp = Path(tempfile.mkdtemp(prefix="phase-a-", dir=temp_root))
    sirets = select_sirets_phase_a(snapshot, plan, phase_a_tmp)
    payloads = sample_payloads(sirets, plan["expected"])
    phase_b_tmp = Path(tempfile.mkdtemp(prefix="phase-b-", dir=temp_root))
    expected_values = project_values_phase_b(
        snapshot, sirets, plan, phase_b_tmp
    )
    mismatch = compare_lookup(
        artifact / "candidate_details.duckdb",
        sirets,
        expected_values,
        batch_max=plan["expected"]["batch_max"],
    )
    rss = peak_rss_bytes()
    if rss > plan["max_rss_bytes"]:
        raise ValueError(f"{STOP}: RSS exceeds limit")
    validate_inputs_unchanged(
        watched, initial, artifact, artifact_initial, plan
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "execution_lock_sha256": lock_hash,
        "plan_sha256": file_sha256(plan_path),
        "snapshot_sha256": plan["snapshot"]["sha256"],
        "artifact_manifest_sha256": plan["artifact"]["files"][
            "manifest.json"
        ]["sha256"],
        "source_hashes": lock["source_hashes"],
        "runtime": runtime(),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = output_root / build_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{STOP}: immutable audit exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    published = False
    try:
        audit_record = {
            "schema_version": SCHEMA_VERSION,
            "sample_count": len(sirets),
            "first_sirets": sirets[:3],
            "last_sirets": sirets[-3:],
            **payloads,
            "mismatch_count": mismatch,
            "peak_rss_bytes": rss,
            "verdict": GO,
        }
        _write_json(staging / "audit.json", audit_record)
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                "audit.json": {
                    "sha256": file_sha256(staging / "audit.json"),
                    "size_bytes": (staging / "audit.json").stat().st_size,
                }
            },
            "verdict": GO,
        }
        _write_json(staging / "manifest.json", manifest)
        validate_publication(
            staging,
            identity,
            audit_record,
            build_id=build_id,
            max_rss_bytes=plan["max_rss_bytes"],
        )
        validate_inputs_unchanged(
            watched, initial, artifact, artifact_initial, plan
        )
        _fsync_directory(staging)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"{STOP}: immutable audit exists")
        os.replace(staging, target)
        published = True
        validate_publication(
            target,
            identity,
            audit_record,
            build_id=build_id,
            max_rss_bytes=plan["max_rss_bytes"],
        )
        _fsync_directory(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if published:
            shutil.rmtree(target, ignore_errors=True)
            _fsync_directory(output_root)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    if args.execution_lock is None:
        raise SystemExit("--execution-lock is required")
    print(audit(execution_lock=args.execution_lock, plan_path=args.plan))


if __name__ == "__main__":
    main()
