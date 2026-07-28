from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import stat
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import audit_v412_unit_retrieval_parity as parity


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(parity.canonical_json(value))


def _schema_description(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "nullable": field.nullable,
            "type": str(field.type),
        }
        for field in schema
    ]


def _parquet_record(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
        "row_count": parquet.metadata.num_rows,
        "schema": _schema_description(parquet.schema_arrow),
        "metadata": None,
    }


def _payloads(
    query_ids: list[str],
    status_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_digest = hashlib.sha256()
    candidate_bytes = 0
    for row in candidate_rows:
        payload = (
            row["query_id"].encode()
            + b"\0"
            + row["candidate_siret"].encode()
            + b"\0"
            + str(row["candidate_rank"]).encode()
            + b"\n"
        )
        candidate_digest.update(payload)
        candidate_bytes += len(payload)
    status_digest = hashlib.sha256()
    status_bytes = 0
    for row in status_rows:
        payload = (
            row["query_id"].encode()
            + b"\0"
            + str(row["candidate_count"]).encode()
            + b"\n"
        )
        status_digest.update(payload)
        status_bytes += len(payload)
    counts = [row["candidate_count"] for row in status_rows]
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


def _query_payload(query_ids: list[str]) -> str:
    return hashlib.sha256(
        b"".join(query_id.encode() + b"\n" for query_id in query_ids)
    ).hexdigest()


def _sandbox_checks() -> dict[str, bool]:
    return {key: True for key in parity.SANDBOX_CHECK_KEYS}


def _worker_checks() -> dict[str, bool]:
    return {
        "allowed_read": True,
        "oracle_denied": True,
        "oracle_audit_denied": True,
        "historical_denied": True,
        "model_denied": True,
        "network_denied": True,
        "write_denied": True,
    }


def _make_fixture(
    tmp_path: Path,
    *,
    status_rows: list[dict[str, Any]] | None = None,
    candidate_rows: list[dict[str, Any]] | None = None,
    status_schema: pa.Schema | None = None,
    candidate_schema: pa.Schema | None = None,
) -> dict[str, Any]:
    source = Path(parity.__file__).resolve()
    profile = source.parents[1] / parity.PROFILE_RELATIVE_PATH
    safe_root = tmp_path / "safe"
    worker_root = tmp_path / "worker"
    temp_root = tmp_path / "temp"
    audit_root = tmp_path / "audit"
    sentinels = tmp_path / "sentinels"
    for path in (safe_root, worker_root, temp_root, audit_root, sentinels):
        path.mkdir()
    query_ids = ["q-alpha", "q-beta", "q-empty"]
    safe_table = pa.Table.from_arrays(
        [
            pa.array(query_ids, type=pa.string()),
            pa.array(["Alpha", "Beta", ""], type=pa.string()),
            pa.array(["1 rue A", "2 rue B", ""], type=pa.string()),
            pa.array(["75001", "69001", ""], type=pa.string()),
            pa.array(["Paris", "Lyon", ""], type=pa.string()),
            pa.array(["75056", "69123", ""], type=pa.string()),
        ],
        schema=parity.SAFE_QUERY_SCHEMA,
    )
    safe_queries = safe_root / "queries_dev.parquet"
    pq.write_table(safe_table, safe_queries)
    safe_manifest = safe_root / "runtime_manifest.json"
    safe_build_id = "safe-build"
    _write_json(
        safe_manifest,
        {
            "schema_version": "synthetic-safe-manifest-1",
            "build_id": safe_build_id,
            "files": {
                "queries_dev.parquet": _parquet_record(safe_queries),
            },
        },
    )
    status_rows = status_rows or [
        {"query_id": "q-alpha", "candidate_count": 2},
        {"query_id": "q-beta", "candidate_count": 1},
        {"query_id": "q-empty", "candidate_count": 0},
    ]
    candidate_rows = candidate_rows or [
        {
            "query_id": "q-alpha",
            "candidate_rank": 1,
            "candidate_siret": "11111111111111",
        },
        {
            "query_id": "q-alpha",
            "candidate_rank": 2,
            "candidate_siret": "22222222222222",
        },
        {
            "query_id": "q-beta",
            "candidate_rank": 1,
            "candidate_siret": "33333333333333",
        },
    ]
    status_path = worker_root / "query_status.parquet"
    candidate_path = worker_root / "candidates_top100.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            status_rows, schema=status_schema or parity.STATUS_SCHEMA
        ),
        status_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            candidate_rows, schema=candidate_schema or parity.CANDIDATE_SCHEMA
        ),
        candidate_path,
    )
    actual = _payloads(query_ids, status_rows, candidate_rows)
    worker_build_id = "worker-build"
    integrity = {
        "schema_version": parity.WORKER_INTEGRITY_SCHEMA,
        "worker_build_id": worker_build_id,
        **actual,
        "lookup_missing_count": 0,
        "sandbox_checks": _worker_checks(),
        "peak_rss_bytes": 1024,
        "durations_ns": {
            "retrieval": 1,
            "lookup": 1,
            "serialization": 1,
            "total": 3,
        },
        "declarations": dict(parity.WORKER_DECLARATIONS),
    }
    integrity_path = worker_root / "integrity.json"
    _write_json(integrity_path, integrity)
    runtime = parity.runtime_identity()
    worker_manifest = {
        "schema_version": parity.WORKER_MANIFEST_SCHEMA,
        "worker_build_id": worker_build_id,
        "safe_input_build_id": safe_build_id,
        "strict_stores_build_id": "strict-build",
        "files": {
            "query_status.parquet": _parquet_record(status_path),
            "candidates_top100.parquet": _parquet_record(candidate_path),
            "integrity.json": {
                "sha256": _sha(integrity_path),
                "size_bytes": integrity_path.stat().st_size,
            },
        },
        "runtime": runtime,
        "declarations": dict(parity.WORKER_DECLARATIONS),
        "verdict": parity.WORKER_VERDICT,
    }
    worker_manifest_path = worker_root / "manifest.json"
    _write_json(worker_manifest_path, worker_manifest)
    worker_paths = {
        "query_status.parquet": str(status_path.resolve()),
        "candidates_top100.parquet": str(candidate_path.resolve()),
        "integrity.json": str(integrity_path.resolve()),
    }
    worker_hashes = {
        name: _sha(Path(path)) for name, path in worker_paths.items()
    }
    sandbox_executable = Path("/usr/bin/sandbox-exec")
    if not sandbox_executable.is_file():
        sandbox_executable = Path(sys.executable).resolve()
    python_executable = parity._default_python_executable()
    run_spec = {
        "schema_version": parity.RUN_SPEC_SCHEMA,
        "worker_build_id": worker_build_id,
        "worker_manifest_path": str(worker_manifest_path.resolve()),
        "worker_manifest_sha256": _sha(worker_manifest_path),
        "worker_file_paths": worker_paths,
        "worker_file_hashes": worker_hashes,
        "safe_input_build_id": safe_build_id,
        "safe_queries_path": str(safe_queries.resolve()),
        "safe_queries_sha256": _sha(safe_queries),
        "safe_manifest_path": str(safe_manifest.resolve()),
        "safe_manifest_sha256": _sha(safe_manifest),
        "safe_query_id_payload_sha256": _query_payload(query_ids),
        "expected": dict(actual),
        "git_commit": "a" * 40,
        "lock_sha256": "b" * 64,
        "parity_source_hashes": {
            parity.SOURCE_RELATIVE_PATH: _sha(source),
        },
        "parity_profile_sha256": _sha(profile),
        "sandbox_executable_path": str(sandbox_executable),
        "sandbox_executable_sha256": _sha(sandbox_executable),
        "python_executable_path": str(python_executable),
        "python_executable_sha256": _sha(python_executable),
        "audit_root_path_sha256": parity.path_commitment(
            audit_root.resolve()
        ),
        "runtime": runtime,
        "temp_root": str(temp_root.resolve()),
        "max_rss_bytes": 2 * 1024 * 1024 * 1024,
        "declarations": dict(parity.DECLARATIONS),
    }
    run_spec_path = tmp_path / "parity_run_spec.json"
    _write_json(run_spec_path, run_spec)
    sentinel_paths: dict[str, Path] = {}
    for name in ("oracle", "oracle_audit", "historical", "model", "store"):
        path = sentinels / name
        path.write_text("forbidden\n", encoding="utf-8")
        sentinel_paths[name] = path
    sentinel_paths["write"] = sentinels / "must-not-be-created"
    return {
        "query_ids": query_ids,
        "actual": actual,
        "spec": run_spec,
        "run_spec": run_spec_path,
        "profile": profile,
        "audit_root": audit_root,
        "safe_queries": safe_queries,
        "safe_manifest": safe_manifest,
        "worker_manifest": worker_manifest_path,
        "integrity": integrity_path,
        "status": status_path,
        "candidates": candidate_path,
        "sentinels": sentinel_paths,
    }


def _open_fixture_fds(fixture: dict[str, Any]) -> dict[str, int]:
    return {
        "safe_queries": os.open(fixture["safe_queries"], os.O_RDONLY),
        "safe_manifest": os.open(fixture["safe_manifest"], os.O_RDONLY),
        "worker_manifest": os.open(fixture["worker_manifest"], os.O_RDONLY),
        "integrity": os.open(fixture["integrity"], os.O_RDONLY),
        "status": os.open(fixture["status"], os.O_RDONLY),
        "candidates": os.open(fixture["candidates"], os.O_RDONLY),
    }


def _evaluate(fixture: dict[str, Any]) -> dict[str, Any]:
    payload = fixture["run_spec"].read_bytes()
    spec = parity.parse_json(payload, "test run-spec")
    descriptors = _open_fixture_fds(fixture)
    try:
        return parity.evaluate_from_fds(
            spec,
            parity_id=parity.parity_build_id(
                spec, hashlib.sha256(payload).hexdigest()
            ),
            safe_queries_fd=descriptors["safe_queries"],
            safe_manifest_fd=descriptors["safe_manifest"],
            worker_manifest_fd=descriptors["worker_manifest"],
            worker_integrity_fd=descriptors["integrity"],
            status_fd=descriptors["status"],
            candidates_fd=descriptors["candidates"],
            sandbox_checks=_sandbox_checks(),
        )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def _reseal_worker(fixture: dict[str, Any]) -> None:
    manifest = json.loads(fixture["worker_manifest"].read_text())
    manifest["files"]["query_status.parquet"] = _parquet_record(
        fixture["status"]
    )
    manifest["files"]["candidates_top100.parquet"] = _parquet_record(
        fixture["candidates"]
    )
    _write_json(fixture["worker_manifest"], manifest)
    spec = json.loads(fixture["run_spec"].read_text())
    for name, key in (
        ("query_status.parquet", "status"),
        ("candidates_top100.parquet", "candidates"),
        ("integrity.json", "integrity"),
    ):
        spec["worker_file_hashes"][name] = _sha(fixture[key])
    spec["worker_manifest_sha256"] = _sha(fixture["worker_manifest"])
    _write_json(fixture["run_spec"], spec)
    fixture["spec"] = spec


def _snapshot_record(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        snapshot = parity._snapshot_fd(descriptor, 1 << 30)
    finally:
        os.close(descriptor)
    return {
        "path": path.name,
        "size_bytes": snapshot["size"],
        "sha256": snapshot["sha256"],
    }


def _publish_recovery_fixture(
    fixture: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    spec_payload = fixture["run_spec"].read_bytes()
    spec = parity.parse_json(spec_payload, "recovery fixture run-spec")
    run_spec_sha256 = hashlib.sha256(spec_payload).hexdigest()
    parity_id = parity.parity_build_id(spec, run_spec_sha256)
    report = _evaluate(fixture)
    final = fixture["audit_root"] / parity_id
    final.mkdir(mode=0o700)
    parity_path = final / "parity.json"
    provenance_path = final / "provenance.json"
    manifest_path = final / "manifest.json"
    parity._write_exclusive(parity_path, parity.canonical_json(report))
    provenance = {
        "schema_version": parity.PROVENANCE_SCHEMA,
        "parity_build_id": parity_id,
        "worker_build_id": spec["worker_build_id"],
        "git_commit": spec["git_commit"],
        "parity_source_hashes": spec["parity_source_hashes"],
        "lock_sha256": spec["lock_sha256"],
        "parity_run_spec_sha256": run_spec_sha256,
        "worker_manifest_sha256": spec["worker_manifest_sha256"],
        "runtime": spec["runtime"],
        "declarations": dict(parity.DECLARATIONS),
    }
    parity._write_exclusive(
        provenance_path,
        parity.canonical_json(provenance),
    )
    manifest = {
        "schema_version": parity.MANIFEST_SCHEMA,
        "parity_build_id": parity_id,
        "worker_build_id": spec["worker_build_id"],
        "files": sorted(
            [
                _snapshot_record(parity_path),
                _snapshot_record(provenance_path),
            ],
            key=lambda row: row["path"].encode(),
        ),
        "runtime": spec["runtime"],
        "declarations": dict(parity.DECLARATIONS),
        "verdict": report["verdict"],
    }
    parity._write_exclusive(manifest_path, parity.canonical_json(manifest))
    os.chmod(final, 0o555)
    return final, report


def test_synthetic_parity_matches_exact_payloads(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    report = _evaluate(fixture)
    assert report["verdict"] == parity.GO
    assert set(report) == parity.PARITY_KEYS
    assert report["candidate_payload_sha256"] == fixture["actual"][
        "candidate_payload_sha256"
    ]
    assert report["status_payload_sha256"] == fixture["actual"][
        "status_payload_sha256"
    ]
    assert all(report["checks"].values())


def test_cryptographic_mismatch_is_aggregate_stop(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    spec = json.loads(fixture["run_spec"].read_text())
    spec["expected"]["candidate_payload_sha256"] = "0" * 64
    _write_json(fixture["run_spec"], spec)
    report = _evaluate(fixture)
    assert report["verdict"] == parity.STOP
    assert report["checks"]["candidate_payload"] is False
    assert "candidate_siret" not in report
    assert "query_id" not in report


def test_rank_gap_is_rejected_even_when_file_is_resealed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    rows = pq.read_table(fixture["candidates"]).to_pylist()
    rows[1]["candidate_rank"] = 3
    pq.write_table(
        pa.Table.from_pylist(rows, schema=parity.CANDIDATE_SCHEMA),
        fixture["candidates"],
    )
    _reseal_worker(fixture)
    with pytest.raises(parity.ParityStopped, match="contiguous rank"):
        _evaluate(fixture)


def test_candidate_ceiling_is_per_query_and_fail_closed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    rows = pq.read_table(fixture["status"]).to_pylist()
    rows[0]["candidate_count"] = 101
    pq.write_table(
        pa.Table.from_pylist(rows, schema=parity.STATUS_SCHEMA),
        fixture["status"],
    )
    _reseal_worker(fixture)
    with pytest.raises(parity.ParityStopped, match="strict per-query ceiling"):
        _evaluate(fixture)


def test_arrow_metadata_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    schema = parity.STATUS_SCHEMA.with_metadata({b"forbidden": b"value"})
    rows = pq.read_table(fixture["status"]).to_pylist()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), fixture["status"])
    _reseal_worker(fixture)
    with pytest.raises(parity.ParityStopped, match="metadata"):
        _evaluate(fixture)


def test_duplicate_siret_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    rows = pq.read_table(fixture["candidates"]).to_pylist()
    rows[1]["candidate_siret"] = rows[0]["candidate_siret"]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=parity.CANDIDATE_SCHEMA),
        fixture["candidates"],
    )
    _reseal_worker(fixture)
    with pytest.raises(parity.ParityStopped, match="duplicate candidate"):
        _evaluate(fixture)


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "/tmp/datasets/legacy",
        "/tmp/audits/oracle-derived-staging",
        "/tmp/AUDITS",
    ],
)
def test_run_spec_rejects_historical_or_oracle_paths(
    tmp_path: Path,
    forbidden_path: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    spec = json.loads(fixture["run_spec"].read_text())
    spec["temp_root"] = forbidden_path
    with pytest.raises(
        parity.ParityStopped,
        match=r"forbidden (?:content|path component)",
    ):
        parity.validate_run_spec(spec)


def test_run_spec_keyset_and_duplicate_json_keys_are_strict(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    spec = json.loads(fixture["run_spec"].read_text())
    spec["unexpected"] = True
    with pytest.raises(parity.ParityStopped, match="keyset"):
        parity.validate_run_spec(spec)
    with pytest.raises(parity.ParityStopped, match="duplicate JSON key"):
        parity.parse_json(b'{"a":1,"a":2}\n', "duplicate")


@pytest.mark.parametrize(
    "target",
    ["sandbox", "python", "sandbox_hash", "audit_root"],
)
def test_parent_refuses_unlocked_executables_and_audit_root(
    tmp_path: Path,
    target: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    sentinels = fixture["sentinels"]
    fake_executable = tmp_path / "fake-executable"
    fake_executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    fake_executable.chmod(0o755)
    arguments: dict[str, Any] = {}
    audit_root = fixture["audit_root"]
    if target == "sandbox":
        arguments["sandbox_executable"] = fake_executable
        message = "sandbox executable differs"
    elif target == "python":
        arguments["python_executable"] = fake_executable
        message = "python executable differs"
    elif target == "sandbox_hash":
        spec = json.loads(fixture["run_spec"].read_text())
        spec["sandbox_executable_sha256"] = "0" * 64
        _write_json(fixture["run_spec"], spec)
        message = "anchored input hash mismatch"
    else:
        audit_root = tmp_path / "other-audit-root"
        message = "audit root differs"
    with pytest.raises(parity.ParityStopped, match=message):
        parity.audit_and_publish(
            fixture["run_spec"],
            fixture["profile"],
            audit_root,
            forbidden_oracle=str(sentinels["oracle"]),
            forbidden_oracle_audit=str(sentinels["oracle_audit"]),
            forbidden_historical=str(sentinels["historical"]),
            forbidden_model=str(sentinels["model"]),
            forbidden_store=str(sentinels["store"]),
            write_sentinel=str(sentinels["write"]),
            **arguments,
        )


def test_anchored_open_rejects_symlink_final_component(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"safe")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(parity.ParityStopped, match="anchored open failed"):
        parity._open_anchored(link, _sha(target), 1 << 30)


def test_same_fd_snapshot_detects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "mutable"
    path.write_bytes(b"before")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        before = parity._snapshot_fd(descriptor, 1 << 30)
        path.write_bytes(b"after!")
        after = parity._snapshot_fd(descriptor, 1 << 30)
        assert before != after
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("recovery_state", ["final", "pending"])
def test_recovery_resnapshots_the_same_anchored_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_state: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    final, _ = _publish_recovery_fixture(fixture)
    pending = final.parent / f".pending-{final.name}"
    if recovery_state == "pending":
        os.rename(final, pending)
    original_validate = parity.validate_published_audit
    mutated = False

    def validate_then_mutate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal mutated
        report = original_validate(*args, **kwargs)
        if not mutated:
            descriptor = os.open(fixture["status"], os.O_WRONLY)
            try:
                os.pwrite(descriptor, b"X", 0)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            mutated = True
        return report

    monkeypatch.setattr(
        parity,
        "validate_published_audit",
        validate_then_mutate,
    )
    sentinels = fixture["sentinels"]
    with pytest.raises(
        parity.ParityStopped,
        match="anchored input changed",
    ):
        parity.audit_and_publish(
            fixture["run_spec"],
            fixture["profile"],
            fixture["audit_root"],
            forbidden_oracle=str(sentinels["oracle"]),
            forbidden_oracle_audit=str(sentinels["oracle_audit"]),
            forbidden_historical=str(sentinels["historical"]),
            forbidden_model=str(sentinels["model"]),
            forbidden_store=str(sentinels["store"]),
            write_sentinel=str(sentinels["write"]),
        )
    if recovery_state == "pending":
        assert pending.is_dir()
        assert not final.exists()


def test_recovery_rederives_checks_from_observed_commitments(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    final, _ = _publish_recovery_fixture(fixture)
    parity_path = final / "parity.json"
    manifest_path = final / "manifest.json"
    os.chmod(final, 0o700)
    os.chmod(parity_path, 0o600)
    os.chmod(manifest_path, 0o600)
    report = json.loads(parity_path.read_text())
    report["candidate_count"] += 1
    assert all(report["checks"].values())
    _write_json(parity_path, report)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = sorted(
        [
            _snapshot_record(parity_path),
            _snapshot_record(final / "provenance.json"),
        ],
        key=lambda row: row["path"].encode(),
    )
    _write_json(manifest_path, manifest)
    os.chmod(parity_path, 0o444)
    os.chmod(manifest_path, 0o444)
    os.chmod(final, 0o555)
    sentinels = fixture["sentinels"]
    with pytest.raises(
        parity.ParityStopped,
        match="checks are not derived",
    ):
        parity.audit_and_publish(
            fixture["run_spec"],
            fixture["profile"],
            fixture["audit_root"],
            forbidden_oracle=str(sentinels["oracle"]),
            forbidden_oracle_audit=str(sentinels["oracle_audit"]),
            forbidden_historical=str(sentinels["historical"]),
            forbidden_model=str(sentinels["model"]),
            forbidden_store=str(sentinels["store"]),
            write_sentinel=str(sentinels["write"]),
        )


def test_exclusive_promotion_never_overwrites_concurrent_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "payload").write_text("sealed", encoding="utf-8")
    original = parity._renameat_exclusive

    def collide_then_rename(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
    ) -> None:
        os.mkdir(destination_name, dir_fd=destination_parent_fd)
        destination_fd = os.open(
            destination_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=destination_parent_fd,
        )
        try:
            marker = os.open(
                "attacker",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=destination_fd,
            )
            os.close(marker)
        finally:
            os.close(destination_fd)
        original(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )

    monkeypatch.setattr(parity, "_renameat_exclusive", collide_then_rename)
    with pytest.raises(
        parity.ParityStopped,
        match="immutable publication destination exists",
    ):
        parity._promote(source, destination)
    assert source.is_dir()
    assert (destination / "attacker").is_file()


def test_pending_recovery_promotes_only_the_validated_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    final, _ = _publish_recovery_fixture(fixture)
    pending = final.parent / f".pending-{final.name}"
    os.rename(final, pending)
    displaced = final.parent / "validated-but-displaced"
    original_validate = parity.validate_published_audit
    substituted = False

    def validate_then_substitute(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal substituted
        report = original_validate(*args, **kwargs)
        if not substituted:
            os.rename(pending, displaced)
            pending.mkdir()
            (pending / "attacker").write_text("do not publish")
            substituted = True
        return report

    monkeypatch.setattr(
        parity,
        "validate_published_audit",
        validate_then_substitute,
    )
    sentinels = fixture["sentinels"]
    with pytest.raises(
        parity.ParityStopped,
        match="validated publication source was substituted",
    ):
        parity.audit_and_publish(
            fixture["run_spec"],
            fixture["profile"],
            fixture["audit_root"],
            forbidden_oracle=str(sentinels["oracle"]),
            forbidden_oracle_audit=str(sentinels["oracle_audit"]),
            forbidden_historical=str(sentinels["historical"]),
            forbidden_model=str(sentinels["model"]),
            forbidden_store=str(sentinels["store"]),
            write_sentinel=str(sentinels["write"]),
        )
    assert (pending / "attacker").is_file()
    assert not final.exists()


def test_profile_has_closed_security_boundary() -> None:
    profile = (
        Path(parity.__file__).resolve().parents[1]
        / parity.PROFILE_RELATIVE_PATH
    ).read_text()
    for marker in (
        "(deny default)",
        "(deny network*)",
        "(deny process-fork)",
        'param "FORBIDDEN_ORACLE"',
        'param "FORBIDDEN_ORACLE_AUDIT"',
        'param "FORBIDDEN_HISTORICAL"',
        'param "FORBIDDEN_MODEL"',
        'param "FORBIDDEN_STORE"',
        'param "WRITE_SENTINEL"',
    ):
        assert marker in profile
    assert '(subpath (param "RUN_ROOT"))' not in profile
    assert profile.count('(subpath (param "RUN_OUTPUT"))') == 3
    assert profile.count('(subpath (param "RUN_TMP"))') == 3
    assert "candidates_sparse_top100.parquet" not in Path(
        parity.__file__
    ).read_text()


@pytest.mark.skipif(
    platform.system() != "Darwin"
    or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="real sandbox-exec is a macOS-only security integration",
)
def test_real_sandbox_publication_and_pending_recovery(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    sentinels = fixture["sentinels"]
    final, report = parity.audit_and_publish(
        fixture["run_spec"],
        fixture["profile"],
        fixture["audit_root"],
        forbidden_oracle=str(sentinels["oracle"]),
        forbidden_oracle_audit=str(sentinels["oracle_audit"]),
        forbidden_historical=str(sentinels["historical"]),
        forbidden_model=str(sentinels["model"]),
        forbidden_store=str(sentinels["store"]),
        write_sentinel=str(sentinels["write"]),
    )
    assert report["verdict"] == parity.GO
    assert stat.S_IMODE(final.stat().st_mode) == 0o555
    assert {path.name for path in final.iterdir()} == {
        "parity.json",
        "provenance.json",
        "manifest.json",
    }
    assert not sentinels["write"].exists()
    pending = final.parent / f".pending-{final.name}"
    os.rename(final, pending)
    recovered, recovered_report = parity.audit_and_publish(
        fixture["run_spec"],
        fixture["profile"],
        fixture["audit_root"],
        forbidden_oracle=str(sentinels["oracle"]),
        forbidden_oracle_audit=str(sentinels["oracle_audit"]),
        forbidden_historical=str(sentinels["historical"]),
        forbidden_model=str(sentinels["model"]),
        forbidden_store=str(sentinels["store"]),
        write_sentinel=str(sentinels["write"]),
    )
    assert recovered == final
    assert recovered_report == report
    assert final.is_dir()
