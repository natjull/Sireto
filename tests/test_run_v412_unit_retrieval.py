from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xgb_matcher import v412_unit_retrieval as worker_core

SPEC = importlib.util.spec_from_file_location(
    "run_v412_unit_retrieval",
    ROOT / "scripts/run_v412_unit_retrieval.py",
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def _plan() -> dict:
    return json.loads(
        (ROOT / "config/v4_12_unit_retrieval_engine_plan.json").read_text()
    )


def _write_worker_outputs(root: Path, build_id: str = "b" * 64) -> list[str]:
    query_ids = ["q1", "q2"]
    status = pa.Table.from_pylist(
        [
            {"query_id": "q1", "candidate_count": 2},
            {"query_id": "q2", "candidate_count": 0},
        ],
        schema=subject.STATUS_SCHEMA,
    )
    candidates = pa.Table.from_pylist(
        [
            {
                "query_id": "q1",
                "candidate_rank": 1,
                "candidate_siret": "11111111100011",
            },
            {
                "query_id": "q1",
                "candidate_rank": 2,
                "candidate_siret": "22222222200022",
            },
        ],
        schema=subject.CANDIDATE_SCHEMA,
    )
    pq.write_table(status, root / "query_status.parquet")
    pq.write_table(candidates, root / "candidates_top100.parquet")
    candidate_payload = (
        b"q1\0" + b"11111111100011\0" + b"1\n"
        + b"q1\0" + b"22222222200022\0" + b"2\n"
    )
    status_payload = b"q1\0" + b"2\n" + b"q2\0" + b"0\n"
    integrity = {
        "schema_version": subject.WORKER_INTEGRITY_SCHEMA,
        "worker_build_id": build_id,
        "query_count": 2,
        "candidate_count": 2,
        "minimum_pool_size": 0,
        "maximum_pool_size": 2,
        "under_ceiling_query_count": 2,
        "empty_query_count": 1,
        "lookup_missing_count": 0,
        "candidate_payload_bytes": len(candidate_payload),
        "candidate_payload_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "status_payload_bytes": len(status_payload),
        "status_payload_sha256": hashlib.sha256(status_payload).hexdigest(),
        "sandbox_checks": {key: True for key in subject.SANDBOX_CHECK_KEYS},
        "peak_rss_bytes": 1,
        "durations_ns": {key: 0 for key in subject.DURATION_KEYS},
        "declarations": dict(subject.DECLARATIONS),
    }
    (root / "integrity.json").write_bytes(subject.canonical_json(integrity))
    return query_ids


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parity_bridge_fixture(tmp_path: Path) -> dict:
    build_id = "b" * 64
    safe_build_id = "safe-synthetic"
    runtime = tmp_path / "runtime" / build_id
    runtime.mkdir(parents=True)
    os.chmod(runtime, 0o700)
    query_ids = _write_worker_outputs(runtime, build_id)
    integrity = json.loads((runtime / "integrity.json").read_text())

    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    safe_queries = safe_root / "queries_dev.parquet"
    safe_table = pa.Table.from_pylist(
        [
            {
                "query_id": query_id,
                "crm_name": f"Name {query_id}",
                "crm_address": "",
                "crm_postcode": "75001",
                "crm_city": "PARIS",
                "crm_insee": "75056",
            }
            for query_id in query_ids
        ],
        schema=pa.schema(
            [
                pa.field(column, pa.string(), nullable=False)
                for column in subject.QUERY_COLUMNS
            ]
        ),
    )
    pq.write_table(safe_table, safe_queries)
    safe_manifest = safe_root / "runtime_manifest.json"
    safe_manifest.write_bytes(
        subject.canonical_json({"build_id": safe_build_id})
    )

    plan = _plan()
    plan["runtime"] = subject._runtime()
    plan["max_rss_bytes"] = 8 * 1024**3
    query_payload = b"".join(
        query_id.encode("utf-8") + b"\n" for query_id in query_ids
    )
    plan["safe_input"].update(
        {
            "build_id": safe_build_id,
            "root": str(safe_root),
            "runtime_manifest_sha256": _sha(safe_manifest),
            "queries_dev_sha256": _sha(safe_queries),
            "query_count": len(query_ids),
            "query_id_payload_sha256": hashlib.sha256(
                query_payload
            ).hexdigest(),
        }
    )
    for key in subject.PARITY_EXPECTED_KEYS:
        plan["historical_parity"][key] = integrity[key]
    plan["prerequisite"]["build_id"] = "strict-synthetic"
    plan["outputs"] = {
        "runtime_root": str(runtime.parent),
        "worker_audit_root": str(tmp_path / "worker-proof"),
        "parity_audit_root": str(tmp_path / "parity-proof"),
        "temp_root": str(tmp_path / "tmp"),
    }
    forbidden_files = {
        "oracle": tmp_path / "oracles" / "manifest.json",
        "oracle_audit": tmp_path / "audits" / "manifest.json",
        "historical": tmp_path / "datasets" / "historical.parquet",
        "model": tmp_path / "model" / "routing.pkl",
    }
    for path in forbidden_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"forbidden")
    plan["parity_controller"]["forbidden_files"] = [
        str(forbidden_files[key])
        for key in ("historical", "oracle", "oracle_audit", "model")
    ]

    manifest = {
        "schema_version": subject.WORKER_MANIFEST_SCHEMA,
        "worker_build_id": build_id,
        "safe_input_build_id": safe_build_id,
        "strict_stores_build_id": plan["prerequisite"]["build_id"],
        "files": subject._runtime_file_records(runtime, 8 * 1024**3),
        "runtime": plan["runtime"],
        "declarations": dict(subject.DECLARATIONS),
        "verdict": subject.SEALED,
    }
    (runtime / "manifest.json").write_bytes(subject.canonical_json(manifest))

    source = ROOT / subject.PARITY_SOURCE_PATH
    profile = ROOT / subject.PARITY_PROFILE_PATH
    sandbox_executable = Path("/usr/bin/sandbox-exec")
    launcher = Path(os.path.realpath(sys.executable))
    python_executable = (
        launcher.parent.parent
        / "Resources"
        / "Python.app"
        / "Contents"
        / "MacOS"
        / "Python"
    )
    if not python_executable.is_file():
        python_executable = launcher
    python_library = launcher.parent.parent / "Python"
    if not python_library.is_file():
        python_library = python_executable
    lock = {
        "git_commit": "a" * 40,
        "source_hashes": {
            str(subject.PARITY_SOURCE_PATH): _sha(source),
            str(subject.PARITY_PROFILE_PATH): _sha(profile),
        },
        "input_paths": {
            "safe_queries_dev": str(safe_queries),
            "safe_runtime_manifest": str(safe_manifest),
        },
        "input_hashes": {
            "safe_queries_dev": _sha(safe_queries),
            "safe_runtime_manifest": _sha(safe_manifest),
        },
        "sandbox": {
            "executable": str(sandbox_executable),
            "executable_sha256": (
                _sha(sandbox_executable)
                if sandbox_executable.is_file()
                else "0" * 64
            ),
            "python_framework_app": str(python_executable),
            "python_framework_app_sha256": _sha(python_executable),
            "python_framework_library": str(python_library),
            "python_framework_library_sha256": _sha(python_library),
        },
    }
    store = tmp_path / "indexes" / "candidate_details.duckdb"
    store.parent.mkdir()
    store.write_bytes(b"forbidden store")
    gate_spec = {
        "allowed_read_files": [
            {"role": "lookup_database", "absolute_path": str(store)}
        ]
    }
    return {
        "plan": plan,
        "lock": lock,
        "runtime": runtime,
        "source": source,
        "gate_spec": gate_spec,
    }


def test_plan_is_strict_and_worker_projection_excludes_history() -> None:
    plan = _plan()
    subject.validate_plan(plan)
    projection = subject._worker_policy(plan)
    payload = subject.canonical_json(projection)
    assert b"historical_parity" not in payload
    assert plan["historical_parity"]["candidate_payload_sha256"].encode() not in payload
    broken = copy.deepcopy(plan)
    broken["unexpected"] = True
    with pytest.raises(subject.RetrievalRunStopped, match="plan schema"):
        subject.validate_plan(broken)


def test_json_rejects_duplicate_keys_and_nan() -> None:
    with pytest.raises(subject.RetrievalRunStopped, match="duplicate"):
        subject._parse_json(b'{"a":1,"a":2}', "fixture")
    with pytest.raises(subject.RetrievalRunStopped, match="canonical"):
        subject.canonical_json({"value": float("nan")})


def test_openat_rejects_symlink_ancestor_and_survives_path_substitution(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    source = real / "value.bin"
    source.write_bytes(b"sealed")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(subject.RetrievalRunStopped, match="anchored open failed"):
        subject._read(alias / "value.bin", 8 * 1024**3)

    expected = subject._snapshot(source, 8 * 1024**3)
    descriptor = subject._open_anchored(source, expected, 8 * 1024**3)
    moved = tmp_path / "moved"
    real.rename(moved)
    real.mkdir()
    (real / "value.bin").write_bytes(b"attacker")
    try:
        assert os.pread(descriptor, 6, 0) == b"sealed"
        assert subject._snapshot_fd(descriptor, 8 * 1024**3) == expected
    finally:
        os.close(descriptor)


def test_profile_is_closed_and_has_explicit_denies(tmp_path: Path) -> None:
    template = (ROOT / "config/v4_12_unit_retrieval.sb").read_text()
    allowed = tmp_path / "allowed"
    allowed.write_bytes(b"allowed")
    forbidden = tmp_path / "oracles"
    forbidden.mkdir()
    profile = subject.render_profile(
        template,
        allowed_files=[allowed],
        forbidden_roots=[forbidden],
        system_roots=[Path("/System"), Path("/usr"), Path("/opt/homebrew")],
        devices=[Path("/dev/null"), Path("/dev/urandom"), Path("/dev/fd")],
        metadata_extra=[tmp_path],
    )
    assert "@@" not in profile
    assert str(allowed) in profile
    assert f"(subpath {json.dumps(str(forbidden))})" in profile
    assert "(deny network*)" in profile
    assert "(deny process-fork)" in profile
    assert '(subpath (param "PRIVATE_PACKAGE_ROOT"))' in profile


def test_profile_refuses_a_missing_marker() -> None:
    with pytest.raises(subject.RetrievalRunStopped, match="marker missing"):
        subject.render_profile(
            "(version 1)",
            allowed_files=[],
            forbidden_roots=[],
            system_roots=[],
            devices=[],
            metadata_extra=[],
        )


def test_worker_outputs_are_validated_end_to_end(tmp_path: Path) -> None:
    query_ids = _write_worker_outputs(tmp_path)
    integrity = subject._validate_outputs(
        tmp_path, "b" * 64, query_ids, 8 * 1024**3
    )
    assert integrity["candidate_count"] == 2
    records = subject._runtime_file_records(tmp_path, 8 * 1024**3)
    assert set(records) == {
        "query_status.parquet",
        "candidates_top100.parquet",
        "integrity.json",
    }
    assert records["query_status.parquet"]["metadata"] is None
    assert records["query_status.parquet"]["row_count"] == 2


@pytest.mark.parametrize("mutation", ["rank", "duplicate", "count", "metadata"])
def test_worker_outputs_reject_invalid_candidate_scenes(
    tmp_path: Path, mutation: str
) -> None:
    query_ids = _write_worker_outputs(tmp_path)
    if mutation == "rank":
        table = pq.read_table(tmp_path / "candidates_top100.parquet")
        table = table.set_column(
            1, "candidate_rank", pa.array([1, 3], type=pa.uint8())
        )
        pq.write_table(table, tmp_path / "candidates_top100.parquet")
    elif mutation == "duplicate":
        table = pq.read_table(tmp_path / "candidates_top100.parquet")
        table = table.set_column(
            2,
            "candidate_siret",
            pa.array(["11111111100011", "11111111100011"]),
        )
        pq.write_table(table, tmp_path / "candidates_top100.parquet")
    elif mutation == "count":
        table = pq.read_table(tmp_path / "query_status.parquet")
        table = table.set_column(
            1, "candidate_count", pa.array([1, 0], type=pa.uint8())
        )
        pq.write_table(table, tmp_path / "query_status.parquet")
    else:
        table = pq.read_table(tmp_path / "query_status.parquet")
        table = table.replace_schema_metadata({b"leak": b"truth"})
        pq.write_table(table, tmp_path / "query_status.parquet")
    with pytest.raises(subject.RetrievalRunStopped):
        subject._validate_outputs(
            tmp_path, "b" * 64, query_ids, 8 * 1024**3
        )


def test_worker_output_rejects_extra_file(tmp_path: Path) -> None:
    query_ids = _write_worker_outputs(tmp_path)
    (tmp_path / "debug.log").write_text("forbidden")
    with pytest.raises(subject.RetrievalRunStopped, match="file-set"):
        subject._validate_outputs(
            tmp_path, "b" * 64, query_ids, 8 * 1024**3
        )


def test_worker_failure_removes_only_registered_private_staging(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / ".run-worker-failure"
    preexisting_orphan = tmp_path / ".run-preexisting-orphan"
    run_root.mkdir(mode=0o700)
    preexisting_orphan.mkdir(mode=0o700)
    (run_root / "partial").write_text("not published")
    (preexisting_orphan / "keep").write_text("untouched")
    subject._register_private(run_root)
    result = subprocess.CompletedProcess(
        ["worker"],
        1,
        stdout="",
        stderr="ModuleNotFoundError: xgb_matcher",
    )
    with pytest.raises(subject.RetrievalRunStopped, match="worker failed rc=1"):
        subject._require_worker_success(run_root, result)
    assert not run_root.exists()
    assert run_root not in subject._ACTIVE_PRIVATE_ROOTS
    assert (preexisting_orphan / "keep").read_text() == "untouched"


def test_routes_use_insee_then_postcode_and_match_payload() -> None:
    queries = pa.table(
        {
            "query_id": ["q1", "q2"],
            "crm_name": ["A", "B"],
            "crm_address": ["", ""],
            "crm_postcode": ["75001", "69001"],
            "crm_city": ["PARIS", "LYON"],
            "crm_insee": ["75056", ""],
        }
    )
    run_spec = {
        "partition_records": [
            {
                "relative_path": "insee/insee=75056/a.parquet",
                "size_bytes": 1,
                "sha256": "a" * 64,
            },
            {
                "relative_path": "cp/postcode=69001/a.parquet",
                "size_bytes": 1,
                "sha256": "b" * 64,
            },
        ],
        "cache_records": [
            {"partition_key": "75056_"},
            {"partition_key": "_69001"},
        ],
    }
    payload = b"q1\0" + b"75056_\n" + b"q2\0" + b"_69001\n"
    expected = {
        "query_count": 2,
        "insee_query_count": 1,
        "cp_query_count": 1,
        "distinct_key_count": 2,
        "missing_key_count": 0,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    routes = subject._derive_routes(queries, run_spec, expected)
    assert [row["partition_key"] for row in routes] == ["75056_", "_69001"]


def test_publication_is_atomic_and_immutable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination_root = tmp_path / "published"
    source.mkdir()
    destination_root.mkdir()
    (source / "value.json").write_text("{}")
    destination = destination_root / "build"
    subject._promote(source, destination)
    assert not source.exists()
    assert stat.S_IMODE(os.lstat(destination).st_mode) == 0o555
    assert stat.S_IMODE(os.lstat(destination / "value.json").st_mode) == 0o444
    with pytest.raises(subject.RetrievalRunStopped, match="destination exists"):
        another = tmp_path / "another"
        another.mkdir()
        subject._promote(another, destination)


def test_worker_identity_uses_only_projected_policy() -> None:
    plan = _plan()
    source_hashes = {
        "scripts/run_v412_unit_retrieval.py": "1" * 64,
        **{key: "2" * 64 for key in plan["worker_sources"]},
    }
    lock = {
        "worker_lock_projection_sha256": "3" * 64,
        "source_hashes": source_hashes,
    }
    first, policy = subject._worker_identity(
        plan, lock, plan["safe_input"]["runtime_manifest_sha256"]
    )
    changed = copy.deepcopy(plan)
    changed["historical_parity"]["candidate_payload_sha256"] = "f" * 64
    second, changed_policy = subject._worker_identity(
        changed, lock, changed["safe_input"]["runtime_manifest_sha256"]
    )
    assert first == second
    assert policy == changed_policy


def test_worker_run_spec_matches_child_contract_exactly() -> None:
    plan = _plan()
    lock = {
        "worker_lock_projection_sha256": "c" * 64,
        "source_hashes": {
            "scripts/run_v412_unit_retrieval.py": "d" * 64,
            **{path: "e" * 64 for path in plan["worker_sources"]},
        },
    }
    spec = subject._worker_run_spec(
        plan,
        lock,
        {"schema_version": "gate-a-fixture"},
        policy_sha="f" * 64,
    )
    assert set(spec) == {
        "schema_version",
        "safe_input_build_id",
        "safe_runtime_manifest_sha256",
        "safe_queries_dev_sha256",
        "query_count",
        "query_id_payload_sha256",
        "routing_payload_sha256",
        "worker_policy_sha256",
        "worker_lock_projection_sha256",
        "parent_runner_sha256",
        "worker_source_hashes",
        "strict_stores_build_id",
        "strict_stores_manifest_sha256",
        "retrieval",
        "tfidf_cache",
        "runtime",
        "max_rss_bytes",
        "gate_a_run_spec",
        "declarations",
    }
    assert spec["schema_version"] == (
        "sireto-v4.12-unit-retrieval-worker-run-spec-1"
    )
    assert set(spec) == worker_core._WORKER_RUN_SPEC_KEYS
    assert spec["retrieval"] == worker_core._RETRIEVAL_POLICY
    assert spec["tfidf_cache"] == worker_core._TFIDF_POLICY
    assert spec["declarations"] == worker_core._WORKER_DECLARATIONS


def test_parity_run_spec_is_built_only_from_sealed_commitments(
    tmp_path: Path,
) -> None:
    fixture = _parity_bridge_fixture(tmp_path)
    spec = subject._build_parity_run_spec(
        repo=ROOT,
        plan=fixture["plan"],
        lock=fixture["lock"],
        runtime=fixture["runtime"],
        lock_sha256="c" * 64,
        limit=8 * 1024**3,
    )
    assert spec["schema_version"] == subject.PARITY_RUN_SPEC_SCHEMA
    assert spec["worker_build_id"] == fixture["runtime"].name
    assert set(spec["expected"]) == subject.PARITY_EXPECTED_KEYS
    assert set(spec["worker_file_paths"]) == subject.PARITY_WORKER_FILES
    assert set(spec["worker_file_hashes"]) == subject.PARITY_WORKER_FILES
    assert spec["expected"]["candidate_payload_sha256"] == json.loads(
        (fixture["runtime"] / "integrity.json").read_text()
    )["candidate_payload_sha256"]
    serialized = subject.canonical_json(spec).lower()
    assert b"/datasets/" not in serialized
    assert b"/oracles/" not in serialized
    assert b"/audits/" not in serialized


def test_parity_bridge_rejects_mutated_worker_manifest(tmp_path: Path) -> None:
    fixture = _parity_bridge_fixture(tmp_path)
    manifest_path = fixture["runtime"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["worker_build_id"] = "attacker"
    manifest_path.write_bytes(subject.canonical_json(manifest))
    with pytest.raises(
        subject.RetrievalRunStopped,
        match="runtime manifest values",
    ):
        subject._build_parity_run_spec(
            repo=ROOT,
            plan=fixture["plan"],
            lock=fixture["lock"],
            runtime=fixture["runtime"],
            lock_sha256="c" * 64,
            limit=8 * 1024**3,
        )


def _prepared_parity_invocation(tmp_path: Path) -> dict:
    fixture = _parity_bridge_fixture(tmp_path)
    staging = (
        Path(fixture["plan"]["outputs"]["temp_root"])
        / ".run-parity-parent-fixture"
    )
    staging.mkdir(parents=True)
    os.chmod(staging, 0o700)
    private_python, framework_root = subject._copy_private_python(
        staging,
        fixture["lock"]["sandbox"],
        8 * 1024**3,
    )
    private_library = (
        staging / "runtime/Python.framework/Versions/3.14/Python"
    )
    spec = subject._build_parity_run_spec(
        repo=ROOT,
        plan=fixture["plan"],
        lock=fixture["lock"],
        runtime=fixture["runtime"],
        lock_sha256="c" * 64,
        limit=8 * 1024**3,
    )
    spec["python_executable_path"] = str(private_python)
    run_spec = staging / "parity_run_spec.json"
    run_spec.write_bytes(subject.canonical_json(spec))
    os.chmod(run_spec, 0o444)
    os.chmod(staging, 0o555)
    sentinels = subject._parity_sentinels(
        fixture["plan"],
        fixture["gate_spec"],
        fixture["runtime"].name,
    )
    fixture.update(
        {
            "spec": spec,
            "staging": staging,
            "run_spec": run_spec,
            "sentinels": sentinels,
            "framework_root": framework_root,
            "private_library": private_library,
            "source_snapshot": subject._snapshot(
                fixture["source"], 8 * 1024**3
            ),
        }
    )
    return fixture


@pytest.mark.parametrize(
    ("returncode", "verdict", "message"),
    [
        (3, "STOP_V412_UNIT_RETRIEVAL", "controller failed"),
        (0, "STOP_V412_UNIT_RETRIEVAL", "non-GO"),
        (0, subject.PARITY_GO, "non-GO"),
    ],
)
def test_parity_bridge_rejects_controller_stop_or_bad_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    verdict: str,
    message: str,
) -> None:
    fixture = _prepared_parity_invocation(tmp_path)
    parity_id = "d" * 64 if message != "non-GO" or verdict != subject.PARITY_GO else "bad"
    response = {
        "verdict": verdict,
        "parity_build_id": parity_id,
        "audit": str(tmp_path / "untrusted"),
    }
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            returncode,
            stdout=json.dumps(response),
            stderr="",
        ),
    )
    with pytest.raises(subject.RetrievalRunStopped, match=message):
        subject._invoke_parity_controller(
            repo=ROOT,
            plan=fixture["plan"],
            lock=fixture["lock"],
            spec=fixture["spec"],
            run_spec_path=fixture["run_spec"],
            sentinels=fixture["sentinels"],
            source_snapshot=fixture["source_snapshot"],
            python_framework_root=fixture["framework_root"],
            python_library_path=fixture["private_library"],
            limit=8 * 1024**3,
        )


def test_parity_controller_source_substitution_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_parity_invocation(tmp_path / "fixture")
    fake_repo = tmp_path / "repo"
    fake_source = fake_repo / subject.PARITY_SOURCE_PATH
    fake_source.parent.mkdir(parents=True)
    fake_source.write_bytes(fixture["source"].read_bytes())
    fake_profile = fake_repo / subject.PARITY_PROFILE_PATH
    fake_profile.parent.mkdir(parents=True)
    fake_profile.write_bytes((ROOT / subject.PARITY_PROFILE_PATH).read_bytes())
    source_snapshot = subject._snapshot(fake_source, 8 * 1024**3)
    fixture["lock"]["source_hashes"][
        str(subject.PARITY_SOURCE_PATH)
    ] = source_snapshot["sha256"]
    observed: dict[str, bytes] = {}

    def substitute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        source_fd = kwargs["pass_fds"][0]  # type: ignore[index]
        displaced = fake_source.with_name("controller.original")
        fake_source.rename(displaced)
        fake_source.write_bytes(b"raise SystemExit('attacker')\n")
        observed["anchored"] = os.pread(source_fd, 32, 0)
        observed["path"] = fake_source.read_bytes()
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="STOP_V412_UNIT_RETRIEVAL: source hash mismatch",
        )

    monkeypatch.setattr(subject.subprocess, "run", substitute)
    with pytest.raises(subject.RetrievalRunStopped, match="controller failed"):
        subject._invoke_parity_controller(
            repo=fake_repo,
            plan=fixture["plan"],
            lock=fixture["lock"],
            spec=fixture["spec"],
            run_spec_path=fixture["run_spec"],
            sentinels=fixture["sentinels"],
            source_snapshot=source_snapshot,
            python_framework_root=fixture["framework_root"],
            python_library_path=fixture["private_library"],
            limit=8 * 1024**3,
        )
    assert observed["anchored"].startswith(b"#!/usr/bin/env python3")
    assert observed["path"].startswith(b"raise SystemExit")


@pytest.mark.parametrize("mutation", ["expected", "path"])
def test_parity_bridge_rejects_noncanonical_or_mutated_run_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _prepared_parity_invocation(tmp_path)
    mutated = copy.deepcopy(fixture["spec"])
    if mutation == "expected":
        mutated["expected"]["candidate_count"] += 1
    else:
        mutated["worker_manifest_path"] = str(tmp_path / "substituted.json")
    os.chmod(fixture["run_spec"], 0o600)
    fixture["run_spec"].write_bytes(subject.canonical_json(mutated))
    os.chmod(fixture["run_spec"], 0o444)
    launched = False

    def forbidden_launch(*args: object, **kwargs: object) -> None:
        nonlocal launched
        launched = True
        raise AssertionError("controller must not launch")

    monkeypatch.setattr(subject.subprocess, "run", forbidden_launch)
    with pytest.raises(
        subject.RetrievalRunStopped,
        match="differs from canonical parent spec",
    ):
        subject._invoke_parity_controller(
            repo=ROOT,
            plan=fixture["plan"],
            lock=fixture["lock"],
            spec=fixture["spec"],
            run_spec_path=fixture["run_spec"],
            sentinels=fixture["sentinels"],
            source_snapshot=fixture["source_snapshot"],
            python_framework_root=fixture["framework_root"],
            python_library_path=fixture["private_library"],
            limit=8 * 1024**3,
        )
    assert launched is False


@pytest.mark.skipif(
    platform.system() != "Darwin"
    or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="production parity bridge requires native macOS Seatbelt",
)
def test_parent_rejects_publication_substituted_after_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_parity_invocation(tmp_path)
    original_run = subprocess.run
    displaced: Path | None = None
    replacement: Path | None = None

    def launch_then_substitute(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess:
        nonlocal displaced, replacement
        result = original_run(command, **kwargs)
        response = json.loads(result.stdout)
        published = Path(response["audit"])
        displaced = published.with_name(published.name + ".validated")
        published.rename(displaced)
        published.mkdir(mode=0o700)
        (published / "attacker.json").write_text("{}")
        replacement = published
        return result

    monkeypatch.setattr(subject.subprocess, "run", launch_then_substitute)
    try:
        with pytest.raises(
            subject.RetrievalRunStopped,
            match="parent parity publication validation failed",
        ):
            subject._invoke_parity_controller(
                repo=ROOT,
                plan=fixture["plan"],
                lock=fixture["lock"],
                spec=fixture["spec"],
                run_spec_path=fixture["run_spec"],
                sentinels=fixture["sentinels"],
                source_snapshot=fixture["source_snapshot"],
                python_framework_root=fixture["framework_root"],
                python_library_path=fixture["private_library"],
                limit=8 * 1024**3,
            )
    finally:
        if replacement is not None and replacement.is_dir():
            (replacement / "attacker.json").unlink()
            replacement.rmdir()
        if displaced is not None and displaced.is_dir():
            displaced.rename(displaced.with_name(displaced.name.removesuffix(".validated")))


def test_parent_rejects_minimal_fake_go_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_parity_invocation(tmp_path)
    run_spec_sha = _sha(fixture["run_spec"])
    parity_id = subject._parity_build_id(fixture["spec"], run_spec_sha)
    parity_root = (
        Path(fixture["plan"]["outputs"]["parity_audit_root"]) / parity_id
    )
    parity_root.mkdir(parents=True)
    for name in ("parity.json", "provenance.json", "manifest.json"):
        path = parity_root / name
        path.write_bytes(
            subject.canonical_json(
                {
                    "parity_build_id": parity_id,
                    "worker_build_id": fixture["runtime"].name,
                    "verdict": subject.PARITY_GO,
                }
            )
        )
        os.chmod(path, 0o444)
    os.chmod(parity_root, 0o555)
    response = {
        "verdict": subject.PARITY_GO,
        "parity_build_id": parity_id,
        "audit": str(parity_root),
    }
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(response),
            stderr="",
        ),
    )
    with pytest.raises(
        subject.RetrievalRunStopped,
        match="parent parity publication validation failed",
    ):
        subject._invoke_parity_controller(
            repo=ROOT,
            plan=fixture["plan"],
            lock=fixture["lock"],
            spec=fixture["spec"],
            run_spec_path=fixture["run_spec"],
            sentinels=fixture["sentinels"],
            source_snapshot=fixture["source_snapshot"],
            python_framework_root=fixture["framework_root"],
            python_library_path=fixture["private_library"],
            limit=8 * 1024**3,
        )


def test_private_python_isolated_from_locked_original_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_parity_invocation(tmp_path)
    original_copy = tmp_path / "locked-original-python"
    original_copy.write_bytes(
        Path(fixture["lock"]["sandbox"]["python_framework_app"]).read_bytes()
    )
    os.chmod(original_copy, 0o555)
    fixture["lock"]["sandbox"]["python_framework_app"] = str(original_copy)
    private_before = fixture["spec"]["python_executable_path"]
    observed: dict[str, str] = {}

    def substitute_original(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess:
        displaced = original_copy.with_suffix(".sealed")
        original_copy.rename(displaced)
        original_copy.write_bytes(b"attacker")
        observed["command"] = command[0]
        observed["private_sha"] = _sha(Path(command[0]))
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="STOP synthetic",
        )

    monkeypatch.setattr(subject.subprocess, "run", substitute_original)
    with pytest.raises(subject.RetrievalRunStopped, match="controller failed"):
        subject._invoke_parity_controller(
            repo=ROOT,
            plan=fixture["plan"],
            lock=fixture["lock"],
            spec=fixture["spec"],
            run_spec_path=fixture["run_spec"],
            sentinels=fixture["sentinels"],
            source_snapshot=fixture["source_snapshot"],
            python_framework_root=fixture["framework_root"],
            python_library_path=fixture["private_library"],
            limit=8 * 1024**3,
        )
    assert observed["command"] == private_before
    assert observed["command"] != str(original_copy)
    assert observed["private_sha"] == fixture["lock"]["sandbox"][
        "python_framework_app_sha256"
    ]


def test_private_python_boundary_blocks_launcher_replace(
    tmp_path: Path,
) -> None:
    fixture = _prepared_parity_invocation(tmp_path)
    launcher = Path(fixture["spec"]["python_executable_path"])
    library = fixture["private_library"]
    runtime_root = fixture["framework_root"]
    assert stat.S_IMODE(os.lstat(launcher).st_mode) == 0o555
    assert stat.S_IMODE(os.lstat(library).st_mode) == 0o444
    for path in [runtime_root, *(
        item for item in runtime_root.rglob("*") if item.is_dir()
    )]:
        assert not path.is_symlink()
        assert stat.S_IMODE(os.lstat(path).st_mode) == 0o555
    attacker = tmp_path / "attacker-python"
    attacker.write_bytes(b"attacker")
    os.chmod(attacker, 0o555)
    with pytest.raises(PermissionError):
        os.replace(attacker, launcher)
    assert _sha(launcher) == fixture["lock"]["sandbox"][
        "python_framework_app_sha256"
    ]


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_private_cleanup_rejects_staging_replaced_after_controller(
    tmp_path: Path,
    replacement: str,
) -> None:
    staging = tmp_path / ".run-parity-parent-fixture"
    staging.mkdir(mode=0o700)
    (staging / "secret").write_text("sealed")
    subject._register_private(staging)
    displaced = tmp_path / "original-staging"
    staging.rename(displaced)
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "must-survive").write_text("value")
    if replacement == "symlink":
        staging.symlink_to(protected, target_is_directory=True)
    else:
        staging.mkdir()
        (staging / "attacker").write_text("value")
    try:
        with pytest.raises(
            subject.RetrievalRunStopped,
            match="staging was substituted",
        ):
            subject._remove_private(staging)
        assert (protected / "must-survive").read_text() == "value"
        assert (displaced / "secret").read_text() == "sealed"
    finally:
        subject._ACTIVE_PRIVATE_ROOTS.pop(staging, None)


def test_private_cleanup_refuses_unregistered_run_prefix(tmp_path: Path) -> None:
    staging = tmp_path / ".run-attacker"
    staging.mkdir()
    with pytest.raises(subject.RetrievalRunStopped, match="refusing"):
        subject._remove_private(staging)
    assert staging.is_dir()


def test_run_end_to_end_calls_worker_spec_controller_and_resnapshots_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = (tmp_path / "plan.json").absolute()
    lock_path = (tmp_path / "lock.json").absolute()
    plan_path.write_bytes(b"{}\n")
    lock_path.write_bytes(b"{}\n")
    runtime = tmp_path / "runtime" / ("b" * 64)
    worker_audit = tmp_path / "worker-proof"
    runtime.mkdir(parents=True)
    worker_audit.mkdir()
    temp_root = tmp_path / "tmp"
    parity_root = tmp_path / "parity-proof" / ("d" * 64)
    plan = {
        "max_rss_bytes": 8 * 1024**3,
        "safe_input": {"runtime_manifest_sha256": "a" * 64},
        "outputs": {
            "temp_root": str(temp_root),
            "parity_audit_root": str(tmp_path / "parity-proof"),
        },
    }
    lock = {"sandbox": {}}
    source = (ROOT / subject.PARITY_SOURCE_PATH).absolute()
    source_snapshot = subject._snapshot(source, 8 * 1024**3)
    original_snapshot = subject._snapshot
    snapshot_counts = {"plan": 0, "lock": 0}
    events: list[str] = []

    def tracking_snapshot(path: Path, limit: int) -> dict:
        absolute = path.absolute()
        if absolute == plan_path:
            snapshot_counts["plan"] += 1
        if absolute == lock_path:
            snapshot_counts["lock"] += 1
        return original_snapshot(path, limit)

    def parse_fixture(payload: bytes, label: str) -> dict:
        return plan if label == "plan" else lock

    def fake_run(plan_arg: Path, lock_arg: Path) -> tuple[Path, Path]:
        assert plan_arg == plan_path
        assert lock_arg == lock_path
        events.append("worker")
        return runtime, worker_audit

    def fake_validate(*args: object, **kwargs: object) -> None:
        events.append("validate-worker")

    def fake_copy(
        staging: Path,
        sandbox: dict,
        limit: int,
    ) -> tuple[Path, Path]:
        events.append("copy-python")
        version = staging / "runtime/Python.framework/Versions/3.14"
        executable = (
            version
            / "Resources/Python.app/Contents/MacOS/Python"
        )
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"python")
        os.chmod(executable, 0o555)
        (version / "Python").write_bytes(b"library")
        return executable, staging / "runtime"

    def fake_build(**kwargs: object) -> dict:
        assert events[-1] == "copy-python"
        events.append("build-spec")
        return {"python_executable_path": ""}

    def fake_invoke(**kwargs: object) -> tuple[Path, dict]:
        assert events[-1] == "build-spec"
        spec = kwargs["spec"]
        assert Path(spec["python_executable_path"]).is_file()
        assert kwargs["run_spec_path"].is_file()
        events.append("controller")
        return parity_root, {
            "parity_build_id": parity_root.name,
            "worker_build_id": runtime.name,
            "verdict": subject.PARITY_GO,
        }

    monkeypatch.setattr(subject, "PLAN_PATH", plan_path)
    monkeypatch.setattr(subject, "LOCK_PATH", lock_path)
    monkeypatch.setattr(subject, "_snapshot", tracking_snapshot)
    monkeypatch.setattr(subject, "_parse_json", parse_fixture)
    monkeypatch.setattr(subject, "validate_plan", lambda value: None)
    monkeypatch.setattr(
        subject,
        "validate_lock",
        lambda *args, **kwargs: {source: source_snapshot},
    )
    monkeypatch.setattr(subject, "run", fake_run)
    monkeypatch.setattr(
        subject,
        "_worker_identity",
        lambda *args, **kwargs: (runtime.name, "policy"),
    )
    monkeypatch.setattr(subject, "_validate_published", fake_validate)
    monkeypatch.setattr(
        subject,
        "_verify_gate_a",
        lambda *args, **kwargs: ({"allowed_read_files": []}, {}),
    )
    monkeypatch.setattr(subject, "_copy_private_python", fake_copy)
    monkeypatch.setattr(subject, "_build_parity_run_spec", fake_build)
    monkeypatch.setattr(
        subject,
        "_parity_sentinels",
        lambda *args, **kwargs: {
            key: str(tmp_path / key)
            for key in (
                "oracle",
                "oracle_audit",
                "historical",
                "model",
                "store",
                "write",
            )
        },
    )
    monkeypatch.setattr(subject, "_invoke_parity_controller", fake_invoke)

    result = subject.run_end_to_end(plan_path, lock_path)
    assert result[0:3] == (runtime, worker_audit, parity_root)
    assert events == [
        "worker",
        "validate-worker",
        "copy-python",
        "build-spec",
        "controller",
        "validate-worker",
    ]
    assert snapshot_counts == {"plan": 3, "lock": 3}
    assert not any(
        path.name.startswith(".run-parity-parent-")
        for path in temp_root.iterdir()
    )


@pytest.mark.skipif(
    platform.system() != "Darwin"
    or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="production parity bridge requires native macOS Seatbelt",
)
def test_synthetic_worker_to_real_parity_controller(tmp_path: Path) -> None:
    fixture = _prepared_parity_invocation(tmp_path)
    parity_root, report = subject._invoke_parity_controller(
        repo=ROOT,
        plan=fixture["plan"],
        lock=fixture["lock"],
        spec=fixture["spec"],
        run_spec_path=fixture["run_spec"],
        sentinels=fixture["sentinels"],
        source_snapshot=fixture["source_snapshot"],
        python_framework_root=fixture["framework_root"],
        python_library_path=fixture["private_library"],
        limit=8 * 1024**3,
    )
    assert report["verdict"] == subject.PARITY_GO
    assert report["worker_build_id"] == fixture["runtime"].name
    assert parity_root == Path(
        fixture["plan"]["outputs"]["parity_audit_root"]
    ) / report["parity_build_id"]


def test_forbidden_sentinels_are_typed_and_unique() -> None:
    sentinels = subject._forbidden_sentinels(
        [
            "/root/datasets/historical.parquet",
            "/root/oracles/build/manifest.json",
            "/root/audits/oracle/manifest.json",
            "/repo/models/routing.pkl",
        ]
    )
    assert set(sentinels) == {"oracle", "oracle_audit", "historical", "model"}
    with pytest.raises(subject.RetrievalRunStopped, match="not unique"):
        subject._forbidden_sentinels(
            [
                "/root/datasets/one.parquet",
                "/root/oracles/one.json",
                "/root/oracles/two.json",
                "/root/audits/one.json",
                "/repo/models/routing.pkl",
            ]
        )


def test_recovery_removes_a_valid_pending_only_runtime(tmp_path: Path) -> None:
    plan = _plan()
    output_root = tmp_path / "runtime"
    audit_root = tmp_path / "audit"
    output_root.mkdir()
    audit_root.mkdir()
    build_id = "b" * 64
    pending = output_root / f".pending-{build_id}"
    pending.mkdir()
    _write_worker_outputs(pending, build_id)
    manifest = {
        "schema_version": subject.WORKER_MANIFEST_SCHEMA,
        "worker_build_id": build_id,
        "safe_input_build_id": plan["safe_input"]["build_id"],
        "strict_stores_build_id": plan["prerequisite"]["build_id"],
        "files": subject._runtime_file_records(pending, 8 * 1024**3),
        "runtime": plan["runtime"],
        "declarations": subject.DECLARATIONS,
        "verdict": subject.SEALED,
    }
    (pending / "manifest.json").write_bytes(subject.canonical_json(manifest))
    for path in pending.iterdir():
        os.chmod(path, 0o444)
    os.chmod(pending, 0o555)
    assert subject._recover(
        output_root,
        audit_root,
        build_id,
        plan,
        {},
        "a" * 64,
        "c" * 64,
        8 * 1024**3,
    ) is None
    assert not pending.exists()


def test_smoke_never_opens_real_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        opened.append(str(path))
        assert "queries_dev.parquet" not in str(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    subject.smoke()
    assert not any("queries_dev.parquet" in value for value in opened)


@pytest.mark.skipif(
    platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").exists(),
    reason="native Seatbelt integration is macOS-only",
)
def test_native_sandbox_runs_private_python_and_enforces_boundaries(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    output = run_root / "output"
    scratch = run_root / "tmp"
    forbidden = tmp_path / "forbidden"
    run_root.mkdir()
    output.mkdir()
    scratch.mkdir()
    forbidden.mkdir()
    allowed = tmp_path / "allowed.txt"
    allowed.write_text("allowed")
    secret = forbidden / "secret.txt"
    secret.write_text("secret")

    app = Path(
        "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/"
        "Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python"
    )
    library = Path(
        "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/"
        "Python.framework/Versions/3.14/Python"
    )
    if not app.exists() or not library.exists():
        pytest.skip("frozen private Python runtime is unavailable")
    sandbox = {
        "python_framework_app": str(app),
        "python_framework_app_sha256": hashlib.sha256(app.read_bytes()).hexdigest(),
        "python_framework_library": str(library),
        "python_framework_library_sha256": hashlib.sha256(
            library.read_bytes()
        ).hexdigest(),
    }
    private_python, framework_root = subject._copy_private_python(
        run_root,
        sandbox,
        8 * 1024**3,
    )
    template = (ROOT / "config/v4_12_unit_retrieval.sb").read_text()
    profile = subject.render_profile(
        template,
        allowed_files=[allowed],
        forbidden_roots=[forbidden],
        system_roots=[Path("/System"), Path("/usr"), Path("/opt/homebrew")],
        devices=[Path("/dev/null"), Path("/dev/urandom"), Path("/dev/fd")],
        metadata_extra=[run_root, output, scratch],
    )
    script = (
        "import errno,pathlib,socket,sys;"
        "assert pathlib.Path(sys.argv[1]).read_text()=='allowed';"
        "\ntry:pathlib.Path(sys.argv[2]).read_text();raise AssertionError('read')"
        "\nexcept PermissionError as e:assert e.errno==errno.EPERM;"
        "\ntry:pathlib.Path(sys.argv[3]).write_text('x');raise AssertionError('write')"
        "\nexcept PermissionError as e:assert e.errno==errno.EPERM;"
        "\ntry:"
        "\n s=socket.socket();s.connect(('127.0.0.1',9));raise AssertionError('network')"
        "\nexcept PermissionError as e:assert e.errno==errno.EPERM"
        "\nfinally:"
        "\n try:s.close()"
        "\n except NameError:pass;"
        "\npathlib.Path(sys.argv[4]).write_text('SANDBOX_OK')"
    )
    probe = run_root / "probe.py"
    probe.write_text(script)
    probe_fd = os.open(probe, os.O_RDONLY)
    probe_fd_path = f"/dev/fd/{probe_fd}"
    blocked_write = run_root / "blocked.txt"
    marker = output / "marker.txt"
    command = [
        "/usr/bin/sandbox-exec",
        "-D",
        f"RUN_ROOT={run_root}",
        "-D",
        "RUN_SPEC=/dev/null",
        "-D",
        "LOOKUP_DESCRIPTOR=/dev/null",
        "-D",
        "QUERIES=/dev/null",
        "-D",
        f"STRICT_SOURCE={probe_fd_path}",
        "-D",
        f"ENGINE_SOURCE={probe_fd_path}",
        "-D",
        f"PRIVATE_PACKAGE_ROOT={run_root}",
        "-D",
        f"RUN_OUTPUT={output}",
        "-D",
        f"RUN_TMP={scratch}",
        "-D",
        f"PYTHON_EXECUTABLE={private_python}",
        "-D",
        f"PYTHON_FRAMEWORK_ROOT={framework_root}",
        "-p",
        profile,
        str(private_python),
        "-B",
        probe_fd_path,
        str(allowed),
        str(secret),
        str(blocked_write),
        str(marker),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=run_root,
            env={
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "JOBLIB_MULTIPROCESSING": "0",
                "TMPDIR": str(scratch),
                "DYLD_FRAMEWORK_PATH": str(framework_root),
            },
            pass_fds=(probe_fd,),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.close(probe_fd)
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "SANDBOX_OK"
    assert not blocked_write.exists()


@pytest.mark.skipif(
    platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").exists(),
    reason="native Seatbelt integration is macOS-only",
)
def test_native_worker_profile_imports_sealed_private_package(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    output = run_root / "output"
    scratch = run_root / "tmp"
    package = run_root / "xgb_matcher"
    forbidden = tmp_path / "forbidden"
    for path in (run_root, output, scratch, package, forbidden):
        path.mkdir()
    for name in ("v412_strict_stores.py", "v412_unit_retrieval.py"):
        source = ROOT / "src/xgb_matcher" / name
        destination = package / name
        destination.write_bytes(source.read_bytes())
        os.chmod(destination, 0o444)
    (package / "__init__.py").write_bytes(b"")
    os.chmod(package / "__init__.py", 0o444)
    os.chmod(package, 0o555)

    app = Path(
        "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/"
        "Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python"
    )
    library = Path(
        "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/"
        "Python.framework/Versions/3.14/Python"
    )
    if not app.exists() or not library.exists():
        pytest.skip("frozen private Python runtime is unavailable")
    sandbox = {
        "python_framework_app": str(app),
        "python_framework_app_sha256": _sha(app),
        "python_framework_library": str(library),
        "python_framework_library_sha256": _sha(library),
    }
    private_python, framework_root = subject._copy_private_python(
        run_root,
        sandbox,
        8 * 1024**3,
    )
    template = (ROOT / "config/v4_12_unit_retrieval.sb").read_text()
    profile = subject.render_profile(
        template,
        allowed_files=[],
        forbidden_roots=[forbidden],
        system_roots=[Path("/System"), Path("/usr"), Path("/opt/homebrew")],
        devices=[Path("/dev/null"), Path("/dev/urandom"), Path("/dev/fd")],
        metadata_extra=[run_root, output, scratch, package],
    )
    command = [
        "/usr/bin/sandbox-exec",
        "-D",
        f"RUN_ROOT={run_root}",
        "-D",
        "RUN_SPEC=/dev/null",
        "-D",
        "LOOKUP_DESCRIPTOR=/dev/null",
        "-D",
        "QUERIES=/dev/null",
        "-D",
        f"STRICT_SOURCE={package / 'v412_strict_stores.py'}",
        "-D",
        f"ENGINE_SOURCE={package / 'v412_unit_retrieval.py'}",
        "-D",
        f"PRIVATE_PACKAGE_ROOT={package}",
        "-D",
        f"RUN_OUTPUT={output}",
        "-D",
        f"RUN_TMP={scratch}",
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
        "--help",
    ]
    result = subprocess.run(
        command,
        cwd=run_root,
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": str(run_root),
            "JOBLIB_MULTIPROCESSING": "0",
            "TMPDIR": str(scratch),
            "DYLD_FRAMEWORK_PATH": str(framework_root),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Run the V4.12 unit retrieval worker" in result.stdout
    assert "--run-spec" in result.stdout

    runtime_command = command[:-5] + [
        str(private_python),
        "-B",
        "-c",
        (
            "import json;"
            "from xgb_matcher.v412_unit_retrieval import _runtime_values;"
            "print(json.dumps(_runtime_values(),sort_keys=True))"
        ),
    ]
    runtime_result = subprocess.run(
        runtime_command,
        cwd=run_root,
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": str(run_root),
            "JOBLIB_MULTIPROCESSING": "0",
            "TMPDIR": str(scratch),
            "DYLD_FRAMEWORK_PATH": str(framework_root),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert runtime_result.returncode == 0, runtime_result.stderr
    observed_runtime = json.loads(runtime_result.stdout)
    expected_runtime = json.loads(
        (ROOT / "config/v4_12_unit_retrieval_engine_plan.json").read_text()
    )["runtime"]
    assert observed_runtime == expected_runtime
