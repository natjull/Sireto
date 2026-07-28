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
