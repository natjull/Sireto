import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

import scripts.build_v412_snapshot_lookup as subject
from src.xgb_matcher.v412_snapshot_lookup import V412SnapshotLookup


def _fixture_snapshot(path: Path) -> Path:
    connection = duckdb.connect()
    connection.execute(
        f"""
        COPY (
            SELECT * FROM (VALUES
                ('00000000000003', ' a ', NULL, 'E2', NULL, 'DU', '01.11Z'),
                ('00000000000001', 'F', 'E1', NULL, NULL, NULL, NULL),
                ('00000000000002', NULL, NULL, NULL, 'E3', '', '62.01Z')
            ) AS source(
                siret,
                etatAdministratifEtablissement,
                enseigne1Etablissement,
                enseigne2Etablissement,
                enseigne3Etablissement,
                denominationUsuelleEtablissement,
                activitePrincipaleEtablissement
            )
        ) TO '{path}' (FORMAT PARQUET)
        """
    )
    connection.close()
    return path


def _fixture_plan(snapshot: Path) -> dict:
    plan = json.loads(subject.DEFAULT_PLAN.read_text())
    plan["snapshot"].update(
        {
            "path": str(snapshot),
            "sha256": subject.file_sha256(snapshot),
            "row_count": 3,
            "unique_siret_count": 3,
            "invalid_siret_count": 0,
        }
    )
    plan["parity"].update(
        {
            "candidate_row_count": 3,
            "candidate_unique_siret_count": 2,
            "snapshot_sample_count": 2,
        }
    )
    ordered = sorted(
        ["00000000000001", "00000000000002", "00000000000003"],
        key=lambda siret: (
            hashlib.sha256(
                f"v412-lookup-parity:{siret}".encode()
            ).hexdigest(),
            siret,
        ),
    )[:2]
    plan["parity"]["snapshot_sample_ordered_newline_sha256"] = hashlib.sha256(
        "".join(f"{siret}\n" for siret in ordered).encode()
    ).hexdigest()
    return plan


def _candidate_file(path: Path) -> Path:
    connection = duckdb.connect()
    connection.execute(
        f"""
        COPY (
            SELECT * FROM (VALUES
                ('00000000000001', 1),
                ('00000000000003', 0),
                ('00000000000001', 0)
            ) AS source(candidate_siret, is_ground_truth)
        ) TO '{path}' (FORMAT PARQUET)
        """
    )
    connection.close()
    return path


def test_repository_plan_is_strict_and_matches_runtime():
    plan = subject._load_plan(subject.DEFAULT_PLAN)
    assert plan["build"]["duckdb"] == subject.EXPECTED_DUCKDB_POLICY
    assert plan["parity"]["candidate_projection"] == ["candidate_siret"]


def test_plan_rejects_frozen_parameter_mutation(tmp_path):
    plan = json.loads(subject.DEFAULT_PLAN.read_text())
    plan["build"]["duckdb"]["threads"] = 8
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="frozen build parameters"):
        subject._load_plan(path)


def test_plan_rejects_projection_snapshot_and_parity_mutations(tmp_path):
    mutations = [
        ("columns", lambda plan: plan["columns"][1].update({"sql": "NULL"})),
        ("snapshot", lambda plan: plan["snapshot"].update({"row_count": 1})),
        (
            "parity",
            lambda plan: plan["parity"].update(
                {"candidate_projection": ["candidate_siret", "is_ground_truth"]}
            ),
        ),
    ]
    for name, mutate in mutations:
        plan = json.loads(subject.DEFAULT_PLAN.read_text())
        mutate(plan)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(plan))
        with pytest.raises(ValueError, match=subject.BUILD_STOP):
            subject._load_plan(path)


def test_projection_sql_is_exact():
    plan = json.loads(subject.DEFAULT_PLAN.read_text())
    assert subject._projection_sql(plan).splitlines() == [
        "CAST(snapshot.siret AS VARCHAR) AS siret,",
        "upper(trim(CAST(snapshot.etatAdministratifEtablissement AS VARCHAR))) AS candidate_state,",
        "CAST(snapshot.enseigne1Etablissement AS VARCHAR) AS enseigne1,",
        "CAST(snapshot.enseigne2Etablissement AS VARCHAR) AS enseigne2,",
        "CAST(snapshot.enseigne3Etablissement AS VARCHAR) AS enseigne3,",
        "CAST(snapshot.denominationUsuelleEtablissement AS VARCHAR) AS denomination_usuelle,",
        "CAST(snapshot.activitePrincipaleEtablissement AS VARCHAR) AS activity_code",
    ]


def test_mini_build_preserves_projection_and_read_only_lookup(tmp_path):
    snapshot = _fixture_snapshot(tmp_path / "snapshot.parquet")
    plan = _fixture_plan(snapshot)
    database = tmp_path / "lookup.duckdb"
    stats = subject._build_database(
        database_path=database,
        snapshot_path=snapshot,
        staging=tmp_path,
        plan=plan,
    )
    assert stats == {
        "row_count": 3,
        "unique_siret_count": 3,
        "invalid_siret_count": 0,
    }
    with V412SnapshotLookup(database) as store:
        details = store.get_candidate_scene_details(["00000000000003"])
    assert details["00000000000003"] == {
        "candidate_state": "A",
        "enseigne1": None,
        "enseigne2": "E2",
        "enseigne3": None,
        "denomination_usuelle": "DU",
        "activity_code": "01.11Z",
    }


@pytest.mark.parametrize(
    "rows",
    [
        [
            ("00000000000001", "A"),
            ("00000000000001", "A"),
            ("00000000000002", "F"),
        ],
        [
            ("bad", "A"),
            ("00000000000001", "A"),
            ("00000000000002", "F"),
        ],
    ],
)
def test_mini_build_refuses_duplicate_or_invalid_siret(tmp_path, rows):
    snapshot = tmp_path / "bad.parquet"
    connection = duckdb.connect()
    connection.execute(
        f"""
        COPY (
          SELECT siret, state AS etatAdministratifEtablissement,
                 NULL::VARCHAR AS enseigne1Etablissement,
                 NULL::VARCHAR AS enseigne2Etablissement,
                 NULL::VARCHAR AS enseigne3Etablissement,
                 NULL::VARCHAR AS denominationUsuelleEtablissement,
                 NULL::VARCHAR AS activitePrincipaleEtablissement
          FROM (VALUES {", ".join(repr(row) for row in rows)})
               AS source(siret, state)
        ) TO '{snapshot}' (FORMAT PARQUET)
        """
    )
    connection.close()
    plan = _fixture_plan(snapshot)
    plan["snapshot"]["sha256"] = subject.file_sha256(snapshot)
    with pytest.raises(ValueError, match="cardinality or SIRET drift"):
        subject._build_database(
            database_path=tmp_path / "bad.duckdb",
            snapshot_path=snapshot,
            staging=tmp_path,
            plan=plan,
        )


def test_reference_and_parity_use_candidate_siret_only(tmp_path):
    snapshot = _fixture_snapshot(tmp_path / "snapshot.parquet")
    candidates = _candidate_file(tmp_path / "candidates.parquet")
    plan = _fixture_plan(snapshot)
    database = tmp_path / "lookup.duckdb"
    subject._build_database(
        database_path=database,
        snapshot_path=snapshot,
        staging=tmp_path,
        plan=plan,
    )
    reference = tmp_path / "reference.parquet"
    stats = subject._build_reference_once(
        snapshot_path=snapshot,
        candidates_path=candidates,
        reference_path=reference,
        plan=plan,
    )
    assert stats["reference_snapshot_scan_count"] == 1
    parity = subject._parity_check(
        database_path=database,
        reference_path=reference,
        plan=plan,
        reference_snapshot_scan_count=stats["reference_snapshot_scan_count"],
    )
    assert parity["mismatch_count"] == 0
    assert parity["candidate_unique_siret_count"] == 2
    assert parity["snapshot_sample_count"] == 2


def test_parity_stops_on_value_drift(tmp_path):
    snapshot = _fixture_snapshot(tmp_path / "snapshot.parquet")
    candidates = _candidate_file(tmp_path / "candidates.parquet")
    plan = _fixture_plan(snapshot)
    database = tmp_path / "lookup.duckdb"
    subject._build_database(
        database_path=database,
        snapshot_path=snapshot,
        staging=tmp_path,
        plan=plan,
    )
    connection = duckdb.connect(str(database))
    connection.execute(
        "UPDATE candidate_details SET enseigne1 = 'DRIFT' "
        "WHERE siret = '00000000000001'"
    )
    connection.execute("CHECKPOINT")
    connection.close()
    reference = tmp_path / "reference.parquet"
    subject._build_reference_once(
        snapshot_path=snapshot,
        candidates_path=candidates,
        reference_path=reference,
        plan=plan,
    )
    with pytest.raises(ValueError, match=subject.PARITY_STOP):
        subject._parity_check(
            database_path=database,
            reference_path=reference,
            plan=plan,
            reference_snapshot_scan_count=1,
        )


def test_execution_lock_rejects_missing_go_before_data(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, "_source_hashes", lambda _: {"source": "a" * 64})
    lock = {
        "schema_version": subject.LOCK_SCHEMA_VERSION,
        "purpose": subject.PURPOSE,
        "audit_verdict": "NO",
        "git_commit": "deadbeef",
        "source_hashes": {"source": "a" * 64},
        "input_paths": {},
        "input_hashes": {},
        "runtime": subject._runtime(),
        "output_root": str(subject.DEFAULT_OUTPUT_ROOT),
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="not independently authorized"):
        subject.validate_execution_lock(
            path,
            plan_path=tmp_path / "must-not-open.json",
            denylist_path=tmp_path / "must-not-open-deny.json",
            verify_git=False,
        )


def _exact_lock(plan_path, denylist_path):
    plan = subject._load_plan(plan_path)
    return {
        "schema_version": subject.LOCK_SCHEMA_VERSION,
        "purpose": subject.PURPOSE,
        "audit_verdict": subject.AUDIT_VERDICT,
        "git_commit": "frozen-commit",
        "source_hashes": subject._source_hashes(subject.REPO_ROOT),
        "input_paths": {
            "plan": str(Path(plan_path).resolve()),
            "denylist": str(Path(denylist_path).resolve()),
            "snapshot": str(Path(plan["snapshot"]["path"]).resolve()),
            "candidates": str(Path(plan["parity"]["candidate_path"]).resolve()),
        },
        "input_hashes": {
            "plan": subject.file_sha256(plan_path),
            "denylist": subject.file_sha256(denylist_path),
            "snapshot": plan["snapshot"]["sha256"],
            "candidates": plan["parity"]["candidate_sha256"],
        },
        "runtime": subject._runtime(),
        "output_root": str(subject.DEFAULT_OUTPUT_ROOT),
    }


def test_execution_lock_accepts_exact_go_and_verifies_git(monkeypatch, tmp_path):
    lock = _exact_lock(subject.DEFAULT_PLAN, subject.DEFAULT_DENYLIST)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))

    class Result:
        stdout = b""

    def fake_run(command, **_):
        relative = command[-1].split(":", 1)[1]
        result = Result()
        result.stdout = (subject.REPO_ROOT / relative).read_bytes()
        return result

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    observed, digest = subject.validate_execution_lock(
        path,
        plan_path=subject.DEFAULT_PLAN,
        denylist_path=subject.DEFAULT_DENYLIST,
    )
    assert observed == lock
    assert digest == subject.file_sha256(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime", {}),
        ("output_root", "/tmp/not-frozen"),
        ("input_paths", {}),
        ("input_hashes", {}),
        ("source_hashes", {}),
    ],
)
def test_execution_lock_rejects_environment_mutations(
    field, value, tmp_path
):
    lock = _exact_lock(subject.DEFAULT_PLAN, subject.DEFAULT_DENYLIST)
    lock[field] = value
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match=subject.BUILD_STOP):
        subject.validate_execution_lock(
            path,
            plan_path=subject.DEFAULT_PLAN,
            denylist_path=subject.DEFAULT_DENYLIST,
            verify_git=False,
        )


def test_build_artifact_has_no_git_verification_bypass():
    import inspect

    assert "verify_git" not in inspect.signature(subject.build_artifact).parameters


@pytest.fixture
def published_mini_artifact(monkeypatch, tmp_path):
    snapshot = _fixture_snapshot(tmp_path / "snapshot.parquet")
    candidates = _candidate_file(tmp_path / "candidates.parquet")
    plan = _fixture_plan(snapshot)
    plan["parity"]["candidate_path"] = str(candidates)
    plan["parity"]["candidate_sha256"] = subject.file_sha256(candidates)
    plan["build"]["output_root"] = str(tmp_path / "published")
    plan["build"]["disk_free_min_bytes"] = 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    denylist_path = tmp_path / "denylist.json"
    denylist_path.write_text("{}")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text('{"authorized":true}')
    source_hashes = subject._source_hashes(subject.REPO_ROOT)
    lock = {
        "source_hashes": source_hashes,
        "runtime": subject._runtime(),
        "output_root": plan["build"]["output_root"],
    }
    monkeypatch.setattr(subject, "DEFAULT_PLAN", plan_path)
    monkeypatch.setattr(subject, "DEFAULT_OUTPUT_ROOT", Path(plan["build"]["output_root"]))
    monkeypatch.setattr(subject, "EXPECTED_SNAPSHOT", copy.deepcopy(plan["snapshot"]))
    monkeypatch.setattr(subject, "EXPECTED_PARITY", copy.deepcopy(plan["parity"]))
    monkeypatch.setattr(subject, "_load_plan", lambda _: plan)
    monkeypatch.setattr(
        subject,
        "_load_denylist",
        lambda _: ({}, set(), []),
    )
    monkeypatch.setattr(
        subject,
        "validate_execution_lock",
        lambda *_, **__: (lock, subject.file_sha256(lock_path)),
    )
    monkeypatch.setattr(
        subject,
        "_external_output",
        lambda path: Path(path).resolve(),
    )
    artifact = subject.build_artifact(
        execution_lock_path=lock_path,
        plan_path=plan_path,
        denylist_path=denylist_path,
    )
    assert artifact.is_dir()
    subject.validate_artifact(artifact)
    return artifact


def _reseal_json(artifact: Path, filename: str, value: dict) -> None:
    path = artifact / filename
    path.write_text(json.dumps(value))
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][filename] = {
        "sha256": subject.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest))


def test_mini_build_artifact_publishes_and_validates(published_mini_artifact):
    assert {item.name for item in published_mini_artifact.iterdir()} == {
        "candidate_details.duckdb",
        "manifest.json",
        "integrity.json",
        "timing.json",
    }
    assert not any(
        item.name.startswith(".") for item in published_mini_artifact.parent.iterdir()
    )
    assert not (
        published_mini_artifact / "candidate_details.duckdb.wal"
    ).exists()


def test_publication_order_is_validate_atomic_fsync_revalidate():
    import inspect

    source = inspect.getsource(subject.build_artifact)
    prevalidate = source.index("validate_artifact(staging")
    atomic = source.index("os.replace(staging, target)")
    parent_fsync = source.index("_fsync_directory(output_root)", atomic)
    postvalidate = source.index("validate_artifact(target)", parent_fsync)
    assert prevalidate < atomic < parent_fsync < postvalidate


def test_json_writer_fsyncs_file_and_directory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subject.os, "fsync", lambda descriptor: calls.append(descriptor))
    subject._write_json(tmp_path / "value.json", {"value": 1})
    assert len(calls) >= 2


def test_build_refuses_existing_immutable_target(published_mini_artifact):
    root = published_mini_artifact.parent.parent
    with pytest.raises(FileExistsError, match="already exists"):
        subject.build_artifact(
            execution_lock_path=root / "lock.json",
            plan_path=root / "plan.json",
            denylist_path=root / "denylist.json",
        )


def test_build_rejects_rss_and_cleans_staging(
    monkeypatch, published_mini_artifact
):
    root = published_mini_artifact.parent.parent
    lock_path = root / "lock.json"
    lock_path.write_text('{"authorized":"second-identity"}')
    monkeypatch.setattr(subject, "peak_rss_bytes", lambda: 9 * 1024**3)
    with pytest.raises(ValueError, match="RSS exceeds 8 GiB"):
        subject.build_artifact(
            execution_lock_path=lock_path,
            plan_path=root / "plan.json",
            denylist_path=root / "denylist.json",
        )
    assert not any(
        item.name.startswith(".") for item in published_mini_artifact.parent.iterdir()
    )


def test_validator_rejects_extra_directory_and_symlink(
    published_mini_artifact,
):
    extra = published_mini_artifact / "extra"
    extra.mkdir()
    with pytest.raises(ValueError, match="artifact file set"):
        subject.validate_artifact(published_mini_artifact)
    extra.rmdir()
    link = published_mini_artifact / "extra"
    link.symlink_to(published_mini_artifact / "timing.json")
    with pytest.raises(ValueError, match="artifact file set"):
        subject.validate_artifact(published_mini_artifact)


@pytest.mark.parametrize(
    ("filename", "mutator"),
    [
        ("integrity.json", lambda value: value.update({"row_count": 99})),
        (
            "integrity.json",
            lambda value: value["parity"].update({"mismatch_count": 1}),
        ),
        ("timing.json", lambda value: value.update({"peak_rss_bytes": 9 * 1024**3})),
    ],
)
def test_validator_rejects_resealed_declaration_tamper(
    published_mini_artifact, filename, mutator
):
    value = json.loads((published_mini_artifact / filename).read_text())
    mutator(value)
    _reseal_json(published_mini_artifact, filename, value)
    with pytest.raises(ValueError, match=subject.BUILD_STOP):
        subject.validate_artifact(published_mini_artifact)


def test_validator_rejects_hash_size_and_provenance_tamper(
    published_mini_artifact,
):
    manifest_path = published_mini_artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["timing.json"]["size_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="output drift"):
        subject.validate_artifact(published_mini_artifact)


def test_validator_rejects_input_path_and_lock_sha_tamper(
    published_mini_artifact,
):
    manifest_path = published_mini_artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["snapshot"]["path"] = "/tmp/fake"
    manifest["inputs"]["execution_lock"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=subject.BUILD_STOP):
        subject.validate_artifact(published_mini_artifact)


@pytest.mark.parametrize("field", ["source_hashes", "runtime"])
def test_validator_rejects_manifest_provenance_tamper(
    published_mini_artifact, field
):
    manifest_path = published_mini_artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = {}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=subject.BUILD_STOP):
        subject.validate_artifact(published_mini_artifact)


def test_validator_rejects_lock_symlink(published_mini_artifact):
    root = published_mini_artifact.parent.parent
    link = root / "lock-link.json"
    link.symlink_to(root / "lock.json")
    manifest_path = published_mini_artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["execution_lock"]["path"] = str(link)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=subject.BUILD_STOP):
        subject.validate_artifact(published_mini_artifact)


def test_validator_rejects_extra_nested_fields(published_mini_artifact):
    integrity = json.loads(
        (published_mini_artifact / "integrity.json").read_text()
    )
    integrity["parity"]["extra"] = True
    _reseal_json(published_mini_artifact, "integrity.json", integrity)
    with pytest.raises(ValueError, match=subject.BUILD_STOP):
        subject.validate_artifact(published_mini_artifact)


def test_validator_rejects_extra_output_record_field(published_mini_artifact):
    manifest_path = published_mini_artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["timing.json"]["extra"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="output drift"):
        subject.validate_artifact(published_mini_artifact)


def test_validator_rejects_database_row_tamper(published_mini_artifact):
    database = published_mini_artifact / "candidate_details.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute("DELETE FROM candidate_details WHERE siret='00000000000001'")
    connection.execute("CHECKPOINT")
    connection.close()
    manifest_path = published_mini_artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["candidate_details.duckdb"] = {
        "sha256": subject.file_sha256(database),
        "size_bytes": database.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="database cardinality"):
        subject.validate_artifact(published_mini_artifact)


def test_build_stops_on_low_disk_before_staging(monkeypatch, tmp_path):
    lock = {"output_root": str(tmp_path)}
    plan = json.loads(subject.DEFAULT_PLAN.read_text())
    monkeypatch.setattr(
        subject, "validate_execution_lock", lambda *_, **__: (lock, "a" * 64)
    )
    monkeypatch.setattr(subject, "_load_plan", lambda _: plan)
    monkeypatch.setattr(subject, "_external_output", lambda _: tmp_path)
    monkeypatch.setattr(subject, "_load_denylist", lambda _: ({}, set(), []))
    monkeypatch.setattr(subject, "_assert_inputs_allowed", lambda *_, **__: None)
    monkeypatch.setattr(
        subject,
        "_fingerprint",
        lambda path: {
            "device": 1,
            "inode": 1,
            "size_bytes": 1,
            "mtime_ns": 1,
            "sha256": (
                plan["snapshot"]["sha256"]
                if "StockEtablissement" in str(path)
                else plan["parity"]["candidate_sha256"]
            ),
        },
    )
    monkeypatch.setattr(
        subject.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=0),
    )
    with pytest.raises(ValueError, match="less than 50 GiB"):
        subject.build_artifact(
            execution_lock_path=tmp_path / "lock",
            plan_path=subject.DEFAULT_PLAN,
            denylist_path=subject.DEFAULT_DENYLIST,
        )


def test_denylist_rejects_relocated_forbidden_hash(tmp_path):
    policy, hashes, roots = subject._load_denylist(subject.DEFAULT_DENYLIST)
    assert len(policy["artifacts"]) == 3
    relocated = tmp_path / "relocated.bin"
    relocated.write_bytes(b"relocated")
    forbidden = next(iter(hashes))
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        subject,
        "file_sha256",
        lambda path: forbidden if Path(path) == relocated else "0" * 64,
    )
    try:
        with pytest.raises(ValueError, match="forbidden challenge hash"):
            subject._assert_inputs_allowed(
                [relocated],
                forbidden_hashes=hashes,
                forbidden_roots=roots,
            )
    finally:
        monkey.undo()


def test_assert_unchanged_closes_toctou_window(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"before")
    fingerprint = subject._fingerprint(path)
    path.write_bytes(b"after")
    with pytest.raises(ValueError, match="changed during execution"):
        subject._assert_unchanged(path, fingerprint)


def test_real_build_requires_external_lock_from_cli(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["build_v412_snapshot_lookup.py"]
    )
    with pytest.raises(SystemExit, match="execution-lock is required"):
        subject.main()
