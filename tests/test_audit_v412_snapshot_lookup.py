import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import duckdb
import pytest

TEST_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEST_REPO_ROOT))

import scripts.audit_v412_snapshot_lookup as subject


def _snapshot(path: Path) -> Path:
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
          SELECT * FROM (VALUES
            ('94410569100017','A','A1',NULL,NULL,'D1','01.1A'),
            ('92883024900019','F',NULL,'A2',NULL,NULL,NULL),
            ('53539062900017','A',NULL,NULL,'A3','D3','62.0Z')
          ) t(siret,etatAdministratifEtablissement,
              enseigne1Etablissement,enseigne2Etablissement,
              enseigne3Etablissement,denominationUsuelleEtablissement,
              activitePrincipaleEtablissement)
        ) TO '{path}' (FORMAT PARQUET)
        """
    )
    con.close()
    return path


def _database(path: Path, snapshot: Path) -> Path:
    con = duckdb.connect(str(path))
    con.execute(
        f"""
        CREATE TABLE candidate_details AS SELECT
          CAST(siret AS VARCHAR) siret,
          upper(trim(CAST(etatAdministratifEtablissement AS VARCHAR)))
            candidate_state,
          CAST(enseigne1Etablissement AS VARCHAR) enseigne1,
          CAST(enseigne2Etablissement AS VARCHAR) enseigne2,
          CAST(enseigne3Etablissement AS VARCHAR) enseigne3,
          CAST(denominationUsuelleEtablissement AS VARCHAR)
            denomination_usuelle,
          CAST(activitePrincipaleEtablissement AS VARCHAR) activity_code
        FROM read_parquet('{snapshot}')
        """
    )
    con.execute(
        "CREATE UNIQUE INDEX candidate_details_siret_uidx "
        "ON candidate_details(siret)"
    )
    con.execute("CHECKPOINT")
    con.close()
    return path


def _plan(snapshot: Path, database: Path, tmp_path: Path) -> dict:
    value = json.loads(subject.DEFAULT_PLAN.read_text())
    sirets = ["94410569100017", "92883024900019", "53539062900017"]
    lf = b"".join(item.encode() + bytes([10]) for item in sirets)
    bad = b"".join(item.encode() + bytes([92, 110]) for item in sirets)
    value["snapshot"] = {
        "path": str(snapshot),
        "sha256": subject.file_sha256(snapshot),
    }
    value["artifact"]["path"] = str(database.parent)
    value["expected"].update(
        {
            "sample_count": 3,
            "first_sirets": sirets,
            "last_sirets": sirets,
            "lf_payload_bytes": len(lf),
            "lf_payload_sha256": hashlib.sha256(lf).hexdigest(),
            "counterexample_bytes": len(bad),
            "counterexample_sha256": hashlib.sha256(bad).hexdigest(),
        }
    )
    value["lookup_schema"].update(
        {"row_count": 3, "unique_siret_count": 3}
    )
    value["duckdb"]["temp_root"] = str(tmp_path / "ducktmp")
    value["output_root"] = str(tmp_path / "output")
    return value


def _official_artifact(
    tmp_path: Path,
    database: Path,
    *,
    snapshot_sha256: str,
) -> tuple[Path, Path]:
    plan_path = tmp_path / "official-plan.json"
    plan_path.write_text("{}\n")
    lock_path = tmp_path / "official-lock.json"
    lock_path.write_text('{"locked":true}\n')
    runtime = {
        "python": "test",
        "platform": "test",
        "machine": "test",
        "duckdb": "1.4.3",
    }
    sources = {"official-test-source": "a" * 64}
    candidate_hash = "b" * 64
    identity = {
        "schema_version": "sireto-v4.12-snapshot-lookup-1",
        "plan_sha256": subject.file_sha256(plan_path),
        "execution_lock_sha256": subject.file_sha256(lock_path),
        "source_hashes": sources,
        "runtime": runtime,
        "snapshot_sha256": snapshot_sha256,
        "candidate_sha256": candidate_hash,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    root = tmp_path / build_id
    root.mkdir()
    database.rename(root / "candidate_details.duckdb")
    integrity = {
        "candidate_projection": ["candidate_siret"],
        "challenge_opened": False,
        "invalid_siret_count": 0,
        "labels_opened": False,
        "lookup_opened_read_only": True,
        "parity": {
            "candidate_unique_siret_count": 3,
            "lookup_batch_max": 100,
            "mismatch_count": 0,
            "reference_row_count": 3,
            "reference_snapshot_scan_count": 1,
            "snapshot_sample_count": 3,
            "snapshot_sample_ordered_newline_sha256": "c" * 64,
        },
        "reference": {
            "candidate_row_count": 3,
            "candidate_unique_siret_count": 3,
            "reference_snapshot_scan_count": 1,
        },
        "row_count": 3,
        "unique_index": "candidate_details_siret_uidx",
        "unique_siret_count": 3,
        "verdict": "GO_V412_SNAPSHOT_LOOKUP",
    }
    timing = {
        "elapsed_seconds": 0.0,
        "peak_rss_bytes": 1,
        "peak_rss_limit_bytes": 8 * 1024**3,
        "serve_latency_gate_evaluated": False,
    }
    (root / "integrity.json").write_text(json.dumps(integrity))
    (root / "timing.json").write_text(json.dumps(timing))
    outputs = {
        name: {
            "sha256": subject.file_sha256(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in (
            "candidate_details.duckdb",
            "integrity.json",
            "timing.json",
        )
    }
    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "snapshot": {
                "path": "mini-snapshot.parquet",
                "sha256": snapshot_sha256,
            },
            "candidates_parity_only": {
                "path": "mini-candidates.parquet",
                "sha256": candidate_hash,
                "projection": ["candidate_siret"],
                "opened_after_lookup_build": True,
            },
            "execution_lock": {
                "path": str(lock_path),
                "sha256": subject.file_sha256(lock_path),
            },
        },
        "outputs": outputs,
        "verdict": "GO_V412_SNAPSHOT_LOOKUP",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    config = {
        "snapshot": {
            "invalid_siret_count": 0,
            "path": "mini-snapshot.parquet",
            "row_count": 3,
            "sha256": snapshot_sha256,
            "unique_siret_count": 3,
        },
        "parity": {
            "candidate_path": "mini-candidates.parquet",
            "candidate_projection": ["candidate_siret"],
            "candidate_row_count": 3,
            "candidate_sha256": candidate_hash,
            "candidate_unique_siret_count": 3,
            "reference_snapshot_scan_count": 1,
            "snapshot_sample_count": 3,
            "snapshot_sample_namespace": "test:",
            "snapshot_sample_ordered_newline_sha256": "c" * 64,
        },
        "plan": str(plan_path),
        "lock_hash": subject.file_sha256(lock_path),
        "runtime": runtime,
        "source_hashes": sources,
    }
    config_path = tmp_path / "official-shim-config.json"
    config_path.write_text(json.dumps(config))
    shim = tmp_path / "official-validator-shim.py"
    shim.write_text(
        "import json,sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0,{str(subject.REPO_ROOT)!r})\n"
        "import scripts.build_v412_snapshot_lookup as official\n"
        "cfg=json.loads(Path(sys.argv[2]).read_text())\n"
        "official.EXPECTED_SNAPSHOT=cfg['snapshot']\n"
        "official.EXPECTED_PARITY=cfg['parity']\n"
        "official.DEFAULT_PLAN=Path(cfg['plan'])\n"
        "official._source_hashes=lambda root: cfg['source_hashes']\n"
        "official._runtime=lambda: cfg['runtime']\n"
        "official.validate_execution_lock=lambda *a,**k: ({},cfg['lock_hash'])\n"
        "official.validate_artifact(Path(sys.argv[1]))\n"
    )
    return root, shim


def _mini_audit_environment(monkeypatch, tmp_path):
    mini_repo = tmp_path / "mini-repo"
    (mini_repo / "scripts").mkdir(parents=True)
    runner_copy = mini_repo / "scripts/audit_v412_snapshot_lookup.py"
    runner_copy.write_bytes(Path(subject.__file__).read_bytes())
    official_marker = tmp_path / "official-validator-called"
    official = mini_repo / "scripts/official-validator.py"
    official.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"Path({str(official_marker)!r}).write_text('called')\n"
        "assert sys.argv[1] == '--validate-artifact'\n"
        "assert Path(sys.argv[2]).is_dir()\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=mini_repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "audit@example.invalid"],
        cwd=mini_repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Audit Test"],
        cwd=mini_repo,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=mini_repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "frozen audit sources"],
        cwd=mini_repo,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mini_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    snapshot = _snapshot(tmp_path / "mini-snapshot.parquet")
    artifact = tmp_path / "mini-artifact"
    artifact.mkdir()
    database = _database(
        artifact / "candidate_details.duckdb", snapshot
    )
    for name in ("integrity.json", "manifest.json", "timing.json"):
        (artifact / name).write_text(json.dumps({"mini": name}))
    plan = json.loads(subject.DEFAULT_PLAN.read_text())
    sirets = [
        row[1]
        for row in sorted(
            (
                hashlib.sha256(
                    (plan["selection"]["namespace"] + value).encode()
                ).hexdigest(),
                value,
            )
            for value in (
                "94410569100017",
                "92883024900019",
                "53539062900017",
            )
        )
    ]
    lf = b"".join(value.encode("ascii") + bytes([10]) for value in sirets)
    bad = b"".join(
        value.encode("ascii") + bytes([92, 110]) for value in sirets
    )
    plan["artifact"] = {
        "path": str(artifact),
        "files": {
            path.name: {
                "sha256": subject.file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact.iterdir()
        },
    }
    plan["snapshot"] = {
        "path": str(snapshot),
        "sha256": subject.file_sha256(snapshot),
    }
    plan["audit_sources"] = ["scripts/audit_v412_snapshot_lookup.py"]
    plan["expected"].update(
        {
            "sample_count": 3,
            "first_sirets": sirets,
            "last_sirets": sirets,
            "lf_payload_bytes": len(lf),
            "lf_payload_sha256": hashlib.sha256(lf).hexdigest(),
            "counterexample_bytes": len(bad),
            "counterexample_sha256": hashlib.sha256(bad).hexdigest(),
        }
    )
    plan["lookup_schema"].update(
        {"row_count": 3, "unique_siret_count": 3}
    )
    ssd_root = tmp_path / "ssd"
    ssd_root.mkdir()
    output_root = ssd_root / "audits"
    temp_root = ssd_root / "tmp"
    plan["output_root"] = str(output_root)
    plan["duckdb"]["temp_root"] = str(temp_root)
    plan_path = tmp_path / "mini-audit-plan.json"
    plan_path.write_text(json.dumps(plan))

    monkeypatch.setattr(subject, "REPO_ROOT", mini_repo)
    monkeypatch.setattr(subject, "OFFICIAL_VALIDATOR", official)
    monkeypatch.setattr(subject, "SSD_ROOT", ssd_root)
    sources = subject.source_hashes(plan)
    locked_runtime = subject.runtime()
    lock = {
        "schema_version": subject.LOCK_SCHEMA_VERSION,
        "purpose": subject.PURPOSE,
        "audit_verdict": subject.AUDIT_VERDICT,
        "git_commit": commit,
        "source_hashes": sources,
        "input_paths": {
            "plan": str(plan_path.resolve()),
            "snapshot": str(snapshot.resolve()),
            "artifact": str(artifact.resolve()),
            "official_validator": str(official.resolve()),
        },
        "input_hashes": {
            "plan": subject.file_sha256(plan_path),
            "snapshot": subject.file_sha256(snapshot),
            "artifact_manifest": plan["artifact"]["files"]["manifest.json"][
                "sha256"
            ],
            "official_validator": subject.file_sha256(official),
        },
        "runtime": locked_runtime,
        "output_root": str(output_root),
        "temp_root": str(temp_root),
    }
    lock_path = tmp_path / "mini-audit-lock.json"
    lock_path.write_text(json.dumps(lock))
    return SimpleNamespace(
        artifact=artifact,
        lock=lock_path,
        marker=official_marker,
        output=output_root,
        plan=plan,
        plan_path=plan_path,
        source=runner_copy,
        temp=temp_root,
    )


def test_payload_distinguishes_lf_from_literal_backslash_n():
    expected = {
        "lf_payload_bytes": 30,
        "lf_payload_sha256": hashlib.sha256(
            b"00000000000001" + bytes([10]) + b"00000000000002" + bytes([10])
        ).hexdigest(),
        "counterexample_bytes": 32,
        "counterexample_sha256": hashlib.sha256(
            b"00000000000001"
            + bytes([92, 110])
            + b"00000000000002"
            + bytes([92, 110])
        ).hexdigest(),
    }
    observed = subject.sample_payloads(
        ["00000000000001", "00000000000002"], expected
    )
    assert observed["lf_payload_sha256"] != observed[
        "counterexample_sha256"
    ]


@pytest.mark.parametrize("count", [0, 1, 100])
def test_lookup_batch_bounds_accept(count, tmp_path):
    snapshot = _snapshot(tmp_path / "snapshot.parquet")
    database = _database(tmp_path / "lookup.duckdb", snapshot)
    requested = ["94410569100017"] * count
    expected = {"94410569100017": ("A", "A1", None, None, "D1", "01.1A")}
    if count == 0:
        expected = {}
    assert subject.compare_lookup(
        database, requested, expected, batch_max=100
    ) == 0


def test_store_refuses_101_sirets(tmp_path):
    snapshot = _snapshot(tmp_path / "snapshot.parquet")
    database = _database(tmp_path / "lookup.duckdb", snapshot)
    with pytest.raises(ValueError):
        subject.compare_lookup(
            database,
            [f"{value:014d}" for value in range(101)],
            {},
            batch_max=101,
        )


def test_official_accepts_resealed_value_mutation_independent_refuses(tmp_path):
    snapshot = _snapshot(tmp_path / "snapshot.parquet")
    database = _database(tmp_path / "lookup.duckdb", snapshot)
    plan = _plan(snapshot, database, tmp_path)
    sirets = subject.select_sirets_phase_a(
        snapshot, plan, tmp_path / "phase-a"
    )
    payload = subject.sample_payloads(sirets, plan["expected"])
    assert payload["lf_payload_bytes"] == 45
    expected = subject.project_values_phase_b(
        snapshot, sirets, plan, tmp_path / "phase-b"
    )
    assert subject.compare_lookup(database, sirets, expected, batch_max=100) == 0
    con = duckdb.connect(str(database))
    con.execute(
        "UPDATE candidate_details SET enseigne1='MUTATED' "
        "WHERE siret='94410569100017'"
    )
    con.execute("CHECKPOINT")
    con.close()
    artifact, official_shim = _official_artifact(
        tmp_path,
        database,
        snapshot_sha256=subject.file_sha256(snapshot),
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutated_database = artifact / "candidate_details.duckdb"
    manifest["outputs"]["candidate_details.duckdb"] = {
        "sha256": subject.file_sha256(mutated_database),
        "size_bytes": mutated_database.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(official_shim), str(artifact), str(tmp_path / "official-shim-config.json")],
        cwd=subject.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError, match=subject.STOP_SAMPLE):
        subject.compare_lookup(
            mutated_database, sirets, expected, batch_max=100
        )


def test_official_validator_is_only_subprocess(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *args, **kwargs: (
            calls.append((args, kwargs))
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    subject.run_official_validator(tmp_path)
    assert calls
    assert "--validate-artifact" in calls[0][0][0]


def _artifact_file_fixture(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    plan = {"artifact": {"files": {}}}
    for name in (
        "candidate_details.duckdb",
        "manifest.json",
        "integrity.json",
        "timing.json",
    ):
        path = root / name
        path.write_bytes(name.encode())
        plan["artifact"]["files"][name] = {
            "sha256": subject.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return root, plan


def test_artifact_files_reject_extra(tmp_path):
    root, plan = _artifact_file_fixture(tmp_path)
    subject.validate_artifact_files(root, plan)
    (root / "extra").write_bytes(b"x")
    with pytest.raises(ValueError):
        subject.validate_artifact_files(root, plan)


def test_artifact_files_reject_wal(tmp_path):
    root, plan = _artifact_file_fixture(tmp_path)
    (root / "candidate_details.duckdb.wal").write_bytes(b"x")
    with pytest.raises(ValueError):
        subject.validate_artifact_files(root, plan)


def test_artifact_files_reject_symlink(tmp_path):
    root, plan = _artifact_file_fixture(tmp_path)
    target = root / "manifest-target"
    (root / "manifest.json").rename(target)
    (root / "manifest.json").symlink_to(target)
    plan["artifact"]["files"]["manifest.json"] = {
        "sha256": subject.file_sha256(target),
        "size_bytes": target.stat().st_size,
    }
    plan["artifact"]["files"]["manifest-target"] = {
        "sha256": subject.file_sha256(target),
        "size_bytes": target.stat().st_size,
    }
    with pytest.raises(ValueError):
        subject.validate_artifact_files(root, plan)


def test_artifact_files_reject_hash_drift(tmp_path):
    root, plan = _artifact_file_fixture(tmp_path)
    (root / "timing.json").write_bytes(b"changed")
    with pytest.raises(ValueError):
        subject.validate_artifact_files(root, plan)


def test_lookup_schema_cardinality_and_unique_index(tmp_path):
    snapshot = _snapshot(tmp_path / "snapshot.parquet")
    database = _database(tmp_path / "lookup.duckdb", snapshot)
    plan = _plan(snapshot, database, tmp_path)
    subject.validate_lookup_integrity(database, plan)
    altered = copy.deepcopy(plan)
    altered["lookup_schema"]["row_count"] = 4
    with pytest.raises(ValueError, match="lookup integrity changed"):
        subject.validate_lookup_integrity(database, altered)


def test_publication_validation_is_strict(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    identity = {
        "schema_version": subject.SCHEMA_VERSION,
        "plan_sha256": "a" * 64,
    }
    audit = {
        "schema_version": subject.SCHEMA_VERSION,
        "sample_count": 3,
        "first_sirets": ["1", "2", "3"],
        "last_sirets": ["1", "2", "3"],
        "lf_payload_bytes": 45,
        "lf_payload_sha256": "b" * 64,
        "counterexample_bytes": 48,
        "counterexample_sha256": "c" * 64,
        "mismatch_count": 0,
        "peak_rss_bytes": 1,
        "verdict": subject.GO,
    }
    subject._write_json(staging / "audit.json", audit)
    manifest = {
        **identity,
        "build_id": "build",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "audit.json": {
                "sha256": subject.file_sha256(staging / "audit.json"),
                "size_bytes": (staging / "audit.json").stat().st_size,
            }
        },
        "verdict": subject.GO,
    }
    subject._write_json(staging / "manifest.json", manifest)
    subject.validate_publication(
        staging,
        identity,
        audit,
        build_id="build",
        max_rss_bytes=10,
    )
    mutations = [
        {**audit, "extra": True},
        {**audit, "sample_count": True},
        {**audit, "peak_rss_bytes": 11},
        {**audit, "verdict": subject.STOP},
    ]
    for mutation in mutations:
        (staging / "audit.json").write_text(json.dumps(mutation))
        resealed = copy.deepcopy(manifest)
        resealed["outputs"]["audit.json"] = {
            "sha256": subject.file_sha256(staging / "audit.json"),
            "size_bytes": (staging / "audit.json").stat().st_size,
        }
        (staging / "manifest.json").write_text(json.dumps(resealed))
        with pytest.raises(ValueError, match="publication"):
            subject.validate_publication(
                staging,
                identity,
                audit,
                build_id="build",
                max_rss_bytes=10,
            )
    (staging / "audit.json").write_text(json.dumps(audit))
    manifest["extra"] = True
    manifest["outputs"]["audit.json"] = {
        "sha256": subject.file_sha256(staging / "audit.json"),
        "size_bytes": (staging / "audit.json").stat().st_size,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="publication"):
        subject.validate_publication(
            staging,
            identity,
            audit,
            build_id="build",
            max_rss_bytes=10,
        )


def test_audit_end_to_end_atomic_publication(monkeypatch, tmp_path):
    environment = _mini_audit_environment(monkeypatch, tmp_path)
    replacements = []
    validations = []
    real_replace = subject.os.replace
    real_validate = subject.validate_publication

    def tracked_replace(source, target):
        replacements.append((Path(source), Path(target)))
        return real_replace(source, target)

    def tracked_validate(root, *args, **kwargs):
        validations.append(Path(root))
        return real_validate(root, *args, **kwargs)

    monkeypatch.setattr(subject.os, "replace", tracked_replace)
    monkeypatch.setattr(subject, "validate_publication", tracked_validate)
    target = subject.audit(
        execution_lock=environment.lock,
        plan_path=environment.plan_path,
    )
    assert environment.marker.read_text() == "called"
    assert target.is_dir()
    assert {path.name for path in target.iterdir()} == {
        "audit.json",
        "manifest.json",
    }
    assert len(replacements) == 1
    assert replacements[0][1] == target
    assert validations[-1] == target
    assert validations[0] != target
    assert not list(environment.output.glob(".*.tmp-*"))


def test_audit_rss_rejection_cleans_up(monkeypatch, tmp_path):
    environment = _mini_audit_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subject,
        "peak_rss_bytes",
        lambda: environment.plan["max_rss_bytes"] + 1,
    )
    with pytest.raises(ValueError, match="RSS exceeds"):
        subject.audit(
            execution_lock=environment.lock,
            plan_path=environment.plan_path,
        )
    assert not any(environment.output.iterdir())
    assert not any(environment.temp.iterdir())


def test_audit_rehashes_after_staging_and_rejects_toctou(
    monkeypatch, tmp_path
):
    environment = _mini_audit_environment(monkeypatch, tmp_path)
    real_validate = subject.validate_publication
    calls = 0

    def mutate_after_staging(root, *args, **kwargs):
        nonlocal calls
        real_validate(root, *args, **kwargs)
        calls += 1
        if calls == 1:
            environment.source.write_bytes(
                environment.source.read_bytes() + b"\n# drift\n"
            )

    monkeypatch.setattr(
        subject, "validate_publication", mutate_after_staging
    )
    with pytest.raises(ValueError, match="changed during audit"):
        subject.audit(
            execution_lock=environment.lock,
            plan_path=environment.plan_path,
        )
    assert calls == 1
    assert not any(environment.output.iterdir())


def test_audit_postvalidation_failure_removes_target(
    monkeypatch, tmp_path
):
    environment = _mini_audit_environment(monkeypatch, tmp_path)
    real_replace = subject.os.replace

    def replace_then_corrupt(source, target):
        real_replace(source, target)
        (Path(target) / "audit.json").write_text("{}")

    monkeypatch.setattr(subject.os, "replace", replace_then_corrupt)
    with pytest.raises(ValueError, match="publication"):
        subject.audit(
            execution_lock=environment.lock,
            plan_path=environment.plan_path,
        )
    assert not any(environment.output.iterdir())


@pytest.mark.parametrize("drift", ["source", "runtime"])
def test_execution_lock_rejects_frozen_drift(
    monkeypatch, tmp_path, drift
):
    snapshot = tmp_path / "snapshot"
    snapshot.write_bytes(b"snapshot")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = artifact / "manifest.json"
    manifest.write_bytes(b"manifest")
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(b"plan")
    plan = {
        "artifact": {
            "path": str(artifact),
            "files": {
                "manifest.json": {
                    "sha256": subject.file_sha256(manifest),
                }
            },
        },
        "snapshot": {
            "path": str(snapshot),
            "sha256": subject.file_sha256(snapshot),
        },
        "output_root": str(tmp_path / "output"),
        "duckdb": {"temp_root": str(tmp_path / "temp")},
    }
    locked_runtime = {"duckdb": "1.4.3"}
    locked_sources = {"runner": "a" * 64}
    monkeypatch.setattr(subject, "load_plan", lambda _: plan)
    monkeypatch.setattr(subject, "source_hashes", lambda _: locked_sources)
    monkeypatch.setattr(subject, "runtime", lambda: locked_runtime)
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=b"source"),
    )
    locked_sources["runner"] = hashlib.sha256(b"source").hexdigest()
    lock = {
        "schema_version": subject.LOCK_SCHEMA_VERSION,
        "purpose": subject.PURPOSE,
        "audit_verdict": subject.AUDIT_VERDICT,
        "git_commit": "deadbeef",
        "source_hashes": dict(locked_sources),
        "input_paths": {
            "plan": str(plan_path.resolve()),
            "snapshot": str(snapshot.resolve()),
            "artifact": str(artifact.resolve()),
            "official_validator": str(subject.OFFICIAL_VALIDATOR.resolve()),
        },
        "input_hashes": {
            "plan": subject.file_sha256(plan_path),
            "snapshot": subject.file_sha256(snapshot),
            "artifact_manifest": subject.file_sha256(manifest),
            "official_validator": subject.file_sha256(
                subject.OFFICIAL_VALIDATOR
            ),
        },
        "runtime": locked_runtime,
        "output_root": plan["output_root"],
        "temp_root": plan["duckdb"]["temp_root"],
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    subject.validate_execution_lock(lock_path, plan_path=plan_path)
    if drift == "source":
        locked_sources["runner"] = "f" * 64
    else:
        lock["runtime"] = {"duckdb": "drift"}
        lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="locked environment changed"):
        subject.validate_execution_lock(lock_path, plan_path=plan_path)


def test_watched_inputs_reject_symlink_and_nonregular(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        subject.fingerprint(link)
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        subject.file_sha256(directory)
    with pytest.raises(ValueError, match="regular file"):
        subject.load_plan(link)
    lock_link = tmp_path / "lock-link"
    lock_link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        subject.validate_execution_lock(lock_link)


def test_toctou_and_rss_helpers(monkeypatch, tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"a")
    before = subject.fingerprint(path)
    path.write_bytes(b"b")
    with pytest.raises(ValueError, match="changed during audit"):
        subject.assert_unchanged(path, before)
    monkeypatch.setattr(subject.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=9 * 1024**3))
    assert subject.peak_rss_bytes() > 8 * 1024**3


def test_lock_is_mandatory_and_has_no_bypass():
    import inspect

    assert "execution_lock" in inspect.signature(subject.audit).parameters
    assert not any(
        "verify" in name or "bypass" in name
        for name in inspect.signature(subject.audit).parameters
    )
    with pytest.raises(SystemExit, match="execution-lock is required"):
        old = subject.sys.argv
        try:
            subject.sys.argv = ["audit_v412_snapshot_lookup.py"]
            subject.main()
        finally:
            subject.sys.argv = old


def test_no_builder_import():
    source = Path(subject.__file__).read_text()
    assert "import scripts.build_v412_snapshot_lookup" not in source
    assert "from scripts.build_v412_snapshot_lookup" not in source
