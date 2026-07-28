from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_v412_direct_evidence as subject


def _queries(count: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": f"q{index}",
                "crm_name": f"SOCIETE {index}",
                "crm_address": f"{index + 1} RUE ALPHA",
                "crm_postcode": "75001",
                "crm_city": "PARIS",
                "crm_insee": "75056",
            }
            for index in range(count)
        ],
        columns=subject.PRESEAL_QUERY_COLUMNS,
    )


def _raw(query_id: str, siret: str) -> dict:
    return {
        "query_id": query_id,
        "candidate_siret": siret,
        "candidate_siren": siret[:9],
        "candidate_state": "A",
        "exact_name_anchor": True,
        "exact_address_anchor": False,
        "direct_evidence_class": "NAME_AND_ADDRESS",
    }


def test_repository_allowlist_and_denylist_are_strictly_understood():
    _, query_path, query_hash, partitions, snapshot = subject.validate_allowlist(
        subject.DEFAULT_ALLOWLIST
    )
    roots, hashes = subject.validate_denylist(subject.DEFAULT_DENYLIST)
    assert query_path.name == "queries.parquet"
    assert len(query_hash) == 64
    assert partitions.name == "candidates_v7_all"
    assert snapshot.name == "StockEtablissement_utf8.parquet"
    assert len(roots) == 3
    assert len(hashes) >= 10


def test_execution_lock_source_set_contains_policy_import_closure():
    assert {
        "scripts/build_benchmark_v4_current_snapshot.py",
        "scripts/build_benchmark_v3_evidence.py",
        "scripts/build_benchmark_v2_qualification.py",
        "scripts/run_v9_retrieval_experiment.py",
        "scripts/freeze_v9_closed_benchmark.py",
        "src/xgb_matcher/retrieval.py",
        "src/xgb_matcher/candidates.py",
        "src/xgb_matcher/fusion.py",
        "src/xgb_matcher/v9_dataset.py",
        "src/xgb_matcher/v9_features.py",
    }.issubset(subject.SOURCE_PATHS)
    package_sources = {
        str(path)
        for path in Path("src/xgb_matcher").glob("*.py")
    }
    assert package_sources.issubset(subject.SOURCE_PATHS)


def test_runtime_pins_numeric_and_similarity_stack():
    runtime = subject._runtime()
    assert {
        "numpy",
        "rapidfuzz",
        "scikit-learn",
        "scipy",
        "duckdb",
    }.issubset(runtime)


def test_publishing_build_cannot_disable_git_verification():
    assert "verify_git" not in inspect.signature(
        subject.build_artifact
    ).parameters


def test_allowlist_rejects_an_extra_input(tmp_path):
    policy = json.loads(subject.DEFAULT_ALLOWLIST.read_text())
    policy["artifacts"][0]["files"]["labels.parquet"] = "a" * 64
    policy["artifacts"][0]["phases"]["labels.parquet"] = "PRE_EVIDENCE_SEAL"
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="artifact files changed"):
        subject.validate_allowlist(path)


def test_allowlist_structure_does_not_open_postseal_artifacts(tmp_path):
    policy = json.loads(subject.DEFAULT_ALLOWLIST.read_text())
    for artifact in policy["artifacts"]:
        artifact["root"] = str(tmp_path / f"absent-{artifact['role']}")
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps(policy))
    _, query_path, _, _, _ = subject.validate_allowlist(path)
    assert not query_path.exists()


def test_query_projection_rejects_extra_model_or_label_column():
    assert len(subject.validate_preseal_queries(_queries(), expected_count=2)) == 2
    frame = _queries()
    frame["ranker_score"] = 1.0
    with pytest.raises(ValueError, match="projection changed"):
        subject.validate_preseal_queries(frame, expected_count=2)


def test_query_file_is_rehashed_immediately_after_read(
    tmp_path, monkeypatch
):
    path = tmp_path / "queries.parquet"
    _queries().to_parquet(path, index=False)
    expected = subject.file_sha256(path)
    original = pd.read_parquet

    def mutate_after_read(source, *, columns):
        frame = original(source, columns=columns)
        path.write_bytes(b"mutated")
        return frame

    monkeypatch.setattr(subject.pd, "read_parquet", mutate_after_read)
    with pytest.raises(ValueError, match="query changed during read"):
        subject.load_preseal_queries(
            path,
            expected_sha256=expected,
            expected_count=2,
        )


def test_denylist_rejects_original_path_and_relocated_hash(tmp_path):
    forbidden_root = tmp_path / "challenge"
    forbidden_root.mkdir()
    original = forbidden_root / "labels.parquet"
    original.write_bytes(b"consumed")
    digest = subject.file_sha256(original)
    with pytest.raises(ValueError, match="forbidden path"):
        subject.validate_inputs_against_denylist(
            [original],
            forbidden_roots={forbidden_root.resolve()},
            forbidden_hashes=set(),
        )
    copied = tmp_path / "innocent-name.bin"
    copied.write_bytes(b"consumed")
    with pytest.raises(ValueError, match="forbidden hash"):
        subject.validate_inputs_against_denylist(
            [copied],
            forbidden_roots=set(),
            forbidden_hashes={digest},
        )


def test_compute_uses_complete_partition_and_frozen_function(monkeypatch):
    queries = _queries()
    monkeypatch.setattr(subject, "PartitionedCandidateStore", lambda path: object())
    monkeypatch.setattr(
        subject, "_planned_partition_key", lambda row, store: "insee:75056"
    )
    complete_partition = [{"siret": str(index)} for index in range(250)]
    monkeypatch.setattr(
        subject, "_load_partition", lambda key, store: complete_partition
    )
    monkeypatch.setattr(
        subject,
        "build_active_partition_index",
        lambda rows: subject.ActivePartitionIndex({}, {}, 250, 250),
    )
    seen = []

    def fake_find(row, index, *, partition_key):
        seen.append(
            (row["postcode"], row["insee"], row["split"], index.active_count)
        )
        if row["query_id"] == "q0":
            return [_raw("q0", f"123456789{index:05d}") for index in range(105)]
        return []

    monkeypatch.setattr(subject, "find_direct_active_candidates", fake_find)
    query_evidence, candidate_evidence, timing = subject.compute_direct_evidence(
        queries, partitions_path=Path("unused")
    )
    assert seen == [
        ("75001", "75056", "v412_label_free", 250),
        ("75001", "75056", "v412_label_free", 250),
    ]
    assert query_evidence.loc[0, "direct_candidate_count"] == 105
    assert len(candidate_evidence) == 105
    assert timing["query_count"] == 2


def test_frozen_storage_hashes_snapshot_without_deserializing(tmp_path):
    partitions = tmp_path / "partitions"
    partitions.mkdir()
    (partitions / "part.parquet").write_bytes(b"partition")
    snapshot = tmp_path / "snapshot.parquet"
    snapshot.write_bytes(b"snapshot")
    subject.validate_frozen_storage(
        partitions_path=partitions,
        snapshot_path=snapshot,
        expected_partitions_signature=subject._path_signature(partitions),
        expected_snapshot_sha256=subject.file_sha256(snapshot),
    )
    snapshot.write_bytes(b"changed")
    with pytest.raises(ValueError, match="snapshot hash changed"):
        subject.validate_frozen_storage(
            partitions_path=partitions,
            snapshot_path=snapshot,
            expected_partitions_signature=subject._path_signature(partitions),
            expected_snapshot_sha256="0" * 64,
        )


def test_compute_rechecks_storage_after_long_calculation(
    tmp_path, monkeypatch
):
    partitions = tmp_path / "partitions"
    partitions.mkdir()
    part = partitions / "part.parquet"
    part.write_bytes(b"before")
    snapshot = tmp_path / "snapshot.parquet"
    snapshot.write_bytes(b"snapshot")
    expected_partitions = subject._path_signature(partitions)
    expected_snapshot = subject.file_sha256(snapshot)

    def mutate_during_compute(queries, *, partitions_path):
        part.write_bytes(b"after")
        return pd.DataFrame(), pd.DataFrame(), {}

    monkeypatch.setattr(
        subject, "compute_direct_evidence", mutate_during_compute
    )
    with pytest.raises(ValueError, match="partition signature changed"):
        subject.compute_direct_evidence_with_recheck(
            _queries(),
            partitions_path=partitions,
            snapshot_path=snapshot,
            expected_partitions_signature=expected_partitions,
            expected_snapshot_sha256=expected_snapshot,
        )


def test_peak_rss_gate_is_blocking(monkeypatch):
    timing = {"peak_rss_limit_bytes": 8 * 1024**3}
    monkeypatch.setattr(
        subject, "_peak_rss_bytes", lambda: 8 * 1024**3 + 1
    )
    with pytest.raises(ValueError, match="peak RSS exceeds"):
        subject.refresh_and_validate_peak_rss(timing)


def _lock(tmp_path: Path) -> tuple[Path, Path, Path]:
    allowlist = tmp_path / "allow.json"
    denylist = tmp_path / "deny.json"
    allowlist.write_text("{}")
    denylist.write_text("{}")
    repo_root = Path(subject.__file__).resolve().parent.parent
    lock = {
        "schema_version": subject.LOCK_SCHEMA_VERSION,
        "purpose": subject.PURPOSE,
        "audit_verdict": subject.AUDIT_VERDICT,
        "git_commit": "fixture",
        "source_hashes": subject._source_hashes(repo_root),
        "input_paths": {
            "allowlist": str(allowlist.resolve()),
            "denylist": str(denylist.resolve()),
            "queries": "/fixture/queries.parquet",
            "partitions": "/fixture/partitions",
            "snapshot": "/fixture/snapshot.parquet",
        },
        "input_hashes": {
            "allowlist": subject.file_sha256(allowlist),
            "denylist": subject.file_sha256(denylist),
            "queries": "1" * 64,
            "partitions": subject.EXPECTED_PARTITIONS_SIGNATURE,
            "snapshot": subject.EXPECTED_SNAPSHOT_SHA256,
        },
        "runtime": subject._runtime(),
        "snapshot_sha256": subject.EXPECTED_SNAPSHOT_SHA256,
        "partitions_signature": subject.EXPECTED_PARTITIONS_SIGNATURE,
    }
    path = tmp_path / "execution-lock.json"
    path.write_text(json.dumps(lock))
    return path, allowlist, denylist


def test_execution_lock_accepts_exact_fixture_and_rejects_extra_field(tmp_path):
    lock_path, allowlist, denylist = _lock(tmp_path)
    lock, digest = subject.validate_execution_lock(
        lock_path,
        allowlist_path=allowlist,
        denylist_path=denylist,
        verify_git=False,
    )
    assert lock["audit_verdict"] == "GO_BUILD_V412_EVIDENCE"
    assert len(digest) == 64
    lock["unexpected"] = True
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="fields changed"):
        subject.validate_execution_lock(
            lock_path,
            allowlist_path=allowlist,
            denylist_path=denylist,
            verify_git=False,
        )


def test_execution_lock_rejects_absent_independent_go(tmp_path):
    lock_path, allowlist, denylist = _lock(tmp_path)
    lock = json.loads(lock_path.read_text())
    lock["audit_verdict"] = "PENDING"
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="not independently authorized"):
        subject.validate_execution_lock(
            lock_path,
            allowlist_path=allowlist,
            denylist_path=denylist,
            verify_git=False,
        )


def _valid_artifact(tmp_path: Path) -> Path:
    candidate = subject.candidate_evidence_record(
        "q0", _raw("q0", "12345678900001")
    )
    query = subject.query_evidence_record(
        query_id="q0",
        partition_key="insee:75056",
        active_universe_count=10,
        candidates=[candidate],
    )
    queries, candidates = subject.build_evidence_frames(
        [query], [candidate]
    )
    identity = {
        "schema_version": subject.SCHEMA_VERSION,
        "execution_lock_sha256": "1" * 64,
        "source_hashes": {"fixture.py": "2" * 64},
        "query_sha256": "3" * 64,
        "query_count": 1,
        "partitions_signature": subject.EXPECTED_PARTITIONS_SIGNATURE,
        "snapshot_sha256": subject.EXPECTED_SNAPSHOT_SHA256,
        "policy_version": subject.POLICY_VERSION,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    root = tmp_path / build_id
    root.mkdir()
    queries.to_parquet(root / "query_evidence.parquet", index=False)
    candidates.to_parquet(root / "candidate_evidence.parquet", index=False)
    evidence_hashes = {
        name: subject.file_sha256(root / name)
        for name in (
            "query_evidence.parquet",
            "candidate_evidence.parquet",
        )
    }
    integrity = {
        "query_count": 1,
        "candidate_count": 1,
        "max_direct_candidate_count": 1,
        "query_ids_unique": True,
        "candidate_sirets_unique_per_query": True,
        "evidence_references_bijective": True,
        "active_candidates_only": True,
        "full_partition_universe": True,
        "ranker_pool_opened": False,
        "ranker_pool_modified": False,
        "retrieval_candidate_cap": 100,
        "labels_opened_before_seal": False,
        "split_opened_before_seal": False,
        "scenes_opened_before_seal": False,
        "models_opened_before_seal": False,
        "challenge_opened": False,
        "sealed_evidence_hashes": evidence_hashes,
    }
    timing = {
        "query_count": 1,
        "total_evidence_ms": 1.0,
        "amortized_batch_per_query_ms": {
            "p50": 1.0,
            "p95": 1.0,
            "max": 1.0,
        },
        "serve_latency_gate_eligible": False,
        "peak_rss_bytes": 1,
        "peak_rss_limit_bytes": 8 * 1024**3,
    }
    subject._write_json(root / "integrity.json", integrity)
    subject._write_json(root / "timing.json", timing)
    outputs = {}
    for filename in subject.OUTPUT_FILENAMES:
        path = root / filename
        outputs[filename] = {
            "sha256": subject.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    outputs["query_evidence.parquet"].update(
        row_count=1,
        columns=subject.QUERY_EVIDENCE_COLUMNS,
    )
    outputs["candidate_evidence.parquet"].update(
        row_count=1,
        columns=subject.CANDIDATE_EVIDENCE_COLUMNS,
    )
    subject._write_json(
        root / "manifest.json",
        {
            **identity,
            "build_id": build_id,
            "outputs": outputs,
        },
    )
    subject.validate_artifact(root)
    return root


def _refresh_output_declaration(root: Path, filename: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = root / filename
    manifest["outputs"][filename]["sha256"] = subject.file_sha256(path)
    manifest["outputs"][filename]["size_bytes"] = path.stat().st_size
    subject._write_json(manifest_path, manifest)


def test_artifact_rejects_extra_file(tmp_path):
    root = _valid_artifact(tmp_path)
    (root / "extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected file set"):
        subject.validate_artifact(root)


def test_artifact_rejects_nested_directory_or_symlink(tmp_path):
    root = _valid_artifact(tmp_path)
    nested = root / "hidden"
    nested.mkdir()
    (nested / "payload.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected file set"):
        subject.validate_artifact(root)


def test_artifact_rejects_false_row_declaration(tmp_path):
    root = _valid_artifact(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["query_evidence.parquet"]["row_count"] = 99
    subject._write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="row declarations changed"):
        subject.validate_artifact(root)


def test_artifact_rejects_false_integrity_seal(tmp_path):
    root = _valid_artifact(tmp_path)
    integrity_path = root / "integrity.json"
    integrity = json.loads(integrity_path.read_text())
    integrity["sealed_evidence_hashes"]["query_evidence.parquet"] = "0" * 64
    subject._write_json(integrity_path, integrity)
    _refresh_output_declaration(root, "integrity.json")
    with pytest.raises(ValueError, match="integrity declarations changed"):
        subject.validate_artifact(root)


def test_artifact_rejects_false_timing_or_rss(tmp_path):
    root = _valid_artifact(tmp_path)
    timing_path = root / "timing.json"
    timing = json.loads(timing_path.read_text())
    timing["peak_rss_bytes"] = timing["peak_rss_limit_bytes"] + 1
    subject._write_json(timing_path, timing)
    _refresh_output_declaration(root, "timing.json")
    with pytest.raises(ValueError, match="timing declarations changed"):
        subject.validate_artifact(root)


def test_artifact_rejects_malformed_amortized_timing(tmp_path):
    root = _valid_artifact(tmp_path)
    timing_path = root / "timing.json"
    timing = json.loads(timing_path.read_text())
    timing["amortized_batch_per_query_ms"] = {}
    subject._write_json(timing_path, timing)
    _refresh_output_declaration(root, "timing.json")
    with pytest.raises(ValueError, match="timing declarations changed"):
        subject.validate_artifact(root)
