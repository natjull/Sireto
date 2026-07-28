from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import build_v412_unit_oracle as oracle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_bytes(oracle.canonical_json_bytes(value))


def _record(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return {
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
        "row_count": parquet.metadata.num_rows,
        "schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
        "metadata": None if schema.metadata is None else dict(schema.metadata),
    }


def _expected(ids, components, kinds, sirets, sirens):
    rows = []
    counts = {
        "threshold_dev": {"total": 0, "MATCH_EXACT": 0, "AMBIGUOUS": 0},
        "comparison_dev": {"total": 0, "MATCH_EXACT": 0, "AMBIGUOUS": 0},
    }
    for query_id, component, kind, siret, siren in zip(
        ids, components, kinds, sirets, sirens, strict=True
    ):
        part = (
            "threshold_dev"
            if hashlib.sha256(("v411-threshold:" + component).encode()).digest()[0] < 128
            else "comparison_dev"
        )
        rows.append((query_id, part, kind, siret, siren))
        counts[part]["total"] += 1
        counts[part][kind] += 1
    id_payload = b"".join(value.encode() + b"\n" for value in ids)
    truth = b"".join(
        b"\0".join(b"\\N" if value is None else value.encode() for value in row) + b"\n"
        for row in rows
    )
    return counts, id_payload, truth


def _restore_permissions(*roots: Path) -> None:
    for root in roots:
        if root.exists():
            os.chmod(root, 0o755)
            for path in root.rglob("*"):
                os.chmod(path, 0o755 if path.is_dir() else 0o644)


@pytest.fixture
def mini(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    ids_unsorted = ["q1", "q2", "q3"]
    ids = sorted(
        ids_unsorted,
        key=lambda value: (
            hashlib.sha256(("v412-unit-engine:" + value).encode()).hexdigest(),
            value.encode(),
        ),
    )
    by_id = {
        "q1": ("component-a", "MATCH_EXACT", "12345678900011", "123456789"),
        "q2": ("component-b", "AMBIGUOUS", None, None),
        "q3": ("component-c", "MATCH_EXACT", "98765432100019", "987654321"),
    }
    components = [by_id[value][0] for value in ids]
    kinds = [by_id[value][1] for value in ids]
    sirets = [by_id[value][2] for value in ids]
    sirens = [by_id[value][3] for value in ids]
    counts, id_payload, truth_payload = _expected(ids, components, kinds, sirets, sirens)

    safe_root = tmp_path / "safe_runtime" / ("a" * 64)
    safe_root.mkdir(parents=True)
    query_schema = pa.schema([pa.field("query_id", pa.string(), nullable=False)])
    safe_queries = safe_root / "queries_dev.parquet"
    pq.write_table(pa.Table.from_arrays([pa.array(ids)], schema=query_schema), safe_queries)
    for name in (
        "queries_all.parquet",
        "partition_inventory.parquet",
        "tfidf_inventory.parquet",
    ):
        pq.write_table(pa.table({"value": [name]}), safe_root / name)
    (safe_root / "integrity.json").write_bytes(b'{"sealed":true}\n')
    files = {
        name: _record(safe_root / name)
        for name in (
            "queries_all.parquet",
            "partition_inventory.parquet",
            "tfidf_inventory.parquet",
        )
    }
    files["integrity.json"] = {
        "sha256": _sha(safe_root / "integrity.json"),
        "size_bytes": (safe_root / "integrity.json").stat().st_size,
    }
    files["queries_dev.parquet"] = _record(safe_queries)
    safe_manifest = {
        "schema_version": "sireto-v4.12-unit-runtime-manifest-1",
        "build_id": safe_root.name,
        "files": files,
        "partition_inventory_sha256": "1" * 64,
        "tfidf_inventory_sha256": "2" * 64,
        "partition_runtime_signature": "3" * 64,
        "tfidf_config_artifact_hash": "4" * 64,
        "runtime": oracle._runtime(),
        "declarations": oracle.SAFE_DECLARATIONS,
    }
    safe_manifest_path = safe_root / "runtime_manifest.json"
    _json(safe_manifest_path, safe_manifest)

    labels_path = tmp_path / "labels.parquet"
    labels_schema = pa.schema(
        [
            pa.field("query_id", pa.string(), False),
            pa.field("label_kind", pa.string(), False),
            pa.field("ground_truth_siret", pa.string(), True),
            pa.field("ground_truth_siren", pa.string(), True),
            pa.field("forbidden_prediction", pa.string(), True),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(ids),
                pa.array(kinds),
                pa.array(sirets),
                pa.array(sirens),
                pa.array(["never-read"] * 3),
            ],
            schema=labels_schema,
        ),
        labels_path,
    )
    split_path = tmp_path / "split.parquet"
    split_schema = pa.schema(
        [
            pa.field("query_id", pa.string(), False),
            pa.field("siren_component_id", pa.string(), False),
            pa.field("split", pa.string(), False),
            pa.field("oof_fold", pa.int64(), False),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array(ids), pa.array(components), pa.array(["dev"] * 3), pa.array([1, 2, 3])],
            schema=split_schema,
        ),
        split_path,
    )
    output_root = tmp_path / "oracles"
    audit_root = tmp_path / "audits"
    temp_root = tmp_path / "temp"
    for root in (output_root, audit_root, temp_root):
        root.mkdir()
    source_a = tmp_path / "source_a.py"
    source_b = tmp_path / "source_b.py"
    source_a.write_text("A=1\n")
    source_b.write_text("B=2\n")
    forbidden = tmp_path / "candidates_sparse_top100.parquet"
    forbidden.write_bytes(b"must never be opened")
    plan_path = tmp_path / "oracle_plan.json"
    lock_path = tmp_path / "oracle_lock.json"
    plan = {
        "audit_output_root": str(audit_root),
        "execution_lock_path": str(lock_path),
        "expected": {
            "comparison_dev": counts["comparison_dev"],
            "first_query_ids": ids[:5],
            "last_query_ids": ids[-5:],
            "logical_payload_bytes": len(truth_payload),
            "logical_payload_sha256": hashlib.sha256(truth_payload).hexdigest(),
            "query_id_payload_bytes": len(id_payload),
            "query_id_payload_sha256": hashlib.sha256(id_payload).hexdigest(),
            "query_count": 3,
            "threshold_dev": counts["threshold_dev"],
        },
        "labels": {
            "path": str(labels_path),
            "projection": [
                "query_id",
                "label_kind",
                "ground_truth_siret",
                "ground_truth_siren",
            ],
            "row_count": 3,
            "sha256": _sha(labels_path),
            "size_bytes": labels_path.stat().st_size,
        },
        "forbidden_artifacts": [{"path": str(forbidden), "sha256": _sha(forbidden)}],
        "max_rss_bytes": 8 * 1024**3,
        "output_root": str(output_root),
        "runtime": oracle._runtime(),
        "safe_input": {
            "build_id": safe_root.name,
            "manifest_path": str(safe_manifest_path),
            "manifest_sha256": _sha(safe_manifest_path),
            "queries_dev_path": str(safe_queries),
            "queries_dev_projection": ["query_id"],
            "queries_dev_sha256": _sha(safe_queries),
            "queries_dev_size_bytes": safe_queries.stat().st_size,
        },
        "schema_version": "sireto-v4.12-unit-oracle-plan-1",
        "sources": [str(source_a), str(source_b)],
        "split_assignments": {
            "path": str(split_path),
            "projection": ["query_id", "siren_component_id", "split"],
            "row_count": 3,
            "sha256": _sha(split_path),
            "size_bytes": split_path.stat().st_size,
        },
        "temp_root": str(temp_root),
    }
    _json(plan_path, plan)
    lock = {
        "schema_version": oracle.LOCK_SCHEMA,
        "purpose": oracle.LOCK_PURPOSE,
        "audit_verdict": oracle.LOCK_VERDICT,
        "git_commit": "f" * 40,
        "source_hashes": {str(source_a): _sha(source_a), str(source_b): _sha(source_b)},
        "input_paths": oracle._expected_input_paths(plan),
        "input_hashes": oracle._expected_input_hashes(plan),
        "safe_input_build_id": safe_root.name,
        "expected_population": oracle._expected_population(plan),
        "expected_id_payload_sha256": plan["expected"]["query_id_payload_sha256"],
        "expected_truth_logical_sha256": plan["expected"]["logical_payload_sha256"],
        "runtime": oracle._runtime(),
        "output_root": str(output_root),
        "audit_output_root": str(audit_root),
        "temp_root": str(temp_root),
        "max_rss_bytes": 8 * 1024**3,
    }
    _json(lock_path, lock)
    staging = temp_root / "run"
    staging.mkdir()
    (staging / "tmp").mkdir()
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("SIRETO_V412_ORACLE_STAGING", str(staging))
    monkeypatch.setenv("TMPDIR", str(staging / "tmp"))
    monkeypatch.setattr(oracle.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(oracle, "CANONICAL_PLAN_PATH", plan_path)

    def fake_sources(repo, sources, expected, commit, max_rss):
        return {Path(path): oracle._snapshot(Path(path), max_rss) for path in sources}

    monkeypatch.setattr(oracle, "_verify_git_sources", fake_sources)
    return {
        "plan": plan,
        "plan_path": plan_path,
        "lock_path": lock_path,
        "safe_root": safe_root,
        "safe_queries": safe_queries,
        "safe_manifest": safe_manifest_path,
        "labels": labels_path,
        "split": split_path,
        "forbidden": forbidden,
        "output_root": output_root,
        "audit_root": audit_root,
    }


def test_projection_is_physically_limited(mini, monkeypatch):
    seen = []
    original = pq.ParquetFile.iter_batches

    def recording(self, *args, **kwargs):
        seen.append(kwargs["columns"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", recording)
    table, _, _, _ = oracle.build_oracle_table(
        mini["safe_queries"], mini["labels"], mini["split"], mini["plan"]
    )
    assert seen == [
        ["query_id"],
        ["query_id", "label_kind", "ground_truth_siret", "ground_truth_siren"],
        ["query_id", "siren_component_id", "split"],
    ]
    assert table.schema == oracle.ORACLE_SCHEMA
    assert "forbidden_prediction" not in table.column_names


def test_complete_mini_publication(mini):
    oracle_dir, audit_dir = oracle.build_oracle(mini["plan_path"], mini["lock_path"])
    oracle.validate_concordance(
        oracle_dir, audit_dir, mini["plan_path"], mini["lock_path"]
    )
    assert {p.name for p in oracle_dir.iterdir()} == oracle.ORACLE_FILES
    assert {p.name for p in audit_dir.iterdir()} == oracle.AUDIT_FILES
    ledger = pq.read_table(audit_dir / "data_inputs.parquet")
    assert ledger.num_rows == 8
    assert set(ledger.column("role").to_pylist()) == oracle.LEDGER_ROLES
    manifest = oracle.load_json_strict(oracle_dir / "manifest.json")
    assert manifest["declarations"] == {
        "safe_runtime_files_opened_for_integrity": True,
        "retrieval_results_opened": False,
        "candidate_results_opened": False,
        "direct_evidence_opened": False,
        "guard_decisions_opened": False,
        "models_opened": False,
        "challenge_or_final_opened": False,
    }
    for root in (oracle_dir, audit_dir):
        assert not (root.stat().st_mode & stat.S_IWUSR)
    _restore_permissions(oracle_dir, audit_dir)


@pytest.mark.parametrize(
    "kind,siret,siren,message",
    [
        ("MATCH_EXACT", "123", "123456789", "SIRET"),
        ("MATCH_EXACT", "12345678900011", "000000000", "SIRET"),
        ("AMBIGUOUS", "12345678900011", None, "AMBIGUOUS"),
        ("NO_MATCH", None, None, "label_kind"),
    ],
)
def test_truth_rules_are_strict(mini, kind, siret, siren, message):
    table = pq.read_table(mini["labels"])
    values = {name: table.column(name).to_pylist() for name in table.column_names}
    values["label_kind"][0] = kind
    values["ground_truth_siret"][0] = siret
    values["ground_truth_siren"][0] = siren
    pq.write_table(pa.table(values), mini["labels"])
    with pytest.raises(oracle.BuildStopped, match=message):
        oracle.build_oracle_table(mini["safe_queries"], mini["labels"], mini["split"], mini["plan"])


def test_safe_manifest_fails_before_sensitive_sources_open(mini, monkeypatch):
    manifest = json.loads(mini["safe_manifest"].read_text())
    manifest["build_id"] = "0" * 64
    _json(mini["safe_manifest"], manifest)
    opened = []
    original = oracle._read_projection

    def recording(path, *args, **kwargs):
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(oracle, "_read_projection", recording)
    with pytest.raises(oracle.BuildStopped):
        oracle.build_oracle(mini["plan_path"], mini["lock_path"])
    assert mini["labels"] not in opened
    assert mini["split"] not in opened


def test_forbidden_artifact_is_never_opened(mini, monkeypatch):
    seen = []
    original = oracle._snapshot

    def recording(path, limit):
        seen.append(Path(path))
        return original(path, limit)

    monkeypatch.setattr(oracle, "_snapshot", recording)
    oracle_dir, audit_dir = oracle.build_oracle(mini["plan_path"], mini["lock_path"])
    assert mini["forbidden"] not in seen
    _restore_permissions(oracle_dir, audit_dir)


def test_safe_inventories_are_never_used_as_truth_inputs(mini, monkeypatch):
    projected = []
    original = oracle._read_projection

    def recording(path, *args, **kwargs):
        projected.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(oracle, "_read_projection", recording)
    oracle_dir, audit_dir = oracle.build_oracle(mini["plan_path"], mini["lock_path"])
    assert set(projected) == {
        mini["safe_queries"],
        mini["labels"],
        mini["split"],
    }
    assert mini["safe_root"] / "partition_inventory.parquet" not in projected
    assert mini["safe_root"] / "tfidf_inventory.parquet" not in projected
    assert mini["safe_root"] / "queries_all.parquet" not in projected
    _restore_permissions(oracle_dir, audit_dir)


def test_lock_refuses_forbidden_artifact(mini):
    lock = oracle.load_json_strict(mini["lock_path"])
    lock["input_paths"]["labels"] = str(mini["forbidden"])
    with pytest.raises(oracle.BuildStopped, match="input paths|forbidden"):
        oracle.validate_lock(lock, mini["plan"], Path.cwd(), 8 * 1024**3)


def test_lock_keyset_and_verdict_are_exact(mini):
    lock = oracle.load_json_strict(mini["lock_path"])
    lock["bypass"] = True
    with pytest.raises(oracle.BuildStopped, match="keyset"):
        oracle.validate_lock(lock, mini["plan"], Path.cwd(), 8 * 1024**3)
    lock.pop("bypass")
    lock["audit_verdict"] = "GO_CONTRACT"
    with pytest.raises(oracle.BuildStopped, match="audit verdict"):
        oracle.validate_lock(lock, mini["plan"], Path.cwd(), 8 * 1024**3)


def test_git_blob_and_worktree_both_must_match(tmp_path, monkeypatch):
    source = tmp_path / "tracked.py"
    source.write_bytes(b"dirty\n")
    expected = hashlib.sha256(b"committed\n").hexdigest()
    monkeypatch.setattr(oracle, "_git", lambda _args: "tracked.py")
    monkeypatch.setattr(oracle, "_git_bytes", lambda _args: b"committed\n")
    with pytest.raises(oracle.BuildStopped, match="worktree"):
        oracle._verify_git_sources(
            tmp_path, ["tracked.py"], {"tracked.py": expected}, "a" * 40, 1024**3
        )


def test_direct_arbitrary_plan_is_rejected(mini, monkeypatch):
    monkeypatch.setattr(
        oracle, "CANONICAL_PLAN_PATH", Path("config/v4_12_unit_oracle_plan.json")
    )
    with pytest.raises(oracle.BuildStopped, match="canonical"):
        oracle.build_oracle(mini["plan_path"], mini["lock_path"])


def test_process_and_rss_guards(mini, monkeypatch):
    monkeypatch.setattr(oracle.sys, "dont_write_bytecode", False)
    with pytest.raises(oracle.BuildStopped, match="python -B"):
        oracle._process_context(mini["plan"])
    monkeypatch.setattr(oracle, "_rss_bytes", lambda: 101)
    with pytest.raises(oracle.BuildStopped, match="RSS"):
        oracle._check_rss(100)


def test_source_swap_then_restore_is_detected(mini, monkeypatch):
    original = oracle.build_oracle_table
    original_bytes = mini["labels"].read_bytes()
    calls = 0

    def swapping(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            table = pq.read_table(mini["labels"])
            values = {name: table.column(name).to_pylist() for name in table.column_names}
            values["ground_truth_siret"][0] = "11111111111111"
            values["ground_truth_siren"][0] = "111111111"
            pq.write_table(pa.table(values), mini["labels"])
            try:
                return original(*args, **kwargs)
            finally:
                mini["labels"].write_bytes(original_bytes)
        return original(*args, **kwargs)

    monkeypatch.setattr(oracle, "build_oracle_table", swapping)
    with pytest.raises(oracle.BuildStopped):
        oracle.build_oracle(mini["plan_path"], mini["lock_path"])


def test_resealed_oracle_manifest_does_not_hide_parquet_mutation(mini):
    oracle_dir, audit_dir = oracle.build_oracle(mini["plan_path"], mini["lock_path"])
    os.chmod(oracle_dir, 0o755)
    parquet_path = oracle_dir / "oracle_dev.parquet"
    manifest_path = oracle_dir / "manifest.json"
    os.chmod(parquet_path, 0o644)
    os.chmod(manifest_path, 0o644)
    table = pq.read_table(parquet_path)
    values = {name: table.column(name).to_pylist() for name in table.column_names}
    values["ground_truth_siret"][0] = "11111111111111"
    values["ground_truth_siren"][0] = "111111111"
    pq.write_table(pa.Table.from_arrays(
        [pa.array(values[field.name], type=field.type) for field in oracle.ORACLE_SCHEMA],
        schema=oracle.ORACLE_SCHEMA,
    ), parquet_path)
    changed = pq.read_table(parquet_path).to_pylist()
    truth_payload = b"".join(
        b"\0".join(
            b"\\N" if row[name] is None else row[name].encode()
            for name in oracle.ORACLE_SCHEMA.names
        )
        + b"\n"
        for row in changed
    )
    truth_sha = hashlib.sha256(truth_payload).hexdigest()
    integrity_path = oracle_dir / "integrity.json"
    os.chmod(integrity_path, 0o644)
    integrity = oracle.load_json_strict(integrity_path)
    integrity["truth_logical_sha256"] = truth_sha
    _json(integrity_path, integrity)
    manifest = oracle.load_json_strict(manifest_path)
    manifest["files"]["oracle_dev.parquet"] = oracle._parquet_record(parquet_path)
    manifest["files"]["integrity.json"] = {
        "sha256": _sha(integrity_path),
        "size_bytes": integrity_path.stat().st_size,
    }
    manifest["truth_logical_sha256"] = truth_sha
    _json(manifest_path, manifest)
    os.chmod(audit_dir, 0o755)
    provenance_path = audit_dir / "provenance.json"
    audit_manifest_path = audit_dir / "manifest.json"
    os.chmod(provenance_path, 0o644)
    os.chmod(audit_manifest_path, 0o644)
    provenance = oracle.load_json_strict(provenance_path)
    provenance["oracle_manifest_sha256"] = _sha(manifest_path)
    _json(provenance_path, provenance)
    audit_manifest = oracle.load_json_strict(audit_manifest_path)
    audit_manifest["files"]["provenance.json"] = {
        "sha256": _sha(provenance_path),
        "size_bytes": provenance_path.stat().st_size,
    }
    _json(audit_manifest_path, audit_manifest)
    with pytest.raises(oracle.BuildStopped, match="external truth"):
        oracle.validate_concordance(oracle_dir, audit_dir, mini["plan_path"], mini["lock_path"])
    _restore_permissions(oracle_dir, audit_dir)


@pytest.mark.parametrize("target", ["plan", "lock"])
def test_plan_or_lock_swap_then_restore_is_detected(mini, monkeypatch, target):
    target_path = mini["plan_path"] if target == "plan" else mini["lock_path"]
    original_bytes = target_path.read_bytes()
    original_loader = oracle.load_json_strict
    calls = 0

    def swapping_loader(path):
        nonlocal calls
        if Path(path) == target_path:
            calls += 1
            if calls == 2:
                value = json.loads(original_bytes)
                if target == "plan":
                    value["expected"]["query_count"] += 1
                else:
                    value["expected_id_payload_sha256"] = "0" * 64
                _json(target_path, value)
                try:
                    return original_loader(path)
                finally:
                    target_path.write_bytes(original_bytes)
        return original_loader(path)

    monkeypatch.setattr(oracle, "load_json_strict", swapping_loader)
    with pytest.raises(oracle.BuildStopped, match="changed while"):
        oracle.build_oracle(mini["plan_path"], mini["lock_path"])


def test_plan_schema_and_keyset_are_exact(mini):
    changed = dict(mini["plan"])
    changed["unexpected"] = True
    with pytest.raises(oracle.BuildStopped, match="plan keyset"):
        oracle.validate_plan(changed)


def test_safe_runtime_siblings_remain_unchanged(mini):
    before = {path.name: _sha(path) for path in mini["safe_root"].iterdir()}
    oracle_dir, audit_dir = oracle.build_oracle(mini["plan_path"], mini["lock_path"])
    after = {path.name: _sha(path) for path in mini["safe_root"].iterdir()}
    assert after == before
    _restore_permissions(oracle_dir, audit_dir)


def test_oracle_extra_file_and_audit_ledger_mutation_stop(mini):
    oracle_dir, audit_dir = oracle.build_oracle(mini["plan_path"], mini["lock_path"])
    os.chmod(oracle_dir, 0o755)
    (oracle_dir / "extra").write_text("forbidden")
    with pytest.raises(oracle.BuildStopped, match="file-set"):
        oracle.validate_concordance(
            oracle_dir, audit_dir, mini["plan_path"], mini["lock_path"]
        )
    (oracle_dir / "extra").unlink()
    os.chmod(audit_dir, 0o755)
    ledger = audit_dir / "data_inputs.parquet"
    audit_manifest = audit_dir / "manifest.json"
    os.chmod(ledger, 0o644)
    os.chmod(audit_manifest, 0o644)
    pq.write_table(pa.table({"wrong": ["schema"]}), ledger)
    manifest = oracle.load_json_strict(audit_manifest)
    manifest["files"]["data_inputs.parquet"] = {
        "sha256": _sha(ledger),
        "size_bytes": ledger.stat().st_size,
    }
    _json(audit_manifest, manifest)
    with pytest.raises(oracle.BuildStopped, match="ledger schema"):
        oracle.validate_concordance(
            oracle_dir, audit_dir, mini["plan_path"], mini["lock_path"]
        )
    _restore_permissions(oracle_dir, audit_dir)


def test_reordered_ledger_resealed_in_audit_manifest_is_stop(mini):
    oracle_dir, audit_dir = oracle.build_oracle(mini["plan_path"], mini["lock_path"])
    os.chmod(audit_dir, 0o755)
    ledger_path = audit_dir / "data_inputs.parquet"
    manifest_path = audit_dir / "manifest.json"
    os.chmod(ledger_path, 0o644)
    os.chmod(manifest_path, 0o644)
    ledger = pq.read_table(ledger_path)
    permutation = pa.array(list(reversed(range(ledger.num_rows))), type=pa.int64())
    pq.write_table(ledger.take(permutation), ledger_path)
    manifest = oracle.load_json_strict(manifest_path)
    manifest["files"]["data_inputs.parquet"] = {
        "sha256": _sha(ledger_path),
        "size_bytes": ledger_path.stat().st_size,
    }
    _json(manifest_path, manifest)
    with pytest.raises(oracle.BuildStopped, match="ledger differs"):
        oracle.validate_concordance(
            oracle_dir, audit_dir, mini["plan_path"], mini["lock_path"]
        )
    _restore_permissions(oracle_dir, audit_dir)


def test_audit_is_promoted_before_oracle(mini, monkeypatch):
    parents = []
    original = oracle._promote

    def recording(source, destination):
        parents.append(destination.parent)
        return original(source, destination)

    monkeypatch.setattr(oracle, "_promote", recording)
    oracle_dir, audit_dir = oracle.build_oracle(mini["plan_path"], mini["lock_path"])
    assert parents == [Path(mini["plan"]["audit_output_root"]), Path(mini["plan"]["output_root"])]
    _restore_permissions(oracle_dir, audit_dir)
