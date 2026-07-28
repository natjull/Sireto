from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import build_v412_unit_inputs as unit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _query_spec(table: pa.Table, namespace: str) -> dict[str, object]:
    order = unit._query_order(table, namespace)
    ids = [table.column("query_id")[index].as_py() for index in order]
    payload = "".join(value + "\n" for value in ids).encode()
    return {
        "namespace": namespace,
        "query_count": len(ids),
        "payload_lf_bytes": len(payload),
        "payload_lf_sha256": hashlib.sha256(payload).hexdigest(),
        "first_query_ids": ids[:5],
        "last_query_ids": ids[-5:],
    }


def _raw_tree_hash(root: Path) -> tuple[int, int, str, str]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().encode(),
    )
    digest = hashlib.sha256()
    historical = hashlib.sha256()
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        sha = _sha(path)
        total += size
        digest.update(
            relative.encode() + b"\0" + str(size).encode() + b"\0" + sha.encode() + b"\n"
        )
        historical.update(relative.encode() + sha.encode())
    return len(files), total, digest.hexdigest(), historical.hexdigest()


def _raw_cache_hash(root: Path) -> str:
    records = []
    for pickle_path in root.glob("*.pkl"):
        sidecar_path = root / f"{pickle_path.name}.sha256.json"
        sidecar = json.loads(sidecar_path.read_text())
        records.append(
            (
                sidecar["partition_key"],
                pickle_path.name,
                pickle_path.stat().st_size,
                _sha(pickle_path),
                sidecar_path.name,
                sidecar_path.stat().st_size,
                _sha(sidecar_path),
            )
        )
    records.sort(key=lambda row: (row[0].encode(), row[1].encode(), row[4].encode()))
    digest = hashlib.sha256()
    for row in records:
        digest.update(b"\0".join(str(value).encode() for value in row) + b"\n")
    return digest.hexdigest()


@pytest.fixture
def mini(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    query_path = tmp_path / "queries.parquet"
    split_path = tmp_path / "split.parquet"
    query_source = pa.table(
        {
            "query_id": ["q1", "q2", "q3"],
            "crm_record_id": ["secret1", "secret2", "secret3"],
            "crm_name": ["Alpha", "Bêta", "Gamma"],
            "crm_address": ["1 rue A", "2 rue B", "3 rue C"],
            "crm_postcode": ["75001", "69001", "13001"],
            "crm_city": ["Paris", "Lyon", "Marseille"],
            "crm_insee": ["75056", "69123", "13055"],
            "crm_name_norm": ["forbidden", "forbidden", "forbidden"],
        }
    )
    split_source = pa.table(
        {
            "query_id": ["q1", "q2", "q3"],
            "siren_component_id": ["secret", "secret", "secret"],
            "split": ["dev", "fit", "dev"],
            "oof_fold": [1, 2, 3],
        }
    )
    pq.write_table(query_source, query_path)
    pq.write_table(split_source, split_path)

    partition_root = tmp_path / "partitions"
    (partition_root / "insee").mkdir(parents=True)
    (partition_root / "manifest").mkdir()
    pq.write_table(pa.table({"siret": ["1", "2"]}), partition_root / "insee" / "a.parquet")
    pq.write_table(pa.table({"key": ["x"]}), partition_root / "manifest" / "insee_counts.parquet")
    (partition_root / "manifest" / "postcode_counts.parquet").write_bytes(b"")
    part_count, part_size, part_hash, historical_hash = _raw_tree_hash(partition_root)

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    pickle_path = cache_root / "12345_.pkl"
    pickle_path.write_bytes(b"not-a-deserialised-pickle")
    sparse_hash = "a" * 64
    _write_json(
        cache_root / "12345_.pkl.sha256.json",
        {
            "schema_version": "sireto-tfidf-cache-integrity-1",
            "config_hash": cache_root.name,
            "partition_key": "12345|",
            "size_bytes": pickle_path.stat().st_size,
            "sha256": _sha(pickle_path),
        },
    )
    cache_files = list(cache_root.iterdir())
    cache_size = sum(path.stat().st_size for path in cache_files)
    cache_hash = _raw_cache_hash(cache_root)

    query_projection = unit.QUERY_SCHEMA.names
    projected_queries = pa.Table.from_arrays(
        [query_source.column(name) for name in query_projection],
        schema=unit.QUERY_SCHEMA,
    )
    dev_queries = projected_queries.filter(pa.array([True, False, True]))
    source_a = tmp_path / "source_a.py"
    source_b = tmp_path / "source_b.py"
    source_a.write_text("A = 1\n")
    source_b.write_text("B = 2\n")
    output_root = tmp_path / "outputs"
    audit_root = tmp_path / "audits"
    temp_root = tmp_path / "staging"
    output_root.mkdir()
    audit_root.mkdir()
    temp_root.mkdir()
    plan_path = tmp_path / "plan.json"
    lock_path = tmp_path / "lock.json"
    plan = {
        "schema_version": "test-plan",
        "cache": {
            "expected_file_count": 2,
            "expected_inventory_sha256": cache_hash,
            "expected_key_count": 1,
            "expected_pickle_count": 1,
            "expected_sidecar_count": 1,
            "expected_size_bytes": cache_size,
            "safe_key_regex": r"^(?:[0-9]{5}_|_[0-9]{5})$",
            "tfidf_config_artifact_hash": "b" * 64,
            "namespace": cache_root.name,
            "path": str(cache_root),
            "sidecar_schema_version": "sireto-tfidf-cache-integrity-1",
            "sparse_config_hash": sparse_hash,
        },
        "max_rss_bytes": 8 * 1024**3,
        "audit_output_root": str(audit_root),
        "execution_lock_path": str(lock_path),
        "orders": {
            "all": _query_spec(projected_queries, "all:"),
            "dev": _query_spec(dev_queries, "dev:"),
        },
        "output_root": str(output_root),
        "partitions": {
            "expected_file_count": part_count,
            "expected_inventory_sha256": part_hash,
            "expected_row_count": 2,
            "expected_runtime_signature": historical_hash,
            "expected_size_bytes": part_size,
            "insee_manifest": {
                "relative_path": "manifest/insee_counts.parquet",
                "sha256": _sha(partition_root / "manifest" / "insee_counts.parquet"),
                "size_bytes": (partition_root / "manifest" / "insee_counts.parquet").stat().st_size,
            },
            "path": str(partition_root),
            "postcode_manifest": {
                "relative_path": "manifest/postcode_counts.parquet",
                "sha256": hashlib.sha256(b"").hexdigest(),
                "size_bytes": 0,
            },
        },
        "queries": {
            "path": str(query_path),
            "projection": query_projection,
            "row_count": 3,
            "sha256": _sha(query_path),
            "size_bytes": query_path.stat().st_size,
        },
        "runtime": unit._runtime_values(),
        "sources": [str(source_a), str(source_b)],
        "split_assignments": {
            "counts": {"dev": 2, "fit": 1},
            "path": str(split_path),
            "projection": ["query_id", "split"],
            "row_count": 3,
            "sha256": _sha(split_path),
            "size_bytes": split_path.stat().st_size,
        },
        "temp_root": str(temp_root),
    }
    _write_json(plan_path, plan)
    lock = {
        "schema_version": unit.LOCK_SCHEMA,
        "purpose": unit.LOCK_PURPOSE,
        "audit_verdict": unit.LOCK_VERDICT,
        "git_commit": "f" * 40,
        "source_hashes": {str(source_a): _sha(source_a), str(source_b): _sha(source_b)},
        "input_paths": {
            "queries": str(query_path),
            "split_assignments": str(split_path),
            "partitions": str(partition_root),
            "cache": str(cache_root),
        },
        "input_hashes": {"queries": _sha(query_path), "split_assignments": _sha(split_path)},
        "partition_inventory_sha256": part_hash,
        "tfidf_inventory_sha256": cache_hash,
        "runtime": unit._runtime_values(),
        "output_root": str(output_root),
        "audit_output_root": str(audit_root),
        "temp_root": str(temp_root),
        "max_rss_bytes": 8 * 1024**3,
    }
    _write_json(lock_path, lock)
    staging = temp_root / "run"
    staging.mkdir()
    (staging / "tmp").mkdir()
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("SIRETO_V412_STAGING", str(staging))
    monkeypatch.setenv("TMPDIR", str(staging / "tmp"))
    monkeypatch.setattr(unit.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(unit, "CANONICAL_PLAN_PATH", plan_path)

    def fake_git_sources(repo, sources, expected, commit, max_rss):
        assert commit == "f" * 40
        return {
            Path(path): unit._snapshot_file(Path(path), max_rss)
            for path in sources
        }

    monkeypatch.setattr(unit, "_verify_git_sources", fake_git_sources)
    return {
        "plan": plan,
        "plan_path": plan_path,
        "lock_path": lock_path,
        "query_path": query_path,
        "split_path": split_path,
        "partition_root": partition_root,
        "cache_root": cache_root,
        "staging": staging,
        "output_root": output_root,
        "audit_root": audit_root,
    }


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"sha256":"a","sha256":"b"}')
    with pytest.raises(unit.BuildStopped, match="duplicate JSON key"):
        unit.load_json_strict(path)


def test_projection_reads_only_authorised_columns(
    mini: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    original = pq.ParquetFile.iter_batches

    def recording(self, *args, **kwargs):
        seen.append(kwargs.get("columns"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", recording)
    all_table, dev_table = unit.build_query_tables(
        mini["query_path"], mini["split_path"], mini["plan"]
    )
    assert seen == [unit.QUERY_SCHEMA.names, ["query_id", "split"]]
    assert all_table.schema == unit.QUERY_SCHEMA
    assert dev_table.schema == unit.QUERY_SCHEMA
    assert "crm_record_id" not in all_table.column_names
    assert all_table.schema.metadata is None


def test_query_null_or_changed_payload_stops(mini: dict[str, object]) -> None:
    plan = mini["plan"]
    bad = pq.read_table(mini["query_path"])
    columns = {
        name: bad.column(name).to_pylist()
        for name in bad.column_names
    }
    columns["crm_city"][0] = None
    pq.write_table(pa.table(columns), mini["query_path"])
    with pytest.raises(unit.BuildStopped, match="null"):
        unit.build_query_tables(mini["query_path"], mini["split_path"], plan)


def test_partition_inventory_rejects_extra_symlink_and_content(
    mini: dict[str, object], tmp_path: Path
) -> None:
    root = mini["partition_root"]
    plan = mini["plan"]
    result = unit.inventory_partitions(root, plan["partitions"], 8 * 1024**3)
    assert result.logical_sha256 == plan["partitions"]["expected_inventory_sha256"]
    extra = root / "insee" / "extra.parquet"
    pq.write_table(pa.table({"x": [1]}), extra)
    with pytest.raises(unit.BuildStopped, match="count mismatch"):
        unit.inventory_partitions(root, plan["partitions"], 8 * 1024**3)
    extra.unlink()
    link = root / "insee" / "linked.parquet"
    link.symlink_to(root / "insee" / "a.parquet")
    with pytest.raises(unit.BuildStopped, match="symlink"):
        unit.inventory_partitions(root, plan["partitions"], 8 * 1024**3)


def test_empty_postcode_manifest_is_never_deserialised(
    mini: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = []
    original = pq.ParquetFile

    def guarded(path, *args, **kwargs):
        opened.append(Path(path))
        assert Path(path).name != "postcode_counts.parquet"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(unit.pq, "ParquetFile", guarded)
    unit.inventory_partitions(
        mini["partition_root"], mini["plan"]["partitions"], 8 * 1024**3
    )
    assert any(path.name == "a.parquet" for path in opened)


@pytest.mark.parametrize("mutation", ["pickle", "reseal", "sidecar_extra", "sidecar_duplicate"])
def test_cache_falsifications_stop(
    mini: dict[str, object], mutation: str
) -> None:
    root = mini["cache_root"]
    spec = mini["plan"]["cache"]
    pickle_path = root / "12345_.pkl"
    sidecar_path = root / "12345_.pkl.sha256.json"
    if mutation in {"pickle", "reseal"}:
        pickle_path.write_bytes(b"tampered-but-still-not-unpickled")
        if mutation == "reseal":
            sidecar = json.loads(sidecar_path.read_text())
            sidecar["size_bytes"] = pickle_path.stat().st_size
            sidecar["sha256"] = _sha(pickle_path)
            _write_json(sidecar_path, sidecar)
    elif mutation == "sidecar_extra":
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["unexpected"] = True
        _write_json(sidecar_path, sidecar)
    else:
        sidecar_path.write_text(
            '{"schema_version":"sireto-tfidf-cache-integrity-1",'
            '"config_hash":"' + "a" * 64 + '","partition_key":"12345|",'
            '"size_bytes":25,"sha256":"x","sha256":"y"}'
        )
    with pytest.raises(unit.BuildStopped):
        unit.inventory_cache(root, spec, 8 * 1024**3)


def test_cache_never_deserialises_pickle(
    mini: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import pickle

    monkeypatch.setattr(
        pickle,
        "load",
        lambda *_args, **_kwargs: pytest.fail("pickle must not be deserialised"),
    )
    result = unit.inventory_cache(
        mini["cache_root"], mini["plan"]["cache"], 8 * 1024**3
    )
    assert result.row_count == 1
    assert mini["plan"]["cache"]["namespace"] != mini["plan"]["cache"]["sparse_config_hash"]


def test_cache_safe_key_and_filename_are_canonical(mini: dict[str, object]) -> None:
    root = mini["cache_root"]
    old_pickle = root / "12345_.pkl"
    old_sidecar = root / "12345_.pkl.sha256.json"
    old_pickle.rename(root / "wrong.pkl")
    old_sidecar.rename(root / "wrong.pkl.sha256.json")
    with pytest.raises(unit.BuildStopped, match="safe_key"):
        unit.inventory_cache(root, mini["plan"]["cache"], 8 * 1024**3)


def test_lock_is_exact_and_pins_every_input(mini: dict[str, object]) -> None:
    lock = unit.load_json_strict(mini["lock_path"])
    lock["bypass"] = True
    with pytest.raises(unit.BuildStopped, match="key set"):
        unit.validate_lock(lock, mini["plan"], Path.cwd(), 8 * 1024**3)
    lock.pop("bypass")
    lock["tfidf_inventory_sha256"] = "0" * 64
    with pytest.raises(unit.BuildStopped, match="cache hash"):
        unit.validate_lock(lock, mini["plan"], Path.cwd(), 8 * 1024**3)


def test_direct_internal_build_rejects_arbitrary_plan(
    mini: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(unit, "CANONICAL_PLAN_PATH", Path("config/v4_12_unit_input_plan.json"))
    with pytest.raises(unit.BuildStopped, match="canonical"):
        unit.build_inputs(mini["plan_path"], mini["lock_path"])


def test_git_blob_must_match_tracked_worktree_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path
    source = repo / "tracked.py"
    source.write_bytes(b"dirty worktree\n")
    expected = _sha(source)

    def fake_git(args):
        return "tracked.py"

    monkeypatch.setattr(unit, "_git", fake_git)
    monkeypatch.setattr(unit, "_git_bytes", lambda _args: b"committed blob\n")
    with pytest.raises(unit.BuildStopped, match="committed Git blob"):
        unit._verify_git_sources(
            repo, ["tracked.py"], {"tracked.py": expected}, "f" * 40, 1024**3
        )


def test_audited_parent_commit_remains_valid_after_unrelated_later_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "tracked.py"
    source.write_bytes(b"unchanged audited source\n")
    expected = _sha(source)
    git_calls = []

    def fake_git(args):
        git_calls.append(args)
        assert "rev-parse" not in args
        return "tracked.py"

    monkeypatch.setattr(unit, "_git", fake_git)
    monkeypatch.setattr(unit, "_git_bytes", lambda _args: source.read_bytes())
    snapshots = unit._verify_git_sources(
        tmp_path,
        ["tracked.py"],
        {"tracked.py": expected},
        "a" * 40,
        1024**3,
    )
    assert list(snapshots) == [source]
    assert len(git_calls) == 1


def test_modified_tracked_source_remains_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "tracked.py"
    committed = b"audited source\n"
    source.write_bytes(b"modified source\n")
    expected = hashlib.sha256(committed).hexdigest()
    monkeypatch.setattr(unit, "_git", lambda _args: "tracked.py")
    monkeypatch.setattr(unit, "_git_bytes", lambda _args: committed)
    with pytest.raises(unit.BuildStopped, match="source hash mismatch"):
        unit._verify_git_sources(
            tmp_path,
            ["tracked.py"],
            {"tracked.py": expected},
            "a" * 40,
            1024**3,
        )


def test_process_guard_requires_python_b_and_private_tmp(
    mini: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(unit.sys, "dont_write_bytecode", False)
    with pytest.raises(unit.BuildStopped, match="python -B"):
        unit._process_context(mini["plan"])
    monkeypatch.setattr(unit.sys, "dont_write_bytecode", True)
    monkeypatch.setenv("TMPDIR", str(Path(mini["plan"]["temp_root"]).parent))
    with pytest.raises(unit.BuildStopped, match="TMPDIR"):
        unit._process_context(mini["plan"])


def test_rss_guard_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unit, "_rss_bytes", lambda: 101)
    with pytest.raises(unit.BuildStopped, match="RSS limit exceeded"):
        unit._check_rss(100)


def test_toc_tou_detects_same_size_change(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"abc")
    before = unit._snapshot_file(path, 1024**3)
    path.write_bytes(b"xyz")
    with pytest.raises(unit.BuildStopped, match="TOCTOU"):
        unit._same_snapshot(path, before, 1024**3)


def test_complete_mini_publication_and_concordance(mini: dict[str, object]) -> None:
    runtime_dir, audit_dir = unit.build_inputs(mini["plan_path"], mini["lock_path"])
    assert {path.name for path in runtime_dir.iterdir()} == unit.RUNTIME_FILES
    assert {path.name for path in audit_dir.iterdir()} == unit.AUDIT_FILES
    unit.validate_concordance(runtime_dir, audit_dir)
    manifest = unit.load_json_strict(runtime_dir / "runtime_manifest.json")
    encoded = json.dumps(manifest)
    assert str(mini["query_path"]) not in encoded
    assert "source_a.py" not in encoded
    ledger = pq.read_table(audit_dir / "data_inputs.parquet")
    assert ledger.num_rows == 7
    assert set(ledger.column("role").to_pylist()) == {
        "queries",
        "split",
        "candidate_partition",
        "partition_manifest",
        "tfidf_pickle",
        "tfidf_sidecar",
    }
    for root in (runtime_dir, audit_dir):
        for path in root.rglob("*"):
            assert not (path.stat().st_mode & stat.S_IWUSR)
    # Restore permissions so pytest can clean its temporary directory.
    for root in (runtime_dir, audit_dir):
        os.chmod(root, 0o755)
        for path in root.rglob("*"):
            os.chmod(path, 0o755 if path.is_dir() else 0o644)


@pytest.mark.parametrize(
    "filename",
    [
        "queries_all.parquet",
        "queries_dev.parquet",
        "partition_inventory.parquet",
        "tfidf_inventory.parquet",
    ],
)
def test_each_runtime_parquet_mutation_is_detected(
    mini: dict[str, object], filename: str
) -> None:
    runtime_dir, audit_dir = unit.build_inputs(mini["plan_path"], mini["lock_path"])
    os.chmod(runtime_dir, 0o755)
    target = runtime_dir / filename
    os.chmod(target, 0o644)
    with target.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(unit.BuildStopped, match="runtime Parquet|runtime file mutation"):
        unit.validate_concordance(runtime_dir, audit_dir)
    os.chmod(audit_dir, 0o755)
    for root in (runtime_dir, audit_dir):
        for path in root.rglob("*"):
            os.chmod(path, 0o755 if path.is_dir() else 0o644)


def test_runtime_extra_file_is_detected(mini: dict[str, object]) -> None:
    runtime_dir, audit_dir = unit.build_inputs(mini["plan_path"], mini["lock_path"])
    os.chmod(runtime_dir, 0o755)
    (runtime_dir / "extra.txt").write_text("forbidden")
    with pytest.raises(unit.BuildStopped, match="file set"):
        unit.validate_concordance(runtime_dir, audit_dir)
    os.chmod(audit_dir, 0o755)
    for root in (runtime_dir, audit_dir):
        for path in root.rglob("*"):
            os.chmod(path, 0o755 if path.is_dir() else 0o644)


def test_ledger_schema_resealed_in_manifest_is_still_rejected(
    mini: dict[str, object]
) -> None:
    runtime_dir, audit_dir = unit.build_inputs(mini["plan_path"], mini["lock_path"])
    os.chmod(audit_dir, 0o755)
    ledger_path = audit_dir / "data_inputs.parquet"
    manifest_path = audit_dir / "manifest.json"
    os.chmod(ledger_path, 0o644)
    os.chmod(manifest_path, 0o644)
    pq.write_table(pa.table({"wrong": ["schema"]}), ledger_path)
    manifest = unit.load_json_strict(manifest_path)
    manifest["files"]["data_inputs.parquet"] = {
        "sha256": _sha(ledger_path),
        "size_bytes": ledger_path.stat().st_size,
    }
    _write_json(manifest_path, manifest)
    with pytest.raises(unit.BuildStopped, match="ledger schema"):
        unit.validate_concordance(runtime_dir, audit_dir)
    os.chmod(runtime_dir, 0o755)
    for root in (runtime_dir, audit_dir):
        for path in root.rglob("*"):
            os.chmod(path, 0o755 if path.is_dir() else 0o644)


def test_crm_swap_then_restore_cannot_evade_pre_promotion_projection(
    mini: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_builder = unit.build_query_tables
    source_path = mini["query_path"]
    source_bytes = source_path.read_bytes()
    calls = 0

    def swapping_builder(queries_path, split_path, plan):
        nonlocal calls
        calls += 1
        if calls == 2:
            table = pq.read_table(source_path)
            values = {name: table.column(name).to_pylist() for name in table.column_names}
            values["crm_city"][0] = "VILLE-SWAP"
            pq.write_table(pa.table(values), source_path)
            try:
                return original_builder(queries_path, split_path, plan)
            finally:
                source_path.write_bytes(source_bytes)
        return original_builder(queries_path, split_path, plan)

    monkeypatch.setattr(unit, "build_query_tables", swapping_builder)
    with pytest.raises(unit.BuildStopped, match="projection changed"):
        unit.build_inputs(mini["plan_path"], mini["lock_path"])


def test_mutated_integrity_and_orphan_audit_fail_closed(mini: dict[str, object]) -> None:
    runtime_dir, audit_dir = unit.build_inputs(mini["plan_path"], mini["lock_path"])
    os.chmod(runtime_dir, 0o755)
    os.chmod(runtime_dir / "integrity.json", 0o644)
    (runtime_dir / "integrity.json").write_text("{}\n")
    with pytest.raises(unit.BuildStopped, match="runtime file mutation"):
        unit.validate_concordance(runtime_dir, audit_dir)
    with pytest.raises(unit.BuildStopped, match="already exists"):
        unit.build_inputs(mini["plan_path"], mini["lock_path"])
    os.chmod(audit_dir, 0o755)
    for root in (runtime_dir, audit_dir):
        for path in root.rglob("*"):
            os.chmod(path, 0o755 if path.is_dir() else 0o644)


def test_audit_is_promoted_before_runtime(
    mini: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    destinations = []
    original = unit._promote

    def recording(source, destination):
        destinations.append(destination.parent)
        return original(source, destination)

    monkeypatch.setattr(unit, "_promote", recording)
    runtime_dir, audit_dir = unit.build_inputs(mini["plan_path"], mini["lock_path"])
    assert destinations == [Path(mini["plan"]["audit_output_root"]), Path(mini["plan"]["output_root"])]
    for root in (runtime_dir, audit_dir):
        os.chmod(root, 0o755)
        for path in root.rglob("*"):
            os.chmod(path, 0o755 if path.is_dir() else 0o644)
