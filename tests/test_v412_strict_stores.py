from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts import certify_v412_strict_stores as cert
from src.xgb_matcher import v412_strict_stores as stores


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allowed(role: str, key: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "partition_key": key,
        "absolute_path": str(path.resolve(strict=True)),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _partition_row() -> dict[str, object]:
    return {
        "siret": "12345678900011",
        "siren": "123456789",
        "denomination": "SAS Alpha",
        "enseigne1": "Alpha",
        "enseigne2": None,
        "enseigne3": None,
        "etablissementSiege": True,
        "is_siege": True,
        "numeroVoie": "1",
        "typeVoie": "RUE",
        "libelleVoie": "DE LA PAIX",
        "complementAdresse": None,
        "postcode": "75001",
        "city": "PARIS",
        "cj_ul": "5710",
        "etat_admin": "A",
        "last_treatment_date": None,
        "sigle_ul": None,
        "denomination_ul": "ALPHA FRANCE",
        "denomination_usuelle_ul": None,
        "nom_ul": None,
        "prenom_usuel_ul": None,
        "pm_dirigeant_names": None,
    }


def _write_insee_partition(root: Path) -> tuple[Path, dict[str, object], list[dict[str, object]]]:
    relative = Path("insee/insee=75056/part.parquet")
    path = root / relative
    path.parent.mkdir(parents=True)
    row = _partition_row()
    schema = pa.schema(
        [pa.field(name, kind) for name, kind in stores._INSEE_FIELDS]
    )
    table = pa.Table.from_pylist([row], schema=schema)
    pq.write_table(table, path)
    record = {
        "relative_path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }
    return path, record, [row]


def _cache_artifacts(aligned: list[dict[str, object]], names: list[str] | None = None):
    expected = stores._expected_tfidf_names(aligned)
    values = expected if names is None else names
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
        norm=None,
    )
    matrix = vectorizer.fit_transform(expected)
    return (
        vectorizer,
        matrix,
        values,
        vectorizer,
        matrix,
        vectorizer,
        matrix,
    )


def _write_cache(
    root: Path,
    key: str,
    aligned: list[dict[str, object]],
    *,
    names: list[str] | None = None,
    duplicate_sidecar: bool = False,
    artifacts: tuple[object, ...] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pickle_path = root / f"{key}.pkl"
    sidecar_path = root / f"{key}.pkl.sha256.json"
    root.mkdir()
    pickle_path.write_bytes(
        pickle.dumps(
            _cache_artifacts(aligned, names)
            if artifacts is None
            else artifacts
        )
    )
    sidecar = {
        "config_hash": stores.CACHE_NAMESPACE,
        "partition_key": key,
        "schema_version": stores.SIDECAR_SCHEMA,
        "sha256": _sha(pickle_path),
        "size_bytes": pickle_path.stat().st_size,
    }
    if duplicate_sidecar:
        sidecar_path.write_text(
            '{"config_hash":"%s","partition_key":"%s",'
            '"schema_version":"%s","sha256":"%s","size_bytes":%d,'
            '"size_bytes":%d}\n'
            % (
                stores.CACHE_NAMESPACE,
                key,
                stores.SIDECAR_SCHEMA,
                _sha(pickle_path),
                pickle_path.stat().st_size,
                pickle_path.stat().st_size,
            ),
            encoding="utf-8",
        )
    else:
        sidecar_path.write_text(
            json.dumps(sidecar, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    records = [
        {
            "partition_key": key,
            "pickle_relative_path": pickle_path.name,
            "pickle_size_bytes": pickle_path.stat().st_size,
            "pickle_sha256": _sha(pickle_path),
            "sidecar_relative_path": sidecar_path.name,
            "sidecar_size_bytes": sidecar_path.stat().st_size,
            "sidecar_sha256": _sha(sidecar_path),
        }
    ]
    allowed = [
        _allowed("cache_pickle", key, pickle_path),
        _allowed("cache_sidecar", key, sidecar_path),
    ]
    return records, allowed


def _write_lookup(root: Path) -> tuple[Path, dict[str, object], list[dict[str, object]]]:
    path = root / "lookup.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE candidate_details (
            siret VARCHAR,
            candidate_state VARCHAR,
            enseigne1 VARCHAR,
            enseigne2 VARCHAR,
            enseigne3 VARCHAR,
            denomination_usuelle VARCHAR,
            activity_code VARCHAR
        )
        """
    )
    connection.execute(
        "INSERT INTO candidate_details VALUES "
        "('12345678900011','OUVERT','ALPHA',NULL,NULL,'ALPHA','62.01Z')"
    )
    connection.execute(
        "CREATE UNIQUE INDEX candidate_details_siret_uidx "
        "ON candidate_details(siret)"
    )
    connection.close()
    descriptor = {
        "schema_version": stores.LOOKUP_DESCRIPTOR_SCHEMA,
        "database_sha256": _sha(path),
        "database_size_bytes": path.stat().st_size,
        "table_name": "candidate_details",
        "columns": list(stores._LOOKUP_COLUMNS),
        "column_types": ["VARCHAR"] * 7,
        "index_name": "candidate_details_siret_uidx",
        "index_unique": True,
        "row_count": 1,
        "max_sirets_per_call": 100,
        "read_only": True,
    }
    return path, descriptor, [_allowed("lookup_database", "", path)]


def _reseal_cache(
    records: list[dict[str, object]],
    allowed: list[dict[str, object]],
    pickle_path: Path,
    sidecar_path: Path,
) -> None:
    record = records[0]
    record.update(
        {
            "pickle_size_bytes": pickle_path.stat().st_size,
            "pickle_sha256": _sha(pickle_path),
            "sidecar_size_bytes": sidecar_path.stat().st_size,
            "sidecar_sha256": _sha(sidecar_path),
        }
    )
    allowed[:] = [
        _allowed("cache_pickle", str(record["partition_key"]), pickle_path),
        _allowed("cache_sidecar", str(record["partition_key"]), sidecar_path),
    ]


def _synthetic_run_spec() -> dict[str, object]:
    digest = "a" * 64
    partition_records: list[dict[str, object]] = []
    cache_records: list[dict[str, object]] = []
    allowed: list[dict[str, object]] = []
    for index in range(648):
        key = f"{index:05d}_"
        partition_relative = f"insee/insee={index:05d}/part.parquet"
        pickle_relative = f"{key}.pkl"
        sidecar_relative = f"{key}.pkl.sha256.json"
        partition_records.append(
            {
                "relative_path": partition_relative,
                "size_bytes": 1,
                "sha256": digest,
            }
        )
        cache_records.append(
            {
                "partition_key": key,
                "pickle_relative_path": pickle_relative,
                "pickle_size_bytes": 1,
                "pickle_sha256": digest,
                "sidecar_relative_path": sidecar_relative,
                "sidecar_size_bytes": 1,
                "sidecar_sha256": digest,
            }
        )
        allowed.extend(
            [
                {
                    "role": "partition",
                    "partition_key": key,
                    "absolute_path": f"/synthetic/partition/{index:05d}.parquet",
                    "size_bytes": 1,
                    "sha256": digest,
                },
                {
                    "role": "cache_pickle",
                    "partition_key": key,
                    "absolute_path": f"/synthetic/cache/{pickle_relative}",
                    "size_bytes": 1,
                    "sha256": digest,
                },
                {
                    "role": "cache_sidecar",
                    "partition_key": key,
                    "absolute_path": f"/synthetic/cache/{sidecar_relative}",
                    "size_bytes": 1,
                    "sha256": digest,
                },
            ]
        )
    allowed.append(
        {
            "role": "lookup_database",
            "partition_key": "",
            "absolute_path": "/synthetic/lookup.duckdb",
            "size_bytes": 1,
            "sha256": digest,
        }
    )
    allowed.sort(key=lambda row: (str(row["role"]).encode(), str(row["absolute_path"]).encode()))
    return {
        "schema_version": stores.RUN_SPEC_SCHEMA,
        "safe_input_build_id": digest,
        "query_count": 1,
        "routing_payload_sha256": digest,
        "partition_records": partition_records,
        "cache_records": cache_records,
        "lookup_descriptor_sha256": digest,
        "allowed_read_files": allowed,
        "staging_dir": "output",
        "tmp_dir": "tmp",
        "max_rss_bytes": 1 << 30,
        "declarations": dict(cert.DECLARATIONS),
    }


def _publication_fixture(
    tmp_path: Path,
    *,
    pending: bool = False,
) -> tuple[Path, Path, str, dict[str, object], Path, Path]:
    output_root = tmp_path / "published"
    audit_root = tmp_path / "audit"
    output_root.mkdir()
    audit_root.mkdir()
    source = tmp_path / "sealed-source.bin"
    source.write_bytes(b"synthetic-sealed-input")
    source_snapshot = cert._snapshot(source, 1 << 30)

    digest = "b" * 64
    plan: dict[str, object] = {
        "safe_input": {
            "runtime_manifest_sha256": digest,
            "partition_inventory_sha256": "c" * 64,
            "tfidf_inventory_sha256": "d" * 64,
        },
        "partitions": {
            "expected_subset_logical_sha256": "e" * 64,
            "expected_subset_file_count": 648,
            "expected_subset_row_count": 777,
        },
        "cache": {
            "expected_subset_logical_sha256": "f" * 64,
            "expected_subset_key_count": 648,
            "expected_aligned_row_count": 700,
        },
        "lookup": {
            "database_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "integrity_sha256": "3" * 64,
            "timing_sha256": "4" * 64,
            "sample_max_sirets": 100,
        },
        "runtime": {"python": "synthetic"},
        "routing": {"query_count": 10, "distinct_key_count": 8},
        "max_rss_bytes": 1 << 30,
    }
    lock: dict[str, object] = {
        "git_commit": "1" * 40,
        "source_hashes": {"synthetic.py": "5" * 64},
    }
    descriptor = {"schema": "synthetic-lookup"}
    run_spec = {"schema": "synthetic-run"}
    profile = b"(version 1)\n(deny default)\n"
    plan_bytes = cert.canonical_json(plan)
    lock_bytes = cert.canonical_json(lock)
    descriptor_sha = hashlib.sha256(cert.canonical_json(descriptor)).hexdigest()
    run_spec_sha = hashlib.sha256(cert.canonical_json(run_spec)).hexdigest()
    profile_sha = hashlib.sha256(profile).hexdigest()
    build_id = cert._build_identity(
        plan_bytes,
        lock_bytes,
        lock,
        plan,
        profile_sha,
        descriptor_sha,
        run_spec_sha,
    )

    cert_path = output_root / (
        f".pending-{build_id}" if pending else build_id
    )
    audit_path = audit_root / build_id
    cert_path.mkdir()
    audit_path.mkdir()
    probe = {
        "schema_version": "sireto-v4.12-strict-stores-probe-1",
        "build_id": build_id,
        "query_count": 10,
        "distinct_key_count": 8,
        "partition_verified_count": 648,
        "partition_raw_row_count": 777,
        "cache_verified_count": 648,
        "aligned_pool_row_count": 700,
        "cache_miss_count": 0,
        "rebuild_count": 0,
        "write_count": 0,
        "lookup_sample_count": 10,
        "lookup_missing_count": 0,
        "lookup_extra_count": 0,
        "sandbox_checks": {
            "allowed_read": True,
            "oracle_denied": True,
            "oracle_audit_denied": True,
            "network_denied": True,
            "write_denied": True,
        },
        "peak_rss_bytes": 1,
        "durations_ns": {
            "partitions": 0,
            "cache": 0,
            "lookup": 0,
            "total": 0,
        },
        "declarations": dict(cert.DECLARATIONS),
    }
    cert._write_json(cert_path / "store_probe.json", probe)
    cert._write_json(cert_path / "lookup_descriptor.json", descriptor)
    cert._write_json(cert_path / "run_spec.json", run_spec)
    (cert_path / "sandbox_profile_effective.sb").write_bytes(profile)

    role_projections = [
        ("safe_runtime_manifest", "JSON_EXACT", 1),
        ("safe_queries_all", "HASH_ONLY", 1),
        ("safe_queries_dev", ",".join(cert.QUERY_COLUMNS), 1),
        ("safe_partition_inventory", ",".join(cert.PARTITION_COLUMNS), 1),
        ("safe_tfidf_inventory", ",".join(cert.CACHE_COLUMNS), 1),
        ("safe_input_integrity", "HASH_ONLY", 1),
        ("partition", "STRICT_PARTITION_SCHEMA", 648),
        ("cache_pickle", "PICKLE_TUPLE_7", 648),
        (
            "cache_sidecar",
            "config_hash,partition_key,schema_version,sha256,size_bytes",
            648,
        ),
        (
            "lookup_database",
            "siret,candidate_state,enseigne1,enseigne2,enseigne3,"
            "denomination_usuelle,activity_code",
            1,
        ),
        ("lookup_manifest", "PARENT_VALIDATION_ONLY", 1),
        ("lookup_integrity", "PARENT_VALIDATION_ONLY", 1),
        ("lookup_timing", "PARENT_VALIDATION_ONLY", 1),
    ]
    ledger_sources: list[tuple[str, Path, str, Mapping[str, Any]]] = []
    ledger_rows: list[tuple[object, ...]] = []
    for role, projection, count in role_projections:
        for _ in range(count):
            ledger_sources.append((role, source, projection, source_snapshot))
            ledger_rows.append(
                (
                    role,
                    str(source.absolute()),
                    projection,
                    source_snapshot["size"],
                    source_snapshot["sha256"],
                    source_snapshot["size"],
                    source_snapshot["sha256"],
                )
            )
    ledger_rows.sort(key=lambda row: (str(row[0]).encode(), str(row[1]).encode()))
    pq.write_table(
        cert._table(ledger_rows),
        audit_path / "open_ledger.parquet",
    )
    integrity = {
        "schema_version": "sireto-v4.12-strict-stores-integrity-1",
        "build_id": build_id,
        "run_spec_sha256": run_spec_sha,
        "lookup_descriptor_sha256": descriptor_sha,
        "sandbox_profile_effective_sha256": profile_sha,
        "store_probe_sha256": _sha(cert_path / "store_probe.json"),
        "data_input_count": 1954,
        "data_ledger_sha256": _sha(audit_path / "open_ledger.parquet"),
        "declarations": dict(cert.DECLARATIONS),
    }
    cert._write_json(cert_path / "integrity.json", integrity)
    certification_manifest = {
        "schema_version": "sireto-v4.12-strict-stores-certification-1",
        "build_id": build_id,
        "files": sorted(
            [
                cert._record(path)
                for path in cert_path.iterdir()
                if path.name != "manifest.json"
            ],
            key=lambda row: row["path"].encode(),
        ),
        "runtime": plan["runtime"],
        "declarations": dict(cert.DECLARATIONS),
        "verdict": cert.GO,
    }
    cert._write_json(cert_path / "manifest.json", certification_manifest)
    provenance = {
        "schema_version": "sireto-v4.12-strict-stores-provenance-1",
        "build_id": build_id,
        "git_commit": lock["git_commit"],
        "source_hashes": lock["source_hashes"],
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "sandbox_profile_effective_sha256": profile_sha,
        "runtime": plan["runtime"],
        "data_input_count": 1954,
        "certification_manifest_sha256": _sha(cert_path / "manifest.json"),
        "declarations": dict(cert.DECLARATIONS),
    }
    cert._write_json(audit_path / "provenance.json", provenance)
    cert._write_json(
        audit_path / "manifest.json",
        {
            "schema_version": "sireto-v4.12-strict-stores-audit-manifest-1",
            "build_id": build_id,
            "files": sorted(
                [
                    cert._record(path)
                    for path in audit_path.iterdir()
                    if path.name != "manifest.json"
                ],
                key=lambda row: row["path"].encode(),
            ),
        },
    )
    current: dict[str, object] = {
        "plan": plan,
        "lock": lock,
        "plan_bytes": plan_bytes,
        "lock_bytes": lock_bytes,
        "descriptor": descriptor,
        "descriptor_sha": descriptor_sha,
        "run_spec": run_spec,
        "run_spec_sha": run_spec_sha,
        "profile_sha": profile_sha,
        "ledger_sources": ledger_sources,
    }
    return output_root, audit_root, build_id, current, cert_path, audit_path


def test_partition_store_reads_exact_schema_and_injects_hive(tmp_path: Path) -> None:
    path, record, _ = _write_insee_partition(tmp_path / "partitions")
    store = stores.StrictPartitionStore(
        [record],
        [_allowed("partition", "75056_", path)],
    )
    status, rows = store.load_with_status("75056_")
    assert status == "VALID_ROWS"
    assert rows[0]["insee"] == "75056"
    assert rows[0]["siret"] == "12345678900011"


def test_core_has_no_historical_project_import() -> None:
    source = Path(stores.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(("." * node.level) + (node.module or ""))
    assert not any(
        name.startswith((".", "src", "xgb_matcher"))
        for name in imported
    )


def test_private_alignment_is_last_wins_in_first_key_order() -> None:
    first = _partition_row()
    replacement = {**first, "denomination": "ALPHA REPLACEMENT"}
    second = {
        **first,
        "siret": "12345678900022",
        "denomination": "BETA",
    }
    closed = {
        **first,
        "siret": "12345678900033",
        "etat_admin": "F",
    }
    unnamed = {
        **first,
        "siret": "12345678900044",
        **{column: None for column in stores._NAME_FILTER_COLUMNS},
    }
    aligned = stores._build_aligned_pool(
        [first, second, closed, unnamed, replacement]
    )
    assert [row["siret"] for row in aligned] == [
        "12345678900011",
        "12345678900022",
    ]
    assert aligned[0]["denomination"] == "ALPHA REPLACEMENT"


def test_partition_store_stops_on_mutation_and_symlink(tmp_path: Path) -> None:
    path, record, _ = _write_insee_partition(tmp_path / "partitions")
    allowed = [_allowed("partition", "75056_", path)]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(stores.StrictPartitionError, match=stores.STOP_PARTITION):
        stores.StrictPartitionStore([record], allowed).load("75056_")

    real_path, record, _ = _write_insee_partition(tmp_path / "real")
    link = tmp_path / "linked.parquet"
    link.symlink_to(real_path)
    linked_allowed = dict(_allowed("partition", "75056_", real_path))
    linked_allowed["absolute_path"] = str(link.absolute())
    with pytest.raises(stores.StrictPartitionError, match=stores.STOP_PARTITION):
        stores.StrictPartitionStore([record], [linked_allowed]).load("75056_")


def test_partition_store_rejects_absent_extra_schema_and_arrow(
    tmp_path: Path,
) -> None:
    absent_path, absent_record, _ = _write_insee_partition(tmp_path / "absent")
    absent_allowed = [_allowed("partition", "75056_", absent_path)]
    absent_path.unlink()
    with pytest.raises(stores.StrictPartitionError, match=stores.STOP_PARTITION):
        stores.StrictPartitionStore([absent_record], absent_allowed).load("75056_")

    path, record, _ = _write_insee_partition(tmp_path / "extra")
    extra_allowed = _allowed("partition", "_75001", path)
    with pytest.raises(stores.StrictPartitionError, match="not bijective"):
        stores.StrictPartitionStore(
            [record],
            [_allowed("partition", "75056_", path), extra_allowed],
        )

    schema_path, schema_record, rows = _write_insee_partition(tmp_path / "schema")
    wrong_schema = pa.schema(
        [pa.field(name, kind) for name, kind in reversed(stores._INSEE_FIELDS)]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=wrong_schema), schema_path)
    schema_record.update(
        {"size_bytes": schema_path.stat().st_size, "sha256": _sha(schema_path)}
    )
    with pytest.raises(stores.StrictPartitionError, match="schema or column order"):
        stores.StrictPartitionStore(
            [schema_record],
            [_allowed("partition", "75056_", schema_path)],
        ).load("75056_")

    arrow_path, arrow_record, _ = _write_insee_partition(tmp_path / "arrow")
    arrow_path.write_bytes(b"not an Arrow or Parquet payload")
    arrow_record.update(
        {"size_bytes": arrow_path.stat().st_size, "sha256": _sha(arrow_path)}
    )
    with pytest.raises(
        stores.StrictPartitionError,
        match="Parquet|verified file operation failed",
    ):
        stores.StrictPartitionStore(
            [arrow_record],
            [_allowed("partition", "75056_", arrow_path)],
        ).load("75056_")


def test_partition_store_detects_same_bytes_mutation_restoration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, record, _ = _write_insee_partition(tmp_path / "partition")
    payload = path.read_bytes()
    before = path.stat()
    original = stores.pq.ParquetFile
    mutated = False

    def mutating_parquet_file(source: object, *args: object, **kwargs: object):
        nonlocal mutated
        parquet_file = original(source, *args, **kwargs)
        if not mutated:
            mutated = True
            path.write_bytes(payload)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return parquet_file

    monkeypatch.setattr(stores.pq, "ParquetFile", mutating_parquet_file)
    with pytest.raises(stores.StrictPartitionError, match="file changed while open"):
        stores.StrictPartitionStore(
            [record],
            [_allowed("partition", "75056_", path)],
        ).load("75056_")


def test_tfidf_cache_is_read_only_and_checks_names(tmp_path: Path) -> None:
    aligned = stores._build_aligned_pool([_partition_row()])
    records, allowed = _write_cache(tmp_path / "cache", "75056_", aligned)
    cache = stores.StrictVerifiedTfidfCache(
        records,
        allowed,
        namespace=stores.CACHE_NAMESPACE,
    )
    assert len(cache.get("75056_", aligned)) == 7
    assert not hasattr(cache, "put")
    assert not hasattr(cache, "clear")

    bad_records, bad_allowed = _write_cache(
        tmp_path / "bad-cache",
        "75056_",
        aligned,
        names=["WRONG"],
    )
    bad = stores.StrictVerifiedTfidfCache(
        bad_records,
        bad_allowed,
        namespace=stores.CACHE_NAMESPACE,
    )
    with pytest.raises(stores.StrictTfidfError, match="names parity"):
        bad.get("75056_", aligned)
    with pytest.raises(stores.StrictTfidfError, match="cache miss"):
        cache.get("_75001", aligned)


def test_tfidf_cache_rejects_duplicate_sidecar_keys(tmp_path: Path) -> None:
    aligned = stores._build_aligned_pool([_partition_row()])
    records, allowed = _write_cache(
        tmp_path / "cache",
        "75056_",
        aligned,
        duplicate_sidecar=True,
    )
    cache = stores.StrictVerifiedTfidfCache(
        records,
        allowed,
        namespace=stores.CACHE_NAMESPACE,
    )
    with pytest.raises(stores.StrictTfidfError, match="duplicate JSON key"):
        cache.get("75056_", aligned)


def test_tfidf_cache_rejects_collisions_and_noncanonical_paths(
    tmp_path: Path,
) -> None:
    aligned = stores._build_aligned_pool([_partition_row()])
    records, allowed = _write_cache(tmp_path / "cache", "75056_", aligned)
    with pytest.raises(stores.StrictTfidfError, match="duplicate cache key"):
        stores.StrictVerifiedTfidfCache(
            records + [dict(records[0])],
            allowed,
            namespace=stores.CACHE_NAMESPACE,
        )
    duplicate_allow = dict(allowed[0])
    with pytest.raises(stores.StrictTfidfError, match="duplicate cache allow record"):
        stores.StrictVerifiedTfidfCache(
            records,
            allowed + [duplicate_allow],
            namespace=stores.CACHE_NAMESPACE,
        )
    noncanonical = copy.deepcopy(records)
    noncanonical[0]["pickle_relative_path"] = "nested/75056_.pkl"
    with pytest.raises(stores.StrictTfidfError, match="non-canonical cache path"):
        stores.StrictVerifiedTfidfCache(
            noncanonical,
            allowed,
            namespace=stores.CACHE_NAMESPACE,
        )


@pytest.mark.parametrize("target", ["pickle", "sidecar"])
def test_tfidf_cache_rejects_post_seal_mutation(
    tmp_path: Path,
    target: str,
) -> None:
    aligned = stores._build_aligned_pool([_partition_row()])
    records, allowed = _write_cache(tmp_path / "cache", "75056_", aligned)
    path = (
        tmp_path / "cache" / "75056_.pkl"
        if target == "pickle"
        else tmp_path / "cache" / "75056_.pkl.sha256.json"
    )
    path.write_bytes(path.read_bytes() + b"x")
    cache = stores.StrictVerifiedTfidfCache(
        records,
        allowed,
        namespace=stores.CACHE_NAMESPACE,
    )
    with pytest.raises(stores.StrictTfidfError, match="mismatch|changed"):
        cache.get("75056_", aligned)


def test_tfidf_cache_rejects_corrupt_pickle_dimensions_and_vectorizer(
    tmp_path: Path,
) -> None:
    aligned = stores._build_aligned_pool([_partition_row()])

    records, allowed = _write_cache(tmp_path / "corrupt", "75056_", aligned)
    pickle_path = tmp_path / "corrupt" / "75056_.pkl"
    sidecar_path = tmp_path / "corrupt" / "75056_.pkl.sha256.json"
    pickle_path.write_bytes(b"not a pickle")
    sidecar_path.write_text(
        json.dumps(
            {
                "config_hash": stores.CACHE_NAMESPACE,
                "partition_key": "75056_",
                "schema_version": stores.SIDECAR_SCHEMA,
                "sha256": _sha(pickle_path),
                "size_bytes": pickle_path.stat().st_size,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    _reseal_cache(records, allowed, pickle_path, sidecar_path)
    with pytest.raises(
        stores.StrictTfidfError,
        match="pickle|verified file operation failed",
    ):
        stores.StrictVerifiedTfidfCache(
            records,
            allowed,
            namespace=stores.CACHE_NAMESPACE,
        ).get("75056_", aligned)

    base = _cache_artifacts(aligned)
    wrong_matrix = sparse.vstack([base[1], base[1]])
    wrong_dimensions = (
        base[0],
        wrong_matrix,
        base[2],
        base[3],
        base[4],
        base[5],
        base[6],
    )
    dimension_records, dimension_allowed = _write_cache(
        tmp_path / "dimensions",
        "75056_",
        aligned,
        artifacts=wrong_dimensions,
    )
    with pytest.raises(stores.StrictTfidfError, match="matrix row mismatch"):
        stores.StrictVerifiedTfidfCache(
            dimension_records,
            dimension_allowed,
            namespace=stores.CACHE_NAMESPACE,
        ).get("75056_", aligned)

    wrong_vectorizer = (object(), *base[1:])
    vectorizer_records, vectorizer_allowed = _write_cache(
        tmp_path / "vectorizer",
        "75056_",
        aligned,
        artifacts=wrong_vectorizer,
    )
    with pytest.raises(stores.StrictTfidfError, match="vectorizer type mismatch"):
        stores.StrictVerifiedTfidfCache(
            vectorizer_records,
            vectorizer_allowed,
            namespace=stores.CACHE_NAMESPACE,
        ).get("75056_", aligned)

    short_records, short_allowed = _write_cache(
        tmp_path / "short",
        "75056_",
        aligned,
        artifacts=base[:6],
    )
    with pytest.raises(stores.StrictTfidfError, match="tuple of seven"):
        stores.StrictVerifiedTfidfCache(
            short_records,
            short_allowed,
            namespace=stores.CACHE_NAMESPACE,
        ).get("75056_", aligned)


@pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
def test_tfidf_cache_rejects_non_finite_idf(
    tmp_path: Path,
    non_finite: float,
) -> None:
    aligned = stores._build_aligned_pool([_partition_row()])
    artifacts = list(_cache_artifacts(aligned))
    vectorizer = copy.deepcopy(artifacts[0])
    idf = vectorizer.idf_.copy()
    idf[0] = non_finite
    vectorizer.idf_ = idf
    artifacts[0] = vectorizer
    records, allowed = _write_cache(
        tmp_path / "cache",
        "75056_",
        aligned,
        artifacts=tuple(artifacts),
    )
    cache = stores.StrictVerifiedTfidfCache(
        records,
        allowed,
        namespace=stores.CACHE_NAMESPACE,
    )
    with pytest.raises(stores.StrictTfidfError, match="IDF mismatch"):
        cache.get("75056_", aligned)


def test_tfidf_cache_rejects_noncontiguous_vocabulary_and_sparse_indices(
    tmp_path: Path,
) -> None:
    aligned = stores._build_aligned_pool([_partition_row()])
    base = list(_cache_artifacts(aligned))

    vectorizer = copy.deepcopy(base[0])
    last_key = next(reversed(vectorizer.vocabulary_))
    vectorizer.vocabulary_[last_key] = len(vectorizer.vocabulary_)
    bad_vocabulary = list(base)
    bad_vocabulary[0] = vectorizer
    records, allowed = _write_cache(
        tmp_path / "vocabulary",
        "75056_",
        aligned,
        artifacts=tuple(bad_vocabulary),
    )
    with pytest.raises(
        stores.StrictTfidfError,
        match="vocabulary indices mismatch",
    ):
        stores.StrictVerifiedTfidfCache(
            records,
            allowed,
            namespace=stores.CACHE_NAMESPACE,
        ).get("75056_", aligned)

    matrix = base[1].copy()
    assert matrix.indices.size
    matrix.indices[0] = matrix.shape[1]
    bad_sparse_index = list(base)
    bad_sparse_index[1] = matrix
    records, allowed = _write_cache(
        tmp_path / "sparse-index",
        "75056_",
        aligned,
        artifacts=tuple(bad_sparse_index),
    )
    with pytest.raises(
        stores.StrictTfidfError,
        match="sparse index out of bounds",
    ):
        stores.StrictVerifiedTfidfCache(
            records,
            allowed,
            namespace=stores.CACHE_NAMESPACE,
        ).get("75056_", aligned)


def test_snapshot_lookup_uses_bounded_read_only_api(tmp_path: Path) -> None:
    path, descriptor, allowed = _write_lookup(tmp_path)
    with stores.StrictSnapshotLookup(descriptor, allowed) as lookup:
        result = lookup.get_candidate_scene_details(["12345678900011"])
        assert result["12345678900011"]["candidate_state"] == "OUVERT"
        with pytest.raises(stores.StrictLookupError, match="too many"):
            lookup.get_candidate_scene_details(
                [f"{value:014d}" for value in range(101)]
            )
    (Path(str(path) + ".tmp")).write_bytes(b"forbidden")
    with pytest.raises(stores.StrictLookupError, match="auxiliary file exists"):
        stores.StrictSnapshotLookup(descriptor, allowed)


@pytest.mark.parametrize("suffix", [".wal", ".tmp"])
def test_snapshot_lookup_rejects_all_writable_siblings(
    tmp_path: Path,
    suffix: str,
) -> None:
    path, descriptor, allowed = _write_lookup(tmp_path)
    Path(str(path) + suffix).write_bytes(b"forbidden")
    with pytest.raises(stores.StrictLookupError, match="auxiliary file exists"):
        stores.StrictSnapshotLookup(descriptor, allowed)


def test_snapshot_lookup_rejects_mutation_and_inode_swap(tmp_path: Path) -> None:
    (tmp_path / "mutated").mkdir()
    path, descriptor, allowed = _write_lookup(tmp_path / "mutated")
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(stores.StrictLookupError, match="mismatch"):
        stores.StrictSnapshotLookup(descriptor, allowed)

    (tmp_path / "swapped").mkdir()
    path, descriptor, allowed = _write_lookup(tmp_path / "swapped")
    replacement = tmp_path / "swapped" / "replacement.duckdb"
    shutil.copyfile(path, replacement)
    lookup = stores.StrictSnapshotLookup(descriptor, allowed)
    os.replace(replacement, path)
    with pytest.raises(stores.StrictLookupError, match="lookup (?:FD|path) changed"):
        lookup.get_candidate_scene_details(["12345678900011"])
    with pytest.raises(
        stores.StrictLookupError,
        match="lookup (?:FD|path) (?:changed|identity changed)",
    ):
        lookup.close()


def test_snapshot_lookup_connection_is_read_only_and_external_access_is_off(
    tmp_path: Path,
) -> None:
    _, descriptor, allowed = _write_lookup(tmp_path)
    with stores.StrictSnapshotLookup(descriptor, allowed) as lookup:
        connection = getattr(
            lookup,
            "_StrictSnapshotLookup__connection",
        )
        with pytest.raises(duckdb.Error):
            connection.execute("CREATE TABLE forbidden(value INTEGER)")
        with pytest.raises(duckdb.Error):
            connection.execute("ATTACH ':memory:' AS forbidden")
        with pytest.raises(duckdb.Error):
            connection.execute("INSTALL httpfs")
        with pytest.raises(stores.StrictLookupError, match="invalid canonical SIRET"):
            lookup.get_candidate_scene_details(["123"])


def test_strict_json_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(stores.StrictProbeError, match="duplicate JSON key"):
        stores.load_json_strict(path)


def test_run_spec_accepts_exact_frozen_keysets_and_counts() -> None:
    run_spec = _synthetic_run_spec()
    validated = stores._validate_run_spec(run_spec)
    assert len(validated["partition_records"]) == 648
    assert len(validated["cache_records"]) == 648
    assert len(validated["allowed_read_files"]) == 1945


@pytest.mark.parametrize("mutation", ["missing", "extra", "record", "declarations"])
def test_run_spec_rejects_keyset_and_declaration_drift(mutation: str) -> None:
    run_spec = _synthetic_run_spec()
    if mutation == "missing":
        del run_spec["query_count"]
    elif mutation == "extra":
        run_spec["unexpected"] = True
    elif mutation == "record":
        run_spec["partition_records"][0]["unexpected"] = True  # type: ignore[index]
    else:
        run_spec["declarations"]["oracle_opened"] = True  # type: ignore[index]
    with pytest.raises(stores.StrictProbeError, match=stores.STOP_PROBE):
        stores._validate_run_spec(run_spec)


def test_run_spec_rejects_allowed_role_counts_paths_and_order() -> None:
    extra_role = _synthetic_run_spec()
    extra_role["allowed_read_files"][0]["role"] = "unknown"  # type: ignore[index]
    with pytest.raises(stores.StrictProbeError, match="role mismatch"):
        stores._validate_run_spec(extra_role)

    duplicate_path = _synthetic_run_spec()
    duplicate_path["allowed_read_files"][1]["absolute_path"] = (  # type: ignore[index]
        duplicate_path["allowed_read_files"][0]["absolute_path"]  # type: ignore[index]
    )
    duplicate_path["allowed_read_files"].sort(  # type: ignore[union-attr]
        key=lambda row: (row["role"].encode(), row["absolute_path"].encode())
    )
    with pytest.raises(stores.StrictProbeError, match="path is not unique"):
        stores._validate_run_spec(duplicate_path)

    unsorted = _synthetic_run_spec()
    unsorted["allowed_read_files"][0], unsorted["allowed_read_files"][1] = (  # type: ignore[index]
        unsorted["allowed_read_files"][1],  # type: ignore[index]
        unsorted["allowed_read_files"][0],  # type: ignore[index]
    )
    with pytest.raises(stores.StrictProbeError, match="not canonically sorted"):
        stores._validate_run_spec(unsorted)


def test_certifier_uses_memory_profile_and_three_fd_anchored_child_controls() -> None:
    source = Path(cert.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    certify_function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "certify"
    )
    command_assignment = next(
        node
        for node in ast.walk(certify_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "command"
            for target in node.targets
        )
    )
    fd_controls = {
        node.slice.value
        for node in ast.walk(command_assignment.value)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "fd_path"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    sandbox_call = next(
        node
        for node in ast.walk(certify_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
        and any(keyword.arg == "pass_fds" for keyword in node.keywords)
    )
    pass_fds = next(
        keyword.value
        for keyword in sandbox_call.keywords
        if keyword.arg == "pass_fds"
    )
    inherited_controls = {
        node.value
        for node in ast.walk(pass_fds)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }
    expected = {"source", "run_spec", "descriptor"}
    assert fd_controls == expected
    assert inherited_controls == expected
    command_elements = command_assignment.value.elts
    profile_switch = next(
        index
        for index, node in enumerate(command_elements)
        if isinstance(node, ast.Constant) and node.value == "-p"
    )
    assert isinstance(command_elements[profile_switch + 1], ast.Name)
    assert command_elements[profile_switch + 1].id == "effective"
    command_literals = {
        node.value for node in command_elements if isinstance(node, ast.Constant)
    }
    assert "-f" not in command_literals
    assert '("profile", profile_canonical, profile_snap)' in source
    assert '("profile", profile_snap)' in source
    assert "profile_snap = _seal_control_file(" in source
    assert 'f"/dev/fd/{descriptor_fd}"' in source


def test_git_calls_are_absolute_and_source_blobs_are_commit_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        stdout: str | bytes = b"blob" if not kwargs.get("text") else "text\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(cert.subprocess, "run", fake_run)
    assert cert._git_text(["status"]) == "text"
    assert cert._git_bytes(["show", "HEAD:file"]) == b"blob"
    assert calls and all(command[0] == "/usr/bin/git" for command in calls)

    payload = b"sealed source"
    source_path = tmp_path / "source.py"
    source_path.write_bytes(payload)
    commit = "1" * 40
    git_text_calls: list[list[str]] = []
    git_bytes_calls: list[list[str]] = []

    def fake_git_text(arguments: list[str]) -> str:
        git_text_calls.append(arguments)
        return "source.py"

    def fake_git_bytes(arguments: list[str]) -> bytes:
        git_bytes_calls.append(arguments)
        return payload

    monkeypatch.setattr(cert, "_git_text", fake_git_text)
    monkeypatch.setattr(cert, "_git_bytes", fake_git_bytes)
    cert._verify_sources(
        tmp_path,
        ["source.py"],
        {"source.py": hashlib.sha256(payload).hexdigest()},
        commit,
        1 << 30,
    )
    assert ["-C", str(tmp_path), "show", f"{commit}:source.py"] in git_bytes_calls
    assert all(
        arguments[:2] == ["-C", str(tmp_path)]
        for arguments in [*git_text_calls, *git_bytes_calls]
    )


def test_plan_requires_absolute_pinned_git_executable() -> None:
    plan: dict[str, object] = {key: None for key in cert.PLAN_KEYS}
    for section, keys in cert.PLAN_NESTED_KEYS.items():
        plan[section] = {key: "synthetic" for key in keys}
    plan["schema_version"] = "sireto-v4.12-strict-stores-plan-1"
    plan["sources"] = []
    sandbox = plan["sandbox"]
    sandbox.update(  # type: ignore[union-attr]
        {
            "network_allowed": False,
            "system_read_roots": ["/System", "/usr", "/opt/homebrew"],
            "device_read_literals": ["/dev/null", "/dev/urandom"],
            "device_read_subpaths": ["/dev/fd"],
            "python_framework_library": str(cert.PYTHON_FRAMEWORK_LIBRARY),
            "python_framework_library_sha256": cert.PYTHON_FRAMEWORK_LIBRARY_SHA256,
            "git_executable": "/usr/bin/git",
        }
    )
    plan["lookup"]["sample_max_sirets"] = 10_000  # type: ignore[index]
    cert.validate_plan(plan)
    plan["sandbox"]["git_executable"] = "git"  # type: ignore[index]
    with pytest.raises(cert.CertificationStopped, match="pinned runtime/tool"):
        cert.validate_plan(plan)


def test_publication_manifests_ledger_and_permissions(tmp_path: Path) -> None:
    output_root, audit_root, build_id, current, cert_path, audit_path = (
        _publication_fixture(tmp_path)
    )
    cert._chmod_tree(cert_path)
    cert._chmod_tree(audit_path)
    cert._validate_publication(cert_path, audit_path, build_id)
    assert stat.S_IMODE(cert_path.stat().st_mode) == 0o555
    assert stat.S_IMODE((cert_path / "manifest.json").stat().st_mode) == 0o444
    assert stat.S_IMODE(audit_path.stat().st_mode) == 0o555
    recovered = cert._publication_recovery(
        output_root,
        audit_root,
        build_id,
        current=current,
    )
    assert recovered == (cert_path, audit_path)


def test_publication_recovery_promotes_pending_with_audit(tmp_path: Path) -> None:
    output_root, audit_root, build_id, current, pending, audit_path = (
        _publication_fixture(tmp_path, pending=True)
    )
    cert._chmod_tree(pending)
    cert._chmod_tree(audit_path)
    recovered = cert._publication_recovery(
        output_root,
        audit_root,
        build_id,
        current=current,
    )
    final = output_root / build_id
    assert recovered == (final, audit_path)
    assert final.is_dir()
    assert not pending.exists()


def test_publication_recovery_removes_pending_without_audit(tmp_path: Path) -> None:
    output_root, audit_root, build_id, current, pending, audit_path = (
        _publication_fixture(tmp_path, pending=True)
    )
    shutil.rmtree(audit_path)
    cert._chmod_tree(pending)
    assert (
        cert._publication_recovery(
            output_root,
            audit_root,
            build_id,
            current=current,
        )
        is None
    )
    assert not pending.exists()


@pytest.mark.parametrize("state", ["final_only", "audit_only", "all_three"])
def test_publication_recovery_rejects_incoherent_states(
    tmp_path: Path,
    state: str,
) -> None:
    output_root, audit_root, build_id, current, final, audit_path = (
        _publication_fixture(tmp_path)
    )
    if state == "final_only":
        shutil.rmtree(audit_path)
    elif state == "audit_only":
        shutil.rmtree(final)
    else:
        pending = output_root / f".pending-{build_id}"
        shutil.copytree(final, pending)
        cert._chmod_tree(pending)
    if final.exists():
        cert._chmod_tree(final)
    if audit_path.exists():
        cert._chmod_tree(audit_path)
    with pytest.raises(cert.CertificationStopped, match="inconsistent durable"):
        cert._publication_recovery(
            output_root,
            audit_root,
            build_id,
            current=current,
        )


def test_publication_recovery_rejects_bad_modes_and_broken_seals(
    tmp_path: Path,
) -> None:
    output_root, audit_root, build_id, current, cert_path, audit_path = (
        _publication_fixture(tmp_path)
    )
    cert._chmod_tree(cert_path)
    cert._chmod_tree(audit_path)
    os.chmod(cert_path / "manifest.json", 0o644)
    with pytest.raises(cert.CertificationStopped, match="mode mismatch"):
        cert._publication_recovery(
            output_root,
            audit_root,
            build_id,
            current=current,
        )

    os.chmod(cert_path / "manifest.json", 0o644)
    payload = (cert_path / "manifest.json").read_bytes()
    (cert_path / "manifest.json").write_bytes(payload + b" ")
    os.chmod(cert_path / "manifest.json", 0o444)
    with pytest.raises(cert.CertificationStopped, match="JSON|records|mismatch"):
        cert._publication_recovery(
            output_root,
            audit_root,
            build_id,
            current=current,
        )


def test_publication_recovery_rejects_resealed_input_mutation(
    tmp_path: Path,
) -> None:
    output_root, audit_root, build_id, current, cert_path, audit_path = (
        _publication_fixture(tmp_path)
    )
    cert._chmod_tree(cert_path)
    cert._chmod_tree(audit_path)
    source = current["ledger_sources"][0][1]  # type: ignore[index]
    source.write_bytes(b"mutated-after-seal")
    with pytest.raises(
        cert.CertificationStopped,
        match="mutation|changed|mismatch",
    ):
        cert._publication_recovery(
            output_root,
            audit_root,
            build_id,
            current=current,
        )


@pytest.mark.skipif(not Path("/usr/bin/sandbox-exec").exists(), reason="macOS only")
def test_real_sandbox_smoke() -> None:
    cert.smoke()


@pytest.mark.skipif(not Path("/usr/bin/sandbox-exec").exists(), reason="macOS only")
def test_real_sandbox_denies_all_unlisted_capabilities(tmp_path: Path) -> None:
    root = tmp_path / "sandbox"
    work = root / "work"
    temp_root = work / "temp"
    output = work / "output"
    private = root / "private"
    oracles = root / "oracles"
    audits = root / "audits"
    for directory in (temp_root, output, private, oracles, audits):
        directory.mkdir(parents=True, exist_ok=True)
    allowed = root / "allowed.bin"
    allowed.write_bytes(b"allowed")
    outside = private / "outside.bin"
    oracle = oracles / "oracle.bin"
    audit = audits / "audit.bin"
    for path in (outside, oracle, audit):
        path.write_bytes(b"forbidden")
    run_spec = work / "run_spec.json"
    descriptor = work / "lookup_descriptor.json"
    lookup = work / "lookup.duckdb"
    for path in (run_spec, descriptor, lookup):
        path.write_bytes(b"synthetic")
    probe = work / "probe.py"
    probe.write_text(
        """
import errno
import os
import socket
import sys

allowed, outside, oracle, audit, output, temp = sys.argv[1:]
assert open(allowed, "rb").read() == b"allowed"
open(os.path.join(temp, "ok"), "wb").write(b"ok")

def denied_read(path):
    try:
        open(path, "rb")
    except OSError as exc:
        assert exc.errno == errno.EPERM, (path, exc.errno)
    else:
        raise AssertionError("read allowed: " + path)

for path in (outside, oracle, audit):
    denied_read(path)
try:
    open(os.path.join(os.path.dirname(output), "forbidden-write"), "wb")
except OSError as exc:
    assert exc.errno == errno.EPERM, exc.errno
else:
    raise AssertionError("write allowed")
for address in (("127.0.0.1", 9), ("1.1.1.1", 53)):
    sock = socket.socket()
    try:
        sock.connect(address)
    except OSError as exc:
        assert exc.errno == errno.EPERM, (address, exc.errno)
    else:
        raise AssertionError("network allowed")
    finally:
        sock.close()
try:
    os.fork()
except OSError as exc:
    assert exc.errno == errno.EPERM, exc.errno
else:
    raise AssertionError("fork allowed")
""".lstrip(),
        encoding="utf-8",
    )
    python_bin = Path(sys.executable).resolve(strict=True)
    python_app = (
        python_bin.parent.parent
        / "Resources/Python.app/Contents/MacOS/Python"
    )
    if not python_app.exists():
        python_app = python_bin
    system_roots = [
        path for path in map(Path, ("/System", "/usr", "/opt/homebrew"))
        if path.exists()
    ]
    plan = {
        "temp_root": str(temp_root),
        "lookup": {"database_path": str(lookup)},
        "sandbox": {
            "python_framework_bin": str(python_bin),
            "python_framework_app": str(python_app),
            "system_read_roots": [str(path) for path in system_roots],
            "device_read_literals": [
                path for path in ("/dev/null", "/dev/urandom")
                if Path(path).exists()
            ],
            "device_read_subpaths": [
                "/dev/fd"
            ] if Path("/dev/fd").exists() else [],
            "forbidden_oracle_manifest": str(oracles / "never-listed.json"),
            "forbidden_audit_manifest": str(audits / "never-listed.json"),
        },
    }
    template = Path("config/v4_12_strict_stores.sb").read_text(encoding="utf-8")
    effective = cert.render_profile(
        template,
        plan,
        [allowed, run_spec, descriptor, lookup],
        [probe],
    )
    profile = work / "effective.sb"
    profile.write_text(effective, encoding="utf-8")
    command = [
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile),
        "-D",
        f"RUN_ROOT={work}",
        "-D",
        f"RUN_SPEC={run_spec}",
        "-D",
        f"LOOKUP_DESCRIPTOR={descriptor}",
        "-D",
        f"RUN_OUTPUT={output}",
        "-D",
        f"RUN_TMP={temp_root}",
        "-D",
        f"PROBE_SOURCE={probe}",
        "-D",
        f"PYTHON_EXECUTABLE={python_app}",
        "-D",
        "PYTHON_FRAMEWORK_ROOT=/opt/homebrew",
        str(python_app),
        str(probe),
        str(allowed),
        str(outside),
        str(oracle),
        str(audit),
        str(output),
        str(temp_root),
    ]
    completed = subprocess.run(
        command,
        cwd=work,
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(temp_root),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (temp_root / "ok").read_bytes() == b"ok"
