from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_legacy_closed_overlay import (
    FILTER_SQL,
    _candidate_select_sql,
    _partition_counts,
    build_overlay,
)


def test_overlay_filter_targets_only_v7_excluded_closed_rows() -> None:
    assert "etatAdministratifEtablissement = 'F'" in FILTER_SQL
    assert "dateDebut IS NULL" in FILTER_SQL
    assert "dateDebut AS DATE) < DATE '2016-01-01'" in FILTER_SQL
    assert ">=" not in FILTER_SQL


def test_candidate_sql_is_label_free() -> None:
    query = _candidate_select_sql(
        establishment_path=Path("/tmp/etab.parquet"),
        legal_unit_path=Path("/tmp/ul.parquet"),
        scope_table="scope_insee",
        source_code_column="codeCommuneEtablissement",
        output_code_column="insee",
    )
    lowered = query.lower()
    assert "ground_truth" not in lowered
    assert "query_id" not in lowered
    assert "scope_insee" in lowered
    assert FILTER_SQL in query


def test_partition_counts_sum_parquet_metadata(tmp_path: Path) -> None:
    first = tmp_path / "insee=01001"
    second = tmp_path / "insee=01002"
    first.mkdir()
    second.mkdir()
    pq.write_table(
        pa.table({"siret": ["1", "2"]}),
        first / "part-0.parquet",
    )
    pq.write_table(
        pa.table({"siret": ["3"]}),
        second / "part-0.parquet",
    )
    table, total = _partition_counts(tmp_path, partition_column="insee")
    assert total == 3
    assert table.to_pylist() == [
        {"insee": "01001", "row_count": 2},
        {"insee": "01002", "row_count": 1},
    ]


def test_builder_rejects_non_positive_thread_count(tmp_path: Path) -> None:
    try:
        build_overlay(
            benchmark_path=tmp_path / "benchmark.parquet",
            benchmark_manifest_path=tmp_path / "manifest.json",
            establishment_path=tmp_path / "etab.parquet",
            legal_unit_path=tmp_path / "ul.parquet",
            output_dir=tmp_path / "output",
            memory_limit="1GB",
            threads=0,
            temp_root=tmp_path,
        )
    except ValueError as exc:
        assert str(exc) == "threads must be positive"
    else:
        raise AssertionError("Expected ValueError")
