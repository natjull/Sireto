from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.xgb_matcher.partitioned_store import PartitionedCandidateStore


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _build_minimal_partitions(root: Path) -> None:
    _write_parquet(
        root / "insee" / "insee=01001" / "data.parquet",
        [{"siret": "01001000000001", "siren": "010010000", "postcode": "01000", "city": "TEST"}],
    )
    _write_parquet(
        root / "insee" / "insee=01002" / "data.parquet",
        [
            {"siret": "01002000000001", "siren": "010020000", "postcode": "01000", "city": "TEST"},
            {"siret": "01002000000002", "siren": "010020000", "postcode": "01000", "city": "TEST"},
        ],
    )
    _write_parquet(
        root / "cp" / "postcode=01000" / "data.parquet",
        [{"siret": "01000000000001", "siren": "010000000", "insee": "01001", "city": "TEST"}],
    )


def test_insee_count_uses_manifest_when_available(tmp_path: Path) -> None:
    _build_minimal_partitions(tmp_path)
    manifest_path = tmp_path / "manifest" / "insee_counts.parquet"
    _write_parquet(
        manifest_path,
        [
            {"insee": "01001", "row_count": 42, "is_mega": False},
            {"insee": "01002", "row_count": 77, "is_mega": False},
        ],
    )

    store = PartitionedCandidateStore(tmp_path)

    assert store._count_insee_rows("01001") == 42
    assert store._count_insee_rows("01002") == 77


def test_insee_count_falls_back_to_dataset_when_manifest_missing_code(tmp_path: Path) -> None:
    _build_minimal_partitions(tmp_path)
    manifest_path = tmp_path / "manifest" / "insee_counts.parquet"
    _write_parquet(
        manifest_path,
        [{"insee": "01001", "row_count": 42, "is_mega": False}],
    )

    store = PartitionedCandidateStore(tmp_path)

    assert store._count_insee_rows("01002") == 2


def test_loaded_pool_returns_exact_dense_partition_key(tmp_path: Path) -> None:
    _build_minimal_partitions(tmp_path)
    store = PartitionedCandidateStore(tmp_path)

    insee_rows, insee_key = store.load_by_insee_then_postcode_with_key(
        "01001",
        "01000",
        mega_insee_policy="full_insee",
    )
    cp_rows, cp_key = store.load_by_insee_then_postcode_with_key(
        None,
        "01000",
    )

    assert len(insee_rows) == 1
    assert insee_key == "01001_"
    assert len(cp_rows) == 1
    assert cp_key == "_01000"


def test_filtered_mega_pool_refuses_incompatible_dense_key(tmp_path: Path) -> None:
    _build_minimal_partitions(tmp_path)
    store = PartitionedCandidateStore(tmp_path)
    store._manifest_insee_counts["01001"] = 200_000

    rows, key = store.load_by_insee_then_postcode_with_key(
        "01001",
        "01000",
        mega_insee_max_rows=100_000,
        mega_insee_policy="cp_filter_insee",
    )

    assert rows
    assert key is None
