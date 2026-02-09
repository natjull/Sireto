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
