#!/usr/bin/env python3
"""Build a memory-mapped SIREN-to-geography lookup for V9 expansion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--batch-size", type=int, default=500_000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Immutable output directory exists: {args.output_dir}"
        )
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sorted_path = args.output_dir / "_sorted_siren_geo.parquet"
    temp_dir = args.output_dir / "_duckdb_tmp"
    temp_dir.mkdir()

    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit = '{args.memory_limit}'")
        connection.execute(f"SET temp_directory = '{_sql_path(temp_dir)}'")
        connection.execute(
            f"""
            COPY (
                SELECT
                    lpad(
                        regexp_replace(CAST(siren AS VARCHAR), '[^0-9]', '', 'g'),
                        9,
                        '0'
                    ) AS siren,
                    coalesce(CAST(insee AS VARCHAR), '') AS insee,
                    coalesce(CAST(postcode AS VARCHAR), '') AS postcode,
                    greatest(coalesce(CAST(siret_count AS INTEGER), 0), 0)
                        AS siret_count
                FROM read_parquet('{_sql_path(args.source)}')
                WHERE length(
                    regexp_replace(CAST(siren AS VARCHAR), '[^0-9]', '', 'g')
                ) > 0
                ORDER BY siren, siret_count DESC, insee, postcode
            )
            TO '{_sql_path(sorted_path)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
            """
        )
    finally:
        connection.close()

    parquet = pq.ParquetFile(sorted_path)
    row_count = parquet.metadata.num_rows
    arrays = {
        "sirens.npy": np.lib.format.open_memmap(
            args.output_dir / "sirens.npy",
            mode="w+",
            dtype="S9",
            shape=(row_count,),
        ),
        "insee.npy": np.lib.format.open_memmap(
            args.output_dir / "insee.npy",
            mode="w+",
            dtype="S5",
            shape=(row_count,),
        ),
        "postcodes.npy": np.lib.format.open_memmap(
            args.output_dir / "postcodes.npy",
            mode="w+",
            dtype="S10",
            shape=(row_count,),
        ),
        "siret_counts.npy": np.lib.format.open_memmap(
            args.output_dir / "siret_counts.npy",
            mode="w+",
            dtype=np.int32,
            shape=(row_count,),
        ),
    }
    offset = 0
    previous_siren = b""
    for batch in parquet.iter_batches(batch_size=args.batch_size):
        size = len(batch)
        sirens = np.asarray(batch.column(0).to_pylist(), dtype="S9")
        insee = np.asarray(batch.column(1).to_pylist(), dtype="S5")
        postcodes = np.asarray(batch.column(2).to_pylist(), dtype="S10")
        counts = np.asarray(batch.column(3).to_numpy(), dtype=np.int32)
        if previous_siren and sirens[0] < previous_siren:
            raise ValueError("Sorted SIREN geography stream is not monotonic")
        if size > 1 and np.any(sirens[1:] < sirens[:-1]):
            raise ValueError("Sorted SIREN geography batch is not monotonic")
        arrays["sirens.npy"][offset : offset + size] = sirens
        arrays["insee.npy"][offset : offset + size] = insee
        arrays["postcodes.npy"][offset : offset + size] = postcodes
        arrays["siret_counts.npy"][offset : offset + size] = counts
        previous_siren = bytes(sirens[-1])
        offset += size
        if offset % 5_000_000 < size:
            print(
                json.dumps(
                    {
                        "stage": "materialize_mmap",
                        "rows": offset,
                        "total": row_count,
                    }
                ),
                flush=True,
            )
    if offset != row_count:
        raise ValueError(
            f"Materialized row count mismatch: {offset} != {row_count}"
        )
    for array in arrays.values():
        array.flush()
    del arrays
    sorted_path.unlink()
    temp_dir.rmdir()

    output_names = [
        "sirens.npy",
        "insee.npy",
        "postcodes.npy",
        "siret_counts.npy",
    ]
    manifest = {
        "schema_version": "v9-siren-geo-mmap-1",
        "source": str(args.source),
        "source_sha256": file_sha256(args.source),
        "row_count": row_count,
        "sort_order": [
            "siren ASC",
            "siret_count DESC",
            "insee ASC",
            "postcode ASC",
        ],
        "outputs": {
            name: file_sha256(args.output_dir / name)
            for name in output_names
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
