#!/usr/bin/env python3
"""Build an indexed DuckDB SIREN→candidate store from V7 INSEE partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256


def parquet_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.parquet")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Immutable output directory exists: {args.output_dir}"
        )
    insee_dir = args.partitions_dir / "insee"
    if not insee_dir.is_dir():
        raise FileNotFoundError(f"Missing INSEE partitions: {insee_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    database_path = args.output_dir / "siren_candidates.duckdb"
    temp_dir = args.output_dir / "_duckdb_tmp"
    temp_dir.mkdir()
    parquet_glob = insee_dir / "**" / "*.parquet"

    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(f"SET memory_limit = '{args.memory_limit}'")
        connection.execute(f"SET temp_directory = '{_sql_path(temp_dir)}'")
        connection.execute(
            f"""
            CREATE TABLE candidates AS
            SELECT
                *,
                count(*) OVER (
                    PARTITION BY siren, insee, postcode
                )::INTEGER AS siren_geo_count
            FROM read_parquet(
                    '{_sql_path(parquet_glob)}',
                    hive_partitioning = true
                )
            ORDER BY siren, siret
            """
        )
        connection.execute(
            "CREATE INDEX candidates_siren_idx ON candidates(siren)"
        )
        row_count = int(
            connection.execute(
                "SELECT count(*) FROM candidates"
            ).fetchone()[0]
        )
        unique_siren_count = int(
            connection.execute(
                "SELECT count(DISTINCT siren) FROM candidates"
            ).fetchone()[0]
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    temp_dir.rmdir()

    manifest = {
        "schema_version": "v9-siren-candidate-duckdb-2",
        "partitions_dir": str(args.partitions_dir),
        "insee_parquet_sha256": parquet_tree_sha256(insee_dir),
        "row_count": row_count,
        "unique_siren_count": unique_siren_count,
        "database_sha256": file_sha256(database_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
