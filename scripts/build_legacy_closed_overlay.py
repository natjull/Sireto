#!/usr/bin/env python3
"""Build an immutable overlay for closed SIRETs excluded by the V7 store."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-legacy-closed-overlay-1"
FILTER_ID = "closed_excluded_by_v7_date_debut_2016"
FILTER_SQL = """
e.etatAdministratifEtablissement = 'F'
AND (
    e.dateDebut IS NULL
    OR CAST(e.dateDebut AS DATE) < DATE '2016-01-01'
)
""".strip()


def _escape_sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _candidate_select_sql(
    *,
    establishment_path: Path,
    legal_unit_path: Path,
    scope_table: str,
    source_code_column: str,
    output_code_column: str,
) -> str:
    """Return the label-free SQL used for one partition family."""
    if scope_table not in {"scope_insee", "scope_postcode"}:
        raise ValueError(f"Unsupported scope table: {scope_table}")
    if source_code_column not in {
        "codeCommuneEtablissement",
        "codePostalEtablissement",
    }:
        raise ValueError(f"Unsupported source code: {source_code_column}")
    if output_code_column not in {"insee", "postcode"}:
        raise ValueError(f"Unsupported output code: {output_code_column}")
    etab = _escape_sql_path(establishment_path)
    legal_unit = _escape_sql_path(legal_unit_path)
    return f"""
        SELECT
            CAST(e.siret AS VARCHAR) AS siret,
            CAST(e.siren AS VARCHAR) AS siren,
            e.denominationUsuelleEtablissement AS denomination,
            e.enseigne1Etablissement AS enseigne1,
            e.enseigne2Etablissement AS enseigne2,
            e.enseigne3Etablissement AS enseigne3,
            CAST(e.etablissementSiege AS BOOLEAN) AS etablissementSiege,
            CAST(e.etablissementSiege AS BOOLEAN) AS is_siege,
            CAST(e.numeroVoieEtablissement AS VARCHAR) AS numeroVoie,
            e.typeVoieEtablissement AS typeVoie,
            e.libelleVoieEtablissement AS libelleVoie,
            e.complementAdresseEtablissement AS complementAdresse,
            CAST(e.codePostalEtablissement AS VARCHAR) AS postcode,
            e.libelleCommuneEtablissement AS city,
            CAST(e.codeCommuneEtablissement AS VARCHAR) AS insee,
            CAST(ul.categorieJuridiqueUniteLegale AS VARCHAR) AS cj_ul,
            e.etatAdministratifEtablissement AS etat_admin,
            CAST(e.dateDernierTraitementEtablissement AS TIMESTAMP)
                AS last_treatment_date,
            ul.sigleUniteLegale AS sigle_ul,
            ul.denominationUniteLegale AS denomination_ul,
            NULLIF(
                TRIM(CONCAT_WS(
                    ' ',
                    ul.denominationUsuelle1UniteLegale,
                    ul.denominationUsuelle2UniteLegale,
                    ul.denominationUsuelle3UniteLegale
                )),
                ''
            ) AS denomination_usuelle_ul,
            ul.nomUniteLegale AS nom_ul,
            ul.prenomUsuelUniteLegale AS prenom_usuel_ul,
            CAST(NULL AS VARCHAR) AS pm_dirigeant_names
        FROM read_parquet('{etab}') e
        LEFT JOIN read_parquet('{legal_unit}') ul
          ON e.siren = ul.siren
        INNER JOIN {scope_table} scope
          ON CAST(e.{source_code_column} AS VARCHAR) = scope.code
        WHERE {FILTER_SQL}
          AND e.siret IS NOT NULL
          AND LENGTH(CAST(e.siret AS VARCHAR)) = 14
          AND e.{source_code_column} IS NOT NULL
    """


def _partition_counts(
    root: Path,
    *,
    partition_column: str,
) -> tuple[pa.Table, int]:
    rows: list[dict[str, Any]] = []
    total = 0
    prefix = f"{partition_column}="
    for directory in sorted(root.glob(f"{prefix}*")):
        if not directory.is_dir():
            continue
        count = 0
        for parquet_path in sorted(directory.glob("*.parquet")):
            count += int(pq.ParquetFile(parquet_path).metadata.num_rows)
        if count:
            rows.append(
                {
                    partition_column: directory.name[len(prefix) :],
                    "row_count": count,
                }
            )
            total += count
    schema = pa.schema(
        [
            (partition_column, pa.string()),
            ("row_count", pa.int64()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema), total


def _data_inventory(root: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for path in sorted(
        (item for item in root.rglob("*.parquet") if "manifest" not in item.parts),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = file_sha256(path)
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(size).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(digest.encode("ascii"))
        hasher.update(b"\n")
        total_bytes += size
        file_count += 1
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "sha256": hasher.hexdigest(),
    }


def _copy_partition_family(
    connection: duckdb.DuckDBPyConnection,
    *,
    query: str,
    output_root: Path,
    partition_column: str,
) -> None:
    output_sql = _escape_sql_path(output_root)
    connection.execute(
        f"""
        COPY ({query})
        TO '{output_sql}'
        (
            FORMAT PARQUET,
            PARTITION_BY ({partition_column}),
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000,
            OVERWRITE_OR_IGNORE
        )
        """
    )


def build_overlay(
    *,
    benchmark_path: Path,
    benchmark_manifest_path: Path,
    establishment_path: Path,
    legal_unit_path: Path,
    output_dir: Path,
    memory_limit: str,
    temp_root: Path,
) -> dict[str, Any]:
    """Build the overlay without selecting or inspecting benchmark labels."""
    if output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {output_dir}")
    benchmark_manifest = json.loads(
        benchmark_manifest_path.read_text(encoding="utf-8")
    )
    expected_benchmark_hash = benchmark_manifest.get("output_sha256", {}).get(
        benchmark_path.name
    )
    if file_sha256(benchmark_path) != expected_benchmark_hash:
        raise ValueError("Benchmark hash does not match its frozen manifest")
    if file_sha256(establishment_path) != benchmark_manifest.get(
        "establishment_snapshot_sha256"
    ):
        raise ValueError("Establishment snapshot hash mismatch")
    if file_sha256(legal_unit_path) != benchmark_manifest.get(
        "legal_unit_snapshot_sha256"
    ):
        raise ValueError("Legal-unit snapshot hash mismatch")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.building-",
            dir=output_dir.parent,
        )
    )
    duckdb_temp = Path(
        tempfile.mkdtemp(prefix="sireto-overlay-duckdb-", dir=temp_root)
    )
    try:
        insee_root = build_root / "insee"
        postcode_root = build_root / "cp"
        insee_root.mkdir()
        postcode_root.mkdir()
        connection = duckdb.connect()
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(
            f"SET temp_directory = '{_escape_sql_path(duckdb_temp)}'"
        )
        benchmark_sql = _escape_sql_path(benchmark_path)
        # Projection is deliberately limited to location fields. Labels and
        # ground-truth identifiers never enter the build relation.
        connection.execute(
            f"""
            CREATE TEMP TABLE scope_insee AS
            SELECT DISTINCT CAST(insee AS VARCHAR) AS code
            FROM read_parquet('{benchmark_sql}')
            WHERE insee IS NOT NULL AND TRIM(CAST(insee AS VARCHAR)) != ''
            """
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE scope_postcode AS
            SELECT DISTINCT CAST(postcode AS VARCHAR) AS code
            FROM read_parquet('{benchmark_sql}')
            WHERE postcode IS NOT NULL
              AND TRIM(CAST(postcode AS VARCHAR)) != ''
            """
        )
        scope_counts = {
            "insee": int(
                connection.execute(
                    "SELECT COUNT(*) FROM scope_insee"
                ).fetchone()[0]
            ),
            "postcode": int(
                connection.execute(
                    "SELECT COUNT(*) FROM scope_postcode"
                ).fetchone()[0]
            ),
        }
        _copy_partition_family(
            connection,
            query=_candidate_select_sql(
                establishment_path=establishment_path,
                legal_unit_path=legal_unit_path,
                scope_table="scope_insee",
                source_code_column="codeCommuneEtablissement",
                output_code_column="insee",
            ),
            output_root=insee_root,
            partition_column="insee",
        )
        _copy_partition_family(
            connection,
            query=_candidate_select_sql(
                establishment_path=establishment_path,
                legal_unit_path=legal_unit_path,
                scope_table="scope_postcode",
                source_code_column="codePostalEtablissement",
                output_code_column="postcode",
            ),
            output_root=postcode_root,
            partition_column="postcode",
        )
        connection.close()

        manifest_dir = build_root / "manifest"
        manifest_dir.mkdir()
        insee_counts, insee_rows = _partition_counts(
            insee_root,
            partition_column="insee",
        )
        postcode_counts, postcode_rows = _partition_counts(
            postcode_root,
            partition_column="postcode",
        )
        if not insee_rows or not postcode_rows:
            raise RuntimeError("Overlay build produced an empty partition family")
        pq.write_table(
            insee_counts,
            manifest_dir / "insee_counts.parquet",
            compression="zstd",
        )
        pq.write_table(
            postcode_counts,
            manifest_dir / "postcode_counts.parquet",
            compression="zstd",
        )
        inventory = _data_inventory(build_root)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "benchmark_build_id": benchmark_manifest["build_id"],
            "benchmark_sha256": expected_benchmark_hash,
            "benchmark_manifest_sha256": file_sha256(
                benchmark_manifest_path
            ),
            "establishment_snapshot_sha256": benchmark_manifest[
                "establishment_snapshot_sha256"
            ],
            "legal_unit_snapshot_sha256": benchmark_manifest[
                "legal_unit_snapshot_sha256"
            ],
            "filter_id": FILTER_ID,
            "filter_sql": FILTER_SQL,
            "scope_columns_read": ["insee", "postcode"],
            "scope_counts": scope_counts,
            "rows": {
                "insee_physical": insee_rows,
                "postcode_physical": postcode_rows,
            },
            "partitions": {
                "insee": int(insee_counts.num_rows),
                "postcode": int(postcode_counts.num_rows),
            },
            "data_inventory": inventory,
            "memory_limit": memory_limit,
            "command": [sys.executable, *sys.argv],
        }
        (build_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(build_root, output_dir)
        return manifest
    except BaseException:
        # Only the private build directory created by this invocation is
        # removed. Existing project artefacts are never touched.
        shutil.rmtree(build_root, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(duckdb_temp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument(
        "--establishment-snapshot",
        type=Path,
        default=Path("data/StockEtablissement_utf8.parquet"),
    )
    parser.add_argument(
        "--legal-unit-snapshot",
        type=Path,
        default=Path("data/StockUniteLegale_utf8.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-limit", default="12GB")
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp"),
    )
    args = parser.parse_args()
    args.temp_root.mkdir(parents=True, exist_ok=True)
    manifest = build_overlay(
        benchmark_path=args.benchmark,
        benchmark_manifest_path=args.benchmark_manifest,
        establishment_path=args.establishment_snapshot,
        legal_unit_path=args.legal_unit_snapshot,
        output_dir=args.output_dir,
        memory_limit=args.memory_limit,
        temp_root=args.temp_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
