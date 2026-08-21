"""Materialize bounded RNE/BODACC temporal layers from SIREN dossier v3.

The national dossier contains one row per observed name/address.  Retrieval
must not concatenate that full history into an unbounded bag.  This module
projects two source-separated canonical layers with at most 16 names and 16
addresses per subject, preferring current and recent evidence.  The resulting
Parquets are consumed by the existing content-addressed Tantivy overlay
builder, so the 98 GB SIRENE index is never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import duckdb


DOSSIER_TEMPORAL_LAYER_SCHEMA_VERSION = "sireto-dossier-temporal-layer-v1"
_SOURCES = ("RNE", "BODACC")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted(
        (value for value in root.rglob("*") if value.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _projection_sql(
    *,
    name_path: Path,
    address_path: Path,
    portfolio_path: Path | None,
    source: str,
    max_names: int,
    max_addresses: int,
) -> str:
    names = _sql_path(name_path)
    addresses = _sql_path(address_path)
    indexed_names_sql = (
        f"""SELECT subject_kind, siren, siret, normalized_value
            FROM read_parquet('{_sql_path(portfolio_path)}')
            WHERE (subject_kind = 'SIRET' AND selected_for_subject)
               OR (subject_kind = 'SIREN' AND selected_for_siren)"""
        if portfolio_path is not None
        else f"""SELECT DISTINCT subject_kind, siren, siret, normalized_value
                 FROM read_parquet('{names}')
                 WHERE source = 'SIRENE_CURRENT' AND normalized_value <> ''"""
    )
    empty_names = "[]::STRUCT(raw_value VARCHAR, normalized_value VARCHAR, kind VARCHAR)[]"
    return f"""
        WITH indexed_names AS (
            {indexed_names_sql}
        ),
        alternate_name_subjects AS (
            SELECT DISTINCT candidate.subject_kind, candidate.siren, candidate.siret
            FROM read_parquet('{names}') candidate
            ANTI JOIN indexed_names current
              USING (subject_kind, siren, siret, normalized_value)
            WHERE candidate.source = '{source}'
              AND candidate.normalized_value <> ''
        ),
        source_address_variants AS (
            SELECT subject_kind, siren, siret, normalized_value,
                   bool_or(coalesce(is_current, false)) AS is_current,
                   max(observed_at) AS observed_at,
                   max(valid_from) AS valid_from,
                   max(source_priority) AS source_priority
            FROM read_parquet('{addresses}')
            WHERE source = '{source}' AND normalized_value <> ''
            GROUP BY subject_kind, siren, siret, normalized_value
        ),
        source_address_ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY subject_kind,
                             CASE WHEN subject_kind = 'SIRET' THEN siret ELSE siren END
                ORDER BY is_current DESC, observed_at DESC NULLS LAST,
                         valid_from DESC NULLS LAST, source_priority DESC,
                         normalized_value
            ) AS source_rank
            FROM source_address_variants
        ),
        alternate_address_subjects AS (
            SELECT DISTINCT subject_kind, siren, siret
            FROM source_address_ranked
            WHERE source_rank > 8 AND source_rank <= 24
        ),
        affected_direct AS (
            SELECT * FROM alternate_name_subjects
            UNION
            SELECT * FROM alternate_address_subjects
        ),
        affected AS (
            SELECT * FROM affected_direct
            UNION
            SELECT 'SIREN' AS subject_kind, siren, '' AS siret
            FROM affected_direct WHERE subject_kind = 'SIRET'
        ),
        name_dedup AS (
            SELECT siren, siret, subject_kind, normalized_value,
                   arg_max(name_kind, source_priority) AS name_kind,
                   bool_or(coalesce(is_current, false)) AS is_current,
                   min(valid_from) AS valid_from,
                   max(valid_to) AS valid_to,
                   max(observed_at) AS observed_at,
                   max(source_priority) AS source_priority,
                   bool_or(indexed.normalized_value IS NOT NULL) AS is_indexed
            FROM read_parquet('{names}') candidate
            SEMI JOIN affected USING (subject_kind, siren, siret)
            LEFT JOIN indexed_names indexed
              USING (subject_kind, siren, siret, normalized_value)
            WHERE candidate.source = '{source}'
              AND normalized_value <> ''
              AND length(siren) = 9
              AND subject_kind IN ('SIREN', 'SIRET')
              AND (subject_kind = 'SIREN' OR length(siret) = 14)
            GROUP BY siren, siret, subject_kind, normalized_value
        ),
        name_ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY subject_kind,
                             CASE WHEN subject_kind = 'SIRET' THEN siret ELSE siren END
                ORDER BY is_indexed ASC, is_current DESC, observed_at DESC NULLS LAST,
                         valid_from DESC NULLS LAST, source_priority DESC,
                         normalized_value
            ) AS evidence_rank
            FROM name_dedup
        ),
        name_group AS (
            SELECT subject_kind, siren, siret,
                   list(struct_pack(
                       raw_value := normalized_value,
                       normalized_value := normalized_value,
                       kind := coalesce(name_kind, 'OTHER')
                   ) ORDER BY evidence_rank) AS names,
                   min(valid_from) AS valid_from,
                   max(valid_to) AS valid_to,
                   max(observed_at) AS observed_at,
                   bool_or(is_current) AS is_current,
                   max(source_priority) AS source_priority
            FROM name_ranked
            WHERE evidence_rank <= {max_names}
            GROUP BY subject_kind, siren, siret
        ),
        address_dedup AS (
            SELECT siren, siret, subject_kind, normalized_value,
                   arg_max(postcode, source_priority) AS postcode,
                   arg_max(insee, source_priority) AS insee,
                   arg_max(street_number, source_priority) AS number,
                   arg_max(street_number_suffix, source_priority) AS number_suffix,
                   bool_or(coalesce(is_current, false)) AS is_current,
                   min(valid_from) AS valid_from,
                   max(valid_to) AS valid_to,
                   max(observed_at) AS observed_at,
                   max(source_priority) AS source_priority
            FROM read_parquet('{addresses}') candidate
            SEMI JOIN affected USING (subject_kind, siren, siret)
            WHERE candidate.source = '{source}'
              AND normalized_value <> ''
              AND length(siren) = 9
              AND subject_kind IN ('SIREN', 'SIRET')
              AND (subject_kind = 'SIREN' OR length(siret) = 14)
              AND (insee <> '' OR postcode <> '')
            GROUP BY siren, siret, subject_kind, normalized_value
        ),
        address_ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY subject_kind,
                             CASE WHEN subject_kind = 'SIRET' THEN siret ELSE siren END
                ORDER BY is_current DESC, observed_at DESC NULLS LAST,
                         valid_from DESC NULLS LAST, source_priority DESC,
                         normalized_value
            ) AS evidence_rank
            FROM address_dedup
        ),
        parent_address_ranked AS (
            SELECT siren, '' AS siret, 'SIREN' AS subject_kind,
                   normalized_value, postcode, insee, number, number_suffix,
                   is_current, valid_from, valid_to, observed_at, source_priority,
                   row_number() OVER (
                       PARTITION BY siren
                       ORDER BY is_current DESC, observed_at DESC NULLS LAST,
                                valid_from DESC NULLS LAST, source_priority DESC,
                                normalized_value
                   ) AS evidence_rank
            FROM address_dedup
            WHERE subject_kind = 'SIRET'
        ),
        address_capped AS (
            SELECT * FROM address_ranked WHERE evidence_rank <= {max_addresses}
            UNION ALL
            SELECT * FROM parent_address_ranked WHERE evidence_rank <= {max_addresses}
        ),
        address_group AS (
            SELECT subject_kind, siren, siret,
                   list(struct_pack(
                       raw_value := normalized_value,
                       normalized_value := normalized_value,
                       postcode := coalesce(postcode, ''),
                       insee := coalesce(insee, ''),
                       number := coalesce(number, ''),
                       number_suffix := coalesce(number_suffix, '')
                   ) ORDER BY evidence_rank) AS addresses,
                   min(valid_from) AS valid_from,
                   max(valid_to) AS valid_to,
                   max(observed_at) AS observed_at,
                   bool_or(is_current) AS is_current,
                   max(source_priority) AS source_priority
            FROM address_capped
            GROUP BY subject_kind, siren, siret
        )
        SELECT '{DOSSIER_TEMPORAL_LAYER_SCHEMA_VERSION}' AS schema_version,
               sha256(concat('{source}|', address.subject_kind, '|',
                             CASE WHEN address.subject_kind = 'SIRET'
                                  THEN address.siret ELSE address.siren END)) AS evidence_id,
               '{source}' AS source,
               concat('DOSSIER_V3|', address.subject_kind, '|',
                      CASE WHEN address.subject_kind = 'SIRET'
                           THEN address.siret ELSE address.siren END) AS source_record_id,
               address.subject_kind,
               CASE WHEN address.subject_kind = 'SIRET'
                    THEN address.siret ELSE address.siren END AS subject_id,
               address.siren,
               CASE WHEN address.subject_kind = 'SIRET' THEN address.siret ELSE '' END AS siret,
               coalesce(name.names, {empty_names}) AS names,
               address.addresses,
               '' AS administrative_state,
               NULL::BOOLEAN AS is_headquarters,
               cast(least(address.valid_from, name.valid_from) AS VARCHAR) AS valid_from,
               cast(greatest(address.valid_to, name.valid_to) AS VARCHAR) AS valid_to,
               cast(greatest(address.observed_at, name.observed_at) AS VARCHAR) AS observed_at,
               coalesce(address.is_current, false) OR coalesce(name.is_current, false) AS is_current,
               greatest(address.source_priority, name.source_priority) AS source_priority
        FROM address_group address
        LEFT JOIN name_group name
          ON name.subject_kind = address.subject_kind
         AND name.siren = address.siren
         AND coalesce(name.siret, '') = coalesce(address.siret, '')
        ORDER BY address.subject_kind, address.siren, address.siret
    """


def materialize_dossier_temporal_layers(
    dossier_path: Path | str,
    output_root: Path | str,
    *,
    name_portfolio_path: Path | str | None = None,
    max_names: int = 16,
    max_addresses: int = 16,
    duckdb_temp_directory: Path | str | None = None,
    duckdb_memory_limit: str = "8GB",
    duckdb_threads: int = 8,
) -> Path:
    """Publish content-addressed source-separated canonical Parquet layers."""
    dossier_path = Path(dossier_path)
    output_root = Path(output_root)
    if not 1 <= max_names <= 64 or not 1 <= max_addresses <= 64:
        raise ValueError("name/address evidence caps must be between 1 and 64")
    if not 1 <= duckdb_threads <= 32:
        raise ValueError("duckdb_threads must be between 1 and 32")
    name_path = dossier_path / "name_evidence.parquet"
    address_path = dossier_path / "address_evidence.parquet"
    relation_path = dossier_path / "relations.parquet"
    manifest_path = dossier_path / "manifest.json"
    portfolio_path = Path(name_portfolio_path) if name_portfolio_path else None
    for path in (name_path, address_path, relation_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if portfolio_path is not None and not portfolio_path.is_file():
        raise FileNotFoundError(portfolio_path)
    identity: dict[str, Any] = {
        "schema_version": DOSSIER_TEMPORAL_LAYER_SCHEMA_VERSION,
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "dossier_manifest_sha256": _sha256_file(manifest_path),
        "name_portfolio_sha256": (
            _sha256_file(portfolio_path) if portfolio_path is not None else None
        ),
        "sources": list(_SOURCES),
        "max_names": max_names,
        "max_addresses": max_addresses,
    }
    build_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / build_hash[:16]
    if output_dir.exists():
        manifest = json.loads((output_dir / "manifest.json").read_text())
        if manifest.get("build_hash") != build_hash:
            raise RuntimeError("temporal layer build identity mismatch")
        return output_dir
    stage = Path(tempfile.mkdtemp(prefix=f".{build_hash[:16]}.building-", dir=output_root))
    connection = duckdb.connect()
    try:
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        connection.execute(f"SET threads={duckdb_threads}")
        if duckdb_temp_directory is not None:
            temp = Path(duckdb_temp_directory)
            temp.mkdir(parents=True, exist_ok=True)
            connection.execute(f"SET temp_directory='{_sql_path(temp)}'")
        layer_manifest: list[dict[str, Any]] = []
        for source in _SOURCES:
            layer = stage / source.lower()
            layer.mkdir()
            evidence_path = layer / "official_evidence.parquet"
            relation_output = layer / "official_relation.parquet"
            projection = _projection_sql(
                name_path=name_path,
                address_path=address_path,
                portfolio_path=portfolio_path,
                source=source,
                max_names=max_names,
                max_addresses=max_addresses,
            )
            connection.execute(
                f"COPY ({projection}) TO '{_sql_path(evidence_path)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
            )
            connection.execute(
                f"COPY (SELECT * FROM read_parquet('{_sql_path(relation_path)}') "
                f"WHERE source = '{source}') TO '{_sql_path(relation_output)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
            )
            evidence_count = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet('{_sql_path(evidence_path)}')"
                ).fetchone()[0]
            )
            relation_count = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet('{_sql_path(relation_output)}')"
                ).fetchone()[0]
            )
            if evidence_count == 0:
                raise RuntimeError(f"{source} temporal projection is empty")
            layer_manifest.append(
                {
                    "source": source,
                    "evidence_path": f"{source.lower()}/official_evidence.parquet",
                    "evidence_sha256": _sha256_file(evidence_path),
                    "evidence_count": evidence_count,
                    "relation_path": f"{source.lower()}/official_relation.parquet",
                    "relation_sha256": _sha256_file(relation_output),
                    "relation_count": relation_count,
                }
            )
        manifest = {
            **identity,
            "build_hash": build_hash,
            "layers": layer_manifest,
            "contains_crm_labels": False,
            "contains_synthetic_data": False,
            "raw_values_published": False,
        }
        payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        (stage / "manifest.json").write_bytes(payload)
        (stage / "manifest.sha256").write_text(
            hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
        )
        _fsync_tree(stage)
        stage.replace(output_dir)
        _fsync_directory(output_root)
        return output_dir
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        connection.close()


__all__ = [
    "DOSSIER_TEMPORAL_LAYER_SCHEMA_VERSION",
    "materialize_dossier_temporal_layers",
]
