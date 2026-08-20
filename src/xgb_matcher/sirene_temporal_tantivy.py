"""Tantivy rescue overlay for the truthful SIRENE temporal projection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import duckdb

from .hierarchical_retrieval import character_ngrams, normalize_text
from .official_evidence_tantivy import official_evidence_tantivy_schema
from .sirene_temporal import SIRENE_TEMPORAL_SCHEMA_VERSION


SIRENE_TEMPORAL_INDEX_SCHEMA_VERSION = "sireto-sirene-temporal-index-v2"


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projection_files(projection_dir: Path) -> tuple[Path, Path, Path, Mapping[str, Any]]:
    manifest_path = projection_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SIRENE_TEMPORAL_SCHEMA_VERSION:
        raise ValueError("unsupported SIRENE temporal projection")
    if manifest.get("semantics", {}).get("historical_addresses_claimed") is not False:
        raise ValueError("temporal projection must not claim historical addresses")
    paths = tuple(
        projection_dir / name
        for name in (
            "historical_siret_identity.parquet",
            "historical_siren_identity.parquet",
            "succession_edges.parquet",
        )
    )
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("temporal projection Parquets are incomplete")
    return (*paths, manifest)


def _rows(cursor: duckdb.DuckDBPyConnection, batch_size: int = 4096) -> Iterator[Mapping[str, Any]]:
    columns = [item[0] for item in cursor.description]
    while batch := cursor.fetchmany(batch_size):
        for values in batch:
            yield dict(zip(columns, values, strict=True))


def iter_temporal_documents(projection_dirs: Sequence[Path | str]) -> Iterator[dict[str, Any]]:
    """Yield historical SIRET and geo/SIREN docs with one-hop links."""
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order=false")
    try:
        for raw_dir in projection_dirs:
            projection_dir = Path(raw_dir).resolve()
            siret_path, siren_path, relation_path, _manifest = _projection_files(projection_dir)
            siret_cursor = connection.execute(
                f"""
                WITH history AS (
                  SELECT * FROM read_parquet('{_sql_path(siret_path)}')
                  QUALIFY row_number() OVER(PARTITION BY siret ORDER BY valid_to DESC NULLS LAST,valid_from DESC NULLS LAST,normalized_name)<=6
                ), links AS (
                  SELECT identifier,list_sort(list_distinct(list(linked))) linked_sirets,
                         list_sort(list_distinct(list(direction||':'||linked))) linked_siret_edges
                  FROM (
                    SELECT predecessor_siret identifier,successor_siret linked,'SUCCESSOR' direction FROM read_parquet('{_sql_path(relation_path)}')
                    UNION ALL
                    SELECT successor_siret,predecessor_siret,'PREDECESSOR' FROM read_parquet('{_sql_path(relation_path)}')
                  ) GROUP BY identifier
                )
                SELECT 'siret' document_type,h.siret,h.siren,h.insee,h.postcode,h.number,h.number_suffix,
                       h.current_state state,h.is_headquarters,
                       list_sort(list_distinct(list(h.normalized_name))) AS name_values,
                       list_sort(list_distinct(list(h.geo_scaffold_address))) AS geo_scaffold_addresses,
                       coalesce(l.linked_sirets,[]) AS linked_sirets,
                       coalesce(l.linked_siret_edges,[]) AS linked_siret_edges,
                       list_sort(list_distinct(list(h.evidence_id))) AS evidence_ids,
                       min(h.valid_from) AS valid_from,max(h.valid_to) AS valid_to
                FROM history h LEFT JOIN links l ON l.identifier=h.siret
                GROUP BY h.siret,h.siren,h.insee,h.postcode,h.number,h.number_suffix,
                         h.current_state,h.is_headquarters,l.linked_sirets,l.linked_siret_edges
                ORDER BY h.siret
                """
            )
            yield from _rows(siret_cursor)
            siren_cursor = connection.execute(
                f"""
                WITH history AS (
                  SELECT * FROM read_parquet('{_sql_path(siren_path)}')
                  QUALIFY row_number() OVER(PARTITION BY siren,insee ORDER BY valid_to DESC NULLS LAST,valid_from DESC NULLS LAST,normalized_name)<=12
                )
                SELECT 'siren' document_type,'' siret,siren,insee,postcode,'' number,'' number_suffix,
                       current_state state,false is_headquarters,
                       list_sort(list_distinct(list(normalized_name))) AS name_values,
                       [] AS geo_scaffold_addresses,[] AS linked_sirets,[] AS linked_siret_edges,
                       list_sort(list_distinct(list(evidence_id))) AS evidence_ids,
                       min(valid_from) AS valid_from,max(valid_to) AS valid_to
                FROM history
                GROUP BY siren,insee,postcode,current_state
                ORDER BY siren,insee,postcode
                """
            )
            yield from _rows(siren_cursor)
            relation_cursor = connection.execute(
                f"""
                SELECT 'siret' AS document_type,identifier AS siret,substr(identifier,1,9) AS siren,
                       '' AS insee,'' AS postcode,'' AS number,'' AS number_suffix,
                       'A' AS state,false AS is_headquarters,
                       cast([] AS VARCHAR[]) AS name_values,
                       cast([] AS VARCHAR[]) AS geo_scaffold_addresses,
                       list_sort(list_distinct(list(linked))) AS linked_sirets,
                       list_sort(list_distinct(list(direction||':'||linked))) AS linked_siret_edges,
                       list_sort(list_distinct(list(relation_id))) AS evidence_ids,
                       min(effective_date) AS valid_from,max(effective_date) AS valid_to
                FROM (
                  SELECT predecessor_siret AS identifier,successor_siret AS linked,'SUCCESSOR' AS direction,relation_id,effective_date
                  FROM read_parquet('{_sql_path(relation_path)}')
                  UNION ALL
                  SELECT successor_siret,predecessor_siret,'PREDECESSOR',relation_id,effective_date
                  FROM read_parquet('{_sql_path(relation_path)}')
                )
                GROUP BY identifier ORDER BY identifier
                """
            )
            yield from _rows(relation_cursor)
    finally:
        connection.close()


def _add_document(writer: Any, tantivy: Any, row: Mapping[str, Any]) -> None:
    document = tantivy.Document()
    names = tuple(dict.fromkeys(normalize_text(item) for item in row["name_values"] if item))
    geo_addresses = tuple(
        dict.fromkeys(normalize_text(item) for item in row["geo_scaffold_addresses"] if item)
    )
    evidence_sources = ["SIRENE_HISTORY"] if names else []
    if row["linked_sirets"]:
        evidence_sources.append("SIRENE_SUCCESSION")
    payload = {
        "siret": str(row["siret"] or ""), "siren": str(row["siren"] or ""),
        "insee": str(row["insee"] or ""), "postcode": str(row["postcode"] or ""),
        "numeroVoie": str(row["number"] or ""), "number_suffix": str(row["number_suffix"] or ""),
        "etat_admin": str(row["state"] or "A"), "is_siege": bool(row["is_headquarters"]),
        "names": list(names), "historical_names": list(names),
        # Stored for candidate comparison, deliberately not added to the indexed
        # ``addresses`` field because it is not a historical address observation.
        "addresses": list(geo_addresses), "geo_scaffold": bool(geo_addresses),
        "geo_scaffold_source": "SIRENE_CURRENT_ESTABLISHMENT" if geo_addresses else "",
        "linked_sirets": list(row["linked_sirets"] or []),
        "official_succession_edges": list(row["linked_siret_edges"] or []),
        "official_evidence_ids": list(row["evidence_ids"] or []),
        "official_evidence_sources": evidence_sources,
        "temporal_valid_from": str(row["valid_from"] or ""),
        "temporal_valid_to": str(row["valid_to"] or ""),
    }
    scalars = {
        "document_type": row["document_type"], "siret": row["siret"], "siren": row["siren"],
        "insee": row["insee"], "postcode": row["postcode"], "number": row["number"],
        "state": row["state"] or "A", "is_siege": "1" if row["is_headquarters"] else "0",
        "linked_sirets": " ".join(row["linked_sirets"] or []), "linked_sirens": "",
        "payload": json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }
    for key, value in scalars.items():
        document.add_text(key, str(value or ""))
    for name in names:
        document.add_text("names", name)
        document.add_text("names_exact", name)
        document.add_text("name_ngrams", " ".join(character_ngrams(name)))
    writer.add_document(document)


def build_temporal_tantivy_overlay(
    projection_dirs: Sequence[Path | str],
    output_root: Path | str,
    *,
    writer_heap_bytes: int = 256_000_000,
    writer_threads: int = 4,
    commit_every: int = 250_000,
) -> Path:
    try:
        import tantivy  # type: ignore
    except ImportError as error:
        raise RuntimeError("temporal overlay requires tantivy==0.25.1") from error
    canonical_dirs = tuple(Path(path).resolve() for path in projection_dirs)
    if not canonical_dirs:
        raise ValueError("at least one temporal projection is required")
    sources = []
    for directory in canonical_dirs:
        siret, siren, relations, manifest = _projection_files(directory)
        sources.append(
            {
                "projection_build_id": manifest["build_id"],
                "historical_siret_sha256": _sha256_file(siret),
                "historical_siren_sha256": _sha256_file(siren),
                "successions_sha256": _sha256_file(relations),
            }
        )
    identity = {
        "schema_version": SIRENE_TEMPORAL_INDEX_SCHEMA_VERSION,
        "sources": sources,
        "config": {"writer_heap_bytes": writer_heap_bytes, "writer_threads": writer_threads},
    }
    build_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / build_hash[:16]
    if (output_dir / "manifest.json").is_file():
        return output_dir
    if output_dir.exists():
        raise RuntimeError(f"incomplete temporal overlay: {output_dir}")
    stage = Path(tempfile.mkdtemp(prefix=f".{build_hash[:16]}.building-", dir=output_root))
    try:
        index_dir = stage / "tantivy"
        index_dir.mkdir()
        index = tantivy.Index(official_evidence_tantivy_schema(tantivy), path=str(index_dir))
        writer = index.writer(heap_size=writer_heap_bytes, num_threads=writer_threads)
        count = siret_count = siren_count = 0
        for row in iter_temporal_documents(canonical_dirs):
            _add_document(writer, tantivy, row)
            count += 1
            siret_count += row["document_type"] == "siret"
            siren_count += row["document_type"] == "siren"
            if count % commit_every == 0:
                writer.commit()
        if not count or not siret_count or not siren_count:
            raise RuntimeError(
                f"temporal index hard gate: documents={count}, siret={siret_count}, siren={siren_count}"
            )
        writer.commit()
        manifest = {
            **identity, "build_hash": build_hash, "backend": "tantivy",
            "overlay_kind": "SIRENE_TEMPORAL_RESCUE", "num_documents": count,
            "num_establishment_documents": int(siret_count), "num_siren_documents": int(siren_count),
            "historical_addresses_indexed": False, "geo_scaffold_stored_only": True,
            "succession_max_hops": 1, "exact_protection_allowed": False,
            "contains_crm_labels": False, "contains_synthetic_rows": False,
        }
        payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        (stage / "manifest.json").write_bytes(payload)
        (stage / "manifest.sha256").write_text(hashlib.sha256(payload).hexdigest()+"\n", encoding="ascii")
        stage.replace(output_dir)
        fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return output_dir
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = ["SIRENE_TEMPORAL_INDEX_SCHEMA_VERSION", "build_temporal_tantivy_overlay", "iter_temporal_documents"]
