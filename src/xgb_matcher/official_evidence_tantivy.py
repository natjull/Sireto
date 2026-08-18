"""Projection and separate Tantivy overlay for canonical official evidence.

The overlay deliberately uses the field contract of ``TantivyBackend`` but is
stored separately from the national index.  It can be rebuilt or discarded
without touching the 98 GB base index.  Only normalized values reach Tantivy;
raw official values remain in ``official_evidence.parquet`` for audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import duckdb

from .hierarchical_retrieval import (
    TantivyBackend,
    character_ngrams,
    normalize_code,
    normalize_insee,
    normalize_text,
)


OFFICIAL_EVIDENCE_INDEX_SCHEMA_VERSION = "sireto-official-evidence-index-v1"
_DUCKDB_MEMORY_LIMIT = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+)?(?:KB|MB|GB|TB)$", re.I)


@dataclass(frozen=True)
class OfficialEvidenceIndexDocument:
    document_type: str
    siret: str
    siren: str
    insee: str
    postcode: str
    number: str
    number_suffix: str
    state: str
    is_headquarters: bool
    names: tuple[str, ...]
    addresses: tuple[str, ...]
    linked_sirets: tuple[str, ...]
    linked_sirens: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.document_type not in {"siret", "siren"}:
            raise ValueError("official index document_type must be siret or siren")

    def retrieval_payload(self) -> dict[str, Any]:
        """Payload intentionally excludes every raw source value."""
        return {
            "siret": self.siret,
            "siren": self.siren,
            "insee": self.insee,
            "postcode": self.postcode,
            "numeroVoie": self.number,
            "number_suffix": self.number_suffix,
            "etat_admin": self.state,
            "is_siege": self.is_headquarters,
            "names": list(self.names),
            "addresses": list(self.addresses),
            "official_evidence_ids": list(self.evidence_ids),
            "official_evidence_sources": list(self.evidence_sources),
            "official_linked_sirens": list(self.linked_sirens),
            "official_overlay": True,
        }


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _rows(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    batch_size: int,
) -> Iterator[Mapping[str, Any]]:
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description]
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        for row in batch:
            yield dict(zip(columns, row, strict=True))


def _tuple_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    else:
        values = value
    return tuple(sorted({str(item) for item in values if item not in (None, "")}))


def _projection_ctes(evidence_path: Path, relation_path: Path) -> str:
    evidence = _sql_path(evidence_path)
    relation = _sql_path(relation_path)
    return f"""
        WITH evidence AS (
            SELECT * FROM read_parquet('{evidence}')
        ),
        relation AS (
            SELECT * FROM read_parquet('{relation}')
        ),
        siren_meta AS (
            SELECT siren,
                   list_distinct(list_transform(flatten(list(names)),
                       item -> item.normalized_value)) AS names,
                   list_distinct(list(evidence_id)) AS evidence_ids,
                   list_distinct(list(source)) AS evidence_sources,
                   arg_max(administrative_state, source_priority)
                       FILTER (WHERE administrative_state <> '') AS state
            FROM evidence
            WHERE subject_kind = 'SIREN'
            GROUP BY siren
        ),
        siret_meta AS (
            SELECT siret, any_value(siren) AS siren,
                   list_distinct(list_transform(flatten(list(names)),
                       item -> item.normalized_value)) AS names,
                   list_distinct(list(evidence_id)) AS evidence_ids,
                   list_distinct(list(source)) AS evidence_sources,
                   arg_max(administrative_state, source_priority)
                       FILTER (WHERE administrative_state <> '') AS state,
                   arg_max(is_headquarters, source_priority)
                       FILTER (WHERE is_headquarters IS NOT NULL) AS is_headquarters
            FROM evidence
            WHERE subject_kind = 'SIRET'
            GROUP BY siret
        ),
        siret_address AS (
            SELECT DISTINCT siret, siren,
                   item.unnest.normalized_value AS address,
                   item.unnest.insee AS insee,
                   item.unnest.postcode AS postcode,
                   item.unnest.number AS number,
                   item.unnest.number_suffix AS number_suffix
            FROM evidence, unnest(addresses) item
            WHERE subject_kind = 'SIRET'
              AND (item.unnest.insee <> '' OR item.unnest.postcode <> '')
        ),
        siren_address AS (
            SELECT DISTINCT siren,
                   item.unnest.normalized_value AS address,
                   item.unnest.insee AS insee,
                   item.unnest.postcode AS postcode
            FROM evidence, unnest(addresses) item
            WHERE subject_kind = 'SIREN'
              AND (item.unnest.insee <> '' OR item.unnest.postcode <> '')
        ),
        siret_links AS (
            SELECT identifier,
                   list_distinct(list(linked_identifier)) AS linked_identifiers
            FROM (
                SELECT from_identifier identifier, to_identifier linked_identifier
                FROM relation
                WHERE relation_type = 'ESTABLISHMENT_SUCCESSION'
                  AND from_kind = 'SIRET' AND to_kind = 'SIRET'
                UNION ALL
                SELECT to_identifier, from_identifier
                FROM relation
                WHERE relation_type = 'ESTABLISHMENT_SUCCESSION'
                  AND from_kind = 'SIRET' AND to_kind = 'SIRET'
            ) links
            GROUP BY identifier
        ),
        siren_links AS (
            SELECT identifier,
                   list_distinct(list(linked_identifier)) AS linked_identifiers
            FROM (
                SELECT from_identifier identifier, to_identifier linked_identifier
                FROM relation
                WHERE relation_type IN ('LEGAL_UNIT_SUCCESSION', 'ASSET_TRANSFER')
                  AND from_kind = 'SIREN' AND to_kind = 'SIREN'
                UNION ALL
                SELECT to_identifier, from_identifier
                FROM relation
                WHERE relation_type IN ('LEGAL_UNIT_SUCCESSION', 'ASSET_TRANSFER')
                  AND from_kind = 'SIREN' AND to_kind = 'SIREN'
            ) links
            GROUP BY identifier
        )
    """


def iter_official_evidence_documents(
    evidence_path: Path | str,
    relation_path: Path | str,
    *,
    batch_size: int = 4096,
    connection: duckdb.DuckDBPyConnection | None = None,
    duckdb_temp_directory: Path | str | None = None,
    duckdb_memory_limit: str | None = None,
) -> Iterator[OfficialEvidenceIndexDocument]:
    """Stream normalized overlay documents, grouping on disk through DuckDB."""
    evidence_path = Path(evidence_path)
    relation_path = Path(relation_path)
    if not evidence_path.is_file() or not relation_path.is_file():
        raise FileNotFoundError("canonical evidence and relation Parquets are required")
    owns_connection = connection is None
    connection = connection or duckdb.connect()
    connection.execute("SET preserve_insertion_order=false")
    if duckdb_temp_directory is not None:
        temp_directory = Path(duckdb_temp_directory)
        temp_directory.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"SET temp_directory='{_sql_path(temp_directory)}'"
        )
    if duckdb_memory_limit is not None:
        if not _DUCKDB_MEMORY_LIMIT.fullmatch(duckdb_memory_limit.strip()):
            raise ValueError("invalid DuckDB memory limit")
        connection.execute(
            f"SET memory_limit='{duckdb_memory_limit.strip().upper()}'"
        )
    ctes = _projection_ctes(evidence_path, relation_path)
    siret_sql = ctes + """
        SELECT 'siret' AS document_type,
               address.siret, address.siren, address.insee, address.postcode,
               address.number, address.number_suffix,
               coalesce(meta.state, '') AS state,
               coalesce(meta.is_headquarters, false) AS is_headquarters,
               list_distinct(list_concat(coalesce(meta.names, []),
                                         coalesce(legal.names, []))) AS names,
               [address.address] AS addresses,
               coalesce(siret_links.linked_identifiers, []) AS linked_sirets,
               coalesce(siren_links.linked_identifiers, []) AS linked_sirens,
               list_distinct(list_concat(coalesce(meta.evidence_ids, []),
                                         coalesce(legal.evidence_ids, []))) AS evidence_ids,
               list_distinct(list_concat(coalesce(meta.evidence_sources, []),
                                         coalesce(legal.evidence_sources, []))) AS evidence_sources
        FROM siret_address address
        JOIN siret_meta meta USING (siret, siren)
        LEFT JOIN siren_meta legal USING (siren)
        LEFT JOIN siret_links ON siret_links.identifier = address.siret
        LEFT JOIN siren_links ON siren_links.identifier = address.siren
        ORDER BY address.siret, address.insee, address.postcode, address.address
    """
    try:
        for row in _rows(connection, siret_sql, batch_size=batch_size):
            yield _document_from_row(row)

        # SIREN documents cover both legal-unit addresses and establishment
        # sites.  This lets a new RNE/BODACC legal name retrieve a SIREN and
        # expand its current SIRENE sites through the unchanged base backend.
        siren_sql = ctes + """
            SELECT 'siren' AS document_type, '' AS siret, address.siren,
                   address.insee, address.postcode, '' AS number,
                   '' AS number_suffix, coalesce(legal.state, '') AS state,
                   false AS is_headquarters,
                   list_distinct(list_concat(coalesce(site.names, []),
                                             coalesce(legal.names, []))) AS names,
                   list_distinct(list(address.address)) AS addresses,
                   [] AS linked_sirets,
                   coalesce(siren_links.linked_identifiers, []) AS linked_sirens,
                   list_distinct(list_concat(coalesce(site.evidence_ids, []),
                                             coalesce(legal.evidence_ids, []))) AS evidence_ids,
                   list_distinct(list_concat(coalesce(site.evidence_sources, []),
                                             coalesce(legal.evidence_sources, []))) AS evidence_sources
            FROM (
                SELECT siren, insee, postcode, address FROM siret_address
                UNION ALL
                SELECT siren, insee, postcode, address FROM siren_address
            ) address
            LEFT JOIN (
                SELECT siren,
                       list_distinct(flatten(list(names))) AS names,
                       list_distinct(flatten(list(evidence_ids))) AS evidence_ids,
                       list_distinct(flatten(list(evidence_sources))) AS evidence_sources
                FROM siret_meta GROUP BY siren
            ) site USING (siren)
            LEFT JOIN siren_meta legal USING (siren)
            LEFT JOIN siren_links ON siren_links.identifier = address.siren
            GROUP BY address.siren, address.insee, address.postcode,
                     legal.state, site.names, legal.names,
                     siren_links.linked_identifiers,
                     site.evidence_ids, legal.evidence_ids,
                     site.evidence_sources, legal.evidence_sources
            ORDER BY address.siren, address.insee, address.postcode
        """
        for row in _rows(connection, siren_sql, batch_size=batch_size):
            yield _document_from_row(row)
    finally:
        if owns_connection:
            connection.close()


def _document_from_row(row: Mapping[str, Any]) -> OfficialEvidenceIndexDocument:
    return OfficialEvidenceIndexDocument(
        document_type=str(row["document_type"]),
        siret=normalize_code(row.get("siret"), 14) if row.get("siret") else "",
        siren=normalize_code(row.get("siren"), 9),
        insee=normalize_insee(row.get("insee")),
        postcode=normalize_code(row.get("postcode"), 5) if row.get("postcode") else "",
        number=normalize_text(row.get("number")),
        number_suffix=normalize_text(row.get("number_suffix")),
        state=normalize_text(row.get("state")),
        is_headquarters=bool(row.get("is_headquarters")),
        names=_tuple_strings(row.get("names")),
        addresses=_tuple_strings(row.get("addresses")),
        linked_sirets=_tuple_strings(row.get("linked_sirets")),
        linked_sirens=_tuple_strings(row.get("linked_sirens")),
        evidence_ids=_tuple_strings(row.get("evidence_ids")),
        evidence_sources=_tuple_strings(row.get("evidence_sources")),
    )


def official_evidence_tantivy_schema(tantivy: Any) -> Any:
    """Build the base-compatible schema plus one overlay-only stored field."""
    builder = tantivy.SchemaBuilder()
    for field_name in ["document_type", "siret", "siren", "insee", "postcode"]:
        builder.add_text_field(
            field_name, stored=True, tokenizer_name="raw", index_option="basic"
        )
    for field_name in ["names", "addresses", "name_ngrams", "address_ngrams"]:
        builder.add_text_field(
            field_name, stored=field_name in {"names", "addresses"}
        )
    for field_name in ["names_exact", "addresses_exact"]:
        builder.add_text_field(
            field_name, stored=False, tokenizer_name="raw", index_option="basic"
        )
    for field_name in [
        "number",
        "state",
        "is_siege",
        "linked_sirets",
        "linked_sirens",
        "payload",
    ]:
        builder.add_text_field(
            field_name, stored=True, tokenizer_name="raw", index_option="basic"
        )
    return builder.build()


def _add_tantivy_document(writer: Any, tantivy: Any, row: OfficialEvidenceIndexDocument) -> None:
    document = tantivy.Document()
    scalar = {
        "document_type": row.document_type,
        "siret": row.siret,
        "siren": row.siren,
        "insee": row.insee,
        "postcode": row.postcode,
        "number": row.number,
        "state": row.state or "A",
        "is_siege": "1" if row.is_headquarters else "0",
        "linked_sirets": " ".join(row.linked_sirets),
        "linked_sirens": " ".join(row.linked_sirens),
        "payload": json.dumps(
            row.retrieval_payload(), sort_keys=True, separators=(",", ":")
        ),
    }
    for key, value in scalar.items():
        document.add_text(key, value)
    for name in row.names:
        document.add_text("names", name)
        document.add_text("names_exact", name)
        document.add_text("name_ngrams", " ".join(character_ngrams(name)))
    for address in row.addresses:
        document.add_text("addresses", address)
        document.add_text("addresses_exact", address)
        document.add_text("address_ngrams", " ".join(character_ngrams(address)))
    writer.add_document(document)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_official_evidence_tantivy_overlay(
    evidence_path: Path | str,
    relation_path: Path | str,
    output_root: Path | str,
    *,
    writer_heap_bytes: int = 256_000_000,
    writer_threads: int = 4,
    commit_every: int = 250_000,
    batch_size: int = 4096,
    duckdb_temp_directory: Path | str | None = None,
    duckdb_memory_limit: str | None = None,
) -> Path:
    """Build a content-addressed overlay; the national base index is untouched."""
    try:
        import tantivy  # type: ignore
    except ImportError as error:
        raise RuntimeError("official evidence overlay requires tantivy==0.25.1") from error
    evidence_path = Path(evidence_path)
    relation_path = Path(relation_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_sha256 = {
        "official_evidence": _sha256_file(evidence_path),
        "official_relation": _sha256_file(relation_path),
    }
    identity = {
        "schema_version": OFFICIAL_EVIDENCE_INDEX_SCHEMA_VERSION,
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "source_sha256": source_sha256,
        "config": {
            "writer_heap_bytes": writer_heap_bytes,
            "writer_threads": writer_threads,
            "commit_every": commit_every,
            "batch_size": batch_size,
            "duckdb_memory_limit": duckdb_memory_limit,
            "duckdb_spill_enabled": duckdb_temp_directory is not None,
        },
    }
    build_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_dir = output_root / build_hash[:16]
    if output_dir.exists():
        manifest_path = output_dir / "manifest.json"
        receipt_path = output_dir / "manifest.sha256"
        if not manifest_path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"incomplete official evidence overlay: {output_dir}")
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if receipt_path.read_text(encoding="ascii").strip() != manifest_sha:
            raise RuntimeError(f"official evidence manifest checksum mismatch: {output_dir}")
        manifest = json.loads(manifest_bytes)
        if manifest.get("build_hash") != build_hash:
            raise RuntimeError(f"official evidence build identity mismatch: {output_dir}")
        return output_dir
    stage = Path(
        tempfile.mkdtemp(prefix=f".{build_hash[:16]}.building-", dir=output_root)
    )
    try:
        index_path = stage / "tantivy"
        index_path.mkdir()
        index = tantivy.Index(
            official_evidence_tantivy_schema(tantivy), path=str(index_path)
        )
        writer = index.writer(heap_size=writer_heap_bytes, num_threads=writer_threads)
        count = 0
        siret_count = 0
        siren_count = 0
        for document in iter_official_evidence_documents(
            evidence_path,
            relation_path,
            batch_size=batch_size,
            duckdb_temp_directory=duckdb_temp_directory,
            duckdb_memory_limit=duckdb_memory_limit,
        ):
            _add_tantivy_document(writer, tantivy, document)
            count += 1
            siret_count += int(document.document_type == "siret")
            siren_count += int(document.document_type == "siren")
            if count % commit_every == 0:
                writer.commit()
        writer.commit()
        index.reload()
        manifest = {
            **identity,
            "build_hash": build_hash,
            "backend": "tantivy",
            "overlay": True,
            "base_index_modified": False,
            "contains_crm_labels": False,
            "raw_values_indexed": False,
            "sources": {
                "official_evidence": {
                    "path": str(evidence_path.resolve()),
                    "sha256": source_sha256["official_evidence"],
                },
                "official_relation": {
                    "path": str(relation_path.resolve()),
                    "sha256": source_sha256["official_relation"],
                },
            },
            "num_documents": count,
            "num_establishment_documents": siret_count,
            "num_siren_documents": siren_count,
            "exclusions": [
                "beneficial_owners",
                "directors",
                "bodacc_full_text",
                "relations_inferred_from_text",
                "crm_labels",
            ],
        }
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        (stage / "manifest.json").write_bytes(manifest_bytes)
        (stage / "manifest.sha256").write_text(
            hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="ascii"
        )
        _fsync_tree(stage)
        stage.replace(output_dir)
        _fsync_directory(output_root)
        return output_dir
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Durably publish Tantivy segments and receipts before atomic rename."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


class OfficialEvidenceTantivyBackend(TantivyBackend):
    """Base-compatible backend exposing overlay-only SIREN relations."""

    def linked_sirens(self, siren: str, limit: int = 256) -> tuple[str, ...]:
        siren = normalize_code(siren, 9)
        parsed = self._tantivy.Query.boolean_query(
            [
                (
                    self._tantivy.Occur.Must,
                    self._tantivy.Query.term_query(
                        self.index.schema, "document_type", "siren", "basic"
                    ),
                ),
                (
                    self._tantivy.Occur.Must,
                    self._tantivy.Query.term_query(
                        self.index.schema, "siren", siren, "basic"
                    ),
                ),
            ]
        )
        result = self.searcher.search(parsed, limit=limit)
        linked: set[str] = set()
        for _score, address in result.hits:
            document = self.searcher.doc(address).to_dict()
            value = self._first(document, "linked_sirens", "")
            linked.update(str(value or "").split())
        linked.discard(siren)
        return tuple(sorted(linked))
