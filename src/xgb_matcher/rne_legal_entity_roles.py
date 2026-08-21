"""Structured RNE legal-entity roles and their bounded retrieval overlay.

Only representatives whose RNE ``typeDePersonne`` is ``ENTREPRISE`` are
accepted.  Natural persons, addresses, dates of birth and beneficial owners
never enter this table or index.  A role denomination is retrieval evidence
for the represented SIREN; it is deliberately kept outside that company's
official name bag.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Iterator, Mapping
import zipfile

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .hierarchical_retrieval import (
    TantivyBackend,
    character_ngrams,
    normalize_code,
    normalize_insee,
    normalize_text,
)


RNE_LEGAL_ENTITY_ROLE_SCHEMA_VERSION = "sireto-rne-legal-entity-role-v1"
RNE_LEGAL_ENTITY_ROLE_INDEX_VERSION = "sireto-rne-legal-entity-role-index-v1"
_SIREN = re.compile(r"^[0-9]{9}$")


ROLE_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("role_id", pa.string()),
        ("source_record_id", pa.string()),
        ("subject_siren", pa.string()),
        ("related_siren", pa.string()),
        ("related_denomination", pa.string()),
        ("normalized_denomination", pa.string()),
        ("role_code", pa.string()),
        ("role_label", pa.string()),
        ("effective_date", pa.string()),
        ("active", pa.bool_()),
        ("establishment_sirets", pa.list_(pa.string())),
        ("observed_at", pa.string()),
        ("source", pa.string()),
    ]
)


@dataclass(frozen=True)
class RneLegalEntityRole:
    source_record_id: str
    subject_siren: str
    related_denomination: str
    related_siren: str = ""
    role_code: str = ""
    role_label: str = ""
    effective_date: str = ""
    active: bool = True
    establishment_sirets: tuple[str, ...] = ()
    observed_at: str = ""

    def __post_init__(self) -> None:
        subject = normalize_code(self.subject_siren, 9)
        related = normalize_code(self.related_siren, 9)
        denomination = str(self.related_denomination or "").strip()
        normalized = normalize_text(denomination)
        if not _SIREN.fullmatch(subject):
            raise ValueError("legal-entity role requires a valid subject SIREN")
        if related and not _SIREN.fullmatch(related):
            raise ValueError("invalid related legal-entity SIREN")
        if related == subject:
            raise ValueError("legal-entity role self-loop")
        if not normalized:
            raise ValueError("legal-entity role requires a denomination")
        record_id = str(self.source_record_id or "").strip()
        if not record_id:
            raise ValueError("legal-entity role requires source_record_id")
        establishments = tuple(
            sorted(
                {
                    value
                    for raw in self.establishment_sirets
                    if len(value := normalize_code(raw, 14)) == 14
                }
            )
        )
        object.__setattr__(self, "source_record_id", record_id)
        object.__setattr__(self, "subject_siren", subject)
        object.__setattr__(self, "related_siren", related)
        object.__setattr__(self, "related_denomination", denomination)
        object.__setattr__(self, "establishment_sirets", establishments)

    @property
    def role_id(self) -> str:
        payload = json.dumps(
            {
                "source_record_id": self.source_record_id,
                "subject_siren": self.subject_siren,
                "related_siren": self.related_siren,
                "denomination": normalize_text(self.related_denomination),
                "role_code": self.role_code,
                "effective_date": self.effective_date,
                "establishments": self.establishment_sirets,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RNE_LEGAL_ENTITY_ROLE_SCHEMA_VERSION,
            "role_id": self.role_id,
            "source_record_id": self.source_record_id,
            "subject_siren": self.subject_siren,
            "related_siren": self.related_siren,
            "related_denomination": self.related_denomination,
            "normalized_denomination": normalize_text(self.related_denomination),
            "role_code": str(self.role_code or ""),
            "role_label": str(self.role_label or ""),
            "effective_date": str(self.effective_date or "")[:10],
            "active": bool(self.active),
            "establishment_sirets": list(self.establishment_sirets),
            "observed_at": str(self.observed_at or "")[:10],
            "source": "RNE",
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _branches(record: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    roots = [record, _mapping(record.get("formality")), _mapping(record.get("formalite"))]
    for root in roots:
        content = _mapping(root.get("content"))
        for key in ("personneMorale", "personnePhysique", "exploitation"):
            branch = content.get(key)
            if isinstance(branch, Mapping):
                yield branch


def _establishment_sirets(power: Mapping[str, Any]) -> tuple[str, ...]:
    output: list[str] = []
    for key in (
        "siretEtablissement",
        "siretEtablissementConcerne",
        "siretEtablissementRattachement",
    ):
        value = normalize_code(power.get(key), 14)
        if len(value) == 14:
            output.append(value)
    for key in ("etablissements", "etablissementsConcernes"):
        values = power.get(key) or []
        if isinstance(values, Mapping):
            values = [values]
        if isinstance(values, list):
            for item in values:
                value = normalize_code(
                    _mapping(item).get("siret") if isinstance(item, Mapping) else item,
                    14,
                )
                if len(value) == 14:
                    output.append(value)
    return tuple(sorted(set(output)))


def extract_legal_entity_roles(record: Mapping[str, Any]) -> tuple[RneLegalEntityRole, ...]:
    root = _mapping(record.get("formality")) or record
    subject = normalize_code(record.get("siren") or root.get("siren"), 9)
    if len(subject) != 9:
        return ()
    record_id = str(record.get("id") or root.get("id") or subject)
    observed_at = str(record.get("updatedAt") or root.get("updatedAt") or "")[:10]
    output: list[RneLegalEntityRole] = []
    for branch in _branches(record):
        composition = _mapping(branch.get("composition"))
        powers = composition.get("pouvoirs") or []
        if not isinstance(powers, list):
            continue
        for ordinal, raw_power in enumerate(powers):
            power = _mapping(raw_power)
            enterprise = _mapping(power.get("entreprise"))
            if str(power.get("typeDePersonne") or "").upper() != "ENTREPRISE" and not enterprise:
                continue
            denomination = str(
                enterprise.get("denomination")
                or enterprise.get("nomEntreprise")
                or power.get("denomination")
                or ""
            ).strip()
            related = normalize_code(
                enterprise.get("siren")
                or _mapping(enterprise.get("descriptionPersonne")).get("siren")
                or power.get("siren"),
                9,
            )
            if not denomination:
                continue
            try:
                output.append(
                    RneLegalEntityRole(
                        source_record_id=f"{record_id}:power:{ordinal}",
                        subject_siren=subject,
                        related_siren=related,
                        related_denomination=denomination,
                        role_code=str(power.get("roleEntreprise") or enterprise.get("roleEntreprise") or ""),
                        role_label=str(power.get("autreRoleEntreprise") or enterprise.get("autreRoleEntreprise") or ""),
                        effective_date=str(
                            power.get("dateEffetRoleDeclarant")
                            or enterprise.get("dateEffetRoleDeclarant")
                            or ""
                        )[:10],
                        active=bool(power.get("actif", True)),
                        establishment_sirets=_establishment_sirets(power),
                        observed_at=observed_at,
                    )
                )
            except ValueError:
                continue
    unique = {role.role_id: role for role in output}
    return tuple(unique[key] for key in sorted(unique))


def iter_rne_zip_records(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        import ijson  # type: ignore
    except ImportError as exc:
        raise RuntimeError("RNE bulk extraction requires ijson") from exc
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() != ".json":
                continue
            with archive.open(info) as stream:
                for value in ijson.items(stream, "item"):
                    if isinstance(value, Mapping):
                        yield value


def write_legal_entity_roles(
    records: Iterable[Mapping[str, Any]], output: Path, *, batch_size: int = 8192
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.unlink(missing_ok=True)
    writer = pq.ParquetWriter(temporary, ROLE_SCHEMA, compression="zstd")
    batch: list[dict[str, Any]] = []
    record_count = role_count = structured_count = 0
    try:
        for record in records:
            record_count += 1
            for role in extract_legal_entity_roles(record):
                batch.append(role.to_dict())
                role_count += 1
                structured_count += int(bool(role.related_siren))
                if len(batch) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=ROLE_SCHEMA))
                    batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=ROLE_SCHEMA))
        writer.close()
        os.replace(temporary, output)
    except Exception:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return {
        "source_records": record_count,
        "legal_entity_roles": role_count,
        "structured_related_siren_roles": structured_count,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_index_schema(tantivy: Any) -> Any:
    builder = tantivy.SchemaBuilder()
    for field_name in ("document_type", "siret", "siren", "insee", "postcode"):
        builder.add_text_field(field_name, stored=True, tokenizer_name="raw", index_option="basic")
    for field_name in (
        "names", "addresses", "name_ngrams", "address_ngrams", "legal_names",
        "trade_names", "site_names", "historical_names", "supporting_names",
        "historical_addresses", "supporting_addresses", "current_name_ngrams",
    ):
        builder.add_text_field(field_name, stored=field_name in {"names", "addresses"})
    for field_name in (
        "names_exact", "addresses_exact", "legal_names_exact", "trade_names_exact",
        "site_names_exact", "number", "state", "is_siege", "linked_sirets",
        "linked_sirens", "official_evidence_ids", "official_evidence_sources", "payload",
    ):
        builder.add_text_field(field_name, stored=True, tokenizer_name="raw", index_option="basic")
    return builder.build()


def build_legal_entity_role_index(
    *, roles_path: Path, siren_documents_path: Path, output_root: Path
) -> Path:
    """Build a geo-filterable SIREN-only Tantivy overlay from structured roles."""
    try:
        import tantivy  # type: ignore
    except ImportError as exc:
        raise RuntimeError("legal-entity role index requires tantivy") from exc
    inputs = {
        "roles_sha256": sha256_file(roles_path),
        "siren_documents_sha256": sha256_file(siren_documents_path),
        "schema_version": RNE_LEGAL_ENTITY_ROLE_INDEX_VERSION,
    }
    build_id = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    target = output_root / build_id[:16]
    if (target / "manifest.json").is_file():
        return target
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id[:16]}.", dir=output_root))
    connection = duckdb.connect()
    try:
        role_sql = str(roles_path.resolve()).replace("'", "''")
        document_sql = str(siren_documents_path.resolve()).replace("'", "''")
        query = f"""
            WITH roles AS (
                SELECT subject_siren AS siren,
                       string_agg(DISTINCT normalized_denomination, ' | ') AS role_names,
                       string_agg(DISTINCT role_id, ' ') AS role_ids,
                       count(*) AS role_count,
                       count(*) FILTER (WHERE related_siren <> '') AS structured_count
                FROM read_parquet('{role_sql}')
                WHERE active AND normalized_denomination <> ''
                GROUP BY subject_siren
            )
            SELECT d.siren, d.insee, d.postcode, r.role_names, r.role_ids,
                   r.role_count, r.structured_count
            FROM read_parquet('{document_sql}') d
            JOIN roles r USING (siren)
            WHERE d.document_kind = 'SIREN'
              AND (d.insee <> '' OR d.postcode <> '')
        """
        schema = _role_index_schema(tantivy)
        (temporary / "tantivy").mkdir()
        index = tantivy.Index(schema, path=str(temporary / "tantivy"), reuse=False)
        writer = index.writer(600_000_000)
        cursor = connection.execute(query)
        count = 0
        while True:
            rows = cursor.fetchmany(8192)
            if not rows:
                break
            for siren, insee, postcode, names_raw, role_ids, role_count, structured_count in rows:
                names = tuple(
                    dict.fromkeys(
                        value
                        for part in str(names_raw or "").split("|")
                        if (value := normalize_text(part))
                    )
                )
                if not names:
                    continue
                document = tantivy.Document()
                scalar = {
                    "document_type": "siren",
                    "siret": "",
                    "siren": normalize_code(siren, 9),
                    "insee": normalize_insee(insee),
                    "postcode": normalize_code(postcode, 5),
                    "number": "",
                    "state": "A",
                    "is_siege": "0",
                    "linked_sirets": "",
                    "linked_sirens": "",
                    "official_evidence_ids": str(role_ids or ""),
                    "official_evidence_sources": "RNE_LEGAL_ENTITY_ROLE",
                    "payload": json.dumps(
                        {"role_count": int(role_count), "structured_role_count": int(structured_count)},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
                for key, value in scalar.items():
                    document.add_text(key, value)
                for name in names:
                    document.add_text("names", name)
                    document.add_text("names_exact", name)
                    document.add_text("name_ngrams", " ".join(character_ngrams(name)))
                writer.add_document(document)
                count += 1
        writer.commit()
        index.reload()
        manifest = {
            "schema_version": RNE_LEGAL_ENTITY_ROLE_INDEX_VERSION,
            "build_id": build_id,
            "num_documents": count,
            "document_type": "siren",
            "overlay_kind": "RNE_LEGAL_ENTITY_ROLE",
            "temporal_complete": True,
            "missing_temporal_sources": [],
            "inputs": inputs,
            "privacy": {
                "natural_persons_indexed": False,
                "beneficial_owners_indexed": False,
                "role_addresses_indexed": False,
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        connection.close()


class LegalEntityRoleTantivyBackend(TantivyBackend):
    """Tantivy backend whose geographic documents are SIREN documents only."""

    def _has_insee(self, insee: str) -> bool:
        normalized = normalize_insee(insee)
        if normalized not in self._insee_exists:
            query = self._tantivy.Query.boolean_query(
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
                            self.index.schema, "insee", normalized, "basic"
                        ),
                    ),
                ]
            )
            self._insee_exists[normalized] = bool(self.searcher.search(query, limit=1).hits)
        return self._insee_exists[normalized]
