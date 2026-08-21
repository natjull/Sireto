"""Structured public-sector facility evidence for SIRET retrieval.

The layer accepts identifiers published by FINESS, the Education directory
and DINUM's administration list.  It never resolves a facility to a SIRET by
text similarity: a structured SIRET/SIREN is mandatory at ingestion.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping
import unicodedata

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .hierarchical_retrieval import TantivyBackend, character_ngrams, normalize_code, normalize_insee, normalize_text


SECTOR_EVIDENCE_SCHEMA_VERSION = "sireto-official-sector-evidence-v1"
SECTOR_INDEX_SCHEMA_VERSION = "sireto-official-sector-index-v1"

SECTOR_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("evidence_id", pa.string()),
        ("source", pa.string()),
        ("source_record_id", pa.string()),
        ("subject_kind", pa.string()),
        ("siret", pa.string()),
        ("siren", pa.string()),
        ("name", pa.string()),
        ("normalized_name", pa.string()),
        ("address", pa.string()),
        ("normalized_address", pa.string()),
        ("insee", pa.string()),
        ("postcode", pa.string()),
        ("number", pa.string()),
        ("active", pa.bool_()),
        ("observed_at", pa.string()),
    ]
)


def _evidence(
    *, source: str, source_record_id: Any, siret: Any = "", siren: Any = "",
    name: Any, address: Any = "", insee: Any = "", postcode: Any = "",
    number: Any = "", active: bool = True, observed_at: str = "",
) -> dict[str, Any] | None:
    normalized_siret = normalize_code(siret, 14)
    normalized_siren = normalize_code(siren or normalized_siret[:9], 9)
    normalized_name = normalize_text(name)
    if len(normalized_siren) != 9 or not normalized_name:
        return None
    subject_kind = "SIRET" if len(normalized_siret) == 14 else "SIREN"
    payload = {
        "source": source,
        "source_record_id": str(source_record_id or ""),
        "subject_kind": subject_kind,
        "siret": normalized_siret,
        "siren": normalized_siren,
        "name": str(name or "").strip(),
        "address": str(address or "").strip(),
        "insee": normalize_insee(insee),
        "postcode": normalize_code(postcode, 5),
        "number": normalize_text(number),
        "active": bool(active),
        "observed_at": str(observed_at or "")[:10],
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SECTOR_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": identity,
        **payload,
        "normalized_name": normalized_name,
        "normalized_address": normalize_text(address),
    }


def iter_finess(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import ijson  # type: ignore
    except ImportError as exc:
        raise RuntimeError("FINESS streaming requires ijson") from exc
    with gzip.open(path, "rb") as stream:
        for entity in ijson.items(stream, "pmej.item"):
            general = entity.get("informationsGeneralesPMEJ") or {}
            siren = general.get("siren")
            for site in entity.get("ege") or []:
                info = site.get("informationsGeneralesEGE") or {}
                addresses = site.get("adresse") or [{}]
                for address in addresses[:1]:
                    value = _evidence(
                        source="FINESS",
                        source_record_id=info.get("numFinessEge") or info.get("egeId"),
                        siret=info.get("siret"),
                        siren=siren,
                        name=info.get("nomEgeLong") or info.get("nomEgeCourt"),
                        address=address.get("ligneQuatre") or " ".join(
                            str(address.get(key) or "")
                            for key in ("numeroVoie", "typeVoie", "libelleVoie")
                        ),
                        insee=address.get("cogCommune"),
                        postcode=address.get("codePostal"),
                        number=address.get("numeroVoie"),
                        active=str(site.get("etatObjet") or "A").upper() != "I",
                        observed_at=str(site.get("dateDerniereMaj") or "")[:10],
                    )
                    if value:
                        yield value
            addresses = entity.get("adresse") or [{}]
            for address in addresses[:1]:
                value = _evidence(
                    source="FINESS",
                    source_record_id=general.get("numFinessPm") or general.get("pmSmsseId"),
                    siren=siren,
                    name=general.get("denominationLonguePmSmsse") or general.get("denominationPm"),
                    address=address.get("ligneQuatre") or "",
                    insee=address.get("cogCommune"),
                    postcode=address.get("codePostal"),
                    number=address.get("numeroVoie"),
                    active=str(entity.get("etatObjet") or "A").upper() != "I",
                    observed_at=str(entity.get("dateDerniereMaj") or "")[:10],
                )
                if value:
                    yield value


def _csv_rows(path: Path, delimiter: str) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter=delimiter)


def _normalized_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(character for character in text if not unicodedata.combining(character))
    return "_".join(normalize_text(ascii_text).lower().split())


def _normalized_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {_normalized_key(key): value for key, value in row.items()}


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    normalized = _normalized_row(row)
    for key in keys:
        value = normalized.get(_normalized_key(key))
        if value not in (None, "", [], {}):
            return value
    return ""


def _iter_records(path: Path) -> Iterator[Mapping[str, Any]]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=8192):
            yield from batch.to_pylist()
        return
    if suffixes.endswith((".jsonl", ".ndjson")):
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        yield value
        return
    if suffixes.endswith((".json", ".json.gz")):
        try:
            import ijson  # type: ignore
        except ImportError as exc:
            raise RuntimeError("JSON sector snapshots require ijson") from exc
        opener = gzip.open if suffixes.endswith(".gz") else open
        with opener(path, "rb") as stream:
            prefix = "item"
            for value in ijson.items(stream, prefix):
                if isinstance(value, Mapping):
                    yield value
        return
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(64 * 1024)
        stream.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ";"
        yield from csv.DictReader(stream, delimiter=delimiter)


def iter_data_es(path: Path) -> Iterator[dict[str, Any]]:
    """Yield sports facilities only when Data ES publishes a structured ID."""
    for raw in _iter_records(path):
        row = _normalized_row(raw)
        siret = _first(
            row,
            "siret", "installation_siret", "inst_siret", "siret_installation",
            "siret_proprietaire", "proprio_siret", "caracteristiques_administratives_siret",
        )
        siren = _first(row, "siren", "installation_siren", "inst_siren", "proprio_siren")
        identifier = normalize_code(siret, 14)
        parent = normalize_code(siren or identifier[:9], 9)
        if len(identifier) != 14 and len(parent) != 9:
            continue
        installation_name = _first(
            row,
            "nom_installation", "installation_nom", "inst_nom", "nom_du_lieu",
            "nom_installation_sportive", "libelle_installation",
        )
        equipment_name = _first(
            row,
            "nom_equipement", "equipement_nom", "equip_nom", "libelle_equipement",
        )
        name = equipment_name or installation_name
        address = _first(
            row,
            "adresse", "installation_adresse", "inst_adresse", "adresse_installation",
            "adresse_complete", "adresse_complete_de_l_installation",
        )
        value = _evidence(
            source="DATA_ES",
            source_record_id=_first(
                row, "numero_equipement", "equip_numero", "numero_installation",
                "inst_numero", "id_equipement", "id_installation",
            ),
            siret=identifier,
            siren=parent,
            name=name,
            address=address,
            insee=_first(row, "code_insee", "insee", "inst_com_insee", "code_commune"),
            postcode=_first(row, "code_postal", "postcode", "inst_cp", "installation_cp"),
            number=_first(row, "numero_voie", "adresse_numero", "inst_num_voie"),
            active=str(_first(row, "etat", "statut", "etat_installation") or "A").upper()
            not in {"F", "I", "FERME", "FERMEE", "INACTIF", "INACTIVE"},
            observed_at=str(_first(row, "date_mise_a_jour", "date_maj", "date_actualisation"))[:10],
        )
        if value:
            yield value
        # Keep the installation label as a second auditable observation when
        # it differs from the individual equipment name.
        if installation_name and normalize_text(installation_name) != normalize_text(name):
            parent_value = _evidence(
                source="DATA_ES",
                source_record_id=str(_first(row, "numero_installation", "inst_numero")) + "|INSTALLATION",
                siret=identifier,
                siren=parent,
                name=installation_name,
                address=address,
                insee=_first(row, "code_insee", "insee", "inst_com_insee", "code_commune"),
                postcode=_first(row, "code_postal", "postcode", "inst_cp", "installation_cp"),
                number=_first(row, "numero_voie", "adresse_numero", "inst_num_voie"),
                active=True,
                observed_at=str(_first(row, "date_mise_a_jour", "date_maj", "date_actualisation"))[:10],
            )
            if parent_value:
                yield parent_value


def _association_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    data = record.get("data")
    return data if isinstance(data, Mapping) else record


def iter_associations(path: Path) -> Iterator[dict[str, Any]]:
    """Project API Association snapshots without indexing free-text objects."""
    for record in _iter_records(path):
        payload = _association_payload(record)
        siren = _first(payload, "siren", "numero_siren")
        headquarters_siret = _first(payload, "siret_siege", "siret", "numero_siret")
        title = _first(payload, "titre", "nom", "denomination", "titre_court")
        address = payload.get("adresse_siege") or payload.get("adresse") or {}
        if not isinstance(address, Mapping):
            address = {"adresse": address}
        legal = _evidence(
            source="API_ASSOCIATION",
            source_record_id=_first(payload, "rna", "id", "numero_rna") or siren,
            siret=headquarters_siret,
            siren=siren,
            name=title,
            address=_first(address, "adresse", "libelle", "voie", "adresse_complete"),
            insee=_first(address, "code_insee", "insee", "code_commune"),
            postcode=_first(address, "code_postal", "postcode"),
            number=_first(address, "numero_voie", "numero"),
            active=str(_first(payload, "active", "etat", "statut") or "A").upper()
            not in {"F", "DISSOUTE", "DISSOLUTION", "INACTIVE"},
            observed_at=str(_first(payload, "date_mise_a_jour", "updated_at", "date_maj"))[:10],
        )
        if legal:
            yield legal
        establishments = payload.get("etablissements") or payload.get("sites") or []
        if isinstance(establishments, Mapping):
            establishments = [establishments]
        for ordinal, raw_site in enumerate(establishments):
            if not isinstance(raw_site, Mapping):
                continue
            site_address = raw_site.get("adresse") or {}
            if not isinstance(site_address, Mapping):
                site_address = {"adresse": site_address}
            site = _evidence(
                source="API_ASSOCIATION",
                source_record_id=f"{_first(payload, 'rna', 'id', 'numero_rna')}|SITE|{ordinal}",
                siret=_first(raw_site, "siret", "numero_siret"),
                siren=siren,
                name=_first(raw_site, "nom", "enseigne", "denomination") or title,
                address=_first(site_address, "adresse", "libelle", "voie", "adresse_complete"),
                insee=_first(site_address, "code_insee", "insee", "code_commune"),
                postcode=_first(site_address, "code_postal", "postcode"),
                number=_first(site_address, "numero_voie", "numero"),
                active=str(_first(raw_site, "active", "etat", "statut") or "A").upper()
                not in {"F", "FERME", "FERMEE", "INACTIF", "INACTIVE"},
                observed_at=str(_first(payload, "date_mise_a_jour", "updated_at", "date_maj"))[:10],
            )
            if site and site["subject_kind"] == "SIRET":
                yield site


def iter_education(path: Path) -> Iterator[dict[str, Any]]:
    for row in _csv_rows(path, ";"):
        identifier = normalize_code(row.get("SIREN_SIRET"), 14)
        if len(identifier) not in {9, 14}:
            continue
        value = _evidence(
            source="EDUCATION_UAI",
            source_record_id=row.get("Identifiant_de_l_etablissement"),
            siret=identifier if len(identifier) == 14 else "",
            siren=identifier[:9],
            name=row.get("Nom_etablissement"),
            address=" ".join(
                str(row.get(key) or "") for key in ("Adresse_1", "Adresse_2", "Adresse_3")
            ),
            insee=row.get("Code_commune"),
            postcode=row.get("Code_postal"),
            active=str(row.get("etat") or "OUVERT").upper() == "OUVERT",
            observed_at=str(row.get("date_maj_ligne") or "")[:10],
        )
        if value:
            yield value


def iter_administrations(path: Path) -> Iterator[dict[str, Any]]:
    for row in _csv_rows(path, ","):
        value = _evidence(
            source="DINUM_ADMINISTRATION",
            source_record_id=row.get("siren"),
            siren=row.get("siren"),
            name=row.get("nom_complet"),
            active=True,
        )
        if value:
            yield value


def write_sector_evidence(rows: Iterable[Mapping[str, Any]], output: Path, batch_size: int = 8192) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    writer = pq.ParquetWriter(temporary, SECTOR_SCHEMA, compression="zstd")
    batch: list[Mapping[str, Any]] = []
    count = 0
    try:
        for row in rows:
            batch.append(row)
            count += 1
            if len(batch) >= batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=SECTOR_SCHEMA))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SECTOR_SCHEMA))
        writer.close()
        os.replace(temporary, output)
        return count
    except Exception:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise


def _schema(tantivy: Any) -> Any:
    builder = tantivy.SchemaBuilder()
    for field in ("document_type", "siret", "siren", "insee", "postcode"):
        builder.add_text_field(field, stored=True, tokenizer_name="raw", index_option="basic")
    for field in (
        "names", "addresses", "name_ngrams", "address_ngrams", "legal_names",
        "trade_names", "site_names", "historical_names", "supporting_names",
        "historical_addresses", "supporting_addresses", "current_name_ngrams",
    ):
        builder.add_text_field(field, stored=field in {"names", "addresses"})
    for field in (
        "names_exact", "addresses_exact", "legal_names_exact", "trade_names_exact",
        "site_names_exact", "number", "state", "is_siege", "linked_sirets",
        "linked_sirens", "official_evidence_ids", "official_evidence_sources", "payload",
    ):
        builder.add_text_field(field, stored=True, tokenizer_name="raw", index_option="basic")
    return builder.build()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sector_index(
    *, evidence_path: Path, siret_documents_path: Path, siren_documents_path: Path,
    output_root: Path,
) -> Path:
    try:
        import tantivy  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sector index requires tantivy") from exc
    inputs = {
        "evidence_sha256": sha256_file(evidence_path),
        "siret_documents_sha256": sha256_file(siret_documents_path),
        "siren_documents_sha256": sha256_file(siren_documents_path),
        "schema_version": SECTOR_INDEX_SCHEMA_VERSION,
    }
    build_id = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target = output_root / build_id[:16]
    if (target / "manifest.json").is_file():
        return target
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id[:16]}.", dir=output_root))
    (temporary / "tantivy").mkdir()
    connection = duckdb.connect()
    try:
        e = str(evidence_path.resolve()).replace("'", "''")
        sd = str(siret_documents_path.resolve()).replace("'", "''")
        ud = str(siren_documents_path.resolve()).replace("'", "''")
        query = f"""
            WITH evidence AS (
              SELECT subject_kind, siret, siren, source,
                     string_agg(DISTINCT normalized_name, ' | ') AS names,
                     string_agg(DISTINCT normalized_address, ' | ') AS addresses,
                     string_agg(DISTINCT evidence_id, ' ') AS evidence_ids,
                     any_value(insee) FILTER (WHERE insee <> '') AS evidence_insee,
                     any_value(postcode) FILTER (WHERE postcode <> '') AS evidence_postcode,
                     any_value(number) FILTER (WHERE number <> '') AS evidence_number,
                     bool_or(active) AS active
              FROM read_parquet('{e}') GROUP BY 1,2,3,4
            ),
            siret_rows AS (
              SELECT 'siret' document_type, x.siret, x.siren,
                     coalesce(nullif(x.evidence_insee,''), d.insee) insee,
                     coalesce(nullif(x.evidence_postcode,''), d.postcode) postcode,
                     coalesce(nullif(x.evidence_number,''), d.number) number,
                     x.names, x.addresses, x.evidence_ids, x.source,
                     coalesce(d.administrative_state, CASE WHEN x.active THEN 'A' ELSE 'F' END) state,
                     coalesce(d.is_headquarters,false) is_siege
              FROM evidence x LEFT JOIN read_parquet('{sd}') d ON x.siret=d.document_id
              WHERE x.subject_kind='SIRET'
            ),
            siren_rows AS (
              SELECT 'siren' document_type, '' siret, x.siren,
                     coalesce(nullif(x.evidence_insee,''), d.insee) insee,
                     coalesce(nullif(x.evidence_postcode,''), d.postcode) postcode,
                     coalesce(nullif(x.evidence_number,''), d.number) number,
                     x.names, x.addresses, x.evidence_ids, x.source,
                     coalesce(d.administrative_state, CASE WHEN x.active THEN 'A' ELSE 'F' END) state,
                     coalesce(d.is_headquarters,false) is_siege
              FROM evidence x JOIN read_parquet('{ud}') d ON x.siren=d.siren
              WHERE x.subject_kind='SIREN'
            )
            SELECT * FROM siret_rows WHERE insee<>'' OR postcode<>''
            UNION ALL SELECT * FROM siren_rows WHERE insee<>'' OR postcode<>''
        """
        index = tantivy.Index(_schema(tantivy), path=str(temporary / "tantivy"), reuse=False)
        writer = index.writer(600_000_000)
        cursor = connection.execute(query)
        count = 0
        while True:
            rows = cursor.fetchmany(8192)
            if not rows:
                break
            for kind, siret, siren, insee, postcode, number, names_raw, addresses_raw, ids, source, state, is_siege in rows:
                names = [normalize_text(part) for part in str(names_raw or "").split("|") if normalize_text(part)]
                addresses = [normalize_text(part) for part in str(addresses_raw or "").split("|") if normalize_text(part)]
                document = tantivy.Document()
                scalar = {
                    "document_type": kind, "siret": normalize_code(siret,14),
                    "siren": normalize_code(siren,9), "insee": normalize_insee(insee),
                    "postcode": normalize_code(postcode,5), "number": normalize_text(number),
                    "state": str(state or "A"), "is_siege": "1" if is_siege else "0",
                    "linked_sirets": "", "linked_sirens": "", "official_evidence_ids": str(ids or ""),
                    "official_evidence_sources": str(source),
                    "payload": json.dumps({"sector_source":source}, sort_keys=True, separators=(",",":")),
                }
                for key,value in scalar.items(): document.add_text(key,value)
                for name in names:
                    document.add_text("names",name); document.add_text("names_exact",name)
                    document.add_text("name_ngrams"," ".join(character_ngrams(name)))
                for address in addresses:
                    document.add_text("addresses",address); document.add_text("addresses_exact",address)
                    document.add_text("address_ngrams"," ".join(character_ngrams(address)))
                writer.add_document(document); count += 1
        writer.commit(); index.reload()
        manifest = {
            "schema_version": SECTOR_INDEX_SCHEMA_VERSION, "build_id": build_id,
            "num_documents": count, "overlay_kind": "OFFICIAL_SECTOR_EVIDENCE",
            "temporal_complete": True, "missing_temporal_sources": [], "inputs": inputs,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
        os.replace(temporary,target)
        return target
    except Exception:
        import shutil
        shutil.rmtree(temporary,ignore_errors=True); raise
    finally:
        connection.close()


class OfficialSectorTantivyBackend(TantivyBackend):
    pass
