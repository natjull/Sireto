"""Streaming builder for the canonical official-evidence layer.

Only an explicit allow-list is read from SIRENE, RNE and BODACC snapshots.
Raw records are never copied to the canonical outputs.  Invalid/conflicting
records are represented by a source id plus SHA-256 fingerprint in quarantine,
which prevents announcement text or sensitive registry sections from leaking.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import ijson

from .official_evidence import (
    OFFICIAL_EVIDENCE_SCHEMA_VERSION,
    OFFICIAL_QUARANTINE_SCHEMA_VERSION,
    OFFICIAL_RELATION_SCHEMA_VERSION,
    OfficialAddress,
    OfficialEvidence,
    OfficialName,
    OfficialNameKind,
    OfficialRelation,
    OfficialRelationType,
    OfficialSource,
    OfficialSubjectKind,
    QuarantinedOfficialRecord,
    QuarantineReason,
    official_evidence_arrow_schema,
    official_quarantine_arrow_schema,
    official_relation_arrow_schema,
    normalize_siren,
    normalize_siret,
    record_fingerprint,
)


BUILDER_SCHEMA_VERSION = "sireto-official-evidence-builder-v1"


class SnapshotRole(str, Enum):
    SIRENE_ESTABLISHMENTS = "SIRENE_ESTABLISHMENTS"
    SIRENE_ESTABLISHMENT_HISTORY = "SIRENE_ESTABLISHMENT_HISTORY"
    SIRENE_LEGAL_UNITS = "SIRENE_LEGAL_UNITS"
    SIRENE_LEGAL_UNIT_HISTORY = "SIRENE_LEGAL_UNIT_HISTORY"
    SIRENE_SUCCESSIONS = "SIRENE_SUCCESSIONS"
    RNE_RECORDS = "RNE_RECORDS"
    BODACC_ANNOUNCEMENTS = "BODACC_ANNOUNCEMENTS"


@dataclass(frozen=True)
class SnapshotSpec:
    path: Path
    source: OfficialSource
    role: SnapshotRole
    observed_at: str | None = None
    batch_size: int = 16_384

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "source", OfficialSource(self.source))
        object.__setattr__(self, "role", SnapshotRole(self.role))
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class CanonicalizedRecord:
    evidence: tuple[OfficialEvidence, ...] = ()
    relations: tuple[OfficialRelation, ...] = ()
    quarantine: tuple[QuarantinedOfficialRecord, ...] = ()


@dataclass(frozen=True)
class OfficialEvidenceBuildResult:
    output_dir: Path
    evidence_path: Path
    relation_path: Path
    quarantine_path: Path
    manifest_path: Path
    input_records: int
    accepted_evidence: int
    accepted_relations: int
    quarantined_records: int


class _ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int = 4096) -> None:
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self.rows: list[Mapping[str, Any]] = []
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.count = 0

    def add(self, row: Mapping[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.writer.write_table(table)
        self.count += len(self.rows)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def snapshot_specs_from_sync_manifest(
    manifest_path: Path | str,
    *,
    role: SnapshotRole | str | None = None,
    batch_size: int = 16_384,
    payload_names: set[str] | None = None,
) -> tuple[SnapshotSpec, ...]:
    """Resolve the immutable payloads emitted by ``official_source_sync``.

    The sync boundary owns transport/provenance.  This builder only consumes
    the manifest and never imports or mutates the sync implementation.
    """
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_name = str(raw.get("source") or "").upper()
    if source_name in {"RNE", "RNE-FTP-BULK"}:
        source = OfficialSource.RNE
        default_role = SnapshotRole.RNE_RECORDS
    elif source_name == "BODACC":
        source = OfficialSource.BODACC
        default_role = SnapshotRole.BODACC_ANNOUNCEMENTS
    else:
        raise ValueError(f"unsupported official-source manifest: {source_name!r}")
    selected_role = SnapshotRole(role) if role else default_role
    provenance = raw.get("provenance") or {}
    observed_at = (
        provenance.get("next_watermark")
        or provenance.get("snapshot_date")
        or raw.get("created_at")
    )
    specs: list[SnapshotSpec] = []
    found_names: set[str] = set()
    payload_items = raw.get("payload") or (
        raw.get("remote") if source_name == "RNE-FTP-BULK" else []
    )
    for item in payload_items:
        item_name = str(item.get("name") or "")
        if payload_names is not None and item_name not in payload_names:
            continue
        found_names.add(item_name)
        path = manifest_path.parent / item_name
        if not path.is_file():
            raise FileNotFoundError(f"manifest payload is missing: {path}")
        expected_size = item.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise ValueError(f"manifest payload size mismatch: {path}")
        expected_sha = str(item.get("sha256") or "")
        if expected_sha and _sha256_file(path) != expected_sha:
            raise ValueError(f"manifest payload SHA-256 mismatch: {path}")
        specs.append(
            SnapshotSpec(
                path=path,
                source=source,
                role=selected_role,
                observed_at=str(observed_at or "") or None,
                batch_size=batch_size,
            )
        )
    if payload_names is not None and found_names != payload_names:
        missing = sorted(payload_names - found_names)
        raise ValueError(f"official-source manifest payload selection missing: {missing}")
    if not specs:
        raise ValueError("official-source manifest has no payload")
    return tuple(specs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8-sig", newline="")
        if path.suffix.lower() == ".gz"
        else path.open("r", encoding="utf-8-sig", newline="")
    )


def _data_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    return suffixes[-2] if suffixes and suffixes[-1] == ".gz" else path.suffix.lower()


def stream_snapshot_rows(spec: SnapshotSpec) -> Iterator[Mapping[str, Any]]:
    """Yield source rows in bounded batches; no raw record is retained."""
    suffix = _data_suffix(spec.path)
    if suffix in {".parquet", ".pq"}:
        parquet = pq.ParquetFile(spec.path)
        for batch in parquet.iter_batches(batch_size=spec.batch_size):
            yield from batch.to_pylist()
        return
    if suffix in {".csv", ".tsv"}:
        with _open_text(spec.path) as stream:
            delimiter = "\t" if suffix == ".tsv" else ","
            yield from csv.DictReader(stream, delimiter=delimiter)
        return
    if suffix in {".jsonl", ".ndjson"}:
        with _open_text(spec.path) as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"{spec.path}:{line_number} is not a JSON object"
                    )
                yield value
        return
    if suffix == ".json":
        # JSON snapshots should normally be NDJSON.  This compatibility path is
        # intended for bounded receipts/fixtures; national JSON arrays should
        # be converted by the transport layer before canonicalization.
        with _open_text(spec.path) as stream:
            value = json.load(stream)
        if isinstance(value, Mapping):
            for key in ("records", "results", "items", "annonces"):
                if isinstance(value.get(key), list):
                    yield from value[key]
                    return
            yield value
            return
        if isinstance(value, list):
            yield from value
            return
    if suffix == ".zip":
        with zipfile.ZipFile(spec.path) as archive:
            for info in archive.infolist():
                if info.is_dir() or info.filename.startswith("__MACOSX/"):
                    continue
                member_suffix = Path(info.filename).suffix.lower()
                if member_suffix not in {".json", ".jsonl", ".ndjson"}:
                    continue
                if info.flag_bits & 0x1:
                    raise ValueError(f"encrypted ZIP member is unsupported: {info.filename}")
                with archive.open(info) as binary:
                    buffered = io.BufferedReader(binary)
                    if member_suffix in {".jsonl", ".ndjson"}:
                        stream = io.TextIOWrapper(
                            buffered, encoding="utf-8-sig", newline=""
                        )
                        for line_number, line in enumerate(stream, start=1):
                            if not line.strip():
                                continue
                            value = json.loads(line)
                            if not isinstance(value, Mapping):
                                raise ValueError(
                                    f"{spec.path}!{info.filename}:{line_number} is not a JSON object"
                                )
                            yield value
                        continue
                    prefix = buffered.peek(4096).lstrip(b"\xef\xbb\xbf \t\r\n")
                    if prefix.startswith(b"["):
                        for ordinal, value in enumerate(
                            ijson.items(buffered, "item"), start=1
                        ):
                            if not isinstance(value, Mapping):
                                raise ValueError(
                                    f"{spec.path}!{info.filename}:{ordinal} is not a JSON object"
                                )
                            yield value
                        continue
                    stream = io.TextIOWrapper(
                        buffered, encoding="utf-8-sig", newline=""
                    )
                    value = json.load(stream)
                    if isinstance(value, Mapping):
                        for key in ("records", "results", "items", "formalites"):
                            if isinstance(value.get(key), list):
                                yield from value[key]
                                break
                        else:
                            yield value
                    elif isinstance(value, list):
                        yield from value
                    else:
                        raise ValueError(
                            f"{spec.path}!{info.filename} is not a JSON object or array"
                        )
        return
    raise ValueError(f"unsupported snapshot format: {spec.path}")


_MISSING = object()


def _path_get(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _first(record: Mapping[str, Any], paths: Sequence[str], default: Any = "") -> Any:
    for path in paths:
        value = _path_get(record, path)
        if value is not _MISSING and value not in (None, "", []):
            return value
    return default


def _values(record: Mapping[str, Any], paths: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for path in paths:
        value = _path_get(record, path)
        if value is _MISSING or value in (None, ""):
            continue
        if isinstance(value, list):
            values.extend(str(item) for item in value if not isinstance(item, Mapping))
        elif not isinstance(value, Mapping):
            values.append(str(value))
    return tuple(dict.fromkeys(item for item in values if item.strip()))


def _source_record_id(record: Mapping[str, Any], fallback: str) -> str:
    value = _first(
        record,
        [
            "id",
            "record_id",
            "idFlux",
            "numeroAnnonce",
            "numero_annonce",
            "formalites.id",
            "formality.id",
            "company.id",
            "company.formality.id",
            "metadata.id",
        ],
    )
    return str(value or fallback)


def _bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().upper()
    if normalized in {"1", "TRUE", "T", "YES", "Y", "OUI", "A"}:
        return True
    if normalized in {"0", "FALSE", "F", "NO", "N", "NON"}:
        return False
    return None


def _names(
    record: Mapping[str, Any],
    fields: Sequence[tuple[str, OfficialNameKind]],
) -> tuple[OfficialName, ...]:
    output: list[OfficialName] = []
    for path, kind in fields:
        for value in _values(record, [path]):
            try:
                output.append(OfficialName(value, kind))
            except ValueError:
                pass
    return tuple(output)


def _address(
    record: Mapping[str, Any],
    *,
    number_paths: Sequence[str],
    suffix_paths: Sequence[str],
    street_type_paths: Sequence[str],
    street_paths: Sequence[str],
    complement_paths: Sequence[str],
    postcode_paths: Sequence[str],
    insee_paths: Sequence[str],
    full_paths: Sequence[str] = (),
) -> tuple[OfficialAddress, ...]:
    number = str(_first(record, number_paths) or "")
    suffix = str(_first(record, suffix_paths) or "")
    postcode = str(_first(record, postcode_paths) or "")
    insee = str(_first(record, insee_paths) or "")
    full = str(_first(record, full_paths) or "")
    if not full:
        full = " ".join(
            str(value)
            for value in (
                number,
                suffix,
                _first(record, street_type_paths),
                _first(record, street_paths),
                _first(record, complement_paths),
            )
            if value not in (None, "")
        )
    if not any((full, postcode, insee)):
        return ()
    try:
        return (
            OfficialAddress(
                raw_value=full,
                postcode=postcode,
                insee=insee,
                number=number,
                number_suffix=suffix,
            ),
        )
    except ValueError:
        return ()


_SIRENE_NAME_FIELDS = (
    ("enseigne1Etablissement", OfficialNameKind.SIGN),
    ("enseigne2Etablissement", OfficialNameKind.SIGN),
    ("enseigne3Etablissement", OfficialNameKind.SIGN),
    ("denominationUsuelleEtablissement", OfficialNameKind.USUAL),
)
_SIRENE_LEGAL_NAME_FIELDS = (
    ("denominationUniteLegale", OfficialNameKind.LEGAL),
    ("denominationUsuelle1UniteLegale", OfficialNameKind.USUAL),
    ("denominationUsuelle2UniteLegale", OfficialNameKind.USUAL),
    ("denominationUsuelle3UniteLegale", OfficialNameKind.USUAL),
    ("sigleUniteLegale", OfficialNameKind.TRADE),
)


def canonicalize_snapshot_record(
    spec: SnapshotSpec,
    record: Mapping[str, Any],
    *,
    ordinal: int,
) -> CanonicalizedRecord:
    fallback_id = f"{spec.path.name}:{ordinal}"
    record_id = _source_record_id(record, fallback_id)
    fingerprint = record_fingerprint(record)
    try:
        if spec.role in {
            SnapshotRole.SIRENE_ESTABLISHMENTS,
            SnapshotRole.SIRENE_ESTABLISHMENT_HISTORY,
        }:
            return _canonicalize_sirene_establishment(
                spec, record, record_id, fingerprint
            )
        if spec.role in {
            SnapshotRole.SIRENE_LEGAL_UNITS,
            SnapshotRole.SIRENE_LEGAL_UNIT_HISTORY,
        }:
            return _canonicalize_sirene_legal_unit(
                spec, record, record_id, fingerprint
            )
        if spec.role is SnapshotRole.SIRENE_SUCCESSIONS:
            return _canonicalize_sirene_succession(
                spec, record, record_id, fingerprint
            )
        if spec.role is SnapshotRole.RNE_RECORDS:
            return _canonicalize_rne(spec, record, record_id, fingerprint)
        if spec.role is SnapshotRole.BODACC_ANNOUNCEMENTS:
            return _canonicalize_bodacc(spec, record, record_id, fingerprint)
    except (TypeError, ValueError, KeyError) as error:
        reason = (
            QuarantineReason.IDENTIFIER_MISMATCH
            if "mismatch" in str(error).lower()
            else QuarantineReason.MALFORMED_RECORD
        )
        return CanonicalizedRecord(
            quarantine=(
                _quarantine(spec, record_id, reason, type(error).__name__, fingerprint),
            )
        )
    return CanonicalizedRecord(
        quarantine=(
            _quarantine(
                spec,
                record_id,
                QuarantineReason.UNSUPPORTED_SNAPSHOT,
                spec.role.value,
                fingerprint,
            ),
        )
    )


def _quarantine(
    spec: SnapshotSpec,
    record_id: str,
    reason: QuarantineReason,
    detail: str,
    fingerprint: str,
) -> QuarantinedOfficialRecord:
    return QuarantinedOfficialRecord(
        source=spec.source,
        snapshot_role=spec.role.value,
        source_record_id=record_id,
        reason=reason,
        detail=detail,
        record_fingerprint=fingerprint,
    )


def _missing_identifier(
    spec: SnapshotSpec, record_id: str, fingerprint: str
) -> CanonicalizedRecord:
    return CanonicalizedRecord(
        quarantine=(
            _quarantine(
                spec,
                record_id,
                QuarantineReason.MISSING_SUBJECT_IDENTIFIER,
                "MISSING STRUCTURED SIREN OR SIRET",
                fingerprint,
            ),
        )
    )


def _canonicalize_sirene_establishment(
    spec: SnapshotSpec,
    record: Mapping[str, Any],
    record_id: str,
    fingerprint: str,
) -> CanonicalizedRecord:
    siret = normalize_siret(_first(record, ["siret", "siretEtablissement"]))
    siren = normalize_siren(_first(record, ["siren", "sirenUniteLegale"]) or siret[:9])
    if not siret or not siren:
        return _missing_identifier(spec, record_id, fingerprint)
    if not siret.startswith(siren):
        return CanonicalizedRecord(
            quarantine=(
                _quarantine(
                    spec,
                    record_id,
                    QuarantineReason.IDENTIFIER_MISMATCH,
                    "SIRET SIREN PREFIX",
                    fingerprint,
                ),
            )
        )
    history = spec.role is SnapshotRole.SIRENE_ESTABLISHMENT_HISTORY
    names = list(_names(record, _SIRENE_NAME_FIELDS))
    if history:
        names = [
            OfficialName(
                item.raw_value,
                OfficialNameKind.HISTORICAL,
                item.normalized_value,
            )
            for item in names
        ]
    addresses = _address(
        record,
        number_paths=["numeroVoieEtablissement"],
        suffix_paths=["indiceRepetitionEtablissement"],
        street_type_paths=["typeVoieEtablissement"],
        street_paths=["libelleVoieEtablissement"],
        complement_paths=["complementAdresseEtablissement"],
        postcode_paths=["codePostalEtablissement"],
        insee_paths=["codeCommuneEtablissement"],
    )
    evidence = OfficialEvidence(
        source=(OfficialSource.SIRENE_HISTORY if history else OfficialSource.SIRENE_CURRENT),
        source_record_id=record_id,
        subject_kind=OfficialSubjectKind.SIRET,
        siren=siren,
        siret=siret,
        names=tuple(names),
        addresses=addresses,
        administrative_state=str(
            _first(record, ["etatAdministratifEtablissement"]) or ""
        ),
        is_headquarters=_bool(_first(record, ["etablissementSiege"], None)),
        valid_from=_first(
            record,
            ["dateDebut", "dateDebutHistorisationEtablissement", "dateCreationEtablissement"],
            None,
        ),
        valid_to=_first(record, ["dateFin", "dateFinHistorisationEtablissement"], None),
        observed_at=spec.observed_at,
        is_current=not history,
    )
    return CanonicalizedRecord(evidence=(evidence,))


def _canonicalize_sirene_legal_unit(
    spec: SnapshotSpec,
    record: Mapping[str, Any],
    record_id: str,
    fingerprint: str,
) -> CanonicalizedRecord:
    siren = normalize_siren(_first(record, ["siren", "sirenUniteLegale"]))
    if not siren:
        return _missing_identifier(spec, record_id, fingerprint)
    history = spec.role is SnapshotRole.SIRENE_LEGAL_UNIT_HISTORY
    names = list(_names(record, _SIRENE_LEGAL_NAME_FIELDS))
    # Natural-person names are intentionally absent from the allow-list.
    if history:
        names = [
            OfficialName(
                item.raw_value,
                OfficialNameKind.HISTORICAL,
                item.normalized_value,
            )
            for item in names
        ]
    try:
        evidence = OfficialEvidence(
            source=(OfficialSource.SIRENE_HISTORY if history else OfficialSource.SIRENE_CURRENT),
            source_record_id=record_id,
            subject_kind=OfficialSubjectKind.SIREN,
            siren=siren,
            names=tuple(names),
            administrative_state=str(
                _first(record, ["etatAdministratifUniteLegale"]) or ""
            ),
            valid_from=_first(record, ["dateDebut", "dateCreationUniteLegale"], None),
            valid_to=_first(record, ["dateFin"], None),
            observed_at=spec.observed_at,
            is_current=not history,
        )
    except ValueError:
        return CanonicalizedRecord(
            quarantine=(
                _quarantine(
                    spec,
                    record_id,
                    QuarantineReason.EMPTY_ALLOWLISTED_EVIDENCE,
                    "NO ALLOWLISTED LEGAL UNIT FIELD",
                    fingerprint,
                ),
            )
        )
    return CanonicalizedRecord(evidence=(evidence,))


def _canonicalize_sirene_succession(
    spec: SnapshotSpec,
    record: Mapping[str, Any],
    record_id: str,
    fingerprint: str,
) -> CanonicalizedRecord:
    predecessor = normalize_siret(
        _first(
            record,
            ["siretEtablissementPredecesseur", "siret_predecesseur", "predecessor_siret"],
        )
    )
    successor = normalize_siret(
        _first(
            record,
            ["siretEtablissementSuccesseur", "siret_successeur", "successor_siret"],
        )
    )
    if not predecessor or not successor:
        return _missing_identifier(spec, record_id, fingerprint)
    if predecessor == successor:
        return CanonicalizedRecord(
            quarantine=(
                _quarantine(
                    spec,
                    record_id,
                    QuarantineReason.RELATION_SELF_LOOP,
                    "SUCCESSION SELF LOOP",
                    fingerprint,
                ),
            )
        )
    relation = OfficialRelation(
        source=OfficialSource.SIRENE_SUCCESSION,
        source_record_id=record_id,
        relation_type=OfficialRelationType.ESTABLISHMENT_SUCCESSION,
        from_kind=OfficialSubjectKind.SIRET,
        from_identifier=predecessor,
        to_kind=OfficialSubjectKind.SIRET,
        to_identifier=successor,
        effective_date=_first(record, ["dateLienSuccession", "dateEffet"], None),
        observed_at=spec.observed_at,
    )
    return CanonicalizedRecord(relations=(relation,))


_RNE_SIREN = [
    "siren",
    "formality.siren",
    "content.personneMorale.identite.entreprise.siren",
    "identite.entreprise.siren",
    "entreprise.siren",
    "company.siren",
    "formality.content.personneMorale.identite.entreprise.siren",
]
_RNE_SIRET = [
    "siret",
    "etablissement.siret",
    "etablissementPrincipal.siret",
    "etablissementPrincipal.descriptionEtablissement.siret",
    "formality.content.personneMorale.etablissementPrincipal.descriptionEtablissement.siret",
    "formality.content.personnePhysique.etablissementPrincipal.descriptionEtablissement.siret",
    "formality.content.exploitation.etablissementPrincipal.descriptionEtablissement.siret",
]
_RNE_NAMES = (
    ("denomination", OfficialNameKind.LEGAL),
    ("content.personneMorale.identite.entreprise.denomination", OfficialNameKind.LEGAL),
    (
        "formality.content.personneMorale.identite.entreprise.denomination",
        OfficialNameKind.LEGAL,
    ),
    ("content.personneMorale.identite.entreprise.nomCommercial", OfficialNameKind.TRADE),
    (
        "formality.content.personneMorale.identite.entreprise.nomCommercial",
        OfficialNameKind.TRADE,
    ),
    ("content.personneMorale.identite.description.sigle", OfficialNameKind.TRADE),
    (
        "formality.content.personneMorale.identite.description.sigle",
        OfficialNameKind.TRADE,
    ),
    (
        "content.personnePhysique.identite.entrepreneur.descriptionPersonne.nom",
        OfficialNameKind.LEGAL,
    ),
    (
        "formality.content.personnePhysique.identite.entrepreneur.descriptionPersonne.nom",
        OfficialNameKind.LEGAL,
    ),
    (
        "content.personnePhysique.identite.entrepreneur.descriptionPersonne.nomUsage",
        OfficialNameKind.USUAL,
    ),
    (
        "formality.content.personnePhysique.identite.entrepreneur.descriptionPersonne.nomUsage",
        OfficialNameKind.USUAL,
    ),
    ("identite.entreprise.denomination", OfficialNameKind.LEGAL),
    ("entreprise.denomination", OfficialNameKind.LEGAL),
    ("nomCommercial", OfficialNameKind.TRADE),
    ("etablissement.nomCommercial", OfficialNameKind.TRADE),
    ("enseigne", OfficialNameKind.SIGN),
    ("etablissement.enseigne", OfficialNameKind.SIGN),
    ("content.exploitation.nomCommercial", OfficialNameKind.TRADE),
    ("formality.content.exploitation.nomCommercial", OfficialNameKind.TRADE),
    ("content.exploitation.nomExploitation", OfficialNameKind.USUAL),
    ("formality.content.exploitation.nomExploitation", OfficialNameKind.USUAL),
    ("content.exploitation.enseigne", OfficialNameKind.SIGN),
    ("formality.content.exploitation.enseigne", OfficialNameKind.SIGN),
)

_RNE_PERSON_BRANCHES = (
    "formality.content.personneMorale",
    "formality.content.personnePhysique",
    "formality.content.exploitation",
    "content.personneMorale",
    "content.personnePhysique",
    "content.exploitation",
)

_RNE_SITE_NAMES = (
    ("descriptionEtablissement.enseigne", OfficialNameKind.SIGN),
    ("descriptionEtablissement.nomCommercial", OfficialNameKind.TRADE),
    ("descriptionEtablissement.nomExploitation", OfficialNameKind.USUAL),
    ("enseigne", OfficialNameKind.SIGN),
    ("nomCommercial", OfficialNameKind.TRADE),
)


def _rne_site_entries(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    for prefix in _RNE_PERSON_BRANCHES:
        person = _first(record, [prefix], None)
        if not isinstance(person, Mapping):
            continue
        for key in ("etablissementPrincipal", "etablissementModifie"):
            value = person.get(key)
            if isinstance(value, Mapping):
                entries.append(value)
        others = person.get("autresEtablissements")
        if isinstance(others, list):
            entries.extend(item for item in others if isinstance(item, Mapping))
    return entries


def _canonicalize_rne_site(
    *,
    spec: SnapshotSpec,
    entry: Mapping[str, Any],
    siren: str,
    record_id: str,
) -> OfficialEvidence | None:
    siret = normalize_siret(
        _first(entry, ["descriptionEtablissement.siret", "siret"], "")
    )
    if not siret or not siret.startswith(siren):
        return None
    addresses = _address(
        entry,
        number_paths=["adresse.numVoie", "adresse.numeroVoie"],
        suffix_paths=["adresse.indiceRepetition"],
        street_type_paths=["adresse.typeVoie"],
        street_paths=["adresse.voie", "adresse.libelleVoie"],
        complement_paths=["adresse.complementLocalisation", "adresse.complement"],
        postcode_paths=["adresse.codePostal"],
        insee_paths=["adresse.codeInseeCommune", "adresse.codeCommune"],
        full_paths=["adresse.adresseComplete"],
    )
    description = _first(entry, ["descriptionEtablissement"], {})
    description = description if isinstance(description, Mapping) else {}
    closed_at = _first(
        entry,
        [
            "descriptionEtablissement.dateEffetFermeture",
            "dateEffetFermeture",
            "dateFermeture",
        ],
        None,
    )
    try:
        return OfficialEvidence(
            source=OfficialSource.RNE,
            source_record_id=f"{record_id}:site:{siret}",
            subject_kind=OfficialSubjectKind.SIRET,
            siren=siren,
            siret=siret,
            names=_names(entry, _RNE_SITE_NAMES),
            addresses=addresses,
            administrative_state=("F" if closed_at else "A"),
            is_headquarters=_bool(description.get("indicateurEtablissementPrincipal")),
            valid_from=_first(entry, ["dateDebut", "dateEffet", "dateCreation"], None),
            valid_to=closed_at,
            observed_at=spec.observed_at,
            is_current=not bool(closed_at),
        )
    except ValueError:
        return None


def _canonicalize_rne(
    spec: SnapshotSpec,
    record: Mapping[str, Any],
    record_id: str,
    fingerprint: str,
) -> CanonicalizedRecord:
    company = record.get("company")
    if isinstance(company, Mapping):
        record = company
    commercial_reuse = _first(
        record,
        [
            "diffusionCommerciale",
            "content.diffusionCommerciale",
            "formality.diffusionCommerciale",
        ],
        None,
    )
    insee_reuse = str(
        _first(
            record,
            ["diffusionINSEE", "content.diffusionINSEE", "formality.diffusionINSEE"],
            "",
        )
        or ""
    ).upper()
    if _bool(commercial_reuse) is False or insee_reuse == "N":
        return CanonicalizedRecord(
            quarantine=(
                _quarantine(
                    spec,
                    record_id,
                    QuarantineReason.OFFICIAL_REUSE_OPPOSITION,
                    "RNE REUSE OR DIFFUSION OPPOSITION",
                    fingerprint,
                ),
            )
        )
    site_entries = _rne_site_entries(record)
    siret = normalize_siret(_first(record, _RNE_SIRET))
    siren = normalize_siren(_first(record, _RNE_SIREN) or siret[:9])
    if not siren:
        return _missing_identifier(spec, record_id, fingerprint)
    if siret and not siret.startswith(siren):
        return CanonicalizedRecord(
            quarantine=(
                _quarantine(
                    spec,
                    record_id,
                    QuarantineReason.IDENTIFIER_MISMATCH,
                    "SIRET SIREN PREFIX",
                    fingerprint,
                ),
            )
        )
    addresses = _address(
        record,
        number_paths=[
            "formality.content.personneMorale.adresseEntreprise.adresse.numVoie",
            "formality.content.personnePhysique.adresseEntreprise.adresse.numVoie",
            "formality.content.exploitation.adresseEntreprise.adresse.numVoie",
            "content.personneMorale.adresseEntreprise.adresse.numVoie",
            "content.personnePhysique.adresseEntreprise.adresse.numVoie",
            "content.exploitation.adresseEntreprise.adresse.numVoie",
            "adresse.numeroVoie",
            "adresse.numVoie",
            "etablissement.adresse.numeroVoie",
        ],
        suffix_paths=[
            "formality.content.personneMorale.adresseEntreprise.adresse.indiceRepetition",
            "formality.content.personnePhysique.adresseEntreprise.adresse.indiceRepetition",
            "formality.content.exploitation.adresseEntreprise.adresse.indiceRepetition",
            "content.personneMorale.adresseEntreprise.adresse.indiceRepetition",
            "content.personnePhysique.adresseEntreprise.adresse.indiceRepetition",
            "content.exploitation.adresseEntreprise.adresse.indiceRepetition",
            "adresse.indiceRepetition",
            "etablissement.adresse.indiceRepetition",
        ],
        street_type_paths=[
            "formality.content.personneMorale.adresseEntreprise.adresse.typeVoie",
            "formality.content.personnePhysique.adresseEntreprise.adresse.typeVoie",
            "formality.content.exploitation.adresseEntreprise.adresse.typeVoie",
            "content.personneMorale.adresseEntreprise.adresse.typeVoie",
            "content.personnePhysique.adresseEntreprise.adresse.typeVoie",
            "content.exploitation.adresseEntreprise.adresse.typeVoie",
            "adresse.typeVoie",
            "etablissement.adresse.typeVoie",
        ],
        street_paths=[
            "formality.content.personneMorale.adresseEntreprise.adresse.voie",
            "formality.content.personnePhysique.adresseEntreprise.adresse.voie",
            "formality.content.exploitation.adresseEntreprise.adresse.voie",
            "content.personneMorale.adresseEntreprise.adresse.voie",
            "content.personnePhysique.adresseEntreprise.adresse.voie",
            "content.exploitation.adresseEntreprise.adresse.voie",
            "adresse.voie",
            "adresse.libelleVoie",
            "etablissement.adresse.libelleVoie",
        ],
        complement_paths=[
            "formality.content.personneMorale.adresseEntreprise.adresse.complementLocalisation",
            "formality.content.personnePhysique.adresseEntreprise.adresse.complementLocalisation",
            "formality.content.exploitation.adresseEntreprise.adresse.complementLocalisation",
            "content.personneMorale.adresseEntreprise.adresse.complementLocalisation",
            "content.personnePhysique.adresseEntreprise.adresse.complementLocalisation",
            "content.exploitation.adresseEntreprise.adresse.complementLocalisation",
            "adresse.complement",
            "etablissement.adresse.complement",
        ],
        postcode_paths=[
            "formality.content.personneMorale.adresseEntreprise.adresse.codePostal",
            "formality.content.personnePhysique.adresseEntreprise.adresse.codePostal",
            "formality.content.exploitation.adresseEntreprise.adresse.codePostal",
            "content.personneMorale.adresseEntreprise.adresse.codePostal",
            "content.personnePhysique.adresseEntreprise.adresse.codePostal",
            "content.exploitation.adresseEntreprise.adresse.codePostal",
            "codePostal",
            "adresse.codePostal",
            "etablissement.adresse.codePostal",
        ],
        insee_paths=[
            "formality.content.personneMorale.adresseEntreprise.adresse.codeInseeCommune",
            "formality.content.personnePhysique.adresseEntreprise.adresse.codeInseeCommune",
            "formality.content.exploitation.adresseEntreprise.adresse.codeInseeCommune",
            "content.personneMorale.adresseEntreprise.adresse.codeInseeCommune",
            "content.personnePhysique.adresseEntreprise.adresse.codeInseeCommune",
            "content.exploitation.adresseEntreprise.adresse.codeInseeCommune",
            "codeCommune",
            "adresse.codeCommune",
            "etablissement.adresse.codeCommune",
        ],
        full_paths=["adresseComplete", "adresse.adresseComplete", "etablissement.adresse.adresseComplete"],
    )
    evidence_values: list[OfficialEvidence] = []
    try:
        evidence_values.append(OfficialEvidence(
            source=OfficialSource.RNE,
            source_record_id=record_id,
            subject_kind=(
                OfficialSubjectKind.SIRET
                if siret and not site_entries
                else OfficialSubjectKind.SIREN
            ),
            siren=siren,
            siret=(siret if siret and not site_entries else ""),
            names=_names(record, _RNE_NAMES),
            addresses=addresses,
            administrative_state=str(
                _first(record, ["etatAdministratif", "statut", "status"]) or ""
            ),
            is_headquarters=_bool(
                _first(record, ["estSiege", "etablissement.estSiege"], None)
            ),
            valid_from=_first(record, ["dateEffet", "dateDebut", "dateCreation"], None),
            valid_to=_first(record, ["dateFin", "dateCessation"], None),
            observed_at=spec.observed_at,
            is_current=not bool(_first(record, ["dateFin", "dateCessation"], None)),
        ))
    except ValueError:
        return CanonicalizedRecord(
            quarantine=(
                _quarantine(
                    spec,
                    record_id,
                    QuarantineReason.EMPTY_ALLOWLISTED_EVIDENCE,
                    "NO ALLOWLISTED RNE FIELD",
                    fingerprint,
                ),
            )
        )
    seen_sites: set[str] = set()
    for entry in site_entries:
        site = _canonicalize_rne_site(
            spec=spec, entry=entry, siren=siren, record_id=record_id
        )
        if site is None or site.siret in seen_sites:
            continue
        seen_sites.add(site.siret)
        evidence_values.append(site)
    return CanonicalizedRecord(evidence=tuple(evidence_values))


_BODACC_SIREN = [
    "siren",
    "personneMorale.siren",
    "personne.siren",
    "entreprise.siren",
    "registre.siren",
]
_BODACC_SIRET = ["siret", "etablissement.siret", "personneMorale.siret"]
_BODACC_NAMES = (
    ("denomination", OfficialNameKind.LEGAL),
    ("personneMorale.denomination", OfficialNameKind.LEGAL),
    ("nomCommercial", OfficialNameKind.TRADE),
    ("enseigne", OfficialNameKind.SIGN),
)


def _canonicalize_bodacc(
    spec: SnapshotSpec,
    record: Mapping[str, Any],
    record_id: str,
    fingerprint: str,
) -> CanonicalizedRecord:
    people = _decode_structured_list(record.get("listepersonnes"))
    establishments = _decode_structured_list(record.get("listeetablissements"))
    predecessors = (
        _decode_structured_list(record.get("listeprecedentproprietaire"))
        + _decode_structured_list(record.get("listeprecedentexploitant"))
    )
    top_siret = normalize_siret(_first(record, _BODACC_SIRET))
    top_siren = _extract_structured_siren(
        _first(record, _BODACC_SIREN + ["registre"]) or top_siret[:9]
    )
    evidence_items: list[OfficialEvidence] = []
    quarantines: list[QuarantinedOfficialRecord] = []
    current_sirens: list[str] = []
    current_sirets: list[str] = [top_siret] if top_siret else []

    # ODS v2.1 serializes these columns either as a JSON string or as nested
    # list/dict values depending on the export protocol.  Only the explicit
    # person identity object is inspected; roles/directors and announcement
    # text are outside this allow-list.
    for index, item in enumerate(people):
        person = _unwrap_structured(
            item,
            ["personne", "personneMorale", "personnePhysique", "personnepm", "personnepp"],
        )
        siren = _extract_structured_siren(
            _first(
                person,
                [
                    "numeroImmatriculation.numeroIdentification",
                    "numeroimmatriculation.numeroidentification",
                    "numeroIdentification",
                    "siren",
                ],
            )
        )
        if not siren:
            continue
        current_sirens.append(siren)
        names = _bodacc_person_names(person)
        addresses = _bodacc_person_addresses(person)
        try:
            evidence_items.append(
                OfficialEvidence(
                    source=OfficialSource.BODACC,
                    source_record_id=f"{record_id}:person:{index}",
                    subject_kind=OfficialSubjectKind.SIREN,
                    siren=siren,
                    names=names,
                    addresses=addresses,
                    valid_from=_first(record, ["dateparution", "dateParution", "dateEffet"], None),
                    observed_at=spec.observed_at,
                    is_current=False,
                )
            )
        except ValueError:
            pass

    if top_siren:
        current_sirens.append(top_siren)
    current_sirens = list(dict.fromkeys(current_sirens))

    # Establishment rows usually provide the place while listepersonnes carries
    # the identifier.  Attach only to a unique structured current SIREN; an
    # ambiguous announcement cannot safely project the address to either one.
    if len(current_sirens) == 1:
        for index, item in enumerate(establishments):
            establishment = _unwrap_structured(
                item, ["etablissement", "etablissementPrincipal"]
            )
            establishment_siret = normalize_siret(
                _first(establishment, ["siret", "numeroSiret"])
            )
            establishment_siren = (
                establishment_siret[:9]
            if establishment_siret
                else current_sirens[0]
            )
            if establishment_siret:
                current_sirets.append(establishment_siret)
            addresses = _bodacc_establishment_addresses(establishment)
            names = _names(
                establishment,
                (
                    ("denomination", OfficialNameKind.USUAL),
                    ("nomCommercial", OfficialNameKind.TRADE),
                    ("enseigne", OfficialNameKind.SIGN),
                ),
            )
            try:
                evidence_items.append(
                    OfficialEvidence(
                        source=OfficialSource.BODACC,
                        source_record_id=f"{record_id}:establishment:{index}",
                        subject_kind=(
                            OfficialSubjectKind.SIRET
                            if establishment_siret
                            else OfficialSubjectKind.SIREN
                        ),
                        siren=establishment_siren,
                        siret=establishment_siret,
                        names=names,
                        addresses=addresses,
                        valid_from=_first(record, ["dateparution", "dateParution", "dateEffet"], None),
                        observed_at=spec.observed_at,
                        is_current=False,
                    )
                )
            except ValueError:
                pass

    # Compatibility with compact curated BODACC snapshots which already expose
    # a top-level structured identifier/name/address.
    if top_siren:
        addresses = _address(
            record,
            number_paths=["adresse.numeroVoie", "etablissement.adresse.numeroVoie"],
            suffix_paths=["adresse.indiceRepetition"],
            street_type_paths=["adresse.typeVoie"],
            street_paths=["adresse.libelleVoie", "adresse.voie"],
            complement_paths=["adresse.complement"],
            postcode_paths=["adresse.codePostal", "codePostal"],
            insee_paths=["adresse.codeCommune", "codeCommune"],
            full_paths=["adresseComplete", "adresse.adresseComplete"],
        )
        try:
            evidence_items.append(
                OfficialEvidence(
                    source=OfficialSource.BODACC,
                    source_record_id=record_id,
                    subject_kind=(OfficialSubjectKind.SIRET if top_siret else OfficialSubjectKind.SIREN),
                    siren=top_siren,
                    siret=top_siret,
                    names=_names(record, _BODACC_NAMES),
                    addresses=addresses,
                    administrative_state="",
                    valid_from=_first(record, ["dateParution", "dateEffet"], None),
                    observed_at=spec.observed_at,
                    is_current=False,
                )
            )
        except ValueError:
            # A structured relation can still be retained without textual
            # evidence; do not quarantine it yet.
            pass

    relations = list(_bodacc_structured_relations(spec, record, record_id))
    if len(current_sirens) == 1:
        current_siren = current_sirens[0]
        for index, item in enumerate(predecessors):
            predecessor = _unwrap_structured(
                item,
                [
                    "precedentProprietairePM",
                    "precedentProprietairePP",
                    "precedentExploitantPM",
                    "precedentExploitantPP",
                    "precedentproprietairepm",
                    "precedentexploitantpm",
                    "personne",
                ],
            )
            predecessor_siren = _extract_structured_siren(
                _first(
                    predecessor,
                    [
                        "numeroImmatriculation.numeroIdentification",
                        "numeroimmatriculation.numeroidentification",
                        "numeroIdentification",
                        "siren",
                    ],
                )
            )
            if predecessor_siren and predecessor_siren != current_siren:
                relations.append(
                    OfficialRelation(
                        source=OfficialSource.BODACC,
                        source_record_id=f"{record_id}:predecessor:{index}",
                        relation_type=OfficialRelationType.ASSET_TRANSFER,
                        from_kind=OfficialSubjectKind.SIREN,
                        from_identifier=predecessor_siren,
                        to_kind=OfficialSubjectKind.SIREN,
                        to_identifier=current_siren,
                        effective_date=_first(record, ["dateparution", "dateParution", "dateEffet"], None),
                        observed_at=spec.observed_at,
                    )
                )
            current_sirets = list(dict.fromkeys(current_sirets))
            predecessor_siret = normalize_siret(
                _first(predecessor, ["siret", "numeroSiret", "etablissement.siret"])
            )
            if (
                len(current_sirets) == 1
                and predecessor_siret
                and predecessor_siret != current_sirets[0]
            ):
                relations.append(
                    OfficialRelation(
                        source=OfficialSource.BODACC,
                        source_record_id=f"{record_id}:predecessor-siret:{index}",
                        relation_type=OfficialRelationType.ESTABLISHMENT_SUCCESSION,
                        from_kind=OfficialSubjectKind.SIRET,
                        from_identifier=predecessor_siret,
                        to_kind=OfficialSubjectKind.SIRET,
                        to_identifier=current_sirets[0],
                        effective_date=_first(record, ["dateparution", "dateParution", "dateEffet"], None),
                        observed_at=spec.observed_at,
                    )
                )
    if not evidence_items and not relations:
        reason = (
            QuarantineReason.UNSTRUCTURED_RELATION
            if any(key in record for key in ("texte", "annonce", "descriptif", "publication"))
            else QuarantineReason.MISSING_SUBJECT_IDENTIFIER
        )
        quarantines.append(
            _quarantine(
                spec,
                record_id,
                reason,
                "NO ALLOWLISTED STRUCTURED IDENTIFIERS",
                fingerprint,
            )
        )
    return CanonicalizedRecord(
        evidence=tuple(evidence_items),
        relations=tuple(relations),
        quarantine=tuple(quarantines),
    )


def _decode_structured_list(value: Any) -> list[Mapping[str, Any]]:
    """Decode ODS JSON columns without inspecting arbitrary free-form text."""
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, Mapping):
        for key in ("items", "liste", "values"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _unwrap_structured(
    value: Mapping[str, Any], wrappers: Sequence[str]
) -> Mapping[str, Any]:
    for wrapper in wrappers:
        nested = value.get(wrapper)
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except json.JSONDecodeError:
                nested = None
        if isinstance(nested, Mapping):
            return nested
    return value


def _extract_structured_siren(value: Any) -> str:
    if isinstance(value, Mapping):
        value = _first(
            value,
            ["numeroIdentification", "numeroidentification", "siren"],
        )
    digits = "".join(character for character in str(value or "") if character.isdigit())
    if len(digits) == 14:
        digits = digits[:9]
    return normalize_siren(digits) if len(digits) == 9 else ""


def _bodacc_person_names(person: Mapping[str, Any]) -> tuple[OfficialName, ...]:
    output = list(
        _names(
            person,
            (
                ("denomination", OfficialNameKind.LEGAL),
                ("nomCommercial", OfficialNameKind.TRADE),
                ("enseigne", OfficialNameKind.SIGN),
            ),
        )
    )
    # Nom/prénom belong to the announced business subject, not a director or
    # beneficial-owner collection.  They are used only when no denomination is
    # supplied by the structured person object.
    if not output:
        name = " ".join(
            str(value)
            for value in (
                _first(person, ["prenom", "prenoms"]),
                _first(person, ["nom", "nomUsage"]),
            )
            if value not in (None, "")
        )
        if name:
            output.append(OfficialName(name, OfficialNameKind.LEGAL))
    return tuple(output)


def _bodacc_address_from_mapping(value: Any) -> tuple[OfficialAddress, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ()
    if not isinstance(value, Mapping):
        return ()
    return _address(
        value,
        number_paths=["numeroVoie", "numero", "numVoie", "numero_voie"],
        suffix_paths=["indiceRepetition", "indice", "suffixe"],
        street_type_paths=["typeVoie", "type_voie"],
        street_paths=["nomVoie", "libelleVoie", "voie", "libelle_voie"],
        complement_paths=["complement", "complementAdresse"],
        postcode_paths=["codePostal", "code_postal", "cp"],
        insee_paths=["codeCommune", "code_commune", "insee"],
        full_paths=["adresseComplete", "adresse", "libelle"],
    )


def _bodacc_person_addresses(person: Mapping[str, Any]) -> tuple[OfficialAddress, ...]:
    output: list[OfficialAddress] = []
    for key in ("adresseSiegeSocial", "adressesiegesocial", "adressePP", "adressepp", "adresse"):
        output.extend(_bodacc_address_from_mapping(person.get(key)))
    return tuple(output)


def _bodacc_establishment_addresses(
    establishment: Mapping[str, Any],
) -> tuple[OfficialAddress, ...]:
    output: list[OfficialAddress] = []
    for key in ("adresse", "adresseEtablissement", "adresseetablissement"):
        output.extend(_bodacc_address_from_mapping(establishment.get(key)))
    if not output:
        output.extend(_bodacc_address_from_mapping(establishment))
    return tuple(output)


def _bodacc_structured_relations(
    spec: SnapshotSpec,
    record: Mapping[str, Any],
    record_id: str,
) -> tuple[OfficialRelation, ...]:
    # Never scan free-form BODACC text for identifiers.  Both endpoints must be
    # present in explicit structured fields.
    pairs = [
        (
            OfficialRelationType.LEGAL_UNIT_SUCCESSION,
            ["predecesseur.siren", "sirenPredecesseur", "predecessor.siren"],
            ["successeur.siren", "sirenSuccesseur", "successor.siren"],
        ),
        (
            OfficialRelationType.ASSET_TRANSFER,
            ["vendeur.siren", "cedant.siren", "seller.siren"],
            ["acquereur.siren", "cessionnaire.siren", "buyer.siren"],
        ),
    ]
    output: list[OfficialRelation] = []
    for relation_type, from_paths, to_paths in pairs:
        from_identifier = normalize_siren(_first(record, from_paths))
        to_identifier = normalize_siren(_first(record, to_paths))
        if not from_identifier or not to_identifier or from_identifier == to_identifier:
            continue
        output.append(
            OfficialRelation(
                source=OfficialSource.BODACC,
                source_record_id=record_id,
                relation_type=relation_type,
                from_kind=OfficialSubjectKind.SIREN,
                from_identifier=from_identifier,
                to_kind=OfficialSubjectKind.SIREN,
                to_identifier=to_identifier,
                effective_date=_first(record, ["dateEffet", "dateParution"], None),
                observed_at=spec.observed_at,
            )
        )
    siret_pairs = [
        (
            ["predecesseur.siret", "siretPredecesseur", "predecessor.siret"],
            ["successeur.siret", "siretSuccesseur", "successor.siret"],
        ),
        (
            ["vendeur.siret", "cedant.siret", "seller.siret"],
            ["acquereur.siret", "cessionnaire.siret", "buyer.siret"],
        ),
    ]
    for from_paths, to_paths in siret_pairs:
        from_identifier = normalize_siret(_first(record, from_paths))
        to_identifier = normalize_siret(_first(record, to_paths))
        if not from_identifier or not to_identifier or from_identifier == to_identifier:
            continue
        output.append(
            OfficialRelation(
                source=OfficialSource.BODACC,
                source_record_id=record_id,
                relation_type=OfficialRelationType.ESTABLISHMENT_SUCCESSION,
                from_kind=OfficialSubjectKind.SIRET,
                from_identifier=from_identifier,
                to_kind=OfficialSubjectKind.SIRET,
                to_identifier=to_identifier,
                effective_date=_first(record, ["dateEffet", "dateParution", "dateparution"], None),
                observed_at=spec.observed_at,
            )
        )
    return tuple(output)


def _address_conflicts(left: OfficialEvidence, right: OfficialEvidence) -> bool:
    for left_address in left.addresses:
        for right_address in right.addresses:
            if left_address.insee and right_address.insee:
                if left_address.insee != right_address.insee:
                    return True
            if left_address.postcode and right_address.postcode:
                if left_address.postcode != right_address.postcode:
                    return True
    return False


def _strip_conflicting_addresses(
    evidence: OfficialEvidence,
) -> OfficialEvidence | None:
    if evidence.names or evidence.administrative_state or evidence.is_headquarters is not None:
        return OfficialEvidence(
            source=evidence.source,
            source_record_id=evidence.source_record_id,
            subject_kind=evidence.subject_kind,
            siren=evidence.siren,
            siret=evidence.siret,
            names=evidence.names,
            addresses=(),
            administrative_state=evidence.administrative_state,
            is_headquarters=evidence.is_headquarters,
            valid_from=evidence.valid_from,
            valid_to=evidence.valid_to,
            observed_at=evidence.observed_at,
            is_current=evidence.is_current,
        )
    return None


def resolve_evidence_precedence(
    records: Sequence[OfficialEvidence],
) -> tuple[tuple[OfficialEvidence, ...], tuple[QuarantinedOfficialRecord, ...]]:
    """Resolve one subject group without learning anything from CRM labels."""
    ordered = sorted(
        records,
        key=lambda item: (
            item.priority,
            item.observed_at or "",
            item.evidence_id,
        ),
        reverse=True,
    )
    deduplicated: list[OfficialEvidence] = []
    quarantined: list[QuarantinedOfficialRecord] = []
    semantic_seen: set[str] = set()
    for item in ordered:
        key = item.semantic_key()
        if key in semantic_seen:
            quarantined.append(
                QuarantinedOfficialRecord(
                    source=item.source,
                    snapshot_role="PRECEDENCE",
                    source_record_id=item.source_record_id,
                    reason=QuarantineReason.DUPLICATE_LOWER_PRECEDENCE,
                    detail="DUPLICATE OFFICIAL FACT",
                    record_fingerprint=item.evidence_id,
                )
            )
            continue
        semantic_seen.add(key)
        deduplicated.append(item)

    current_geo = [
        item
        for item in deduplicated
        if item.subject_kind is OfficialSubjectKind.SIRET
        and item.is_current
        and item.addresses
    ]
    if not current_geo:
        return tuple(deduplicated), tuple(quarantined)
    top_priority = max(item.priority for item in current_geo)
    anchors = [item for item in current_geo if item.priority == top_priority]
    ambiguous_top = any(
        _address_conflicts(left, right)
        for index, left in enumerate(anchors)
        for right in anchors[index + 1 :]
    )
    resolved: list[OfficialEvidence] = []
    for item in deduplicated:
        reason: QuarantineReason | None = None
        if item in current_geo:
            if ambiguous_top:
                reason = QuarantineReason.AMBIGUOUS_TOP_PRECEDENCE_CURRENT_GEO
            elif item.priority < top_priority and any(
                _address_conflicts(item, anchor) for anchor in anchors
            ):
                reason = QuarantineReason.LOWER_PRECEDENCE_CURRENT_GEO_CONFLICT
        if reason is None:
            resolved.append(item)
            continue
        quarantined.append(
            QuarantinedOfficialRecord(
                source=item.source,
                snapshot_role="PRECEDENCE",
                source_record_id=item.source_record_id,
                reason=reason,
                detail="CURRENT GEO CONFLICT ADDRESS REMOVED",
                record_fingerprint=item.evidence_id,
            )
        )
        sanitized = _strip_conflicting_addresses(item)
        if sanitized is not None:
            resolved.append(sanitized)
    return tuple(resolved), tuple(quarantined)


def _row_to_evidence(row: Mapping[str, Any]) -> OfficialEvidence:
    return OfficialEvidence.from_dict(row)


def _stream_ordered_groups(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    key_columns: Sequence[str],
    order_suffix: str,
    batch_size: int,
) -> Iterator[list[Mapping[str, Any]]]:
    escaped = str(path.resolve()).replace("'", "''")
    order = ", ".join(key_columns)
    cursor = connection.execute(
        f"SELECT * FROM read_parquet('{escaped}') ORDER BY {order}, {order_suffix}"
    )
    columns = [item[0] for item in cursor.description]
    current_key: tuple[Any, ...] | None = None
    group: list[Mapping[str, Any]] = []
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for raw in rows:
            row = dict(zip(columns, raw, strict=True))
            key = tuple(row[column] for column in key_columns)
            if current_key is not None and key != current_key:
                yield group
                group = []
            current_key = key
            group.append(row)
    if group:
        yield group


def build_official_evidence_layer(
    specs: Sequence[SnapshotSpec],
    output_dir: Path | str,
    *,
    work_dir: Path | str | None = None,
    batch_size: int = 4096,
) -> OfficialEvidenceBuildResult:
    """Build canonical Parquet artifacts with bounded-memory source scans."""
    if not specs:
        raise ValueError("at least one snapshot is required")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"official evidence output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_parent = Path(work_dir) if work_dir else output_dir.parent
    build_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=build_parent)
    )
    input_records = 0
    try:
        staged_evidence = temporary / "staged_evidence.parquet"
        staged_relations = temporary / "staged_relations.parquet"
        evidence_stage_sink = _ParquetSink(staged_evidence, official_evidence_arrow_schema(), batch_size)
        relation_stage_sink = _ParquetSink(staged_relations, official_relation_arrow_schema(), batch_size)
        quarantine_sink = _ParquetSink(
            temporary / "quarantine.parquet", official_quarantine_arrow_schema(), batch_size
        )
        for spec in specs:
            if not spec.path.is_file():
                raise FileNotFoundError(spec.path)
            for ordinal, record in enumerate(stream_snapshot_rows(spec), start=1):
                input_records += 1
                canonical = canonicalize_snapshot_record(
                    spec, record, ordinal=ordinal
                )
                for item in canonical.evidence:
                    evidence_stage_sink.add(item.to_dict())
                for item in canonical.relations:
                    relation_stage_sink.add(item.to_dict())
                for item in canonical.quarantine:
                    quarantine_sink.add(item.to_dict())
        evidence_stage_sink.close()
        relation_stage_sink.close()

        connection = duckdb.connect(str(temporary / "precedence.duckdb"))
        connection.execute("SET preserve_insertion_order=false")
        evidence_sink = _ParquetSink(
            temporary / "official_evidence.parquet",
            official_evidence_arrow_schema(),
            batch_size,
        )
        for group in _stream_ordered_groups(
            connection,
            staged_evidence,
            key_columns=["subject_kind", "subject_id"],
            order_suffix="source_priority DESC, observed_at DESC NULLS LAST, evidence_id",
            batch_size=batch_size,
        ):
            accepted, conflicts = resolve_evidence_precedence(
                [_row_to_evidence(row) for row in group]
            )
            for item in accepted:
                evidence_sink.add(item.to_dict())
            for item in conflicts:
                quarantine_sink.add(item.to_dict())
        evidence_sink.close()

        relation_sink = _ParquetSink(
            temporary / "official_relation.parquet",
            official_relation_arrow_schema(),
            batch_size,
        )
        for group in _stream_ordered_groups(
            connection,
            staged_relations,
            key_columns=[
                "relation_type",
                "from_kind",
                "from_identifier",
                "to_kind",
                "to_identifier",
                "effective_date",
            ],
            order_suffix="source_priority DESC, observed_at DESC NULLS LAST, relation_id",
            batch_size=batch_size,
        ):
            first = OfficialRelation.from_dict(group[0])
            relation_sink.add(first.to_dict())
            for duplicate in group[1:]:
                item = OfficialRelation.from_dict(duplicate)
                quarantine_sink.add(
                    QuarantinedOfficialRecord(
                        source=item.source,
                        snapshot_role="PRECEDENCE",
                        source_record_id=item.source_record_id,
                        reason=QuarantineReason.DUPLICATE_LOWER_PRECEDENCE,
                        detail="DUPLICATE STRUCTURED RELATION",
                        record_fingerprint=item.relation_id,
                    ).to_dict()
                )
        relation_sink.close()
        quarantine_sink.close()
        connection.close()

        staged_evidence.unlink()
        staged_relations.unlink()
        (temporary / "precedence.duckdb").unlink(missing_ok=True)

        evidence_count = pq.ParquetFile(temporary / "official_evidence.parquet").metadata.num_rows
        relation_count = pq.ParquetFile(temporary / "official_relation.parquet").metadata.num_rows
        quarantine_count = pq.ParquetFile(temporary / "quarantine.parquet").metadata.num_rows
        manifest = {
            "schema_version": BUILDER_SCHEMA_VERSION,
            "contracts": {
                "official_evidence": OFFICIAL_EVIDENCE_SCHEMA_VERSION,
                "official_relation": OFFICIAL_RELATION_SCHEMA_VERSION,
                "quarantine": OFFICIAL_QUARANTINE_SCHEMA_VERSION,
            },
            "counts": {
                "input_records": input_records,
                "accepted_evidence": evidence_count,
                "accepted_relations": relation_count,
                "quarantined_records": quarantine_count,
            },
            "sources": [
                {
                    "path": str(spec.path.resolve()),
                    "source": spec.source.value,
                    "role": spec.role.value,
                    "size_bytes": spec.path.stat().st_size,
                }
                for spec in specs
            ],
            "precedence": {
                "SIRENE_CURRENT": 500,
                "SIRENE_SUCCESSION": 500,
                "SIRENE_HISTORY": 400,
                "RNE": 300,
                "BODACC": 200,
            },
            "exclusions": [
                "beneficial_owners",
                "directors",
                "bodacc_full_text",
                "relations_inferred_from_text",
                "crm_labels",
            ],
            "streaming": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
        return OfficialEvidenceBuildResult(
            output_dir=output_dir,
            evidence_path=output_dir / "official_evidence.parquet",
            relation_path=output_dir / "official_relation.parquet",
            quarantine_path=output_dir / "quarantine.parquet",
            manifest_path=output_dir / "manifest.json",
            input_records=input_records,
            accepted_evidence=evidence_count,
            accepted_relations=relation_count,
            quarantined_records=quarantine_count,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
