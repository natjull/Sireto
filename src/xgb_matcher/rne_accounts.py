"""Privacy-minimal RNE annual-account deposit metadata for the SIREN dossier."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Mapping
import zipfile

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

from .official_source_sync import canonical_json, sha256_file


RNE_ACCOUNT_DEPOSITS_SCHEMA_VERSION = "sireto-rne-account-deposits-v1"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(_text(value)[:10])
    except ValueError:
        return None


def _nested(record: Mapping[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


ACCOUNT_DEPOSIT_SCHEMA = pa.schema(
    [
        pa.field("source_record_uid", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("archive_member", pa.string(), nullable=False),
        pa.field("source_record_ordinal", pa.int64(), nullable=False),
        pa.field("filing_id", pa.string(), nullable=False),
        pa.field("siren", pa.string(), nullable=False),
        pa.field("denomination", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32()),
        pa.field("closing_date", pa.date32()),
        pa.field("previous_closing_date", pa.date32()),
        pa.field("updated_at", pa.string(), nullable=False),
        pa.field("chronology_number", pa.string(), nullable=False),
        pa.field("confidentiality", pa.string(), nullable=False),
        pa.field("is_public", pa.bool_(), nullable=False),
        pa.field("is_deleted", pa.bool_(), nullable=False),
        pa.field("account_type", pa.string(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("duration_months", pa.int16()),
        pa.field("activity_code", pa.string(), nullable=False),
        pa.field("structured_accounts_present", pa.bool_(), nullable=False),
    ]
)


@dataclass(frozen=True)
class RneAccountDepositsBuild:
    output_dir: Path
    deposits_path: Path
    manifest_path: Path
    count: int


def _iter_zip_records(path: Path) -> Iterator[tuple[str, int, Mapping[str, Any]]]:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() != ".json":
                continue
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted RNE account member: {info.filename}")
            with archive.open(info) as binary:
                buffered = io.BufferedReader(binary)
                prefix = buffered.peek(4096).lstrip(b"\xef\xbb\xbf \t\r\n")
                if not prefix.startswith(b"["):
                    raise ValueError(
                        f"RNE account member is not a root JSON array: {info.filename}"
                    )
                for ordinal, record in enumerate(ijson.items(buffered, "item"), start=1):
                    if not isinstance(record, Mapping):
                        raise ValueError(
                            f"RNE account member record is not an object: {info.filename}:{ordinal}"
                        )
                    yield info.filename, ordinal, record


def _canonical_row(
    record: Mapping[str, Any], *, snapshot_id: str, member: str, ordinal: int
) -> dict[str, Any] | None:
    siren = "".join(character for character in _text(record.get("siren")) if character.isdigit())
    if len(siren) != 9:
        return None
    identity = _nested(record, "bilanSaisi", "bilan", "identite")
    identity = identity if isinstance(identity, Mapping) else {}
    filing_id = _text(record.get("id")) or f"{member}:{ordinal}"
    confidentiality = _text(record.get("confidentiality"))
    is_public = confidentiality.casefold() == "public"
    duration = _text(identity.get("dureeExerciceN"))
    uid = hashlib.sha256(
        f"{snapshot_id}|{member}|{filing_id}|{ordinal}".encode("utf-8")
    ).hexdigest()
    return {
        "source_record_uid": uid,
        "snapshot_id": snapshot_id,
        "archive_member": member,
        "source_record_ordinal": ordinal,
        "filing_id": filing_id,
        "siren": siren,
        "denomination": _text(record.get("denomination")),
        "filing_date": _date(record.get("dateDepot") or identity.get("dateDepot")),
        "closing_date": _date(record.get("dateCloture") or identity.get("dateClotureExercice")),
        "previous_closing_date": _date(identity.get("dateClotureExerciceNMoins1")),
        "updated_at": _text(record.get("updatedAt")),
        "chronology_number": _text(record.get("numChrono") or identity.get("numDepot")),
        "confidentiality": confidentiality,
        "is_public": is_public,
        "is_deleted": bool(record.get("deleted", False)),
        "account_type": _text(record.get("typeBilan") or identity.get("codeTypeBilan")),
        "currency": _text(identity.get("codeDevise")),
        "duration_months": int(duration) if duration.isdigit() else None,
        "activity_code": _text(identity.get("codeActivite")),
        # Deliberately records availability only. Detailed liasse cells remain
        # outside the dossier/model contract until a separate approved mapping.
        "structured_accounts_present": bool(_nested(record, "bilanSaisi", "bilan", "detail")),
    }


def build_rne_account_deposits(
    *, manifest_path: Path, payload_name: str, output_root: Path, batch_size: int = 4096
) -> RneAccountDepositsBuild:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("payload") or manifest.get("observed_payload") or []
    selected = next((item for item in items if item.get("name") == payload_name), None)
    if not isinstance(selected, Mapping):
        raise ValueError(f"RNE account payload not found: {payload_name}")
    archive = manifest_path.parent / payload_name
    if not archive.is_file():
        raise FileNotFoundError(archive)
    actual_sha = sha256_file(archive)
    expected_sha = _text(selected.get("sha256"))
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("RNE account archive SHA-256 mismatch")
    snapshot_id = _text(manifest.get("build_id")) or hashlib.sha256(
        canonical_json(manifest)
    ).hexdigest()
    identity = {
        "schema_version": RNE_ACCOUNT_DEPOSITS_SCHEMA_VERSION,
        "input": {"name": payload_name, "size_bytes": archive.stat().st_size, "sha256": actual_sha},
        "snapshot_id": snapshot_id,
        "policy": {
            "siren_level_only": True,
            "detailed_account_cells_present": False,
            "personal_data_present": False,
            "model_use_enabled": False,
        },
    }
    build_id = hashlib.sha256(canonical_json(identity)).hexdigest()
    output_root = Path(output_root)
    final = output_root / build_id[:16]
    if final.exists():
        saved = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        return RneAccountDepositsBuild(final, final / "account_deposits.parquet", final / "manifest.json", int(saved["count"]))
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".rne-accounts-", dir=output_root))
    writer = pq.ParquetWriter(stage / "account_deposits.parquet", ACCOUNT_DEPOSIT_SCHEMA, compression="zstd")
    count = 0
    rows: list[dict[str, Any]] = []
    try:
        for member, ordinal, record in _iter_zip_records(archive):
            row = _canonical_row(record, snapshot_id=snapshot_id, member=member, ordinal=ordinal)
            if row is None:
                continue
            rows.append(row)
            if len(rows) >= batch_size:
                writer.write_table(pa.Table.from_pylist(rows, schema=ACCOUNT_DEPOSIT_SCHEMA))
                count += len(rows)
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=ACCOUNT_DEPOSIT_SCHEMA))
            count += len(rows)
        writer.close()
        output = stage / "account_deposits.parquet"
        result_manifest = {
            **identity,
            "build_id": build_id,
            "count": count,
            "output": {"name": output.name, "size_bytes": output.stat().st_size, "sha256": sha256_file(output)},
        }
        (stage / "manifest.json").write_bytes(canonical_json(result_manifest))
        os.rename(stage, final)
        return RneAccountDepositsBuild(final, final / output.name, final / "manifest.json", count)
    except Exception:
        writer.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise
