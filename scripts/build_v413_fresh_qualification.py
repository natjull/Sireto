#!/usr/bin/env python3
"""Build V4.13 qualification artifacts from synthetic fixtures only.

This module deliberately has no retrieval/model import and cannot operate on
the real V4.13 inbox.  Its file-writing API is restricted to the operating
system temporary directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CRM_COLUMNS = [
    "source_record_id",
    "source_group_id",
    "source_record_created_at_utc",
    "reference_date",
    "crm_name_raw",
    "crm_address_raw",
    "crm_postcode_raw",
    "crm_city_raw",
    "crm_insee_raw",
    "source_record_id_equivalence_attested",
]
MAPPING_COLUMNS = [
    "source_record_id",
    "authority_type",
    "authority_issuer_id",
    "authority_system_id",
    "authority_record_id",
    "authoritative_siret",
    "authoritative_siren",
    "valid_from",
    "valid_to",
    "evidence_created_at_utc",
    "evidence_payload_sha256",
    "matching_pipeline_used",
]
QUERY_COLUMNS = [
    "query_id",
    "reference_date",
    "crm_name_raw",
    "crm_address_raw",
    "crm_postcode_raw",
    "crm_city_raw",
    "crm_insee_raw",
]
ORACLE_COLUMNS = [
    "query_id",
    "label",
    "authoritative_siret",
    "authoritative_siren",
    "reason_code",
    "evidence_count",
    "evidence_payload_sha256s",
]
ALLOWED_AUTHORITY_TYPES = {
    "SOURCE_SYSTEM_OFFICIAL_SIRET",
    "CONTRACT_OR_BILLING_SIRET",
    "SEALED_ADMINISTRATIVE_DOCUMENT",
}
MANIFEST_FIELDS = [
    "authority_catalog_id",
    "collection_id",
    "created_at_utc",
    "crm_file",
    "crm_format",
    "crm_row_count",
    "crm_sha256",
    "crm_size_bytes",
    "export_cutoff_utc",
    "export_id",
    "mapping_file",
    "mapping_format",
    "mapping_row_count",
    "mapping_sha256",
    "mapping_size_bytes",
    "matching_based_exclusions",
    "period_end_utc",
    "period_start_utc",
    "plan_git_commit",
    "plan_sha256",
    "population_definition",
    "population_exclusions",
    "population_is_exhaustive",
    "preregistration_lock_sha256",
    "producer_id",
    "reference_date",
    "schema_version",
    "source_record_id_semantics",
]
OPAQUE_DOMAIN = b"SIRETO-V413-OPAQUE-QUERY-ID\0"
HEX_TO_OPAQUE = str.maketrans("0123456789abcdef", "abcdefghijklmnop")


class QualificationError(ValueError):
    """The synthetic fixture violates the frozen V4.13 qualification rules."""


def _exact_keys(row: Mapping[str, Any], expected: Sequence[str], kind: str) -> None:
    if list(row) != list(expected):
        raise QualificationError(
            f"{kind} columns must be exact and ordered: {list(expected)!r}"
        )


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise QualificationError(f"{field} must be a string or null")
    return value


def _parse_date(value: Any, field: str) -> date:
    text = _nonempty_string(value, field)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise QualificationError(f"{field} must be an ISO-8601 date") from exc
    if parsed.isoformat() != text:
        raise QualificationError(f"{field} must use canonical YYYY-MM-DD")
    return parsed


def _parse_utc(value: Any, field: str) -> datetime:
    text = _nonempty_string(value, field)
    if not text.endswith("Z"):
        raise QualificationError(f"{field} must be RFC3339 UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise QualificationError(f"{field} must be RFC3339 UTC") from exc
    if parsed.tzinfo != timezone.utc:
        raise QualificationError(f"{field} must be UTC")
    return parsed


def _optional_identifier(value: Any, length: int, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != length or not value.isascii():
        raise QualificationError(f"{field} must be {length} ASCII digits or null")
    if not value.isdigit():
        raise QualificationError(f"{field} must be {length} ASCII digits or null")
    return value


def _hex64(value: Any, field: str) -> str:
    text = _nonempty_string(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise QualificationError(f"{field} must be 64 lowercase hexadecimal chars")
    return text


def _compact_canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )


def opaque_query_id(
    collection_manifest_sha256: str,
    source_file_sha256: str,
    source_row_ordinal_1_based: int,
    source_record_id: str,
) -> str:
    """Return the frozen a..p opaque query identifier."""
    _hex64(collection_manifest_sha256, "collection_manifest_sha256")
    _hex64(source_file_sha256, "source_file_sha256")
    if type(source_row_ordinal_1_based) is not int or source_row_ordinal_1_based < 1:
        raise QualificationError("source_row_ordinal_1_based must be positive")
    _nonempty_string(source_record_id, "source_record_id")
    projection = [
        collection_manifest_sha256,
        source_file_sha256,
        source_row_ordinal_1_based,
        source_record_id,
    ]
    digest = hashlib.sha256(OPAQUE_DOMAIN + _compact_canonical_json(projection))
    return digest.hexdigest().translate(HEX_TO_OPAQUE)


def _standalone_digit_run(value: str) -> tuple[int, str] | None:
    projected = unicodedata.normalize("NFKC", value)
    run: list[str] = []
    for character in projected + "\0":
        if character.isdecimal():
            run.append(character)
            continue
        if len(run) in {9, 14}:
            return len(run), "".join(run)
        run.clear()
    return None


def scan_query_for_truth(query: Mapping[str, Any]) -> None:
    """Reject any standalone 9/14 decimal run after NFKC."""
    _exact_keys(query, QUERY_COLUMNS, "query")
    for field, value in query.items():
        if value is None:
            continue
        if not isinstance(value, str):
            raise QualificationError(f"query field {field} must be string or null")
        leaked = _standalone_digit_run(value)
        if leaked:
            raise QualificationError(
                f"query truth leak in {field}: standalone {leaked[0]}-digit sequence"
            )


def _authority_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["authority_type"]),
        str(row["authority_issuer_id"]),
        str(row["authority_system_id"]),
    )


def _synthetic_allowlist(catalog: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    if catalog.get("real_collection_open_authorized") is not False:
        raise QualificationError("synthetic catalog must keep real collection opening false")
    entries = catalog.get("synthetic_test_authorities")
    if not isinstance(entries, list) or not entries:
        raise QualificationError("an explicit synthetic fixture allowlist is required")
    allowed: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "authority_type",
            "authority_issuer_id",
            "authority_system_id",
            "test_only",
        }:
            raise QualificationError("synthetic authority entries have a closed schema")
        if entry["test_only"] is not True:
            raise QualificationError("synthetic authority must be marked test_only")
        authority_type = _nonempty_string(entry["authority_type"], "authority_type")
        issuer = _nonempty_string(entry["authority_issuer_id"], "authority_issuer_id")
        system = _nonempty_string(entry["authority_system_id"], "authority_system_id")
        if authority_type not in ALLOWED_AUTHORITY_TYPES:
            raise QualificationError("unsupported authority type")
        if not issuer.startswith("TEST_") or not system.startswith("TEST_"):
            raise QualificationError("synthetic issuer and system must start with TEST_")
        allowed.add((authority_type, issuer, system))
    return allowed


def _positive_integer(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise QualificationError(f"{field} must be a positive integer")
    return value


def _safe_basename(value: Any, field: str) -> str:
    text = _nonempty_string(value, field)
    if Path(text).name != text or text in {".", ".."}:
        raise QualificationError(f"{field} must be a safe basename")
    return text


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[date, datetime]:
    _exact_keys(manifest, MANIFEST_FIELDS, "collection manifest")
    if manifest["authority_catalog_id"] != "v413-independent-authorities-1":
        raise QualificationError("unexpected authority_catalog_id")
    for field in (
        "collection_id",
        "export_id",
        "population_definition",
        "producer_id",
        "source_record_id_semantics",
    ):
        _nonempty_string(manifest[field], field)
    _safe_basename(manifest["crm_file"], "crm_file")
    _safe_basename(manifest["mapping_file"], "mapping_file")
    if manifest["crm_format"] not in {"CSV", "PARQUET"}:
        raise QualificationError("crm_format must be CSV or PARQUET")
    if manifest["mapping_format"] not in {"CSV", "PARQUET"}:
        raise QualificationError("mapping_format must be CSV or PARQUET")
    for field in (
        "crm_row_count",
        "crm_size_bytes",
        "mapping_row_count",
        "mapping_size_bytes",
    ):
        _positive_integer(manifest[field], field)
    for field in (
        "crm_sha256",
        "mapping_sha256",
        "plan_sha256",
        "preregistration_lock_sha256",
    ):
        _hex64(manifest[field], field)
    commit = _nonempty_string(manifest["plan_git_commit"], "plan_git_commit")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise QualificationError("plan_git_commit must be 40 lowercase hexadecimal chars")
    if manifest["matching_based_exclusions"] is not False:
        raise QualificationError("matching_based_exclusions must be boolean false")
    if manifest["population_is_exhaustive"] is not True:
        raise QualificationError("population_is_exhaustive must be boolean true")
    if not isinstance(manifest["population_exclusions"], list):
        raise QualificationError("population_exclusions must be an array")
    if manifest["schema_version"] != "sireto-v4.13-collection-manifest-1":
        raise QualificationError("unexpected collection manifest schema_version")
    start = _parse_utc(manifest["period_start_utc"], "period_start_utc")
    end = _parse_utc(manifest["period_end_utc"], "period_end_utc")
    cutoff = _parse_utc(manifest["export_cutoff_utc"], "export_cutoff_utc")
    created = _parse_utc(manifest["created_at_utc"], "created_at_utc")
    reference = _parse_date(manifest["reference_date"], "reference_date")
    if not start <= end <= cutoff <= created:
        raise QualificationError(
            "manifest must satisfy period_start <= period_end <= export_cutoff <= created_at"
        )
    if not (start.date() <= reference <= end.date() or reference == cutoff.date()):
        raise QualificationError("manifest reference_date is outside the frozen period")
    return reference, cutoff


def _validate_crm_row(row: Mapping[str, Any], manifest_reference: date) -> None:
    _exact_keys(row, CRM_COLUMNS, "CRM")
    _nonempty_string(row["source_record_id"], "source_record_id")
    _optional_string(row["source_group_id"], "source_group_id")
    _parse_utc(row["source_record_created_at_utc"], "source_record_created_at_utc")
    if _parse_date(row["reference_date"], "reference_date") != manifest_reference:
        raise QualificationError("CRM reference_date must equal manifest reference_date")
    for field in CRM_COLUMNS[4:9]:
        _optional_string(row[field], field)
    if row["source_record_id_equivalence_attested"] is not True:
        raise QualificationError(
            "source_record_id_equivalence_attested must be true for every row"
        )


def _validate_mapping_row(
    row: Mapping[str, Any],
    allowed: set[tuple[str, str, str]],
    export_cutoff: datetime,
) -> tuple[str | None, str | None, date | None, date | None]:
    _exact_keys(row, MAPPING_COLUMNS, "authoritative mapping")
    _nonempty_string(row["source_record_id"], "source_record_id")
    authority_type = _nonempty_string(row["authority_type"], "authority_type")
    if authority_type not in ALLOWED_AUTHORITY_TYPES or _authority_key(row) not in allowed:
        raise QualificationError("authority is absent from the explicit synthetic allowlist")
    _nonempty_string(row["authority_issuer_id"], "authority_issuer_id")
    _nonempty_string(row["authority_system_id"], "authority_system_id")
    _nonempty_string(row["authority_record_id"], "authority_record_id")
    siret = _optional_identifier(row["authoritative_siret"], 14, "authoritative_siret")
    siren = _optional_identifier(row["authoritative_siren"], 9, "authoritative_siren")
    if siret is not None and siren is not None and siret[:9] != siren:
        raise QualificationError("authoritative SIRET/SIREN are inconsistent")
    valid_from = (
        _parse_date(row["valid_from"], "valid_from")
        if row["valid_from"] is not None
        else None
    )
    valid_to = (
        _parse_date(row["valid_to"], "valid_to")
        if row["valid_to"] is not None
        else None
    )
    if valid_from is not None and valid_to is not None and valid_from > valid_to:
        raise QualificationError("valid_from must be <= valid_to")
    if _parse_utc(row["evidence_created_at_utc"], "evidence_created_at_utc") > export_cutoff:
        raise QualificationError("evidence_created_at_utc must be <= export_cutoff")
    _hex64(row["evidence_payload_sha256"], "evidence_payload_sha256")
    if row["matching_pipeline_used"] is not False:
        raise QualificationError("matching_pipeline_used must be boolean false")
    return siret, siren, valid_from, valid_to


def qualify_fixture_rows(
    *,
    manifest: Mapping[str, Any],
    collection_manifest_sha256: str,
    source_file_sha256: str,
    crm_rows: Sequence[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, Any]],
    authority_catalog: Mapping[str, Any],
    synthetic_fixtures_only: bool,
) -> dict[str, Any]:
    """Validate and qualify a complete in-memory synthetic frame."""
    if synthetic_fixtures_only is not True:
        raise QualificationError("this builder is authorized for synthetic fixtures only")
    _hex64(collection_manifest_sha256, "collection_manifest_sha256")
    _hex64(source_file_sha256, "source_file_sha256")
    reference, cutoff = _validate_manifest(manifest)
    computed_manifest_sha = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    if collection_manifest_sha256 != computed_manifest_sha:
        raise QualificationError("collection manifest SHA does not match canonical bytes")
    if source_file_sha256 != manifest["crm_sha256"]:
        raise QualificationError("source file SHA does not match manifest crm_sha256")
    if manifest["crm_row_count"] != len(crm_rows):
        raise QualificationError("CRM row count does not match manifest")
    if manifest["mapping_row_count"] != len(mapping_rows):
        raise QualificationError("mapping row count does not match manifest")
    allowed = _synthetic_allowlist(authority_catalog)

    source_ids: set[str] = set()
    for row in crm_rows:
        _validate_crm_row(row, reference)
        source_id = str(row["source_record_id"])
        if source_id in source_ids:
            raise QualificationError("source_record_id must be unique")
        source_ids.add(source_id)

    mappings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_row in mapping_rows:
        parsed = _validate_mapping_row(raw_row, allowed, cutoff)
        source_id = str(raw_row["source_record_id"])
        if source_id not in source_ids:
            raise QualificationError("mapping contains an orphan source_record_id")
        row = dict(raw_row)
        row["_parsed"] = parsed
        mappings[source_id].append(row)

    queries: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for ordinal, crm in enumerate(crm_rows, start=1):
        source_id = str(crm["source_record_id"])
        query_id = opaque_query_id(
            collection_manifest_sha256,
            source_file_sha256,
            ordinal,
            source_id,
        )
        query = {
            "query_id": query_id,
            "reference_date": crm["reference_date"],
            "crm_name_raw": crm["crm_name_raw"],
            "crm_address_raw": crm["crm_address_raw"],
            "crm_postcode_raw": crm["crm_postcode_raw"],
            "crm_city_raw": crm["crm_city_raw"],
            "crm_insee_raw": crm["crm_insee_raw"],
        }
        scan_query_for_truth(query)

        eligible: list[dict[str, Any]] = []
        for evidence in mappings[source_id]:
            siret, siren, valid_from, valid_to = evidence["_parsed"]
            if valid_from is not None and reference < valid_from:
                continue
            if valid_to is not None and reference > valid_to:
                continue
            if siret is None and siren is None:
                continue
            eligible.append(evidence)

        sirets = {
            evidence["_parsed"][0]
            for evidence in eligible
            if evidence["_parsed"][0] is not None
        }
        sirens = {
            evidence["_parsed"][1] or evidence["_parsed"][0][:9]
            for evidence in eligible
            if evidence["_parsed"][1] is not None
            or evidence["_parsed"][0] is not None
        }
        has_siren_only = any(
            evidence["_parsed"][0] is None and evidence["_parsed"][1] is not None
            for evidence in eligible
        )
        if len(sirets) == 1 and len(sirens) == 1 and not has_siren_only:
            label = "MATCH_EXACT"
            authoritative_siret = next(iter(sirets))
            authoritative_siren = authoritative_siret[:9]
            reason = "UNIQUE_TEMPORALLY_COHERENT_SYNTHETIC_AUTHORITY"
        elif eligible and (sirets or sirens):
            label = "AMBIGUOUS"
            authoritative_siret = None
            authoritative_siren = next(iter(sirens)) if len(sirens) == 1 else None
            reason = (
                "SIREN_ONLY"
                if has_siren_only and not sirets
                else "MULTIPLE_OR_CONTRADICTORY_AUTHORITIES"
            )
        else:
            label = "UNRESOLVED"
            authoritative_siret = None
            authoritative_siren = None
            reason = "NO_ADMISSIBLE_AUTHORITY_AT_REFERENCE_DATE"

        hashes = sorted(
            {str(evidence["evidence_payload_sha256"]) for evidence in eligible}
        )
        all_known_sirens = sorted(
            {
                parsed_siren or parsed_siret[:9]
                for evidence in mappings[source_id]
                for parsed_siret, parsed_siren, _, _ in [evidence["_parsed"]]
                if parsed_siren is not None or parsed_siret is not None
            }
        )
        queries.append(query)
        oracle.append(
            {
                "query_id": query_id,
                "label": label,
                "authoritative_siret": authoritative_siret,
                "authoritative_siren": authoritative_siren,
                "reason_code": reason,
                "evidence_count": len(hashes),
                "evidence_payload_sha256s": hashes,
            }
        )
        split_rows.append(
            {
                "schema_version": "sireto-v4.13-split-input-row-1",
                "query_id": query_id,
                "source_group_id": crm["source_group_id"],
                "authoritative_sirens": all_known_sirens,
            }
        )

    counts = {
        label: sum(row["label"] == label for row in oracle)
        for label in ("MATCH_EXACT", "AMBIGUOUS", "UNRESOLVED")
    }
    counts["source_rows"] = len(crm_rows)
    return {
        "queries": queries,
        "oracle": oracle,
        "split_rows": split_rows,
        "counts": counts,
    }


def _assert_temporary_output(output_root: Path) -> Path:
    root = output_root.resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    if root == temporary or temporary not in root.parents:
        raise QualificationError("fixture outputs must be below the OS temporary directory")
    if output_root.is_symlink():
        raise QualificationError("output root cannot be a symlink")
    return root


def _validate_result_for_write(result: Mapping[str, Any]) -> None:
    if set(result) != {"queries", "oracle", "split_rows", "counts"}:
        raise QualificationError("qualification result schema mismatch")
    queries = result["queries"]
    oracle = result["oracle"]
    split_rows = result["split_rows"]
    if (
        not isinstance(queries, list)
        or not isinstance(oracle, list)
        or not isinstance(split_rows, list)
        or not queries
        or len(queries) != len(oracle)
        or len(queries) != len(split_rows)
    ):
        raise QualificationError("qualification result row sets mismatch")
    query_ids: list[str] = []
    labels: list[str] = []
    for query in queries:
        scan_query_for_truth(query)
        query_id = _nonempty_string(query["query_id"], "query_id")
        if len(query_id) != 64 or any(character not in "abcdefghijklmnop" for character in query_id):
            raise QualificationError("query_id must be opaque a-p lowercase")
        query_ids.append(query_id)
    for row in oracle:
        _exact_keys(row, ORACLE_COLUMNS, "oracle")
        query_id = _nonempty_string(row["query_id"], "query_id")
        label = row["label"]
        if label not in {"MATCH_EXACT", "AMBIGUOUS", "UNRESOLVED"}:
            raise QualificationError("oracle label invalid")
        siret = _optional_identifier(row["authoritative_siret"], 14, "authoritative_siret")
        siren = _optional_identifier(row["authoritative_siren"], 9, "authoritative_siren")
        if siret is not None and siren is not None and siret[:9] != siren:
            raise QualificationError("oracle SIRET/SIREN mismatch")
        hashes = row["evidence_payload_sha256s"]
        if (
            not isinstance(hashes, list)
            or hashes != sorted(set(hashes))
            or any(_hex64(value, "evidence_payload_sha256") != value for value in hashes)
            or type(row["evidence_count"]) is not int
            or row["evidence_count"] != len(hashes)
        ):
            raise QualificationError("oracle evidence inventory invalid")
        if label == "MATCH_EXACT" and (
            siret is None or siren is None or not hashes
        ):
            raise QualificationError("MATCH_EXACT oracle invariant")
        if label == "AMBIGUOUS" and (siret is not None or not hashes):
            raise QualificationError("AMBIGUOUS oracle invariant")
        if label == "UNRESOLVED" and (
            siret is not None or siren is not None or hashes
        ):
            raise QualificationError("UNRESOLVED oracle invariant")
        _nonempty_string(row["reason_code"], "reason_code")
        query_ids.append(query_id)
        labels.append(label)
    query_half = query_ids[: len(queries)]
    oracle_half = query_ids[len(queries) :]
    if (
        len(set(query_half)) != len(query_half)
        or len(set(oracle_half)) != len(oracle_half)
        or set(query_half) != set(oracle_half)
    ):
        raise QualificationError("query/oracle ID sets mismatch")
    split_ids: list[str] = []
    for row in split_rows:
        if set(row) != {
            "schema_version",
            "query_id",
            "source_group_id",
            "authoritative_sirens",
        }:
            raise QualificationError("split row schema mismatch")
        if row["schema_version"] != "sireto-v4.13-split-input-row-1":
            raise QualificationError("split row schema_version mismatch")
        split_ids.append(_nonempty_string(row["query_id"], "query_id"))
        group = row["source_group_id"]
        if group is not None and (not isinstance(group, str) or not group):
            raise QualificationError("split source_group_id invalid")
        sirens = row["authoritative_sirens"]
        if (
            not isinstance(sirens, list)
            or sirens != sorted(set(sirens))
            or any(_optional_identifier(value, 9, "authoritative_siren") is None for value in sirens)
        ):
            raise QualificationError("split authoritative_sirens invalid")
    if split_ids != query_half:
        raise QualificationError("query/split ID order mismatch")
    expected_counts = {
        label: labels.count(label)
        for label in ("MATCH_EXACT", "AMBIGUOUS", "UNRESOLVED")
    }
    expected_counts["source_rows"] = len(queries)
    if result["counts"] != expected_counts:
        raise QualificationError("qualification counts mismatch")


def _exclusive_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            if "evidence_payload_sha256s" in encoded:
                encoded["evidence_payload_sha256s"] = json.dumps(
                    encoded["evidence_payload_sha256s"], separators=(",", ":")
                )
            writer.writerow(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def write_fixture_outputs(result: Mapping[str, Any], output_root: Path) -> dict[str, Path]:
    """Write separated fixture artifacts, exclusively below the temp directory."""
    root = _assert_temporary_output(output_root)
    _validate_result_for_write(result)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    query_dir = root / "queries"
    oracle_dir = root / "oracle"
    audit_dir = root / "audit"
    for directory in (query_dir, oracle_dir, audit_dir):
        directory.mkdir(mode=0o700)
    query_path = query_dir / "queries.csv"
    oracle_path = oracle_dir / "oracle.csv"
    audit_path = audit_dir / "qualification.json"
    split_rows_path = audit_dir / "private_split_rows.jsonl"
    _exclusive_csv(query_path, QUERY_COLUMNS, result["queries"])
    _exclusive_csv(oracle_path, ORACLE_COLUMNS, result["oracle"])
    audit_payload = {
        "schema_version": "sireto-v4.13-synthetic-qualification-audit-1",
        "synthetic_fixtures_only": True,
        "counts": result["counts"],
    }
    fd = os.open(audit_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(_compact_canonical_json(audit_payload))
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(
        split_rows_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(fd, "wb") as handle:
        for row in result["split_rows"]:
            handle.write(_compact_canonical_json(row))
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "queries": query_path,
        "oracle": oracle_path,
        "audit": audit_path,
        "private_split_rows": split_rows_path,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crm-rows", type=Path, required=True)
    parser.add_argument("--mapping-rows", type=Path, required=True)
    parser.add_argument("--authority-catalog", type=Path, required=True)
    parser.add_argument("--collection-manifest-sha256", required=True)
    parser.add_argument("--source-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--synthetic-fixtures-only", action="store_true")
    args = parser.parse_args(argv)
    temporary = Path(tempfile.gettempdir()).resolve()
    for input_path in (
        args.manifest,
        args.crm_rows,
        args.mapping_rows,
        args.authority_catalog,
    ):
        resolved = input_path.resolve()
        if temporary not in resolved.parents:
            raise QualificationError("all fixture inputs must be below the OS temp directory")
    result = qualify_fixture_rows(
        manifest=_load_json(args.manifest),
        collection_manifest_sha256=args.collection_manifest_sha256,
        source_file_sha256=args.source_file_sha256,
        crm_rows=_load_json(args.crm_rows),
        mapping_rows=_load_json(args.mapping_rows),
        authority_catalog=_load_json(args.authority_catalog),
        synthetic_fixtures_only=args.synthetic_fixtures_only,
    )
    write_fixture_outputs(result, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
