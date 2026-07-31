#!/usr/bin/env python3
"""Validate V4.13 query/oracle separation without retrieval or model imports."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


OPAQUE_RE = re.compile(r"^[a-p]{64}$")
SIRET_RE = re.compile(r"^[0-9]{14}$")
SIREN_RE = re.compile(r"^[0-9]{9}$")
DIGIT_LEAK_RE = re.compile(r"(?<!\d)(?:\d{14}|\d{9})(?!\d)")
QUERY_KEYS = {
    "query_id",
    "reference_date",
    "crm_name_raw",
    "crm_address_raw",
    "crm_postcode_raw",
    "crm_city_raw",
    "crm_insee_raw",
}
ORACLE_KEYS = {
    "query_id",
    "label",
    "authoritative_siret",
    "authoritative_siren",
    "reason_code",
    "evidence_count",
    "evidence_payload_sha256s",
}
LABELS = {"MATCH_EXACT", "AMBIGUOUS", "UNRESOLVED"}
FORBIDDEN_COLUMN_TOKENS = {
    "SIRET",
    "SIREN",
    "EVIDENCE",
    "LABEL",
    "CANDIDATE",
    "HIT",
    "RANK",
    "SCORE",
    "PREDICTION",
}


class ValidationStopped(RuntimeError):
    pass


def _stop(message: str) -> None:
    raise ValidationStopped(f"STOP_V413_ARTIFACT_VALIDATION: {message}")


def _validate_query(row: dict[str, Any], ordinal: int) -> str:
    if set(row) != QUERY_KEYS:
        _stop(f"query {ordinal} fields mismatch")
    for key in row:
        upper = key.upper()
        if any(token in upper for token in FORBIDDEN_COLUMN_TOKENS):
            _stop(f"query {ordinal} forbidden column: {key}")
    query_id = row["query_id"]
    if not isinstance(query_id, str) or not OPAQUE_RE.fullmatch(query_id):
        _stop(f"query {ordinal} invalid query_id")
    for key, value in row.items():
        if key == "query_id" or value is None:
            continue
        if not isinstance(value, str):
            _stop(f"query {ordinal} {key} must be string or null")
        normalized = unicodedata.normalize("NFKC", value)
        if DIGIT_LEAK_RE.search(normalized):
            _stop(f"query {ordinal} autonomous 9/14 digit leak in {key}")
    return query_id


def _validate_oracle(row: dict[str, Any], ordinal: int) -> tuple[str, str]:
    if set(row) != ORACLE_KEYS:
        _stop(f"oracle {ordinal} fields mismatch")
    query_id = row["query_id"]
    if not isinstance(query_id, str) or not OPAQUE_RE.fullmatch(query_id):
        _stop(f"oracle {ordinal} invalid query_id")
    label = row["label"]
    if label not in LABELS:
        _stop(f"oracle {ordinal} invalid label")
    siret, siren = row["authoritative_siret"], row["authoritative_siren"]
    if siret is not None and (not isinstance(siret, str) or not SIRET_RE.fullmatch(siret)):
        _stop(f"oracle {ordinal} invalid SIRET")
    if siren is not None and (not isinstance(siren, str) or not SIREN_RE.fullmatch(siren)):
        _stop(f"oracle {ordinal} invalid SIREN")
    if siret is not None and siren is not None and siret[:9] != siren:
        _stop(f"oracle {ordinal} SIRET/SIREN mismatch")
    count = row["evidence_count"]
    hashes = row["evidence_payload_sha256s"]
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or not isinstance(hashes, list)
        or len(hashes) != count
        or hashes != sorted(set(hashes))
        or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)
    ):
        _stop(f"oracle {ordinal} invalid evidence inventory")
    if label == "MATCH_EXACT" and (siret is None or siren is None or count < 1):
        _stop(f"oracle {ordinal} MATCH_EXACT requires unique evidence and SIRET")
    if label == "UNRESOLVED" and (siret is not None or siren is not None):
        _stop(f"oracle {ordinal} UNRESOLVED cannot expose truth")
    if not isinstance(row["reason_code"], str) or not row["reason_code"]:
        _stop(f"oracle {ordinal} reason_code required")
    return query_id, label


def validate_artifacts(
    queries: Iterable[dict[str, Any]],
    oracle: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    query_rows, oracle_rows = list(queries), list(oracle)
    if not query_rows or len(query_rows) != len(oracle_rows):
        _stop("queries and oracle must be nonempty and equal length")
    query_ids = [_validate_query(row, i) for i, row in enumerate(query_rows, 1)]
    oracle_values = [_validate_oracle(row, i) for i, row in enumerate(oracle_rows, 1)]
    oracle_ids = [value[0] for value in oracle_values]
    if len(set(query_ids)) != len(query_ids) or len(set(oracle_ids)) != len(oracle_ids):
        _stop("duplicate query_id")
    if set(query_ids) != set(oracle_ids):
        _stop("query/oracle ID sets differ")
    counts = {label: 0 for label in sorted(LABELS)}
    for _, label in oracle_values:
        counts[label] += 1
    total = len(query_rows)
    return {
        "schema_version": "sireto-v4.13-artifact-validation-result-1",
        "row_count": total,
        "label_counts": counts,
        "match_exact_coverage": counts["MATCH_EXACT"] / total,
        "query_leak_count": 0,
        "verdict": "VALID",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            _stop(f"{path.name} line {ordinal} is not object")
        rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--oracle-jsonl", type=Path, required=True)
    args = parser.parse_args()
    result = validate_artifacts(
        _read_jsonl(args.queries_jsonl), _read_jsonl(args.oracle_jsonl)
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
