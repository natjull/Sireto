from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_v413", ROOT / "scripts/validate_v413_fresh_artifacts.py"
)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def query(query_id: str = "a" * 64) -> dict:
    return {
        "query_id": query_id,
        "reference_date": "2026-08-01",
        "crm_name_raw": "École des Lilas",
        "crm_address_raw": "1 rue des Lilas",
        "crm_postcode_raw": "75001",
        "crm_city_raw": "Paris",
        "crm_insee_raw": "75101",
    }


def oracle(query_id: str = "a" * 64) -> dict:
    return {
        "query_id": query_id,
        "label": "MATCH_EXACT",
        "authoritative_siret": "12345678900012",
        "authoritative_siren": "123456789",
        "reason_code": "UNIQUE_VALID_AUTHORITY",
        "evidence_count": 1,
        "evidence_payload_sha256s": ["1" * 64],
    }


def test_valid_pair_reports_exact_denominator_and_coverage() -> None:
    result = subject.validate_artifacts([query()], [oracle()])
    assert result["row_count"] == 1
    assert result["label_counts"]["MATCH_EXACT"] == 1
    assert result["match_exact_coverage"] == 1.0
    assert result["query_leak_count"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("crm_name_raw", "SIRET 12345678901234"),
        ("crm_name_raw", "SIREN 123456789"),
        ("crm_name_raw", "Unicode １２３４５６７８９"),
        ("crm_address_raw", "12345678901234"),
    ],
)
def test_ascii_and_unicode_truth_leaks_are_rejected(field: str, value: str) -> None:
    candidate = query()
    candidate[field] = value
    with pytest.raises(subject.ValidationStopped):
        subject.validate_artifacts([candidate], [oracle()])


def test_postcode_and_insee_are_not_false_positive_leaks() -> None:
    subject.validate_artifacts([query()], [oracle()])


@pytest.mark.parametrize(
    "mutation",
    ["extra_query_field", "id_mismatch", "bad_siret_siren", "unresolved_truth", "bad_evidence"],
)
def test_schema_join_and_oracle_invariants_fail_closed(mutation: str) -> None:
    q, o = query(), oracle()
    if mutation == "extra_query_field":
        q["candidate_score"] = 1.0
    elif mutation == "id_mismatch":
        o["query_id"] = "b" * 64
    elif mutation == "bad_siret_siren":
        o["authoritative_siren"] = "987654321"
    elif mutation == "unresolved_truth":
        o["label"] = "UNRESOLVED"
    else:
        o["evidence_count"] = 2
    with pytest.raises(subject.ValidationStopped):
        subject.validate_artifacts([q], [o])
