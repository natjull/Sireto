from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from scripts.build_v413_fresh_qualification import (
    CRM_COLUMNS,
    MANIFEST_FIELDS,
    MAPPING_COLUMNS,
    QUERY_COLUMNS,
    QualificationError,
    opaque_query_id,
    qualify_fixture_rows,
    write_fixture_outputs,
)


SOURCE_SHA = "e" * 64


def manifest(*, crm_count: int = 1, mapping_count: int = 1) -> dict:
    values = {
        "authority_catalog_id": "v413-independent-authorities-1",
        "collection_id": "TEST_COLLECTION",
        "created_at_utc": "2026-07-31T13:00:00Z",
        "crm_file": "crm_source.csv",
        "crm_format": "CSV",
        "crm_row_count": crm_count,
        "crm_sha256": SOURCE_SHA,
        "crm_size_bytes": 100,
        "export_cutoff_utc": "2026-07-31T12:00:00Z",
        "export_id": "TEST_EXPORT",
        "mapping_file": "authoritative_mapping.csv",
        "mapping_format": "CSV",
        "mapping_row_count": mapping_count,
        "mapping_sha256": "d" * 64,
        "mapping_size_bytes": 100,
        "matching_based_exclusions": False,
        "period_end_utc": "2026-07-31T00:00:00Z",
        "period_start_utc": "2026-07-01T00:00:00Z",
        "plan_git_commit": "c" * 40,
        "plan_sha256": "b" * 64,
        "population_definition": "all synthetic fixture rows",
        "population_exclusions": [],
        "population_is_exhaustive": True,
        "preregistration_lock_sha256": "a" * 64,
        "producer_id": "TEST_PRODUCER",
        "reference_date": "2026-07-31",
        "schema_version": "sireto-v4.13-collection-manifest-1",
        "source_record_id_semantics": "synthetic stable identifier",
    }
    return {field: values[field] for field in MANIFEST_FIELDS}


def manifest_sha(value: dict) -> str:
    raw = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode()
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def catalog() -> dict:
    return {
        "real_collection_open_authorized": False,
        "synthetic_test_authorities": [
            {
                "authority_type": "CONTRACT_OR_BILLING_SIRET",
                "authority_issuer_id": "TEST_ISSUER",
                "authority_system_id": "TEST_BILLING",
                "test_only": True,
            }
        ],
    }


def crm_row(source_record_id: str, *, name: str = "Société Alpha") -> dict:
    return dict(
        zip(
            CRM_COLUMNS,
            [
                source_record_id,
                None,
                "2026-07-15T10:00:00Z",
                "2026-07-31",
                name,
                "1 rue de la Paix",
                "75001",
                "Paris",
                "75101",
                True,
            ],
            strict=True,
        )
    )


def mapping_row(
    source_record_id: str,
    *,
    siret: str | None = "12345678900011",
    siren: str | None = "123456789",
    record_id: str = "invoice-1",
    valid_from: str | None = "2026-01-01",
    valid_to: str | None = None,
) -> dict:
    return dict(
        zip(
            MAPPING_COLUMNS,
            [
                source_record_id,
                "CONTRACT_OR_BILLING_SIRET",
                "TEST_ISSUER",
                "TEST_BILLING",
                record_id,
                siret,
                siren,
                valid_from,
                valid_to,
                "2026-07-30T10:00:00Z",
                "a" * 64,
                False,
            ],
            strict=True,
        )
    )


def qualify(crm_rows: list[dict], mapping_rows: list[dict]) -> dict:
    effective_mappings = list(mapping_rows)
    if not effective_mappings:
        effective_mappings = [
            mapping_row(crm_rows[0]["source_record_id"], siret=None, siren=None)
        ]
    fixture_manifest = manifest(
        crm_count=len(crm_rows), mapping_count=len(effective_mappings)
    )
    return qualify_fixture_rows(
        manifest=fixture_manifest,
        collection_manifest_sha256=manifest_sha(fixture_manifest),
        source_file_sha256=SOURCE_SHA,
        crm_rows=crm_rows,
        mapping_rows=effective_mappings,
        authority_catalog=catalog(),
        synthetic_fixtures_only=True,
    )


def test_all_rows_remain_in_denominator_and_outputs_are_truth_separated(
    tmp_path: Path,
) -> None:
    rows = [crm_row("crm-1"), crm_row("crm-2"), crm_row("crm-3"), crm_row("crm-4")]
    mappings = [
        mapping_row("crm-1"),
        mapping_row("crm-2", siret="98765432100019", siren="987654321"),
        mapping_row(
            "crm-2",
            siret="98765432100027",
            siren="987654321",
            record_id="invoice-2",
        ),
        mapping_row("crm-3", siret=None, siren="111222333"),
    ]
    result = qualify(rows, mappings)

    assert result["counts"] == {
        "MATCH_EXACT": 1,
        "AMBIGUOUS": 2,
        "UNRESOLVED": 1,
        "source_rows": 4,
    }
    assert [row["label"] for row in result["oracle"]] == [
        "MATCH_EXACT",
        "AMBIGUOUS",
        "AMBIGUOUS",
        "UNRESOLVED",
    ]
    assert list(result["queries"][0]) == QUERY_COLUMNS
    assert not {
        "source_record_id",
        "label",
        "authoritative_siret",
        "authoritative_siren",
        "evidence_payload_sha256s",
    } & set(result["queries"][0])
    assert result["oracle"][0]["authoritative_siret"] == "12345678900011"
    assert result["oracle"][0]["authoritative_siren"] == "123456789"
    assert result["split_rows"][0] == {
        "schema_version": "sireto-v4.13-split-input-row-1",
        "query_id": result["queries"][0]["query_id"],
        "source_group_id": None,
        "authoritative_sirens": ["123456789"],
    }
    assert result["split_rows"][1]["authoritative_sirens"] == ["987654321"]
    assert result["split_rows"][2]["authoritative_sirens"] == ["111222333"]

    paths = write_fixture_outputs(result, tmp_path / "qualification")
    assert paths["queries"].parent != paths["oracle"].parent
    with paths["queries"].open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == QUERY_COLUMNS
        assert "12345678900011" not in handle.read()
    assert json.loads(paths["audit"].read_text())["counts"]["source_rows"] == 4
    split_rows = [
        json.loads(line)
        for line in paths["private_split_rows"].read_text().splitlines()
    ]
    assert split_rows == result["split_rows"]


def test_opaque_query_id_matches_frozen_vector() -> None:
    assert opaque_query_id("f" * 64, SOURCE_SHA, 1, "CRM-42") == (
        "hchckchghiebinmcgfdfbaifncohacngafphapcneafambbgeheamcplanppofhj"
    )


@pytest.mark.parametrize(
    "name",
    [
        "Client 123456789",
        "Client 12345678901234",
        "Client １２３４５６７８９０１２３４",
        "Client ¹²³⁴⁵⁶⁷⁸⁹",
        "Client 12٣٤٥٦789",
    ],
)
def test_nfkc_scanner_rejects_standalone_truth_sequences(name: str) -> None:
    with pytest.raises(QualificationError, match="truth leak"):
        qualify([crm_row("crm-1", name=name)], [])


@pytest.mark.parametrize(
    "name",
    [
        "Client 12345678",
        "Client 1234567890",
        "Client 1234567890123",
        "Client 123456789012345",
    ],
)
def test_nfkc_scanner_does_not_match_substrings_of_other_lengths(name: str) -> None:
    assert qualify([crm_row("crm-1", name=name)], [])["counts"]["UNRESOLVED"] == 1


def test_exact_join_rejects_orphan_and_duplicate_source_ids() -> None:
    with pytest.raises(QualificationError, match="orphan"):
        qualify([crm_row("crm-1")], [mapping_row("crm-2")])
    with pytest.raises(QualificationError, match="unique"):
        qualify([crm_row("crm-1"), crm_row("crm-1")], [])


def test_authority_must_be_explicitly_synthetic_and_allowlisted() -> None:
    unknown = mapping_row("crm-1")
    unknown["authority_system_id"] = "TEST_UNKNOWN"
    with pytest.raises(QualificationError, match="allowlist"):
        qualify([crm_row("crm-1")], [unknown])

    bad_catalog = catalog()
    bad_catalog["synthetic_test_authorities"][0]["test_only"] = False
    with pytest.raises(QualificationError, match="test_only"):
        fixture_manifest = manifest()
        qualify_fixture_rows(
            manifest=fixture_manifest,
            collection_manifest_sha256=manifest_sha(fixture_manifest),
            source_file_sha256=SOURCE_SHA,
            crm_rows=[crm_row("crm-1")],
            mapping_rows=[mapping_row("crm-1", siret=None, siren=None)],
            authority_catalog=bad_catalog,
            synthetic_fixtures_only=True,
        )
    with pytest.raises(QualificationError, match="synthetic fixtures only"):
        fixture_manifest = manifest()
        qualify_fixture_rows(
            manifest=fixture_manifest,
            collection_manifest_sha256=manifest_sha(fixture_manifest),
            source_file_sha256=SOURCE_SHA,
            crm_rows=[crm_row("crm-1")],
            mapping_rows=[mapping_row("crm-1", siret=None, siren=None)],
            authority_catalog=catalog(),
            synthetic_fixtures_only=False,
        )


def test_schema_boolean_and_siret_siren_constraints_are_strict() -> None:
    bad = mapping_row("crm-1")
    bad["matching_pipeline_used"] = "false"
    with pytest.raises(QualificationError, match="boolean false"):
        qualify([crm_row("crm-1")], [bad])

    inconsistent = mapping_row("crm-1", siren="999999999")
    with pytest.raises(QualificationError, match="inconsistent"):
        qualify([crm_row("crm-1")], [inconsistent])

    unattested = crm_row("crm-1")
    unattested["source_record_id_equivalence_attested"] = False
    with pytest.raises(QualificationError, match="must be true"):
        qualify([unattested], [])


def test_temporal_constraints_and_reference_date_eligibility() -> None:
    invalid_manifest = manifest(mapping_count=1)
    invalid_manifest["created_at_utc"] = "2026-07-30T00:00:00Z"
    with pytest.raises(QualificationError, match="period_start"):
        qualify_fixture_rows(
            manifest=invalid_manifest,
            collection_manifest_sha256=manifest_sha(invalid_manifest),
            source_file_sha256=SOURCE_SHA,
            crm_rows=[crm_row("crm-1")],
            mapping_rows=[],
            authority_catalog=catalog(),
            synthetic_fixtures_only=True,
        )

    after_cutoff = mapping_row("crm-1")
    after_cutoff["evidence_created_at_utc"] = "2026-08-01T00:00:00Z"
    with pytest.raises(QualificationError, match="export_cutoff"):
        qualify([crm_row("crm-1")], [after_cutoff])

    reversed_validity = mapping_row(
        "crm-1", valid_from="2026-08-01", valid_to="2026-01-01"
    )
    with pytest.raises(QualificationError, match="valid_from"):
        qualify([crm_row("crm-1")], [reversed_validity])

    expired = mapping_row("crm-1", valid_to="2026-07-01")
    result = qualify([crm_row("crm-1")], [expired])
    assert result["counts"]["UNRESOLVED"] == 1
    assert result["oracle"][0]["evidence_count"] == 0


def test_exact_ordered_schemas_reject_extra_or_reordered_fields() -> None:
    extra = crm_row("crm-1")
    extra["label"] = "MATCH_EXACT"
    with pytest.raises(QualificationError, match="exact and ordered"):
        qualify([extra], [])

    reordered = mapping_row("crm-1")
    reordered = {"authority_type": reordered["authority_type"], **reordered}
    with pytest.raises(QualificationError, match="exact and ordered"):
        qualify([crm_row("crm-1")], [reordered])


def test_duplicate_evidence_payload_hash_has_consistent_unique_inventory() -> None:
    first = mapping_row("crm-1")
    second = mapping_row("crm-1", record_id="invoice-2")
    second["evidence_payload_sha256"] = first["evidence_payload_sha256"]
    result = qualify([crm_row("crm-1")], [first, second])
    assert result["oracle"][0]["evidence_count"] == 1
    assert len(result["oracle"][0]["evidence_payload_sha256s"]) == 1


def test_writer_rescans_mutated_result_before_any_output(tmp_path: Path) -> None:
    result = qualify([crm_row("crm-1")], [mapping_row("crm-1")])
    result["queries"][0]["crm_name_raw"] = "LEAK 12345678901234"
    output = tmp_path / "must-not-exist"
    with pytest.raises(QualificationError, match="truth leak"):
        write_fixture_outputs(result, output)
    assert not output.exists()


def test_output_writer_refuses_non_temporary_path() -> None:
    with pytest.raises(QualificationError, match="temporary"):
        write_fixture_outputs(
            {"queries": [], "oracle": [], "counts": {"source_rows": 0}},
            Path("/Volumes/CATNAT_DATA/v413-forbidden"),
        )
