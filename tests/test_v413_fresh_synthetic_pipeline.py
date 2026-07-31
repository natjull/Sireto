from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from scripts.build_v413_fresh_qualification import (
    CRM_COLUMNS,
    MANIFEST_FIELDS,
    MAPPING_COLUMNS,
    qualify_fixture_rows,
    write_fixture_outputs,
)
from scripts.seal_v413_fresh_splits import build_manifests, seal_manifests
from scripts.validate_v413_fresh_artifacts import validate_artifacts


def _manifest(crm_rows: int, mapping_rows: int) -> dict:
    values = {
        "authority_catalog_id": "v413-independent-authorities-1",
        "collection_id": "TEST_INTEGRATED",
        "created_at_utc": "2026-08-02T00:00:00Z",
        "crm_file": "crm_source.csv",
        "crm_format": "CSV",
        "crm_row_count": crm_rows,
        "crm_sha256": "e" * 64,
        "crm_size_bytes": 1,
        "export_cutoff_utc": "2026-08-01T23:00:00Z",
        "export_id": "TEST_INTEGRATED_EXPORT",
        "mapping_file": "authoritative_mapping.csv",
        "mapping_format": "CSV",
        "mapping_row_count": mapping_rows,
        "mapping_sha256": "d" * 64,
        "mapping_size_bytes": 1,
        "matching_based_exclusions": False,
        "period_end_utc": "2026-08-01T00:00:00Z",
        "period_start_utc": "2026-08-01T00:00:00Z",
        "plan_git_commit": "c" * 40,
        "plan_sha256": "b" * 64,
        "population_definition": "all integrated synthetic rows",
        "population_exclusions": [],
        "population_is_exhaustive": True,
        "preregistration_lock_sha256": "a" * 64,
        "producer_id": "TEST_PRODUCER",
        "reference_date": "2026-08-01",
        "schema_version": "sireto-v4.13-collection-manifest-1",
        "source_record_id_semantics": "stable synthetic key",
    }
    return {field: values[field] for field in MANIFEST_FIELDS}


def _crm(source_id: str, group: str | None) -> dict:
    values = [
        source_id,
        group,
        "2026-08-01T01:00:00Z",
        "2026-08-01",
        f"Site {source_id}",
        "1 rue de la Paix",
        "75001",
        "Paris",
        "75101",
        True,
    ]
    return dict(zip(CRM_COLUMNS, values, strict=True))


def _mapping(source_id: str, siret: str, record: str) -> dict:
    values = [
        source_id,
        "CONTRACT_OR_BILLING_SIRET",
        "TEST_ISSUER",
        "TEST_BILLING",
        record,
        siret,
        siret[:9],
        "2026-01-01",
        None,
        "2026-07-01T00:00:00Z",
        hashlib.sha256(record.encode()).hexdigest(),
        False,
    ]
    return dict(zip(MAPPING_COLUMNS, values, strict=True))


def test_qualification_validation_and_split_sealing_connect_end_to_end(
    tmp_path,
) -> None:
    crm = [_crm("row-a", "group-1"), _crm("row-b", "group-1"), _crm("row-c", None)]
    mapping = [
        _mapping("row-a", "12345678900011", "proof-a"),
        _mapping("row-b", "98765432100019", "proof-b1"),
        _mapping("row-b", "98765432100027", "proof-b2"),
    ]
    manifest = _manifest(len(crm), len(mapping))
    manifest_sha = hashlib.sha256(
        (
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode()
    ).hexdigest()
    result = qualify_fixture_rows(
        manifest=manifest,
        collection_manifest_sha256=manifest_sha,
        source_file_sha256="e" * 64,
        crm_rows=crm,
        mapping_rows=mapping,
        authority_catalog={
            "real_collection_open_authorized": False,
            "synthetic_test_authorities": [
                {
                    "authority_type": "CONTRACT_OR_BILLING_SIRET",
                    "authority_issuer_id": "TEST_ISSUER",
                    "authority_system_id": "TEST_BILLING",
                    "test_only": True,
                }
            ],
        },
        synthetic_fixtures_only=True,
    )

    assert result["counts"] == {
        "MATCH_EXACT": 1,
        "AMBIGUOUS": 1,
        "UNRESOLVED": 1,
        "source_rows": 3,
    }
    validation = validate_artifacts(result["queries"], result["oracle"])
    assert validation["verdict"] == "VALID"
    assert validation["row_count"] == 3

    manifests = build_manifests(result["split_rows"])
    assignments = {
        item["query_id"]: (split, item["component_sha256"])
        for split, split_manifest in manifests.items()
        for item in split_manifest["assignments"]
    }
    first, second = result["queries"][0]["query_id"], result["queries"][1]["query_id"]
    assert assignments[first] == assignments[second]
    assert len(assignments) == 3

    output_paths = write_fixture_outputs(result, tmp_path / "qualification")
    assert output_paths["private_split_rows"].is_file()
    hashes = seal_manifests(result["split_rows"], tmp_path / "splits")
    assert set(hashes) == {"fit", "dev", "test"}
