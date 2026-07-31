from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_v413_synthetic_contamination as subject
from scripts import build_v412_consumed_compatibility_registry as historical


KEY = bytes(range(32))


def crm() -> dict:
    return {
        "source_record_id": "NEW-42",
        "source_record_id_equivalence_attested": True,
        "crm_name_raw": "École Élémentaire",
        "crm_address_raw": "1 rue de l’Église",
        "crm_postcode_raw": "75001",
        "crm_city_raw": "Paris",
        "crm_insee_raw": "75101",
    }


def split() -> dict:
    return {"authoritative_sirens": ["123456789"]}


def empty_sets() -> dict[str, set[str]]:
    return {
        "service_id": set(),
        "siret_masked": set(),
        "fuzzy_historical": set(),
        "consumed_sirens": set(),
    }


def projection() -> dict:
    row = crm()
    return {
        "SITE": row["crm_name_raw"],
        "CODE_POSTAL": row["crm_postcode_raw"],
        "CODE_INSEE": row["crm_insee_raw"],
        "SERVICE ID": row["source_record_id"],
        "COMMUNE": row["crm_city_raw"],
        "SIRET": "",
        "SITE_CLI_ADRESSE": row["crm_address_raw"],
        "SITE_CLI_COMMUNE": row["crm_city_raw"],
    }


def test_zero_overlap_reports_all_three_applicable_projections() -> None:
    report = subject.audit_synthetic_contamination(
        crm_rows=[crm()],
        split_rows=[split()],
        keysets=empty_sets(),
        hmac_key=KEY,
        synthetic_only=True,
    )
    assert report["verdict"] == "ZERO_FORBIDDEN_OVERLAP"
    assert report["applicable_keysets"] == [
        "service_id",
        "siret_masked",
        "fuzzy_historical",
    ]
    assert report["excluded_keyset"] == "input_siret_lineage"
    assert report["hit_row_counts"] == {
        "service_id": 0,
        "siret_masked": 0,
        "fuzzy_historical": 0,
    }


@pytest.mark.parametrize("kind", ["service_id", "siret_masked", "fuzzy", "siren"])
def test_every_overlap_family_stops(kind: str) -> None:
    keysets = empty_sets()
    source = projection()
    if kind == "service_id":
        keysets["service_id"].add(
            historical.lineage_hmac(
                KEY,
                historical.SERVICE_DOMAIN,
                historical.canonical_text(source["SERVICE ID"]),
            )
        )
    elif kind == "siret_masked":
        keysets["siret_masked"].add(
            historical.siret_masked_fingerprint(source)
        )
    elif kind == "fuzzy":
        keysets["fuzzy_historical"].add(
            historical.fuzzy_singletons(source)[0][2]
        )
    else:
        keysets["consumed_sirens"].add("123456789")
    with pytest.raises(subject.ContaminationStop, match="forbidden historical overlap"):
        subject.audit_synthetic_contamination(
            crm_rows=[crm()],
            split_rows=[split()],
            keysets=keysets,
            hmac_key=KEY,
            synthetic_only=True,
        )


def test_noncomparable_service_id_stops_instead_of_becoming_zero_hit() -> None:
    source = crm()
    source["source_record_id_equivalence_attested"] = False
    with pytest.raises(subject.ContaminationStop, match="equivalence"):
        subject.audit_synthetic_contamination(
            crm_rows=[source],
            split_rows=[split()],
            keysets=empty_sets(),
            hmac_key=KEY,
            synthetic_only=True,
        )


def test_real_mode_and_fourth_keyset_are_rejected() -> None:
    with pytest.raises(subject.ContaminationStop, match="real registry"):
        subject.audit_synthetic_contamination(
            crm_rows=[crm()],
            split_rows=[split()],
            keysets=empty_sets(),
            hmac_key=KEY,
            synthetic_only=False,
        )
    keysets = empty_sets()
    keysets["input_siret_lineage"] = set()
    with pytest.raises(subject.ContaminationStop, match="exactly"):
        subject.audit_synthetic_contamination(
            crm_rows=[crm()],
            split_rows=[split()],
            keysets=keysets,
            hmac_key=KEY,
            synthetic_only=True,
        )
