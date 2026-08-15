from __future__ import annotations

import pandas as pd

from scripts import build_synthetic_gt_inspiration_bank as bank


def record() -> dict:
    return {
        "names": ["SAS BOULANGERIE DU PORT"],
        "enseigne": ["LE FOURNIL DU PORT"],
        "number": "12",
        "street_type": "AVENUE",
        "street": "DU PORT",
        "postcode": "13002",
        "city": "MARSEILLE",
        "insee": "13202",
        "state": "A",
    }


def crm(**updates: str) -> pd.Series:
    value = {
        "crm_name": "BOULANGERIE PORT",
        "crm_adresse": "12 AV DU PORT",
        "crm_cp": "13002",
        "crm_commune": "MARSEILLE",
        "crm_insee": "13202",
        "oof_fold": 2,
        "legacy_split": "train",
    }
    value.update(updates)
    return pd.Series(value)


def test_bank_keeps_real_compound_pair_without_identity_fields() -> None:
    value = bank.inspiration_row(
        crm(), record(), "12345678900012", "salt"
    )
    assert value is not None
    assert value["structural_signature"]["changed_fields"] == ["name", "address"]
    assert value["official"]["address"] == "12 AVENUE DU PORT"
    assert value["observed_crm"]["address"] == "12 AV DU PORT"
    encoded = str(value)
    assert "12345678900012" not in encoded
    assert "123456789" not in encoded


def test_bank_rejects_single_field_and_conflicting_geo_pairs() -> None:
    assert bank.inspiration_row(
        crm(crm_adresse="12 AVENUE DU PORT"), record(), "12345678900012", "salt"
    ) is None
    assert bank.inspiration_row(
        crm(crm_cp="75001"), record(), "12345678900012", "salt"
    ) is None


def test_bank_rejects_missing_house_number_anchor() -> None:
    assert bank.inspiration_row(
        crm(crm_adresse="AV DU PORT"), record(), "12345678900012", "salt"
    ) is None


def test_bank_rejects_business_tokens_and_numbers_absent_from_official() -> None:
    assert bank.inspiration_row(
        crm(crm_name="BOULANGERIE DU PORT GROUPE OUEST"),
        record(),
        "12345678900012",
        "salt",
    ) is None
    changed_record = record()
    changed_record["number"] = ""
    assert bank.inspiration_row(
        crm(), changed_record, "12345678900012", "salt"
    ) is None
