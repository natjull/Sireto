from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import build_synthetic_gt_official_context_v2 as context
from scripts import run_synthetic_gt_agentic_loop as loop


def target() -> dict:
    return {
        "source_siret": "12345678900012",
        "source_siren": "123456789",
        "source_record_sha256": "a" * 64,
        "official_fields": {
            "siret": "12345678900012",
            "siren": "123456789",
            "state": "F",
            "name": "SAS BOULANGERIE DU PORT",
            "legal_name_options": ["SAS BOULANGERIE DU PORT"],
            "enseigne": "LE FOURNIL DU PORT",
            "street_number": "12",
            "street_type": "AVENUE",
            "street": "DU PORT",
            "postcode": "13002",
            "city": "MARSEILLE",
            "insee": "13202",
        },
    }


def competitor(siret: str, siren: str, address: str = "DU PORT") -> dict:
    return {
        "siret": siret,
        "siren": siren,
        "state": "A",
        "is_headquarters": False,
        "usual_name": "BOULANGERIE TEST",
        "enseigne1": "",
        "enseigne2": "",
        "enseigne3": "",
        "number": "12",
        "repetition_index": "",
        "street_type": "AVENUE",
        "street": address,
        "postcode": "13002",
        "city": "MARSEILLE",
        "insee": "13202",
        "relation_tags": ["SAME_OFFICIAL_ADDRESS"],
    }


def test_same_siren_same_site_is_operational_not_exact() -> None:
    sibling = competitor("12345678900020", "123456789")
    sibling["relation_tags"].append("SAME_SIREN")
    value = context.context_row(target(), [sibling], set(), 32)
    assert value["qualification"]["operational_equivalence"] is True
    assert value["qualification"]["exact_identifiable_at_official_baseline"] is False
    assert value["qualification"]["pre_generation_exact_eligible"] is False
    assert value["qualification"]["operational_equivalent_sirets"] == ["12345678900020"]


def test_protected_collision_is_kept_internal_but_hidden_from_luna() -> None:
    protected = competitor("98765432100019", "987654321")
    value = context.context_row(target(), [protected], {"987654321"}, 32)
    assert value["qualification"]["protected_conflict"] is True
    assert value["qualification"]["pre_generation_exact_eligible"] is False
    assert len(value["internal_context"]) == 1
    assert value["llm_view"]["official_context"] == []
    assert value["llm_view"]["context_summary"]["protected_exact_conflicts"] == 1


def test_different_site_sibling_is_visible_and_exact_eligible() -> None:
    sibling = competitor("12345678900020", "123456789", address="DES FLEURS")
    sibling["relation_tags"] = ["SAME_SIREN"]
    value = context.context_row(target(), [sibling], set(), 32)
    assert value["qualification"]["pre_generation_exact_eligible"] is True
    assert value["llm_view"]["official_context"][0]["site_ref"].startswith("CTX-")
    assert "siret" not in value["llm_view"]["official_context"][0]
    assert "siren" not in value["llm_view"]["official_context"][0]


def test_context_hash_is_deterministic() -> None:
    first = context.context_row(target(), [], set(), 32)
    second = context.context_row(target(), [], set(), 32)
    assert first["context_sha256"] == second["context_sha256"]


def test_query_context_uses_same_lowercase_normalization_aliases_and_physical_names(tmp_path: Path) -> None:
    establishments = pd.DataFrame([{
        "siret": "98765432100019", "siren": "987654321",
        "etatAdministratifEtablissement": "A", "etablissementSiege": True,
        "denominationUsuelleEtablissement": "", "enseigne1Etablissement": "",
        "enseigne2Etablissement": "", "enseigne3Etablissement": "",
        "numeroVoieEtablissement": "12", "indiceRepetitionEtablissement": "",
        "typeVoieEtablissement": "che", "libelleVoieEtablissement": "du port",
        "codePostalEtablissement": "13002", "libelleCommuneEtablissement": "marseille",
        "codeCommuneEtablissement": "13202",
    }])
    units = pd.DataFrame([{
        "siren": "987654321", "denominationUniteLegale": "",
        "denominationUsuelle1UniteLegale": "", "denominationUsuelle2UniteLegale": "",
        "denominationUsuelle3UniteLegale": "", "sigleUniteLegale": "",
        "prenomUsuelUniteLegale": "Jean", "prenom1UniteLegale": "Jean",
        "nomUniteLegale": "Dupont", "nomUsageUniteLegale": "",
    }])
    establishment_path = tmp_path / "establishments.parquet"
    legal_path = tmp_path / "units.parquet"
    establishments.to_parquet(establishment_path, index=False)
    units.to_parquet(legal_path, index=False)
    targets = pd.DataFrame([{
        "target_siret": "12345678900012", "target_siren": "123456789",
        "target_state": "F", "number_key": "12", "index_key": "",
        "street_type_key": loop.normalized_alnum(context.canonical_street_type("CHEMIN")),
        "street_key": "duport", "postcode_key": "13002", "insee_key": "13202",
        "city_key": "marseille", "name_values_json": '["UNRELATED TARGET"]',
    }])
    names = pd.DataFrame([{
        "target_siret": "12345678900012", "insee_key": "13202",
        "name_key": "unrelatedtarget",
    }])
    result = context.query_context(
        establishment_path, legal_path, targets, names, tmp_path / "duckdb"
    )
    assert len(result["12345678900012"]) == 1
    assert result["12345678900012"][0]["relation_tags"] == ["SAME_OFFICIAL_ADDRESS"]
    assert "Jean Dupont" in result["12345678900012"][0]["name_values"]


def test_llm_candidate_keeps_legal_name_merged_after_address_discovery() -> None:
    row = competitor("98765432100019", "987654321")
    row["name_values"] = ["BOULANGERIE TEST", "SCP COSTA OUDIN"]
    value = context.llm_candidate("12345678900012", row)
    assert value["names"] == ["BOULANGERIE TEST", "SCP COSTA OUDIN"]
