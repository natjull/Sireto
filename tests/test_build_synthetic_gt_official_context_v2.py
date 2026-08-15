from __future__ import annotations

from scripts import build_synthetic_gt_official_context_v2 as context


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
