from scripts import enrich_sirene_agentic_candidates as enrich


def test_enrich_row_uses_official_legal_name_and_requires_full_address():
    row = {
        "source_siret": "12345678900012",
        "source_siren": "123456789",
        "official_fields": {
            "name": None, "enseigne": None, "street_number": "12",
            "street_type": "RUE", "street": "DES LILAS", "postcode": "75001",
            "city": "PARIS", "insee": "75056",
        },
    }
    unit = {"denominationUniteLegale": "SOCIETE DES FLEURS"}
    value = enrich.enrich_row(row, unit)
    assert value["official_fields"]["name"] == "SOCIETE DES FLEURS"
    assert value["selection_eligibility"]["legal_unit_enriched"] is True
    row["official_fields"]["street"] = "[ND]"
    assert enrich.enrich_row(row, unit) is None
