from scripts import audit_synthetic_gt_full_sirene_exact as audit


def candidate(siret: str, name: str, address: str, *, insee: str = "93066", postcode: str = "75001"):
    words = address.split()
    return {
        "siret": siret, "siren": siret[:9], "insee": insee, "postcode": postcode,
        "number": words[0], "repetition_index": "", "street_type": words[1],
        "street": " ".join(words[2:]), "establishment_usual": name,
        "enseigne1": "", "enseigne2": "", "enseigne3": "",
        "legal_denomination": name, "legal_sigle": "", "legal_usual1": "",
        "legal_usual2": "", "legal_usual3": "", "legal_last_name": "",
        "legal_usage_name": "", "legal_usual_first": "", "legal_first": "",
    }


def crm(name="FLEURS MAISON", address="12 R DES LILAS"):
    return {"name": name, "address": address, "postcode": "75001", "city": "SAINT DENIS", "insee": "93066"}


def test_exact_address_naturally_returns_target_without_injection():
    target = candidate("12345678900012", "MAISON DES FLEURS", "12 RUE DES LILAS")
    other = candidate("98765432100019", "AUTRE SOCIETE", "8 RUE DES LILAS")
    result = audit.qualify_variant(target["siret"], crm(), [target, other])
    assert result["decision"] == "EXACT_IDENTIFIABLE"
    assert result["exact_witness"] in {"G_N_A", "G_A", "G_N"}
    assert result["target_naturally_returned"] is True


def test_missing_target_is_not_injected():
    other = candidate("98765432100019", "AUTRE SOCIETE", "8 RUE DES LILAS")
    result = audit.qualify_variant("12345678900012", crm(), [other])
    assert result["decision"] == "TARGET_NOT_NATURALLY_MATCHED"
    assert result["target_naturally_returned"] is False


def test_same_name_and_address_is_ambiguous_not_exact():
    first = candidate("12345678900012", "MAISON DES FLEURS", "12 RUE DES LILAS")
    second = candidate("98765432100019", "MAISON DES FLEURS", "12 RUE DES LILAS")
    result = audit.qualify_variant(first["siret"], crm(), [first, second])
    assert result["decision"] == "AMBIGUOUS_OFFICIAL"
    assert result["candidate_counts"]["G_N_A"] == 2


def test_same_siren_same_site_is_reported_operationally_but_not_exact():
    first = candidate("12345678900012", "MAISON DES FLEURS", "12 RUE DES LILAS")
    second = candidate("12345678900020", "MAISON DES FLEURS", "12 RUE DES LILAS")
    result = audit.qualify_variant(first["siret"], crm(), [first, second])
    assert result["decision"] == "AMBIGUOUS_OFFICIAL"
    assert result["operational_equivalence"] is True
    assert result["operational_equivalent_sirets"] == [second["siret"]]


def test_finite_language_allows_whole_token_delete_reorder_and_join_not_letter_anagram():
    assert audit.whole_token_language("FLEURS MAISON", "MAISON DES FLEURS")
    assert audit.whole_token_language("SAO", "S A O REYMOND")
    assert not audit.whole_token_language("NOSA", "S A O REYMOND")
    assert not audit.whole_token_language("FLEUSR", "MAISON DES FLEURS")
