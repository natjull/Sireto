from __future__ import annotations

from scripts import build_synthetic_gt_compound_evidence as evidence


def test_name_classifiers_measure_real_delta_classes() -> None:
    assert evidence.classify_name(["DUPONT JEAN"], [], "JEAN DUPONT") == {
        "family": "TOKEN_ORDER",
        "delta_class": "two_token_reverse",
        "official": "DUPONT JEAN",
    }
    assert evidence.classify_name(["SAS ALPHA BETA"], [], "ALPHA BETA")["delta_class"] == "remove:SAS"
    accent = evidence.classify_name(["SOCIÉTÉ L’ETOILE"], [], "SOCIETE LETOILE")
    assert accent["family"] == "ACCENT_PUNCTUATION"
    assert "diacritic_removed" in accent["delta_class"]


def test_official_enseigne_is_distinguished_from_legal_name() -> None:
    value = evidence.classify_name(
        ["ALPHA HOLDING"], ["CHEZ MIMI"], "CHEZ MIMI"
    )
    assert value["family"] == "ENSEIGNE_VS_DENOMINATION"
    assert value["delta_class"] == "official_enseigne"


def test_hyphen_deletion_is_not_mislabeled_as_acronym() -> None:
    value = evidence.classify_name(["CORSE-DU-SUD"], [], "CORSEDUSUD")
    assert value["family"] == "ACCENT_PUNCTUATION"
    assert value["delta_class"] == "punctuation_removed:--"


def test_address_abbreviation_requires_exact_canonical_expansion() -> None:
    assert evidence.classify_address(
        "12 RUE DES LILAS", "12 R DES LILAS"
    )["delta_class"] == "RUE->R"
    assert evidence.classify_address(
        "12 RUE DES LILAS", "12 RUE DES ROSES"
    )["family"] == "UNCLASSIFIED"


def test_geo_classification_keeps_unknown_aliases_unproven() -> None:
    assert evidence.classify_geo(
        "city", "SAINT-ÉTIENNE", "SAINT ETIENNE"
    )["family"] == "ACCENT_PUNCTUATION"
    assert evidence.classify_geo(
        "city", "PARIS", "PARIS CEDEX"
    )["family"] == "UNCLASSIFIED"
