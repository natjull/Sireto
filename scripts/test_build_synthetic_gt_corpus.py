from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_synthetic_gt_corpus.py")
SPEC = importlib.util.spec_from_file_location("synthetic_builder", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalization_and_decimal_guard_are_deterministic() -> None:
    assert MODULE.normalize_text("École-de l’Île") == "ECOLE DE L ILE"
    assert MODULE.autonomous_decimal_leak("Client 12345678901234")
    assert MODULE.autonomous_decimal_leak("Rue ١٢٣٤٥٦٧٨٩")
    assert not MODULE.autonomous_decimal_leak("12 rue Alpha")


def test_rng_is_stable_and_family_order_is_pinned() -> None:
    first = MODULE.deterministic_rng(42, "q1", "OCR_LIMITED", 3).random()
    second = MODULE.deterministic_rng(42, "q1", "OCR_LIMITED", 3).random()
    assert first == second
    assert MODULE.FAMILY_ORDER == (
        "LEGAL_FORM",
        "ACRONYM_TOKENIZATION",
        "ACCENT_PUNCTUATION",
        "OCR_LIMITED",
        "TOKEN_ORDER",
        "FIELD_MISSING",
        "ADDRESS_ABBREVIATION",
        "ADDRESS_TOKEN_ORDER",
        "ADDRESS_OCR",
        "COMMUNE_VARIANT",
        "ENSEIGNE_VS_DENOMINATION",
    )


def test_guard_accepts_target_with_local_competitor_when_margin_is_clear() -> None:
    target = {
        "siret": "12345678900011",
        "siren": "123456789",
        "state": "A",
        "postcode": "75001",
        "city": "PARIS",
        "insee": "75101",
        "address": "1 RUE ALPHA 75001 PARIS",
        "address_signature": "1 RUE ALPHA 75001 75101",
        "street_signature": "RUE ALPHA 75001",
        "names": ["MAISON ALPHA"],
        "denomination_usuelle": "MAISON ALPHA",
        "legal_denomination": "MAISON ALPHA",
        "enseigne1": "",
        "enseigne2": "",
        "enseigne3": "",
        "legal_usual_1": "",
        "legal_usual_2": "",
        "legal_usual_3": "",
        "sigle": "",
    }
    other = dict(target)
    other.update(
        {
            "siret": "98765432100019",
            "siren": "987654321",
            "names": ["ATELIER BETA"],
            "address": "9 RUE BETA 75001 PARIS",
            "address_signature": "9 RUE BETA 75001 75101",
            "street_signature": "RUE BETA 75001",
        }
    )
    result = MODULE.deterministic_guard(
        {"name": "Maison Alpha", "address": "1 rue Alpha", "postcode": "75001", "city": "Paris", "insee": "75101"},
        target,
        {target["siret"]: target, other["siret"]: other},
    )
    assert result["status"] == "PASS"
    assert result["margin"] > 0


def test_guard_rejects_decimal_identity_leak() -> None:
    target = {"siret": "12345678900011", "siren": "123456789", "state": "A", "postcode": "75001", "city": "PARIS", "insee": "75101", "address": "1 RUE ALPHA", "address_signature": "1 RUE ALPHA 75001 75101", "street_signature": "RUE ALPHA 75001", "names": ["MAISON ALPHA"]}
    result = MODULE.deterministic_guard(
        {"name": "Maison Alpha 123456789", "address": "1 rue Alpha", "postcode": "75001", "city": "Paris", "insee": "75101"},
        target,
        {target["siret"]: target},
    )
    assert result["status"] == "REJECT"
    assert result["reason"] == "AUTONOMOUS_DECIMAL_LEAK"
