from __future__ import annotations

import copy
import hashlib

import pytest

from scripts import run_synthetic_gt_agentic_loop as loop


BASELINE_NAME = "BOULANGERIE MARTIN SAS"
OFFICIAL_ALIAS = "L'ÉPI D'OR"


def official_alias_fragment() -> dict:
    return {
        "inspiration_ref": "a" * 64,
        "field": "name",
        "relation": loop.OFFICIAL_NAME_ALIAS_RELATION,
        "evidence_source_type": loop.OFFICIAL_NAME_ALIAS_EVIDENCE_SOURCE,
        "source_fold": -1,
        "official_value": BASELINE_NAME,
        "observed_crm_value": OFFICIAL_ALIAS,
        "operation_parameters": {
            "official_alias_value": OFFICIAL_ALIAS,
            "official_alias_normalized": "l'épi d'or",
            "official_alias_sha256": hashlib.sha256(
                OFFICIAL_ALIAS.encode("utf-8")
            ).hexdigest(),
        },
    }


def seed() -> dict:
    address_fragment = {
        "inspiration_ref": "b" * 64,
        "field": "address",
        "relation": "ADDRESS_ABBREVIATE",
        "source_fold": 2,
        "official_value": "12 RUE DES LILAS",
        "observed_crm_value": "12 R DES LILAS",
        "operation_parameters": {
            "pairs": [{"source": "RUE", "target": "R"}],
        },
    }
    return {
        "seed_id": "official-alias-seed",
        "target_siret": "12345678900012",
        "target_siren": "123456789",
        "source_kind": "SIRENE_ONLY_TRAIN",
        "oof_fold": -1,
        "legacy_split": "train_synthetic",
        "seed_card": {
            "generation_mode": "OBSERVED_COMPOSITE_ANALOGY_V2",
            "name_options": [BASELINE_NAME, OFFICIAL_ALIAS],
            "enseigne_options": [],
            "address": "12 RUE DES LILAS",
            "postcode": "75001",
            "city": "PARIS",
            "insee": "75056",
            "street_number": "12",
            "street_type": "RUE",
            "composite_contracts": [{
                "variant_id": "v1",
                "requested_family": loop.COMPOSITE_FAMILY,
                "target_fields": ["name", "address"],
                "field_relations": {
                    "name": loop.OFFICIAL_NAME_ALIAS_RELATION,
                    "address": "ADDRESS_ABBREVIATE",
                },
                "field_inspirations": {
                    "name": official_alias_fragment(),
                    "address": address_fragment,
                },
            }],
            "internal_context": [],
            "qualification": {
                "pre_generation_exact_eligible": True,
                "siblings_complete": True,
                "same_address_complete": True,
                "same_name_geography_complete": True,
            },
        },
        "observed_train_profile": {
            "rows": 50,
            "supported_families": [loop.COMPOSITE_FAMILY],
            "source_sha256": "c" * 64,
        },
        "risk_flags": [],
    }


def crm(name: str = OFFICIAL_ALIAS) -> dict[str, str]:
    return {
        "name": name,
        "address": "12 R DES LILAS",
        "postcode": "75001",
        "city": "PARIS",
        "insee": "75056",
    }


def test_official_alias_initialization_accepts_typed_non_train_evidence() -> None:
    validated = loop.validate_seed(seed())
    fragment = validated["seed_card"]["composite_contracts"][0][
        "field_inspirations"
    ]["name"]
    assert fragment["source_fold"] == -1
    assert fragment["evidence_source_type"] == "SIRENE_OFFICIAL_NAME_OPTION"


def test_official_alias_allows_only_the_byte_exact_bound_sirene_value() -> None:
    card = seed()["seed_card"]
    contract = card["composite_contracts"][0]
    baseline = loop.official_baseline(card)

    assert loop.composite_change_errors(baseline, crm(), contract, card) == []

    errors = loop.composite_change_errors(
        baseline, crm("l'épi d'or"), contract, card,
    )
    assert any("NAME_RELATION_MISMATCH:OFFICIAL_NAME_ALIAS" in value for value in errors)
    assert "NAME_OPERATOR_PARAMETERS_MISMATCH" in errors

    errors = loop.composite_change_errors(
        baseline, crm(OFFICIAL_ALIAS + " PARIS"), contract, card,
    )
    assert any("NAME_RELATION_MISMATCH:OFFICIAL_NAME_ALIAS" in value for value in errors)


@pytest.mark.parametrize(
    ("source", "target", "street_type", "expected"),
    [
        (
            "2 RTE DE ROUTE DE LA LEUQUEUE", "2 RTE DE RTE DE LA LEUQUEUE", "RTE",
            "STREET_TYPE_COMPONENT_LOCK",
        ),
        (
            "636 RTE CHE DES GAYS", "636 RTE CHEMIN DES GAYS", "RTE",
            "STREET_TYPE_COMPONENT_LOCK",
        ),
        ("12 ROUTE DES LILAS", "12 RTE DES LILAS", "ROUTE", None),
        ("34 GRANDE RUE DES STUARTS", "34 GRANDE R DES STUARTS", "GRANDE RUE", None),
    ],
)
def test_street_alias_is_bound_to_unique_primary_type(
    source: str, target: str, street_type: str, expected: str | None,
) -> None:
    errors = loop.street_alias_structure_errors(
        source, target, {
            "street_number": source.split()[0], "street_type": street_type,
            "official_context": {"target": {"address": {
                "number": source.split()[0], "repetition_index": "",
                "street_type": street_type,
            }}},
        },
    )
    if expected is None:
        assert errors == []
    else:
        assert expected in errors


def test_final_surface_quality_rejects_double_space_broken_elision_and_acronym() -> None:
    contract = {
        "field_relations": {
            "name": "PUNCTUATION_REMOVED", "address": "ADDRESS_TOKEN_SUBSET",
        },
        "field_inspirations": {"address": {"operation_parameters": {
            "source_token_count": 5, "retained_positions": [0, 1, 2, 4],
        }}},
    }
    baseline = {
        "name": "A.B.C SERVICES", "address": "1 RUE DE L'ÉTOILE",
        "postcode": "75001", "city": "SAINT-DENIS", "insee": "75056",
    }
    crm_value = {
        "name": "AB.C  SERVICES", "address": "1 RUE DE ÉTOILE",
        "postcode": "75001", "city": "SAINT DENIS", "insee": "75056",
    }
    errors = loop.final_surface_quality_errors(
        baseline, crm_value, contract, {"street_number": "1", "street_type": "RUE"},
    )
    assert "NAME_DOUBLE_SPACE" in errors
    assert "NAME_ASYMMETRIC_ACRONYM" in errors
    assert "BROKEN_DE_L_ELISION" in errors


def test_official_alias_proof_exposes_authoritative_byte_match_to_critic() -> None:
    card = seed()["seed_card"]
    proof = loop.deterministic_variant_proof(
        card, "v1", crm(), {"passed": True, "errors": []},
    )
    name = proof["fields"]["name"]
    assert name["actual_relation"] == loop.OFFICIAL_NAME_ALIAS_RELATION
    assert name["operator_match"] is True
    assert name["authoritative_value_byte_exact"] is True
    assert name["evidence_source_type"] == loop.OFFICIAL_NAME_ALIAS_EVIDENCE_SOURCE
    assert name["actual_operation_parameters"] == official_alias_fragment()[
        "operation_parameters"
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda f: f.update(source_fold=2), "invalid official name alias evidence"),
        (
            lambda f: f.update(evidence_source_type="TRAIN_OBSERVED"),
            "invalid official name alias evidence",
        ),
        (
            lambda f: f["operation_parameters"].update(
                official_alias_sha256="0" * 64
            ),
            "not bound to a distinct SIRENE option",
        ),
        (
            lambda f: f.update(observed_crm_value="ENSEIGNE INVENTEE"),
            "not bound to a distinct SIRENE option",
        ),
    ],
)
def test_official_alias_contract_fails_closed_on_unbound_evidence(
    mutation, message: str,
) -> None:
    value = copy.deepcopy(seed())
    fragment = value["seed_card"]["composite_contracts"][0][
        "field_inspirations"
    ]["name"]
    mutation(fragment)
    with pytest.raises(ValueError, match=message):
        loop.validate_seed(value)


def test_source_fold_minus_one_is_forbidden_for_train_observed_relation() -> None:
    value = copy.deepcopy(seed())
    fragment = value["seed_card"]["composite_contracts"][0][
        "field_inspirations"
    ]["address"]
    fragment["source_fold"] = -1
    fragment["evidence_source_type"] = loop.OFFICIAL_NAME_ALIAS_EVIDENCE_SOURCE
    with pytest.raises(ValueError, match="invalid train-observed inspiration source"):
        loop.validate_seed(value)


def test_official_alias_cannot_use_legacy_train_inspiration_envelope() -> None:
    value = copy.deepcopy(seed())
    contract = value["seed_card"]["composite_contracts"][0]
    contract.pop("field_inspirations")
    contract["inspiration_ref"] = "d" * 64
    contract["inspiration"] = {
        "inspiration_ref": "d" * 64,
        "source_fold": 2,
        "structural_signature": {
            "changed_fields": ["name", "address"],
            "missing_fields": [],
        },
        "official": {
            "name": BASELINE_NAME,
            "address": "12 RUE DES LILAS",
        },
        "observed_crm": {
            "name": OFFICIAL_ALIAS,
            "address": "12 R DES LILAS",
        },
    }
    with pytest.raises(ValueError, match="requires typed field evidence"):
        loop.validate_seed(value)


def test_same_relation_pair_is_allowed_only_for_distinct_exact_operators() -> None:
    value = copy.deepcopy(seed())
    card = value["seed_card"]
    card["address"] = "12 RUE DES LILAS"
    card["city"] = "SAINT-PIERRE-SUR-MER"
    contracts = []
    for index, observed_city in enumerate(
        ("SAINT PIERRE-SUR-MER", "SAINT-PIERRE SUR-MER"), start=1
    ):
        city_fragment = {
            "inspiration_ref": str(index + 1) * 64,
            "field": "city",
            "relation": "PUNCTUATION_REMOVED",
            "source_fold": 2,
            "official_value": card["city"],
            "observed_crm_value": observed_city,
            "operation_parameters": loop.composite_operation_parameters(
                "city", "PUNCTUATION_REMOVED", card["city"], observed_city,
            ),
        }
        contracts.append({
            "variant_id": f"v{index}",
            "requested_family": loop.COMPOSITE_FAMILY,
            "target_fields": ["name", "city"],
            "field_relations": {
                "name": loop.OFFICIAL_NAME_ALIAS_RELATION,
                "city": "PUNCTUATION_REMOVED",
            },
            "field_inspirations": {
                "name": official_alias_fragment(),
                "city": city_fragment,
            },
        })
    card["composite_contracts"] = contracts

    loop.validate_seed(value)

    duplicate = copy.deepcopy(value)
    duplicate_city = duplicate["seed_card"]["composite_contracts"][1][
        "field_inspirations"
    ]["city"]
    first_city = duplicate["seed_card"]["composite_contracts"][0][
        "field_inspirations"
    ]["city"]
    duplicate_city["observed_crm_value"] = first_city["observed_crm_value"]
    duplicate_city["operation_parameters"] = first_city["operation_parameters"]
    with pytest.raises(ValueError, match="operation signatures must differ"):
        loop.validate_seed(duplicate)
