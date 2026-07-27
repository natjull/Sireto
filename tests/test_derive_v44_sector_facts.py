from __future__ import annotations

import json

import pandas as pd

from scripts.derive_v44_sector_facts import (
    build_adjudication_priority,
    derive_producer_fact,
)


def _observation(kind: str, identifier: str, siret: str) -> dict:
    return {
        "audit_case_id": "case-1",
        "service_id": "service-1",
        "observed_siret": siret,
        "identifier_kind": kind,
        "identifier": identifier,
        "producer_request_key": "request-1",
        "origin_query_kinds_json": '["TOP1_SIRET"]',
        "origin_occurrence_count": 1,
    }


def _response(producer: str, payload) -> dict:
    return {
        "producer": producer,
        "http_status": 200,
        "result_count": 1,
        "response_excerpt_json": json.dumps(payload),
        "producer_data_date": "2026-07-27",
        "collected_at": "2026-07-27T12:00:00Z",
        "response_url": "https://producer.test",
        "raw_response_path": "raw/response.json",
        "raw_response_sha256": "abc",
    }


def test_uai_fact_extracts_explicit_siret_name_address_and_dates() -> None:
    fact = derive_producer_fact(
        _observation("UAI", "0141396S", "21140409000030"),
        _response(
            "Education nationale",
            {
                "total_count": 1,
                "results": [
                    {
                        "identifiant_de_l_etablissement": "0141396S",
                        "siren_siret": "21140409000030",
                        "nom_etablissement": "Ecole primaire",
                        "adresse_1": "4 avenue de Lavergne",
                        "code_postal": "14810",
                        "nom_commune": "Merville",
                        "date_ouverture": "1970-05-25",
                        "date_maj_ligne": "2026-07-24",
                        "etat": "OUVERT",
                    }
                ],
            },
        ),
    )

    assert fact["producer_identifier_returned"] is True
    assert fact["producer_observed_siret_returned"] is True
    assert json.loads(fact["producer_sirets_json"]) == ["21140409000030"]
    assert "Ecole primaire" in fact["producer_names_json"]
    assert "14810" in fact["producer_addresses_json"]
    assert "2026-07-24" in fact["producer_dates_json"]
    assert fact["correctness_conclusion"] == "NOT_DERIVED"
    assert fact["training_eligible"] is False


def test_bio_fact_records_explicit_siret_difference_without_correctness() -> None:
    fact = derive_producer_fact(
        _observation("BIO", "110415", "11111111100011"),
        _response(
            "Agence Bio",
            [
                {
                    "numeroBio": 110415,
                    "siret": "33812649300026",
                    "raisonSociale": "S.A.S CHAMPILAND",
                    "adressesOperateurs": [
                        {
                            "lieu": "390 avenue Joseph Lacoste",
                            "codePostal": "40990",
                            "ville": "Herm",
                        }
                    ],
                    "dateMaj": "2026-07-21",
                    "certificats": [
                        {
                            "etatCertification": "ENGAGEE",
                            "dateEngagement": "2017-04-03",
                        }
                    ],
                }
            ],
        ),
    )

    assert fact["producer_identifier_returned"] is True
    assert fact["producer_explicit_siret_present"] is True
    assert fact["producer_observed_siret_returned"] is False
    assert "PRODUCER_SIRET_DIFFERS_FROM_OBSERVED" in fact[
        "producer_fact_codes_json"
    ]
    assert fact["correctness_conclusion"] == "NOT_DERIVED"


def test_finess_fact_uses_full_snapshot_index_for_address() -> None:
    fact = derive_producer_fact(
        _observation("FINESS", "110002862", "77555569100275"),
        _response("ANS", [{"numFinessEge": "110002862"}]),
        finess_index={
            "110002862": [
                {
                    "info": {
                        "numFinessEge": "110002862",
                        "siret": "77555569100275",
                        "nomEgeLong": "EANM LES HIRONDELLES",
                        "dateOuverture": "1996-01-02",
                    },
                    "addresses": [
                        {
                            "ligneQuatre": "47 RUE DES POTIERS",
                            "ligneSix": "11400 CASTELNAUDARY",
                        }
                    ],
                    "state": "A",
                    "date_last_update": "2024-02-15",
                }
            ]
        },
    )

    assert fact["producer_identifier_returned"] is True
    assert fact["producer_observed_siret_returned"] is True
    assert "47 RUE DES POTIERS" in fact["producer_addresses_json"]
    assert "2024-02-15" in fact["producer_dates_json"]


def test_priority_distinguishes_top1_input_other_and_conflict() -> None:
    cases = pd.DataFrame(
        [
            {
                "audit_case_id": "conflict",
                "service_id": "s1",
                "top1_siret": "11111111100011",
                "input_siret": "11111111100011",
            },
            {
                "audit_case_id": "input",
                "service_id": "s2",
                "top1_siret": "22222222200022",
                "input_siret": "33333333300033",
            },
            {
                "audit_case_id": "none",
                "service_id": "s3",
                "top1_siret": "44444444400044",
                "input_siret": "",
            },
        ]
    )
    facts = pd.DataFrame(
        [
            {
                "audit_case_id": "conflict",
                "observed_siret": "11111111100011",
                "identifier_kind": "BIO",
                "producer": "Agence Bio",
                "producer_identifier_returned": True,
                "producer_explicit_siret_present": True,
                "producer_observed_siret_returned": False,
            },
            {
                "audit_case_id": "input",
                "observed_siret": "33333333300033",
                "identifier_kind": "UAI",
                "producer": "Education",
                "producer_identifier_returned": True,
                "producer_explicit_siret_present": True,
                "producer_observed_siret_returned": True,
            },
        ]
    )

    priority = build_adjudication_priority(cases, facts).set_index(
        "audit_case_id"
    )

    assert priority.loc["conflict", "adjudication_priority_code"] == (
        "P1_PRODUCER_SIRET_CONFLICT"
    )
    assert priority.loc["input", "adjudication_priority_code"] == (
        "P2_NON_TOP1_SECTOR_OBSERVATION"
    )
    assert priority.loc["input", "sector_input_only_count"] == 1
    assert priority.loc["none", "adjudication_priority_code"] == (
        "P5_NO_SECTOR_OBSERVATION"
    )
    assert priority["sector_evidence_correctness_conclusion"].eq(
        "NOT_DERIVED"
    ).all()
