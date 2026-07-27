from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.derive_v44_evidence_facts import derive_case_facts, derive_facts


def _result(siret: str, name: str, *, insee: str = "75101") -> dict:
    return {
        "siren": siret[:9],
        "nom_complet": name,
        "nom_raison_sociale": name,
        "matching_etablissements": [
            {
                "siret": siret,
                "etat_administratif": "A",
                "commune": insee,
                "code_postal": "75001",
                "nom_commercial": None,
                "liste_enseignes": None,
            }
        ],
    }


def _evidence_row(case_id, kind, payload, params=None):
    return {
        "audit_case_id": case_id,
        "service_id": "service-1",
        "query_kind": kind,
        "query_params_json": json.dumps(params or {}),
        "http_status": 200,
        "result_count": int(payload.get("total_results") or 0),
        "payload_json": json.dumps(payload),
        "collected_at": "2026-07-27T00:00:00+00:00",
        "source_url": "https://recherche-entreprises.api.gouv.fr/search?q=x",
    }


def _case():
    return pd.Series(
        {
            "audit_case_id": "case-1",
            "service_id": "service-1",
            "decision": "AUTO_MATCH",
            "SITE": "Alpha Paris",
            "CODE_INSEE": "75101",
            "CODE_POSTAL": "75001",
            "top1_siret": "11111111100001",
            "predicted_siret": "11111111100001",
            "input_siret": "22222222200002",
            "input_siret_state": "ACTIVE",
            "ranker_score": 0.999999,
            "top1_address": "1 RUE TEST",
        }
    )


def test_direct_identity_and_name_geo_are_facts_not_adjudication():
    rows = [
        _evidence_row(
            "case-1",
            "TOP1_SIRET",
            {
                "results": [_result("11111111100001", "ALPHA PARIS")],
                "total_results": 1,
            },
        ),
        _evidence_row(
            "case-1",
            "INPUT_SIRET",
            {
                "results": [_result("22222222200002", "BETA")],
                "total_results": 1,
            },
        ),
        _evidence_row(
            "case-1",
            "CRM_NAME_GEO",
            {
                "results": [_result("11111111100001", "ALPHA PARIS")],
                "total_results": 1,
            },
            {"q": "Alpha Paris", "code_commune": "75101"},
        ),
    ]

    facts = derive_case_facts(_case(), pd.DataFrame(rows))

    assert facts["top1_direct_exact_siret_returned"] is True
    assert facts["name_geo_top1_exact_hit"] is True
    assert facts["crm_name_exact_top1_official_name"] is True
    assert facts["official_view_count"] == 3
    assert facts["independent_source_family_count"] == 1
    assert facts["independent_non_sirene_source_count"] == 0
    assert facts["same_source_views_not_counted_as_corroboration"] is True
    assert facts["correctness_conclusion"] == "NOT_DERIVED"
    assert facts["training_eligible"] is False


def test_address_and_model_score_never_create_correctness_fact():
    rows = [
        _evidence_row(
            "case-1",
            "TOP1_SIRET",
            {
                "results": [_result("11111111100001", "UNRELATED")],
                "total_results": 1,
            },
        ),
        _evidence_row(
            "case-1",
            "CRM_NAME_GEO",
            {"results": [], "total_results": 0},
            {"q": "Alpha Paris", "code_commune": "75101"},
        ),
    ]

    facts = derive_case_facts(_case(), pd.DataFrame(rows))

    assert facts["top1_direct_exact_siret_returned"] is True
    assert facts["name_geo_top1_exact_hit"] is False
    assert facts["model_score_used_for_facts"] is False
    assert facts["address_used_as_correctness_proof"] is False
    assert facts["correctness_conclusion"] == "NOT_DERIVED"
    assert facts["requires_human_or_independent_evidence"] is True


def test_equal_input_reuses_single_direct_view_without_fake_corroboration():
    case = _case()
    case["input_siret"] = case["top1_siret"]
    evidence = pd.DataFrame(
        [
            _evidence_row(
                "case-1",
                "TOP1_SIRET",
                {
                    "results": [_result("11111111100001", "ALPHA PARIS")],
                    "total_results": 1,
                },
            ),
            _evidence_row(
                "case-1",
                "CRM_NAME_GEO",
                {"results": [], "total_results": 0},
            ),
        ]
    )

    facts = derive_case_facts(case, evidence)

    assert facts["input_direct_exact_siret_returned"] is True
    assert facts["input_direct_evidence_query_kind"] == "TOP1_SIRET_SHARED"
    assert facts["official_view_count"] == 2
    assert facts["independent_source_family_count"] == 1
    assert facts["correctness_conclusion"] == "NOT_DERIVED"


def test_derive_facts_requires_frozen_auto_coverage(monkeypatch):
    import scripts.derive_v44_evidence_facts as module

    monkeypatch.setattr(module, "EXPECTED_AUTO_COUNT", 1)
    queue = pd.DataFrame([_case()])
    evidence = pd.DataFrame(
        [
            _evidence_row(
                "case-1",
                "TOP1_SIRET",
                {
                    "results": [_result("11111111100001", "ALPHA")],
                    "total_results": 1,
                },
            ),
            _evidence_row(
                "case-1",
                "CRM_NAME_GEO",
                {"results": [], "total_results": 0},
            ),
        ]
    )

    facts = derive_facts(queue, evidence)

    assert len(facts) == 1
    assert facts.iloc[0]["correctness_conclusion"] == "NOT_DERIVED"
    assert not bool(facts.iloc[0]["training_eligible"])

    with pytest.raises(ValueError, match="duplicate"):
        derive_facts(queue, pd.concat([evidence, evidence.iloc[[0]]]))
