from __future__ import annotations

from src.xgb_matcher.retrieval_selective_qualification import (
    QualificationPolicy,
    qualify_direct_evidence,
)


def policy() -> QualificationPolicy:
    return QualificationPolicy.load(
        "config/retrieval_selective_qualification_v4.json"
    )


def base_row() -> dict:
    return {
        "crm_name": "ACME SERVICES",
        "crm_address": "12 RUE VICTOR HUGO",
        "crm_number": "12",
        "candidate_number": "12",
        "legal_current_names": "ACME SERVICES",
        "trade_current_names": "",
        "site_current_names": "",
        "historical_names": "",
        "supporting_names": "",
        "sirene_names": "",
        "rne_names": "",
        "bodacc_names": "",
        "current_address_text": "12 RUE VICTOR HUGO",
        "historical_address_text": "",
        "supporting_address_text": "",
        "sirene_addresses": "",
        "rne_addresses": "",
        "bodacc_addresses": "",
    }


def test_strong_direct_evidence_is_identifiable() -> None:
    result = qualify_direct_evidence(base_row(), policy())
    assert result["identifiable_exact_v4"] is True
    assert result["qualification_reason_v4"] == "STRONG_NAME"


def test_moderate_joint_requires_compatible_number() -> None:
    row = base_row()
    row.update(
        {
            "crm_name": "ACME LOCAL",
            "legal_current_names": "ACME REGIONALE",
            "crm_address": "12 RUE VICTOR",
            "current_address_text": "12 RUE VICTOR HUGO",
        }
    )
    assert qualify_direct_evidence(row, policy())["identifiable_exact_v4"] is True
    row["candidate_number"] = "99"
    result = qualify_direct_evidence(row, policy())
    assert result["moderate_joint"] is False


def test_no_direct_evidence_stays_review() -> None:
    row = base_row()
    row.update(
        {
            "crm_name": "MARQUE SANS RAPPORT",
            "crm_address": "8 AVENUE AUTRE",
            "legal_current_names": "SOCIETE DIFFERENTE",
            "current_address_text": "99 CHEMIN LOINTAIN",
            "candidate_number": "99",
        }
    )
    result = qualify_direct_evidence(row, policy())
    assert result["identifiable_exact_v4"] is False
    assert result["qualification_reason_v4"] == "INSUFFICIENT_DIRECT_EVIDENCE"
