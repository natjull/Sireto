import pandas as pd
import pytest

from scripts.build_benchmark_v4_current_snapshot import (
    apply_current_snapshot_policy,
    build_active_partition_index,
    canonical_address_anchor,
    cross_split_audit,
    find_direct_active_candidates,
    reject_model_outputs,
)


def _candidate(
    siret: str,
    name: str,
    *,
    state: str = "A",
) -> dict:
    return {
        "siret": siret,
        "siren": siret[:9],
        "denomination": name,
        "enseigne1": name,
        "enseigne2": None,
        "enseigne3": None,
        "is_siege": True,
        "numeroVoie": "9",
        "typeVoie": "RUE",
        "libelleVoie": "HENRI BECQUEREL",
        "complementAdresse": None,
        "postcode": "77500",
        "city": "CHELLES",
        "insee": "77108",
        "cj_ul": "5499",
        "etat_admin": state,
        "sigle_ul": None,
        "denomination_ul": name,
        "denomination_usuelle_ul": None,
        "nom_ul": None,
        "prenom_usuel_ul": None,
        "pm_dirigeant_names": "",
    }


def _query(query_id: str = "1") -> dict:
    return {
        "query_id": query_id,
        "split": "dev",
        "crm_name": "VISSELECT SARL",
        "crm_address": "9 Rue Henri Becquerel",
        "crm_city": "Chelles",
        "postcode": "77500",
        "insee": "77108",
    }


def test_v4_finds_unique_active_direct_match_and_ignores_closed() -> None:
    index = build_active_partition_index(
        [
            _candidate("62820158400024", "VISSELECT", state="A"),
            _candidate("52381510800015", "VISSELECT", state="F"),
        ]
    )

    evidence = find_direct_active_candidates(
        _query(),
        index,
        partition_key="insee:77108",
    )

    assert [row["candidate_siret"] for row in evidence] == [
        "62820158400024"
    ]
    assert evidence[0]["candidate_state"] == "A"
    assert evidence[0]["exact_name_anchor"]
    assert evidence[0]["exact_address_anchor"]


def test_v4_keeps_multiple_active_direct_matches_ambiguous() -> None:
    index = build_active_partition_index(
        [
            _candidate("62820158400024", "VISSELECT"),
            _candidate("62820158400032", "VISSELECT"),
        ]
    )
    evidence = find_direct_active_candidates(
        _query(),
        index,
        partition_key="insee:77108",
    )
    assert len(evidence) == 2


def test_v4_address_anchor_neutralizes_broken_apostrophe() -> None:
    assert canonical_address_anchor("1|DE L?INDUSTRIE") == (
        canonical_address_anchor("1|DE L'INDUSTRIE")
    )


def _v3() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **_query("1"),
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "52381510800015",
                "ground_truth_siren": "523815108",
                "historical_ground_truth_siret": "52381510800015",
                "historical_ground_truth_siren": "523815108",
                "qualification_reason": "V3_EXACT",
            },
            {
                **_query("2"),
                "label_kind": "UNRESOLVED",
                "ground_truth_siret": None,
                "ground_truth_siren": None,
                "historical_ground_truth_siret": "22222222200002",
                "historical_ground_truth_siren": "222222222",
                "qualification_reason": "V3_UNRESOLVED",
            },
            {
                **_query("3"),
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "33333333300003",
                "ground_truth_siren": "333333333",
                "historical_ground_truth_siret": "33333333300003",
                "historical_ground_truth_siren": "333333333",
                "qualification_reason": "V3_EXACT",
            },
        ]
    )


def _audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "1",
                "split": "dev",
                "direct_active_candidate_count": 1,
                "direct_active_sirets_json": '["62820158400024"]',
                "selected_active_siret": "62820158400024",
                "selected_active_siren": "628201584",
            },
            {
                "query_id": "2",
                "split": "dev",
                "direct_active_candidate_count": 2,
                "direct_active_sirets_json": (
                    '["22222222200002", "22222222200010"]'
                ),
                "selected_active_siret": None,
                "selected_active_siren": None,
            },
            {
                "query_id": "3",
                "split": "dev",
                "direct_active_candidate_count": 0,
                "direct_active_sirets_json": "[]",
                "selected_active_siret": None,
                "selected_active_siren": None,
            },
        ]
    )


def test_v4_policy_promotes_unique_current_match_only() -> None:
    qualified = apply_current_snapshot_policy(
        _v3(),
        _audit(),
        snapshot_id="snapshot",
    ).set_index("query_id")

    assert qualified.at["1", "label_kind"] == "MATCH_EXACT"
    assert qualified.at["1", "ground_truth_siret"] == "62820158400024"
    assert qualified.at["1", "siret_changed_from_historical"]
    assert qualified.at["1", "siren_changed_from_historical"]
    assert qualified.at["2", "label_kind"] == "AMBIGUOUS"
    assert pd.isna(qualified.at["2", "ground_truth_siret"])
    assert qualified.at["3", "label_kind"] == "UNRESOLVED"
    assert pd.isna(qualified.at["3", "ground_truth_siret"])


def test_v4_refuses_model_outputs() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        reject_model_outputs(
            pd.DataFrame({"query_id": ["1"], "retrieval_rank": [1]}),
            source="fixture",
        )


def test_v4_cross_split_audit_detects_new_siren_leakage() -> None:
    train = pd.DataFrame(
        {
            "label_kind": ["MATCH_EXACT", "UNRESOLVED"],
            "ground_truth_siren": ["111111111", None],
        }
    )
    dev = pd.DataFrame(
        {
            "label_kind": ["MATCH_EXACT", "MATCH_EXACT"],
            "ground_truth_siren": ["111111111", "222222222"],
        }
    )

    audit = cross_split_audit(train, dev)

    assert audit["shared_exact_siren_count"] == 1
    assert audit["shared_exact_sirens"] == ["111111111"]
