import pandas as pd

from scripts.build_v4_fresh_expansion import (
    assign_fresh_roles,
    build_fresh_benchmark,
    exact_siren_overlap_by_role,
    role_for_group,
)


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SITE": "Alpha",
                "CODE_POSTAL": "75001",
                "CODE_INSEE": "75101",
                "SERVICE ID": "known",
                "COMMUNE": "PARIS",
                "SIRET": "11111111100001",
                "SITE_CLI_ADRESSE": "1 rue A",
                "SITE_CLI_COMMUNE": "PARIS",
            },
            {
                "SITE": "Beta",
                "CODE_POSTAL": "69001",
                "CODE_INSEE": "69381",
                "SERVICE ID": "fresh",
                "COMMUNE": "LYON",
                "SIRET": "invalid",
                "SITE_CLI_ADRESSE": "2 rue B",
                "SITE_CLI_COMMUNE": "LYON",
            },
        ]
    )


def test_fresh_pool_excludes_frozen_service_ids() -> None:
    fresh = build_fresh_benchmark(
        _raw(),
        benchmark_service_ids={"known"},
    )

    assert fresh["crm_record_id"].tolist() == ["fresh"]
    assert fresh["query_id"].tolist() == ["fresh:fresh"]
    assert fresh["historical_ground_truth_siret"].isna().all()
    assert fresh["label_kind"].tolist() == ["UNRESOLVED"]


def test_fresh_roles_are_grouped_and_existing_siren_is_forced_to_fit() -> None:
    qualified = pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3"],
            "label_kind": ["MATCH_EXACT", "MATCH_EXACT", "UNRESOLVED"],
            "ground_truth_siren": ["111111111", "222222222", None],
            "historical_ground_truth_siren": [
                "999999999",
                "888888888",
                "222222222",
            ],
        }
    )
    assigned = assign_fresh_roles(
        qualified,
        existing_sirens={"111111111"},
    )

    assert assigned.loc[0, "fresh_role"] == "fit_addition"
    assert assigned.loc[1, "fresh_group_key"] == "SIREN:222222222"
    assert assigned.loc[1, "fresh_role"] == assigned.loc[2, "fresh_role"]


def test_role_hash_is_deterministic_and_exact_sirens_do_not_cross_roles() -> None:
    assert role_for_group("SIREN:123456789") == role_for_group(
        "SIREN:123456789"
    )
    assigned = pd.DataFrame(
        {
            "label_kind": ["MATCH_EXACT", "MATCH_EXACT", "MATCH_EXACT"],
            "ground_truth_siren": ["111111111", "222222222", "333333333"],
            "fresh_role": ["fit_addition", "dev_new", "holdout_sealed"],
        }
    )

    audit = exact_siren_overlap_by_role(assigned)

    assert not any(audit["overlap_counts"].values())
