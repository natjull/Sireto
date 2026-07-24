import pandas as pd
import pytest

from scripts.build_benchmark_v2_qualification import (
    apply_policy,
    evaluate_retrieval,
    qualify_site_label,
)


def _benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "1",
                "split": "dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100001",
                "ground_truth_siren": "111111111",
                "crm_name": "Alpha",
            },
            {
                "query_id": "2",
                "split": "dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "22222222200002",
                "ground_truth_siren": "222222222",
                "crm_name": "Beta",
            },
            {
                "query_id": "3",
                "split": "dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "33333333300003",
                "ground_truth_siren": "333333333",
                "crm_name": "Gamma",
            },
        ]
    )


def _audit() -> pd.DataFrame:
    rows = [
        (
            "1",
            "NO_EXACT_SIBLING",
            0,
            0,
        ),
        (
            "2",
            "ACTIVE_GT_HAS_ACTIVE_EXACT_SIBLING",
            1,
            1,
        ),
        (
            "3",
            "CLOSED_GT_UNIQUE_ACTIVE_EXACT_SIBLING",
            1,
            1,
        ),
    ]
    return pd.DataFrame(
        [
            {
                "query_id": query_id,
                "split": "dev",
                "ground_truth_siret": f"{int(query_id) * 111111111:09d}0000{query_id}",
                "ground_truth_siren": f"{int(query_id) * 111111111:09d}",
                "site_label_class": site_class,
                "exact_sibling_count": exact_count,
                "active_exact_sibling_count": active_count,
                "exact_sibling_sirets_json": "[]",
                "active_exact_sibling_sirets_json": "[]",
            }
            for query_id, site_class, exact_count, active_count in rows
        ]
    )


def test_policy_separates_ambiguity_and_stale_reference() -> None:
    assert qualify_site_label("NO_EXACT_SIBLING")[0] == "MATCH_EXACT"
    assert (
        qualify_site_label("ACTIVE_GT_HAS_ACTIVE_EXACT_SIBLING")[0]
        == "AMBIGUOUS"
    )
    assert (
        qualify_site_label("CLOSED_GT_UNIQUE_ACTIVE_EXACT_SIBLING")[0]
        == "UNRESOLVED"
    )


def test_policy_clears_open_labels_but_preserves_historical_provenance() -> None:
    qualified = apply_policy(_benchmark(), _audit()).set_index("query_id")

    assert qualified.at["1", "ground_truth_siret"] == "11111111100001"
    assert qualified.at["1", "exact_metric_eligible"]
    assert pd.isna(qualified.at["2", "ground_truth_siret"])
    assert qualified.at["2", "historical_ground_truth_siret"] == "22222222200002"
    assert qualified.at["2", "label_kind"] == "AMBIGUOUS"
    assert qualified.at["3", "label_kind"] == "UNRESOLVED"
    assert not qualified["qualification_is_human_validated"].any()


def test_policy_rejects_ground_truth_mismatch() -> None:
    audit = _audit()
    audit.loc[audit["query_id"].eq("1"), "ground_truth_siret"] = "99999999900009"
    with pytest.raises(ValueError, match="SIRET labels differ"):
        apply_policy(_benchmark(), audit)


def test_retrieval_metrics_keep_historical_and_exact_denominators() -> None:
    qualified = apply_policy(_benchmark(), _audit())
    retrieval = pd.DataFrame(
        {
            "query_id": ["1", "2", "3"],
            "hit_at_100": [True, True, False],
            "baseline_hit_at_100": [False, True, False],
            "oracle_hit": [True, True, True],
            "candidate_count": [100, 100, 99],
        }
    )

    metrics = evaluate_retrieval(qualified, retrieval)

    assert metrics["historical_all_queries"]["admission_at_100"]["total"] == 3
    assert metrics["historical_all_queries"]["admission_at_100"]["successes"] == 2
    assert metrics["v2_exact_metric"]["admission_at_100"]["total"] == 1
    assert metrics["v2_exact_metric"]["admission_at_100"]["successes"] == 1
    assert metrics["by_v2_label_kind"]["AMBIGUOUS"]["query_count"] == 1


def test_retrieval_metrics_enforce_candidate_ceiling() -> None:
    qualified = apply_policy(_benchmark(), _audit())
    retrieval = pd.DataFrame(
        {
            "query_id": ["1", "2", "3"],
            "hit_at_100": [True, True, False],
            "baseline_hit_at_100": [True, True, False],
            "oracle_hit": [True, True, True],
            "candidate_count": [101, 100, 100],
        }
    )
    with pytest.raises(ValueError, match="100-candidate ceiling"):
        evaluate_retrieval(qualified, retrieval)
