import pandas as pd

from scripts.build_benchmark_v3_evidence import (
    apply_evidence_policy,
    classify_direct_evidence,
    evaluate_retrieval,
)


def _features(**overrides: float) -> dict[str, float]:
    values = {
        "name_norm_exact": 0.0,
        "name_jaro_max": 0.0,
        "name_token_overlap_max": 0.0,
        "name_contains_crm_max": 0.0,
        "name_crm_contains_cand_max": 0.0,
        "acronym_match_max": 0.0,
        "postcode_match": 0.0,
        "street_name_jaro": 0.0,
        "street_number_match": 0.0,
    }
    values.update(overrides)
    return values


def test_direct_evidence_classes_are_deterministic() -> None:
    assert classify_direct_evidence(
        _features(name_norm_exact=1.0),
        exact_address_hash=True,
        crm_number_present=True,
        candidate_number_present=True,
    )[0] == "NAME_AND_ADDRESS"
    assert classify_direct_evidence(
        _features(name_jaro_max=0.85, name_token_overlap_max=0.50),
        exact_address_hash=False,
        crm_number_present=True,
        candidate_number_present=True,
    )[0] == "NAME_ONLY"
    assert classify_direct_evidence(
        _features(),
        exact_address_hash=True,
        crm_number_present=True,
        candidate_number_present=True,
    )[0] == "ADDRESS_ONLY"
    assert classify_direct_evidence(
        _features(),
        exact_address_hash=False,
        crm_number_present=True,
        candidate_number_present=True,
    )[0] == "NO_DIRECT_EVIDENCE"


def _v2() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "1",
                "split": "dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100001",
                "ground_truth_siren": "111111111",
                "qualification_reason": "V2_EXACT",
                "exact_metric_eligible": True,
                "historical_ground_truth_siret": "11111111100001",
                "historical_ground_truth_siren": "111111111",
            },
            {
                "query_id": "2",
                "split": "dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "22222222200002",
                "ground_truth_siren": "222222222",
                "qualification_reason": "V2_EXACT",
                "exact_metric_eligible": True,
                "historical_ground_truth_siret": "22222222200002",
                "historical_ground_truth_siren": "222222222",
            },
            {
                "query_id": "3",
                "split": "dev",
                "label_kind": "AMBIGUOUS",
                "ground_truth_siret": None,
                "ground_truth_siren": None,
                "qualification_reason": "V2_AMBIGUOUS",
                "exact_metric_eligible": False,
                "historical_ground_truth_siret": "33333333300003",
                "historical_ground_truth_siren": "333333333",
            },
        ]
    )


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["1", "2", "3"],
            "direct_evidence_class": [
                "NAME_ONLY",
                "NO_DIRECT_EVIDENCE",
                "NAME_AND_ADDRESS",
            ],
            "strong_name_evidence": [True, False, True],
            "strong_address_evidence": [False, False, True],
        }
    )


def test_v3_only_closes_v2_exact_without_direct_evidence() -> None:
    qualified = apply_evidence_policy(_v2(), _evidence()).set_index("query_id")

    assert qualified.at["1", "label_kind"] == "MATCH_EXACT"
    assert qualified.at["2", "label_kind"] == "UNRESOLVED"
    assert pd.isna(qualified.at["2", "ground_truth_siret"])
    assert qualified.at["2", "historical_ground_truth_siret"] == "22222222200002"
    assert qualified.at["3", "label_kind"] == "AMBIGUOUS"
    assert qualified.at["3", "v2_label_kind"] == "AMBIGUOUS"


def test_v3_metrics_publish_all_three_denominators() -> None:
    qualified = apply_evidence_policy(_v2(), _evidence())
    retrieval = pd.DataFrame(
        {
            "query_id": ["1", "2", "3"],
            "hit_at_100": [True, False, True],
            "baseline_hit_at_100": [True, False, False],
            "oracle_hit": [True, False, True],
            "candidate_count": [100, 100, 99],
        }
    )

    metrics = evaluate_retrieval(qualified, retrieval)

    assert metrics["historical_all_queries"]["admission_at_100"]["total"] == 3
    assert metrics["v2_exact_metric"]["admission_at_100"]["total"] == 2
    assert metrics["v3_exact_metric"]["admission_at_100"]["total"] == 1
    assert metrics["v3_exact_metric"]["losses"]["seen_then_pruned"] == 0
