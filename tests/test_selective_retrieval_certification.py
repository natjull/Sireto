import pandas as pd

from scripts.certify_selective_retrieval_test import certify


def _inputs(exact_count: int, *, over_budget: bool = False):
    total = 100
    qualified = pd.DataFrame(
        {
            "query_id": [str(index) for index in range(total)],
            "label_kind": [
                "MATCH_EXACT" if index < exact_count else "UNRESOLVED"
                for index in range(total)
            ],
            "v2_label_kind": ["MATCH_EXACT"] * total,
            "ground_truth_state": ["A"] * total,
        }
    )
    admission = pd.DataFrame(
        {
            "query_id": [str(index) for index in range(total)],
            "hit_at_100": [True] * total,
            "baseline_hit_at_100": [True] * total,
            "oracle_hit": [True] * total,
            "candidate_count": [101 if over_budget and index == 0 else 100 for index in range(total)],
            "mega_base_pool": [False] * total,
            "multi_site_siren": [False] * total,
            "location_match_type": ["insee"] * total,
        }
    )
    return qualified, admission


def test_certification_goes_when_global_gates_pass() -> None:
    qualified, admission = _inputs(100)
    result = certify(qualified, admission)
    assert result["decision"] == "GO"
    assert result["gates"]["coverage"]["passed"]
    assert result["gates"]["recall_at_100"]["passed"]


def test_certification_pivots_when_coverage_is_too_low() -> None:
    qualified, admission = _inputs(79)
    result = certify(qualified, admission)
    assert result["decision"] == "PIVOT"
    assert not result["gates"]["coverage"]["passed"]


def test_certification_pivots_on_candidate_ceiling_violation() -> None:
    qualified, admission = _inputs(100, over_budget=True)
    result = certify(qualified, admission)
    assert result["decision"] == "PIVOT"
    assert not result["gates"]["candidate_ceiling"]["passed"]
