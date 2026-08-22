from __future__ import annotations

from scripts.profile_retrieval_all_human_v11 import (
    candidate_order_digest,
    deterministic_query_ids,
)


def test_runtime_profile_sample_is_deterministic_and_label_agnostic() -> None:
    rows = [
        {"query_id": f"q{index}", "human_ground_truth_siret": str(index)}
        for index in range(20)
    ]
    first = deterministic_query_ids(rows, sample_size=7, salt="fixed")
    changed_labels = [dict(row, human_ground_truth_siret="changed") for row in rows]
    second = deterministic_query_ids(changed_labels, sample_size=7, salt="fixed")
    assert first == second
    assert len(first) == 7


def test_candidate_order_digest_is_order_sensitive_but_query_order_stable() -> None:
    expected = candidate_order_digest({"q2": ["2"], "q1": ["1", "3"]})
    assert expected == candidate_order_digest({"q1": ["1", "3"], "q2": ["2"]})
    assert expected != candidate_order_digest({"q1": ["3", "1"], "q2": ["2"]})
