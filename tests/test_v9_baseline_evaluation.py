import pandas as pd

from scripts.evaluate_v9_baseline import evaluate_baseline


def test_baseline_metrics_are_strict_siret_and_count_open_set_auto_as_error():
    labels = pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "label_kind": ["MATCH_EXACT", "NO_MATCH"],
            "ground_truth_siret": ["12345678900011", None],
            "ground_truth_siren": ["123456789", None],
        }
    )
    topk = pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "candidate_siret": ["12345678999999", "22222222200022"],
            "score": [0.9, 0.8],
            "routing_status": ["AUTO", "AUTO"],
        }
    )
    report = evaluate_baseline(topk, labels)
    assert report["hit_at_1_siren"] == 1.0
    assert report["hit_at_1_siret"] == 0.0
    assert report["auto_exact_siret_precision"] == 0.0
    assert report["auto_error_count"] == 2
