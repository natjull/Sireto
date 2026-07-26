import pandas as pd

from scripts.evaluate_v4_ranker_e1 import ranking_summary, verdict


def test_ranking_summary_counts_exact_siret_and_siren() -> None:
    labels = pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "ground_truth_siret": ["11111111100001", "22222222200002"],
            "ground_truth_siren": ["111111111", "222222222"],
        }
    )
    top1 = pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "candidate_siret": ["11111111100001", "22222222299999"],
            "candidate_siren": ["111111111", "222222222"],
        }
    )
    summary = ranking_summary(top1, labels)
    assert summary["siret_successes"] == 1
    assert summary["siren_successes"] == 2
    assert summary["hit_at_1_siret"] == 0.5


def test_v4_ranker_verdicts() -> None:
    assert verdict(0.10, 0.95, 0.98) == "GO_ACCEPTEUR_V4"
    assert verdict(0.10, 0.98, 0.95) == "KEEP_OLD_RANKER"
    assert verdict(0.95, 0.90, 0.95) == "PIVOT_RANKER_V4"
