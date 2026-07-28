from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_v46_aligned_ranker as subject


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q1",
                "candidate_siret": "11111111100001",
                "candidate_siren": "111111111",
                "retrieval_rank": 2,
                "is_ground_truth": 0,
            },
            {
                "query_id": "q1",
                "candidate_siret": "11111111100002",
                "candidate_siren": "111111111",
                "retrieval_rank": 1,
                "is_ground_truth": 1,
            },
            {
                "query_id": "q1",
                "candidate_siret": "11111111100000",
                "candidate_siren": "111111111",
                "retrieval_rank": 1,
                "is_ground_truth": 0,
            },
        ]
    )


def test_rank_scored_rows_uses_frozen_tie_break():
    ranked = subject.rank_scored_rows(
        _candidate_rows(),
        [0.8, 0.8, 0.8],
        origin="test",
        fold=None,
    )

    assert ranked["candidate_siret"].tolist() == [
        "11111111100000",
        "11111111100002",
        "11111111100001",
    ]
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_rank_scored_rows_places_higher_score_first():
    ranked = subject.rank_scored_rows(
        _candidate_rows(),
        [0.9, 0.8, 0.8],
        origin="test",
        fold=2,
    )

    assert ranked.iloc[0]["candidate_siret"] == "11111111100001"
    assert set(ranked["fold"]) == {2}


def test_paired_statistics_counts_end_to_end_and_is_reproducible():
    a = np.ones(subject.EXPECTED_DEV_EXACT_COUNT, dtype=bool)
    b = a.copy()
    a[-10:] = False
    b[-10:-2] = True
    b[-2:] = False
    a[-20:-18] = True
    b[-20:-18] = False

    first = subject.paired_statistics(a, b)
    second = subject.paired_statistics(a, b)

    assert first == second
    assert first["a_wrong_b_correct"] == 8
    assert first["a_correct_b_wrong"] == 2
    assert first["net_corrections"] == 6
    assert first["bootstrap_95_high"] >= first["bootstrap_95_low"]


def test_paired_statistics_requires_exact_frozen_population():
    with pytest.raises(ValueError, match="1,217"):
        subject.paired_statistics([True], [True])


def test_exact_hits_counts_missing_top1_as_failure():
    labels = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "split": "dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100001",
                "ground_truth_siren": "111111111",
            },
            {
                "query_id": "q2",
                "split": "dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "22222222200001",
                "ground_truth_siren": "222222222",
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "candidate_siret": "11111111100001",
                "candidate_siren": "111111111",
                "score": 0.9,
                "rank": 1,
            }
        ]
    )

    result = subject.exact_hits(predictions, labels)

    assert result["siret_hit"].tolist() == [True, False]
    assert result["siren_hit"].tolist() == [True, False]


def test_validate_repeat_predictions_detects_score_and_top1_drift():
    base = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "candidate_siret": "11111111100001",
                "prediction_origin": "v46_b_dev",
                "fold": np.nan,
                "score": 0.8,
                "rank": 1,
            },
            {
                "query_id": "q1",
                "candidate_siret": "11111111100002",
                "prediction_origin": "v46_b_dev",
                "fold": np.nan,
                "score": 0.7,
                "rank": 2,
            },
        ]
    )
    same = base.copy()
    result = subject.validate_repeat_predictions(base, same)
    assert result["top1_identical"] is True
    assert result["scores_within_1e_12"] is True

    changed = base.copy()
    changed.loc[0, "score"] += 1e-6
    result = subject.validate_repeat_predictions(base, changed)
    assert result["scores_within_1e_12"] is False


def test_missing_prediction_sentinel_preserves_zero_candidate_query():
    predictions = subject.rank_scored_rows(
        _candidate_rows(),
        [0.8, 0.7, 0.6],
        origin="v46_b_oof",
        fold=0,
    )
    assignments = pd.DataFrame(
        [
            {"query_id": "q1", "split": "fit", "oof_fold": 0},
            {"query_id": "q2", "split": "fit", "oof_fold": 1},
        ]
    )

    result = subject.append_missing_prediction_sentinels(
        predictions,
        assignments,
        include_splits={"fit"},
        origin_by_split={"fit": "v46_b_oof"},
    )

    sentinel = result[result["query_id"].eq("q2")].iloc[0]
    assert pd.isna(sentinel["candidate_siret"])
    assert sentinel["rank"] == 1
    assert sentinel["score"] == -np.inf
    assert result[result["rank"].eq(1)]["query_id"].nunique() == 2


def test_segment_metrics_enforces_large_and_small_family_gates():
    count = 101
    queries = pd.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(count)],
            "input_siret_state": ["ACTIVE"] * count,
            "source_segment": ["dev"] * count,
        }
    )
    a = pd.DataFrame(
        {"query_id": queries["query_id"], "siret_hit": [True] * count}
    )
    b = pd.DataFrame(
        {
            "query_id": queries["query_id"],
            "siret_hit": [False] * 3 + [True] * (count - 3),
        }
    )

    metrics, large_ok, family_ok = subject.segment_metrics(queries, a, b)

    assert metrics["GLOBAL"]["count"] == count
    assert metrics["GLOBAL"]["net_loss"] == 3
    assert metrics["input_siret_state=ACTIVE"]["net_loss"] == 3
    assert large_ok is False
    assert family_ok is False


def _gate_hits(successes: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "siret_hit": [True] * successes
            + [False] * (subject.EXPECTED_DEV_EXACT_COUNT - successes),
            "siren_hit": [True] * successes
            + [False] * (subject.EXPECTED_DEV_EXACT_COUNT - successes),
        }
    )


def test_gate_is_mechanical():
    a = _gate_hits(1200)
    b = _gate_hits(1210)
    paired = {
        "net_corrections": 10,
        "bootstrap_95_low": 0.001,
        "mcnemar_exact_two_sided_p": 0.01,
    }
    deterministic = {
        "top1_identical": True,
        "scores_within_1e_12": True,
    }
    latency_a = {"p95": 1.0}
    latency_b = {"p95": 1.1}

    checks, verdict = subject.evaluate_gates(
        a=a,
        b=b,
        paired=paired,
        segment_large_ok=True,
        segment_family_ok=True,
        deterministic=deterministic,
        latency_a=latency_a,
        latency_b=latency_b,
        total_seconds=100,
        integrity_ok=True,
    )

    assert all(checks.values())
    assert verdict == "GO_ALIGN_RANKER"

    paired["mcnemar_exact_two_sided_p"] = 0.1
    checks, verdict = subject.evaluate_gates(
        a=a,
        b=b,
        paired=paired,
        segment_large_ok=True,
        segment_family_ok=True,
        deterministic=deterministic,
        latency_a=latency_a,
        latency_b=latency_b,
        total_seconds=100,
        integrity_ok=True,
    )
    assert checks["mcnemar_p_lt_0_05"] is False
    assert verdict == "KEEP_RANKER_A"


def test_writable_output_must_be_external():
    with pytest.raises(ValueError, match="must be located"):
        subject._external_path(Path("/tmp/v46-evaluation"), name="output_root")


def test_primary_and_replica_paths_must_be_distinct(tmp_path):
    with pytest.raises(ValueError, match="must be distinct"):
        subject.load_aligned_dataset(tmp_path, tmp_path)
