from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_v411_ranker_c_development as subject


class _Scorer:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def predict(self, matrix):
        assert len(matrix) == len(self.scores)
        return self.scores


def _candidate(
    query_id: str,
    siret: str,
    *,
    rank: int,
    truth: int = 0,
    split: str = "fit",
    fold: int = 0,
) -> dict:
    row = {
        "query_id": query_id,
        "candidate_siret": siret,
        "candidate_siren": siret[:9],
        "candidate_state": "A",
        "is_ground_truth": truth,
        "retrieval_rank": rank,
        "retrieval_source": "v4.11-sparse",
        "retrieval_channel_count": 2,
        "retrieval_agreement": 1,
        "enseigne1": None,
        "enseigne2": None,
        "enseigne3": None,
        "denomination_usuelle": None,
        "activity_code": "85.20Z",
        "split": split,
        "oof_fold": fold,
    }
    row.update({name: 0.0 for name in subject.RANKER_C_FEATURE_ORDER})
    row["retrieval_rank_recip"] = 1.0 / rank
    return row


def test_ranker_c_frozen_contract_is_exact():
    assert len(subject.RANKER_C_FEATURE_ORDER) == 45
    assert subject.RANKER_PARAMS == {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@1",
        "n_estimators": 800,
        "learning_rate": 0.035,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    assert subject.PREDICTION_COLUMNS[-2:] == ["oof_fold", "ranker_rank"]


def test_eligible_rows_require_exact_label_and_positive_in_pool():
    candidates = pd.DataFrame(
        [
            _candidate("q1", "11111111100011", rank=1, truth=1),
            _candidate("q1", "22222222200022", rank=2),
            _candidate("q2", "33333333300033", rank=1),
            _candidate("q3", "44444444400044", rank=1),
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100011",
            },
            {
                "query_id": "q2",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "99999999900099",
            },
            {
                "query_id": "q3",
                "label_kind": "AMBIGUOUS",
                "ground_truth_siret": None,
            },
        ]
    )

    output = subject.eligible_ranker_rows(candidates, labels)

    assert set(output["query_id"]) == {"q1"}
    assert len(output) == 2
    assert output["is_ground_truth"].sum() == 1


def test_score_rows_uses_frozen_tie_break_and_schema():
    candidates = pd.DataFrame(
        [
            _candidate("q1", "22222222200022", rank=1),
            _candidate("q1", "11111111100011", rank=2),
            _candidate("q1", "33333333300033", rank=3),
        ]
    )
    # Equal score for the first two: retrieval rank wins before SIRET.
    output = subject.score_rows(
        _Scorer([0.5, 0.5, 0.1]),
        candidates,
        origin="ranker_c_oof",
        fold=3,
    )

    assert list(output.columns) == subject.PREDICTION_COLUMNS
    assert output["candidate_siret"].tolist() == [
        "22222222200022",
        "11111111100011",
        "33333333300033",
    ]
    assert output["ranker_rank"].tolist() == [1, 2, 3]
    assert output["oof_fold"].tolist() == [3, 3, 3]


def test_repetition_check_is_strictly_bit_exact():
    predictions = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "candidate_siret": "11111111100011",
                "candidate_siren": "111111111",
                "retrieval_rank": 1,
                "is_ground_truth": 1,
                "ranker_score": np.float32(0.5),
                "prediction_origin": "ranker_c_oof",
                "oof_fold": 0,
                "ranker_rank": 1,
            }
        ]
    )
    result = subject.assert_bit_exact_repetitions(
        predictions,
        predictions.copy(),
        {"full_fit": "abc"},
        {"full_fit": "abc"},
    )
    assert result["scores_bit_exact"] is True
    changed = predictions.copy()
    changed["ranker_score"] = np.nextafter(
        changed["ranker_score"].to_numpy(dtype=np.float32),
        np.float32(np.inf),
    )
    with pytest.raises(ValueError, match="not bit-exact"):
        subject.assert_bit_exact_repetitions(
            predictions,
            changed,
            {"full_fit": "abc"},
            {"full_fit": "abc"},
        )


def test_masked_ranker_b_projection_never_exposes_identifier_evidence():
    candidates = pd.DataFrame(
        [_candidate("q1", "11111111100011", rank=4)]
    )
    feature_order = [
        *subject.RANKER_C_FEATURE_ORDER[:-1],
        "admission_rank_recip",
        "admission_fusion_score",
        "admission_channel_count",
        "admission_overlay_quota",
        "admission_current_sparse_rank_recip",
        "admission_name_word_rank_recip",
        "admission_name_char_rank_recip",
        "admission_address_word_rank_recip",
        "admission_siren_head_rank_recip",
        "admission_name_exact_rank_recip",
        "admission_address_exact_rank_recip",
        "input_siret_exact_match",
        "input_siren_exact_match",
        "candidate_is_active",
        "candidate_is_closed",
        "candidate_state_unknown",
        "candidate_from_sparse",
        "candidate_from_input_siret",
        "candidate_from_input_siren",
        "candidate_from_closed_alias",
    ]
    matrix = subject.masked_ranker_b_matrix(candidates, feature_order)
    projected = dict(zip(feature_order, matrix[0], strict=True))
    assert projected["admission_rank_recip"] == pytest.approx(0.25)
    assert projected["admission_current_sparse_rank_recip"] == pytest.approx(0.25)
    assert projected["admission_fusion_score"] == pytest.approx(1.0 / 64.0)
    assert projected["admission_channel_count"] == 1
    assert projected["candidate_is_active"] == 1
    assert projected["candidate_from_sparse"] == 1
    assert subject.masked_ranker_b_projection_metadata()[
        "admission_channel_count"
    ] == 1.0
    assert subject.masked_ranker_b_projection_metadata()[
        "admission_fusion_score"
    ] == "1/(60+retrieval_rank)"
    for name in (
        "input_siret_exact_match",
        "input_siren_exact_match",
        "candidate_from_input_siret",
        "candidate_from_input_siren",
        "candidate_from_closed_alias",
    ):
        assert projected[name] == 0


def test_top1_keeps_zero_candidate_queries_and_exact_metric_counts_miss():
    predictions = subject.score_rows(
        _Scorer([0.7, 0.6]),
        pd.DataFrame(
            [
                _candidate(
                    "q1",
                    "11111111100011",
                    rank=1,
                    truth=1,
                ),
                _candidate("q3", "33333333300033", rank=1),
            ]
        ),
        origin="ranker_c_oof",
        fold=0,
    )
    assignments = pd.DataFrame(
        [
            {"query_id": "q1", "split": "fit", "oof_fold": 0},
            {"query_id": "q2", "split": "fit", "oof_fold": 1},
            {"query_id": "q3", "split": "fit", "oof_fold": 0},
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "split": "fit",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100011",
                "ground_truth_siren": "111111111",
            },
            {
                "query_id": "q2",
                "split": "fit",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "22222222200022",
                "ground_truth_siren": "222222222",
            },
            {
                "query_id": "q3",
                "split": "fit",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "44444444400044",
                "ground_truth_siren": "444444444",
            },
        ]
    )
    top1 = subject.build_top1(predictions, assignments)
    evaluated, metrics = subject.exact_metrics(top1, labels, split="fit")
    assert len(top1) == 3
    assert evaluated["siret_hit"].tolist() == [True, False, False]
    assert metrics["siret_hit_at_1"] == pytest.approx(1.0 / 3.0)
    assert metrics["empty_pool_count"] == 1
    assert metrics["truth_absent_from_pool_count"] == 2
    assert metrics["retrieval_miss_count"] == 2


def test_segment_gate_only_applies_at_one_hundred_cases():
    query_ids = [f"q{i}" for i in range(101)]
    c = pd.DataFrame({"query_id": query_ids, "siret_hit": [False] * 101})
    b = pd.DataFrame({"query_id": query_ids, "siret_hit": [True] * 101})
    audit = pd.DataFrame(
        {
            "query_id": query_ids,
            "input_siret_state": ["ACTIVE"] * 99 + ["CLOSED"] * 2,
            "source_segment": ["S"] * 101,
        }
    )
    metrics, passed = subject.segment_comparison(c, b, audit)
    assert metrics["input_siret_state"]["ACTIVE"]["gated"] is False
    assert metrics["source_segment"]["S"]["gated"] is True
    assert passed is False


def test_output_root_must_be_external():
    with pytest.raises(ValueError, match="CATNAT_DATA"):
        subject._external_path(Path("/tmp/v411"), name="output_root")
