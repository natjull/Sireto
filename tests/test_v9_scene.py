import pandas as pd
import pytest

from src.xgb_matcher.v9_scene import (
    assert_oof_training_scenes,
    build_query_scenes,
    split_dev_roles,
)


def _labels():
    return pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3"],
            "label_kind": ["MATCH_EXACT", "MATCH_EXACT", "NO_MATCH"],
            "ground_truth_siret": [
                "12345678900011",
                "22222222200022",
                None,
            ],
            "ground_truth_siren": ["123456789", "222222222", None],
            "split": ["train", "dev", "test"],
        }
    )


def test_scene_correctness_is_strict_siret_and_keeps_misses():
    predictions = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "candidate_siret": [
                "12345678999999",
                "12345678900011",
                "22222222200022",
            ],
            "score": [0.9, 0.8, 0.7],
            "prediction_origin": ["oof", "oof", "holdout"],
        }
    )
    scenes = build_query_scenes(predictions, _labels()).set_index("query_id")
    assert scenes.at["q1", "is_exact_siret_correct"] == 0
    assert scenes.at["q1", "predicted_siren"] == "123456789"
    assert scenes.at["q3", "retrieval_miss"] == 1.0
    assert scenes.at["q3", "is_exact_siret_correct"] == 0


def test_scene_keeps_top_candidate_evidence_and_removes_duplicate_sirets():
    predictions = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "candidate_siret": [
                "12345678900011",
                "12345678900011",
                "99999999900099",
            ],
            "score": [0.9, 0.8, 0.7],
            "rank": [1, 2, 3],
            "prediction_origin": ["oof", "oof", "oof"],
            "name_jaro_max": [0.95, 0.95, 0.40],
            "addr_jaro": [0.80, 0.80, 0.20],
        }
    )

    scene = build_query_scenes(predictions, _labels()).set_index("query_id").loc["q1"]

    assert scene["candidate_count"] == 2.0
    assert scene["top1_name_jaro_max"] == pytest.approx(0.95)
    assert scene["top2_name_jaro_max"] == pytest.approx(0.40)
    assert scene["delta_name_jaro_max"] == pytest.approx(0.55)
    assert scene["top1_addr_jaro"] == pytest.approx(0.80)


def test_training_scenes_must_be_oof():
    scenes = build_query_scenes(
        pd.DataFrame(
            {
                "query_id": ["q1"],
                "candidate_siret": ["12345678900011"],
                "score": [1.0],
                "prediction_origin": ["in_sample"],
            }
        ),
        _labels(),
    )
    with pytest.raises(ValueError, match="out-of-fold"):
        assert_oof_training_scenes(scenes)
    scenes.loc[scenes["query_id"].eq("q1"), "prediction_origin"] = "oof"
    assert_oof_training_scenes(scenes)


def test_training_scenes_allow_ranker_unseen_ambiguous():
    scenes = pd.DataFrame(
        {
            "split": ["train", "train"],
            "label_kind": ["MATCH_EXACT", "AMBIGUOUS"],
            "prediction_origin": ["oof", "out_of_sample_ambiguous"],
        }
    )
    assert_oof_training_scenes(scenes)


def test_dev_roles_are_deterministic_and_disjoint():
    ids = pd.Series([f"q{i}" for i in range(30)])
    first = split_dev_roles(ids)
    second = split_dev_roles(ids)
    assert first.equals(second)
    assert set(first) == {"calibration", "threshold"}
