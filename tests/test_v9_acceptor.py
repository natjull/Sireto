import numpy as np
import pandas as pd

from src.xgb_matcher.v9_acceptor import (
    V9AcceptorBundle,
    candidate_models,
    train_selective_acceptor,
)
from src.xgb_matcher.v9_scene import V9_SCENE_FEATURE_NAMES


class ScoreModel:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        scores = np.clip(np.asarray(X)[:, 2], 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])


def test_acceptor_never_uses_test_for_selection():
    rows = []
    for split, role, count in [
        ("train", "", 8),
        ("dev", "calibration", 6),
        ("dev", "threshold", 6),
        ("test", "", 6),
    ]:
        for index in range(count):
            correct = int(index % 2 == 0)
            row = {name: 0.0 for name in V9_SCENE_FEATURE_NAMES}
            row.update(
                {
                    "query_id": f"{split}-{role}-{index}",
                    "split": split,
                    "dev_role": role,
                    "is_exact_siret_correct": correct,
                    "score_top1": 0.9 if correct else 0.1,
                    "has_candidate": 1.0,
                }
            )
            rows.append(row)
    bundle, report = train_selective_acceptor(
        pd.DataFrame(rows),
        dataset_manifest_id="dataset",
        target_precision=1.0,
        models={"score": ScoreModel()},
    )
    assert report["test_used_for_selection"] is False
    assert report["final_holdout_evaluated"] is False
    assert report["test"] is None
    assert set(report["models"]["score"]["operating_points"]) == {
        "0.990",
        "0.995",
        "0.998",
    }
    assert report["models"]["score"]["threshold_risk_coverage_curve"]
    assert bundle.model_family == "score"

    _, evaluated = train_selective_acceptor(
        pd.DataFrame(rows),
        dataset_manifest_id="dataset",
        target_precision=1.0,
        models={"score": ScoreModel()},
        evaluate_test=True,
    )
    assert evaluated["final_holdout_evaluated"] is True
    assert evaluated["test"]["count"] == 6


def test_e2b_compares_frozen_score_transformations_and_reloads_bundle(tmp_path):
    rows = []
    for split, role, count in [
        ("train", "", 8),
        ("dev", "calibration", 6),
        ("dev", "threshold", 6),
    ]:
        for index in range(count):
            correct = int(index % 2 == 0)
            row = {name: 0.0 for name in V9_SCENE_FEATURE_NAMES}
            row.update(
                {
                    "query_id": f"{split}-{role}-{index}",
                    "split": split,
                    "dev_role": role,
                    "is_exact_siret_correct": correct,
                    "score_top1": 0.9 if correct else 0.1,
                    "has_candidate": 1.0,
                }
            )
            rows.append(row)
    scenes = pd.DataFrame(rows)
    bundle, report = train_selective_acceptor(
        scenes,
        dataset_manifest_id="dataset",
        target_precision=1.0,
        models={"score": ScoreModel()},
        calibration_methods=("raw", "sigmoid", "isotonic"),
        minimum_gate_coverage=0.25,
    )

    assert set(report["models"]) == {
        "score__raw",
        "score__sigmoid",
        "score__isotonic",
    }
    assert {value["calibration_method"] for value in report["models"].values()} == {
        "raw",
        "sigmoid",
        "isotonic",
    }
    assert report["verdict"] == "PASS_E2B"
    assert report["test"] is None

    output_dir = tmp_path / "acceptor"
    expected = bundle.confidence(scenes)
    bundle.save(output_dir, {"training_report": report})
    restored = V9AcceptorBundle.load(output_dir)
    assert restored.calibration_method == bundle.calibration_method
    np.testing.assert_allclose(restored.confidence(scenes), expected)


def test_logistic_acceptor_standardizes_features():
    logistic = candidate_models()["logistic_scaled"]
    assert list(logistic.named_steps) == ["scaler", "model"]
