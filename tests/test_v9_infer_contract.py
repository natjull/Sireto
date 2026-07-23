import numpy as np

from src.xgb_matcher.contracts import MatchDecision, ReviewReason
from src.xgb_matcher.v9_acceptor import V9AcceptorBundle
from src.xgb_matcher.v9_scene import V9_SCENE_FEATURE_NAMES


class ConstantModel:
    def predict_proba(self, X):
        return np.asarray([[0.1, 0.9] for _ in range(len(X))])


class IdentityCalibrator:
    def predict(self, values):
        return values


def bundle():
    return V9AcceptorBundle(
        model=ConstantModel(),
        calibrator=IdentityCalibrator(),
        threshold=0.8,
        feature_order=V9_SCENE_FEATURE_NAMES,
        model_bundle_id="model",
        dataset_manifest_id="dataset",
        model_family="test",
    )


def test_v9_output_never_auto_matches_without_candidate():
    scene = {name: 0.0 for name in V9_SCENE_FEATURE_NAMES}
    scene["query_id"] = "q1"
    result = bundle().decide(scene)
    assert result.decision == MatchDecision.REVIEW
    assert result.review_reason == ReviewReason.NO_CANDIDATE
    assert result.routing_status == "REVIEW"


def test_v9_output_keeps_legacy_auto_status():
    scene = {name: 0.0 for name in V9_SCENE_FEATURE_NAMES}
    scene.update(
        {
            "query_id": "q2",
            "predicted_siret": "12345678900011",
            "predicted_siren": "123456789",
            "has_candidate": 1.0,
        }
    )
    result = bundle().decide(scene)
    assert result.decision == MatchDecision.AUTO_MATCH
    assert result.routing_status == "AUTO"
