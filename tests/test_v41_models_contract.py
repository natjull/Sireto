from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.xgb_matcher.v41_acceptor import (
    V41_CONFIDENCE_KIND,
    V41RawLogisticAcceptor,
    fit_v41_raw_logistic_acceptor,
)
from src.xgb_matcher.v41_decision import (
    V41Decision,
    V41ReviewReason,
    decide_v41,
)
from src.xgb_matcher.v41_features import (
    V41_CANDIDATE_FEATURE_NAMES,
    build_v41_candidate_features,
    validate_v41_model_feature_order,
)
from src.xgb_matcher.v41_release import V41ReleaseManifest
from src.xgb_matcher.v41_split import (
    assign_connected_siren_splits,
    validate_connected_siren_split,
)


class _FixedModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        positive = np.full(len(matrix), self.probability)
        return np.column_stack((1.0 - positive, positive))


def _acceptor(probability: float = 0.99) -> V41RawLogisticAcceptor:
    return V41RawLogisticAcceptor(
        model=_FixedModel(probability),
        threshold=0.9,
        feature_order=["scene_signal"],
        model_bundle_id="acceptor-1",
        dataset_manifest_id="acceptor-data",
        retrieval_signature="retrieval-A",
    )


def _candidate(
    siret: str,
    *,
    rank: int,
    state: str = "A",
    direct: bool = False,
    evidence: bool = False,
) -> dict:
    return {
        "candidate_siret": siret,
        "rank": rank,
        "score": 1.0 / rank,
        "candidate_state": state,
        "is_direct_candidate": direct,
        "has_direct_evidence": evidence,
    }


def test_candidate_features_only_expose_relations_state_and_provenance() -> None:
    candidate = {
        "candidate_siret": "12345678900012",
        "etat_admin": "A",
        "from_sparse": True,
        "from_input_siret": True,
    }
    features = build_v41_candidate_features(
        candidate,
        input_siret="123 456 789 00012",
    )
    assert list(features) == V41_CANDIDATE_FEATURE_NAMES
    assert features["input_siret_exact_match"] == 1.0
    assert features["input_siren_exact_match"] == 1.0
    assert features["candidate_is_active"] == 1.0
    assert features["candidate_from_sparse"] == 1.0
    assert set(features.values()) <= {0.0, 1.0}
    invalid_features = build_v41_candidate_features(
        candidate,
        input_siret="abc12345678900012",
    )
    assert invalid_features["input_siret_exact_match"] == 0.0

    validate_v41_model_feature_order(["legacy_score", *V41_CANDIDATE_FEATURE_NAMES])
    with pytest.raises(ValueError, match="Raw identifiers"):
        validate_v41_model_feature_order(
            ["input_siret", *V41_CANDIDATE_FEATURE_NAMES]
        )
    with pytest.raises(ValueError, match="Raw identifiers"):
        validate_v41_model_feature_order(
            ["raw_input_siret", *V41_CANDIDATE_FEATURE_NAMES]
        )


def test_connected_component_split_blocks_transitive_siren_leakage() -> None:
    rows = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "input_siren": "111111111",
                "target_siren": "222222222",
            },
            {
                "query_id": "q2",
                "input_siren": "222222222",
                "target_siren": "333333333",
            },
            {
                "query_id": "q3",
                "input_siren": "444444444",
                "target_siren": "555555555",
            },
            {"query_id": "q4", "input_siren": None, "target_siren": None},
        ]
    )
    first = assign_connected_siren_splits(rows, seed=7)
    second = assign_connected_siren_splits(rows.sample(frac=1.0), seed=7).set_index(
        "query_id"
    )
    validate_connected_siren_split(first)

    q1 = first.set_index("query_id").loc["q1"]
    q2 = first.set_index("query_id").loc["q2"]
    assert q1["siren_component_id"] == q2["siren_component_id"]
    assert q1["split"] == q2["split"]
    assert q1["oof_fold"] == q2["oof_fold"]
    assert first.set_index("query_id")["split"].to_dict() == second["split"].to_dict()


@pytest.mark.parametrize(
    ("candidates", "input_siret", "input_state", "reason"),
    [
        (
            [_candidate("11111111100011", rank=1, state="F")],
            None,
            "UNKNOWN",
            V41ReviewReason.NO_ACTIVE_CANDIDATE,
        ),
        (
            [
                _candidate("11111111100011", rank=1, direct=True),
                _candidate("11111111100029", rank=2, direct=True),
            ],
            "11111111100011",
            "A",
            V41ReviewReason.AMBIGUOUS_DIRECT,
        ),
        (
            [
                _candidate("11111111100011", rank=1, state="F"),
                _candidate("22222222200022", rank=2, state="A"),
            ],
            None,
            "UNKNOWN",
            V41ReviewReason.CLOSED_TOP1,
        ),
        (
            [
                _candidate("22222222200022", rank=1, evidence=True),
                _candidate("11111111100011", rank=2, evidence=True),
            ],
            "11111111100011",
            "A",
            V41ReviewReason.INPUT_CONFLICT,
        ),
    ],
)
def test_runtime_prechecks_are_applied_before_acceptor(
    candidates: list[dict],
    input_siret: str | None,
    input_state: str,
    reason: V41ReviewReason,
) -> None:
    result = decide_v41(
        query_id="q",
        input_siret=input_siret,
        input_siret_state=input_state,
        candidates=candidates,
        scene={"scene_signal": 1.0},
        acceptor=_acceptor(),
    )
    assert result.decision == V41Decision.REVIEW
    assert result.review_reason == reason
    assert result.confidence == 0.0
    assert result.routing_status == "REVIEW"


def test_acceptor_is_raw_logistic_and_never_consumes_test() -> None:
    rows = []
    for split, count in (("fit", 40), ("dev", 20)):
        for index in range(count):
            correct = int(index % 4 != 0)
            rows.append(
                {
                    "split": split,
                    "signal": float(correct),
                    "is_exact_siret_correct": correct,
                    "ranker_prediction_is_out_of_sample": True,
                }
            )
    scenes = pd.DataFrame(rows)
    bundle, report = fit_v41_raw_logistic_acceptor(
        scenes,
        feature_order=["signal"],
        dataset_manifest_id="acceptor-data",
        retrieval_signature="retrieval-A",
        target_precision=0.9,
        min_auto_count=5,
    )
    assert bundle.model_family == "logistic_scaled"
    assert bundle.calibration_method == "raw"
    assert bundle.confidence_kind == V41_CONFIDENCE_KIND
    assert report["final_test_evaluated"] is False

    with pytest.raises(ValueError, match="test is forbidden"):
        fit_v41_raw_logistic_acceptor(
            pd.concat(
                [
                    scenes,
                    pd.DataFrame(
                        [
                            {
                                "split": "test",
                                "signal": 1.0,
                                "is_exact_siret_correct": 1,
                                "ranker_prediction_is_out_of_sample": True,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            ),
            feature_order=["signal"],
            dataset_manifest_id="acceptor-data",
            retrieval_signature="retrieval-A",
            target_precision=0.9,
            min_auto_count=5,
        )


def test_release_manifest_allows_distinct_dataset_ids_but_checks_signatures(
    tmp_path,
) -> None:
    ranker_features = ["legacy_score", *V41_CANDIDATE_FEATURE_NAMES]
    manifest = V41ReleaseManifest.build(
        retrieval_signature="retrieval-A",
        ranker_bundle_id="ranker-1",
        acceptor_bundle_id="acceptor-1",
        ranker_dataset_manifest_id="ranker-data",
        acceptor_dataset_manifest_id="acceptor-data",
        ranker_feature_order=ranker_features,
        acceptor_feature_order=["scene_signal"],
    )
    ranker_metadata = {
        "model_bundle_id": "ranker-1",
        "dataset_manifest_id": "ranker-data",
        "retrieval_signature": "retrieval-A",
        "feature_order": ranker_features,
    }
    acceptor_metadata = {
        "model_bundle_id": "acceptor-1",
        "dataset_manifest_id": "acceptor-data",
        "retrieval_signature": "retrieval-A",
        "feature_order": ["scene_signal"],
        "calibration_method": "raw",
        "confidence_kind": V41_CONFIDENCE_KIND,
    }
    manifest.validate_components(
        ranker_metadata=ranker_metadata,
        acceptor_metadata=acceptor_metadata,
    )
    path = tmp_path / "release.json"
    manifest.save(path)
    assert V41ReleaseManifest.load(path) == manifest
    assert json.loads(path.read_text())["ranker_dataset_manifest_id"] != json.loads(
        path.read_text()
    )["acceptor_dataset_manifest_id"]

    acceptor_metadata["retrieval_signature"] = "retrieval-B"
    with pytest.raises(ValueError, match="acceptor retrieval signature"):
        manifest.validate_components(
            ranker_metadata=ranker_metadata,
            acceptor_metadata=acceptor_metadata,
        )


def test_legacy_routing_status_is_preserved_for_auto_match() -> None:
    result = decide_v41(
        query_id="q",
        input_siret=None,
        input_siret_state="UNKNOWN",
        candidates=[_candidate("11111111100011", rank=1)],
        scene={"scene_signal": 1.0},
        acceptor=_acceptor(),
        shadow_run_id="shadow-1",
    )
    assert result.decision == V41Decision.AUTO_MATCH
    assert result.review_reason is None
    assert result.to_dict()["routing_status"] == "AUTO"
    assert result.to_dict()["confidence_kind"] == V41_CONFIDENCE_KIND

    with pytest.raises(ValueError, match="between 0 and 100"):
        type(result)(**{**result.__dict__, "candidate_count": 101})
