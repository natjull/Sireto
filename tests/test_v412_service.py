from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.xgb_matcher.v411_scene import V411_ACCEPTOR_FEATURE_NAMES
from src.xgb_matcher.v412_service import (
    FIXED_THRESHOLD,
    RANKER_C_FEATURE_ORDER,
    V412DownstreamService,
)


FEATURES = list(RANKER_C_FEATURE_ORDER)


class _Ranker:
    def predict(self, matrix):
        return np.asarray(matrix[:, 0], dtype=np.float32)


class _Acceptor:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score

    def predict_proba(self, matrix):
        assert matrix.shape == (1, len(V411_ACCEPTOR_FEATURE_NAMES))
        return np.asarray([[1.0 - self.score, self.score]], dtype=np.float64)


def _scene_builder(_query, candidates, _taxonomy):
    top = candidates.iloc[0] if len(candidates) else None
    return {
        "predicted_siret": None if top is None else str(top["candidate_siret"]),
        "predicted_siren": None if top is None else str(top["candidate_siren"]),
        **{name: 0.0 for name in V411_ACCEPTOR_FEATURE_NAMES},
    }


def _engine(*, score: float = 0.9) -> V412DownstreamService:
    return V412DownstreamService(
        ranker=_Ranker(),
        acceptor=_Acceptor(score),
        taxonomy=object(),
        ranker_feature_order=FEATURES,
        scene_builder=_scene_builder,
    )


def _candidates(count: int = 2) -> pd.DataFrame:
    rows = []
    for index in range(count):
        siret = f"123456789{index:05d}"
        rows.append(
            {
                "query_id": "q1",
                "candidate_siret": siret,
                "candidate_siren": siret[:9],
                "candidate_state": "A",
                "retrieval_rank": index + 1,
                **{
                    feature: float(count - index)
                    if position == 0
                    else 0.0
                    for position, feature in enumerate(FEATURES)
                },
            }
        )
    return pd.DataFrame(rows)


def _evidence(count: int, sole: str | None) -> dict:
    return {
        "query_id": "q1",
        "direct_candidate_count": count,
        "sole_direct_siret": sole,
    }


def test_service_scores_one_query_and_guard_is_only_a_veto() -> None:
    candidates = _candidates()
    top = str(candidates.iloc[0]["candidate_siret"])
    accepted = _engine().infer_one(
        query={"query_id": "q1"},
        candidates=candidates,
        direct_evidence=_evidence(1, top),
    )
    assert accepted.predicted_siret == top
    assert accepted.decision_v411 == "AUTO_MATCH"
    assert accepted.decision_v412 == "AUTO_MATCH"
    assert accepted.review_reason_v412 is None
    assert accepted.scored_candidates["ranker_rank"].tolist() == [1, 2]
    assert accepted.timings.downstream_ns > 0

    vetoed = _engine().infer_one(
        query={"query_id": "q1"},
        candidates=candidates,
        direct_evidence=_evidence(2, None),
    )
    assert vetoed.decision_v411 == "AUTO_MATCH"
    assert vetoed.decision_v412 == "REVIEW"
    assert (
        vetoed.review_reason_v412
        == "MULTIPLE_STRONG_DIRECT_CANDIDATES"
    )


def test_v411_stage_can_run_without_loading_or_applying_evidence() -> None:
    candidates = _candidates()
    engine = _engine()

    trace = engine.rank_and_accept_one(
        query={"query_id": "q1"},
        candidates=candidates,
    )

    assert trace.predicted_siret == str(candidates.iloc[0]["candidate_siret"])
    assert trace.decision_v411 == "AUTO_MATCH"
    assert trace.ranker_ns > 0
    assert trace.scene_acceptor_ns > 0

    guarded = engine.apply_guard_to_trace(
        trace=trace,
        direct_evidence=_evidence(2, None),
    )
    assert guarded.decision_v411 == "AUTO_MATCH"
    assert guarded.decision_v412 == "REVIEW"


def test_low_acceptor_score_cannot_be_upgraded_by_guard() -> None:
    candidates = _candidates()
    top = str(candidates.iloc[0]["candidate_siret"])
    trace = _engine(score=0.2).infer_one(
        query={"query_id": "q1"},
        candidates=candidates,
        direct_evidence=_evidence(1, top),
    )
    assert trace.decision_v411 == "REVIEW"
    assert trace.decision_v412 == "REVIEW"
    assert trace.review_reason_v412 == "LOW_CONFIDENCE"


def test_empty_pool_is_review_without_calling_ranker() -> None:
    candidates = _candidates(1).iloc[0:0].copy()
    trace = _engine().infer_one(
        query={"query_id": "q1"},
        candidates=candidates,
        direct_evidence=_evidence(0, None),
    )
    assert trace.predicted_siret is None
    assert trace.decision_v411 == "REVIEW"
    assert trace.decision_v412 == "REVIEW"
    assert trace.review_reason_v412 == "NO_CANDIDATE"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("candidate_truth", "candidate pool contains"),
        ("query_truth", "query contains"),
        ("evidence_truth", "direct evidence contains"),
        ("candidate_101", "candidate ceiling exceeded"),
        ("rank_gap", "retrieval ranks not contiguous"),
        ("query_mismatch", "candidate query mismatch"),
        ("state", "invalid candidate identity/state"),
        ("nonfinite", "non-finite ranker feature"),
    ],
)
def test_service_fails_closed_on_boundary_mutations(
    mutation: str,
    match: str,
) -> None:
    candidates = _candidates(101 if mutation == "candidate_101" else 2)
    query = {"query_id": "q1"}
    evidence = _evidence(
        1,
        str(candidates.iloc[0]["candidate_siret"]),
    )
    if mutation == "candidate_truth":
        candidates["is_ground_truth"] = 0
    elif mutation == "query_truth":
        query["ground_truth_siret"] = "12345678900000"
    elif mutation == "evidence_truth":
        evidence["label_kind"] = "MATCH_EXACT"
    elif mutation == "rank_gap":
        candidates.loc[candidates.index[-1], "retrieval_rank"] = 3
    elif mutation == "query_mismatch":
        candidates.loc[candidates.index[-1], "query_id"] = "q2"
    elif mutation == "state":
        candidates.loc[candidates.index[-1], "candidate_state"] = "F"
    elif mutation == "nonfinite":
        candidates.loc[candidates.index[-1], FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match=match):
        _engine().infer_one(
            query=query,
            candidates=candidates,
            direct_evidence=evidence,
        )


def test_threshold_and_feature_order_are_frozen() -> None:
    with pytest.raises(ValueError, match="threshold changed"):
        V412DownstreamService(
            ranker=_Ranker(),
            acceptor=_Acceptor(),
            taxonomy=object(),
            ranker_feature_order=FEATURES,
            threshold=FIXED_THRESHOLD + 1e-6,
            scene_builder=_scene_builder,
        )
    with pytest.raises(ValueError, match="feature order changed"):
        V412DownstreamService(
            ranker=_Ranker(),
            acceptor=_Acceptor(),
            taxonomy=object(),
            ranker_feature_order=FEATURES[:-1],
            scene_builder=_scene_builder,
        )
