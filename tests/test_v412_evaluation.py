from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.xgb_matcher import v412_evaluation as subject


def _component(prefix: str, *, threshold: bool) -> str:
    index = 0
    while True:
        value = f"{prefix}-{index}"
        observed = (
            hashlib.sha256(
                f"v411-threshold:{value}".encode()
            ).digest()[0]
            < 128
        )
        if observed is threshold:
            return value
        index += 1


def _split() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("fit-1", "fit-component", "fit", 0),
            ("fit-2", "fit-component-2", "fit", 1),
            ("threshold-1", _component("t", threshold=True), "dev", None),
            ("comparison-1", _component("c", threshold=False), "dev", None),
        ],
        columns=subject.SPLIT_COLUMNS,
    )


def _scenes(split: pd.DataFrame) -> pd.DataFrame:
    populations = subject.assign_populations(split)
    rows = []
    for row in populations.to_dict("records"):
        rows.append(
            {
                "query_id": row["query_id"],
                "split": row["split"],
                "dev_partition": (
                    "" if row["split"] == "fit" else row["population"]
                ),
                "oof_fold": row["oof_fold"],
                "siren_component_id": row["siren_component_id"],
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100011",
                "predicted_siret": "11111111100011",
                "acceptor_target": 1,
                "ranker_prediction_is_out_of_sample": True,
                "prediction_origin": (
                    "ranker_c_oof" if row["split"] == "fit" else "ranker_c_dev"
                ),
                "input_siret_state": "ACTIVE",
                "source_segment": "train",
                "top1_siren_candidate_count": 1.0,
                "role_crm_count": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_populations_are_recomputed_by_siren_component():
    split = _split()
    populations = subject.assign_populations(split)
    assert populations["population"].tolist() == [
        "fit",
        "fit",
        "threshold_dev",
        "comparison_dev",
    ]
    returned = subject.validate_population_parity(
        split, _scenes(split), enforce_canonical=False
    )
    assert set(returned["query_id"]) == set(split["query_id"])


def test_population_parity_rejects_scene_partition_tampering():
    split = _split()
    scenes = _scenes(split)
    scenes.loc[scenes["query_id"].eq("comparison-1"), "dev_partition"] = (
        "threshold_dev"
    )
    with pytest.raises(ValueError, match="dev partition"):
        subject.validate_population_parity(
            split, scenes, enforce_canonical=False
        )


def _ranker(split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in split.to_dict("records"):
        for rank, siret in enumerate(
            ["11111111100011", "22222222200022"], start=1
        ):
            rows.append(
                {
                    "query_id": row["query_id"],
                    "candidate_siret": siret,
                    "candidate_siren": siret[:9],
                    "retrieval_rank": rank,
                    "ranker_score": float(3 - rank),
                    "prediction_origin": (
                        "ranker_c_oof"
                        if row["split"] == "fit"
                        else "ranker_c_dev"
                    ),
                    "oof_fold": (
                        row["oof_fold"] if row["split"] == "fit" else None
                    ),
                    "ranker_rank": rank,
                }
            )
    return pd.DataFrame(rows, columns=subject.RANKER_COLUMNS)


def test_ranker_projection_enforces_top100_oof_and_top1():
    split = _split()
    scenes = _scenes(split)
    pools = subject.validate_ranker_projection(_ranker(split), split, scenes)
    assert pools["fit-1"] == {"11111111100011", "22222222200022"}

    leaked = _ranker(split)
    leaked["is_ground_truth"] = 0
    with pytest.raises(ValueError, match="projection changed"):
        subject.validate_ranker_projection(leaked, split, scenes)


def test_ranker_rejects_more_than_100_candidates():
    split = _split().iloc[[0]].copy()
    scenes = _scenes(split)
    rows = []
    for rank in range(1, 102):
        siren = f"{rank:09d}"
        rows.append(
            {
                "query_id": "fit-1",
                "candidate_siret": f"{siren}00000",
                "candidate_siren": siren,
                "retrieval_rank": min(rank, 100),
                "ranker_score": float(-rank),
                "prediction_origin": "ranker_c_oof",
                "oof_fold": 0,
                "ranker_rank": rank,
            }
        )
    ranker = pd.DataFrame(rows, columns=subject.RANKER_COLUMNS)
    with pytest.raises(ValueError, match="exceeds 100"):
        subject.validate_ranker_projection(ranker, split, scenes)


def test_ranker_rejects_fractional_ranks_and_provenance_drift():
    split = _split()
    scenes = _scenes(split)
    fractional = _ranker(split)
    fractional["ranker_rank"] = fractional["ranker_rank"].astype(float)
    with pytest.raises(ValueError, match="invalid ranker_rank"):
        subject.validate_ranker_projection(fractional, split, scenes)

    changed = _ranker(split)
    scenes.loc[0, "prediction_origin"] = "forged"
    with pytest.raises(ValueError, match="provenance differs"):
        subject.validate_ranker_projection(changed, split, scenes)


class _Model:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, matrix):
        assert matrix.shape[1] == 80
        return np.column_stack([1.0 - self.scores, self.scores])


class _UnstableModel(_Model):
    def predict_proba(self, matrix):
        self.scores = self.scores - 0.01
        return super().predict_proba(matrix)


def _scenes_with_features(split: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    scenes = _scenes(split)
    features = [f"f{index}" for index in range(80)]
    for feature in features:
        scenes[feature] = 0.0
    return scenes, features


def test_v411_scoring_is_frozen_and_target_is_exact_siret():
    scenes, features = _scenes_with_features(_split())
    scored = subject.score_v411(
        _Model([0.9, 0.1, 0.9, 0.9]),
        scenes,
        feature_order=features,
    )
    assert scored["decision_v411"].tolist() == [
        "AUTO_MATCH",
        "REVIEW",
        "AUTO_MATCH",
        "AUTO_MATCH",
    ]
    with pytest.raises(ValueError, match="threshold changed"):
        subject.score_v411(
            _Model([0.9] * 4),
            scenes,
            feature_order=features,
            threshold=0.5,
        )
    with pytest.raises(ValueError, match="not bit-exact"):
        subject.score_v411(
            _UnstableModel([0.9] * 4),
            scenes,
            feature_order=features,
        )


def test_guard_frame_controls_unique_direct_presence_in_top100():
    split = _split()
    scenes, features = _scenes_with_features(split)
    v411 = subject.score_v411(
        _Model([0.9] * 4), scenes, feature_order=features
    )
    evidence = pd.DataFrame(
        [
            {
                "query_id": query_id,
                "direct_candidate_count": 1,
                "direct_siren_count": 1,
                "sole_direct_siret": (
                    "22222222200022"
                    if query_id == "comparison-1"
                    else "11111111100011"
                ),
                "sole_direct_siren": (
                    "222222222"
                    if query_id == "comparison-1"
                    else "111111111"
                ),
            }
            for query_id in split["query_id"]
        ]
    )
    pools = {
        query_id: {"11111111100011"}
        for query_id in split["query_id"]
    }
    decisions = subject.apply_guard_frame(
        v411,
        evidence,
        ranker_sirets=pools,
        populations=subject.assign_populations(split)[
            ["query_id", "population"]
        ],
    )
    rejected = decisions[decisions["query_id"].eq("comparison-1")].iloc[0]
    assert bool(rejected["sole_direct_in_top100"]) is False
    assert rejected["review_reason_v412"] == (
        "DIRECT_EVIDENCE_DISAGREES_TOP1"
    )


def _canonical_decisions(v412_auto: int) -> pd.DataFrame:
    rows = []
    for index in range(746):
        v411_auto = index < 614
        rows.append(
            {
                "query_id": f"q{index}",
                "population": "comparison_dev",
                "label_kind": "MATCH_EXACT" if index < 634 else "AMBIGUOUS",
                "ground_truth_siret": "11111111100011",
                "predicted_siret": "11111111100011",
                "acceptor_target": 1 if index < 634 else 0,
                "acceptor_score": 0.9 if v411_auto else 0.1,
                "decision_v411": "AUTO_MATCH" if v411_auto else "REVIEW",
                "review_reason_v411": None if v411_auto else "LOW_CONFIDENCE",
                "direct_candidate_count": 1,
                "direct_siren_count": 1,
                "sole_direct_siret": "11111111100011",
                "sole_direct_siren": "111111111",
                "sole_direct_in_top100": True,
                "decision_v412": (
                    "AUTO_MATCH" if index < v412_auto else "REVIEW"
                ),
                "review_reason_v412": (
                    None if index < v412_auto else "NO_DIRECT_EVIDENCE"
                ),
                "correct_exact_siret": index < 634,
                "input_siret_state": "ACTIVE",
                "source_segment": "train",
                "top1_siren_candidate_count": 1.0,
                "role_crm_count": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=subject.DECISION_COLUMNS)


def test_integer_gate_and_segment_noninferiority():
    metrics, segments = subject.evaluate_comparison_gate(
        _canonical_decisions(600), enforce_canonical=True
    )
    assert metrics["v412_g"]["auto_count"] == 600
    assert metrics["verdict"] == "GO_V412_HISTORICAL_GATE"
    assert metrics["latency_gate_evaluated"] is False
    assert metrics["production_certified"] is False
    assert all(row["coverage_noninferiority_pass"] for row in segments)

    metrics, _ = subject.evaluate_comparison_gate(
        _canonical_decisions(599), enforce_canonical=True
    )
    assert metrics["verdict"] == "STOP_V412_GUARD"
