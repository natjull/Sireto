"""Query-level V9 scenes built from ranked candidate predictions."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .contracts import GroundTruthKind


V9_SCENE_FEATURE_NAMES = [
    "has_candidate",
    "candidate_count",
    "score_top1",
    "score_top2",
    "score_gap",
    "score_ratio",
    "score_mean",
    "score_std",
    "score_entropy",
    "top3_mean",
    "unique_siren_count",
    "top1_siren_candidate_count",
    "same_siren_top2",
    "siren_score_gap",
    "top1_retrieval_channel_count",
    "top1_retrieval_agreement",
    "top1_rrf_score",
    "sparse_dense_top1_agreement",
    "retrieval_disagreement",
    "retrieval_miss",
]


def _normalise_siret(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits.zfill(14) if digits else None


def _score_entropy(scores: np.ndarray) -> float:
    if len(scores) <= 1:
        return 0.0
    shifted = scores - np.max(scores)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return entropy / float(np.log(len(scores)))


def _scene_for_candidates(rows: pd.DataFrame) -> tuple[dict[str, float], dict[str, Any]]:
    if rows.empty:
        features = {name: 0.0 for name in V9_SCENE_FEATURE_NAMES}
        features["retrieval_miss"] = 1.0
        return features, {
            "predicted_siret": None,
            "predicted_siren": None,
            "prediction_origin": None,
        }

    ranked = rows.copy()
    scene_origin = (
        str(ranked["prediction_origin"].dropna().iloc[0])
        if "prediction_origin" in ranked.columns
        and not ranked["prediction_origin"].dropna().empty
        else None
    )
    ranked["score"] = pd.to_numeric(ranked["score"], errors="coerce").fillna(0.0)
    if "rank" in ranked.columns:
        ranked["rank"] = pd.to_numeric(ranked["rank"], errors="coerce")
        ranked = ranked.sort_values(["rank", "score"], ascending=[True, False])
    else:
        ranked = ranked.sort_values("score", ascending=False)

    candidate_column = (
        "candidate_siret"
        if "candidate_siret" in ranked.columns
        else "siret_candidate"
    )
    ranked["candidate_siret"] = ranked[candidate_column].map(_normalise_siret)
    ranked = ranked[ranked["candidate_siret"].notna()].copy()
    if ranked.empty:
        features, prediction = _scene_for_candidates(ranked)
        prediction["prediction_origin"] = scene_origin
        return features, prediction
    ranked["candidate_siren"] = ranked["candidate_siret"].str[:9]

    scores = ranked["score"].to_numpy(dtype=np.float64)
    top1 = ranked.iloc[0]
    top2_score = float(scores[1]) if len(scores) > 1 else 0.0
    top1_score = float(scores[0])
    denominator = max(abs(top2_score), 1e-6)
    siren_best = (
        ranked.groupby("candidate_siren", sort=False)["score"]
        .max()
        .sort_values(ascending=False)
        .to_numpy(dtype=np.float64)
    )
    top1_siren = str(top1["candidate_siren"])
    top1_siren_count = int((ranked["candidate_siren"] == top1_siren).sum())
    same_siren_top2 = (
        int(str(ranked.iloc[1]["candidate_siren"]) == top1_siren)
        if len(ranked) > 1
        else 0
    )

    sparse_dense_agreement = 0
    if {"sparse_rank", "dense_rank"}.issubset(ranked.columns):
        sparse = ranked[pd.to_numeric(ranked["sparse_rank"], errors="coerce").eq(1)]
        dense = ranked[pd.to_numeric(ranked["dense_rank"], errors="coerce").eq(1)]
        if not sparse.empty and not dense.empty:
            sparse_dense_agreement = int(
                sparse.iloc[0]["candidate_siret"] == dense.iloc[0]["candidate_siret"]
            )

    top1_channel_count = float(top1.get("retrieval_channel_count") or 0.0)
    top1_agreement = float(top1.get("retrieval_agreement") or 0.0)
    features = {
        "has_candidate": 1.0,
        "candidate_count": float(len(ranked)),
        "score_top1": top1_score,
        "score_top2": top2_score,
        "score_gap": top1_score - top2_score,
        "score_ratio": top1_score / denominator,
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_entropy": _score_entropy(scores),
        "top3_mean": float(scores[:3].mean()),
        "unique_siren_count": float(ranked["candidate_siren"].nunique()),
        "top1_siren_candidate_count": float(top1_siren_count),
        "same_siren_top2": float(same_siren_top2),
        "siren_score_gap": (
            float(siren_best[0] - siren_best[1]) if len(siren_best) > 1 else float(siren_best[0])
        ),
        "top1_retrieval_channel_count": top1_channel_count,
        "top1_retrieval_agreement": top1_agreement,
        "top1_rrf_score": float(top1.get("rrf_score") or 0.0),
        "sparse_dense_top1_agreement": float(sparse_dense_agreement),
        "retrieval_disagreement": float(
            top1_channel_count > 1 and not (top1_agreement or sparse_dense_agreement)
        ),
        "retrieval_miss": 0.0,
    }
    return features, {
        "predicted_siret": str(top1["candidate_siret"]),
        "predicted_siren": top1_siren,
        "prediction_origin": scene_origin,
    }


def build_query_scenes(
    candidate_predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Build one scene per labelled query, including zero-candidate scenes."""
    required_labels = {
        "query_id",
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
        "split",
    }
    missing = required_labels - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    if "query_id" not in candidate_predictions.columns:
        raise ValueError("Candidate predictions require query_id")
    if not candidate_predictions.empty and "score" not in candidate_predictions.columns:
        raise ValueError("Candidate predictions require score")

    grouped = {
        str(query_id): group
        for query_id, group in candidate_predictions.groupby("query_id", sort=False)
    }
    scenes: list[dict[str, Any]] = []
    for label in labels.to_dict("records"):
        query_id = str(label["query_id"])
        features, prediction = _scene_for_candidates(
            grouped.get(query_id, candidate_predictions.iloc[0:0])
        )
        ground_truth_siret = _normalise_siret(label.get("ground_truth_siret"))
        kind = str(label["label_kind"])
        correct = int(
            kind == GroundTruthKind.MATCH_EXACT.value
            and prediction["predicted_siret"] == ground_truth_siret
        )
        scenes.append(
            {
                "query_id": query_id,
                "split": label["split"],
                "label_kind": kind,
                "ground_truth_siret": ground_truth_siret,
                "ground_truth_siren": label.get("ground_truth_siren"),
                **prediction,
                "is_exact_siret_correct": correct,
                **features,
            }
        )
    return pd.DataFrame(scenes)


def build_inference_scene(
    query_id: str,
    candidate_predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Build an unlabeled scene for online acceptance."""
    features, prediction = _scene_for_candidates(candidate_predictions)
    return {"query_id": str(query_id), **prediction, **features}


def assert_oof_training_scenes(scenes: pd.DataFrame) -> None:
    """Reject train scenes that came from in-sample ranker predictions."""
    train = scenes[scenes["split"].eq("train")]
    origins = set(train["prediction_origin"].dropna().astype(str))
    if origins != {"oof"}:
        raise ValueError(
            "All train scenes must be produced by out-of-fold ranker predictions "
            f"(observed origins: {sorted(origins) or ['missing']})"
        )


def split_dev_roles(query_ids: pd.Series, seed: int = 42) -> pd.Series:
    """Deterministically separate model calibration from threshold selection."""
    return query_ids.astype(str).map(
        lambda query_id: (
            "calibration"
            if hashlib.sha256(f"{seed}:cal:{query_id}".encode()).digest()[0] < 128
            else "threshold"
        )
    )


__all__ = [
    "V9_SCENE_FEATURE_NAMES",
    "build_query_scenes",
    "build_inference_scene",
    "assert_oof_training_scenes",
    "split_dev_roles",
]
