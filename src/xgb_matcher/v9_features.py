"""Retrieval and SIREN-scene features shared by V9 train and serve."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


V9_RETRIEVAL_FEATURE_NAMES = [
    "retrieval_sparse_rank_recip",
    "retrieval_dense_rank_recip",
    "retrieval_global_siren_rank_recip",
    "retrieval_rrf_score",
    "retrieval_channel_count",
    "retrieval_agreement",
    "siren_candidate_count",
    "siren_rrf_max",
    "siren_sparse_best_rank_recip",
    "siren_dense_best_rank_recip",
    "candidate_vs_siren_rrf_gap",
]


def _rank_recip(value: Any) -> float:
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / rank if rank > 0 else 0.0


def inject_retrieval_siren_features(
    feature_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    if len(feature_rows) != len(candidates):
        raise ValueError("feature_rows and candidates must have the same length")

    by_siren: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_siren[str(candidate.get("siren") or "")].append(candidate)

    aggregates: dict[str, dict[str, float]] = {}
    for siren, siblings in by_siren.items():
        sparse_recip = [_rank_recip(candidate.get("sparse_rank")) for candidate in siblings]
        dense_recip = [_rank_recip(candidate.get("dense_rank")) for candidate in siblings]
        rrf_scores = [float(candidate.get("rrf_score") or 0.0) for candidate in siblings]
        aggregates[siren] = {
            "count": float(len(siblings)),
            "rrf_max": max(rrf_scores, default=0.0),
            "sparse_best": max(sparse_recip, default=0.0),
            "dense_best": max(dense_recip, default=0.0),
        }

    for feature_row, candidate in zip(feature_rows, candidates, strict=True):
        siren = str(candidate.get("siren") or "")
        aggregate = aggregates[siren]
        candidate_rrf = float(candidate.get("rrf_score") or 0.0)
        feature_row.update(
            {
                "retrieval_sparse_rank_recip": _rank_recip(candidate.get("sparse_rank")),
                "retrieval_dense_rank_recip": _rank_recip(candidate.get("dense_rank")),
                "retrieval_global_siren_rank_recip": _rank_recip(
                    candidate.get("global_siren_rank")
                ),
                "retrieval_rrf_score": candidate_rrf,
                "retrieval_channel_count": float(
                    candidate.get("retrieval_channel_count") or 0
                ),
                "retrieval_agreement": float(
                    candidate.get("retrieval_agreement") or 0
                ),
                "siren_candidate_count": aggregate["count"],
                "siren_rrf_max": aggregate["rrf_max"],
                "siren_sparse_best_rank_recip": aggregate["sparse_best"],
                "siren_dense_best_rank_recip": aggregate["dense_best"],
                "candidate_vs_siren_rrf_gap": aggregate["rrf_max"] - candidate_rrf,
            }
        )


__all__ = ["V9_RETRIEVAL_FEATURE_NAMES", "inject_retrieval_siren_features"]
