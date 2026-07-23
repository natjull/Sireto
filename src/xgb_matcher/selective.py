"""Selective prediction utilities for the V9 query-level acceptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectiveThreshold:
    threshold: float
    auto_count: int
    total_count: int
    error_count: int
    coverage: float
    precision: float


def prepare_top1(inference_rows: pd.DataFrame) -> pd.DataFrame:
    """Return one top-ranked row per query and compute its score margin."""
    required = {"crm_id", "score"}
    missing = required - set(inference_rows.columns)
    if missing:
        raise ValueError(f"Missing inference columns: {sorted(missing)}")

    rows = inference_rows.copy()
    rows["score"] = pd.to_numeric(rows["score"], errors="coerce").fillna(0.0)
    if "rank" in rows.columns:
        rows["rank"] = pd.to_numeric(rows["rank"], errors="coerce")
        rows = rows.sort_values(
            ["crm_id", "rank", "score"],
            ascending=[True, True, False],
        )
    else:
        rows = rows.sort_values(["crm_id", "score"], ascending=[True, False])

    grouped = rows.groupby("crm_id", sort=False)
    top1 = grouped.nth(0).reset_index()
    top2 = grouped.nth(1).reset_index()[["crm_id", "score"]]
    top2 = top2.rename(columns={"score": "score_top2"})
    top1 = top1.rename(columns={"score": "score_top1"})
    top1 = top1.merge(top2, on="crm_id", how="left")
    top1["score_top2"] = top1["score_top2"].fillna(0.0)
    top1["score_gap"] = top1["score_top1"] - top1["score_top2"]
    top1["score"] = top1["score_top1"]
    return top1


# Backward-compatible private name used by the former calibration script.
_prepare_top1 = prepare_top1


def risk_coverage_curve(
    confidence: Iterable[float],
    correct: Iterable[int | bool],
) -> pd.DataFrame:
    """Build the exact empirical risk/coverage curve at observed thresholds."""
    scores = np.asarray(list(confidence), dtype=np.float64)
    labels = np.asarray(list(correct), dtype=np.int8)
    if scores.shape != labels.shape:
        raise ValueError("confidence and correct must have identical shapes")
    if scores.ndim != 1:
        raise ValueError("confidence and correct must be one-dimensional")
    if len(scores) == 0:
        return pd.DataFrame(
            columns=[
                "threshold",
                "auto_count",
                "coverage",
                "precision",
                "risk",
                "error_count",
            ]
        )

    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_correct = labels[order]
    cumulative_correct = np.cumsum(sorted_correct)
    endpoints = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )

    rows = []
    total = len(scores)
    for endpoint in endpoints:
        auto_count = int(endpoint + 1)
        correct_count = int(cumulative_correct[endpoint])
        error_count = auto_count - correct_count
        precision = correct_count / auto_count
        rows.append(
            {
                "threshold": float(sorted_scores[endpoint]),
                "auto_count": auto_count,
                "coverage": auto_count / total,
                "precision": precision,
                "risk": 1.0 - precision,
                "error_count": error_count,
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    confidence: Iterable[float],
    correct: Iterable[int | bool],
    *,
    target_precision: float = 0.998,
    min_auto_count: int = 1,
) -> SelectiveThreshold | None:
    """Select maximum empirical coverage satisfying a target precision on dev."""
    curve = risk_coverage_curve(confidence, correct)
    eligible = curve[
        (curve["precision"] >= target_precision)
        & (curve["auto_count"] >= min_auto_count)
    ]
    if eligible.empty:
        return None
    row = eligible.sort_values(
        ["coverage", "threshold"],
        ascending=[False, False],
    ).iloc[0]
    return SelectiveThreshold(
        threshold=float(row["threshold"]),
        auto_count=int(row["auto_count"]),
        total_count=int(curve["auto_count"].max()),
        error_count=int(row["error_count"]),
        coverage=float(row["coverage"]),
        precision=float(row["precision"]),
    )


def clopper_pearson_error_upper(
    error_count: int,
    sample_count: int,
    *,
    confidence_level: float = 0.99,
) -> float:
    """One-sided exact upper confidence bound for an observed error rate."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if error_count < 0 or error_count > sample_count:
        raise ValueError("error_count must be between 0 and sample_count")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if error_count == sample_count:
        return 1.0

    from scipy.stats import beta

    return float(
        beta.ppf(
            confidence_level,
            error_count + 1,
            sample_count - error_count,
        )
    )


def certified_precision_lower(
    error_count: int,
    sample_count: int,
    *,
    confidence_level: float = 0.99,
) -> float:
    """Lower confidence bound on precision for a frozen AUTO policy."""
    return 1.0 - clopper_pearson_error_upper(
        error_count,
        sample_count,
        confidence_level=confidence_level,
    )


__all__ = [
    "SelectiveThreshold",
    "prepare_top1",
    "risk_coverage_curve",
    "select_threshold",
    "clopper_pearson_error_upper",
    "certified_precision_lower",
]
