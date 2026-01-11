#!/usr/bin/env python3
"""
Evaluate XGBoost decider on labeled samples (candidate-level ground truth).

Outputs:
- AUC / PR-AUC / Brier / LogLoss
- Calibration bins + ECE
- Hit@K (ranking) by query_id
- Precision / coverage at high-score thresholds

Usage:
  python scripts/evaluate_decider_on_samples.py \
    --samples data/samples_v4_with_ranker.parquet \
    --model models/xgb_decider_20260103_132351.json \
    --calibrator models/xgb_decider_calibrator_isotonic_20260103_132351.pkl \
    --meta models/xgb_two_stage_meta_20260103_132351.json \
    --output reports/decider_eval.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import FEATURE_NAMES


# Calibrator wrapper classes for pickle compatibility
class IsotonicCalibrator:
    def __init__(self, base_estimator, iso_reg):
        self.base_estimator = base_estimator
        self.iso_reg = iso_reg

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.iso_reg.predict(proba)
        calibrated = np.clip(calibrated, 0, 1)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SigmoidCalibrator:
    def __init__(self, base_estimator, lr):
        self.base_estimator = base_estimator
        self.lr = lr

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.lr.predict_proba(proba.reshape(-1, 1))
        return calibrated

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _load_feature_order(meta_path: Path | None) -> List[str]:
    if meta_path and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
    return FEATURE_NAMES


def _ece(scores: np.ndarray, labels: np.ndarray, bins: List[float]) -> float:
    ece = 0.0
    n = len(scores)
    if n == 0:
        return 0.0
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (scores > low) & (scores <= high)
        if not mask.any():
            continue
        bin_scores = scores[mask]
        bin_labels = labels[mask]
        ece += (len(bin_scores) / n) * abs(bin_scores.mean() - bin_labels.mean())
    return float(ece)


def _calibration_bins(scores: np.ndarray, labels: np.ndarray, bins: List[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (scores > low) & (scores <= high)
        if not mask.any():
            rows.append(
                {
                    "bin": f"({low:.3f}, {high:.3f}]",
                    "count": 0,
                    "mean_score": 0.0,
                    "empirical_rate": 0.0,
                }
            )
            continue
        bin_scores = scores[mask]
        bin_labels = labels[mask]
        rows.append(
            {
                "bin": f"({low:.3f}, {high:.3f}]",
                "count": int(mask.sum()),
                "mean_score": float(bin_scores.mean()),
                "empirical_rate": float(bin_labels.mean()),
            }
        )
    return rows


def _hit_at_k(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int) -> float:
    hits = 0
    start = 0
    for group_size in groups:
        end = start + group_size
        group_scores = scores[start:end]
        group_labels = labels[start:end]
        top_k = np.argsort(group_scores)[::-1][:k]
        if any(group_labels[i] == 1 for i in top_k):
            hits += 1
        start = end
    return hits / len(groups) if len(groups) else 0.0


def _precision_at_threshold(scores: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    mask = scores >= thr
    if not mask.any():
        return {"precision": 0.0, "coverage": 0.0, "fp_rate": 0.0}
    precision = float(labels[mask].mean())
    coverage = float(mask.mean())
    fp_rate = 1.0 - precision
    return {"precision": precision, "coverage": coverage, "fp_rate": fp_rate}


def _eval_split(df: pd.DataFrame, feature_order: List[str], model, calibrator, split_name: str) -> Dict[str, Any]:
    if df.empty:
        return {"split": split_name, "error": "empty"}

    X = df[feature_order].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    raw_scores = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        cal_scores = calibrator.predict_proba(X)[:, 1]
    else:
        cal_scores = raw_scores

    # Grouping for ranking metrics
    df_sorted = df.sort_values("query_id").reset_index(drop=True)
    groups = df_sorted.groupby("query_id", sort=False).size().values

    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0]

    metrics = {
        "split": split_name,
        "rows": int(len(df)),
        "queries": int(df["query_id"].nunique()),
        "pos_rate": float(df["label"].mean()),
        "raw": {
            "auc": float(roc_auc_score(y, raw_scores)) if len(np.unique(y)) > 1 else 0.0,
            "avg_precision": float(average_precision_score(y, raw_scores)) if len(np.unique(y)) > 1 else 0.0,
            "brier": float(brier_score_loss(y, raw_scores)),
            "logloss": float(log_loss(y, raw_scores, labels=[0, 1])),
            "ece": _ece(raw_scores, y, bins),
            "calibration_bins": _calibration_bins(raw_scores, y, bins),
        },
        "calibrated": {
            "auc": float(roc_auc_score(y, cal_scores)) if len(np.unique(y)) > 1 else 0.0,
            "avg_precision": float(average_precision_score(y, cal_scores)) if len(np.unique(y)) > 1 else 0.0,
            "brier": float(brier_score_loss(y, cal_scores)),
            "logloss": float(log_loss(y, cal_scores, labels=[0, 1])),
            "ece": _ece(cal_scores, y, bins),
            "calibration_bins": _calibration_bins(cal_scores, y, bins),
        },
    }

    # Ranking metrics using calibrated scores
    df_sorted["_score"] = cal_scores
    df_sorted = df_sorted.sort_values("query_id").reset_index(drop=True)
    y_sorted = df_sorted["label"].values.astype(np.float32)
    scores_sorted = df_sorted["_score"].values.astype(np.float32)
    groups_sorted = df_sorted.groupby("query_id", sort=False).size().values

    metrics["ranking"] = {
        "hit_at_1": _hit_at_k(scores_sorted, y_sorted, groups_sorted, 1),
        "hit_at_3": _hit_at_k(scores_sorted, y_sorted, groups_sorted, 3),
        "hit_at_5": _hit_at_k(scores_sorted, y_sorted, groups_sorted, 5),
        "hit_at_10": _hit_at_k(scores_sorted, y_sorted, groups_sorted, 10),
    }

    thresholds = [0.95, 0.97, 0.98, 0.99, 0.995]
    metrics["precision_at_threshold"] = {
        f"{thr:.3f}": _precision_at_threshold(cal_scores, y, thr) for thr in thresholds
    }

    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate XGB decider on samples parquet")
    p.add_argument("--samples", type=Path, default=Path("data/samples_v4_with_ranker.parquet"))
    p.add_argument("--model", type=Path, default=Path("models/xgb_decider_20260103_132351.json"))
    p.add_argument(
        "--calibrator", type=Path, default=Path("models/xgb_decider_calibrator_isotonic_20260103_132351.pkl")
    )
    p.add_argument("--meta", type=Path, default=Path("models/xgb_two_stage_meta_20260103_132351.json"))
    p.add_argument("--output", type=Path, default=Path("reports/decider_eval.json"))
    args = p.parse_args()

    df = pd.read_parquet(args.samples)
    feature_order = _load_feature_order(args.meta)

    model = xgb.XGBClassifier()
    model.load_model(str(args.model))

    calibrator = None
    if args.calibrator and args.calibrator.exists():
        with open(args.calibrator, "rb") as f:
            calibrator = pickle.load(f)

    results: Dict[str, Any] = {
        "samples": str(args.samples),
        "model": str(args.model),
        "calibrator": str(args.calibrator) if args.calibrator else None,
        "meta": str(args.meta) if args.meta else None,
        "splits": {},
    }

    for split in ["train", "dev", "test"]:
        df_split = df[df["split"] == split].copy()
        results["splits"][split] = _eval_split(df_split, feature_order, model, calibrator, split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
