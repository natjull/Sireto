#!/usr/bin/env python3
"""
Train XGBoost decider (Stage 2 classifier) for the 2-stage pipeline.

Optionally filters training samples to top-K candidates per query using a ranker,
to better reflect inference-time candidate distributions.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import FEATURE_NAMES


# Calibrator wrapper classes (must match inference for pickle compatibility)
class IsotonicCalibrator:
    """Wrapper for isotonic calibration of a base classifier."""
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
    """Wrapper for sigmoid (Platt) calibration of a base classifier."""
    def __init__(self, base_estimator, lr):
        self.base_estimator = base_estimator
        self.lr = lr

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.lr.predict_proba(proba.reshape(-1, 1))
        return calibrated

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


CLASSIFIER_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": ["auc", "logloss"],
    "seed": 42,
    "scale_pos_weight": 100,
}

NUM_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 20


def load_samples(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def prepare_data(df: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_sorted = df.sort_values("query_id").reset_index(drop=True)
    X = df_sorted[feature_names].values.astype(np.float32)
    y = df_sorted["label"].values.astype(np.float32)
    groups = df_sorted.groupby("query_id", sort=False).size().values
    return X, y, groups


def _zero_out_features(X: np.ndarray, feature_names: List[str], features_to_zero: List[str]) -> np.ndarray:
    if not features_to_zero:
        return X
    idx = [feature_names.index(f) for f in features_to_zero if f in feature_names]
    if not idx:
        return X
    Xz = X.copy()
    Xz[:, idx] = 0.0
    return Xz


def compute_hit_at_k(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int = 1) -> float:
    hits = 0
    start = 0
    for group_size in groups:
        end = start + group_size
        group_labels = y_true[start:end]
        group_scores = y_pred[start:end]
        top_k_indices = np.argsort(group_scores)[::-1][:k]
        if any(group_labels[i] == 1 for i in top_k_indices):
            hits += 1
        start = end
    return hits / len(groups) if len(groups) else 0.0


def filter_topk_by_ranker(
    df: pd.DataFrame,
    ranker: xgb.Booster,
    feature_order: List[str],
    top_k: int,
    zero_features: List[str] | None = None,
) -> pd.DataFrame:
    if top_k <= 0:
        return df
    df_sorted = df.sort_values("query_id").reset_index(drop=True)
    X = df_sorted[feature_order].values.astype(np.float32)
    if zero_features:
        X = _zero_out_features(X, feature_order, zero_features)
    scores = ranker.predict(xgb.DMatrix(X, feature_names=feature_order))
    df_sorted["_ranker_score"] = scores

    selected_idx = []
    for _, group in df_sorted.groupby("query_id", sort=False):
        top = group.nlargest(top_k, "_ranker_score")
        pos = group[group["label"] == 1]
        combined = pd.concat([top, pos]).drop_duplicates()
        selected_idx.extend(combined.index.tolist())
    filtered = df_sorted.loc[selected_idx].sort_values("query_id").reset_index(drop=True)
    return filtered


def _load_meta(meta_path: Path) -> Dict:
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_meta(meta_path: Path, meta: Dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def calibrate_classifier(
    classifier: xgb.XGBClassifier,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str,
):
    if method == "none":
        return None
    from sklearn.calibration import CalibratedClassifierCV
    try:
        calibrator = CalibratedClassifierCV(estimator=classifier, method=method, cv="prefit", ensemble=False)
        calibrator.fit(X_calib, y_calib)
    except (TypeError, ValueError):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        y_proba = classifier.predict_proba(X_calib)[:, 1]
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_proba, y_calib)
            calibrator = IsotonicCalibrator(classifier, iso)
        else:
            lr = LogisticRegression()
            lr.fit(y_proba.reshape(-1, 1), y_calib)
            calibrator = SigmoidCalibrator(classifier, lr)
    return calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost Decider (Stage 2)")
    parser.add_argument("--samples", type=Path, default=Path("data/samples_v4_with_ranker.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--meta-path", type=Path, default=None)
    parser.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"], default="isotonic")
    parser.add_argument("--ranker-model", type=Path, default=None)
    parser.add_argument("--ranker-meta", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = args.meta_path or (args.output_dir / f"xgb_two_stage_meta_{timestamp}.json")

    df = load_samples(args.samples)
    if "split" not in df.columns:
        raise ValueError("Samples must include a 'split' column (train/dev/test).")

    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"]
    test_df = df[df["split"] == "test"]

    semantic_enabled_samples = None
    if "semantic_enabled" in df.columns:
        vals = sorted(df["semantic_enabled"].dropna().unique().tolist())
        if vals:
            semantic_enabled_samples = int(vals[0])
            if len(vals) > 1:
                print(f"[WARN] semantic_enabled has multiple values in samples: {vals}")

    semantic_enabled_env = int(os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1")
    if semantic_enabled_samples is not None and semantic_enabled_samples != semantic_enabled_env:
        print(
            f"[WARN] semantic_enabled mismatch: samples={semantic_enabled_samples} vs env={semantic_enabled_env} "
            "(train/serve skew risk)"
        )

    # Optional top-k filtering using ranker
    ranker = None
    feature_order = FEATURE_NAMES
    zero_features: List[str] = []
    if args.ranker_meta and args.ranker_meta.exists():
        meta = _load_meta(args.ranker_meta)
        feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
        zero_features = meta.get("semantic_features_zeroed_for_ranker_fast") or []
        if not args.ranker_model:
            ranker_path = meta.get("ranker_fast_model") or meta.get("ranker_model")
            if ranker_path:
                args.ranker_model = Path(ranker_path)

    if args.ranker_model and args.ranker_model.exists():
        ranker = xgb.Booster()
        ranker.load_model(str(args.ranker_model))

    if ranker is not None and args.top_k > 0:
        train_df = filter_topk_by_ranker(train_df, ranker, feature_order, args.top_k, zero_features)
        dev_df = filter_topk_by_ranker(dev_df, ranker, feature_order, args.top_k, zero_features)
        test_df = filter_topk_by_ranker(test_df, ranker, feature_order, args.top_k, zero_features)

    X_train, y_train, groups_train = prepare_data(train_df, FEATURE_NAMES)
    X_dev, y_dev, groups_dev = prepare_data(dev_df, FEATURE_NAMES)
    X_test, y_test, groups_test = prepare_data(test_df, FEATURE_NAMES)

    classifier = xgb.XGBClassifier(
        n_estimators=NUM_BOOST_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **CLASSIFIER_PARAMS,
    )
    classifier.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_dev, y_dev)],
        verbose=20,
    )

    calibrator = calibrate_classifier(classifier, X_dev, y_dev, args.calibration)

    def _eval(split_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_name: str, model) -> Dict:
        if hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(xgb.DMatrix(X))
        return {
            "split": split_name,
            "model": model_name,
            "auc": roc_auc_score(y, y_pred) if len(np.unique(y)) > 1 else 0.0,
            "avg_precision": average_precision_score(y, y_pred) if len(np.unique(y)) > 1 else 0.0,
            "brier": brier_score_loss(y, y_pred) if len(np.unique(y)) > 1 else 0.0,
            "hit_at_1": compute_hit_at_k(y, y_pred, groups, k=1),
            "hit_at_3": compute_hit_at_k(y, y_pred, groups, k=3),
            "hit_at_5": compute_hit_at_k(y, y_pred, groups, k=5),
        }

    metrics = []
    metrics.append(_eval("dev", X_dev, y_dev, groups_dev, "decider", classifier))
    metrics.append(_eval("test", X_test, y_test, groups_test, "decider", classifier))
    if calibrator is not None:
        metrics.append(_eval("dev", X_dev, y_dev, groups_dev, "calibrated", calibrator))
        metrics.append(_eval("test", X_test, y_test, groups_test, "calibrated", calibrator))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decider_path = args.output_dir / f"xgb_decider_{timestamp}.json"
    classifier.get_booster().save_model(str(decider_path))

    calibrator_path = None
    if calibrator is not None:
        calibrator_path = args.output_dir / f"xgb_decider_calibrator_{args.calibration}_{timestamp}.pkl"
        with open(calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)

    meta = _load_meta(meta_path)
    meta.update(
        {
            "timestamp": timestamp,
            "feature_names": FEATURE_NAMES,
            "feature_order": FEATURE_NAMES,
            "decider_params": CLASSIFIER_PARAMS,
            "decider_model": str(decider_path),
            "decider_calibrator": str(calibrator_path) if calibrator_path else None,
            "decider_calibration_method": args.calibration,
            "decider_top_k_filter": args.top_k if ranker is not None else 0,
            "decider_metrics": metrics,
            "samples_file": str(args.samples),
            "semantic_enabled_samples": semantic_enabled_samples,
            "semantic_enabled_env_train": semantic_enabled_env,
        }
    )
    _save_meta(meta_path, meta)

    print("\n--- Models saved ---")
    print(f"  Decider: {decider_path}")
    if calibrator_path:
        print(f"  Calibrator: {calibrator_path}")
    print(f"  Meta: {meta_path}")


if __name__ == "__main__":
    main()
