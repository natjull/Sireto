#!/usr/bin/env python3
"""
Train a query-level routing risk model (predict top1 correctness).

Supports two model types:
  - logit: LogisticRegression (baseline)
  - xgb: XGBClassifier (recommended - captures feature interactions)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.routing_risk import BASE_FEATURES, default_feature_columns


class IsotonicCalibrator:
    def __init__(self, base_estimator, iso_reg):
        self.base_estimator = base_estimator
        self.iso_reg = iso_reg

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.iso_reg.predict(proba)
        calibrated = np.clip(calibrated, 0, 1)
        return np.column_stack([1 - calibrated, calibrated])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train routing risk model (top1 correctness).")
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("models"))
    p.add_argument("--target", choices=["strict", "siren"], default="siren")
    p.add_argument("--calibration", choices=["none", "isotonic"], default="isotonic")
    p.add_argument("--target-fp-rate", type=float, default=0.01)
    p.add_argument("--report-path", type=Path, default=Path("reports/routing_risk_report.json"))
    p.add_argument(
        "--model-type",
        choices=["logit", "xgb"],
        default="xgb",
        help="Model type: logit (LogisticRegression) or xgb (XGBClassifier, recommended).",
    )
    p.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=100,
        help="Number of boosting rounds for XGBClassifier.",
    )
    p.add_argument(
        "--xgb-max-depth",
        type=int,
        default=4,
        help="Max tree depth for XGBClassifier.",
    )
    p.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for XGBClassifier.",
    )
    return p.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str)


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_feature_columns(df: pd.DataFrame) -> List[str]:
    desired = default_feature_columns(BASE_FEATURES)
    return [c for c in desired if c in df.columns]


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        dev_df = df[df["split"] == "dev"].copy()
        test_df = df[df["split"] == "test"].copy()
        if len(train_df) and len(dev_df):
            return train_df, dev_df, test_df

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx].copy()
    dev_df = df.iloc[split_idx:].copy()
    return train_df, dev_df, pd.DataFrame()


def evaluate_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    target_fp_rate: float,
) -> Tuple[float, List[dict], bool]:
    """
    Evaluate thresholds and find the best one respecting target FP rate.
    
    Returns:
        (best_threshold, rows, target_met)
        - target_met: True if a threshold meeting target_fp_rate was found
    """
    rows: List[dict] = []
    best_threshold = 0.5
    best_auto_rate = -1.0
    target_met = False

    # Use finer grid for better threshold selection
    thresholds = np.arange(0.50, 1.00, 0.005)
    
    for thr in thresholds:
        auto_mask = probs >= thr
        auto_count = int(auto_mask.sum())
        if auto_count == 0:
            continue
        auto_correct = int(y_true[auto_mask].sum())
        fp_count = auto_count - auto_correct
        fp_rate = fp_count / auto_count if auto_count else 0.0
        auto_rate = auto_count / len(y_true) if len(y_true) else 0.0

        rows.append(
            {
                "threshold": float(thr),
                "auto_rate": float(auto_rate),
                "fp_rate": float(fp_rate),
                "auto_count": auto_count,
                "fp_count": fp_count,
            }
        )

        if fp_rate <= target_fp_rate and auto_rate > best_auto_rate:
            best_threshold = float(thr)
            best_auto_rate = auto_rate
            target_met = True

    # If no threshold meets target, pick the one with lowest FP rate (then max auto_rate)
    if not target_met and rows:
        fallback = min(rows, key=lambda r: (r["fp_rate"], -r["auto_rate"]))
        best_threshold = fallback["threshold"]
        print(f"WARNING: No threshold meets target FP rate {target_fp_rate:.2%}. "
              f"Using best available: thr={best_threshold:.3f}, fp_rate={fallback['fp_rate']:.2%}")

    return best_threshold, rows, target_met


def main() -> None:
    args = _parse_args()

    df = load_dataset(args.dataset_path)

    label_col = "label_top1_strict" if args.target == "strict" else "label_top1_siren"
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    feature_cols = build_feature_columns(df)
    if not feature_cols:
        raise ValueError("No feature columns found in dataset")

    df = _coerce_numeric(df, feature_cols + [label_col])
    df = df.dropna(subset=feature_cols + [label_col])

    train_df, dev_df, test_df = split_dataset(df)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.int32)
    X_dev = dev_df[feature_cols].values.astype(np.float32)
    y_dev = dev_df[label_col].values.astype(np.int32)

    # Build classifier based on model type
    if args.model_type == "xgb":
        # Compute scale_pos_weight for class imbalance
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        clf = XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=10,
        )
        # Use dev set for early stopping
        clf.fit(
            X_train, y_train,
            eval_set=[(X_dev, y_dev)],
            verbose=False,
        )
        print(f"XGBoost trained: {clf.best_iteration} rounds (early stopping)")
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        print("LogisticRegression trained")

    calibrator = None
    if args.calibration == "isotonic" and len(dev_df) > 0:
        proba_dev = clf.predict_proba(X_dev)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(proba_dev, y_dev)
        calibrator = IsotonicCalibrator(clf, iso)

    if calibrator:
        proba = calibrator.predict_proba(X_dev)[:, 1]
    else:
        proba = clf.predict_proba(X_dev)[:, 1]

    best_threshold, rows, target_met = evaluate_thresholds(proba, y_dev, args.target_fp_rate)

    pred_dev = (proba >= best_threshold).astype(int)
    precision = precision_score(y_dev, pred_dev, zero_division=0.0)
    recall = recall_score(y_dev, pred_dev, zero_division=0.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "routing_risk_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    calibrator_path = None
    if calibrator:
        calibrator_path = args.output_dir / "routing_risk_calibrator.pkl"
        with open(calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)

    meta = {
        "target": args.target,
        "model_type": args.model_type,
        "features": feature_cols,
        "threshold": best_threshold,
        "calibration": args.calibration,
        "model_path": str(model_path),
        "calibrator_path": str(calibrator_path) if calibrator_path else None,
    }
    meta_path = args.output_dir / "routing_risk_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report = {
        "target": args.target,
        "model_type": args.model_type,
        "target_fp_rate": args.target_fp_rate,
        "target_met": target_met,
        "best_threshold": best_threshold,
        "precision": float(precision),
        "recall": float(recall),
        "feature_count": len(feature_cols),
        "thresholds": rows,
        "train_size": int(len(train_df)),
        "dev_size": int(len(dev_df)),
        "test_size": int(len(test_df)),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved meta to {meta_path}")
    print(f"Saved report to {args.report_path}")
    print(f"Best threshold: {best_threshold:.3f}, Precision: {precision:.2%}, Recall: {recall:.2%}")


if __name__ == "__main__":
    main()
