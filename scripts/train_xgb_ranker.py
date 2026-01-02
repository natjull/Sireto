#!/usr/bin/env python3
"""
Train XGBoost ranker for the 2-stage pipeline (Stage 1 retrieval).

Outputs:
  - xgbranker_<ts>.json
  - xgbranker_fast_<ts>.json (optional, semantic features zeroed)
  - xgb_two_stage_meta_<ts>.json (feature order + model paths)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import FEATURE_NAMES


RANKER_PARAMS = {
    "objective": "rank:ndcg",
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": ["ndcg@5", "map@5"],
    "seed": 42,
}

NUM_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 20

SEMANTIC_FEATURES = [f for f in FEATURE_NAMES if f.startswith("name_semantic_")]


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


def compute_hit_at_k(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int = 5) -> float:
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


def compute_recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    recalls = []
    start = 0
    for group_size in groups:
        end = start + group_size
        group_labels = y_true[start:end]
        group_scores = y_pred[start:end]
        total_pos = float(np.sum(group_labels))
        if total_pos <= 0:
            start = end
            continue
        top_k_indices = np.argsort(group_scores)[::-1][:k]
        hits = float(np.sum(group_labels[top_k_indices]))
        recalls.append(hits / total_pos)
        start = end
    return float(np.mean(recalls)) if recalls else 0.0


def compute_mrr(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    reciprocal_ranks = []
    start = 0
    for group_size in groups:
        end = start + group_size
        group_labels = y_true[start:end]
        group_scores = y_pred[start:end]
        ranking = np.argsort(group_scores)[::-1]
        for rank, idx in enumerate(ranking, 1):
            if group_labels[idx] == 1:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
        start = end
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def evaluate_ranker(
    model: xgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    split: str,
    name: str,
) -> Dict[str, float]:
    dmat = xgb.DMatrix(X)
    y_pred = model.predict(dmat)
    return {
        "split": split,
        "model": name,
        "hit_at_1": compute_hit_at_k(y, y_pred, groups, k=1),
        "hit_at_3": compute_hit_at_k(y, y_pred, groups, k=3),
        "hit_at_5": compute_hit_at_k(y, y_pred, groups, k=5),
        "recall_at_10": compute_recall_at_k(y, y_pred, groups, k=10),
        "recall_at_20": compute_recall_at_k(y, y_pred, groups, k=20),
        "mrr": compute_mrr(y, y_pred, groups),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost Ranker (Stage 1)")
    parser.add_argument("--samples", type=Path, default=Path("data/samples_aligned_v4.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--meta-path", type=Path, default=None)
    parser.add_argument("--skip-semantic", action="store_true", help="Zero semantic features for ranker training.")
    parser.add_argument("--train-ranker-fast", action="store_true", default=True)
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = args.meta_path or (args.output_dir / f"xgb_two_stage_meta_{timestamp}.json")

    print("=" * 60)
    print("XGBoost Ranker Training (Stage 1)")
    print("=" * 60)

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

    X_train, y_train, groups_train = prepare_data(train_df, FEATURE_NAMES)
    X_dev, y_dev, groups_dev = prepare_data(dev_df, FEATURE_NAMES)
    X_test, y_test, groups_test = prepare_data(test_df, FEATURE_NAMES)

    if args.skip_semantic and SEMANTIC_FEATURES:
        X_train = _zero_out_features(X_train, FEATURE_NAMES, SEMANTIC_FEATURES)
        X_dev = _zero_out_features(X_dev, FEATURE_NAMES, SEMANTIC_FEATURES)
        X_test = _zero_out_features(X_test, FEATURE_NAMES, SEMANTIC_FEATURES)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(groups_train)
    ddev = xgb.DMatrix(X_dev, label=y_dev)
    ddev.set_group(groups_dev)

    ranker = xgb.train(
        RANKER_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(dtrain, "train"), (ddev, "dev")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=20,
    )

    ranker_fast = None
    if args.train_ranker_fast and SEMANTIC_FEATURES and not args.skip_semantic:
        print("\n--- Training Ranker FAST (semantic zeroed) ---")
        X_train_fast = _zero_out_features(X_train, FEATURE_NAMES, SEMANTIC_FEATURES)
        X_dev_fast = _zero_out_features(X_dev, FEATURE_NAMES, SEMANTIC_FEATURES)
        dtrain_fast = xgb.DMatrix(X_train_fast, label=y_train)
        dtrain_fast.set_group(groups_train)
        ddev_fast = xgb.DMatrix(X_dev_fast, label=y_dev)
        ddev_fast.set_group(groups_dev)
        ranker_fast = xgb.train(
            RANKER_PARAMS,
            dtrain_fast,
            num_boost_round=NUM_BOOST_ROUNDS,
            evals=[(dtrain_fast, "train"), (ddev_fast, "dev")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=20,
        )

    metrics = []
    for split_name, X, y, groups in [("dev", X_dev, y_dev, groups_dev), ("test", X_test, y_test, groups_test)]:
        metrics.append(evaluate_ranker(ranker, X, y, groups, split_name, "ranker"))
    if ranker_fast is not None:
        X_dev_fast = _zero_out_features(X_dev, FEATURE_NAMES, SEMANTIC_FEATURES)
        X_test_fast = _zero_out_features(X_test, FEATURE_NAMES, SEMANTIC_FEATURES)
        metrics.append(evaluate_ranker(ranker_fast, X_dev_fast, y_dev, groups_dev, "dev", "ranker_fast"))
        metrics.append(evaluate_ranker(ranker_fast, X_test_fast, y_test, groups_test, "test", "ranker_fast"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranker_path = args.output_dir / f"xgbranker_{timestamp}.json"
    ranker.save_model(str(ranker_path))
    ranker_fast_path = None
    if ranker_fast is not None:
        ranker_fast_path = args.output_dir / f"xgbranker_fast_{timestamp}.json"
        ranker_fast.save_model(str(ranker_fast_path))

    meta = _load_meta(meta_path)
    meta.update(
        {
            "timestamp": timestamp,
            "feature_names": FEATURE_NAMES,
            "feature_order": FEATURE_NAMES,
            "semantic_features_zeroed_for_ranker_fast": SEMANTIC_FEATURES,
            "semantic_enabled_samples": semantic_enabled_samples,
            "semantic_enabled_env_train": semantic_enabled_env,
            "ranker_params": RANKER_PARAMS,
            "ranker_skip_semantic": bool(args.skip_semantic),
            "ranker_model": str(ranker_path),
            "ranker_fast_model": str(ranker_fast_path) if ranker_fast_path else None,
            "ranker_metrics": metrics,
            "samples_file": str(args.samples),
        }
    )
    _save_meta(meta_path, meta)

    print("\n--- Models saved ---")
    print(f"  Ranker: {ranker_path}")
    if ranker_fast_path:
        print(f"  Ranker FAST: {ranker_fast_path}")
    print(f"  Meta: {meta_path}")


if __name__ == "__main__":
    main()
