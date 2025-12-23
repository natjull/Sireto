#!/usr/bin/env python3
"""
Train XGBoost matcher v2 with aligned training samples.

This script uses training samples generated with realistic candidate pools,
addressing the train/serve skew identified in the original training.

Key improvements:
- Uses samples from generate_training_samples.py (real pool, hard negatives)
- Includes new features (is_siege, is_association, alias_match, etc.)
- Trains both Ranker (LambdaMART) and Classifier (binary)
- Evaluates on hold-out test set with proper metrics

Usage:
    python scripts/train_xgb_matcher_v2.py [--samples data/samples_aligned.parquet]
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import FEATURE_NAMES

# Configuration
DEFAULT_SAMPLES = Path("data/samples_aligned.parquet")
MODEL_DIR = Path("models")

# XGBoost hyperparameters
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

CLASSIFIER_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": ["auc", "logloss"],
    "seed": 42,
    "scale_pos_weight": 100,  # Handle imbalance
}

NUM_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 20


def load_samples(path: Path) -> pd.DataFrame:
    """Load training samples from parquet."""
    return pd.read_parquet(path)


def prepare_data(
    df: pd.DataFrame, 
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for XGBoost training.
    
    IMPORTANT: Data is sorted by query_id to ensure groups align with feature matrix.
    
    Returns:
        X: Feature matrix
        y: Labels
        groups: Query group sizes (for ranking)
    """
    # Sort by query_id to ensure groups are aligned with data order
    df_sorted = df.sort_values("query_id").reset_index(drop=True)
    
    X = df_sorted[feature_names].values.astype(np.float32)
    y = df_sorted["label"].values.astype(np.float32)
    
    # Compute group sizes in sorted order (preserves order)
    groups = df_sorted.groupby("query_id", sort=False).size().values
    
    return X, y, groups


def compute_hit_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    k: int = 5,
) -> float:
    """Compute Hit@K metric for ranking evaluation."""
    hits = 0
    start = 0
    
    for group_size in groups:
        end = start + group_size
        group_labels = y_true[start:end]
        group_scores = y_pred[start:end]
        
        # Check if any positive is in top-K
        top_k_indices = np.argsort(group_scores)[::-1][:k]
        if any(group_labels[i] == 1 for i in top_k_indices):
            hits += 1
        
        start = end
    
    return hits / len(groups)


def compute_mrr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Compute Mean Reciprocal Rank."""
    reciprocal_ranks = []
    start = 0
    
    for group_size in groups:
        end = start + group_size
        group_labels = y_true[start:end]
        group_scores = y_pred[start:end]
        
        # Rank by score
        ranking = np.argsort(group_scores)[::-1]
        
        # Find rank of first positive
        for rank, idx in enumerate(ranking, 1):
            if group_labels[idx] == 1:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
        
        start = end
    
    return np.mean(reciprocal_ranks)


def train_ranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    groups_dev: np.ndarray,
) -> xgb.Booster:
    """Train XGBoost ranker (LambdaMART)."""
    print("\n--- Training Ranker (LambdaMART) ---")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(groups_train)
    
    ddev = xgb.DMatrix(X_dev, label=y_dev)
    ddev.set_group(groups_dev)
    
    evals = [(dtrain, "train"), (ddev, "dev")]
    
    ranker = xgb.train(
        RANKER_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=20,
    )
    
    return ranker


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
) -> xgb.XGBClassifier:
    """Train XGBoost binary classifier."""
    print("\n--- Training Classifier (Binary) ---")
    
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
    
    return classifier


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str,
    split_name: str,
) -> Dict:
    """Evaluate model and return metrics."""
    if model_type == "ranker":
        dmatrix = xgb.DMatrix(X)
        y_pred = model.predict(dmatrix)
    else:
        y_pred = model.predict_proba(X)[:, 1]
    
    # Compute metrics
    metrics = {
        "split": split_name,
        "model_type": model_type,
        "hit_at_1": compute_hit_at_k(y, y_pred, groups, k=1),
        "hit_at_3": compute_hit_at_k(y, y_pred, groups, k=3),
        "hit_at_5": compute_hit_at_k(y, y_pred, groups, k=5),
        "mrr": compute_mrr(y, y_pred, groups),
        "auc": roc_auc_score(y, y_pred) if len(np.unique(y)) > 1 else 0.0,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost matcher v2")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    
    print("=" * 60)
    print("XGBoost Matcher Training v2 (Aligned with Inference)")
    print("=" * 60)
    
    # Load samples
    print(f"\n1. Loading samples from {args.samples}...")
    df = load_samples(args.samples)
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {len(FEATURE_NAMES)}")
    
    # Split by split column
    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"]
    test_df = df[df["split"] == "test"]
    
    print(f"\n2. Data splits:")
    print(f"   Train: {len(train_df)} samples ({train_df['label'].sum()} positives)")
    print(f"   Dev:   {len(dev_df)} samples ({dev_df['label'].sum()} positives)")
    print(f"   Test:  {len(test_df)} samples ({test_df['label'].sum()} positives)")
    
    # Prepare data
    X_train, y_train, groups_train = prepare_data(train_df, FEATURE_NAMES)
    X_dev, y_dev, groups_dev = prepare_data(dev_df, FEATURE_NAMES)
    X_test, y_test, groups_test = prepare_data(test_df, FEATURE_NAMES)
    
    # Train ranker
    ranker = train_ranker(X_train, y_train, groups_train, X_dev, y_dev, groups_dev)
    
    # Train classifier
    classifier = train_classifier(X_train, y_train, X_dev, y_dev)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    all_metrics = []
    
    for split_name, X, y, groups in [
        ("dev", X_dev, y_dev, groups_dev),
        ("test", X_test, y_test, groups_test),
    ]:
        for model_type, model in [("ranker", ranker), ("classifier", classifier)]:
            metrics = evaluate_model(model, X, y, groups, model_type, split_name)
            all_metrics.append(metrics)
            print(f"\n{model_type.upper()} on {split_name}:")
            print(f"  Hit@1: {metrics['hit_at_1']:.4f}")
            print(f"  Hit@3: {metrics['hit_at_3']:.4f}")
            print(f"  Hit@5: {metrics['hit_at_5']:.4f}")
            print(f"  MRR:   {metrics['mrr']:.4f}")
            print(f"  AUC:   {metrics['auc']:.4f}")
    
    # Save models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    ranker_path = args.output_dir / f"xgbranker_{timestamp}.json"
    classifier_path = args.output_dir / f"xgbclassifier_{timestamp}.json"
    meta_path = args.output_dir / f"xgb_matcher_features_{timestamp}.json"
    
    ranker.save_model(str(ranker_path))
    classifier.save_model(str(classifier_path))
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "feature_names": FEATURE_NAMES,
        "ranker_params": RANKER_PARAMS,
        "classifier_params": CLASSIFIER_PARAMS,
        "metrics": all_metrics,
        "samples_file": str(args.samples),
        "train_size": len(train_df),
        "dev_size": len(dev_df),
        "test_size": len(test_df),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n--- Models saved ---")
    print(f"  Ranker: {ranker_path}")
    print(f"  Classifier: {classifier_path}")
    print(f"  Metadata: {meta_path}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
