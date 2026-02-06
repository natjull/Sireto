"""
Train XGBoost Ranker (Stage 1) for SIRETO.
SSOT-aligned version.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import (
    FEATURE_NAMES,
)
from src.xgb_matcher.retrieval_config import (
    RetrievalConfigV1,
    RetrievalSignatureV1,
)

# --- Configuration ---
RANKER_PARAMS = {
    "objective": "rank:pairwise",
    "tree_method": "hist",
    "eta": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": ["ndcg@5", "map@5"],
}
NUM_BOOST_ROUND = 200
EARLY_STOPPING_ROUNDS = 20

SEMANTIC_FEATURES = [f for f in FEATURE_NAMES if f.startswith("name_semantic_")]


def _load_samples_meta(samples_path: Path) -> Dict[str, Any]:
    meta_path = Path(samples_path).with_suffix(".json")
    if not meta_path.exists():
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def _load_meta(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_meta(path: Path, meta: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def prepare_data(df: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare X, y, and groups for XGBoost ranking."""
    df = df.sort_values("query_id")
    X = df[feature_names].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    groups = df.groupby("query_id").size().values
    return X, y, groups


def evaluate_ranker(
    booster: xgb.Booster, X: np.ndarray, y: np.ndarray, groups: np.ndarray, split_name: str, model_name: str
) -> Dict[str, Any]:
    """Simple evaluation of the ranker."""
    dmat = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    dmat.set_group(groups)
    res = booster.eval_set([(dmat, "eval")])
    print(f"Metrics for {model_name} on {split_name}: {res}")

    preds = booster.predict(dmat)
    start = 0
    hits = 0
    total = len(groups)
    for glen in groups:
        g_preds = preds[start : start + glen]
        g_labels = y[start : start + glen]
        if len(g_labels) > 0 and g_labels[np.argmax(g_preds)] == 1:
            hits += 1
        start += glen
    hit_rate = hits / total if total > 0 else 0
    print(f"  Hit@1 ({split_name}): {hit_rate:.2%}")
    return {"split": split_name, "model": model_name, "hit@1": hit_rate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SIRETO Stage 1 Ranker")
    parser.add_argument("--samples", type=Path, required=True, help="Path to training samples parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory to save models")
    parser.add_argument("--skip-semantic", action="store_true", help="Zero out semantic features")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = args.output_dir / f"xgb_two_stage_meta_{timestamp}.json"

    print("Loading data...")
    df = pd.read_parquet(args.samples)
    
    semantic_enabled_samples = bool(df["semantic_enabled"].iloc[0]) if "semantic_enabled" in df.columns else False
    semantic_enabled_env = os.environ.get("XGB_SEMANTIC_ENABLED", "0") == "1"

    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"]
    test_df = df[df["split"] == "test"]

    print(f"Training on {len(train_df)} rows, validating on {len(dev_df)} rows.")

    X_train, y_train, groups_train = prepare_data(train_df, FEATURE_NAMES)
    X_dev, y_dev, groups_dev = prepare_data(dev_df, FEATURE_NAMES)
    X_test, y_test, groups_test = prepare_data(test_df, FEATURE_NAMES)

    if args.skip_semantic:
        print("Zeroing out semantic features for training.")
        idx = [FEATURE_NAMES.index(f) for f in SEMANTIC_FEATURES]
        X_train[:, idx] = 0.0
        X_dev[:, idx] = 0.0
        X_test[:, idx] = 0.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dtrain.set_group(groups_train)
    ddev = xgb.DMatrix(X_dev, label=y_dev, feature_names=FEATURE_NAMES)
    ddev.set_group(groups_dev)

    print("\n--- Training Full Ranker ---")
    ranker = xgb.train(
        RANKER_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (ddev, "dev")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=20,
    )

    ranker_fast = None
    if not args.skip_semantic:
        print("\n--- Training Ranker FAST (semantic zeroed) ---")
        X_train_fast = X_train.copy()
        idx = [FEATURE_NAMES.index(f) for f in SEMANTIC_FEATURES]
        X_train_fast[:, idx] = 0.0
        X_dev_fast = X_dev.copy()
        X_dev_fast[:, idx] = 0.0

        dtrain_f = xgb.DMatrix(X_train_fast, label=y_train, feature_names=FEATURE_NAMES)
        dtrain_f.set_group(groups_train)
        ddev_f = xgb.DMatrix(X_dev_fast, label=y_dev, feature_names=FEATURE_NAMES)
        ddev_f.set_group(groups_dev)

        ranker_fast = xgb.train(
            RANKER_PARAMS,
            dtrain_f,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain_f, "train"), (ddev_f, "dev")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=20,
        )

    metrics = []
    for split_name, X, y, groups in [("dev", X_dev, y_dev, groups_dev), ("test", X_test, y_test, groups_test)]:
        metrics.append(evaluate_ranker(ranker, X, y, groups, split_name, "ranker"))
    if ranker_fast is not None:
        idx = [FEATURE_NAMES.index(f) for f in SEMANTIC_FEATURES]
        X_dev_fast = X_dev.copy(); X_dev_fast[:, idx] = 0.0
        X_test_fast = X_test.copy(); X_test_fast[:, idx] = 0.0
        metrics.append(evaluate_ranker(ranker_fast, X_dev_fast, y_dev, groups_dev, "dev", "ranker_fast"))
        metrics.append(evaluate_ranker(ranker_fast, X_test_fast, y_test, groups_test, "test", "ranker_fast"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranker_path = args.output_dir / f"xgbranker_{timestamp}.json"
    ranker.save_model(str(ranker_path))
    ranker_fast_path = None
    if ranker_fast is not None:
        ranker_fast_path = args.output_dir / f"xgbranker_fast_{timestamp}.json"
        ranker_fast.save_model(str(ranker_fast_path))

    samples_meta = _load_samples_meta(args.samples)
    sig_v1_raw = samples_meta.get("retrieval_signature_v1")
    retrieval_config = None
    retrieval_signature = None
    if sig_v1_raw:
        config_raw = sig_v1_raw.get("config")
        if config_raw:
            retrieval_config = RetrievalConfigV1.from_dict(config_raw)
            retrieval_signature = RetrievalSignatureV1.from_dict(sig_v1_raw)
            if not retrieval_signature.matches(retrieval_config):
                print("WARNING: Retrieval signature internal mismatch in samples metadata.")
        else:
            print("WARNING: Missing config inside retrieval_signature_v1.")
    else:
        print("WARNING: Missing retrieval_signature_v1 in samples metadata JSON.")

    meta = _load_meta(meta_path)
    meta.update(
        {
            "timestamp": timestamp,
            "feature_names": FEATURE_NAMES,
            "feature_order": FEATURE_NAMES,
            "ranker_feature_order": FEATURE_NAMES,
            "ranker_fast_feature_order": FEATURE_NAMES,
            "semantic_features_zeroed_for_ranker_fast": SEMANTIC_FEATURES,
            "semantic_enabled_samples": semantic_enabled_samples,
            "semantic_enabled_env_train": semantic_enabled_env,
            "ranker_params": RANKER_PARAMS,
            "ranker_skip_semantic": bool(args.skip_semantic),
            "ranker_model": str(ranker_path.name),
            "ranker_fast_model": str(ranker_fast_path.name) if ranker_fast_path else None,
            "ranker_metrics": metrics,
            "samples_file": str(args.samples.name),
            "retrieval_signature_v1": sig_v1_raw,
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
