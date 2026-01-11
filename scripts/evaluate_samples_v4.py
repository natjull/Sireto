#!/usr/bin/env python3
"""
Evaluate samples v4 quality and ranker recall.

Outputs:
  - coverage by split (queries with positives)
  - avg candidates per query
  - hardness ratio (same address / different name negatives)
  - semantic feature sanity (mean, non-zero rate)
  - recall@K from latest ranker_fast / ranker (if available)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import FEATURE_NAMES


def find_latest_ranker(model_dir: Path) -> tuple[Path | None, Path | None]:
    fast = sorted(model_dir.glob("xgbranker_fast_*.json"), reverse=True)
    if fast:
        ts = fast[0].stem.replace("xgbranker_fast_", "")
        meta = model_dir / f"xgb_matcher_features_{ts}.json"
        return fast[0], meta if meta.exists() else None
    regular = sorted(model_dir.glob("xgbranker_*.json"), reverse=True)
    if regular:
        ts = regular[0].stem.replace("xgbranker_", "")
        meta = model_dir / f"xgb_matcher_features_{ts}.json"
        return regular[0], meta if meta.exists() else None
    return None, None


def _hit_at_k(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int) -> float:
    hits = 0
    start = 0
    for group_size in groups:
        end = start + group_size
        group_labels = y_true[start:end]
        group_scores = y_pred[start:end]
        top_k = np.argsort(group_scores)[::-1][:k]
        if any(group_labels[i] == 1 for i in top_k):
            hits += 1
        start = end
    return hits / len(groups) if len(groups) else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate samples v4 quality.")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, default=None, help="DEPRECATED: CSV splits dir (default: use parquet split column)")
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--ranker-model", type=Path, default=None, help="Optional explicit ranker model path")
    p.add_argument("--meta-path", type=Path, default=None, help="Optional explicit meta/features path")
    p.add_argument("--k", type=str, default="1,3,5,10,20")
    args = p.parse_args()

    df = pd.read_parquet(args.samples)
    if "split" not in df.columns:
        raise ValueError("samples parquet must contain 'split' column")

    k_list = [int(x) for x in args.k.split(",") if x.strip()]

    print("=" * 60)
    print("SAMPLES V4 EVALUATION")
    print("=" * 60)

    # Coverage by split
    for split in ["train", "dev", "test"]:
        # Compute total queries: try CSV if splits_dir provided, else use parquet
        total_queries = None
        if args.splits_dir is not None:
            csv_path = args.splits_dir / f"{split}.csv"
            if csv_path.exists():
                base = pd.read_csv(csv_path)
                total_queries = base["crm_id"].nunique()
        
        if total_queries is None:
            # Fallback: use parquet query_id count for this split
            total_queries = df[df["split"] == split]["query_id"].nunique()

        df_split = df[df["split"] == split]
        queries_in_samples = df_split["query_id"].nunique()
        queries_with_pos = df_split[df_split["label"] == 1]["query_id"].nunique()
        avg_cands = len(df_split) / max(queries_in_samples, 1)

        print(f"\n[{split}]")
        print(f"  queries total: {total_queries}")
        print(f"  queries in samples: {queries_in_samples}")
        print(f"  queries with positive: {queries_with_pos} ({queries_with_pos/max(total_queries,1):.2%})")
        print(f"  avg candidates/query: {avg_cands:.2f}")

    # Hardness ratio: same address / different name negatives
    if {"addr_jaro", "name_jaro_max"}.issubset(df.columns):
        neg = df[df["label"] == 0]
        hard = neg[(neg["addr_jaro"] >= 0.95) & (neg["name_jaro_max"] < 0.50)]
        print(f"\nHard negatives (addr>=0.95 & name<0.50): {len(hard)}/{len(neg)} ({len(hard)/max(len(neg),1):.2%})")

    # Semantic sanity
    if "name_semantic_max" in df.columns:
        mean_sem = df["name_semantic_max"].mean()
        nz_sem = (df["name_semantic_max"] > 0).mean()
        print(f"\nSemantic max mean: {mean_sem:.6f} | non-zero rate: {nz_sem:.2%}")

    # Ranker recall@K
    ranker_path = args.ranker_model
    meta_path = args.meta_path
    if ranker_path is None:
        ranker_path, meta_path = find_latest_ranker(args.model_dir)
    if ranker_path is None:
        print("\nNo ranker model found → skip recall@K.")
        return

    feature_order = FEATURE_NAMES
    if meta_path and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES

    print(f"\nRanker: {ranker_path.name}")
    ranker = xgb.Booster()
    ranker.load_model(str(ranker_path))

    for split in ["dev", "test"]:
        df_split = df[df["split"] == split].copy()
        if df_split.empty:
            continue
        df_split = df_split.sort_values("query_id").reset_index(drop=True)
        X = df_split[feature_order].values.astype(np.float32)
        y = df_split["label"].values.astype(np.float32)
        groups = df_split.groupby("query_id", sort=False).size().values
        dmat = xgb.DMatrix(X, feature_names=feature_order)
        scores = ranker.predict(dmat)
        print(f"\n[{split}] Recall@K (ranker)")
        for k in k_list:
            hit = _hit_at_k(y, scores, groups, k)
            print(f"  Hit@{k}: {hit:.4f}")


if __name__ == "__main__":
    main()
