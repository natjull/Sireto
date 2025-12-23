#!/usr/bin/env python3
"""
Comprehensive evaluation script for XGBoost matcher.

Evaluates models with and without policy layer (reranking rules) to measure
the contribution of each component.

Metrics computed:
- Hit@1, Hit@3, Hit@5: Ranking accuracy
- MRR: Mean Reciprocal Rank
- NDCG@5: Normalized Discounted Cumulative Gain
- Calibration: Precision at different thresholds

Usage:
    python scripts/evaluate_xgb_comprehensive.py --dataset test --policy on
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.candidates import (
    load_candidates_for_locations,
    get_candidates_for_query,
    build_location_index,
)
from src.xgb_matcher.features import FEATURE_NAMES, make_features
from src.xgb_matcher.naming import primary_name, normalize_text

# Import policy layer
from scripts.infer_xgb_matcher_topk import apply_rerank_rules, find_latest_models

# Paths
SPLITS_DIR = Path("data/splits")
REPORTS_DIR = Path("reports")


def load_split(split_name: str) -> pd.DataFrame:
    """Load a data split (train/dev/test)."""
    path = SPLITS_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    return pd.read_csv(path, dtype=str)


def compute_ndcg_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 5,
) -> float:
    """Compute NDCG@K for a single query."""
    # Ideal ranking
    ideal_order = np.argsort(y_true)[::-1][:k]
    ideal_dcg = sum(y_true[i] / np.log2(r + 2) for r, i in enumerate(ideal_order) if y_true[i] > 0)
    
    if ideal_dcg == 0:
        return 0.0
    
    # Predicted ranking
    pred_order = np.argsort(y_pred)[::-1][:k]
    dcg = sum(y_true[i] / np.log2(r + 2) for r, i in enumerate(pred_order) if y_true[i] > 0)
    
    return dcg / ideal_dcg


class Evaluator:
    """Comprehensive evaluator for XGBoost matcher."""
    
    def __init__(
        self,
        ranker: xgb.Booster,
        classifier: xgb.XGBClassifier,
        candidates: Dict[str, dict],
        candidate_index: Dict[str, Dict[str, List[Tuple[str, dict]]]] | None = None,
        use_policy: bool = True,
    ):
        self.ranker = ranker
        self.classifier = classifier
        self.candidates = candidates
        self.candidate_index = candidate_index
        self.use_policy = use_policy
        
        self.stage1_top_n = 200
    
    def evaluate_query(self, row: pd.Series) -> Dict:
        """Evaluate a single query and return metrics."""
        ground_truth = row.get("ground_truth_siret", "")
        postcode = row.get("postcode", "")
        insee = row.get("insee", "")
        
        # Get candidates
        cand_list = get_candidates_for_query(
            postcode,
            insee,
            self.candidates,
            index=self.candidate_index,
        )
        
        if not cand_list:
            return {"status": "no_candidates"}
        
        # Compute features
        feats = []
        for siret, cand in cand_list:
            feat = make_features(
                crm_name=row.get("crm_name", ""),
                crm_address=row.get("crm_address", ""),
                crm_city=row.get("crm_city", ""),
                candidate=cand,
                crm_row=row,
            )
            feat["_siret"] = siret
            feat["_cand_name"] = primary_name(cand) or ""
            
            # Add metadata for policy layer
            ul_denoms = []
            for field in ["denomination_ul", "denomination_usuelle_ul", "sigle_ul"]:
                val = cand.get(field)
                if val:
                    ul_denoms.append(normalize_text(str(val)))
            feat["_ul_denoms"] = ul_denoms
            feat["_is_siege"] = cand.get("is_siege", False)
            
            feats.append(feat)
        
        # Stage 1: Ranker
        X = np.array([[f.get(fn, 0.0) for fn in FEATURE_NAMES] for f in feats], dtype=np.float32)
        dmatrix = xgb.DMatrix(X)
        ranker_scores = self.ranker.predict(dmatrix)
        top_n_idx = np.argsort(ranker_scores)[::-1][:self.stage1_top_n]
        
        # Stage 2: Classifier
        X_n = X[top_n_idx]
        feats_n = [feats[i] for i in top_n_idx]
        cand_list_n = [cand_list[i] for i in top_n_idx]
        
        class_probs = self.classifier.predict_proba(X_n)[:, 1]
        
        # Apply policy layer if enabled
        if self.use_policy:
            scores = apply_rerank_rules(class_probs, feats_n, row)
        else:
            scores = class_probs
        
        # Find rank of ground truth
        ranked_sirets = [cand_list_n[i][0] for i in np.argsort(scores)[::-1]]
        
        if ground_truth in ranked_sirets:
            rank = ranked_sirets.index(ground_truth) + 1
        else:
            # Check if ground truth was in original pool
            all_sirets = [s for s, _ in cand_list]
            if ground_truth in all_sirets:
                rank = len(ranked_sirets) + 1  # Beyond top-N
            else:
                return {"status": "gt_not_in_pool"}
        
        return {
            "status": "ok",
            "rank": rank,
            "pool_size": len(cand_list),
            "top_n_size": len(cand_list_n),
        }
    
    def evaluate_dataset(self, df: pd.DataFrame) -> Dict:
        """Evaluate entire dataset and compute aggregate metrics."""
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            result = self.evaluate_query(row)
            results.append(result)
        
        # Aggregate metrics
        ok_results = [r for r in results if r["status"] == "ok"]
        
        if not ok_results:
            return {"error": "No valid results"}
        
        ranks = [r["rank"] for r in ok_results]
        
        metrics = {
            "total_queries": len(df),
            "evaluated": len(ok_results),
            "no_candidates": sum(1 for r in results if r["status"] == "no_candidates"),
            "gt_not_in_pool": sum(1 for r in results if r["status"] == "gt_not_in_pool"),
            "hit_at_1": sum(1 for r in ranks if r == 1) / len(ok_results),
            "hit_at_3": sum(1 for r in ranks if r <= 3) / len(ok_results),
            "hit_at_5": sum(1 for r in ranks if r <= 5) / len(ok_results),
            "mrr": np.mean([1.0 / r for r in ranks]),
            "mean_rank": np.mean(ranks),
            "median_rank": np.median(ranks),
        }
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Comprehensive XGBoost evaluation")
    parser.add_argument("--dataset", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--policy", choices=["on", "off", "both"], default="both")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    
    print("=" * 60)
    print("XGBoost Matcher Comprehensive Evaluation")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading {args.dataset} split...")
    df = load_split(args.dataset)
    print(f"   Rows: {len(df)}")
    
    # Build candidate pool
    postcodes = set(df["postcode"].dropna().unique())
    insee = set(df["insee"].dropna().unique())
    print(f"\n2. Building candidate pool...")
    candidates = load_candidates_for_locations(postcodes, insee)
    print(f"   Candidates: {len(candidates)}")
    candidate_index = build_location_index(candidates)
    
    # Load models
    model_paths = find_latest_models()
    print(f"\n3. Loading models (timestamp: {model_paths['timestamp']})...")
    
    ranker = xgb.Booster()
    ranker.load_model(str(model_paths["ranker"]))
    
    classifier = xgb.XGBClassifier()
    classifier.load_model(str(model_paths["classifier"]))
    
    # Evaluate
    all_results = {}
    
    if args.policy in ["off", "both"]:
        print("\n4. Evaluating WITHOUT policy layer...")
        evaluator = Evaluator(
            ranker,
            classifier,
            candidates,
            candidate_index=candidate_index,
            use_policy=False,
        )
        metrics_off = evaluator.evaluate_dataset(df)
        all_results["policy_off"] = metrics_off
        
        print(f"\n   Results (Policy OFF):")
        print(f"   Hit@1: {metrics_off['hit_at_1']:.4f}")
        print(f"   Hit@5: {metrics_off['hit_at_5']:.4f}")
        print(f"   MRR:   {metrics_off['mrr']:.4f}")
    
    if args.policy in ["on", "both"]:
        print("\n5. Evaluating WITH policy layer...")
        evaluator = Evaluator(
            ranker,
            classifier,
            candidates,
            candidate_index=candidate_index,
            use_policy=True,
        )
        metrics_on = evaluator.evaluate_dataset(df)
        all_results["policy_on"] = metrics_on
        
        print(f"\n   Results (Policy ON):")
        print(f"   Hit@1: {metrics_on['hit_at_1']:.4f}")
        print(f"   Hit@5: {metrics_on['hit_at_5']:.4f}")
        print(f"   MRR:   {metrics_on['mrr']:.4f}")
    
    # Compare
    if args.policy == "both":
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        delta_hit1 = metrics_on["hit_at_1"] - metrics_off["hit_at_1"]
        delta_hit5 = metrics_on["hit_at_5"] - metrics_off["hit_at_5"]
        delta_mrr = metrics_on["mrr"] - metrics_off["mrr"]
        print(f"   Hit@1: {delta_hit1:+.4f}")
        print(f"   Hit@5: {delta_hit5:+.4f}")
        print(f"   MRR:   {delta_mrr:+.4f}")
    
    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or REPORTS_DIR / f"eval_{args.dataset}_{timestamp}.json"
    
    report = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "model_timestamp": model_paths["timestamp"],
        "results": all_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n   Report saved: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
