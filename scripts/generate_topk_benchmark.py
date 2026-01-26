#!/usr/bin/env python3
"""
Prepare a Top-K CSV for routing benchmark from a samples parquet file.
Uses the V6_A model to score the test split and formats the output for route_xgb_results.py.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xgb_matcher.features import FEATURE_NAMES

# Wrapper classes for unpickling
class IsotonicCalibrator:
    def __init__(self, base_estimator, iso_reg):
        self.base_estimator = base_estimator
        self.iso_reg = iso_reg

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.iso_reg.predict(proba)
        calibrated = np.clip(calibrated, 0, 1)
        return np.column_stack([1 - calibrated, calibrated])

class SigmoidCalibrator:
    def __init__(self, base_estimator, lr):
        self.base_estimator = base_estimator
        self.lr = lr

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.lr.predict_proba(proba.reshape(-1, 1))
        return calibrated


def _normalize_siret(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().replace(" ", "")
    if not s or s.lower() == "nan":
        return None
    if "e" in s.lower():
        try:
            s = str(int(float(s)))
        except (ValueError, OverflowError):
            pass
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(14)

def _load_feature_order(meta_path: Path | None) -> list[str]:
    if meta_path and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
    return FEATURE_NAMES

def main():
    parser = argparse.ArgumentParser(description="Generate Top-K CSV for routing benchmark")
    parser.add_argument("--samples", type=Path, default=Path("data/samples_v6_A.parquet"))
    parser.add_argument("--model", type=Path, default=Path("models/xgb_decider_20260124_210218.json"))
    parser.add_argument("--calibrator", type=Path, default=Path("models/xgb_decider_calibrator_isotonic_20260124_210218.pkl"))
    parser.add_argument("--meta", type=Path, default=Path("models/xgb_two_stage_meta_20260124_210218.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_v6a_topk.csv"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test", "all"])
    args = parser.parse_args()

    print(f"Loading samples from {args.samples}...")
    df = pd.read_parquet(args.samples)
    
    # Filter for test split
    if "split" in df.columns:
        if args.split != "all":
            df = df[df["split"] == args.split].copy()
            print(f"Filtered to {args.split} split: {len(df)} rows")
        else:
            print(f"Using all splits: {len(df)} rows")
    else:
        print("Warning: No 'split' column found, using all data")

    # Load model artifacts
    feature_order = _load_feature_order(args.meta)
    print(f"Loading model from {args.model}...")
    model = xgb.XGBClassifier()
    model.load_model(str(args.model))

    calibrator = None
    if args.calibrator and args.calibrator.exists():
        print(f"Loading calibrator from {args.calibrator}...")
        with open(args.calibrator, "rb") as f:
            calibrator = pickle.load(f)

    # Prepare features
    X = df[feature_order].values.astype(np.float32)

    # Predict
    print("Scoring candidates...")
    if calibrator:
        scores = calibrator.predict_proba(X)[:, 1]
    else:
        scores = model.predict_proba(X)[:, 1]
    
    df["score"] = scores
    if "siret" in df.columns:
        df["siret"] = df["siret"].apply(_normalize_siret)

    # Rank and filter Top-K
    print(f"Ranking and filtering Top-{args.top_k}...")
    df_sorted = df.sort_values(["query_id", "score"], ascending=[True, False])
    
    topk_rows = []
    
    for query_id, group in df_sorted.groupby("query_id", sort=False):
        # Calculate gap and ratio for the top candidate
        top_candidates = group.head(args.top_k).copy()
        top_candidates = top_candidates.reset_index(drop=True)
        
        # Add rank
        top_candidates["rank"] = top_candidates.index
        
        if len(top_candidates) >= 2:
            score_1 = top_candidates.iloc[0]["score"]
            score_2 = top_candidates.iloc[1]["score"]
            gap = score_1 - score_2
            ratio = score_1 / score_2 if score_2 > 0 else 100.0
        else:
            gap = 0.0
            ratio = 1.0
            
        # Assign gap/ratio to all rows in the group (or at least the top one needs it for routing)
        # route_xgb_results.py expects these columns on the row
        top_candidates["score_gap"] = gap
        top_candidates["score_ratio"] = ratio
        
        topk_rows.append(top_candidates)

    df_topk = pd.concat(topk_rows, ignore_index=True)

    # Rename columns for routing script
    # route_xgb_results expects: crm_id, siret_candidate
    df_topk = df_topk.rename(columns={
        "query_id": "crm_id",
        "siret": "siret_candidate"
    })

    df_topk["crm_id"] = df_topk["crm_id"].astype(str)
    if "siret_candidate" in df_topk.columns:
        df_topk["siret_candidate"] = df_topk["siret_candidate"].apply(_normalize_siret)

    # Ensure required columns for routing rules are present
    # Some routing rules might look for crm_name, crm_address etc. which might be missing in the samples parquet
    # The samples parquet is optimized for training (features only).
    # However, route_xgb_results.py handles missing optional columns gracefully or via features.
    # The 'features' in the parquet like 'name_jaro_max' are sufficient for the 'certainty_rules'.
    # But some logic like segment detection uses 'crm_name' word count.
    # If crm_name is missing, it might default to "default" segment.
    # We can try to fake crm_name if not present, or accept that segment detection will be default.
    
    if "crm_name" not in df_topk.columns:
        df_topk["crm_name"] = "" # Will cause segment=default

    print(f"Saving {len(df_topk)} rows to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_topk.to_csv(args.output, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
