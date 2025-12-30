#!/usr/bin/env python3
"""
Generate training samples v3 - Single parquet scan, per-commune grouping.

Strategy:
1. ONE scan of parquet, building candidates by INSEE/CP
2. Group CRM queries by commune  
3. For each commune group, generate samples using local pool
4. Split by SIREN for no leakage

This is MUCH faster than rescanning parquet for each commune.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import gc
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import FEATURE_NAMES, make_features, normalize_text, jaro_sim
from src.xgb_matcher.candidates import load_candidates_for_locations

# Configuration
DEFAULT_OUTPUT = Path("data/samples_aligned_v3.parquet")
DEFAULT_SPLITS_DIR = Path("data/splits")
TRAINING_DATA = Path("data/entrainements.csv")
PARQUET_PATH = Path("data/StockEtablissement_utf8.parquet")
UL_PATH = Path("data/StockUniteLegale_utf8.parquet")
MODEL_DIR = Path("models")

MAX_NEGATIVES = 50  # Per query
HARD_RATIO = 0.5
SAME_ADDR_NEG_MAX = 10  # Max extra negatives with same address but different name
SEED = 42
BATCH_SIZE = 100_000
PREFILTER_TOP_K = 500  # Max candidates to fully score with make_features()
SEMANTIC_ENABLED = os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"
MAX_POOL_FOR_SCORING = int(os.getenv("XGB_MAX_POOL_FOR_SCORING", "10000"))
SLOW_QUERY_SEC = float(os.getenv("XGB_SAMPLE_SLOW_SEC", "30"))
GC_EVERY = int(os.getenv("XGB_GC_EVERY", "2000"))


def load_training_data(path: Path) -> pd.DataFrame:
    """Load CRM training data."""
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    df = df.rename(columns={
        "SITE": "crm_name",
        "SITE_CLI_ADRESSE": "crm_address",
        "SITE_CLI_COMMUNE": "crm_city",
        "CODE_POSTAL": "postcode",
        "CODE_INSEE": "insee",
        "SIRET": "ground_truth_siret",
    })
    df = df[df["ground_truth_siret"].notna() & (df["ground_truth_siret"].str.len() == 14)]
    df["crm_id"] = range(len(df))
    df["siren"] = df["ground_truth_siret"].str[:9]
    # Location key: prefer INSEE, fallback to CP
    df["loc_key"] = df["insee"].fillna("") + "|" + df["postcode"].fillna("")
    return df


def create_siren_split(df: pd.DataFrame, seed: int = SEED):
    """Split by SIREN."""
    sirens = df["siren"].unique().tolist()
    random.seed(seed)
    random.shuffle(sirens)
    n = len(sirens)
    train_sirens = set(sirens[:int(n * 0.70)])
    dev_sirens = set(sirens[int(n * 0.70):int(n * 0.85)])
    test_sirens = set(sirens[int(n * 0.85):])
    return (
        df[df["siren"].isin(train_sirens)].copy(),
        df[df["siren"].isin(dev_sirens)].copy(),
        df[df["siren"].isin(test_sirens)].copy(),
    )


def load_candidates_by_location(
    insee_codes: Set[str], 
    postcodes: Set[str]
) -> Dict[str, List[dict]]:
    """
    Load candidates indexed by INSEE and postcode using shared logic.
    
    This version uses src.xgb_matcher.candidates.load_candidates_for_locations 
    to ensure full enrichment (UL names, PM dirigeants from SQLite).
    """
    print("  Loading and enriching candidates (this may take a few minutes)...")
    full_mapping = load_candidates_for_locations(
        postcodes=postcodes,
        insee_codes=insee_codes,
        include_closed_establishments=True,
        verbose=True
    )
    
    by_insee: Dict[str, List[dict]] = defaultdict(list)
    by_cp: Dict[str, List[dict]] = defaultdict(list)
    
    for cand in full_mapping.values():
        insee = cand.get("insee")
        cp = cand.get("postcode")
        
        if insee and insee in insee_codes:
            by_insee[insee].append(cand)
        if cp and cp in postcodes:
            by_cp[cp].append(cand)
            
    # Merge into single dict with prefixes for get_candidates_for_query
    result: Dict[str, List[dict]] = {}
    for k, v in by_insee.items():
        result[f"insee:{k}"] = v
    for k, v in by_cp.items():
        result[f"cp:{k}"] = v
    
    total_locations = len(result)
    print(f"  Indexed {len(full_mapping)} unique candidates across {total_locations} location keys")
    
    return result


def get_candidates_for_query(
    insee: Optional[str],
    postcode: Optional[str],
    candidates_by_loc: Dict[str, List[dict]],
) -> List[dict]:
    """Get candidates for a query, preferring INSEE."""
    if insee:
        key = f"insee:{insee}"
        if key in candidates_by_loc:
            return candidates_by_loc[key]
    if postcode:
        key = f"cp:{postcode}"
        if key in candidates_by_loc:
            return candidates_by_loc[key]
    return []


def compute_score(feat: dict) -> float:
    """Quick scoring for hard negative selection."""
    return 0.6 * feat.get("name_jaro_max", 0.0) + 0.4 * feat.get("addr_jaro", 0.0)


def quick_score(crm_name: str, cand: dict) -> float:
    """Ultra-fast scoring with just one jaro on primary name.
    
    Used for pre-filtering candidates before expensive make_features().
    """
    cand_name = cand.get("denomination") or cand.get("enseigne1") or ""
    if not cand_name:
        return 0.0
    return jaro_sim(normalize_text(crm_name), normalize_text(cand_name))


def find_latest_ranker(models_dir: Path) -> Tuple[Optional[Path], Optional[Path], bool]:
    """Return (ranker_path, meta_path, is_fast)."""
    fast = sorted(models_dir.glob("xgbranker_fast_*.json"), reverse=True)
    if fast:
        ranker_path = fast[0]
        ts = ranker_path.stem.replace("xgbranker_fast_", "")
        meta_path = models_dir / f"xgb_matcher_features_{ts}.json"
        return ranker_path, (meta_path if meta_path.exists() else None), True
    regular = sorted(models_dir.glob("xgbranker_*.json"), reverse=True)
    if regular:
        ranker_path = regular[0]
        ts = ranker_path.stem.replace("xgbranker_", "")
        meta_path = models_dir / f"xgb_matcher_features_{ts}.json"
        return ranker_path, (meta_path if meta_path.exists() else None), False
    return None, None, False


def load_ranker_meta(meta_path: Optional[Path]) -> Tuple[List[str], List[str]]:
    """Load feature order and semantic-zero list from metadata."""
    if not meta_path or not meta_path.exists():
        return FEATURE_NAMES, [f for f in FEATURE_NAMES if f.startswith("name_semantic_")]
    with open(meta_path) as f:
        meta = json.load(f)
    feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
    semantic_zero = meta.get("semantic_features_zeroed_for_ranker_fast") or []
    return feature_order, semantic_zero


def score_with_ranker(
    feats: List[dict],
    *,
    ranker: xgb.Booster,
    feature_order: List[str],
    zero_features: List[str] | None = None,
) -> np.ndarray:
    """Score candidates using the provided ranker model."""
    X = np.array([[f.get(fn, 0.0) for fn in feature_order] for f in feats], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if zero_features:
        idx = [feature_order.index(f) for f in zero_features if f in feature_order]
        if idx:
            X[:, idx] = 0.0
    dmat = xgb.DMatrix(X, feature_names=feature_order)
    return ranker.predict(dmat)


def generate_samples_for_query(
    crm_row: pd.Series,
    candidates: List[dict],
    max_neg: int = MAX_NEGATIVES,
    prefilter_k: int = PREFILTER_TOP_K,
    ranker: xgb.Booster | None = None,
    ranker_feature_order: Optional[List[str]] = None,
    ranker_zero_features: Optional[List[str]] = None,
) -> List[dict]:
    """Generate samples for one query with pre-filtering."""
    t0 = time.perf_counter()
    gt_siret = crm_row.get("ground_truth_siret", "")
    query_id = crm_row["crm_id"]
    crm_name = crm_row.get("crm_name", "")
    
    if not candidates:
        return []
    
    # ===== PRE-FILTER: Keep only top-K by quick jaro score =====
    # This reduces from ~5000 candidates to ~500 before expensive make_features()
    t_prefilter_start = time.perf_counter()
    
    if len(candidates) > prefilter_k:
        # Ensure ground truth is always included
        gt_cand = None
        other_cands = []
        for c in candidates:
            if c["siret"] == gt_siret:
                gt_cand = c
            else:
                other_cands.append(c)
        
        # If pool is HUGE (e.g. Paris, Lille), randomly sample BEFORE scoring
        # This trades some recall for speed — acceptable for training data
        if MAX_POOL_FOR_SCORING > 0 and len(other_cands) > MAX_POOL_FOR_SCORING:
            other_cands = random.sample(other_cands, MAX_POOL_FOR_SCORING)
        
        # Score and filter non-GT candidates
        scored = [(quick_score(crm_name, c), c) for c in other_cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered_cands = [c for _, c in scored[:prefilter_k - 1]]  # Leave room for GT
        
        # Add back ground truth if found
        if gt_cand:
            filtered_cands.append(gt_cand)
        
        candidates = filtered_cands

    t_prefilter_end = time.perf_counter()

    # ===== FULL FEATURE COMPUTATION on filtered candidates =====
    all_feats = []
    for cand in candidates:
        feat = make_features(crm_row, cand, skip_semantic=not SEMANTIC_ENABLED)
        feat["_siret"] = cand["siret"]
        feat["_is_pos"] = (cand["siret"] == gt_siret)
        all_feats.append(feat)
    t_features = time.perf_counter()

    # Score for hard negatives: prefer ranker if provided
    if ranker is not None:
        feature_order = ranker_feature_order or FEATURE_NAMES
        scores = score_with_ranker(
            all_feats,
            ranker=ranker,
            feature_order=feature_order,
            zero_features=ranker_zero_features or [],
        )
        for f, s in zip(all_feats, scores):
            f["_score"] = float(s)
    else:
        for f in all_feats:
            f["_score"] = compute_score(f)
    t_ranker = time.perf_counter()
    
    positives = [f for f in all_feats if f["_is_pos"]]
    negatives = [f for f in all_feats if not f["_is_pos"]]
    
    if not positives:
        return []
    
    # Sample negatives
    num_hard = int(max_neg * HARD_RATIO)
    neg_sorted = sorted(negatives, key=lambda x: x["_score"], reverse=True)
    hard = neg_sorted[:num_hard]
    rest = neg_sorted[num_hard:]
    rand = random.sample(rest, min(max_neg - num_hard, len(rest))) if rest else []

    # Add "same address, different name" negatives (critical FP pattern)
    same_addr = [
        f
        for f in negatives
        if float(f.get("addr_jaro", 0.0)) >= 0.95
        and float(f.get("name_jaro_max", 0.0)) < 0.50
    ]
    same_addr = sorted(same_addr, key=lambda x: x.get("addr_jaro", 0.0), reverse=True)
    same_addr = same_addr[: min(SAME_ADDR_NEG_MAX, max_neg)]

    # Merge with priority: hard -> same_addr -> random, while preserving max_neg and uniqueness
    selected = []
    seen = set()
    for group in (hard, same_addr, rand):
        for f in group:
            siret = f.get("_siret")
            if siret in seen:
                continue
            selected.append(f)
            if siret:
                seen.add(siret)
            if len(selected) >= max_neg:
                break
        if len(selected) >= max_neg:
            break
    
    samples = []
    for f in positives:
        s = {fn: f.get(fn, 0.0) for fn in FEATURE_NAMES}
        s["label"] = 1
        s["query_id"] = query_id
        s["siret"] = f["_siret"]
        samples.append(s)
    
    for f in selected:
        s = {fn: f.get(fn, 0.0) for fn in FEATURE_NAMES}
        s["label"] = 0
        s["query_id"] = query_id
        s["siret"] = f["_siret"]
        samples.append(s)

    total_time = time.perf_counter() - t0
    if total_time >= SLOW_QUERY_SEC:
        insee = crm_row.get("insee")
        postcode = crm_row.get("postcode")
        print(
            "[slow] crm_id=%s insee=%s cp=%s cand=%s prefilter=%s "
            "t_prefilter=%.2fs t_features=%.2fs t_ranker=%.2fs total=%.2fs",
            query_id,
            insee,
            postcode,
            len(candidates),
            prefilter_k,
            t_prefilter_end - t_prefilter_start,
            t_features - t_prefilter_end,
            t_ranker - t_features,
            total_time,
        )

    return samples


def generate_all_samples(
    df: pd.DataFrame,
    candidates_by_loc: Dict[str, List[dict]],
    max_neg: int,
    ranker: xgb.Booster | None = None,
    ranker_feature_order: Optional[List[str]] = None,
    ranker_zero_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate samples for all queries."""
    all_samples = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        cands = get_candidates_for_query(
            row.get("insee"), row.get("postcode"), candidates_by_loc
        )
        samples = generate_samples_for_query(
            row,
            cands,
            max_neg,
            ranker=ranker,
            ranker_feature_order=ranker_feature_order,
            ranker_zero_features=ranker_zero_features,
        )
        all_samples.extend(samples)
        if GC_EVERY > 0 and idx > 0 and idx % GC_EVERY == 0:
            gc.collect()
    
    return pd.DataFrame(all_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--max-negatives", type=int, default=MAX_NEGATIVES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ranker-model", type=Path, default=None, help="Path to ranker model for hard negatives")
    parser.add_argument("--ranker-meta", type=Path, default=None, help="Path to ranker metadata JSON")
    parser.add_argument(
        "--disable-ranker-hard-negatives",
        action="store_true",
        help="Fallback to heuristic hard negatives if no ranker is available",
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Sample Generation v3 (Single Scan, Per-Commune)")
    print("=" * 60)
    
    print("\n1. Loading CRM data...")
    df = load_training_data(TRAINING_DATA)
    print(f"   Rows: {len(df)}, SIRENs: {df['siren'].nunique()}")
    
    print("\n2. SIREN split...")
    train_df, dev_df, test_df = create_siren_split(df, args.seed)
    print(f"   Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.splits_dir / "train.csv", index=False)
    dev_df.to_csv(args.splits_dir / "dev.csv", index=False)
    test_df.to_csv(args.splits_dir / "test.csv", index=False)
    
    print("\n3. Loading candidates (single parquet scan)...")
    insee_codes = set(df["insee"].dropna().unique())
    postcodes = set(df["postcode"].dropna().unique())
    print(f"   INSEE: {len(insee_codes)}, Postcodes: {len(postcodes)}")
    
    candidates_by_loc = load_candidates_by_location(insee_codes, postcodes)
    
    # Load ranker model for hard negatives if available
    ranker = None
    ranker_feature_order: Optional[List[str]] = None
    ranker_zero_features: List[str] = []
    is_fast = False
    ranker_info = None
    if not args.disable_ranker_hard_negatives:
        ranker_path = args.ranker_model
        meta_path = args.ranker_meta
        if ranker_path is None:
            ranker_path, meta_path, is_fast = find_latest_ranker(MODEL_DIR)
        else:
            is_fast = "fast" in ranker_path.name.lower()
        if ranker_path and ranker_path.exists():
            ranker = xgb.Booster()
            ranker.load_model(str(ranker_path))
            ranker_feature_order, semantic_zero = load_ranker_meta(meta_path)
            ranker_zero_features = semantic_zero if is_fast else []
            ranker_info = {
                "path": str(ranker_path),
                "meta": str(meta_path) if meta_path else None,
                "is_fast": is_fast,
            }
            print(f"\n[HardNeg] Using ranker: {ranker_path} (fast={is_fast})")
        else:
            print("\n[HardNeg] No ranker model found → fallback to heuristic scoring.")

    print("\n4. Generating train samples...")
    train_samples = generate_all_samples(
        train_df,
        candidates_by_loc,
        args.max_negatives,
        ranker=ranker,
        ranker_feature_order=ranker_feature_order,
        ranker_zero_features=ranker_zero_features,
    )
    print(f"   → {len(train_samples)} samples ({train_samples['label'].sum()} pos)")
    
    print("\n5. Generating dev samples...")
    dev_samples = generate_all_samples(
        dev_df,
        candidates_by_loc,
        args.max_negatives,
        ranker=ranker,
        ranker_feature_order=ranker_feature_order,
        ranker_zero_features=ranker_zero_features,
    )
    print(f"   → {len(dev_samples)} samples ({dev_samples['label'].sum()} pos)")
    
    print("\n6. Generating test samples...")
    test_samples = generate_all_samples(
        test_df,
        candidates_by_loc,
        args.max_negatives,
        ranker=ranker,
        ranker_feature_order=ranker_feature_order,
        ranker_zero_features=ranker_zero_features,
    )
    print(f"   → {len(test_samples)} samples ({test_samples['label'].sum()} pos)")
    
    train_samples["split"] = "train"
    dev_samples["split"] = "dev"
    test_samples["split"] = "test"
    
    all_samples = pd.concat([train_samples, dev_samples, test_samples], ignore_index=True)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_samples.to_parquet(args.output, index=False)
    print(f"\n7. Saved {len(all_samples)} samples → {args.output}")
    
    meta = {
        "generated": datetime.now().isoformat(),
        "seed": args.seed,
        "train": len(train_samples),
        "dev": len(dev_samples),
        "test": len(test_samples),
        "hard_negative_ranker": ranker_info,
    }
    with open(args.output.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Done!")


if __name__ == "__main__":
    main()
