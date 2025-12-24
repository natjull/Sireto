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
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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

MAX_NEGATIVES = 50  # Per query
HARD_RATIO = 0.5
SEED = 42
BATCH_SIZE = 100_000
PREFILTER_TOP_K = 500  # Max candidates to fully score with make_features()


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


def generate_samples_for_query(
    crm_row: pd.Series,
    candidates: List[dict],
    max_neg: int = MAX_NEGATIVES,
    prefilter_k: int = PREFILTER_TOP_K,
) -> List[dict]:
    """Generate samples for one query with pre-filtering."""
    gt_siret = crm_row.get("ground_truth_siret", "")
    query_id = crm_row["crm_id"]
    crm_name = crm_row.get("crm_name", "")
    
    if not candidates:
        return []
    
    # ===== PRE-FILTER: Keep only top-K by quick jaro score =====
    # This reduces from ~5000 candidates to ~500 before expensive make_features()
    if len(candidates) > prefilter_k:
        # Ensure ground truth is always included
        gt_cand = None
        other_cands = []
        for c in candidates:
            if c["siret"] == gt_siret:
                gt_cand = c
            else:
                other_cands.append(c)
        
        # Score and filter non-GT candidates
        scored = [(quick_score(crm_name, c), c) for c in other_cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered_cands = [c for _, c in scored[:prefilter_k - 1]]  # Leave room for GT
        
        # Add back ground truth if found
        if gt_cand:
            filtered_cands.append(gt_cand)
        
        candidates = filtered_cands
    
    # ===== FULL FEATURE COMPUTATION on filtered candidates =====
    all_feats = []
    for cand in candidates:
        feat = make_features(crm_row, cand)
        feat["_siret"] = cand["siret"]
        feat["_is_pos"] = (cand["siret"] == gt_siret)
        feat["_score"] = compute_score(feat)
        all_feats.append(feat)
    
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
    selected = hard + rand
    
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
    
    return samples


def generate_all_samples(
    df: pd.DataFrame,
    candidates_by_loc: Dict[str, List[dict]],
    max_neg: int,
) -> pd.DataFrame:
    """Generate samples for all queries."""
    all_samples = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        cands = get_candidates_for_query(
            row.get("insee"), row.get("postcode"), candidates_by_loc
        )
        samples = generate_samples_for_query(row, cands, max_neg)
        all_samples.extend(samples)
    
    return pd.DataFrame(all_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--max-negatives", type=int, default=MAX_NEGATIVES)
    parser.add_argument("--seed", type=int, default=SEED)
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
    
    print("\n4. Generating train samples...")
    train_samples = generate_all_samples(train_df, candidates_by_loc, args.max_negatives)
    print(f"   → {len(train_samples)} samples ({train_samples['label'].sum()} pos)")
    
    print("\n5. Generating dev samples...")
    dev_samples = generate_all_samples(dev_df, candidates_by_loc, args.max_negatives)
    print(f"   → {len(dev_samples)} samples ({dev_samples['label'].sum()} pos)")
    
    print("\n6. Generating test samples...")
    test_samples = generate_all_samples(test_df, candidates_by_loc, args.max_negatives)
    print(f"   → {len(test_samples)} samples ({test_samples['label'].sum()} pos)")
    
    train_samples["split"] = "train"
    dev_samples["split"] = "dev"
    test_samples["split"] = "test"
    
    all_samples = pd.concat([train_samples, dev_samples, test_samples], ignore_index=True)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_samples.to_parquet(args.output, index=False)
    print(f"\n7. Saved {len(all_samples)} samples → {args.output}")
    
    meta = {"generated": datetime.now().isoformat(), "seed": args.seed, 
            "train": len(train_samples), "dev": len(dev_samples), "test": len(test_samples)}
    with open(args.output.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Done!")


if __name__ == "__main__":
    main()
