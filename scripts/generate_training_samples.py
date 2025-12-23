#!/usr/bin/env python3
"""
Generate training samples with realistic candidate pool.

This script aligns training data generation with inference reality:
- Uses the same candidate pooling as inference (all SIRET in CP/INSEE)
- Generates hard negatives from ranker top-K (excluding ground truth)
- Random negatives from the pool for diversity
- Split by SIREN to avoid entity leakage

Usage:
    python scripts/generate_training_samples.py [--output data/samples_aligned.parquet]
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
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.candidates import (
    load_candidates_for_locations,
    get_candidates_for_query,
    build_location_index,
)
from src.xgb_matcher.features import FEATURE_NAMES, make_features
from src.xgb_matcher.naming import primary_name

# Configuration
DEFAULT_OUTPUT = Path("data/samples_aligned.parquet")
DEFAULT_SPLITS_DIR = Path("data/splits")
TRAINING_DATA = Path("data/entrainements.csv")

# Sampling parameters
MAX_NEGATIVES_PER_QUERY = 200  # Total negatives per query
HARD_NEGATIVE_RATIO = 0.5     # Ratio of hard negatives vs random
RANDOM_SEED = 42


def load_training_data(path: Path) -> pd.DataFrame:
    """Load training data with ground truth SIRET."""
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    
    # Map columns to standard names
    df = df.rename(columns={
        "SITE": "crm_name",
        "SITE_CLI_ADRESSE": "crm_address",
        "SITE_CLI_COMMUNE": "crm_city",
        "CODE_POSTAL": "postcode",
        "CODE_INSEE": "insee",
        "SIRET": "ground_truth_siret",
    })
    
    # Filter rows with valid SIRET ground truth
    df = df[df["ground_truth_siret"].notna() & (df["ground_truth_siret"].str.len() == 14)]
    df["crm_id"] = df.index
    
    # Extract SIREN for grouping
    df["siren"] = df["ground_truth_siret"].str[:9]
    
    return df


def create_siren_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by SIREN to avoid entity leakage.
    
    All rows with the same SIREN go to the same split.
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 0.001
    
    # Get unique SIRENs
    sirens = df["siren"].unique().tolist()
    random.seed(seed)
    random.shuffle(sirens)
    
    # Calculate split points
    n = len(sirens)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))
    
    train_sirens = set(sirens[:train_end])
    dev_sirens = set(sirens[train_end:dev_end])
    test_sirens = set(sirens[dev_end:])
    
    train_df = df[df["siren"].isin(train_sirens)].copy()
    dev_df = df[df["siren"].isin(dev_sirens)].copy()
    test_df = df[df["siren"].isin(test_sirens)].copy()
    
    print(f"Split by SIREN:")
    print(f"  Train: {len(train_df)} rows ({len(train_sirens)} SIRENs)")
    print(f"  Dev:   {len(dev_df)} rows ({len(dev_sirens)} SIRENs)")
    print(f"  Test:  {len(test_df)} rows ({len(test_sirens)} SIRENs)")
    
    return train_df, dev_df, test_df


def compute_simple_score(features: Dict) -> float:
    """
    Simple heuristic score for hard negative selection.
    Combines name and address similarities.
    """
    name_score = features.get("name_jaro_max", 0.0)
    addr_score = features.get("addr_jaro", 0.0)
    return 0.6 * name_score + 0.4 * addr_score


def generate_samples_for_query(
    crm_row: pd.Series,
    candidates: Dict[str, dict],
    candidate_index: Optional[Dict[str, Dict[str, List[Tuple[str, dict]]]]] = None,
    max_negatives: int = MAX_NEGATIVES_PER_QUERY,
    hard_negative_ratio: float = HARD_NEGATIVE_RATIO,
) -> List[Dict]:
    """
    Generate training samples for a single CRM query.
    
    Returns list of {features..., label, query_id} dicts.
    """
    ground_truth = crm_row.get("ground_truth_siret", "")
    postcode = crm_row.get("postcode", "")
    insee = crm_row.get("insee", "")
    query_id = crm_row.name  # Use index as query_id
    
    # Get candidates for this location
    cand_list = get_candidates_for_query(postcode, insee, candidates, index=candidate_index)
    
    if not cand_list:
        return []
    
    # Compute features for all candidates
    all_features = []
    for siret, cand in cand_list:
        feat = make_features(crm_row, cand)
        feat["_siret"] = siret
        feat["_is_positive"] = (siret == ground_truth)
        feat["_score"] = compute_simple_score(feat)
        all_features.append(feat)
    
    # Separate positive and negatives
    positives = [f for f in all_features if f["_is_positive"]]
    negatives = [f for f in all_features if not f["_is_positive"]]
    
    if not positives:
        # Ground truth not in pool - skip this query
        return []
    
    # Sample negatives: mix of hard and random
    num_hard = int(max_negatives * hard_negative_ratio)
    num_random = max_negatives - num_hard
    
    # Hard negatives: highest scoring non-positives
    negatives_sorted = sorted(negatives, key=lambda x: x["_score"], reverse=True)
    hard_negs = negatives_sorted[:num_hard]
    
    # Random negatives from the rest
    remaining = negatives_sorted[num_hard:]
    if remaining:
        random_negs = random.sample(remaining, min(num_random, len(remaining)))
    else:
        random_negs = []
    
    selected_negatives = hard_negs + random_negs
    
    # Build samples
    samples = []
    
    # Add positive(s)
    for pos in positives:
        sample = {fn: pos.get(fn, 0.0) for fn in FEATURE_NAMES}
        sample["label"] = 1
        sample["query_id"] = query_id
        sample["siret"] = pos["_siret"]
        samples.append(sample)
    
    # Add negatives
    for neg in selected_negatives:
        sample = {fn: neg.get(fn, 0.0) for fn in FEATURE_NAMES}
        sample["label"] = 0
        sample["query_id"] = query_id
        sample["siret"] = neg["_siret"]
        samples.append(sample)
    
    return samples


def generate_all_samples(
    df: pd.DataFrame,
    candidates: Dict[str, dict],
    candidate_index: Optional[Dict[str, Dict[str, List[Tuple[str, dict]]]]] = None,
    max_negatives: int = MAX_NEGATIVES_PER_QUERY,
) -> pd.DataFrame:
    """Generate training samples for all queries in dataframe."""
    all_samples = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating samples"):
        samples = generate_samples_for_query(row, candidates, candidate_index, max_negatives)
        all_samples.extend(samples)
    
    return pd.DataFrame(all_samples)


def main():
    parser = argparse.ArgumentParser(description="Generate aligned training samples")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--max-negatives", type=int, default=MAX_NEGATIVES_PER_QUERY)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Training Sample Generation (Aligned with Inference)")
    print("=" * 60)
    
    # Load training data
    print(f"\n1. Loading training data from {TRAINING_DATA}...")
    df = load_training_data(TRAINING_DATA)
    print(f"   Total rows with valid SIRET: {len(df)}")
    print(f"   Unique SIRENs: {df['siren'].nunique()}")
    
    # Create SIREN-based split
    print("\n2. Creating SIREN-based split...")
    train_df, dev_df, test_df = create_siren_split(df, seed=args.seed)
    
    # Save splits
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.splits_dir / "train.csv", index=False)
    dev_df.to_csv(args.splits_dir / "dev.csv", index=False)
    test_df.to_csv(args.splits_dir / "test.csv", index=False)
    print(f"   Saved splits to {args.splits_dir}/")
    
    # Build candidate pool
    all_postcodes = set(df["postcode"].dropna().unique())
    all_insee = set(df["insee"].dropna().unique())
    print(f"\n3. Building candidate pool...")
    print(f"   Postcodes: {len(all_postcodes)}, INSEE codes: {len(all_insee)}")
    
    candidates = load_candidates_for_locations(all_postcodes, all_insee)
    print(f"   Candidates loaded: {len(candidates)}")
    candidate_index = build_location_index(candidates)
    
    # Generate samples for each split
    print("\n4. Generating training samples...")
    train_samples = generate_all_samples(train_df, candidates, candidate_index, args.max_negatives)
    print(f"   Train: {len(train_samples)} samples ({train_samples['label'].sum()} positives)")
    
    print("\n5. Generating dev samples...")
    dev_samples = generate_all_samples(dev_df, candidates, candidate_index, args.max_negatives)
    print(f"   Dev: {len(dev_samples)} samples ({dev_samples['label'].sum()} positives)")
    
    print("\n6. Generating test samples...")
    test_samples = generate_all_samples(test_df, candidates, candidate_index, args.max_negatives)
    print(f"   Test: {len(test_samples)} samples ({test_samples['label'].sum()} positives)")
    
    # Save samples
    train_samples["split"] = "train"
    dev_samples["split"] = "dev"
    test_samples["split"] = "test"
    
    all_samples = pd.concat([train_samples, dev_samples, test_samples], ignore_index=True)
    
    # Save as parquet
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_samples.to_parquet(args.output, index=False)
    print(f"\n7. Saved {len(all_samples)} samples to {args.output}")
    
    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "max_negatives": args.max_negatives,
        "hard_negative_ratio": HARD_NEGATIVE_RATIO,
        "feature_names": FEATURE_NAMES,
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "test_samples": len(test_samples),
        "train_positives": int(train_samples["label"].sum()),
        "dev_positives": int(dev_samples["label"].sum()),
        "test_positives": int(test_samples["label"].sum()),
    }
    meta_path = args.output.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved metadata to {meta_path}")
    
    print("\n" + "=" * 60)
    print("✅ Sample generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
