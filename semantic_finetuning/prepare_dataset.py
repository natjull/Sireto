#!/usr/bin/env python3
"""
SIRET-BERT Dataset Preparation V2

Generates training pairs for semantic fine-tuning using the FULL bag-of-names strategy:
1. Positive pairs: CRM name ↔ ALL Official SIRENE names (enseigne1/2/3, denomination, sigle_ul, etc.)  
2. Hard negatives: Names from same commune with high lexical similarity but different SIRET
3. Random negatives: Sampled from mini-batch (via MultipleNegativesRankingLoss)

Uses the same naming strategy as features.py for consistency.

Optimized for MacBook M4 Pro:
- Single parquet scan with batch processing
- Memory-efficient streaming
- Consistent preprocessing with semantic.py
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.naming import (
    normalize_name, 
    normalize_text, 
    build_candidate_names,
    CandidateName,
    NameSource,
)
from src.xgb_matcher.semantic import _normalize_for_embedding


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_OUTPUT = Path("data/semantic_train.jsonl")
DEFAULT_EVAL_OUTPUT = Path("data/semantic_eval.jsonl")
TRAINING_DATA = Path("data/entrainements.csv")
PARQUET_PATH = Path("data/StockEtablissement_utf8.parquet")
UL_PATH = Path("data/StockUniteLegale_utf8.parquet")
HARVEST_DB = Path("data/harvest_full.sqlite")

HARD_NEGATIVES_PER_SAMPLE = 3
MIN_HARD_NEG_JARO = 0.5  # Minimum similarity for hard negative
MAX_HARD_NEG_JARO = 0.95  # Maximum similarity (avoid near-duplicates)
SEED = 42
BATCH_SIZE = 100_000
TRAIN_RATIO = 0.85  # 85% train, 15% eval


# ============================================================================
# Preprocessing (consistent with semantic.py)
# ============================================================================

def preprocess_for_embedding(text: str) -> str:
    """
    Apply the same preprocessing as semantic.py for consistency.
    
    1. Normalize (accents, uppercase, etc.)
    2. CamelCase split, digit separation (from _normalize_for_embedding)
    """
    if not text:
        return ""
    # First normalize
    normalized = normalize_name(text, uppercase=False)
    # Then apply embedding-specific normalization
    return _normalize_for_embedding(normalized)


# ============================================================================
# Full Bag-of-Names Extraction (mirrors features.py/naming.py)
# ============================================================================

def build_candidate_dict_from_row(row: dict) -> dict:
    """
    Build a candidate dict from parquet row with ALL name fields.
    This mirrors what features.py expects.
    """
    return {
        # Etablissement level
        "enseigne1": row.get("enseigne1Etablissement"),
        "enseigne2": row.get("enseigne2Etablissement"),
        "enseigne3": row.get("enseigne3Etablissement"),
        "denomination": row.get("denominationUsuelleEtablissement"),
        # Unite Legale level  
        "sigle_ul": row.get("sigleUniteLegale"),
        "denomination_usuelle_ul": row.get("denominationUsuelle1UniteLegale"),
        "denomination_ul": row.get("denominationUniteLegale"),
        # Person (for EI)
        "prenom_usuel_ul": row.get("prenomUsuelUniteLegale"),
        "nom_ul": row.get("nomUniteLegale"),
        "cj_ul": row.get("categorieJuridiqueUniteLegale"),
        # (PM dirigeant names would come from harvest_full.sqlite - added separately)
        "pm_dirigeant_names": row.get("pm_dirigeant_names", []),
    }


def extract_all_names(cand_dict: dict) -> List[str]:
    """
    Extract ALL names using build_candidate_names from naming.py.
    Returns preprocessed names ready for embedding.
    """
    names = build_candidate_names(cand_dict)
    result = []
    for nm in names:
        preprocessed = preprocess_for_embedding(nm.text)
        if preprocessed and len(preprocessed) > 2:
            result.append(preprocessed)
    return result


def extract_best_name(cand_dict: dict) -> str:
    """
    Extract the best (primary) name using naming.py logic.
    """
    names = build_candidate_names(cand_dict)
    if not names:
        return ""
    # Use same priority as naming.py
    priority_sources = [
        NameSource.ETAB_ENSEIGNE,
        NameSource.ETAB_DENOM,
        NameSource.ETAB_ENSEIGNE_X,
        NameSource.UL_SIGLE,
        NameSource.UL_DENOM_USUELLE,
        NameSource.UL_DENOM,
        NameSource.PM_DIRIGEANT,
        NameSource.PERSON_NAME,
    ]
    for source in priority_sources:
        for nm in names:
            if nm.source == source:
                return preprocess_for_embedding(nm.text)
    return preprocess_for_embedding(names[0].text) if names else ""


# ============================================================================
# Data Loading
# ============================================================================

def load_training_data(path: Path) -> pd.DataFrame:
    """Load and clean CRM training data."""
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    df = df.rename(columns={
        "SITE": "crm_name",
        "SITE_CLI_ADRESSE": "crm_address",
        "SITE_CLI_COMMUNE": "crm_city",
        "CODE_POSTAL": "postcode",
        "CODE_INSEE": "insee",
        "SIRET": "ground_truth_siret",
    })
    # Filter valid SIRETs
    df = df[df["ground_truth_siret"].notna() & (df["ground_truth_siret"].str.len() == 14)]
    df["siren"] = df["ground_truth_siret"].str[:9]
    df["crm_id"] = range(len(df))
    return df


def load_ul_lookup() -> Dict[str, dict]:
    """Load Unite Legale data indexed by SIREN."""
    print("  Loading Unite Legale data...")
    ul_cols = [
        "siren",
        "sigleUniteLegale",
        "denominationUniteLegale", 
        "denominationUsuelle1UniteLegale",
        "prenomUsuelUniteLegale",
        "nomUniteLegale",
        "categorieJuridiqueUniteLegale",
    ]
    
    ul_lookup = {}
    pf = pq.ParquetFile(UL_PATH)
    
    for batch in tqdm(pf.iter_batches(columns=ul_cols, batch_size=BATCH_SIZE), 
                      desc="  UL scan"):
        pdf = batch.to_pandas()
        for _, row in pdf.iterrows():
            siren = row.get("siren")
            if siren:
                ul_lookup[siren] = {
                    "sigleUniteLegale": row.get("sigleUniteLegale"),
                    "denominationUniteLegale": row.get("denominationUniteLegale"),
                    "denominationUsuelle1UniteLegale": row.get("denominationUsuelle1UniteLegale"),
                    "prenomUsuelUniteLegale": row.get("prenomUsuelUniteLegale"),
                    "nomUniteLegale": row.get("nomUniteLegale"),
                    "categorieJuridiqueUniteLegale": row.get("categorieJuridiqueUniteLegale"),
                }
    
    print(f"  → Loaded {len(ul_lookup)} Unite Legale records")
    return ul_lookup


def load_pm_dirigeant_lookup() -> Dict[str, List[str]]:
    """Load PM dirigeant names from harvest_full.sqlite."""
    if not HARVEST_DB.exists():
        print("  Warning: harvest_full.sqlite not found, skipping PM dirigeant names")
        return {}
    
    print("  Loading PM dirigeant names...")
    pm_lookup: Dict[str, List[str]] = defaultdict(list)
    
    try:
        conn = sqlite3.connect(str(HARVEST_DB))
        cursor = conn.cursor()
        
        # Get PM dirigeant names (personne morale)
        cursor.execute("""
            SELECT siren, nom 
            FROM dirigeants 
            WHERE type_personne = 'PM' AND nom IS NOT NULL AND nom != ''
        """)
        
        for siren, nom in cursor.fetchall():
            if siren and nom:
                pm_lookup[siren].append(nom)
        
        conn.close()
        print(f"  → Loaded PM dirigeant names for {len(pm_lookup)} SIRENs")
    except Exception as e:
        print(f"  Warning: Failed to load PM dirigeant: {e}")
    
    return pm_lookup


def load_sirene_data_full(
    sirets: Set[str],
    ul_lookup: Dict[str, dict],
    pm_lookup: Dict[str, List[str]],
) -> Dict[str, dict]:
    """
    Load SIRENE data with FULL bag-of-names for the given SIRETs.
    Enriches with UL and PM dirigeant data.
    """
    print("  Loading SIRENE etablissement data...")
    
    etab_cols = [
        "siret", "siren",
        "enseigne1Etablissement", 
        "enseigne2Etablissement",
        "enseigne3Etablissement",
        "denominationUsuelleEtablissement",
    ]
    
    lookup = {}
    pf = pq.ParquetFile(PARQUET_PATH)
    
    for batch in tqdm(pf.iter_batches(columns=etab_cols, batch_size=BATCH_SIZE), 
                      desc="  SIRENE scan"):
        pdf = batch.to_pandas()
        mask = pdf["siret"].isin(sirets)
        
        for _, row in pdf[mask].iterrows():
            siret = row["siret"]
            siren = row.get("siren", siret[:9])
            
            # Start with etablissement data
            data = {
                "siret": siret,
                "siren": siren,
                "enseigne1Etablissement": row.get("enseigne1Etablissement"),
                "enseigne2Etablissement": row.get("enseigne2Etablissement"),
                "enseigne3Etablissement": row.get("enseigne3Etablissement"),
                "denominationUsuelleEtablissement": row.get("denominationUsuelleEtablissement"),
            }
            
            # Enrich with UL data
            if siren in ul_lookup:
                data.update(ul_lookup[siren])
            
            # Enrich with PM dirigeant
            if siren in pm_lookup:
                data["pm_dirigeant_names"] = pm_lookup[siren]
            
            lookup[siret] = data
    
    print(f"  → Loaded full data for {len(lookup)}/{len(sirets)} SIRETs")
    return lookup


def load_candidates_by_commune(
    insee_codes: Set[str],
    ul_lookup: Dict[str, dict],
    pm_lookup: Dict[str, List[str]],
) -> Dict[str, List[Dict]]:
    """
    Load ALL candidates by commune with FULL bag-of-names.
    Single parquet scan - memory efficient.
    """
    etab_cols = [
        "siret", "siren",
        "enseigne1Etablissement", 
        "enseigne2Etablissement",
        "enseigne3Etablissement",
        "denominationUsuelleEtablissement",
        "codeCommuneEtablissement",
        "etatAdministratifEtablissement",
    ]
    
    by_commune: Dict[str, List[Dict]] = defaultdict(list)
    pf = pq.ParquetFile(PARQUET_PATH)
    
    for batch in tqdm(pf.iter_batches(columns=etab_cols, batch_size=BATCH_SIZE),
                      desc="  Building commune index"):
        pdf = batch.to_pandas()
        
        # Filter to relevant communes + active only
        mask = (
            pdf["codeCommuneEtablissement"].isin(insee_codes) &
            (pdf["etatAdministratifEtablissement"] != "F")
        )
        
        for _, row in pdf[mask].iterrows():
            siren = row.get("siren", row["siret"][:9])
            
            # Build full candidate dict
            data = {
                "siret": row["siret"],
                "siren": siren,
                "enseigne1Etablissement": row.get("enseigne1Etablissement"),
                "enseigne2Etablissement": row.get("enseigne2Etablissement"),
                "enseigne3Etablissement": row.get("enseigne3Etablissement"),
                "denominationUsuelleEtablissement": row.get("denominationUsuelleEtablissement"),
            }
            
            # Enrich with UL data
            if siren in ul_lookup:
                data.update(ul_lookup[siren])
            
            # Enrich with PM dirigeant
            if siren in pm_lookup:
                data["pm_dirigeant_names"] = pm_lookup[siren]
            
            # Build candidate dict for naming.py
            cand_dict = build_candidate_dict_from_row(data)
            all_names = extract_all_names(cand_dict)
            best_name = extract_best_name(cand_dict)
            
            if not all_names and not best_name:
                continue
            
            commune = row["codeCommuneEtablissement"]
            by_commune[commune].append({
                "siret": row["siret"],
                "siren": siren,
                "all_names": all_names,
                "best_name": best_name,
            })
    
    return by_commune


# ============================================================================
# Hard Negative Mining
# ============================================================================

def find_hard_negatives(
    crm_name: str,
    crm_siret: str,
    commune_candidates: List[Dict],
    n: int = HARD_NEGATIVES_PER_SAMPLE,
) -> List[Dict]:
    """
    Find hard negatives: high lexical similarity but wrong SIRET.
    Uses the BEST name from each candidate for comparison.
    """
    if not commune_candidates:
        return []
    
    crm_normalized = preprocess_for_embedding(crm_name)
    if not crm_normalized:
        return []
    
    scored = []
    for cand in commune_candidates:
        if cand["siret"] == crm_siret:
            continue  # Skip the ground truth
        
        best_name = cand.get("best_name", "")
        if not best_name:
            continue
            
        # Compute Jaro-Winkler similarity
        sim = JaroWinkler.similarity(crm_normalized, best_name)
        
        # Filter by similarity range
        if MIN_HARD_NEG_JARO <= sim <= MAX_HARD_NEG_JARO:
            scored.append((sim, cand))
    
    # Sort by similarity (descending) and take top n
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cand for _, cand in scored[:n]]


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_training_pairs(
    crm_df: pd.DataFrame,
    sirene_lookup: Dict[str, dict],
    candidates_by_commune: Dict[str, List[Dict]],
    n_hard_negatives: int = HARD_NEGATIVES_PER_SAMPLE,
) -> List[Dict]:
    """
    Generate training pairs using FULL bag-of-names.
    
    For each CRM query, creates multiple positive pairs:
    - anchor (CRM name) paired with EACH SIRENE name (enseigne1, enseigne2, sigle, etc.)
    
    Output format (each line of JSONL):
    {
        "anchor": "normalized CRM name",
        "positive": "normalized SIRENE name",
        "negative": "hard negative name" (optional)
    }
    """
    pairs = []
    
    for _, row in tqdm(crm_df.iterrows(), total=len(crm_df), desc="Generating pairs"):
        crm_name = row["crm_name"]
        siret = row["ground_truth_siret"]
        insee = row.get("insee", "")
        
        # Get the full SIRENE data
        sirene_data = sirene_lookup.get(siret)
        if not sirene_data:
            continue
        
        # Build candidate dict and extract ALL names
        cand_dict = build_candidate_dict_from_row(sirene_data)
        all_sirene_names = extract_all_names(cand_dict)
        
        if not all_sirene_names:
            continue
        
        # Preprocess CRM name (anchor)
        anchor = preprocess_for_embedding(crm_name)
        if not anchor:
            continue
        
        # Find hard negatives from the same commune
        commune_candidates = candidates_by_commune.get(insee, [])
        hard_negatives = find_hard_negatives(
            crm_name, siret, commune_candidates, n=n_hard_negatives
        )
        
        # Create pairs for EACH SIRENE name (bag-of-names approach)
        for sirene_name in all_sirene_names:
            if not sirene_name:
                continue
                
            if hard_negatives:
                # With hard negatives (triplet style)
                for neg in hard_negatives:
                    # Use a random name from the hard negative's bag
                    neg_names = neg.get("all_names", [])
                    neg_name = random.choice(neg_names) if neg_names else neg.get("best_name", "")
                    if neg_name:
                        pairs.append({
                            "anchor": anchor,
                            "positive": sirene_name,
                            "negative": neg_name,
                        })
            else:
                # Without hard negatives (pair style)
                pairs.append({
                    "anchor": anchor,
                    "positive": sirene_name,
                })
    
    return pairs


def split_by_siren(
    crm_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by SIREN to avoid data leakage.
    Different establishments of the same company should be in the same split.
    """
    sirens = crm_df["siren"].unique().tolist()
    random.seed(seed)
    random.shuffle(sirens)
    
    n_train = int(len(sirens) * train_ratio)
    train_sirens = set(sirens[:n_train])
    eval_sirens = set(sirens[n_train:])
    
    train_df = crm_df[crm_df["siren"].isin(train_sirens)]
    eval_df = crm_df[crm_df["siren"].isin(eval_sirens)]
    
    return train_df, eval_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for semantic fine-tuning")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--hard-negatives", type=int, default=HARD_NEGATIVES_PER_SAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("SIRET-BERT Dataset Preparation V2 (Full Bag-of-Names)")
    print("=" * 60)
    
    # 1. Load CRM data
    print("\n1. Loading CRM training data...")
    crm_df = load_training_data(TRAINING_DATA)
    print(f"   → {len(crm_df)} CRM queries, {crm_df['siren'].nunique()} unique SIRENs")
    
    # 2. Split by SIREN
    print("\n2. Splitting by SIREN...")
    train_df, eval_df = split_by_siren(crm_df, args.train_ratio, args.seed)
    print(f"   → Train: {len(train_df)} queries, Eval: {len(eval_df)} queries")
    
    # 3. Load Unite Legale data
    print("\n3. Loading enrichment data...")
    ul_lookup = load_ul_lookup()
    pm_lookup = load_pm_dirigeant_lookup()
    
    # 4. Load full SIRENE data for ground truth SIRETs
    print("\n4. Loading SIRENE data for ground truth...")
    all_sirets = set(crm_df["ground_truth_siret"])
    sirene_lookup = load_sirene_data_full(all_sirets, ul_lookup, pm_lookup)
    
    # 5. Load candidates by commune for hard negative mining
    print("\n5. Loading candidates by commune (single scan)...")
    insee_codes = set(crm_df["insee"].dropna())
    print(f"   → {len(insee_codes)} unique INSEE codes")
    candidates_by_commune = load_candidates_by_commune(insee_codes, ul_lookup, pm_lookup)
    total_candidates = sum(len(v) for v in candidates_by_commune.values())
    print(f"   → {total_candidates} candidates across {len(candidates_by_commune)} communes")
    
    # 6. Generate training pairs
    print("\n6. Generating training pairs (bag-of-names)...")
    train_pairs = generate_training_pairs(
        train_df, sirene_lookup, candidates_by_commune, args.hard_negatives
    )
    print(f"   → {len(train_pairs)} training pairs")
    
    # 7. Generate eval pairs
    print("\n7. Generating eval pairs...")
    eval_pairs = generate_training_pairs(
        eval_df, sirene_lookup, candidates_by_commune, args.hard_negatives
    )
    print(f"   → {len(eval_pairs)} eval pairs")
    
    # 8. Save to JSONL
    print("\n8. Saving datasets...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"   → Train: {args.output}")
    
    with open(args.eval_output, "w", encoding="utf-8") as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"   → Eval: {args.eval_output}")
    
    # 9. Save metadata
    meta = {
        "generated": datetime.now().isoformat(),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "hard_negatives_per_sample": args.hard_negatives,
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "unique_sirens_train": train_df["siren"].nunique(),
        "unique_sirens_eval": eval_df["siren"].nunique(),
        "ul_records": len(ul_lookup),
        "pm_records": len(pm_lookup),
        "communes": len(candidates_by_commune),
        "strategy": "full_bag_of_names",
    }
    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   → Metadata: {meta_path}")
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    
    # Show sample pairs
    print("\nSample pairs:")
    for pair in train_pairs[:5]:
        print(f"  anchor:   {pair['anchor'][:50]}...")
        print(f"  positive: {pair['positive'][:50]}...")
        if "negative" in pair:
            print(f"  negative: {pair['negative'][:50]}...")
        print()


if __name__ == "__main__":
    main()
