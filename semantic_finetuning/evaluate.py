#!/usr/bin/env python3
"""
SIRET-BERT Evaluation Script

Evaluates fine-tuned models using:
1. Recall@K (K=1, 5, 10)
2. Mean Reciprocal Rank (MRR)
3. Hit Rate
4. Comparison with baseline model

Tests on known difficult cases from the project.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.semantic import _normalize_for_embedding
from src.xgb_matcher.naming import normalize_name


# ============================================================================
# Configuration  
# ============================================================================

TRAINING_DATA = Path("data/entrainements.csv")
PARQUET_PATH = Path("data/StockEtablissement_utf8.parquet")
BASELINE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 100_000

# Known difficult cases from the project
DIFFICULT_CASES = [
    {"crm_name": "Timcod Rhône-Alpes", "target_siret": "48048580400037"},
    {"crm_name": "DigitBoxing and coall", "target_siret": "92962682200017"},
    {"crm_name": "Groupe ADF", "target_siret": None},  # Will find from data
]


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_for_embedding(text: str) -> str:
    """Apply the same preprocessing as training."""
    if not text:
        return ""
    normalized = normalize_name(text, uppercase=False)
    return _normalize_for_embedding(normalized)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_recall_at_k(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    ground_truth_indices: List[int],
    k: int,
) -> float:
    """
    Compute Recall@K: fraction of queries where correct answer is in top-K.
    """
    hits = 0
    
    # Compute all similarities
    similarities = np.dot(query_embeddings, candidate_embeddings.T)
    
    for i, gt_idx in enumerate(ground_truth_indices):
        if gt_idx < 0:
            continue  # Skip if no ground truth
            
        # Get top-K indices
        top_k_indices = np.argsort(similarities[i])[-k:][::-1]
        
        if gt_idx in top_k_indices:
            hits += 1
    
    n_valid = sum(1 for idx in ground_truth_indices if idx >= 0)
    return hits / n_valid if n_valid > 0 else 0.0


def compute_mrr(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    ground_truth_indices: List[int],
) -> float:
    """
    Compute Mean Reciprocal Rank.
    MRR = average of 1/rank for each query.
    """
    reciprocal_ranks = []
    
    similarities = np.dot(query_embeddings, candidate_embeddings.T)
    
    for i, gt_idx in enumerate(ground_truth_indices):
        if gt_idx < 0:
            continue
            
        # Get sorted indices (highest similarity first)
        sorted_indices = np.argsort(similarities[i])[::-1]
        
        # Find the rank of the ground truth
        rank = np.where(sorted_indices == gt_idx)[0]
        if len(rank) > 0:
            reciprocal_ranks.append(1.0 / (rank[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


# ============================================================================
# Evaluation Pipeline
# ============================================================================

class Evaluator:
    """Evaluates a sentence transformer on the SIRET matching task."""
    
    def __init__(self, model_path: str, device: str = "mps"):
        self.device = device
        print(f"Loading model: {model_path}")
        self.model = SentenceTransformer(model_path, device=device)
        
    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of texts."""
        preprocessed = [preprocess_for_embedding(t) for t in texts]
        embeddings = self.model.encode(
            preprocessed,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings
    
    def evaluate_commune(
        self,
        crm_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Evaluate on a per-commune basis.
        For each query, retrieval is done against candidates from the same commune.
        """
        results = {
            "recall@1": [],
            "recall@5": [],
            "recall@10": [],
            "mrr": [],
        }
        
        # Group by commune
        crm_by_commune = crm_df.groupby("insee")
        cand_by_commune = candidates_df.groupby("insee")
        
        communes = set(crm_df["insee"].unique()) & set(candidates_df["insee"].unique())
        
        for insee in tqdm(communes, desc="Evaluating communes"):
            commune_crm = crm_by_commune.get_group(insee)
            commune_cands = cand_by_commune.get_group(insee)
            
            if len(commune_cands) < 2:
                continue
            
            # Embed everything
            query_texts = commune_crm["crm_name"].tolist()
            cand_texts = commune_cands["sirene_name"].tolist()
            cand_sirets = commune_cands["siret"].tolist()
            
            query_embs = self.embed_texts(query_texts, batch_size=32)
            cand_embs = self.embed_texts(cand_texts, batch_size=32)
            
            # Ground truth indices
            gt_sirets = commune_crm["ground_truth_siret"].tolist()
            gt_indices = []
            for gt_siret in gt_sirets:
                try:
                    idx = cand_sirets.index(gt_siret)
                except ValueError:
                    idx = -1
                gt_indices.append(idx)
            
            # Compute metrics
            if any(idx >= 0 for idx in gt_indices):
                results["recall@1"].append(compute_recall_at_k(query_embs, cand_embs, gt_indices, 1))
                results["recall@5"].append(compute_recall_at_k(query_embs, cand_embs, gt_indices, 5))
                results["recall@10"].append(compute_recall_at_k(query_embs, cand_embs, gt_indices, 10))
                results["mrr"].append(compute_mrr(query_embs, cand_embs, gt_indices))
        
        # Average across communes
        return {
            k: np.mean(v) if v else 0.0 
            for k, v in results.items()
        }
    
    def evaluate_difficult_cases(self, candidates_df: pd.DataFrame) -> Dict[str, any]:
        """Evaluate on known difficult cases."""
        results = []
        
        for case in DIFFICULT_CASES:
            crm_name = case["crm_name"]
            target_siret = case.get("target_siret")
            
            if not target_siret:
                continue
            
            # Get candidates from same commune (if we have the info)
            # For now, search everywhere
            query_emb = self.embed_texts([crm_name])[0:1]
            
            # Find the target in candidates
            mask = candidates_df["siret"] == target_siret
            if not mask.any():
                print(f"  Warning: {target_siret} not found in candidates")
                continue
            
            target_row = candidates_df[mask].iloc[0]
            target_insee = target_row.get("insee", "")
            
            # Get commune candidates
            if target_insee:
                commune_cands = candidates_df[candidates_df["insee"] == target_insee]
            else:
                commune_cands = candidates_df.sample(min(1000, len(candidates_df)))
            
            cand_texts = commune_cands["sirene_name"].tolist()
            cand_sirets = commune_cands["siret"].tolist()
            cand_embs = self.embed_texts(cand_texts, batch_size=32)
            
            # Find rank of target
            similarities = np.dot(query_emb, cand_embs.T)[0]
            sorted_indices = np.argsort(similarities)[::-1]
            
            try:
                target_idx = cand_sirets.index(target_siret)
                rank = np.where(sorted_indices == target_idx)[0][0] + 1
            except (ValueError, IndexError):
                rank = -1
            
            results.append({
                "crm_name": crm_name,
                "target_siret": target_siret,
                "rank": rank,
                "top_5_in": rank <= 5 if rank > 0 else False,
                "similarity": similarities[target_idx] if rank > 0 else 0.0,
            })
            
            print(f"  {crm_name}: rank={rank}, sim={similarities[target_idx] if rank > 0 else 0:.4f}")
        
        return results


# ============================================================================
# Data Loading
# ============================================================================

def load_evaluation_data(
    sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CRM data and SIRENE candidates for evaluation."""
    
    # Load CRM data
    print("Loading CRM data...")
    crm_df = pd.read_csv(TRAINING_DATA, sep=";", dtype=str, encoding="utf-8-sig")
    crm_df = crm_df.rename(columns={
        "SITE": "crm_name",
        "CODE_POSTAL": "postcode",
        "CODE_INSEE": "insee",
        "SIRET": "ground_truth_siret",
    })
    crm_df = crm_df[crm_df["ground_truth_siret"].notna() & (crm_df["ground_truth_siret"].str.len() == 14)]
    
    if sample_size:
        crm_df = crm_df.sample(min(sample_size, len(crm_df)), random_state=42)
    
    print(f"  {len(crm_df)} CRM queries")
    
    # Load SIRENE candidates from relevant communes
    print("Loading SIRENE candidates...")
    insee_codes = set(crm_df["insee"].dropna())
    
    cols = [
        "siret",
        "enseigne1Etablissement",
        "denominationUsuelleEtablissement", 
        "codeCommuneEtablissement",
        "etatAdministratifEtablissement",
    ]
    
    candidates = []
    pf = pq.ParquetFile(PARQUET_PATH)
    
    for batch in tqdm(pf.iter_batches(columns=cols, batch_size=BATCH_SIZE), desc="Scanning SIRENE"):
        pdf = batch.to_pandas()
        mask = (
            pdf["codeCommuneEtablissement"].isin(insee_codes) &
            (pdf["etatAdministratifEtablissement"] != "F")
        )
        
        for _, row in pdf[mask].iterrows():
            name = row.get("enseigne1Etablissement") or row.get("denominationUsuelleEtablissement")
            if name and str(name).strip() and str(name) != "nan":
                candidates.append({
                    "siret": row["siret"],
                    "insee": row["codeCommuneEtablissement"],
                    "sirene_name": str(name).strip(),
                })
    
    candidates_df = pd.DataFrame(candidates)
    print(f"  {len(candidates_df)} SIRENE candidates")
    
    return crm_df, candidates_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SIRET-BERT model")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--baseline", type=str, default=BASELINE_MODEL, help="Baseline model for comparison")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of queries to evaluate")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--output", type=Path, default=Path("reports/semantic_eval.json"))
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIRET-BERT Evaluation")
    print("=" * 60)
    
    # Load data
    crm_df, candidates_df = load_evaluation_data(args.sample_size)
    
    # Evaluate fine-tuned model
    print(f"\n1. Evaluating fine-tuned model: {args.model}")
    finetuned_eval = Evaluator(args.model, args.device)
    finetuned_metrics = finetuned_eval.evaluate_commune(crm_df, candidates_df)
    
    print("\n   Results:")
    for k, v in finetuned_metrics.items():
        print(f"     {k}: {v:.4f}")
    
    # Evaluate baseline
    print(f"\n2. Evaluating baseline model: {args.baseline}")
    baseline_eval = Evaluator(args.baseline, args.device)
    baseline_metrics = baseline_eval.evaluate_commune(crm_df, candidates_df)
    
    print("\n   Results:")
    for k, v in baseline_metrics.items():
        print(f"     {k}: {v:.4f}")
    
    # Comparison
    print("\n3. Comparison (Fine-tuned vs Baseline):")
    for k in finetuned_metrics:
        diff = finetuned_metrics[k] - baseline_metrics[k]
        symbol = "+" if diff > 0 else ""
        print(f"     {k}: {symbol}{diff:.4f}")
    
    # Difficult cases
    print("\n4. Difficult cases (Fine-tuned model):")
    difficult_results = finetuned_eval.evaluate_difficult_cases(candidates_df)
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "baseline": args.baseline,
        "sample_size": args.sample_size,
        "finetuned_metrics": finetuned_metrics,
        "baseline_metrics": baseline_metrics,
        "difficult_cases": difficult_results,
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
