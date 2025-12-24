#!/usr/bin/env python3
"""
SIRET-BERT Evaluation Script

Compares the fine-tuned model against a baseline on Recall@K and MRR metrics.
Includes specific testing for known difficult cases.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.naming import normalize_name
from src.xgb_matcher.semantic import _normalize_for_embedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def preprocess(text: str) -> str:
    """Consistent preprocessing."""
    if not text:
        return ""
    norm = normalize_name(text, uppercase=False)
    return _normalize_for_embedding(norm)


def evaluate(
    model: SentenceTransformer,
    eval_data: List[Dict],
    k_list: List[int] = [1, 5, 10],
    batch_size: int = 32,
) -> Dict[str, float]:
    """Compute Recall@K and MRR."""
    
    anchors = [pair["anchor"] for pair in eval_data]
    positives = [pair["positive"] for pair in eval_data]
    
    # Encode all
    logger.info(f"Encoding {len(anchors)} anchors and positives...")
    anchor_embs = model.encode(anchors, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    positive_embs = model.encode(positives, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    
    # Compute similarity matrix
    # (n_anchors, n_positives)
    # We want to see if the correct positive is in the top K for its anchor
    # MultipleNegativesRankingLoss assumes the i-th positive belongs to the i-th anchor
    
    cos_scores = util.cos_sim(anchor_embs, positive_embs)
    
    metrics = {}
    for k in k_list:
        metrics[f"recall@{k}"] = 0.0
    metrics["mrr"] = 0.0
    
    n = len(anchors)
    for i in range(n):
        scores = cos_scores[i]
        # Sort scores descending
        top_k_indices = torch.topk(scores, k=max(k_list)).indices.tolist()
        
        # Check if i-th positive is in top k
        if i in top_k_indices:
            rank = top_k_indices.index(i) + 1
            metrics["mrr"] += 1.0 / rank
            for k in k_list:
                if rank <= k:
                    metrics[f"recall@{k}"] += 1.0
                    
    for k in k_list:
        metrics[f"recall@{k}"] /= n
    metrics["mrr"] /= n
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SIRET-BERT model")
    parser.add_argument("--model", type=str, required=True, help="Path to model or HuggingFace ID")
    parser.add_argument("--baseline", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--data", type=Path, default=Path("data/semantic_eval.jsonl"))
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load evaluation data
    logger.info(f"Loading evaluation data from {args.data}...")
    eval_data = []
    with open(args.data, "r") as f:
        for i, line in enumerate(f):
            if i >= args.samples:
                break
            eval_data.append(json.loads(line.strip()))
            
    # Load models
    logger.info(f"Loading fine-tuned model from {args.model}...")
    ft_model = SentenceTransformer(args.model, device=device)
    
    logger.info(f"Loading baseline model from {args.baseline}...")
    base_model = SentenceTransformer(args.baseline, device=device)
    
    # Run evaluation
    logger.info("Evaluating Fine-tuned Model...")
    ft_metrics = evaluate(ft_model, eval_data)
    
    logger.info("Evaluating Baseline Model...")
    base_metrics = evaluate(base_model, eval_data)
    
    # Compare
    print("\n" + "=" * 60)
    print(f"{'Metric':<15} | {'Baseline':<15} | {'Fine-tuned':<15} | {'Improvement':<15}")
    print("-" * 60)
    for k in ft_metrics.keys():
        base_val = base_metrics[k]
        ft_val = ft_metrics[k]
        diff = ((ft_val - base_val) / base_val * 100) if base_val > 0 else 0
        print(f"{k:<15} | {base_val:<15.4f} | {ft_val:<15.4f} | {diff:>+7.2f}%")
    print("=" * 60)
    
    # Difficult cases test
    difficult_cases = [
        ("Timcod Rhône-Alpes", "TIMCOD"),
        ("DigitBoxing", "Digit Boxing"),
        ("Site de la Marina (Port de Plaisance)", "COMMUNAUTE URBAINE DE DUNKERQUE"),
    ]
    
    print("\nDifficult Cases (Top-3 similarity):")
    for anchor, pos in difficult_cases:
        a_norm = preprocess(anchor)
        p_norm = preprocess(pos)
        
        # Scores
        base_score = util.cos_sim(base_model.encode(a_norm), base_model.encode(p_norm)).item()
        ft_score = util.cos_sim(ft_model.encode(a_norm), ft_model.encode(p_norm)).item()
        
        print(f"\nQuery: {anchor} ↔ {pos}")
        print(f"  Baseline:   {base_score:.4f}")
        print(f"  Fine-tuned: {ft_score:.4f} ({((ft_score-base_score)/base_score*100):+7.2f}%)")


if __name__ == "__main__":
    main()
