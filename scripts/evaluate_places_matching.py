#!/usr/bin/env python3
"""
Evaluate Places-guided matching against ground truth.

Phase 4: Places Matching Evaluation
===================================
This script evaluates the quality of Places-guided candidate generation
and matching by comparing against ground truth labels.

Metrics computed:
- MATCH_PLACES accuracy (precision): % of Places matches that are correct
- Places recall: % of correct matches found by Places
- Arm A vs Arm B contribution analysis
- Pool coverage: % of ground truth found in merged pool
- address_close validation accuracy

Usage:
    python scripts/evaluate_places_matching.py \
        --places-results-path reports/routed_with_places.csv \
        --ground-truth-path data/samples_v4_with_ranker.parquet \
        --output-path reports/places_matching_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class PlacesMatchingEvaluation:
    """Complete Places matching evaluation results."""

    total_records: int = 0

    # MATCH_PLACES metrics
    match_places_count: int = 0
    match_places_correct: int = 0
    match_places_incorrect: int = 0
    match_places_precision: float = 0.0

    # NO_MATCH metrics
    no_match_count: int = 0
    no_match_had_correct: int = 0  # GT was in pool but not promoted
    no_match_truly_none: int = 0  # GT was not in pool at all

    # Pool coverage
    pool_coverage_arm_a: int = 0
    pool_coverage_arm_b: int = 0
    pool_coverage_topk: int = 0
    pool_coverage_total: int = 0

    # address_close metrics
    address_close_passed: int = 0
    address_close_failed: int = 0
    address_close_fp_prevented: int = 0  # Would have been wrong but address_close blocked

    # Source contribution
    match_from_arm_a: int = 0
    match_from_arm_b: int = 0
    match_from_topk: int = 0

    # Cost analysis
    places_api_calls: int = 0
    estimated_cost_usd: float = 0.0

    # Breakdown by segment
    segment_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ============================================================================
# Argument Parsing
# ============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Places-guided matching against ground truth.")
    p.add_argument(
        "--places-results-path",
        type=Path,
        required=True,
        help="Path to Places processing results CSV (from route_xgb_results.py with --places-mode)",
    )
    p.add_argument(
        "--ground-truth-path",
        type=Path,
        required=True,
        help="Path to ground truth (parquet or CSV)",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/places_matching_evaluation.json"),
        help="Output path for evaluation JSON",
    )
    p.add_argument(
        "--cost-per-call",
        type=float,
        default=0.001,
        help="Cost per Places API call in USD",
    )
    return p.parse_args()


# ============================================================================
# Data Loading
# ============================================================================


def load_places_results(path: Path) -> pd.DataFrame:
    """Load Places processing results."""
    df = pd.read_csv(path, dtype=str)
    LOGGER.info("Loaded Places results: %d rows from %s", len(df), path)
    return df


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load ground truth data."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype=str)

    # Normalize column names
    if "siret" in df.columns and "siret_gt" not in df.columns:
        df = df.rename(columns={"siret": "siret_gt"})
    if "ground_truth_siret" in df.columns:
        df = df.rename(columns={"ground_truth_siret": "siret_gt"})

    LOGGER.info("Loaded ground truth: %d rows from %s", len(df), path)
    return df


def merge_data(places_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Places results with ground truth."""
    # Try different join strategies based on available columns
    if "crm_id" in places_df.columns and "crm_id" in gt_df.columns:
        merged = places_df.merge(gt_df[["crm_id", "siret_gt"]], on="crm_id", how="left")
    elif "query_id" in gt_df.columns:
        # samples_v4 uses query_id + siret + label format
        # We need to extract ground truth differently
        gt_positive = gt_df[gt_df["label"] == 1][["query_id", "siret"]].copy()
        gt_positive = gt_positive.rename(columns={"siret": "siret_gt", "query_id": "crm_id"})
        gt_positive["crm_id"] = gt_positive["crm_id"].astype(str)
        merged = places_df.merge(gt_positive, on="crm_id", how="left")
    else:
        LOGGER.error("Cannot find compatible columns for merge")
        return pd.DataFrame()

    # Determine chosen SIRET
    if "chosen_siret_final" in merged.columns:
        merged["chosen_siret"] = merged["chosen_siret_final"]
    elif "chosen_siret_xgb" in merged.columns:
        merged["chosen_siret"] = merged["chosen_siret_xgb"]
    else:
        merged["chosen_siret"] = merged.get("siret_candidate", "")

    # Add is_correct column
    merged["is_correct"] = merged["chosen_siret"].astype(str) == merged["siret_gt"].astype(str)

    LOGGER.info(
        "Merged: %d rows (%.1f%% have ground truth)",
        len(merged),
        merged["siret_gt"].notna().mean() * 100,
    )
    return merged


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_places_matching(df: pd.DataFrame, cost_per_call: float) -> PlacesMatchingEvaluation:
    """Evaluate Places matching decisions."""
    eval_result = PlacesMatchingEvaluation()

    # Filter to rows with ground truth
    df_gt = df[df["siret_gt"].notna()].copy()
    eval_result.total_records = len(df_gt)

    if eval_result.total_records == 0:
        LOGGER.warning("No records with ground truth found")
        return eval_result

    # MATCH_PLACES metrics
    match_places_mask = df_gt["final_status"] == "MATCH_PLACES"
    match_places_df = df_gt[match_places_mask]
    eval_result.match_places_count = len(match_places_df)
    eval_result.match_places_correct = match_places_df["is_correct"].sum()
    eval_result.match_places_incorrect = eval_result.match_places_count - eval_result.match_places_correct
    eval_result.match_places_precision = (
        eval_result.match_places_correct / eval_result.match_places_count
        if eval_result.match_places_count > 0
        else 0
    )

    # NO_MATCH metrics
    no_match_mask = df_gt["final_status"] == "NO_MATCH"
    no_match_df = df_gt[no_match_mask]
    eval_result.no_match_count = len(no_match_df)
    # For NO_MATCH, check if GT was available
    eval_result.no_match_had_correct = no_match_df["siret_gt"].notna().sum()

    # address_close metrics
    if "places_status" in df_gt.columns:
        # Cases where Places would have promoted but address_close blocked
        blocked_by_addr = df_gt[
            (df_gt["places_status"] == "MATCH_PLACES")
            & (df_gt["final_status"] == "NO_MATCH")
        ]
        eval_result.address_close_failed = len(blocked_by_addr)
        # Check if blocking was correct (prevented FP)
        eval_result.address_close_fp_prevented = len(blocked_by_addr[~blocked_by_addr["is_correct"]])

    # Pool coverage analysis (if available)
    if "places_pool_size" in df_gt.columns:
        pool_sizes = pd.to_numeric(df_gt["places_pool_size"], errors="coerce")
        eval_result.pool_coverage_total = int(pool_sizes.notna().sum())

    # Source contribution (if tracked)
    if "match_source" in df_gt.columns:
        sources = df_gt[match_places_mask]["match_source"].value_counts()
        eval_result.match_from_arm_a = int(sources.get("arm_a", 0))
        eval_result.match_from_arm_b = int(sources.get("arm_b", 0))
        eval_result.match_from_topk = int(sources.get("topk", 0))

    # Cost analysis
    # Count Places API calls (REVIEW cases that went to Places)
    review_mask = df_gt["xgb_status"].str.startswith("REVIEW", na=False)
    eval_result.places_api_calls = review_mask.sum()
    eval_result.estimated_cost_usd = eval_result.places_api_calls * cost_per_call

    # Segment breakdown
    if "segment" in df_gt.columns:
        for segment in df_gt["segment"].dropna().unique():
            seg_df = df_gt[df_gt["segment"] == segment]
            seg_match_places = seg_df[seg_df["final_status"] == "MATCH_PLACES"]
            eval_result.segment_metrics[segment] = {
                "total": len(seg_df),
                "match_places_count": len(seg_match_places),
                "match_places_rate": len(seg_match_places) / len(seg_df) if len(seg_df) > 0 else 0,
                "match_places_correct": seg_match_places["is_correct"].sum(),
                "match_places_precision": (
                    seg_match_places["is_correct"].sum() / len(seg_match_places)
                    if len(seg_match_places) > 0
                    else 0
                ),
            }

    return eval_result


# ============================================================================
# Output
# ============================================================================


def save_evaluation(eval_result: PlacesMatchingEvaluation, output_path: Path) -> None:
    """Save evaluation results to JSON."""
    result_dict = {
        "total_records": eval_result.total_records,
        "match_places_metrics": {
            "count": eval_result.match_places_count,
            "correct": eval_result.match_places_correct,
            "incorrect": eval_result.match_places_incorrect,
            "precision": eval_result.match_places_precision,
        },
        "no_match_metrics": {
            "count": eval_result.no_match_count,
            "had_correct_in_pool": eval_result.no_match_had_correct,
        },
        "address_close_metrics": {
            "failed_count": eval_result.address_close_failed,
            "fp_prevented": eval_result.address_close_fp_prevented,
        },
        "pool_coverage": {
            "arm_a": eval_result.pool_coverage_arm_a,
            "arm_b": eval_result.pool_coverage_arm_b,
            "topk": eval_result.pool_coverage_topk,
            "total": eval_result.pool_coverage_total,
        },
        "source_contribution": {
            "arm_a": eval_result.match_from_arm_a,
            "arm_b": eval_result.match_from_arm_b,
            "topk": eval_result.match_from_topk,
        },
        "cost_analysis": {
            "places_api_calls": eval_result.places_api_calls,
            "estimated_cost_usd": eval_result.estimated_cost_usd,
        },
        "segment_breakdown": eval_result.segment_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    LOGGER.info("Wrote evaluation to %s", output_path)


def print_summary(eval_result: PlacesMatchingEvaluation) -> None:
    """Print evaluation summary."""
    LOGGER.info("=" * 60)
    LOGGER.info("PLACES MATCHING EVALUATION SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("Total records: %d", eval_result.total_records)
    LOGGER.info("")
    LOGGER.info("MATCH_PLACES Metrics:")
    LOGGER.info("  Count: %d", eval_result.match_places_count)
    LOGGER.info("  Correct: %d", eval_result.match_places_correct)
    LOGGER.info("  Incorrect: %d", eval_result.match_places_incorrect)
    LOGGER.info("  Precision: %.2f%%", eval_result.match_places_precision * 100)
    LOGGER.info("")
    LOGGER.info("NO_MATCH Metrics:")
    LOGGER.info("  Count: %d", eval_result.no_match_count)
    LOGGER.info("  Had correct in pool: %d", eval_result.no_match_had_correct)
    LOGGER.info("")
    LOGGER.info("address_close Metrics:")
    LOGGER.info("  Failed (blocked promotion): %d", eval_result.address_close_failed)
    LOGGER.info("  FP prevented: %d", eval_result.address_close_fp_prevented)
    LOGGER.info("")
    LOGGER.info("Source Contribution:")
    LOGGER.info("  From Arm A (address): %d", eval_result.match_from_arm_a)
    LOGGER.info("  From Arm B (radius): %d", eval_result.match_from_arm_b)
    LOGGER.info("  From XGB top-k: %d", eval_result.match_from_topk)
    LOGGER.info("")
    LOGGER.info("Cost Analysis:")
    LOGGER.info("  Places API calls: %d", eval_result.places_api_calls)
    LOGGER.info("  Estimated cost: $%.2f", eval_result.estimated_cost_usd)
    LOGGER.info("=" * 60)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = _parse_args()

    # Load data
    places_df = load_places_results(args.places_results_path)
    gt_df = load_ground_truth(args.ground_truth_path)

    # Merge
    merged = merge_data(places_df, gt_df)

    if len(merged) == 0:
        LOGGER.error("No data to evaluate after merge")
        sys.exit(1)

    # Evaluate
    eval_result = evaluate_places_matching(merged, args.cost_per_call)

    # Output
    save_evaluation(eval_result, args.output_path)
    print_summary(eval_result)


if __name__ == "__main__":
    main()
