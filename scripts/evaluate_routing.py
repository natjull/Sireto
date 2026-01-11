#!/usr/bin/env python3
"""
Evaluate routing decisions against ground truth.

Phase 4: Routing Evaluation
===========================
This script evaluates the quality of AUTO/REVIEW routing decisions
by comparing against ground truth labels.

Metrics computed:
- AUTO accuracy (precision): % of AUTO decisions that are correct
- REVIEW coverage: % of incorrect matches sent to REVIEW
- FP rate: false positive rate (incorrect AUTO)
- FN rate: false negative rate (correct matches in REVIEW)
- Cost-benefit analysis: Places API calls saved vs errors

Usage:
    python scripts/evaluate_routing.py \
        --routed-path reports/routed_results.csv \
        --ground-truth-path data/samples_v4_with_ranker.parquet \
        --output-path reports/routing_evaluation.json
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
class RoutingEvaluation:
    """Complete routing evaluation results."""

    total_records: int = 0

    # AUTO metrics
    auto_count: int = 0
    auto_correct: int = 0
    auto_incorrect: int = 0
    auto_rate: float = 0.0
    auto_precision: float = 0.0

    # REVIEW metrics
    review_count: int = 0
    review_would_be_correct: int = 0
    review_would_be_incorrect: int = 0

    # MATCH_PLACES metrics
    match_places_count: int = 0
    match_places_correct: int = 0
    match_places_incorrect: int = 0

    # NO_MATCH metrics
    no_match_count: int = 0
    no_match_had_correct: int = 0

    # Global metrics
    fp_rate: float = 0.0  # AUTO incorrect / AUTO total
    fn_rate: float = 0.0  # Correct in REVIEW / Total correct

    # Cost analysis
    estimated_places_calls: int = 0
    estimated_cost_usd: float = 0.0
    places_calls_saved: int = 0
    savings_usd: float = 0.0

    # Breakdown by segment
    segment_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Breakdown by certainty rule
    certainty_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ============================================================================
# Argument Parsing
# ============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate routing decisions against ground truth.")
    p.add_argument(
        "--routed-path",
        type=Path,
        required=True,
        help="Path to routed results CSV (from route_xgb_results.py)",
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
        default=Path("reports/routing_evaluation.json"),
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


def load_routed(path: Path) -> pd.DataFrame:
    """Load routed results."""
    df = pd.read_csv(path, dtype=str)
    LOGGER.info("Loaded routed results: %d rows from %s", len(df), path)
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


def merge_data(routed_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """Merge routed results with ground truth."""
    # Merge on crm_id
    merged = routed_df.merge(gt_df[["crm_id", "siret_gt"]], on="crm_id", how="left")

    # Determine chosen SIRET
    if "chosen_siret_final" in merged.columns:
        merged["chosen_siret"] = merged["chosen_siret_final"]
    elif "chosen_siret_xgb" in merged.columns:
        merged["chosen_siret"] = merged["chosen_siret_xgb"]
    else:
        merged["chosen_siret"] = merged.get("siret_candidate", "")

    # FIX: Normalize SIRETs to 14 chars with zero-padding before comparison
    # This fixes cases like "7565021800318" vs "07565021800318"
    def normalize_siret(siret):
        if pd.isna(siret):
            return None
        s = str(siret).replace(" ", "").strip()
        return s.zfill(14) if s else None

    merged["chosen_siret_norm"] = merged["chosen_siret"].apply(normalize_siret)
    merged["siret_gt_norm"] = merged["siret_gt"].apply(normalize_siret)

    # Add is_correct column (using normalized SIRETs)
    merged["is_correct"] = merged["chosen_siret_norm"] == merged["siret_gt_norm"]

    # Determine status
    if "final_status" in merged.columns:
        merged["status"] = merged["final_status"]
    elif "xgb_status" in merged.columns:
        merged["status"] = merged["xgb_status"]
    else:
        merged["status"] = "UNKNOWN"

    LOGGER.info(
        "Merged: %d rows (%.1f%% have ground truth)",
        len(merged),
        merged["siret_gt"].notna().mean() * 100,
    )
    return merged


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_routing(df: pd.DataFrame, cost_per_call: float) -> RoutingEvaluation:
    """Evaluate routing decisions."""
    eval_result = RoutingEvaluation()

    # Filter to rows with ground truth
    df_gt = df[df["siret_gt"].notna()].copy()
    eval_result.total_records = len(df_gt)

    if eval_result.total_records == 0:
        LOGGER.warning("No records with ground truth found")
        return eval_result

    # Count by status
    status_counts = df_gt["status"].value_counts().to_dict()

    # AUTO metrics (includes AUTO_CERTAIN, AUTO_SAME_SIREN, AUTO_BUDGET)
    auto_mask = df_gt["status"].str.startswith("AUTO", na=False)
    auto_df = df_gt[auto_mask]
    eval_result.auto_count = len(auto_df)
    eval_result.auto_correct = auto_df["is_correct"].sum()
    eval_result.auto_incorrect = eval_result.auto_count - eval_result.auto_correct
    eval_result.auto_rate = eval_result.auto_count / eval_result.total_records
    eval_result.auto_precision = (
        eval_result.auto_correct / eval_result.auto_count
        if eval_result.auto_count > 0
        else 0
    )

    # REVIEW metrics
    review_mask = df_gt["status"].str.startswith("REVIEW", na=False)
    review_df = df_gt[review_mask]
    eval_result.review_count = len(review_df)
    eval_result.review_would_be_correct = review_df["is_correct"].sum()
    eval_result.review_would_be_incorrect = eval_result.review_count - eval_result.review_would_be_correct

    # MATCH_PLACES metrics
    places_mask = df_gt["status"] == "MATCH_PLACES"
    places_df = df_gt[places_mask]
    eval_result.match_places_count = len(places_df)
    eval_result.match_places_correct = places_df["is_correct"].sum()
    eval_result.match_places_incorrect = eval_result.match_places_count - eval_result.match_places_correct

    # NO_MATCH metrics
    no_match_mask = df_gt["status"] == "NO_MATCH"
    no_match_df = df_gt[no_match_mask]
    eval_result.no_match_count = len(no_match_df)
    # For NO_MATCH, "correct" means there truly was no match (siret_gt is null or special value)
    eval_result.no_match_had_correct = (no_match_df["siret_gt"].notna() & (no_match_df["siret_gt"] != "")).sum()

    # FP/FN rates
    eval_result.fp_rate = (
        eval_result.auto_incorrect / eval_result.auto_count
        if eval_result.auto_count > 0
        else 0
    )

    total_correct = df_gt["is_correct"].sum()
    eval_result.fn_rate = (
        eval_result.review_would_be_correct / total_correct
        if total_correct > 0
        else 0
    )

    # Cost analysis
    eval_result.estimated_places_calls = eval_result.review_count
    eval_result.estimated_cost_usd = eval_result.review_count * cost_per_call
    eval_result.places_calls_saved = eval_result.auto_count
    eval_result.savings_usd = eval_result.auto_count * cost_per_call

    # Segment breakdown
    if "segment" in df_gt.columns:
        for segment in df_gt["segment"].dropna().unique():
            seg_df = df_gt[df_gt["segment"] == segment]
            seg_auto = seg_df[seg_df["status"].str.startswith("AUTO", na=False)]
            eval_result.segment_metrics[segment] = {
                "total": len(seg_df),
                "auto_count": len(seg_auto),
                "auto_rate": len(seg_auto) / len(seg_df) if len(seg_df) > 0 else 0,
                "auto_correct": seg_auto["is_correct"].sum(),
                "auto_precision": (
                    seg_auto["is_correct"].sum() / len(seg_auto)
                    if len(seg_auto) > 0
                    else 0
                ),
            }

    # Certainty rule breakdown
    if "route_certainty_rule" in df_gt.columns:
        for rule in df_gt["route_certainty_rule"].dropna().unique():
            rule_df = df_gt[df_gt["route_certainty_rule"] == rule]
            eval_result.certainty_metrics[rule] = {
                "count": len(rule_df),
                "correct": rule_df["is_correct"].sum(),
                "precision": rule_df["is_correct"].mean(),
            }

    return eval_result


# ============================================================================
# Output
# ============================================================================


def save_evaluation(eval_result: RoutingEvaluation, output_path: Path) -> None:
    """Save evaluation results to JSON."""
    def _to_builtin(obj):
        if isinstance(obj, dict):
            return {k: _to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_builtin(v) for v in obj]
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        return obj

    result_dict = {
        "total_records": eval_result.total_records,
        "auto_metrics": {
            "count": eval_result.auto_count,
            "correct": eval_result.auto_correct,
            "incorrect": eval_result.auto_incorrect,
            "rate": eval_result.auto_rate,
            "precision": eval_result.auto_precision,
        },
        "review_metrics": {
            "count": eval_result.review_count,
            "would_be_correct": eval_result.review_would_be_correct,
            "would_be_incorrect": eval_result.review_would_be_incorrect,
        },
        "match_places_metrics": {
            "count": eval_result.match_places_count,
            "correct": eval_result.match_places_correct,
            "incorrect": eval_result.match_places_incorrect,
        },
        "no_match_metrics": {
            "count": eval_result.no_match_count,
            "had_correct_match": eval_result.no_match_had_correct,
        },
        "error_rates": {
            "fp_rate": eval_result.fp_rate,
            "fn_rate": eval_result.fn_rate,
        },
        "cost_analysis": {
            "estimated_places_calls": eval_result.estimated_places_calls,
            "estimated_cost_usd": eval_result.estimated_cost_usd,
            "places_calls_saved": eval_result.places_calls_saved,
            "savings_usd": eval_result.savings_usd,
        },
        "segment_breakdown": eval_result.segment_metrics,
        "certainty_rule_breakdown": eval_result.certainty_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(result_dict), f, indent=2, ensure_ascii=False)

    LOGGER.info("Wrote evaluation to %s", output_path)


def print_summary(eval_result: RoutingEvaluation) -> None:
    """Print evaluation summary."""
    LOGGER.info("=" * 60)
    LOGGER.info("ROUTING EVALUATION SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("Total records: %d", eval_result.total_records)
    LOGGER.info("")
    LOGGER.info("AUTO Metrics:")
    LOGGER.info("  Count: %d (%.1f%%)", eval_result.auto_count, eval_result.auto_rate * 100)
    LOGGER.info("  Correct: %d", eval_result.auto_correct)
    LOGGER.info("  Incorrect: %d", eval_result.auto_incorrect)
    LOGGER.info("  Precision: %.2f%%", eval_result.auto_precision * 100)
    LOGGER.info("  FP Rate: %.3f%%", eval_result.fp_rate * 100)
    LOGGER.info("")
    LOGGER.info("REVIEW Metrics:")
    LOGGER.info("  Count: %d", eval_result.review_count)
    LOGGER.info("  Would be correct: %d", eval_result.review_would_be_correct)
    LOGGER.info("  FN Rate: %.2f%%", eval_result.fn_rate * 100)
    LOGGER.info("")
    LOGGER.info("MATCH_PLACES Metrics:")
    LOGGER.info("  Count: %d", eval_result.match_places_count)
    LOGGER.info("  Correct: %d", eval_result.match_places_correct)
    LOGGER.info("  Incorrect: %d", eval_result.match_places_incorrect)
    LOGGER.info("")
    LOGGER.info("Cost Analysis:")
    LOGGER.info("  Places API calls: %d", eval_result.estimated_places_calls)
    LOGGER.info("  Estimated cost: $%.2f", eval_result.estimated_cost_usd)
    LOGGER.info("  Calls saved by AUTO: %d", eval_result.places_calls_saved)
    LOGGER.info("  Savings: $%.2f", eval_result.savings_usd)
    LOGGER.info("=" * 60)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = _parse_args()

    # Load data
    routed_df = load_routed(args.routed_path)
    gt_df = load_ground_truth(args.ground_truth_path)

    # Merge
    merged = merge_data(routed_df, gt_df)

    # Evaluate
    eval_result = evaluate_routing(merged, args.cost_per_call)

    # Output
    save_evaluation(eval_result, args.output_path)
    print_summary(eval_result)


if __name__ == "__main__":
    main()
