#!/usr/bin/env python3
"""
Calibrate Places promotion thresholds using ground truth data.

Phase 4: Places Threshold Calibration
=====================================
This script analyzes Places processing results with ground truth to find optimal
thresholds for MATCH_PLACES promotion that achieve:
- Target FP rate <= 0.1%
- Maximum promotion rate (minimize REVIEW/NO_MATCH)

Works with labeled training samples where we have ground truth SIRET.

Usage:
    python scripts/calibrate_places_thresholds.py \
        --samples-path data/samples_v4_with_ranker.parquet \
        --output-path configs/places_thresholds_calibrated.yaml

Output:
    - YAML config with calibrated thresholds
    - Calibration report with metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import yaml
except ImportError:
    yaml = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class PlacesCalibrationResult:
    """Result of Places threshold calibration."""

    score_min: float
    gap_min: float
    total_samples: int
    match_places_count: int
    match_places_rate: float
    fp_count: int
    fp_rate: float
    fn_count: int
    fn_rate: float
    precision: float
    recall: float


@dataclass
class PlacesCalibrationReport:
    """Complete Places calibration report."""

    main_result: PlacesCalibrationResult | None = None
    address_close_metrics: Dict[str, Any] = field(default_factory=dict)
    segment_breakdown: Dict[str, PlacesCalibrationResult] = field(default_factory=dict)
    global_metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Argument Parsing
# ============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate Places promotion thresholds for Phase 4.")
    p.add_argument(
        "--samples-path",
        type=Path,
        required=True,
        help="Path to samples with ground truth (parquet or CSV)",
    )
    p.add_argument(
        "--places-results-path",
        type=Path,
        default=None,
        help="Path to Places processing results (if available)",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("configs/places_thresholds_calibrated.yaml"),
        help="Output path for calibrated thresholds YAML",
    )
    p.add_argument(
        "--target-fp-rate",
        type=float,
        default=0.001,
        help="Target false positive rate (default: 0.1%%)",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path to save calibration report JSON",
    )
    return p.parse_args()


# ============================================================================
# Data Loading
# ============================================================================


def load_samples(path: Path) -> pd.DataFrame:
    """Load samples with ground truth."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype=str)

    LOGGER.info("Loaded samples: %d rows from %s", len(df), path)
    return df


def load_places_results(path: Path) -> pd.DataFrame:
    """Load Places processing results."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    LOGGER.info("Loaded Places results: %d rows from %s", len(df), path)
    return df


# ============================================================================
# Threshold Calibration
# ============================================================================


def calibrate_places_thresholds(
    df: pd.DataFrame,
    target_fp_rate: float,
) -> PlacesCalibrationResult:
    """Calibrate Places promotion thresholds.

    If no Places results are available, we simulate by looking at XGB scores
    and address matching features to estimate optimal thresholds.

    Args:
        df: DataFrame with score, gap, and label columns
        target_fp_rate: Target FP rate

    Returns:
        CalibrationResult with optimal thresholds
    """
    n = len(df)
    if n == 0:
        return PlacesCalibrationResult(
            score_min=0.97,
            gap_min=0.05,
            total_samples=0,
            match_places_count=0,
            match_places_rate=0,
            fp_count=0,
            fp_rate=0,
            fn_count=0,
            fn_rate=0,
            precision=0,
            recall=0,
        )

    # Filter to positive labels (correct matches)
    positive_df = df[df["label"] == 1].copy()
    negative_df = df[df["label"] == 0].copy()

    LOGGER.info("Positive samples: %d, Negative samples: %d", len(positive_df), len(negative_df))

    # Grid search for optimal thresholds
    best_score_min = 0.97
    best_gap_min = 0.05
    best_f1 = 0.0

    # We need a score column - use name_semantic_max or addr_jaro as proxy for Places promotion
    score_col = "name_semantic_max" if "name_semantic_max" in df.columns else "name_jaro_max"

    for score_min in np.arange(0.90, 0.99, 0.01):
        for gap_min in [0.03, 0.05, 0.08, 0.10]:
            # Apply thresholds
            # For positive samples: how many would be correctly promoted (score >= threshold)
            pos_promoted = positive_df[positive_df[score_col] >= score_min]
            # For negative samples: how many would be incorrectly promoted (FP)
            neg_promoted = negative_df[negative_df[score_col] >= score_min]

            if len(pos_promoted) == 0:
                continue

            tp = len(pos_promoted)
            fp = len(neg_promoted)
            fn = len(positive_df) - tp

            # Check if FP rate is acceptable
            fp_rate = fp / (tp + fp) if (tp + fp) > 0 else 0

            if fp_rate <= target_fp_rate:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                if f1 > best_f1:
                    best_f1 = f1
                    best_score_min = score_min
                    best_gap_min = gap_min

    # Calculate final metrics with best thresholds
    pos_promoted = positive_df[positive_df[score_col] >= best_score_min]
    neg_promoted = negative_df[negative_df[score_col] >= best_score_min]

    tp = len(pos_promoted)
    fp = len(neg_promoted)
    fn = len(positive_df) - tp
    total_promoted = tp + fp

    match_places_rate = total_promoted / n if n > 0 else 0
    fp_rate = fp / total_promoted if total_promoted > 0 else 0
    fn_rate = fn / len(positive_df) if len(positive_df) > 0 else 0
    precision = tp / total_promoted if total_promoted > 0 else 0
    recall = tp / len(positive_df) if len(positive_df) > 0 else 0

    return PlacesCalibrationResult(
        score_min=best_score_min,
        gap_min=best_gap_min,
        total_samples=n,
        match_places_count=total_promoted,
        match_places_rate=match_places_rate,
        fp_count=fp,
        fp_rate=fp_rate,
        fn_count=fn,
        fn_rate=fn_rate,
        precision=precision,
        recall=recall,
    )


def calibrate_address_close(df: pd.DataFrame, target_fp_rate: float) -> Dict[str, Any]:
    """Calibrate address_close parameters.

    Uses address features from training data to find optimal thresholds
    for postcode, addr_jaro, street_number_diff that minimize FP.
    """
    if len(df) == 0:
        return {}

    positive_df = df[df["label"] == 1].copy()
    negative_df = df[df["label"] == 0].copy()

    # Analyze address features
    metrics: Dict[str, Any] = {}

    # Postcode match
    if "postcode_match" in df.columns:
        pos_with_pc = positive_df["postcode_match"].mean() if len(positive_df) > 0 else 0
        neg_with_pc = negative_df["postcode_match"].mean() if len(negative_df) > 0 else 0
        metrics["postcode_match"] = {
            "positive_rate": float(pos_with_pc),
            "negative_rate": float(neg_with_pc),
            "recommendation": "require_exact" if pos_with_pc > 0.95 else "allow_mismatch",
        }

    # Address Jaro
    if "addr_jaro" in df.columns:
        pos_addr_jaro = positive_df["addr_jaro"].mean() if len(positive_df) > 0 else 0
        neg_addr_jaro = negative_df["addr_jaro"].mean() if len(negative_df) > 0 else 0

        # Find threshold that separates positive from negative
        thresholds = [0.80, 0.85, 0.90, 0.95]
        best_threshold = 0.90
        best_separation = 0

        for thresh in thresholds:
            pos_above = (positive_df["addr_jaro"] >= thresh).mean() if len(positive_df) > 0 else 0
            neg_above = (negative_df["addr_jaro"] >= thresh).mean() if len(negative_df) > 0 else 0
            separation = pos_above - neg_above
            if separation > best_separation:
                best_separation = separation
                best_threshold = thresh

        metrics["addr_jaro"] = {
            "positive_mean": float(pos_addr_jaro),
            "negative_mean": float(neg_addr_jaro),
            "recommended_threshold": float(best_threshold),
        }

    # Street number diff
    if "street_number_diff" in df.columns:
        pos_sn_diff = positive_df["street_number_diff"].mean() if len(positive_df) > 0 else 999
        neg_sn_diff = negative_df["street_number_diff"].mean() if len(negative_df) > 0 else 999

        # Find threshold that works
        for max_diff in [0, 1, 2, 3, 5]:
            pos_ok = (positive_df["street_number_diff"] <= max_diff).mean() if len(positive_df) > 0 else 0
            neg_ok = (negative_df["street_number_diff"] <= max_diff).mean() if len(negative_df) > 0 else 0

            if pos_ok >= 0.9:  # Accept if 90% of positives pass
                metrics["street_number_diff"] = {
                    "positive_mean": float(pos_sn_diff),
                    "negative_mean": float(neg_sn_diff),
                    "recommended_max": int(max_diff),
                }
                break

    return metrics


# ============================================================================
# Output Generation
# ============================================================================


def generate_yaml(report: PlacesCalibrationReport, output_path: Path) -> None:
    """Generate YAML config from calibration report."""
    if yaml is None:
        LOGGER.warning("PyYAML not available, skipping YAML output")
        return

    config = {
        "version": "1.0",
        "calibrated_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "models": {
            "decider_model": "models/xgb_decider_20260124_210218.json",
            "calibrator_path": "models/xgb_decider_calibrator_isotonic_20260124_210218.pkl",
            "meta_path": "models/xgb_two_stage_meta_20260124_210218.json",
        },
        "pool": {
            "xgb_topk": 20,
            "arm_a_enabled": True,
            "arm_b_enabled": True,
            "arm_b_radius_m": 100,
        },
        "gate": {
            "enabled": True,
            "min_addr_overlap": 0.5,
            "min_name_semantic": 0.3,
        },
        "promotion": {
            "enabled": True,
        },
        "address_close": {
            "enabled": True,
        },
    }

    # Add calibrated values
    if report.main_result:
        config["promotion"]["score_min"] = float(report.main_result.score_min)
        config["promotion"]["gap_min"] = float(report.main_result.gap_min)
        config["promotion"]["calibration_metrics"] = {
            "samples": report.main_result.total_samples,
            "match_rate": round(report.main_result.match_places_rate, 4),
            "fp_rate": round(report.main_result.fp_rate, 4),
            "precision": round(report.main_result.precision, 4),
            "recall": round(report.main_result.recall, 4),
        }

    # Add address_close calibration
    if report.address_close_metrics:
        if "addr_jaro" in report.address_close_metrics:
            config["address_close"]["addr_jaro_min"] = report.address_close_metrics["addr_jaro"].get(
                "recommended_threshold", 0.90
            )
        if "street_number_diff" in report.address_close_metrics:
            config["address_close"]["street_number_diff_max"] = report.address_close_metrics[
                "street_number_diff"
            ].get("recommended_max", 2)
        if "postcode_match" in report.address_close_metrics:
            config["address_close"]["postcode_exact"] = (
                report.address_close_metrics["postcode_match"].get("recommendation") == "require_exact"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    LOGGER.info("Wrote calibrated config to %s", output_path)


def generate_report(report: PlacesCalibrationReport, output_path: Path) -> None:
    """Generate JSON report from calibration."""
    report_dict: Dict[str, Any] = {
        "global_metrics": report.global_metrics,
    }

    if report.main_result:
        report_dict["main_result"] = {
            "score_min": report.main_result.score_min,
            "gap_min": report.main_result.gap_min,
            "total_samples": report.main_result.total_samples,
            "match_places_count": report.main_result.match_places_count,
            "match_places_rate": report.main_result.match_places_rate,
            "fp_count": report.main_result.fp_count,
            "fp_rate": report.main_result.fp_rate,
            "fn_count": report.main_result.fn_count,
            "fn_rate": report.main_result.fn_rate,
            "precision": report.main_result.precision,
            "recall": report.main_result.recall,
        }

    report_dict["address_close_metrics"] = report.address_close_metrics
    report_dict["segment_breakdown"] = {
        seg: {
            "score_min": r.score_min,
            "gap_min": r.gap_min,
            "total_samples": r.total_samples,
            "match_places_count": r.match_places_count,
            "match_places_rate": r.match_places_rate,
            "fp_rate": r.fp_rate,
            "precision": r.precision,
            "recall": r.recall,
        }
        for seg, r in report.segment_breakdown.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    LOGGER.info("Wrote calibration report to %s", output_path)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = _parse_args()

    # Load data
    samples_df = load_samples(args.samples_path)

    if args.places_results_path and args.places_results_path.exists():
        places_df = load_places_results(args.places_results_path)
        # Merge samples with places results
        # For now, we just use samples directly
        df = samples_df
    else:
        df = samples_df

    if len(df) == 0:
        LOGGER.error("No samples to calibrate")
        sys.exit(1)

    # Initialize report
    report = PlacesCalibrationReport()

    # Main calibration
    LOGGER.info("Calibrating Places promotion thresholds...")
    report.main_result = calibrate_places_thresholds(df, args.target_fp_rate)

    if report.main_result:
        LOGGER.info(
            "Main result: score_min=%.3f gap_min=%.2f FP=%.3f%% Precision=%.1f%% Recall=%.1f%%",
            report.main_result.score_min,
            report.main_result.gap_min,
            report.main_result.fp_rate * 100,
            report.main_result.precision * 100,
            report.main_result.recall * 100,
        )

    # Address_close calibration
    LOGGER.info("Calibrating address_close parameters...")
    report.address_close_metrics = calibrate_address_close(df, args.target_fp_rate)

    # Global metrics
    report.global_metrics = {
        "total_samples": len(df),
        "positive_samples": int((df["label"] == 1).sum()) if "label" in df.columns else 0,
        "negative_samples": int((df["label"] == 0).sum()) if "label" in df.columns else 0,
    }

    # Output
    generate_yaml(report, args.output_path)

    if args.report_path:
        generate_report(report, args.report_path)

    # Summary
    LOGGER.info("=" * 60)
    LOGGER.info("PLACES CALIBRATION SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("Total samples: %d", report.global_metrics.get("total_samples", 0))
    if report.main_result:
        LOGGER.info("Optimal score_min: %.3f", report.main_result.score_min)
        LOGGER.info("Optimal gap_min: %.3f", report.main_result.gap_min)
        LOGGER.info("Match rate: %.1f%%", report.main_result.match_places_rate * 100)
        LOGGER.info("FP rate: %.3f%%", report.main_result.fp_rate * 100)
        LOGGER.info("Precision: %.1f%%", report.main_result.precision * 100)
        LOGGER.info("Recall: %.1f%%", report.main_result.recall * 100)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
