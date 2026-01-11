#!/usr/bin/env python3
"""
Calibrate routing thresholds using ground truth data.

Phase 4: SOTA Routing Calibration
=================================
This script analyzes XGB inference results with ground truth to find optimal
thresholds for AUTO/REVIEW routing that achieve:
- Target FP rate <= 0.1%
- Maximum AUTO rate (minimize Places API calls)

Usage:
    python scripts/calibrate_routing_thresholds.py \
        --inference-path reports/xgb_two_stage_topk.csv \
        --ground-truth-path data/samples_v4_with_ranker.parquet \
        --output-path configs/routing_thresholds_calibrated.yaml

Output:
    - YAML config with calibrated thresholds
    - Calibration report with metrics per segment
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
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
class CalibrationResult:
    """Result of threshold calibration for a segment."""

    segment: str
    threshold: float
    gap_min: float
    total_samples: int
    auto_count: int
    auto_rate: float
    fp_count: int
    fp_rate: float
    fn_count: int
    fn_rate: float
    precision: float
    recall: float


@dataclass
class CalibrationReport:
    """Complete calibration report."""

    segments: Dict[str, CalibrationResult] = field(default_factory=dict)
    global_metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Argument Parsing
# ============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate routing thresholds for Phase 4.")
    p.add_argument(
        "--inference-path",
        type=Path,
        required=True,
        help="Path to XGB inference results CSV (with top-k per CRM)",
    )
    p.add_argument(
        "--ground-truth-path",
        type=Path,
        required=True,
        help="Path to ground truth (parquet or CSV with crm_id, siret_gt columns)",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("configs/routing_thresholds_calibrated.yaml"),
        help="Output path for calibrated thresholds YAML",
    )
    p.add_argument(
        "--target-fp-rate",
        type=float,
        default=0.001,
        help="Target false positive rate (default: 0.1%%)",
    )
    p.add_argument(
        "--min-auto-rate",
        type=float,
        default=0.60,
        help="Minimum AUTO rate target (default: 60%%)",
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


def load_inference(path: Path) -> pd.DataFrame:
    """Load XGB inference results."""
    df = pd.read_csv(path, dtype={"siret_candidate": str, "crm_id": str})
    LOGGER.info("Loaded inference: %d rows from %s", len(df), path)
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

    # Keep only relevant columns
    gt_cols = ["crm_id", "siret_gt"]
    for col in gt_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    LOGGER.info("Loaded ground truth: %d rows from %s", len(df), path)
    return df[gt_cols].drop_duplicates()


def merge_with_ground_truth(infer_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """Merge inference with ground truth."""
    # Get top-1 per CRM
    if "rank" in infer_df.columns:
        top1 = infer_df[infer_df["rank"] == 1].copy()
    else:
        top1 = infer_df.sort_values("score", ascending=False).groupby("crm_id").first().reset_index()

    # Merge with ground truth
    merged = top1.merge(gt_df, on="crm_id", how="inner")

    # Add is_correct column
    merged["is_correct"] = merged["siret_candidate"].astype(str) == merged["siret_gt"].astype(str)

    LOGGER.info("Merged: %d rows (%.1f%% correct)", len(merged), merged["is_correct"].mean() * 100)
    return merged


# ============================================================================
# Segment Detection
# ============================================================================


def _has_complete_address(row: pd.Series) -> bool:
    """Match routing logic for address completeness (street_number/street_name or crm_address)."""
    import re

    street_number = str(row.get("street_number", "") or "").strip()
    street_name = str(row.get("street_name", "") or "").strip()
    crm_address = str(row.get("crm_address", "") or "").strip()

    has_number = bool(street_number) or bool(re.match(r"^\s*\d+", crm_address))
    if street_name:
        has_street = True
    else:
        addr_tokens = re.findall(r"[A-Z0-9]+", crm_address.upper()) if crm_address else []
        has_street = len(addr_tokens) >= 2
    return bool(has_number and has_street)


def detect_segment(row: pd.Series) -> str:
    """Detect segment for a row based on features (aligned with route_xgb_results)."""
    # Extract features
    idf_name = float(row.get("idf_name", 0) or 0)
    name_length = float(row.get("name_length_max", 0) or 0)
    crm_name = str(row.get("crm_name", "") or "")

    # Count words in CRM name
    import re

    words = re.findall(r"[A-Z0-9]+", crm_name.upper())
    word_count = len(words)

    # Check address completeness (aligned with routing logic)
    addr_complete = _has_complete_address(row)

    # Segment logic
    if word_count <= 2 or (word_count == 0 and name_length <= 6):
        return "short_name"

    if idf_name >= 5.0:
        if addr_complete:
            return "unique_name_full_addr"
        else:
            return "unique_name_partial_addr"
    else:
        if addr_complete:
            return "common_name_full_addr"
        else:
            return "common_name_partial_addr"


# ============================================================================
# Threshold Calibration
# ============================================================================


def _ensure_score_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure score_gap is populated (fallback to score_top1/score_top2)."""
    out = df.copy()
    if "score_gap" not in out.columns or out["score_gap"].isna().all():
        if "score_top1" in out.columns and "score_top2" in out.columns:
            out["score_gap"] = pd.to_numeric(out["score_top1"], errors="coerce") - pd.to_numeric(
                out["score_top2"], errors="coerce"
            )
    out["score_gap"] = pd.to_numeric(out.get("score_gap"), errors="coerce").fillna(0.0)
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce").fillna(0.0)
    return out


def calibrate_segment(
    df: pd.DataFrame,
    segment: str,
    target_fp_rate: float,
    min_auto_rate: float,
) -> CalibrationResult:
    """Calibrate thresholds for a single segment."""
    n = len(df)
    if n == 0:
        return CalibrationResult(
            segment=segment,
            threshold=0.99,
            gap_min=0.05,
            total_samples=0,
            auto_count=0,
            auto_rate=0,
            fp_count=0,
            fp_rate=0,
            fn_count=0,
            fn_rate=0,
            precision=0,
            recall=0,
        )

    # Normalize score/gap
    df = _ensure_score_gap(df)
    df = df.sort_values("score", ascending=False)

    # Find threshold that achieves target FP rate
    best_threshold = 0.999
    best_gap_min = 0.10
    best_auto_rate = -1.0
    best_fp_rate = 1.0
    best_ok = False
    best_fallback = None  # (fp_rate, auto_rate, threshold, gap)

    # Grid search over thresholds
    for threshold in np.arange(0.995, 0.90, -0.005):
        for gap_min in [0.03, 0.05, 0.08, 0.10]:
            # Apply threshold
            mask = (df["score"] >= threshold) & (df["score_gap"] >= gap_min)
            auto_df = df[mask]
            review_df = df[~mask]

            if len(auto_df) == 0:
                continue

            # Calculate FP rate
            auto_correct = auto_df["is_correct"].sum()
            auto_incorrect = len(auto_df) - auto_correct
            fp_rate = auto_incorrect / len(auto_df) if len(auto_df) > 0 else 0

            auto_rate = len(auto_df) / n

            if fp_rate <= target_fp_rate:
                ok = auto_rate >= min_auto_rate
                # Prefer candidates meeting min_auto_rate; then maximize auto_rate
                if (ok and not best_ok) or (
                    ok == best_ok and (auto_rate > best_auto_rate + 1e-9)
                ) or (
                    ok == best_ok
                    and abs(auto_rate - best_auto_rate) <= 1e-9
                    and fp_rate < best_fp_rate
                ):
                    best_threshold = threshold
                    best_gap_min = gap_min
                    best_auto_rate = auto_rate
                    best_fp_rate = fp_rate
                    best_ok = ok
            else:
                # Fallback candidate if no thresholds meet FP target
                if best_fallback is None or fp_rate < best_fallback[0] or (
                    fp_rate == best_fallback[0] and auto_rate > best_fallback[1]
                ):
                    best_fallback = (fp_rate, auto_rate, threshold, gap_min)

    if not best_ok and best_auto_rate < 0 and best_fallback is not None:
        _, _, best_threshold, best_gap_min = best_fallback

    # Calculate final metrics with best threshold
    mask = (df["score"] >= best_threshold) & (df["score_gap"] >= best_gap_min)
    auto_df = df[mask]

    auto_count = len(auto_df)
    auto_rate = auto_count / n if n > 0 else 0
    fp_count = len(auto_df[~auto_df["is_correct"]]) if len(auto_df) > 0 else 0
    fp_rate = fp_count / auto_count if auto_count > 0 else 0

    # FN: correct samples that went to REVIEW
    review_df = df[~mask]
    fn_count = len(review_df[review_df["is_correct"]])
    fn_rate = fn_count / n if n > 0 else 0

    # Precision/Recall
    precision = (auto_count - fp_count) / auto_count if auto_count > 0 else 0
    recall = (auto_count - fp_count) / df["is_correct"].sum() if df["is_correct"].sum() > 0 else 0

    return CalibrationResult(
        segment=segment,
        threshold=best_threshold,
        gap_min=best_gap_min,
        total_samples=n,
        auto_count=auto_count,
        auto_rate=auto_rate,
        fp_count=fp_count,
        fp_rate=fp_rate,
        fn_count=fn_count,
        fn_rate=fn_rate,
        precision=precision,
        recall=recall,
    )


def calibrate_all_segments(
    df: pd.DataFrame,
    target_fp_rate: float,
    min_auto_rate: float,
) -> CalibrationReport:
    """Calibrate thresholds for all segments."""
    report = CalibrationReport()

    # Add segment column
    df = df.copy()
    df["segment"] = df.apply(detect_segment, axis=1)

    # Calibrate each segment
    for segment in df["segment"].unique():
        segment_df = df[df["segment"] == segment]
        result = calibrate_segment(segment_df, segment, target_fp_rate, min_auto_rate)
        report.segments[segment] = result
        LOGGER.info(
            "Segment %s: threshold=%.3f gap=%.2f AUTO=%.1f%% FP=%.2f%%",
            segment,
            result.threshold,
            result.gap_min,
            result.auto_rate * 100,
            result.fp_rate * 100,
        )

    # Calibrate default (all data)
    default_result = calibrate_segment(df, "default", target_fp_rate, min_auto_rate)
    report.segments["default"] = default_result

    # Global metrics
    total_auto = sum(r.auto_count for r in report.segments.values() if r.segment != "default")
    total_samples = sum(r.total_samples for r in report.segments.values() if r.segment != "default")
    total_fp = sum(r.fp_count for r in report.segments.values() if r.segment != "default")

    report.global_metrics = {
        "total_samples": total_samples,
        "total_auto": total_auto,
        "global_auto_rate": total_auto / total_samples if total_samples > 0 else 0,
        "global_fp_count": total_fp,
        "global_fp_rate": total_fp / total_auto if total_auto > 0 else 0,
    }

    return report


# ============================================================================
# Output Generation
# ============================================================================


def generate_yaml(report: CalibrationReport, output_path: Path) -> None:
    """Generate YAML config from calibration report."""
    if yaml is None:
        LOGGER.warning("PyYAML not available, skipping YAML output")
        return

    config = {
        "version": "1.0",
        "calibrated_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "segments": {},
        "certainty_rules": {"enabled": True},
        "same_siren_resolution": {"enabled": True},
        "blocking_rules": {},
        "promotion_rules": {},
        "budget": {"mode": "normal"},
    }

    for segment, result in report.segments.items():
        config["segments"][segment] = {
            "threshold": float(result.threshold),
            "gap_min": float(result.gap_min),
            "places_value": "medium",
            "calibration_metrics": {
                "samples": result.total_samples,
                "auto_rate": round(result.auto_rate, 4),
                "fp_rate": round(result.fp_rate, 4),
            },
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    LOGGER.info("Wrote calibrated config to %s", output_path)


def generate_report(report: CalibrationReport, output_path: Path) -> None:
    """Generate JSON report from calibration."""
    report_dict = {
        "global_metrics": report.global_metrics,
        "segments": {
            seg: {
                "threshold": r.threshold,
                "gap_min": r.gap_min,
                "total_samples": r.total_samples,
                "auto_count": r.auto_count,
                "auto_rate": r.auto_rate,
                "fp_count": r.fp_count,
                "fp_rate": r.fp_rate,
                "fn_count": r.fn_count,
                "fn_rate": r.fn_rate,
                "precision": r.precision,
                "recall": r.recall,
            }
            for seg, r in report.segments.items()
        },
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
    infer_df = load_inference(args.inference_path)
    gt_df = load_ground_truth(args.ground_truth_path)

    # Merge
    merged = merge_with_ground_truth(infer_df, gt_df)

    if len(merged) == 0:
        LOGGER.error("No matching records between inference and ground truth")
        sys.exit(1)

    # Calibrate
    report = calibrate_all_segments(merged, args.target_fp_rate, args.min_auto_rate)

    # Output
    generate_yaml(report, args.output_path)

    if args.report_path:
        generate_report(report, args.report_path)

    # Summary
    LOGGER.info("=" * 60)
    LOGGER.info("CALIBRATION SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("Total samples: %d", report.global_metrics.get("total_samples", 0))
    LOGGER.info(
        "Global AUTO rate: %.1f%%", report.global_metrics.get("global_auto_rate", 0) * 100
    )
    LOGGER.info(
        "Global FP rate: %.3f%%", report.global_metrics.get("global_fp_rate", 0) * 100
    )
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
