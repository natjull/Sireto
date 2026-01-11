#!/usr/bin/env python3
"""
Simulate Places API costs based on routing decisions.

Phase 4: Cost Simulation
========================
This script simulates the cost of Places API calls based on different
routing strategies and budget modes.

Features:
- Simulate costs for different AUTO thresholds
- Compare budget modes (aggressive/normal/permissive)
- Project monthly costs
- Identify cost-effective threshold configurations

Usage:
    python scripts/simulate_places_costs.py \
        --inference-path reports/xgb_two_stage_topk.csv \
        --output-path reports/places_cost_simulation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Serper Places API pricing (as of 2026)
SERPER_FREE_TIER = 2500  # Free calls per month
SERPER_COST_PER_CALL = 0.001  # USD per call after free tier

# Google Places API pricing
GOOGLE_COST_PER_CALL = 0.017  # USD per call

# Average expected volumes
DAILY_CRM_VOLUME_LOW = 50
DAILY_CRM_VOLUME_MED = 200
DAILY_CRM_VOLUME_HIGH = 1000


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CostSimulation:
    """Cost simulation for a routing configuration."""

    name: str
    threshold: float
    gap_min: float
    total_records: int
    auto_count: int
    review_count: int
    auto_rate: float
    places_calls: int
    monthly_calls: int
    monthly_cost_serper: float
    monthly_cost_google: float
    annual_cost_serper: float
    annual_cost_google: float


@dataclass
class CostReport:
    """Complete cost simulation report."""

    simulations: List[CostSimulation] = field(default_factory=list)
    recommendations: Dict[str, Any] = field(default_factory=dict)
    volume_projections: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ============================================================================
# Argument Parsing
# ============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate Places API costs for routing.")
    p.add_argument(
        "--inference-path",
        type=Path,
        required=True,
        help="Path to XGB inference results CSV",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/places_cost_simulation.json"),
        help="Output path for cost simulation JSON",
    )
    p.add_argument(
        "--daily-volume",
        type=int,
        default=100,
        help="Expected daily CRM volume for projections",
    )
    p.add_argument(
        "--ground-truth-path",
        type=Path,
        default=None,
        help="Optional ground truth for accuracy-adjusted costs",
    )
    return p.parse_args()


# ============================================================================
# Simulation
# ============================================================================


def simulate_threshold(
    df: pd.DataFrame,
    threshold: float,
    gap_min: float,
    name: str,
    daily_volume: int,
) -> CostSimulation:
    """Simulate costs for a specific threshold configuration."""
    # Get top-1 per CRM
    if "rank" in df.columns:
        top1 = df[df["rank"] == 1].copy()
    else:
        top1 = df.sort_values("score", ascending=False).groupby("crm_id").first().reset_index()

    total = len(top1)

    # Ensure score_gap exists
    if "score_gap" not in top1.columns:
        top1["score_gap"] = 0.0

    # Apply threshold
    auto_mask = (top1["score"].astype(float) >= threshold) & (
        top1["score_gap"].astype(float).fillna(0) >= gap_min
    )
    auto_count = auto_mask.sum()
    review_count = total - auto_count

    auto_rate = auto_count / total if total > 0 else 0

    # Project monthly
    sample_rate = total / 1  # Assuming this is 1 day of data
    monthly_multiplier = (daily_volume / total) * 30 if total > 0 else 30

    monthly_calls = int(review_count * monthly_multiplier)

    # Calculate costs
    # Serper: 2500 free, then $0.001 per call
    if monthly_calls <= SERPER_FREE_TIER:
        monthly_cost_serper = 0
    else:
        monthly_cost_serper = (monthly_calls - SERPER_FREE_TIER) * SERPER_COST_PER_CALL

    # Google: $0.017 per call
    monthly_cost_google = monthly_calls * GOOGLE_COST_PER_CALL

    return CostSimulation(
        name=name,
        threshold=threshold,
        gap_min=gap_min,
        total_records=total,
        auto_count=int(auto_count),
        review_count=int(review_count),
        auto_rate=auto_rate,
        places_calls=int(review_count),
        monthly_calls=monthly_calls,
        monthly_cost_serper=monthly_cost_serper,
        monthly_cost_google=monthly_cost_google,
        annual_cost_serper=monthly_cost_serper * 12,
        annual_cost_google=monthly_cost_google * 12,
    )


def run_simulations(df: pd.DataFrame, daily_volume: int) -> CostReport:
    """Run cost simulations for various configurations."""
    report = CostReport()

    # Define configurations to simulate
    configs = [
        ("aggressive", 0.95, 0.03),
        ("normal", 0.98, 0.05),
        ("permissive", 0.99, 0.08),
        ("conservative", 0.995, 0.10),
        ("ultra_conservative", 0.999, 0.15),
    ]

    # Also test a range of thresholds
    for threshold in [0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999]:
        for gap in [0.03, 0.05, 0.08]:
            name = f"t{int(threshold*1000)}_g{int(gap*100)}"
            sim = simulate_threshold(df, threshold, gap, name, daily_volume)
            report.simulations.append(sim)

    # Add named configs
    for name, threshold, gap in configs:
        sim = simulate_threshold(df, threshold, gap, f"config_{name}", daily_volume)
        report.simulations.append(sim)

    # Volume projections
    for label, volume in [
        ("low", DAILY_CRM_VOLUME_LOW),
        ("medium", DAILY_CRM_VOLUME_MED),
        ("high", DAILY_CRM_VOLUME_HIGH),
    ]:
        # Use normal config for projections
        sim = simulate_threshold(df, 0.98, 0.05, f"proj_{label}", volume)
        report.volume_projections[label] = {
            "daily_volume": volume,
            "monthly_calls": sim.monthly_calls,
            "monthly_cost_serper": sim.monthly_cost_serper,
            "monthly_cost_google": sim.monthly_cost_google,
            "annual_cost_serper": sim.annual_cost_serper,
            "annual_cost_google": sim.annual_cost_google,
        }

    # Generate recommendations
    # Find config with lowest cost while staying under free tier
    under_free_tier = [s for s in report.simulations if s.monthly_calls <= SERPER_FREE_TIER]
    if under_free_tier:
        best_free = max(under_free_tier, key=lambda s: s.auto_rate)
        report.recommendations["stay_free"] = {
            "name": best_free.name,
            "threshold": best_free.threshold,
            "gap_min": best_free.gap_min,
            "auto_rate": best_free.auto_rate,
            "monthly_calls": best_free.monthly_calls,
        }

    # Find best cost-efficiency
    if report.simulations:
        best_efficiency = min(
            report.simulations,
            key=lambda s: s.monthly_cost_serper / max(s.auto_rate, 0.01),
        )
        report.recommendations["best_efficiency"] = {
            "name": best_efficiency.name,
            "threshold": best_efficiency.threshold,
            "gap_min": best_efficiency.gap_min,
            "auto_rate": best_efficiency.auto_rate,
            "monthly_cost": best_efficiency.monthly_cost_serper,
        }

    return report


# ============================================================================
# Output
# ============================================================================


def save_report(report: CostReport, output_path: Path) -> None:
    """Save cost report to JSON."""
    result = {
        "simulations": [
            {
                "name": s.name,
                "threshold": s.threshold,
                "gap_min": s.gap_min,
                "total_records": s.total_records,
                "auto_count": s.auto_count,
                "review_count": s.review_count,
                "auto_rate": round(s.auto_rate, 4),
                "places_calls": s.places_calls,
                "monthly_calls": s.monthly_calls,
                "monthly_cost_serper": round(s.monthly_cost_serper, 2),
                "monthly_cost_google": round(s.monthly_cost_google, 2),
                "annual_cost_serper": round(s.annual_cost_serper, 2),
                "annual_cost_google": round(s.annual_cost_google, 2),
            }
            for s in report.simulations
        ],
        "volume_projections": report.volume_projections,
        "recommendations": report.recommendations,
        "pricing_info": {
            "serper_free_tier": SERPER_FREE_TIER,
            "serper_cost_per_call": SERPER_COST_PER_CALL,
            "google_cost_per_call": GOOGLE_COST_PER_CALL,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    LOGGER.info("Wrote cost report to %s", output_path)


def print_summary(report: CostReport) -> None:
    """Print cost simulation summary."""
    LOGGER.info("=" * 70)
    LOGGER.info("PLACES API COST SIMULATION")
    LOGGER.info("=" * 70)

    LOGGER.info("\nTop 5 Configurations by AUTO Rate:")
    top_auto = sorted(report.simulations, key=lambda s: s.auto_rate, reverse=True)[:5]
    for s in top_auto:
        LOGGER.info(
            "  %s: AUTO=%.1f%% | Calls=%d | Cost=$%.2f/mo (Serper)",
            s.name,
            s.auto_rate * 100,
            s.monthly_calls,
            s.monthly_cost_serper,
        )

    LOGGER.info("\nTop 5 Configurations by Cost (Serper):")
    top_cost = sorted(report.simulations, key=lambda s: s.monthly_cost_serper)[:5]
    for s in top_cost:
        LOGGER.info(
            "  %s: Cost=$%.2f/mo | AUTO=%.1f%% | Calls=%d",
            s.name,
            s.monthly_cost_serper,
            s.auto_rate * 100,
            s.monthly_calls,
        )

    LOGGER.info("\nVolume Projections (Normal Config):")
    for label, proj in report.volume_projections.items():
        LOGGER.info(
            "  %s (%d/day): %d calls/mo | $%.2f/mo (Serper) | $%.2f/mo (Google)",
            label.upper(),
            proj["daily_volume"],
            proj["monthly_calls"],
            proj["monthly_cost_serper"],
            proj["monthly_cost_google"],
        )

    if "stay_free" in report.recommendations:
        rec = report.recommendations["stay_free"]
        LOGGER.info("\nRecommendation (Stay Under Free Tier):")
        LOGGER.info(
            "  Use threshold=%.3f gap=%.2f for %.1f%% AUTO rate",
            rec["threshold"],
            rec["gap_min"],
            rec["auto_rate"] * 100,
        )

    LOGGER.info("=" * 70)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = _parse_args()

    # Load data
    df = pd.read_csv(args.inference_path, dtype={"crm_id": str, "siret_candidate": str})
    LOGGER.info("Loaded %d rows from %s", len(df), args.inference_path)

    # Run simulations
    report = run_simulations(df, args.daily_volume)

    # Output
    save_report(report, args.output_path)
    print_summary(report)


if __name__ == "__main__":
    main()
