"""Route XGBoost top-k results into AUTO / REVIEW with optional Places fallback.

SIMPLIFIED V7 ROUTING
=====================
- Risk Model decides AUTO vs REVIEW (threshold 0.835)
- Places fallback: simple "Places as CRM Repair" approach
- Post-Places: MATCH_PLACES or NO_MATCH (binary, no REVIEW)

Expected input: CSV with top-k rows per CRM (from infer_xgb_two_stage.py).

Usage:
    python scripts/route_xgb_results.py --input-path data/topk.csv --output-path output/routed.csv
    python scripts/route_xgb_results.py --input-path data/topk.csv --output-path output/routed.csv --places-mode
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xgb_matcher.routing_risk import build_feature_row, default_feature_columns


class IsotonicCalibrator:
    def __init__(self, base_estimator, iso_reg):
        self.base_estimator = base_estimator
        self.iso_reg = iso_reg

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.iso_reg.predict(proba)
        calibrated = np.clip(calibrated, 0, 1)
        return np.column_stack([1 - calibrated, calibrated])


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


# ============================================================================
# Argument Parsing
# ============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Route XGB top-k results into AUTO/REVIEW with optional Places fallback."
    )
    p.add_argument("--input-path", type=Path, required=True, help="Input CSV with XGB top-k results")
    p.add_argument("--output-path", type=Path, required=True, help="Output CSV with routing decisions")
    p.add_argument("--only-review", action="store_true", help="Export only REVIEW rows")
    p.add_argument("--places-mode", action="store_true", help="Enable Places fallback for REVIEW cases")
    p.add_argument("--config-path", type=Path, default=None, help="Path to config.yaml")
    p.add_argument("--dry-run", action="store_true", help="Don't make Places API calls (use cache only)")
    p.add_argument("--risk-model", type=Path, default=None, help="Path to routing risk model pickle")
    p.add_argument("--risk-calibrator", type=Path, default=None, help="Path to routing risk calibrator pickle")
    p.add_argument("--risk-meta", type=Path, default=None, help="Path to routing risk meta JSON")
    p.add_argument("--risk-threshold", type=float, default=None, help="Override risk threshold")
    p.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v4_active"), help="Partitions directory")
    p.add_argument("--ranker-model", type=Path, default=None, help="Ranker model path")
    p.add_argument("--decider-model", type=Path, default=None, help="Decider model path")
    return p.parse_args()


# ============================================================================
# Utility Functions
# ============================================================================


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_str(value, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value).strip() or default


def _normalize_siret(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().replace(" ", "")
    if not s or s.lower() == "nan":
        return None
    if "e" in s.lower():
        try:
            s = str(int(float(s)))
        except (ValueError, OverflowError):
            pass
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(14)


def _pick_top_rows(group: pd.DataFrame) -> Tuple[pd.Series | None, pd.Series | None, List[pd.Series]]:
    """Pick top rows from a CRM group, return (top1, top2, all_sorted)."""
    if "rank" in group.columns:
        group_sorted = group.sort_values(["rank", "score"], ascending=[True, False])
    else:
        group_sorted = group.sort_values(["score"], ascending=[False])

    rows_list = [row for _, row in group_sorted.iterrows()]
    top1 = rows_list[0] if rows_list else None
    top2 = rows_list[1] if len(rows_list) > 1 else None
    return top1, top2, rows_list


def _get_feature(row: pd.Series, name: str, default: float = 0.0) -> float:
    """Fetch feature value from row."""
    if name in row.index:
        val = row.get(name)
        if pd.notna(val):
            return _safe_float(val, default)
    return default


# ============================================================================
# Risk Model Routing
# ============================================================================


def _load_risk_assets(
    model_path: Path | None,
    calibrator_path: Path | None,
    meta_path: Path | None,
) -> Tuple[object | None, object | None, List[str], float | None]:
    model = None
    calibrator = None
    feature_names: List[str] = []
    threshold = None

    if meta_path and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        feature_names = meta.get("features") or []
        threshold = meta.get("threshold")
        if not model_path and meta.get("model_path"):
            model_path = Path(meta.get("model_path"))
        if not calibrator_path and meta.get("calibrator_path"):
            calibrator_path = Path(meta.get("calibrator_path"))

    if model_path and model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    if calibrator_path and calibrator_path.exists():
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)

    if not feature_names:
        feature_names = default_feature_columns()

    return model, calibrator, feature_names, threshold


def route_with_risk_model(
    top1: pd.Series,
    top2: pd.Series | None,
    risk_model: object,
    risk_calibrator: object | None,
    risk_features: List[str],
    risk_threshold: float,
) -> Tuple[str, Dict[str, Any]]:
    """Route using risk metamodel."""
    metadata: Dict[str, Any] = {}

    # Risk model scoring
    features = build_feature_row(top1, top2, risk_features)
    if risk_calibrator is not None:
        proba = risk_calibrator.predict_proba([features])[:, 1]  # type: ignore
    else:
        proba = risk_model.predict_proba([features])[:, 1]  # type: ignore

    score = float(proba[0])
    metadata["risk_score"] = score
    metadata["risk_threshold"] = risk_threshold

    if score >= risk_threshold:
        return "AUTO_RISK", metadata
    return "REVIEW", metadata


# ============================================================================
# Metrics Tracking
# ============================================================================


@dataclass
class RoutingMetrics:
    """Routing metrics."""

    total_records: int = 0
    auto_count: int = 0
    review_count: int = 0
    match_places_count: int = 0
    no_match_count: int = 0

    def add_decision(self, decision: str):
        self.total_records += 1
        if decision.startswith("AUTO"):
            self.auto_count += 1
        elif decision == "REVIEW":
            self.review_count += 1
        elif decision == "MATCH_PLACES":
            self.match_places_count += 1
        elif decision == "NO_MATCH":
            self.no_match_count += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "auto_count": self.auto_count,
            "auto_rate": self.auto_count / self.total_records if self.total_records else 0,
            "review_count": self.review_count,
            "review_rate": self.review_count / self.total_records if self.total_records else 0,
            "match_places": self.match_places_count,
            "no_match": self.no_match_count,
        }


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    args = _parse_args()

    # Load risk model
    risk_model, risk_calibrator, risk_features, risk_threshold = _load_risk_assets(
        args.risk_model,
        args.risk_calibrator,
        args.risk_meta,
    )
    if args.risk_threshold is not None:
        risk_threshold = args.risk_threshold
    if risk_model is not None and risk_threshold is None:
        risk_threshold = 0.835

    if risk_model is None:
        # Try default paths
        default_meta = Path("models/routing_risk_meta.json")
        default_model = Path("models/routing_risk_model.pkl")
        if default_meta.exists() and default_model.exists():
            risk_model, risk_calibrator, risk_features, risk_threshold = _load_risk_assets(
                default_model, None, default_meta
            )
            LOGGER.info("Loaded default risk model from %s", default_model)

    LOGGER.info("Loading input from %s", args.input_path)
    df = pd.read_csv(args.input_path, dtype={"crm_id": str, "siret_candidate": str})

    if "siret_candidate" in df.columns:
        df["siret_candidate"] = df["siret_candidate"].apply(_normalize_siret)

    if "crm_id" not in df.columns:
        raise ValueError("Missing required column: crm_id")
    if "score" not in df.columns:
        raise ValueError("Missing required column: score")

    if risk_model is not None:
        LOGGER.info("Routing risk model enabled (threshold=%.3f)", risk_threshold)
    else:
        LOGGER.warning("No risk model loaded - all cases will be REVIEW")

    # Initialize Places fallback engine if needed
    xgb_engine = None
    config = None
    if args.places_mode:
        from pipe_v6.config import load_config
        from xgb_matcher.infer import XgbInferenceEngine

        config = load_config(args.config_path)

        # Find model paths
        ranker_path = args.ranker_model
        decider_path = args.decider_model
        if not ranker_path or not decider_path:
            model_dir = Path("models")
            ranker_files = sorted(model_dir.glob("xgbranker_*.json"), reverse=True)
            decider_files = sorted(model_dir.glob("xgb_decider_*.json"), reverse=True)
            if ranker_files and not ranker_path:
                ranker_path = ranker_files[0]
            if decider_files and not decider_path:
                decider_path = decider_files[0]

        if ranker_path and decider_path:
            try:
                xgb_engine = XgbInferenceEngine.from_models(
                    ranker_path=ranker_path,
                    decider_path=decider_path,
                    partitions_dir=args.partitions_dir,
                    risk_model_path=args.risk_model or Path("models/routing_risk_model.pkl"),
                    risk_meta_path=args.risk_meta or Path("models/routing_risk_meta.json"),
                )
                LOGGER.info("Places fallback engine initialized")
            except Exception as exc:
                LOGGER.error("Failed to initialize Places fallback engine: %s", exc)
                xgb_engine = None
        else:
            LOGGER.warning("Could not find ranker/decider models for Places fallback")

    rows_out: List[dict] = []
    metrics = RoutingMetrics()

    groups = list(df.groupby("crm_id", dropna=False))
    total = len(groups)

    for idx, (crm_id, group) in enumerate(groups, 1):
        top1, top2, all_rows = _pick_top_rows(group)

        if top1 is None:
            continue

        # Route with risk model
        if risk_model is not None:
            xgb_status, route_meta = route_with_risk_model(
                top1,
                top2,
                risk_model,
                risk_calibrator,
                risk_features,
                float(risk_threshold) if risk_threshold is not None else 0.835,
            )
        else:
            xgb_status = "REVIEW"
            route_meta = {}

        # Skip AUTO cases if requested
        if args.only_review and xgb_status.startswith("AUTO"):
            continue

        # Base output row
        row_out = {
            "crm_id": crm_id,
            "xgb_status": xgb_status,
            "chosen_siret_xgb": _safe_str(top1.get("siret_candidate")),
            "score_top1": _safe_float(top1.get("score")),
            "score_top2": _safe_float(top2.get("score") if top2 is not None else None),
            "score_gap": _safe_float(top1.get("score_gap")),
            "score_ratio": _safe_float(top1.get("score_ratio")),
            "crm_name": _safe_str(top1.get("crm_name")),
            "crm_address": _safe_str(top1.get("crm_address")),
            "crm_postcode": _safe_str(top1.get("crm_postcode")),
            "crm_city": _safe_str(top1.get("crm_city")),
        }

        # Add routing metadata
        for key in ["risk_score", "risk_threshold"]:
            if key in route_meta:
                row_out[f"route_{key}"] = route_meta[key]

        # Determine final status
        final_status = xgb_status
        final_siret = str(row_out["chosen_siret_xgb"]).strip().replace(".0", "").zfill(14) if pd.notna(row_out["chosen_siret_xgb"]) else None

        # Places fallback for REVIEW cases
        if args.places_mode and xgb_status == "REVIEW" and xgb_engine is not None and config is not None:
            LOGGER.info("[%d/%d] Processing REVIEW case: crm_id=%s", idx, total, crm_id)

            from pipe_v6.places_fallback import fallback_with_places

            try:
                result = fallback_with_places(
                    crm_id=str(crm_id),
                    crm_name=_safe_str(top1.get("crm_name")),
                    crm_address=_safe_str(top1.get("crm_address")) or None,
                    crm_postcode=_safe_str(top1.get("crm_postcode")) or None,
                    crm_city=_safe_str(top1.get("crm_city")) or None,
                    crm_insee=None,
                    config=config,
                    engine=xgb_engine,
                    logger=LOGGER,
                    cache_only=args.dry_run,
                )

                final_status = result.final_status
                if result.final_siret:
                    final_siret = result.final_siret

                row_out["places_used"] = result.places_used
                row_out["places_title"] = result.places_title
                row_out["places_address"] = result.places_address
                row_out["places_postcode"] = result.places_postcode
                row_out["places_dept_guard_passed"] = result.dept_guard_passed
                row_out["places_xgb_rerun_status"] = result.xgb_rerun_status
                row_out["places_xgb_rerun_score"] = result.xgb_rerun_score
                row_out["places_reason"] = result.reason

            except Exception as exc:
                LOGGER.error("Places fallback failed for crm_id=%s: %s", crm_id, exc)
                row_out["places_error"] = str(exc)

        row_out["final_status"] = final_status
        row_out["chosen_siret_final"] = final_siret
        rows_out.append(row_out)
        metrics.add_decision(final_status)

    # Write output
    out_df = pd.DataFrame(rows_out)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    LOGGER.info("Wrote %d rows to %s", len(out_df), args.output_path)

    # Summary
    summary = metrics.summary()
    LOGGER.info("=" * 60)
    LOGGER.info("ROUTING SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("Total records: %d", summary["total_records"])
    LOGGER.info("AUTO rate: %.1f%% (%d)", summary["auto_rate"] * 100, summary["auto_count"])
    LOGGER.info("REVIEW rate: %.1f%% (%d)", summary["review_rate"] * 100, summary["review_count"])
    if args.places_mode:
        LOGGER.info("MATCH_PLACES: %d", summary["match_places"])
        LOGGER.info("NO_MATCH (post-Places): %d", summary["no_match"])
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
