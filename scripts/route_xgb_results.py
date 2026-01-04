"""
Route XGBoost top-k results into AUTO / REVIEW with optional Places lookup.

PHASE 4: SOTA Routing
=====================
- Output: AUTO or REVIEW only (NO_MATCH is only possible AFTER Places)
- Certainty rules: identify cases where FP is impossible → direct AUTO
- Same-SIREN resolution: auto-resolve ambiguity within same company
- Cost-aware segmentation: threshold varies by segment + Places utility
- Budget modes: aggressive/normal/permissive

Expected input: CSV with top-k rows per CRM (from infer_xgb_two_stage.py).
Required columns (minimum):
  - crm_id
  - rank (optional but recommended)
  - siret_candidate
  - score
Optional columns used for AUTO routing:
  - name_semantic_max
  - name_semantic_second
  - score_gap, score_ratio
Optional columns for Places lookup:
  - crm_name, street_number, street_name, postcode, city
Optional columns for explainability:
  - shap

Usage:
    python scripts/route_xgb_results.py --input-path data/topk.csv --output-path output/routed.csv
    python scripts/route_xgb_results.py --input-path data/topk.csv --output-path output/routed.csv --places-mode
    python scripts/route_xgb_results.py --input-path data/topk.csv --output-path output/routed.csv --budget-mode aggressive
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import yaml
except ImportError:
    yaml = None

from pipe_v6.config import load_config
from pipe_v6.serper_places_client import search_places, build_places_query
from pipe_v6.places_candidate_generator import (
    generate_candidates_by_address,
    generate_candidates_by_radius,
    merge_candidate_pools,
)
from pipe_v6.places_orchestrator import (
    CrmRow,
    XgbTopkCandidate,
    process_review_case,
    enrich_candidate_from_cache,
)
from pipe_v6.places_validator import build_observability_record
from pipe_v6.places_xgb_rescorer import PlacesXgbRescorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


# ============================================================================
# Configuration Loading
# ============================================================================


@dataclass
class RoutingConfig:
    """Parsed routing configuration from YAML."""

    segments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    certainty_rules: Dict[str, Any] = field(default_factory=dict)
    same_siren: Dict[str, Any] = field(default_factory=dict)
    blocking_rules: Dict[str, Any] = field(default_factory=dict)
    promotion_rules: Dict[str, Any] = field(default_factory=dict)
    budget: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "RoutingConfig":
        if yaml is None:
            raise ImportError("PyYAML required for routing config")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            segments=data.get("segments", {}),
            certainty_rules=data.get("certainty_rules", {}),
            same_siren=data.get("same_siren_resolution", {}),
            blocking_rules=data.get("blocking_rules", {}),
            promotion_rules=data.get("promotion_rules", {}),
            budget=data.get("budget", {}),
        )

    @classmethod
    def default(cls) -> "RoutingConfig":
        """Return default configuration without YAML."""
        return cls(
            segments={
                "default": {
                    "threshold": 0.99,
                    "gap_min": 0.05,
                    "places_value": "medium",
                }
            },
            certainty_rules={"enabled": True},
            same_siren={"enabled": True, "min_score": 0.90, "min_name_jaro": 0.80},
            blocking_rules={},
            promotion_rules={},
            budget={"mode": "normal"},
        )


def load_routing_config(path: Path | None) -> RoutingConfig:
    """Load routing config from YAML or return defaults."""
    if path and path.exists():
        try:
            return RoutingConfig.from_yaml(path)
        except Exception as e:
            LOGGER.warning("Failed to load routing config: %s (using defaults)", e)
    return RoutingConfig.default()


# ============================================================================
# Argument Parsing
# ============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Route XGB top-k results into AUTO/REVIEW with optional Places lookup (Phase 4)."
    )
    p.add_argument("--input-path", type=Path, required=True, help="Input CSV with XGB top-k results")
    p.add_argument("--output-path", type=Path, required=True, help="Output CSV with routing decisions")
    p.add_argument("--only-review", action="store_true", help="Export only REVIEW rows")
    p.add_argument("--places-mode", action="store_true", help="Enable Places-guided lookup for REVIEW cases")
    p.add_argument("--legacy-mode", action="store_true", help="Use legacy Google CSE + Brave lookup")
    p.add_argument("--config-path", type=Path, default=None, help="Path to config.yaml")
    p.add_argument(
        "--thresholds",
        type=Path,
        default=Path("configs/routing_thresholds.yaml"),
        help="Path to routing thresholds YAML",
    )
    p.add_argument("--sirene-db", type=Path, default=None, help="Path to SIRENE cache SQLite")
    p.add_argument("--log-places", type=Path, default=None, help="Path to log Places observability JSON")
    p.add_argument("--dry-run", action="store_true", help="Don't make Places API calls (use cache only)")
    p.add_argument(
        "--budget-mode",
        choices=["aggressive", "normal", "permissive"],
        default="normal",
        help="Budget mode: aggressive=minimize Places calls, permissive=maximize recall",
    )
    p.add_argument("--ground-truth", type=Path, default=None, help="Path to ground truth CSV for validation")
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


def _crm_word_count(raw_name: str) -> int:
    if not raw_name:
        return 0
    tokens = re.findall(r"[A-Z0-9]+", raw_name.upper())
    return len([t for t in tokens if t])


def _has_complete_address(row: pd.Series) -> bool:
    street_number = _safe_str(row.get("street_number"))
    street_name = _safe_str(row.get("street_name"))
    crm_address = _safe_str(row.get("crm_address"))
    has_number = bool(street_number) or bool(re.match(r"^\s*\d+", crm_address))
    if street_name:
        has_street = True
    else:
        addr_tokens = re.findall(r"[A-Z0-9]+", crm_address.upper()) if crm_address else []
        has_street = len(addr_tokens) >= 2
    return bool(has_number and has_street)


def _pick_top_rows(group: pd.DataFrame) -> Tuple[pd.Series | None, pd.Series | None, List[pd.Series]]:
    """Pick top rows from a CRM group, return (top1, top2, all_sorted)."""
    if "rank" in group.columns:
        group_sorted = group.sort_values(["rank", "score"], ascending=[True, False])
    else:
        group_sorted = group.sort_values(["score"], ascending=[False])

    rows_list = [row for _, row in group_sorted.iterrows()]
    top1 = rows_list[0] if rows_list else None
    top2 = rows_list[1] if len(rows_list) > 1 else None

    # Parse SHAP features for optional fallback (routing prefers direct columns)
    if top1 is not None:
        shap_raw = top1.get("shap")
        if pd.notna(shap_raw) and shap_raw:
            try:
                shap_data = json.loads(shap_raw)
                features = {f["feature"]: f["value"] for f in shap_data.get("top", [])}
                top1 = top1.copy()
                top1["_shap_features"] = features
            except Exception:
                pass
    return top1, top2, rows_list


def _get_feature(row: pd.Series, name: str, default: float = 0.0) -> float:
    """Fetch feature value from direct columns, fallback to SHAP payload if present."""
    if name in row:
        val = row.get(name)
        if pd.notna(val):
            return _safe_float(val, default)
    shap_data = row.get("_shap_features", {})
    if isinstance(shap_data, dict) and name in shap_data:
        return _safe_float(shap_data.get(name), default)
    return default


# ============================================================================
# Phase 4: Certainty Rules (FP Impossible)
# ============================================================================


def is_absolute_certainty(row: pd.Series, cfg: RoutingConfig) -> Tuple[bool, str]:
    """
    Check if this case has absolute certainty (FP impossible).
    Returns (is_certain, rule_name).
    """
    if not cfg.certainty_rules.get("enabled", True):
        return False, ""

    score = _safe_float(row.get("score"))
    jaro = _get_feature(row, "name_jaro_max")
    tok_overlap = _get_feature(row, "name_token_overlap_max")
    addr_jaro = _get_feature(row, "addr_jaro")
    street_diff = _get_feature(row, "street_number_diff", 9999)
    contains = _get_feature(row, "name_crm_contains_cand_max")

    # R1: Perfect lexical + address match
    r1 = cfg.certainty_rules.get("perfect_match", {})
    if r1.get("enabled", True):
        if (
            jaro >= r1.get("name_jaro_min", 0.98)
            and tok_overlap >= r1.get("name_token_overlap_min", 0.90)
            and addr_jaro >= r1.get("addr_jaro_min", 0.98)
            and street_diff <= r1.get("street_number_diff_max", 0)
            and score >= r1.get("score_min", 0.90)
        ):
            return True, "perfect_match"

    # R2: Identical name (jaro=1.0) + high score + sufficient gap
    # FIX: Added gap check to prevent homonyme FPs (e.g., "CABANON", "SCHLUMBERGER")
    r2 = cfg.certainty_rules.get("identical_name", {})
    if r2.get("enabled", True):
        score_gap = _safe_float(row.get("score_gap"))
        gap_min = r2.get("gap_min", 0.05)  # Default: require 5% gap for identical names
        if jaro >= r2.get("name_jaro_exact", 1.0) and score >= r2.get("score_min", 0.95) and score_gap >= gap_min:
            return True, "identical_name"

    # R3: Model certainty + lexical evidence
    r3 = cfg.certainty_rules.get("model_certainty", {})
    if r3.get("enabled", True):
        if score >= r3.get("score_min", 0.999) and (
            jaro >= r3.get("name_jaro_min", 0.90) or tok_overlap >= r3.get("name_token_overlap_min", 0.70)
        ):
            return True, "model_certainty"

    # R4: Contains match + high score + address match
    r4 = cfg.certainty_rules.get("contains_match", {})
    if r4.get("enabled", True):
        if (
            contains >= r4.get("name_crm_contains_cand_min", 0.95)
            and score >= r4.get("score_min", 0.95)
            and addr_jaro >= r4.get("addr_jaro_min", 0.80)
        ):
            return True, "contains_match"

    return False, ""


# ============================================================================
# Phase 4: Same-SIREN Resolution
# ============================================================================


def resolve_same_siren(
    rows_list: List[pd.Series], cfg: RoutingConfig
) -> Tuple[pd.Series | None, str | None]:
    """
    When top candidates share the same SIREN, resolve automatically.
    Returns (chosen_row, decision_label) or (None, None) if not applicable.
    """
    if not cfg.same_siren.get("enabled", True):
        return None, None

    if len(rows_list) < 2:
        return None, None

    top1 = rows_list[0]
    top2 = rows_list[1]

    siret1 = _safe_str(top1.get("siret_candidate"))
    siret2 = _safe_str(top2.get("siret_candidate"))

    if not siret1 or not siret2:
        return None, None

    siren1 = siret1[:9]
    siren2 = siret2[:9]

    # Not same SIREN
    if siren1 != siren2:
        return None, None

    # Same SIREN: resolve by preferring OUVERT and higher score
    same_siren_rows = [r for r in rows_list if _safe_str(r.get("siret_candidate"))[:9] == siren1]

    prefer_ouvert = cfg.same_siren.get("prefer_ouvert", True)

    def sort_key(r: pd.Series) -> Tuple[int, float]:
        state = _safe_str(r.get("candidate_state", "")).upper()
        state_priority = 0 if state == "OUVERT" else (1 if state == "FERME" else 2)
        if not prefer_ouvert:
            state_priority = 0
        return (state_priority, -_safe_float(r.get("score")))

    same_siren_rows.sort(key=sort_key)
    best = same_siren_rows[0]

    # Check if we can auto-resolve to AUTO or need REVIEW
    min_score = cfg.same_siren.get("min_score", 0.90)
    min_jaro = cfg.same_siren.get("min_name_jaro", 0.80)

    score = _safe_float(top1.get("score"))
    jaro = _get_feature(top1, "name_jaro_max")

    if score >= min_score and jaro >= min_jaro:
        return best, "AUTO_SAME_SIREN"
    else:
        return best, "REVIEW_SAME_SIREN"


# ============================================================================
# Phase 4: Blocking Rules (Force REVIEW)
# ============================================================================


def check_blocking_rules(row: pd.Series, cfg: RoutingConfig) -> Tuple[bool, str]:
    """
    Check if blocking rules force REVIEW.
    Returns (is_blocked, rule_name).
    """
    score = _safe_float(row.get("score"))
    jaro = _get_feature(row, "name_jaro_max")
    tok_overlap = _get_feature(row, "name_token_overlap_max")
    addr_tok_overlap = _get_feature(row, "addr_token_overlap")
    etab = _get_feature(row, "name_sim_max_etab")
    contains = _get_feature(row, "name_crm_contains_cand_max")
    pm = _get_feature(row, "name_sim_max_pm_dirigeant")
    density = _get_feature(row, "address_density", 1.0)
    score_gap = _safe_float(row.get("score_gap"))
    score_ratio = _safe_float(row.get("score_ratio"), 1.0)
    has_evidence = int(_safe_float(row.get("has_name_evidence"), 1.0))

    rules = cfg.blocking_rules

    # B1: No lexical evidence at all
    b1 = rules.get("no_lexical_evidence", {})
    if b1.get("enabled", True):
        if jaro < b1.get("name_jaro_max", 0.50) and tok_overlap < b1.get("name_token_overlap_max", 0.20):
            return True, "no_lexical_evidence"

    # B2: Semantic-only match
    b2 = rules.get("semantic_only", {})
    if b2.get("enabled", True):
        bypass = b2.get("score_bypass", 0.998)
        if score < bypass:
            is_semantic_only = (
                jaro < b2.get("name_jaro_max", 0.60)
                and etab < b2.get("name_sim_max_etab_max", 0.50)
                and contains < b2.get("name_crm_contains_cand_max", 0.50)
                and pm < b2.get("name_sim_max_pm_dirigeant_max", 0.50)
                and tok_overlap < b2.get("name_token_overlap_max", 0.45)
            )
            if is_semantic_only and score >= 0.95:
                return True, "semantic_only"

    # B3: Address-only match (co-location risk)
    b3 = rules.get("address_only", {})
    if b3.get("enabled", True):
        if tok_overlap < b3.get("name_token_overlap_max", 0.10) and addr_tok_overlap > b3.get(
            "addr_token_overlap_min", 0.80
        ):
            return True, "address_only"

    # B4: High address density
    b4 = rules.get("high_density", {})
    if b4.get("enabled", True):
        if density >= b4.get("address_density_min", 10) and jaro < b4.get("name_jaro_max", 0.80):
            return True, "high_density"

    # B5: Weak gap (high ambiguity)
    b5 = rules.get("weak_gap", {})
    if b5.get("enabled", True):
        if score_gap > 0 and score_gap < b5.get("score_gap_max", 0.05):
            return True, "weak_gap"
        if score_ratio > 0 and score_ratio < b5.get("score_ratio_max", 1.02):
            return True, "weak_ratio"

    # B6: No name evidence flag
    b6 = rules.get("no_name_evidence", {})
    if b6.get("enabled", True):
        if has_evidence == 0:
            return True, "no_name_evidence"

    return False, ""


# ============================================================================
# Phase 4: Promotion Rules (Force AUTO)
# ============================================================================


def check_promotion_rules(row: pd.Series, cfg: RoutingConfig) -> Tuple[bool, str]:
    """
    Check if promotion rules force AUTO.
    Returns (is_promoted, rule_name).
    """
    score = _safe_float(row.get("score"))
    etab = _get_feature(row, "name_sim_max_etab")
    contains = _get_feature(row, "name_crm_contains_cand_max")
    pm = _get_feature(row, "name_sim_max_pm_dirigeant")
    tok_overlap = _get_feature(row, "name_token_overlap_max")

    rules = cfg.promotion_rules

    # P1: Strong establishment match
    p1 = rules.get("strong_etab", {})
    if p1.get("enabled", True):
        if score >= p1.get("score_min", 0.95) and etab >= p1.get("name_sim_max_etab_min", 0.70):
            return True, "strong_etab"

    # P2: Contains match
    p2 = rules.get("contains", {})
    if p2.get("enabled", True):
        if score >= p2.get("score_min", 0.90) and contains >= p2.get("name_crm_contains_cand_min", 0.90):
            return True, "contains"

    # P3: PM dirigeant match
    p3 = rules.get("pm_dirigeant", {})
    if p3.get("enabled", True):
        if score >= p3.get("score_min", 0.95) and pm >= p3.get("name_sim_max_pm_dirigeant_min", 0.70):
            return True, "pm_dirigeant"

    # P4: High token overlap
    p4 = rules.get("token_overlap", {})
    if p4.get("enabled", True):
        if score >= p4.get("score_min", 0.98) and tok_overlap >= p4.get("name_token_overlap_min", 0.50):
            return True, "token_overlap"

    return False, ""


# ============================================================================
# Phase 4: Segment Detection & Thresholds
# ============================================================================


def get_segment_config(row: pd.Series, cfg: RoutingConfig) -> Dict[str, Any]:
    """
    Determine segment and return its configuration.
    """
    crm_name = _safe_str(row.get("crm_name"))
    name_words = _crm_word_count(crm_name)
    name_length = _get_feature(row, "name_length_max")
    idf_name = _get_feature(row, "idf_name")
    addr_complete = _has_complete_address(row)

    segments = cfg.segments

    # Priority: short_name > specific segments > default
    if name_words <= 2 or (name_words == 0 and name_length <= 6):
        seg = segments.get("short_name", {})
        if seg:
            return {**segments.get("default", {}), **seg, "segment": "short_name"}

    # Check IDF-based segments
    if idf_name >= 5.0:
        if addr_complete:
            seg = segments.get("unique_name_full_addr", {})
            if seg:
                return {**segments.get("default", {}), **seg, "segment": "unique_name_full_addr"}
        else:
            seg = segments.get("unique_name_partial_addr", {})
            if seg:
                return {**segments.get("default", {}), **seg, "segment": "unique_name_partial_addr"}
    else:
        if addr_complete:
            seg = segments.get("common_name_full_addr", {})
            if seg:
                return {**segments.get("default", {}), **seg, "segment": "common_name_full_addr"}
        else:
            seg = segments.get("common_name_partial_addr", {})
            if seg:
                return {**segments.get("default", {}), **seg, "segment": "common_name_partial_addr"}

    # Default
    default = segments.get("default", {"threshold": 0.99, "gap_min": 0.05, "places_value": "medium"})
    return {**default, "segment": "default"}


def apply_budget_mode(
    threshold: float, gap_min: float, budget_mode: str, cfg: RoutingConfig
) -> Tuple[float, float]:
    """Apply budget mode modifiers to thresholds."""
    budget = cfg.budget
    mode_cfg = budget.get(budget_mode, {})

    thr_mult = mode_cfg.get("threshold_multiplier", 1.0)
    gap_mult = mode_cfg.get("gap_multiplier", 1.0)

    return threshold * thr_mult, gap_min * gap_mult


# ============================================================================
# Phase 4: Main Routing Function (Cost-Aware)
# ============================================================================


def route_cost_aware(
    top1: pd.Series,
    top2: pd.Series | None,
    rows_list: List[pd.Series],
    cfg: RoutingConfig,
    budget_mode: str = "normal",
) -> Tuple[str, Dict[str, Any]]:
    """
    Phase 4 cost-aware routing.
    Returns (decision, metadata).
    """
    metadata: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────
    # STEP 0: Certainty rules (bypass everything)
    # ─────────────────────────────────────────────────────────────
    is_certain, certainty_rule = is_absolute_certainty(top1, cfg)
    if is_certain:
        metadata["certainty_rule"] = certainty_rule
        return "AUTO_CERTAIN", metadata

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Same-SIREN resolution
    # ─────────────────────────────────────────────────────────────
    resolved_row, siren_decision = resolve_same_siren(rows_list, cfg)
    if siren_decision:
        metadata["same_siren_resolved"] = True
        if resolved_row is not None:
            metadata["resolved_siret"] = _safe_str(resolved_row.get("siret_candidate"))
            metadata["resolved_score"] = _safe_float(resolved_row.get("score"))
            metadata["resolved_rank"] = int(_safe_float(resolved_row.get("rank"), 0))
        return siren_decision, metadata

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Blocking rules (force REVIEW)
    # ─────────────────────────────────────────────────────────────
    is_blocked, block_rule = check_blocking_rules(top1, cfg)
    if is_blocked:
        metadata["blocked_by"] = block_rule
        return "REVIEW", metadata

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Get segment configuration
    # ─────────────────────────────────────────────────────────────
    seg_config = get_segment_config(top1, cfg)
    metadata["segment"] = seg_config.get("segment", "default")
    metadata["places_value"] = seg_config.get("places_value", "medium")

    threshold = seg_config.get("threshold", 0.99)
    gap_min = seg_config.get("gap_min", 0.05)

    # Apply budget mode
    threshold, gap_min = apply_budget_mode(threshold, gap_min, budget_mode, cfg)
    metadata["threshold"] = threshold
    metadata["gap_min"] = gap_min

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Promotion rules (force AUTO)
    # ─────────────────────────────────────────────────────────────
    is_promoted, promo_rule = check_promotion_rules(top1, cfg)
    if is_promoted:
        metadata["promoted_by"] = promo_rule
        return "AUTO", metadata

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Gap check (ambiguity)
    # ─────────────────────────────────────────────────────────────
    score_gap = _safe_float(top1.get("score_gap"))
    score_ratio = _safe_float(top1.get("score_ratio"), 1.0)

    if score_gap > 0 and score_gap < gap_min:
        places_value = seg_config.get("places_value", "medium")
        if places_value == "low":
            metadata["review_reason"] = "low_gap_low_places_value"
            return "REVIEW_LOW_PLACES_VALUE", metadata
        metadata["review_reason"] = "low_gap"
        return "REVIEW", metadata

    # ─────────────────────────────────────────────────────────────
    # STEP 6: Final threshold decision
    # ─────────────────────────────────────────────────────────────
    score = _safe_float(top1.get("score"))

    if score >= threshold:
        metadata["passed_threshold"] = True
        return "AUTO", metadata

    # Score below threshold → REVIEW (not NO_MATCH before Places)
    metadata["review_reason"] = "below_threshold"

    # Budget mode: aggressive + low places value → can still AUTO
    if budget_mode == "aggressive" and seg_config.get("places_value") == "low":
        metadata["auto_budget"] = True
        return "AUTO_BUDGET", metadata

    return "REVIEW", metadata


# ============================================================================
# Legacy Routing (Backward Compatibility)
# ============================================================================


def _route_xgb_legacy(top1: pd.Series) -> str:
    """Legacy routing for backward compatibility (Phase 1 rules)."""
    score = _safe_float(top1.get("score"))
    name_semantic_max = _safe_float(top1.get("name_semantic_max"))
    name_semantic_second = _safe_float(top1.get("name_semantic_second"))

    name_jaro_max = _get_feature(top1, "name_jaro_max")
    name_token_overlap_max = _get_feature(top1, "name_token_overlap_max")
    name_sim_max_etab = _get_feature(top1, "name_sim_max_etab")
    name_crm_contains_cand_max = _get_feature(top1, "name_crm_contains_cand_max")
    name_sim_max_pm_dirigeant = _get_feature(top1, "name_sim_max_pm_dirigeant")
    addr_token_overlap = _get_feature(top1, "addr_token_overlap")
    name_length_max = _get_feature(top1, "name_length_max")

    crm_name = _safe_str(top1.get("crm_name"))
    name_word_count = _crm_word_count(crm_name)
    if name_word_count == 0:
        name_word_count = 2 if name_length_max <= 6 else 3
    address_complete = _has_complete_address(top1)

    # Blocking rules
    if name_jaro_max < 0.50 and name_token_overlap_max < 0.20:
        return "REVIEW"

    is_semantic_only = (
        score < 0.998
        and name_jaro_max < 0.6
        and name_sim_max_etab < 0.5
        and name_crm_contains_cand_max < 0.5
        and name_sim_max_pm_dirigeant < 0.5
        and name_token_overlap_max < 0.45
    )
    if is_semantic_only and score >= 0.95:
        return "REVIEW"

    if name_token_overlap_max < 0.10 and addr_token_overlap > 0.80:
        return "REVIEW"

    if name_jaro_max < 0.60 and name_token_overlap_max < 0.30:
        return "REVIEW"

    # Promotion rules
    if score >= 0.95 and name_sim_max_etab >= 0.70:
        return "AUTO"
    if score >= 0.90 and name_crm_contains_cand_max >= 0.9:
        return "AUTO"
    if score >= 0.95 and name_sim_max_pm_dirigeant >= 0.70:
        return "AUTO"
    if score >= 0.98 and name_token_overlap_max >= 0.50:
        return "AUTO"

    # Segmented thresholds
    if name_word_count <= 2:
        threshold = 0.995
    elif name_word_count <= 4:
        threshold = 0.99
    else:
        threshold = 0.98
    if not address_complete:
        threshold = min(0.999, threshold + 0.005)

    if score >= threshold or (score >= 0.95 and name_semantic_max >= 0.75):
        return "AUTO"

    # Phase 4: Everything else is REVIEW (not NO_MATCH)
    return "REVIEW"


# ============================================================================
# CRM / Candidate Helpers (for Places mode)
# ============================================================================


def _build_crm_row(group: pd.DataFrame) -> CrmRow:
    """Build CrmRow from first row of group."""
    first = group.iloc[0]

    crm_address = _safe_str(first.get("crm_address"))
    if crm_address:
        match = re.match(
            r"^\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*(?:[A-Za-z]|BIS|TER|QUATER))?)\s+(.+)",
            crm_address,
            flags=re.IGNORECASE,
        )
        if match:
            street_number = match.group(1).strip()
            street_name = match.group(2).strip()
            street_name = re.sub(r"^\d+\s*(?:[-–]\s*\d+)?\s+", "", street_name).strip()
        else:
            street_number = None
            street_name = crm_address
    else:
        street_number = _safe_str(first.get("street_number")) or None
        street_name = _safe_str(first.get("street_name")) or None

    postcode = _safe_str(first.get("crm_postcode")) or _safe_str(first.get("postcode")) or None
    city = _safe_str(first.get("crm_city")) or _safe_str(first.get("city")) or None
    crm_address_raw = crm_address or " ".join(p for p in [street_number, street_name] if p)

    return CrmRow(
        crm_id=_safe_str(first.get("crm_id")),
        crm_name=_safe_str(first.get("crm_name")),
        crm_address=crm_address_raw or None,
        street_number=street_number,
        street_name=street_name,
        postcode=postcode,
        city=city,
        insee_code=_safe_str(first.get("insee_code")) or _safe_str(first.get("candidate_insee")) or None,
    )


def _build_xgb_topk(rows: List[pd.Series]) -> List[XgbTopkCandidate]:
    """Build XgbTopkCandidate list from sorted rows."""
    topk = []
    for i, row in enumerate(rows[:20]):  # Phase 4: recall@20
        topk.append(
            XgbTopkCandidate(
                siret=_safe_str(row.get("siret_candidate")),
                score=_safe_float(row.get("score")),
                rank=i,
                features={
                    "denomination": _safe_str(row.get("candidate_name")) or _safe_str(row.get("denomination")),
                    "enseigne1": _safe_str(row.get("enseigne1")),
                    "street_number": _safe_str(row.get("street_number_candidate")),
                    "street_name": _safe_str(row.get("street_name_candidate")),
                    "address_full": _safe_str(row.get("candidate_addr")),
                    "postcode": _safe_str(row.get("candidate_postcode")) or _safe_str(row.get("postcode_candidate")),
                    "city": _safe_str(row.get("candidate_city")) or _safe_str(row.get("city_candidate")),
                    "insee_code": _safe_str(row.get("candidate_insee")),
                },
            )
        )
    return topk


# ============================================================================
# Metrics Tracking
# ============================================================================


@dataclass
class RoutingMetrics:
    """Phase 4 routing metrics."""

    total_records: int = 0
    auto_count: int = 0
    auto_certain_count: int = 0
    auto_same_siren_count: int = 0
    auto_budget_count: int = 0
    review_count: int = 0
    review_low_places_value_count: int = 0
    review_same_siren_count: int = 0
    match_places_count: int = 0
    no_match_count: int = 0  # Only after Places

    # Quality metrics (if ground truth provided)
    auto_correct: int = 0
    auto_incorrect: int = 0
    review_would_be_correct: int = 0

    def add_decision(self, decision: str):
        self.total_records += 1
        if decision.startswith("AUTO"):
            self.auto_count += 1
            if decision == "AUTO_CERTAIN":
                self.auto_certain_count += 1
            elif decision == "AUTO_SAME_SIREN":
                self.auto_same_siren_count += 1
            elif decision == "AUTO_BUDGET":
                self.auto_budget_count += 1
        elif decision.startswith("REVIEW"):
            self.review_count += 1
            if decision == "REVIEW_LOW_PLACES_VALUE":
                self.review_low_places_value_count += 1
            elif decision == "REVIEW_SAME_SIREN":
                self.review_same_siren_count += 1
        elif decision == "MATCH_PLACES":
            self.match_places_count += 1
        elif decision == "NO_MATCH":
            self.no_match_count += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "auto_count": self.auto_count,
            "auto_rate": self.auto_count / self.total_records if self.total_records else 0,
            "auto_certain": self.auto_certain_count,
            "auto_same_siren": self.auto_same_siren_count,
            "auto_budget": self.auto_budget_count,
            "review_count": self.review_count,
            "review_rate": self.review_count / self.total_records if self.total_records else 0,
            "review_low_places_value": self.review_low_places_value_count,
            "match_places": self.match_places_count,
            "no_match": self.no_match_count,
            "estimated_places_calls": self.review_count,
            "estimated_cost_usd": self.review_count * 0.001,
        }


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    args = _parse_args()

    # Load configs
    config = load_config(args.config_path)
    routing_cfg = load_routing_config(args.thresholds)

    # Override places mode from args
    if args.places_mode:
        config.places_lookup_mode = "places"
    elif args.legacy_mode:
        config.places_lookup_mode = "legacy"

    LOGGER.info("Loading input from %s", args.input_path)
    df = pd.read_csv(args.input_path)

    if "crm_id" not in df.columns:
        raise ValueError("Missing required column: crm_id")
    if "score" not in df.columns:
        raise ValueError("Missing required column: score")

    LOGGER.info("Budget mode: %s", args.budget_mode)
    LOGGER.info("Using routing config: %s", args.thresholds if args.thresholds.exists() else "defaults")

    # Open SIRENE cache if Places mode
    conn = None
    if args.places_mode:
        sirene_path = args.sirene_db or config.sqlite_path
        if not sirene_path.exists():
            LOGGER.warning("SIRENE cache not found at %s, Places lookup will be limited", sirene_path)
        else:
            conn = sqlite3.connect(sirene_path)
            conn.row_factory = sqlite3.Row
            LOGGER.info("Opened SIRENE cache at %s", sirene_path)

    # Initialize XGB rescorer (once) for Places mode
    xgb_rescorer = None
    if args.places_mode:
        try:
            xgb_rescorer = PlacesXgbRescorer(config)
        except Exception as exc:
            LOGGER.error("Failed to load XGB rescorer: %s (fallback to heuristic)", exc)

    rows_out: List[dict] = []
    places_logs: List[dict] = []
    metrics = RoutingMetrics()

    groups = list(df.groupby("crm_id", dropna=False))
    total = len(groups)

    for idx, (crm_id, group) in enumerate(groups, 1):
        top1, top2, all_rows = _pick_top_rows(group)

        if top1 is None:
            continue

        # Phase 4 routing
        xgb_status, route_meta = route_cost_aware(top1, top2, all_rows, routing_cfg, args.budget_mode)

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
            "name_semantic_max": _safe_float(top1.get("name_semantic_max")),
            "name_jaro_max": _get_feature(top1, "name_jaro_max"),
            "name_token_overlap_max": _get_feature(top1, "name_token_overlap_max"),
            "segment": route_meta.get("segment"),
            "threshold_used": route_meta.get("threshold"),
            "places_value": route_meta.get("places_value"),
        }

        # Add routing metadata
        for key in ["certainty_rule", "blocked_by", "promoted_by", "review_reason", "same_siren_resolved"]:
            if key in route_meta:
                row_out[f"route_{key}"] = route_meta[key]
        for key in ["resolved_siret", "resolved_score", "resolved_rank"]:
            if key in route_meta:
                row_out[key] = route_meta[key]

        # If same-SIREN resolution picked a different establishment, honor it in outputs
        if route_meta.get("resolved_siret"):
            row_out["chosen_siret_xgb"] = route_meta["resolved_siret"]

        # Places lookup for REVIEW cases
        final_status = xgb_status
        final_siret = row_out["chosen_siret_xgb"]

        if args.places_mode and xgb_status.startswith("REVIEW") and conn is not None:
            LOGGER.info("[%d/%d] Processing REVIEW case: crm_id=%s", idx, total, crm_id)

            try:
                crm_row = _build_crm_row(group)
                xgb_topk = _build_xgb_topk(all_rows)

                result = process_review_case(
                    crm_row=crm_row,
                    xgb_topk=xgb_topk,
                    config=config,
                    conn=conn,
                    xgb_status=xgb_status,
                    logger=LOGGER,
                    xgb_rescorer=xgb_rescorer,
                    cache_only=args.dry_run,
                )

                # Update outputs
                final_status = result.final_status
                final_siret = result.final_siret

                row_out["places_status"] = result.places_decision.decision if result.places_decision else None
                row_out["places_score_after"] = (
                    result.places_decision.score_after if result.places_decision else None
                )
                row_out["places_gap_after"] = result.places_decision.gap_after if result.places_decision else None
                row_out["places_pool_size"] = result.places_decision.pool_size if result.places_decision else None
                row_out["places_top1_changed"] = (
                    result.places_decision.top1_changed if result.places_decision else None
                )

                # Log observability
                if result.observability:
                    places_logs.append(result.observability)

            except Exception as exc:
                LOGGER.error("Places processing failed for crm_id=%s: %s", crm_id, exc)
                row_out["places_error"] = str(exc)

        row_out["final_status"] = final_status
        row_out["chosen_siret_final"] = final_siret
        rows_out.append(row_out)
        metrics.add_decision(final_status)

    # Close DB
    if conn:
        conn.close()

    # Write output
    out_df = pd.DataFrame(rows_out)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    LOGGER.info("Wrote %d rows to %s", len(out_df), args.output_path)

    # Write Places logs
    if args.log_places and places_logs:
        args.log_places.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_places, "w", encoding="utf-8") as f:
            json.dump(places_logs, f, ensure_ascii=False, indent=2)
        LOGGER.info("Wrote %d Places observability records to %s", len(places_logs), args.log_places)

    # Summary
    summary = metrics.summary()
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 4 ROUTING SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("Total records: %d", summary["total_records"])
    LOGGER.info("AUTO rate: %.1f%% (%d)", summary["auto_rate"] * 100, summary["auto_count"])
    LOGGER.info("  - AUTO_CERTAIN: %d", summary["auto_certain"])
    LOGGER.info("  - AUTO_SAME_SIREN: %d", summary["auto_same_siren"])
    LOGGER.info("  - AUTO_BUDGET: %d", summary["auto_budget"])
    LOGGER.info("REVIEW rate: %.1f%% (%d)", summary["review_rate"] * 100, summary["review_count"])
    LOGGER.info("  - REVIEW_LOW_PLACES_VALUE: %d", summary["review_low_places_value"])
    if args.places_mode:
        LOGGER.info("MATCH_PLACES: %d", summary["match_places"])
        LOGGER.info("NO_MATCH (post-Places): %d", summary["no_match"])
    LOGGER.info("Estimated Places API calls: %d", summary["estimated_places_calls"])
    LOGGER.info("Estimated cost: $%.2f", summary["estimated_cost_usd"])
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
