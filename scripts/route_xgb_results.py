"""
Route XGBoost top-k results into AUTO / REVIEW / NO_MATCH with optional Places lookup.

Expected input: CSV with top-k rows per CRM (from infer_xgb_matcher_topk.py).
Required columns (minimum):
  - crm_id
  - rank (optional but recommended)
  - siret_candidate
  - score
Optional columns used for AUTO routing:
  - name_semantic_max
  - name_semantic_second
Optional columns for Places lookup:
  - crm_name, street_number, street_name, postcode, city
Optional columns for explainability:
  - shap

Usage:
    python scripts/route_xgb_results.py --input-path data/topk.csv --output-path output/routed.csv
    python scripts/route_xgb_results.py --input-path data/topk.csv --output-path output/routed.csv --places-mode
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Route XGB top-k results into AUTO/REVIEW/NO_MATCH with optional Places lookup."
    )
    p.add_argument("--input-path", type=Path, required=True, help="Input CSV with XGB top-k results")
    p.add_argument("--output-path", type=Path, required=True, help="Output CSV with routing decisions")
    p.add_argument("--only-review", action="store_true", help="Export only REVIEW/NO_MATCH rows")
    p.add_argument("--places-mode", action="store_true", help="Enable Places-guided lookup for REVIEW cases")
    p.add_argument("--legacy-mode", action="store_true", help="Use legacy Google CSE + Brave lookup")
    p.add_argument("--config-path", type=Path, default=None, help="Path to config.yaml")
    p.add_argument("--sirene-db", type=Path, default=None, help="Path to SIRENE cache SQLite")
    p.add_argument("--log-places", type=Path, default=None, help="Path to log Places observability JSON")
    p.add_argument("--dry-run", action="store_true", help="Don't make Places API calls (use cache only)")
    return p.parse_args()


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


def _pick_top_rows(group: pd.DataFrame) -> tuple[pd.Series, pd.Series | None, list[pd.Series]]:
    """Pick top rows from a CRM group, return (top1, top2, all_sorted)."""
    if "rank" in group.columns:
        group_sorted = group.sort_values(["rank", "score"], ascending=[True, False])
    else:
        group_sorted = group.sort_values(["score"], ascending=[False])
    
    rows_list = [row for _, row in group_sorted.iterrows()]
    top1 = rows_list[0] if rows_list else None
    top2 = rows_list[1] if len(rows_list) > 1 else None
    
    # Parse SHAP features for routing rules
    if top1 is not None:
        shap_raw = top1.get("shap")
        if pd.notna(shap_raw) and shap_raw:
            try:
                import json
                shap_data = json.loads(shap_raw)
                features = {f['feature']: f['value'] for f in shap_data.get('top', [])}
                top1 = top1.copy()
                top1["_shap_features"] = features
            except Exception:
                pass
    return top1, top2, rows_list


def _route_xgb(top1: pd.Series) -> str:
    """Apply XGBoost routing rules v2.1.
    
    Changes from v2.0:
    - Stricter blocking for semantic-only matches even with low token overlap
    - Lower score threshold for PM dirigeant matches
    - Promote contains matches at lower score threshold
    """
    score = _safe_float(top1.get("score"))
    name_semantic_max = _safe_float(top1.get("name_semantic_max"))
    name_semantic_second = _safe_float(top1.get("name_semantic_second"))
    
    # Access SHAP features from parsed shap column if available
    shap_data = top1.get("_shap_features", {})
    name_jaro_max = _safe_float(shap_data.get("name_jaro_max", top1.get("name_jaro_max")))
    name_token_overlap_max = _safe_float(shap_data.get("name_token_overlap_max", top1.get("name_token_overlap_max")))
    name_sim_max_etab = _safe_float(shap_data.get("name_sim_max_etab", top1.get("name_sim_max_etab")))
    name_crm_contains_cand_max = _safe_float(shap_data.get("name_crm_contains_cand_max", top1.get("name_crm_contains_cand_max")))
    name_sim_max_pm_dirigeant = _safe_float(shap_data.get("name_sim_max_pm_dirigeant", top1.get("name_sim_max_pm_dirigeant")))
    
    # BLOCK: Semantic-only match without solid lexical evidence
    # Must have LOW values for ALL: jaro, etab, pm, token_overlap
    # E.g., "RUBIX FRANCE" -> "FRANCE MECANIQUE" (token_overlap=0.33, etab=0)
    # Exception: score >= 0.998 bypasses blocking (model is very confident)
    is_semantic_only = (
        score < 0.998 and  # Allow very high scores through
        name_jaro_max < 0.6 and
        name_sim_max_etab < 0.5 and
        name_crm_contains_cand_max < 0.5 and
        name_sim_max_pm_dirigeant < 0.5 and
        name_token_overlap_max < 0.45
    )
    
    if is_semantic_only and score >= 0.95:
        # Force REVIEW even if score is high
        return "REVIEW"
    
    # PROMOTE: Strong match via various signals
    
    # P1: Strong establishment name match (score >= 0.95)
    is_strong_etab_match = (
        score >= 0.95 and
        name_sim_max_etab >= 0.70
    )
    if is_strong_etab_match:
        return "AUTO"
    
    # P2: Contains match at lower threshold (score >= 0.90)
    # E.g., "Timcod Rhone-Alpes" -> APLISTORE (Timcod in UL name)
    is_contains_match = (
        score >= 0.90 and
        name_crm_contains_cand_max >= 0.9
    )
    if is_contains_match:
        return "AUTO"
    
    # P3: Strong PM dirigeant match (score >= 0.95, pm >= 0.70)
    # E.g., "METALDYNE" -> METALDYNE INTERNATIONAL via PM dirigeant
    is_pm_match = (
        score >= 0.95 and
        name_sim_max_pm_dirigeant >= 0.70
    )
    if is_pm_match:
        return "AUTO"
    
    # P4: High token overlap (score >= 0.98, tok_overlap >= 0.50)
    # E.g., "ISOJET" -> "ISOJET EQUIPEMENTS" (tok_overlap=0.50)
    # E.g., "JOURDAN ET MARZE" -> "JOURDAN ET MARZE SUC" (tok_overlap=0.75)
    is_token_match = (
        score >= 0.98 and
        name_token_overlap_max >= 0.50
    )
    if is_token_match:
        return "AUTO"
    
    # Standard v1.0 rules
    auto = (
        score >= 0.99
        or (score >= 0.95 and name_semantic_max >= 0.75)
        or (score >= 0.98 and name_semantic_second >= 0.65)
    )
    if auto:
        return "AUTO"
    if score >= 0.70:
        return "REVIEW"
    return "NO_MATCH"


def _build_crm_row(group: pd.DataFrame) -> CrmRow:
    """Build CrmRow from first row of group.
    
    Handles both column naming conventions:
    - Legacy: crm_address, crm_postcode, crm_city
    - New: street_number, street_name, postcode, city
    """
    first = group.iloc[0]
    
    # Handle address: try crm_address first, then street_number/street_name
    crm_address = _safe_str(first.get("crm_address"))
    if crm_address:
        # Parse address to extract street number and name
        import re
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
    
    # Handle postcode: try crm_postcode first, then postcode
    postcode = _safe_str(first.get("crm_postcode")) or _safe_str(first.get("postcode")) or None
    
    # Handle city: try crm_city first, then city
    city = _safe_str(first.get("crm_city")) or _safe_str(first.get("city")) or None
    
    # Best-effort raw address
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


def _build_xgb_topk(rows: list[pd.Series]) -> list[XgbTopkCandidate]:
    """Build XgbTopkCandidate list from sorted rows.
    
    Handles both column naming conventions for candidate data.
    """
    topk = []
    for i, row in enumerate(rows[:10]):  # Max 10
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


def main() -> None:
    args = _parse_args()
    
    # Load config
    config = load_config(args.config_path)
    
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

    rows_out: list[dict] = []
    places_logs: list[dict] = []
    
    groups = list(df.groupby("crm_id", dropna=False))
    total = len(groups)
    
    for idx, (crm_id, group) in enumerate(groups, 1):
        top1, top2, all_rows = _pick_top_rows(group)
        
        if top1 is None:
            continue
            
        xgb_status = _route_xgb(top1)
        
        # Skip AUTO cases if requested
        if args.only_review and xgb_status == "AUTO":
            continue

        # Base output row
        row_out = {
            "crm_id": crm_id,
            "xgb_status": xgb_status,
            "chosen_siret_xgb": _safe_str(top1.get("siret_candidate")),
            "score_top1": _safe_float(top1.get("score")),
            "score_top2": _safe_float(top2.get("score") if top2 is not None else None),
            "name_semantic_max": _safe_float(top1.get("name_semantic_max")),
            "name_semantic_second": _safe_float(top1.get("name_semantic_second")),
            "xgb_shap_top1": top1.get("shap"),
        }

        # Places lookup for REVIEW/NO_MATCH cases
        final_status = xgb_status
        final_siret = row_out["chosen_siret_xgb"]
        
        if args.places_mode and xgb_status in ("REVIEW", "NO_MATCH") and conn is not None:
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
                row_out["places_gap_after"] = (
                    result.places_decision.gap_after if result.places_decision else None
                )
                row_out["places_pool_size"] = (
                    result.places_decision.pool_size if result.places_decision else None
                )
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
    if rows_out:
        status_counts = out_df["final_status"].value_counts().to_dict()
        LOGGER.info("Routing summary: %s", status_counts)
        
        if args.places_mode:
            places_promoted = len(out_df[out_df["final_status"] == "MATCH_PLACES"])
            LOGGER.info("Places promoted %d REVIEW cases to MATCH_PLACES", places_promoted)


if __name__ == "__main__":
    main()
