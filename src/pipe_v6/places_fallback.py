"""Simple Places fallback: "Places as CRM Repair".

This module replaces the complex legacy Places system (orchestrator, candidate_generator,
validator, xgb_rescorer) with a radically simpler approach:

1. Call Serper Places API with CRM name + address
2. Dept-guard: reject if Places postcode[:2] != CRM postcode[:2]
3. Use Google's top-1 as "repaired CRM" and rerun EXACT same XGB pipeline
4. If XGB returns AUTO -> MATCH_PLACES, else -> NO_MATCH

Key insight: Google knows business identity (successions, name changes, fund transfers).
We trust Google for "which business" and XGB for "which SIRET".
The same Risk Model (calibrated at 0.835) validates both.

Usage:
    from pipe_v6.places_fallback import fallback_with_places, PlacesFallbackResult

    result = fallback_with_places(
        crm_input=crm,
        engine=xgb_engine,
        config=config,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from .config import PipelineConfig
from .serper_places_client import (
    PlacesResult,
    build_places_query,
    search_places,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PlacesFallbackResult:
    """Result from Places fallback processing."""
    
    crm_id: str
    original_status: str
    final_status: str  # "MATCH_PLACES" or "NO_MATCH"
    final_siret: str | None
    places_used: bool
    places_title: str | None = None
    places_address: str | None = None
    places_postcode: str | None = None
    dept_guard_passed: bool | None = None
    xgb_rerun_status: str | None = None
    xgb_rerun_score: float | None = None
    reason: str | None = None


def _normalize_postcode(p: str | None) -> str:
    """Normalize postcode to 5 digits."""
    if not p:
        return ""
    digits = "".join(c for c in str(p) if c.isdigit())
    return digits.zfill(5) if digits else ""


def _dept_from_postcode(postcode: str | None) -> str:
    """Extract department code (2 chars) from postcode."""
    norm = _normalize_postcode(postcode)
    if len(norm) >= 2:
        return norm[:2]
    return ""


def fallback_with_places(
    crm_id: str,
    crm_name: str,
    crm_address: str | None,
    crm_postcode: str | None,
    crm_city: str | None,
    crm_insee: str | None,
    config: PipelineConfig,
    engine: Any,  # XgbInferenceEngine - avoid circular import
    logger: logging.Logger | None = None,
    cache_only: bool = False,
) -> PlacesFallbackResult:
    """Simple Places fallback: use Google top-1 as "repaired CRM" and rerun XGB.
    
    Args:
        crm_id: CRM identifier
        crm_name: Original CRM name
        crm_address: Original CRM address
        crm_postcode: Original CRM postcode
        crm_city: Original CRM city
        crm_insee: Original CRM INSEE code
        config: Pipeline configuration with Serper API key
        engine: XgbInferenceEngine instance for rerun
        logger: Optional logger
        cache_only: If True, only use cached Places results
    
    Returns:
        PlacesFallbackResult with final_status = "MATCH_PLACES" or "NO_MATCH"
    """
    log = logger or LOGGER
    
    # Build Places query
    query = build_places_query(
        crm_name=crm_name,
        street_number=None,  # Let Google parse it
        street_name=crm_address,
        postcode=crm_postcode,
        city=crm_city,
    )

    if not query.strip():
        log.info("[Places Fallback] Empty query for crm_id=%s", crm_id)
        return PlacesFallbackResult(
            crm_id=crm_id,
            original_status="REVIEW",
            final_status="NO_MATCH",
            final_siret=None,
            places_used=False,
            reason="empty_query",
        )
    
    log.info("[Places Fallback] crm_id=%s query='%s'", crm_id, query[:80])
    
    # Step 1: Call Serper Places API
    try:
        response = search_places(query, config, log, cache_only=cache_only)
    except Exception as exc:
        log.warning("[Places Fallback] API failed for crm_id=%s: %s", crm_id, exc)
        return PlacesFallbackResult(
            crm_id=crm_id,
            original_status="REVIEW",
            final_status="NO_MATCH",
            final_siret=None,
            places_used=False,
            reason=f"api_error: {exc}",
        )
    
    if not response.places:
        log.info("[Places Fallback] No results for crm_id=%s", crm_id)
        return PlacesFallbackResult(
            crm_id=crm_id,
            original_status="REVIEW",
            final_status="NO_MATCH",
            final_siret=None,
            places_used=False,
            reason="no_places_results",
        )
    
    # Use first Places result (Google's best match)
    top1: PlacesResult = response.places[0]
    
    # Step 2: Dept-guard
    crm_dept = _dept_from_postcode(crm_postcode)
    places_dept = _dept_from_postcode(top1.postcode)
    
    if not crm_dept or not places_dept:
        log.info(
            "[Places Fallback] Dept guard unavailable for crm_id=%s: crm_postcode=%s places_postcode=%s",
            crm_id, crm_postcode, top1.postcode,
        )
        return PlacesFallbackResult(
            crm_id=crm_id,
            original_status="REVIEW",
            final_status="NO_MATCH",
            final_siret=None,
            places_used=True,
            places_title=top1.title,
            places_address=top1.address,
            places_postcode=top1.postcode,
            dept_guard_passed=False,
            reason="missing_dept_guard",
        )

    if crm_dept != places_dept:
        log.info(
            "[Places Fallback] Dept guard failed for crm_id=%s: CRM=%s vs Places=%s",
            crm_id, crm_dept, places_dept,
        )
        return PlacesFallbackResult(
            crm_id=crm_id,
            original_status="REVIEW",
            final_status="NO_MATCH",
            final_siret=None,
            places_used=True,
            places_title=top1.title,
            places_address=top1.address,
            places_postcode=top1.postcode,
            dept_guard_passed=False,
            reason=f"dept_mismatch: crm={crm_dept} places={places_dept}",
        )
    
    # Step 3: Build "repaired CRM" from Places top-1
    from xgb_matcher.infer import CrmInput
    
    repaired_crm = CrmInput(
        crm_id=crm_id,
        crm_name=top1.title,  # Google's name for this business
        crm_address=top1.address,  # Google's address
        postcode=top1.postcode or crm_postcode,  # Prefer Places, fallback to CRM
        insee=crm_insee,  # Keep original INSEE (Google doesn't provide this)
        crm_city=top1.city or crm_city,
    )
    
    log.info(
        "[Places Fallback] Rerunning XGB with repaired CRM: name='%s' addr='%s'",
        repaired_crm.crm_name[:50], repaired_crm.crm_address[:50] if repaired_crm.crm_address else "",
    )
    
    # Step 4: Rerun EXACT same XGB pipeline
    try:
        result = engine.infer_single(repaired_crm)
    except Exception as exc:
        log.warning("[Places Fallback] XGB rerun failed for crm_id=%s: %s", crm_id, exc)
        return PlacesFallbackResult(
            crm_id=crm_id,
            original_status="REVIEW",
            final_status="NO_MATCH",
            final_siret=None,
            places_used=True,
            places_title=top1.title,
            places_address=top1.address,
            places_postcode=top1.postcode,
            dept_guard_passed=True,
            reason=f"xgb_rerun_error: {exc}",
        )
    
    # Step 5: Binary decision based on XGB result
    if result.status == "AUTO":
        log.info(
            "[Places Fallback] MATCH_PLACES for crm_id=%s: siret=%s score=%.3f",
            crm_id, result.top1_siret, result.top1_score,
        )
        return PlacesFallbackResult(
            crm_id=crm_id,
            original_status="REVIEW",
            final_status="MATCH_PLACES",
            final_siret=result.top1_siret,
            places_used=True,
            places_title=top1.title,
            places_address=top1.address,
            places_postcode=top1.postcode,
            dept_guard_passed=True,
            xgb_rerun_status=result.status,
            xgb_rerun_score=result.top1_score,
        )
    else:
        # Post-Places is binary: NO REVIEW possible
        log.info(
            "[Places Fallback] NO_MATCH for crm_id=%s: xgb_status=%s score=%.3f",
            crm_id, result.status, result.top1_score,
        )
        return PlacesFallbackResult(
            crm_id=crm_id,
            original_status="REVIEW",
            final_status="NO_MATCH",
            final_siret=None,
            places_used=True,
            places_title=top1.title,
            places_address=top1.address,
            places_postcode=top1.postcode,
            dept_guard_passed=True,
            xgb_rerun_status=result.status,
            xgb_rerun_score=result.top1_score,
            reason="xgb_rerun_not_auto",
        )


__all__ = [
    "PlacesFallbackResult",
    "fallback_with_places",
]
