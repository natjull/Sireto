"""Orchestrator for Places-guided candidate generation and XGB rescoring.

This module ties together:
- Serper Places API client
- Candidate generation (Arm A + Arm B)
- XGB rescoring with Places-normalized data
- Decision logic with adaptive thresholds

Usage:
    from pipe_v6.places_orchestrator import process_review_case

    result = process_review_case(
        crm_row=...,
        xgb_topk=...,
        config=config,
        conn=sirene_conn,
    )
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from .config import PipelineConfig
from .serper_places_client import (
    PlacesResult,
    PlacesSearchResponse,
    build_places_query,
    search_places,
)
from .places_candidate_generator import (
    SireneCandidate,
    generate_candidates_by_address,
    generate_candidates_by_radius,
    merge_candidate_pools,
)
from .places_validator import (
    PlacesDecisionResult,
    PlacesGateResult,
    build_observability_record,
    make_places_decision,
    validate_places_result,
)
from .places_xgb_rescorer import PlacesXgbRescorer

LOGGER = logging.getLogger(__name__)


@dataclass
class CrmRow:
    """Minimal CRM row representation for Places processing."""

    crm_id: str
    crm_name: str
    crm_address: str | None
    street_number: str | None
    street_name: str | None
    postcode: str | None
    city: str | None
    insee_code: str | None = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CrmRow":
        """Create from dictionary (e.g., pandas row.to_dict())."""
        return cls(
            crm_id=str(d.get("crm_id", "")),
            crm_name=str(d.get("crm_name", "") or ""),
            crm_address=str(d.get("crm_address", "") or "") or None,
            street_number=str(d.get("street_number", "") or "") or None,
            street_name=str(d.get("street_name", "") or "") or None,
            postcode=str(d.get("postcode", "") or "") or None,
            city=str(d.get("city", "") or "") or None,
            insee_code=str(d.get("insee_code", "") or "") or None,
        )


@dataclass
class XgbTopkCandidate:
    """XGB top-k candidate with score."""

    siret: str
    score: float
    rank: int
    features: Dict[str, Any] | None = None

    def to_sirene_candidate(self) -> SireneCandidate:
        """Convert to SireneCandidate (minimal, for pool merging)."""
        return SireneCandidate(
            siret=self.siret,
            siren=self.siret[:9],
            denomination=self.features.get("denomination") if self.features else None,
            denomination_ci=self.features.get("denomination_ci") if self.features else None,
            enseigne1=self.features.get("enseigne1") if self.features else None,
            enseigne2=self.features.get("enseigne2") if self.features else None,
            enseigne3=self.features.get("enseigne3") if self.features else None,
            denomination_unite_legale=(
                self.features.get("denomination_unite_legale") if self.features else None
            ),
            nom_unite_legale=self.features.get("nom_unite_legale") if self.features else None,
            prenom1_unite_legale=self.features.get("prenom1_unite_legale") if self.features else None,
            street_number=self.features.get("street_number") if self.features else None,
            street_type=self.features.get("street_type") if self.features else None,
            street_name=self.features.get("street_name") if self.features else None,
            address_full=self.features.get("address_full") if self.features else None,
            postcode=self.features.get("postcode") if self.features else None,
            city=self.features.get("city") if self.features else None,
            insee_code=self.features.get("insee_code") if self.features else None,
            geo_latitude=self.features.get("geo_latitude") if self.features else None,
            geo_longitude=self.features.get("geo_longitude") if self.features else None,
            etat_administratif=None,
            etablissement_siege=None,
            activite_principale=None,
            legal_nature=None,
            source="topk",
        )


@dataclass
class PlacesProcessingResult:
    """Complete result of Places-guided processing."""

    crm_id: str
    xgb_status: str
    places_response: PlacesSearchResponse | None
    places_decision: PlacesDecisionResult | None
    final_status: str  # "MATCH_PLACES", "MATCH_XGB", "REVIEW", "NO_MATCH"
    final_siret: str | None
    observability: Dict[str, Any]
    error: str | None = None


def _crm_address_from_row(crm_row: CrmRow) -> str:
    """Best-effort CRM address string."""
    if crm_row.crm_address:
        return crm_row.crm_address
    parts = [crm_row.street_number, crm_row.street_name]
    return " ".join(p for p in parts if p)


def enrich_candidate_from_cache(
    candidate: SireneCandidate,
    conn: sqlite3.Connection,
) -> SireneCandidate:
    """Enrich a candidate with full data from SIRENE cache."""
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT denomination, denomination_ci, enseigne1, enseigne2, enseigne3,
               denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
               street_number, street_type, street_name, address_full, postcode, city, insee_code,
               geo_latitude, geo_longitude, etat_administratif, etablissement_siege,
               activite_principale, legal_nature
        FROM establishments
        WHERE siret = ?
        """,
        (candidate.siret,),
    )
    row = cursor.fetchone()
    cursor.close()

    if row:
        candidate.denomination = row["denomination"]
        candidate.denomination_ci = row["denomination_ci"]
        candidate.enseigne1 = row["enseigne1"]
        candidate.enseigne2 = row["enseigne2"]
        candidate.enseigne3 = row["enseigne3"]
        candidate.denomination_unite_legale = row["denomination_unite_legale"]
        candidate.nom_unite_legale = row["nom_unite_legale"]
        candidate.prenom1_unite_legale = row["prenom1_unite_legale"]
        candidate.street_number = row["street_number"]
        candidate.street_type = row["street_type"]
        candidate.street_name = row["street_name"]
        candidate.address_full = row["address_full"]
        candidate.postcode = row["postcode"]
        candidate.city = row["city"]
        candidate.insee_code = row["insee_code"]
        candidate.geo_latitude = row["geo_latitude"]
        candidate.geo_longitude = row["geo_longitude"]
        candidate.etat_administratif = row["etat_administratif"]
        candidate.etablissement_siege = row["etablissement_siege"]
        candidate.activite_principale = row["activite_principale"]
        candidate.legal_nature = row["legal_nature"]

    return candidate


def process_review_case(
    crm_row: CrmRow,
    xgb_topk: List[XgbTopkCandidate],
    config: PipelineConfig,
    conn: sqlite3.Connection,
    xgb_status: str = "REVIEW",
    logger: logging.Logger | None = None,
    *,
    score_fn: Callable[[SireneCandidate, CrmRow], float] | None = None,
    semantic_model: Any = None,
    xgb_rescorer: PlacesXgbRescorer | None = None,
    cache_only: bool = False,
) -> PlacesProcessingResult:
    """Process a REVIEW case using Places-guided candidate generation.

    Full pipeline:
    1. Call Serper Places API
    2. Generate candidates via Arm A (address) and Arm B (radius)
    3. Merge with XGB top-k
    4. Rescore all candidates (using provided score_fn or simple heuristic)
    5. Apply decision logic with adaptive thresholds

    Args:
        crm_row: CRM row data
        xgb_topk: Original XGB top-k candidates with scores
        config: Pipeline configuration
        conn: SQLite connection to SIRENE cache
        xgb_status: Original XGB routing status ("REVIEW" or "NO_MATCH")
        logger: Optional logger
        score_fn: Optional scoring function (candidate, crm) -> score
        semantic_model: Optional SentenceTransformer for name similarity

    Returns:
        PlacesProcessingResult with decision and observability data
    """
    log = logger or LOGGER

    # Build Places query
    query = build_places_query(
        crm_name=crm_row.crm_name,
        street_number=crm_row.street_number,
        street_name=crm_row.street_name,
        postcode=crm_row.postcode,
        city=crm_row.city,
    )

    log.info("Processing REVIEW case: crm_id=%s query='%s'", crm_row.crm_id, query[:80])

    # Step 1: Call Places API
    try:
        places_response = search_places(query, config, log, cache_only=cache_only)
    except Exception as exc:
        log.error("Places API failed for crm_id=%s: %s", crm_row.crm_id, exc)
        return PlacesProcessingResult(
            crm_id=crm_row.crm_id,
            xgb_status=xgb_status,
            places_response=None,
            places_decision=None,
            final_status=xgb_status,
            final_siret=xgb_topk[0].siret if xgb_topk else None,
            observability={"error": str(exc)},
            error=str(exc),
        )

    if not places_response.places:
        log.info("No Places results for crm_id=%s", crm_row.crm_id)
        return PlacesProcessingResult(
            crm_id=crm_row.crm_id,
            xgb_status=xgb_status,
            places_response=places_response,
            places_decision=None,
            final_status=xgb_status,
            final_siret=xgb_topk[0].siret if xgb_topk else None,
            observability={"reason": "no_places_results"},
        )

    # Use first Places result (best match from Google)
    best_place = places_response.places[0]

    # Pre-gate: validate Places vs CRM (avoid wrong Places hit)
    crm_address = _crm_address_from_row(crm_row)
    places_gate = validate_places_result(
        best_place,
        crm_name=crm_row.crm_name,
        crm_address=crm_address,
        config=config,
        logger=log,
        semantic_model=semantic_model,
    )
    if not places_gate.is_valid:
        log.info(
            "Places gate failed for crm_id=%s: %s", crm_row.crm_id, places_gate.reason
        )
        return PlacesProcessingResult(
            crm_id=crm_row.crm_id,
            xgb_status=xgb_status,
            places_response=places_response,
            places_decision=None,
            final_status=xgb_status,
            final_siret=xgb_topk[0].siret if xgb_topk else None,
            observability={
                "reason": "places_gate_failed",
                "places_gate": {
                    "valid": places_gate.is_valid,
                    "reason": places_gate.reason,
                    **places_gate.checks,
                },
            },
        )

    # Step 2: Generate candidates
    arm_a = generate_candidates_by_address(best_place, conn, config, log)
    arm_b = generate_candidates_by_radius(best_place, conn, config, log)
    topk_candidates = [c.to_sirene_candidate() for c in xgb_topk]

    # Enrich topk candidates from cache
    for cand in topk_candidates:
        enrich_candidate_from_cache(cand, conn)

    # Step 3: Merge pools
    pool = merge_candidate_pools(arm_a, arm_b, topk_candidates, log)
    pool_sources = {
        "arm_a": len(arm_a),
        "arm_b": len(arm_b),
        "topk": len(topk_candidates),
    }

    if not pool:
        log.info("Empty candidate pool for crm_id=%s", crm_row.crm_id)
        return PlacesProcessingResult(
            crm_id=crm_row.crm_id,
            xgb_status=xgb_status,
            places_response=places_response,
            places_decision=None,
            final_status=xgb_status,
            final_siret=None,
            observability={"reason": "empty_pool"},
        )

    # Step 4: Rescore candidates
    scored: List[Tuple[SireneCandidate, float]] = []

    if xgb_rescorer is not None:
        scores = xgb_rescorer.score_candidates(best_place, crm_row, pool)
        scored = list(zip(pool, scores))
    else:
        for candidate in pool:
            if score_fn is not None:
                score = score_fn(candidate, crm_row)
            else:
                # Simple heuristic scoring if no XGB scorer provided
                score = _simple_score(candidate, best_place, crm_row)
            scored.append((candidate, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Get original XGB metrics
    original_top1_siret = xgb_topk[0].siret if xgb_topk else None
    original_score = xgb_topk[0].score if xgb_topk else 0.0
    original_gap = (
        xgb_topk[0].score - xgb_topk[1].score if len(xgb_topk) > 1 else 0.0
    )

    # Step 5: Make decision
    decision = make_places_decision(
        places_result=best_place,
        candidates_scored=scored,
        original_top1_siret=original_top1_siret,
        original_score=original_score,
        original_gap=original_gap,
        config=config,
        pool_sources=pool_sources,
        logger=log,
        semantic_model=semantic_model,
    )

    # Determine final status
    if decision.decision == "MATCH_PLACES":
        final_status = "MATCH_PLACES"
        final_siret = decision.chosen_siret
    else:
        final_status = xgb_status  # Keep original REVIEW/NO_MATCH
        final_siret = original_top1_siret

    # Build observability record
    observability = build_observability_record(
        crm_id=crm_row.crm_id,
        xgb_status=xgb_status,
        places_decision=decision,
        places_query=query,
        places_gate=places_gate,
    )

    return PlacesProcessingResult(
        crm_id=crm_row.crm_id,
        xgb_status=xgb_status,
        places_response=places_response,
        places_decision=decision,
        final_status=final_status,
        final_siret=final_siret,
        observability=observability,
    )


def _simple_score(
    candidate: SireneCandidate,
    places: PlacesResult,
    crm: CrmRow,
) -> float:
    """Simple heuristic scoring for candidates (fallback if no XGB scorer)."""
    from .places_validator import semantic_name_similarity, token_overlap

    score = 0.0

    # Name similarity with Places
    places_name_sim = semantic_name_similarity(candidate.display_name or "", places.title)
    score += 0.4 * places_name_sim

    # Name similarity with CRM
    crm_name_sim = semantic_name_similarity(candidate.display_name or "", crm.crm_name)
    score += 0.3 * crm_name_sim

    # Address overlap with Places
    addr_overlap = token_overlap(candidate.full_address or "", places.address)
    score += 0.2 * addr_overlap

    # Postcode match
    if candidate.postcode and crm.postcode and candidate.postcode == crm.postcode:
        score += 0.1

    return score


__all__ = [
    "CrmRow",
    "XgbTopkCandidate",
    "PlacesProcessingResult",
    "process_review_case",
    "enrich_candidate_from_cache",
]
