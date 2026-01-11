"""Places validation and decision logic for V7 pipeline.

Implements:
- Validation gate with 4 checks (position, name semantic, address overlap, distance)
- Adaptive threshold selection based on pool size
- Decision rules for MATCH_PLACES promotion
- Observability logging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .config import PipelineConfig
from .serper_places_client import PlacesResult
from .places_candidate_generator import (
    SireneCandidate,
    haversine_distance,
    jaro_winkler,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PlacesValidationResult:
    """Result of Places validation check."""

    is_valid: bool
    confidence_boost: float
    reason: str
    checks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacesDecisionResult:
    """Result of Places-guided decision process."""

    decision: str  # "MATCH_PLACES", "REVIEW", "NO_MATCH"
    chosen_siret: str | None
    score_before: float
    score_after: float
    gap_before: float
    gap_after: float
    pool_size: int
    pool_sources: Dict[str, int]
    threshold_used: float
    validation_result: PlacesValidationResult | None
    top1_changed: bool
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacesGateResult:
    """Result of CRM ↔ Places gate validation."""

    is_valid: bool
    reason: str
    checks: Dict[str, Any] = field(default_factory=dict)


def token_overlap(text1: str, text2: str) -> float:
    """Calculate token overlap ratio between two strings.

    Returns the Jaccard similarity of word tokens after normalization.
    Handles French street type abbreviations (Av. → AVENUE, etc.).
    """
    if not text1 or not text2:
        return 0.0

    import unicodedata

    # French street type abbreviations to normalize
    ABBREVIATIONS = {
        "AV": "AVENUE",
        "AV.": "AVENUE",
        "BD": "BOULEVARD",
        "BD.": "BOULEVARD",
        "R": "RUE",
        "R.": "RUE",
        "PL": "PLACE",
        "PL.": "PLACE",
        "ALL": "ALLEE",
        "ALL.": "ALLEE",
        "IMP": "IMPASSE",
        "IMP.": "IMPASSE",
        "CHE": "CHEMIN",
        "CHE.": "CHEMIN",
        "CHEM": "CHEMIN",
        "CHEM.": "CHEMIN",
        "PROM": "PROMENADE",
        "PROM.": "PROMENADE",
        "SQ": "SQUARE",
        "SQ.": "SQUARE",
        "RTE": "ROUTE",
        "RTE.": "ROUTE",
        "PAS": "PASSAGE",
        "PAS.": "PASSAGE",
        "QU": "QUAI",
        "QU.": "QUAI",
    }

    # Common stopwords to remove
    STOPWORDS = {"DE", "LA", "LE", "LES", "DU", "DES", "ET", "A", "AU", "AUX", "D", "L"}

    def normalize(t: str) -> set:
        # Remove accents
        t = unicodedata.normalize("NFKD", t)
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        t = t.upper()
        
        # Replace hyphens and commas with spaces for better token splitting
        t = t.replace("-", " ").replace(",", " ")
        
        # Split into tokens
        tokens = set(t.split())
        
        # Normalize abbreviations
        normalized_tokens = set()
        for token in tokens:
            # Remove trailing punctuation for lookup
            clean_token = token.rstrip(".,;:")
            if clean_token in ABBREVIATIONS:
                normalized_tokens.add(ABBREVIATIONS[clean_token])
            elif token in ABBREVIATIONS:
                normalized_tokens.add(ABBREVIATIONS[token])
            else:
                normalized_tokens.add(token)
        
        # Remove stopwords
        return normalized_tokens - STOPWORDS

    tokens1 = normalize(text1)
    tokens2 = normalize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union) if union else 0.0


def semantic_name_similarity(
    name1: str,
    name2: str,
    model: Any = None,
) -> float:
    """Calculate semantic similarity between business names.

    If a sentence-transformers model is provided, uses embeddings.
    Otherwise falls back to Jaro-Winkler on normalized names.

    Args:
        name1: First business name
        name2: Second business name
        model: Optional SentenceTransformer model

    Returns:
        Similarity score between 0 and 1
    """
    if not name1 or not name2:
        return 0.0

    # Normalize names for comparison
    import unicodedata

    def normalize(n: str) -> str:
        n = unicodedata.normalize("NFKD", n)
        n = "".join(ch for ch in n if not unicodedata.combining(ch))
        n = n.upper()
        n = " ".join(n.split())
        return n

    n1_norm = normalize(name1)
    n2_norm = normalize(name2)

    if model is not None:
        try:
            from sentence_transformers import util

            emb1 = model.encode(n1_norm, convert_to_tensor=True)
            emb2 = model.encode(n2_norm, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
            return max(0.0, min(1.0, similarity))
        except Exception:
            pass

    # Fallback to Jaro-Winkler
    return jaro_winkler(n1_norm, n2_norm)


def validate_places_result(
    places_result: PlacesResult,
    crm_name: str,
    crm_address: str,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
    *,
    semantic_model: Any = None,
) -> PlacesGateResult:
    """Validate the Places result against CRM data (pre-gate).

    Checks:
      1. Places position <= config.places_position_max
      2. Semantic name similarity (CRM vs Places) >= threshold
      3. Address token overlap (CRM vs Places) >= threshold
    """
    log = logger or LOGGER
    checks: Dict[str, Any] = {}

    # Position gate
    checks["places_position"] = places_result.position
    if places_result.position > getattr(config, "places_position_max", 2):
        return PlacesGateResult(
            is_valid=False,
            reason=f"places_position={places_result.position} > {getattr(config, 'places_position_max', 2)}",
            checks=checks,
        )

    # Name semantic gate (CRM vs Places)
    name_sim = semantic_name_similarity(crm_name or "", places_result.title or "", semantic_model)
    checks["crm_places_name_sim"] = name_sim
    name_min = getattr(config, "places_crm_name_semantic_min", config.places_name_semantic_min)

    # Address overlap gate (CRM vs Places)
    addr_overlap = token_overlap(crm_address or "", places_result.address or "")
    checks["crm_places_addr_overlap"] = addr_overlap
    addr_min = getattr(config, "places_crm_addr_overlap_min", config.places_addr_overlap_min)
    addr_strict = getattr(config, "places_crm_addr_overlap_strict", max(addr_min, 0.80))

    if addr_overlap < addr_min:
        return PlacesGateResult(
            is_valid=False,
            reason=(
                f"crm_places_addr_overlap={addr_overlap:.2f} < "
                f"{addr_min}"
            ),
            checks=checks,
        )

    # Allow low name similarity if address overlap is very strong (address-dominant gate)
    if name_sim < name_min and addr_overlap < addr_strict:
        return PlacesGateResult(
            is_valid=False,
            reason=(
                f"crm_places_name_sim={name_sim:.2f} < {name_min} "
                f"and crm_places_addr_overlap={addr_overlap:.2f} < {addr_strict}"
            ),
            checks=checks,
        )

    log.debug(
        "Places gate passed: name_sim=%.2f addr_overlap=%.2f",
        name_sim,
        addr_overlap,
    )
    return PlacesGateResult(is_valid=True, reason="crm_places_gate_ok", checks=checks)


def address_close(
    crm_address: str | None,
    crm_postcode: str | None,
    candidate_address: str | None,
    candidate_postcode: str | None,
    candidate_lat: float | None = None,
    candidate_lon: float | None = None,
    places_lat: float | None = None,
    places_lon: float | None = None,
    *,
    addr_jaro_min: float = 0.90,
    street_name_jaro_min: float = 0.85,
    street_number_diff_max: int = 2,
    distance_max_m: float = 80.0,
    postcode_exact: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """
    Phase 4: Check if candidate address is "close" to CRM/Places address.

    address_close combines:
    1. Postcode match (exact or prefix)
    2. Street address similarity (Jaro-Winkler)
    3. Street number proximity (if extractable)
    4. Geo distance (if coordinates available)

    Args:
        crm_address: CRM street address
        crm_postcode: CRM postcode
        candidate_address: Candidate street address
        candidate_postcode: Candidate postcode
        candidate_lat/lon: Candidate coordinates (optional)
        places_lat/lon: Places coordinates (optional)
        addr_jaro_min: Minimum Jaro similarity for full address
        street_name_jaro_min: Minimum Jaro similarity for street name only
        street_number_diff_max: Maximum street number difference
        distance_max_m: Maximum distance in meters
        postcode_exact: Require exact postcode match

    Returns:
        (is_close, checks_dict) tuple
    """
    import re

    checks: dict[str, Any] = {}

    # Normalize inputs
    crm_addr = (crm_address or "").strip().upper()
    cand_addr = (candidate_address or "").strip().upper()
    crm_pc = (crm_postcode or "").strip()
    cand_pc = (candidate_postcode or "").strip()

    # Check 1: Postcode
    if postcode_exact:
        postcode_ok = crm_pc == cand_pc and bool(crm_pc)
    else:
        # Allow prefix match (e.g., 69 matches 69003)
        postcode_ok = (
            crm_pc == cand_pc
            or (crm_pc and cand_pc and cand_pc.startswith(crm_pc[:2]))
        )
    checks["postcode_match"] = postcode_ok

    if not postcode_ok and postcode_exact:
        checks["reason"] = "postcode_mismatch"
        return False, checks

    # Check 2: Full address Jaro similarity
    if crm_addr and cand_addr:
        addr_jaro = jaro_winkler(crm_addr, cand_addr)
        checks["addr_jaro"] = addr_jaro

        if addr_jaro >= addr_jaro_min:
            checks["reason"] = "addr_jaro_pass"
            return True, checks
    else:
        checks["addr_jaro"] = 0.0

    # Check 3: Extract and compare street numbers
    def extract_street_number(addr: str) -> int | None:
        match = re.match(r"^\s*(\d+)", addr)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None

    crm_num = extract_street_number(crm_addr)
    cand_num = extract_street_number(cand_addr)
    checks["crm_street_num"] = crm_num
    checks["cand_street_num"] = cand_num

    if crm_num is not None and cand_num is not None:
        num_diff = abs(crm_num - cand_num)
        checks["street_number_diff"] = num_diff

        if num_diff > street_number_diff_max:
            checks["reason"] = "street_number_too_far"
            return False, checks

    # Check 4: Street name similarity (remove numbers first)
    def extract_street_name(addr: str) -> str:
        # Remove leading numbers and common prefixes
        cleaned = re.sub(r"^\s*\d+\s*[-–]?\s*\d*\s*", "", addr)
        cleaned = re.sub(r"^(BIS|TER|QUATER)\s+", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    crm_street = extract_street_name(crm_addr)
    cand_street = extract_street_name(cand_addr)

    if crm_street and cand_street:
        street_jaro = jaro_winkler(crm_street, cand_street)
        checks["street_name_jaro"] = street_jaro

        if street_jaro >= street_name_jaro_min:
            checks["reason"] = "street_name_pass"
            return True, checks
    else:
        checks["street_name_jaro"] = 0.0

    # Check 5: Geo distance (if coordinates available)
    if (
        candidate_lat is not None
        and candidate_lon is not None
        and places_lat is not None
        and places_lon is not None
    ):
        distance = haversine_distance(places_lat, places_lon, candidate_lat, candidate_lon)
        checks["distance_m"] = distance

        if distance <= distance_max_m:
            checks["reason"] = "distance_pass"
            return True, checks
    else:
        checks["distance_m"] = None

    # No check passed
    checks["reason"] = "no_check_passed"
    return False, checks


def validate_places_candidate(
    places_result: PlacesResult,
    candidate: SireneCandidate,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
    *,
    semantic_model: Any = None,
) -> PlacesValidationResult:
    """Validate a candidate against Places result using 4-check gate.

    Checks:
    1. Places position <= 2 (unicité)
    2. Name semantic similarity >= threshold
    3. Address token overlap >= threshold
    4. Distance <= threshold (if coordinates available)

    Args:
        places_result: Parsed Places API result
        candidate: SIRENE candidate to validate
        config: Pipeline configuration with thresholds
        logger: Optional logger
        semantic_model: Optional SentenceTransformer for semantic similarity

    Returns:
        PlacesValidationResult with is_valid, confidence_boost, and checks
    """
    log = logger or LOGGER
    checks: Dict[str, Any] = {}

    # Check 1: Position (unicité)
    max_pos = getattr(config, "places_position_max", 2)
    checks["position"] = places_result.position
    if places_result.position > max_pos:
        return PlacesValidationResult(
            is_valid=False,
            confidence_boost=0.0,
            reason=f"places_position={places_result.position} > {max_pos}",
            checks=checks,
        )

    # Check 2: Name semantic similarity
    candidate_name = candidate.display_name or ""
    places_name = places_result.title or ""

    name_sim = semantic_name_similarity(candidate_name, places_name, semantic_model)
    checks["name_semantic"] = name_sim

    if name_sim < config.places_name_semantic_min:
        return PlacesValidationResult(
            is_valid=False,
            confidence_boost=0.0,
            reason=f"name_semantic={name_sim:.2f} < {config.places_name_semantic_min}",
            checks=checks,
        )

    # Check 3: Address token overlap
    candidate_addr = candidate.full_address or ""
    places_addr = places_result.address or ""

    addr_overlap = token_overlap(candidate_addr, places_addr)
    checks["addr_overlap"] = addr_overlap

    if addr_overlap < config.places_addr_overlap_min:
        return PlacesValidationResult(
            is_valid=False,
            confidence_boost=0.0,
            reason=f"addr_overlap={addr_overlap:.2f} < {config.places_addr_overlap_min}",
            checks=checks,
        )

    # Check 4: Distance (if coordinates available)
    distance_m: float | None = None
    if (
        places_result.latitude is not None
        and places_result.longitude is not None
        and candidate.geo_latitude is not None
        and candidate.geo_longitude is not None
    ):
        distance_m = haversine_distance(
            places_result.latitude,
            places_result.longitude,
            candidate.geo_latitude,
            candidate.geo_longitude,
        )
        checks["distance_m"] = distance_m

        if distance_m > config.places_distance_max_m:
            return PlacesValidationResult(
                is_valid=False,
                confidence_boost=0.0,
                reason=f"distance={distance_m:.0f}m > {config.places_distance_max_m}m",
                checks=checks,
            )
    else:
        checks["distance_m"] = None
        checks["distance_skipped"] = True

    # All checks passed
    confidence_boost = 0.1 * name_sim + 0.05 * addr_overlap
    if distance_m is not None and distance_m < 100:
        confidence_boost += 0.05

    log.debug(
        "Places validation passed: siret=%s name_sim=%.2f addr_overlap=%.2f dist=%s",
        candidate.siret,
        name_sim,
        addr_overlap,
        f"{distance_m:.0f}m" if distance_m else "N/A",
    )

    return PlacesValidationResult(
        is_valid=True,
        confidence_boost=confidence_boost,
        reason="all_checks_passed",
        checks=checks,
    )


def get_adaptive_threshold(pool_size: int, config: PipelineConfig) -> float:
    """Get score threshold based on candidate pool size.

    Larger pools = higher threshold required (more conservative).

    Args:
        pool_size: Number of candidates in the pool
        config: Pipeline configuration with threshold values

    Returns:
        Score threshold τ for MATCH decision
    """
    if pool_size <= 5:
        return config.places_threshold_small
    elif pool_size <= 20:
        return config.places_threshold_medium
    else:
        return config.places_threshold_large


def make_places_decision(
    places_result: PlacesResult,
    candidates_scored: List[Tuple[SireneCandidate, float]],
    original_top1_siret: str | None,
    original_score: float,
    original_gap: float,
    config: PipelineConfig,
    pool_sources: Dict[str, int],
    logger: logging.Logger | None = None,
    *,
    semantic_model: Any = None,
) -> PlacesDecisionResult:
    """Make the final decision for Places-guided matching.

    Decision rules:
    - score_after >= τ(pool_size)
    - gap_after >= config.places_gap_min
    - Top candidate passes validation gate

    Args:
        places_result: Parsed Places API result
        candidates_scored: List of (candidate, score) tuples, sorted by score desc
        original_top1_siret: SIRET of original XGB top-1
        original_score: Score of original top-1
        original_gap: Gap between original top-1 and top-2
        config: Pipeline configuration
        pool_sources: Dict with counts per source {arm_a: N, arm_b: M, topk: K}
        logger: Optional logger
        semantic_model: Optional SentenceTransformer

    Returns:
        PlacesDecisionResult with decision and evidence
    """
    log = logger or LOGGER
    pool_size = len(candidates_scored)

    if pool_size == 0:
        return PlacesDecisionResult(
            decision="NO_MATCH",
            chosen_siret=None,
            score_before=original_score,
            score_after=0.0,
            gap_before=original_gap,
            gap_after=0.0,
            pool_size=0,
            pool_sources=pool_sources,
            threshold_used=0.0,
            validation_result=None,
            top1_changed=False,
            evidence={"reason": "empty_pool"},
        )

    # Get top candidates
    top1_candidate, score_after = candidates_scored[0]
    score_top2 = candidates_scored[1][1] if pool_size > 1 else 0.0
    gap_after = score_after - score_top2
    ratio_after = (
        score_after / (score_top2 + 1e-9) if pool_size > 1 and score_top2 > 0 else float("inf")
    )

    # Check if top-1 changed
    top1_changed = top1_candidate.siret != original_top1_siret

    # Get adaptive threshold
    threshold = get_adaptive_threshold(pool_size, config)

    # Validate top candidate against Places
    validation = validate_places_candidate(
        places_result,
        top1_candidate,
        config,
        logger=log,
        semantic_model=semantic_model,
    )

    # Build evidence
    evidence: Dict[str, Any] = {
        "top1_siret": top1_candidate.siret,
        "top1_name": top1_candidate.display_name,
        "top1_source": top1_candidate.source,
        "places_title": places_result.title,
        "places_address": places_result.address,
        "ratio_after": ratio_after,
        "validation_checks": validation.checks,
    }

    # Decision logic
    if not validation.is_valid:
        decision = "REVIEW"
        evidence["reason"] = f"validation_failed: {validation.reason}"
    elif score_after < threshold:
        decision = "REVIEW"
        evidence["reason"] = f"score_after={score_after:.4f} < threshold={threshold}"
    elif gap_after < config.places_gap_min:
        decision = "REVIEW"
        evidence["reason"] = f"gap_after={gap_after:.4f} < {config.places_gap_min}"
    elif ratio_after < getattr(config, "places_ratio_min", 1.0):
        decision = "REVIEW"
        evidence["reason"] = f"ratio_after={ratio_after:.4f} < {getattr(config, 'places_ratio_min', 1.0)}"
    else:
        decision = "MATCH_PLACES"
        evidence["reason"] = "all_conditions_met"

    log.info(
        "Places decision: %s siret=%s score=%.4f->%.4f gap=%.4f->%.4f pool=%d threshold=%.2f",
        decision,
        top1_candidate.siret,
        original_score,
        score_after,
        original_gap,
        gap_after,
        pool_size,
        threshold,
    )

    return PlacesDecisionResult(
        decision=decision,
        chosen_siret=top1_candidate.siret if decision == "MATCH_PLACES" else None,
        score_before=original_score,
        score_after=score_after,
        gap_before=original_gap,
        gap_after=gap_after,
        pool_size=pool_size,
        pool_sources=pool_sources,
        threshold_used=threshold,
        validation_result=validation,
        top1_changed=top1_changed,
        evidence=evidence,
    )


def build_observability_record(
    crm_id: str,
    xgb_status: str,
    places_decision: PlacesDecisionResult,
    places_query: str,
    places_gate: PlacesGateResult | None = None,
) -> Dict[str, Any]:
    """Build a complete observability record for logging.

    Includes all fields needed for analysis and threshold calibration.
    """
    record = {
        "crm_id": crm_id,
        "xgb_status": xgb_status,
        "places_query": places_query,
        "places_decision": places_decision.decision,
        "chosen_siret": places_decision.chosen_siret,
        "score_before": places_decision.score_before,
        "score_after": places_decision.score_after,
        "gap_before": places_decision.gap_before,
        "gap_after": places_decision.gap_after,
        "pool_size": places_decision.pool_size,
        "pool_arm_a": places_decision.pool_sources.get("arm_a", 0),
        "pool_arm_b": places_decision.pool_sources.get("arm_b", 0),
        "pool_topk": places_decision.pool_sources.get("topk", 0),
        "threshold_used": places_decision.threshold_used,
        "top1_changed": places_decision.top1_changed,
        "validation_reason": (
            places_decision.validation_result.reason
            if places_decision.validation_result
            else None
        ),
        "validation_name_sim": places_decision.evidence.get("validation_checks", {}).get(
            "name_semantic"
        ),
        "validation_addr_overlap": places_decision.evidence.get(
            "validation_checks", {}
        ).get("addr_overlap"),
        "validation_distance_m": places_decision.evidence.get(
            "validation_checks", {}
        ).get("distance_m"),
        "evidence_reason": places_decision.evidence.get("reason"),
    }
    if places_gate is not None:
        record.update(
            {
                "places_gate_valid": places_gate.is_valid,
                "places_gate_reason": places_gate.reason,
                "places_gate_position": places_gate.checks.get("places_position"),
                "places_gate_name_sim": places_gate.checks.get("crm_places_name_sim"),
                "places_gate_addr_overlap": places_gate.checks.get("crm_places_addr_overlap"),
            }
        )
    return record


__all__ = [
    "PlacesValidationResult",
    "PlacesDecisionResult",
    "PlacesGateResult",
    "token_overlap",
    "semantic_name_similarity",
    "validate_places_result",
    "validate_places_candidate",
    "get_adaptive_threshold",
    "make_places_decision",
    "build_observability_record",
    "address_close",
]
