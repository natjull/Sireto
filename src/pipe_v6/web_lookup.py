"""Unified web lookup interface for V7 pipeline.

This module provides a clean interface for web-based candidate lookup with two modes:
- "places": New Serper Google Places API (V7 architecture)
- "legacy": Google CSE + Brave multisite search (V6 architecture)

The mode is selected via config.places_lookup_mode.

Usage:
    from pipe_v6.web_lookup import lookup_candidates

    candidates = lookup_candidates(
        crm_row=crm_row,
        xgb_topk=xgb_topk,
        config=config,
        conn=sirene_conn,
    )
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, List

from .config import PipelineConfig
from .candidate_store import RawCandidate

LOGGER = logging.getLogger(__name__)


def lookup_candidates(
    crm_name: str,
    street_number: str | None,
    street_name: str | None,
    postcode: str | None,
    city: str | None,
    config: PipelineConfig,
    conn: sqlite3.Connection | None = None,
    logger: logging.Logger | None = None,
) -> List[RawCandidate]:
    """Unified entry point for web-based candidate lookup.

    Routes to either Places-guided candidate generation or legacy
    Google/Brave multisite search based on config.places_lookup_mode.

    Args:
        crm_name: CRM business name
        street_number: Optional street number
        street_name: Optional street name
        postcode: Postal code
        city: City name
        config: Pipeline configuration
        conn: SQLite connection (required for Places mode)
        logger: Optional logger

    Returns:
        List of RawCandidate objects
    """
    log = logger or LOGGER
    mode = config.places_lookup_mode

    if mode == "places":
        return _lookup_places(
            crm_name=crm_name,
            street_number=street_number,
            street_name=street_name,
            postcode=postcode,
            city=city,
            config=config,
            conn=conn,
            logger=log,
        )
    else:
        return _lookup_legacy(
            crm_name=crm_name,
            street_number=street_number,
            street_name=street_name,
            postcode=postcode,
            config=config,
            logger=log,
        )


def _lookup_places(
    crm_name: str,
    street_number: str | None,
    street_name: str | None,
    postcode: str | None,
    city: str | None,
    config: PipelineConfig,
    conn: sqlite3.Connection | None,
    logger: logging.Logger,
) -> List[RawCandidate]:
    """Places-guided candidate lookup (V7 architecture).

    Uses Serper Google Places API to get structured business data,
    then generates candidates from SIRENE cache via address/radius matching.
    """
    from .serper_places_client import search_places, build_places_query, PlacesResult
    from .places_candidate_generator import (
        generate_candidates_by_address,
        generate_candidates_by_radius,
        merge_candidate_pools,
        SireneCandidate,
    )
    from .candidate_store import create_raw_candidate

    if conn is None:
        logger.warning("Places mode requires SIRENE cache connection, falling back to legacy")
        return _lookup_legacy(
            crm_name=crm_name,
            street_number=street_number,
            street_name=street_name,
            postcode=postcode,
            config=config,
            logger=logger,
        )

    # Build and execute Places query
    query = build_places_query(
        crm_name=crm_name,
        street_number=street_number,
        street_name=street_name,
        postcode=postcode,
        city=city,
    )

    try:
        response = search_places(query, config, logger)
    except Exception as exc:
        logger.error("Places API failed: %s, falling back to legacy", exc)
        return _lookup_legacy(
            crm_name=crm_name,
            street_number=street_number,
            street_name=street_name,
            postcode=postcode,
            config=config,
            logger=logger,
        )

    if not response.places:
        logger.info("No Places results, falling back to legacy")
        return _lookup_legacy(
            crm_name=crm_name,
            street_number=street_number,
            street_name=street_name,
            postcode=postcode,
            config=config,
            logger=logger,
        )

    # Use best Places result
    best_place = response.places[0]

    # Generate candidates from SIRENE cache
    arm_a = generate_candidates_by_address(best_place, conn, config, logger)
    arm_b = generate_candidates_by_radius(best_place, conn, config, logger)

    pool = merge_candidate_pools(arm_a, arm_b, [], logger)

    # Convert to RawCandidate format for compatibility
    candidates: List[RawCandidate] = []
    for cand in pool:
        try:
            raw_cand = create_raw_candidate(
                source=f"PLACES_{cand.source.upper()}",
                siren=cand.siren,
                siret=cand.siret,
                label=cand.display_name,
                url=None,
                extra={
                    "places_title": best_place.title,
                    "places_address": best_place.address,
                    "places_position": best_place.position,
                    "candidate_source": cand.source,
                    "distance_m": cand.distance_m,
                },
            )
            candidates.append(raw_cand)
        except Exception as exc:
            logger.debug("Failed to create candidate: %s", exc)

    logger.info(
        "Places lookup: query='%s' -> %d candidates (arm_a=%d, arm_b=%d)",
        query[:50],
        len(candidates),
        len(arm_a),
        len(arm_b),
    )

    return candidates


def _lookup_legacy(
    crm_name: str,
    street_number: str | None,
    street_name: str | None,
    postcode: str | None,
    config: PipelineConfig,
    logger: logging.Logger,
) -> List[RawCandidate]:
    """Legacy Google CSE + Brave multisite search (V6 architecture).

    Uses existing external_sources.search_web_sites function.
    """
    import pandas as pd
    from .external_sources import search_web_sites

    # Build a row-like object for search_web_sites
    row = pd.Series({
        "crm_name": crm_name,
        "street_number": street_number,
        "street_name": street_name,
        "postcode": postcode,
    })

    try:
        candidates = search_web_sites(row, config, logger)
    except Exception as exc:
        logger.error("Legacy web search failed: %s", exc)
        return []

    logger.info(
        "Legacy lookup: name='%s' postcode=%s -> %d candidates",
        crm_name[:30],
        postcode,
        len(candidates),
    )

    return candidates


def is_places_mode(config: PipelineConfig) -> bool:
    """Check if Places mode is enabled."""
    return (
        config.places_lookup_enabled
        and config.places_lookup_mode == "places"
        and bool(config.serper_api_key)
    )


def is_legacy_mode(config: PipelineConfig) -> bool:
    """Check if legacy mode is enabled."""
    return (
        config.web_search_enabled
        and config.places_lookup_mode == "legacy"
        and (bool(config.google_api_key) or bool(config.brave_api_key))
    )


__all__ = [
    "lookup_candidates",
    "is_places_mode",
    "is_legacy_mode",
]
