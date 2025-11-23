"""Orchestration helpers for Pipe V6 (tasks 8.x)."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Sequence

from .candidate_store import (
    group_raw_candidates,
    enrich_candidates_from_sirene,
)
from .commune_detection import CommuneKey
from .config import PipelineConfig
from .external_sources import search_datagouv, search_qwant_sites, search_rne
from .llm_matcher import classify_final_status, decide_match, filter_candidates_by_category
from .llm_normalizer import NormalizationParseError, normalize_crm_entry
from .llm_utils import LLMCallError, OllamaClient
from .rne_client import RneClient
from .sirene_cache import get_or_fetch_commune


LOGGER = logging.getLogger(__name__)


def preload_sirene(
    communes: Sequence[CommuneKey],
    config: PipelineConfig,
    logger: logging.Logger | None = None,
    *,
    conn=None,
) -> None:
    """Preload SIRENE data for all communes before processing the CRM.

    This is an I/O-bound preparatory step: it loops over the list of communes
    computed from the CRM and ensures each one is present in the local SQLite
    cache. Any exception from the underlying SIRENE client/cache is allowed to
    propagate, as the cache is a hard requirement for the pipeline.

    Args:
        communes: Ordered list of :class:`CommuneKey` to fetch.
        config: Pipeline configuration.
        logger: Optional logger (defaults to module logger).
        conn: Optional SQLite connection to reuse; if ``None``,
            :func:`get_or_fetch_commune` will open/close internally.
    """

    log = logger or LOGGER

    if not communes:
        log.info("No communes to preload (empty CRM input).")
        return

    log.info("Preloading SIRENE cache for %d commune(s)...", len(communes))

    for idx, commune in enumerate(communes, start=1):
        log.debug(
            "[%d/%d] Preload commune insee=%s postcode=%s city=%s",
            idx,
            len(communes),
            commune.insee_code,
            commune.postcode,
            commune.city,
        )
        get_or_fetch_commune(commune, config, conn=conn, logger=log)

    log.info("SIRENE preload completed for %d commune(s).", len(communes))


__all__ = ["preload_sirene"]


def _get(row: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve a value from pandas Series or itertuples row."""

    try:
        if hasattr(row, "__getitem__"):
            return row[key]
    except Exception:
        pass
    return getattr(row, key, default)


def process_crm_row(
    row: Any,
    config: PipelineConfig,
    conn: sqlite3.Connection,
    logger: logging.Logger,
    *,
    rne_client: RneClient | None = None,
    llm_client: OllamaClient | None = None,
) -> dict:
    """Process a single CRM line from normalization to LLM #2 arbitration.

    The function is resilient: expected errors from normalization or external
    sources are handled gracefully; unexpected exceptions are caught and turned
    into a NO_MATCH result so that one bad record does not break the pipeline.
    """

    crm_id = _get(row, "crm_id")

    def _base_result(
        *,
        status: str,
        chosen_siret: str | None,
        confidence: float,
        reason: str | None,
        crm_category: str,
        normalized_name: str,
        normalized_address: str,
        candidate_count_total: int = 0,
        candidate_count_used: int = 0,
        sources: list[str] | None = None,
    ) -> dict:
        return {
            "crm_id": crm_id,
            "status": status,
            "chosen_siret": chosen_siret,
            "confidence": confidence,
            "reason": reason,
            "crm_category": crm_category,
            "normalized_name": normalized_name,
            "normalized_address": normalized_address,
            "candidate_count_total": candidate_count_total,
            "candidate_count_used": candidate_count_used,
            "sources": sources or [],
        }

    try:
        # 1) Normalisation LLM #1
        try:
            norm_entry = normalize_crm_entry(row, config, logger=logger, client=llm_client)
        except (LLMCallError, NormalizationParseError) as exc:
            logger.error("CRM %s: normalization failed: %s", crm_id, exc)
            return _base_result(
                status="NO_MATCH",
                chosen_siret=None,
                confidence=0.0,
                reason=f"LLM_NORMALIZATION_ERROR: {type(exc).__name__} - {exc}",
                crm_category="INCONNU",
                normalized_name="",
                normalized_address="",
            )

        # 2) Court-circuit EQUIPEMENT_URBAIN
        if norm_entry.category == "EQUIPEMENT_URBAIN":
            logger.info(
                "CRM %s: category EQUIPEMENT_URBAIN -> short-circuit NO_MATCH", crm_id
            )
            return _base_result(
                status="NO_MATCH",
                chosen_siret=None,
                confidence=0.0,
                reason="EQUIPEMENT_URBAIN: pas d'entité légale attendue",
                crm_category=norm_entry.category,
                normalized_name=norm_entry.normalized_name,
                normalized_address=norm_entry.normalized_address,
            )

        # 3) Collecte des candidats externes (RNE, DataGouv, Qwant)
        all_candidates: list = []

        try:
            all_candidates.extend(
                search_rne(
                    norm_entry.normalized_name,
                    city=_get(row, "city", ""),
                    postcode=_get(row, "postcode", None),
                    config=config,
                    logger=logger,
                    client=rne_client,
                )
            )
        except Exception as exc:  # RneApiError or other
            logger.error("CRM %s: RNE search failed: %s", crm_id, exc)

        try:
            all_candidates.extend(
                search_datagouv(
                    norm_entry.normalized_name,
                    city=_get(row, "city", ""),
                    postcode=_get(row, "postcode", None),
                    config=config,
                    logger=logger,
                )
            )
        except Exception as exc:
            logger.error("CRM %s: DataGouv search failed: %s", crm_id, exc)

        try:
            all_candidates.extend(search_qwant_sites(row, config=config, logger=logger))
        except Exception as exc:
            logger.error("CRM %s: Qwant search failed: %s", crm_id, exc)

        # 4) Agrégation et enrichissement SIRENE
        groups = group_raw_candidates(all_candidates)
        candidates = enrich_candidates_from_sirene(groups, conn, logger)
        candidate_count_total = len(candidates)

        filtered = filter_candidates_by_category(
            norm_entry.category, candidates, config, logger=logger
        )
        candidate_count_used = len(filtered)

        # 5) LLM #2 arbitrage
        decision = decide_match(
            row,
            norm_entry,
            filtered,
            config,
            logger=logger,
            client=llm_client,
        )
        status = classify_final_status(decision, config)

        chosen_siret = None if status == "NO_MATCH" else decision.chosen_siret
        sources: list[str] = []
        if chosen_siret:
            chosen_candidate = next((c for c in filtered if c.siret == chosen_siret), None)
            if chosen_candidate:
                sources = chosen_candidate.sources
            else:
                logger.error(
                    "CRM %s: chosen_siret %s not found in filtered candidates", crm_id, chosen_siret
                )

        return _base_result(
            status=status,
            chosen_siret=chosen_siret,
            confidence=decision.confidence,
            reason=decision.reason,
            crm_category=norm_entry.category,
            normalized_name=norm_entry.normalized_name,
            normalized_address=norm_entry.normalized_address,
            candidate_count_total=candidate_count_total,
            candidate_count_used=candidate_count_used,
            sources=sources,
        )

    except Exception as exc:  # safety net
        logger.error(
            "CRM %s: unexpected pipeline error: %s", crm_id, exc, exc_info=True
        )
        return {
            "crm_id": crm_id,
            "status": "NO_MATCH",
            "chosen_siret": None,
            "confidence": 0.0,
            "reason": f"PIPELINE_ERROR: {type(exc).__name__}",
            "crm_category": "INCONNU",
            "normalized_name": "",
            "normalized_address": "",
            "candidate_count_total": 0,
            "candidate_count_used": 0,
            "sources": [],
        }


__all__.append("process_crm_row")
