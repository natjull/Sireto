"""Orchestration helpers for Pipe V6 (tasks 8.x)."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Iterable, Sequence

import pandas as pd

from .candidate_store import (
    group_raw_candidates,
    enrich_candidates_from_sirene,
)
from .commune_detection import CommuneKey, extract_communes
from .config import PipelineConfig
from .crm_loader import load_crm
from .external_sources import search_datagouv, search_qwant_sites, search_rne
from .llm_matcher import classify_final_status, decide_match, filter_candidates_by_category
from .llm_normalizer import NormalizationParseError, normalize_crm_entry
from .llm_utils import LLMCallError, OllamaClient
from .rne_client import RneClient
from .sirene_cache import get_cache_connection, get_or_fetch_commune


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


__all__ = ["preload_sirene", "run_pipeline", "process_crm_row"]


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
    chosen_name: str | None,
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
        "chosen_name": chosen_name,
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
                chosen_name=None,
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
                chosen_name=None,
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
                chosen_name = chosen_candidate.name
            else:
                logger.error(
                    "CRM %s: chosen_siret %s not found in filtered candidates", crm_id, chosen_siret
                )
                chosen_name = None
        else:
            chosen_name = None

        return _base_result(
            status=status,
            chosen_siret=chosen_siret,
            chosen_name=chosen_name,
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
            "chosen_name": None,
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


# ---------------------------------------------------------------------------
# Pipeline complet (8.3)
# ---------------------------------------------------------------------------


RESULT_COLUMNS = [
    "crm_id",
    "status",
    "chosen_siret",
    "chosen_name",
    "confidence",
    "reason",
    "crm_category",
    "normalized_name",
    "normalized_address",
    "candidate_count_total",
    "candidate_count_used",
    "sources",
]


def _progress(iterable: Iterable, total: int, description: str):
    """Yield from iterable with tqdm if available, otherwise passthrough.

    Falls back silently when tqdm is not installed to avoid an extra dependency
    requirement at runtime.
    """

    try:  # pragma: no cover - dependency optional
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=description)
    except Exception:  # noqa: BLE001 - broad to keep fallback simple
        return iterable


def run_pipeline(
    config: PipelineConfig,
    *,
    max_rows: int | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Run the full Pipe V6 pipeline over the CRM CSV.

    Steps:
    1. Load CRM (optional row cap via ``max_rows``).
    2. Extract communes and preload SIRENE cache.
    3. Process each CRM row sequentially with shared RNE + LLM clients.
    4. Return a DataFrame containing only pipeline result columns.
    """

    log = logger or LOGGER

    df = load_crm(config.crm_path)
    if max_rows is not None:
        df = df.head(max_rows)

    if df.empty:
        log.warning("CRM input is empty; returning empty results DataFrame.")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    communes = extract_communes(df)

    conn = get_cache_connection(config)
    llm_client: OllamaClient | None = None
    rne_client: RneClient | None = None

    try:
        preload_sirene(communes, config, logger=log, conn=conn)

        rne_client = RneClient(config=config, logger=log)
        llm_client = OllamaClient(config=config, logger=log)

        results: list[dict] = []
        iterable = _progress(
            df.itertuples(index=False, name="CRMRow"),
            total=len(df),
            description="Processing CRM",
        )

        for row in iterable:
            result = process_crm_row(
                row,
                config,
                conn,
                log,
                rne_client=rne_client,
                llm_client=llm_client,
            )
            results.append(result)

        return pd.DataFrame(results, columns=RESULT_COLUMNS)

    finally:
        try:
            if llm_client:
                llm_client.close()
        except Exception:
            log.exception("Failed to close Ollama client")

        try:
            conn.close()
        except Exception:
            log.exception("Failed to close SQLite connection")


__all__.append("run_pipeline")
