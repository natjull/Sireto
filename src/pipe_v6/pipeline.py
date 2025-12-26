"""Orchestration helpers for Pipe V6 (tasks 8.x)."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
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
from .external_sources import search_datagouv, search_web_sites, search_rne
from .llm_matcher import classify_final_status, decide_match, filter_candidates_by_category
from .llm_normalizer import NormalizationParseError, normalize_crm_entry
from .llm_utils import LLMCallError, create_llm_client
from .rne_client import RneClient
from .sirene_cache import get_cache_connection, get_or_fetch_commune
from xgb_matcher.features import (
    normalize_text,
    jaro_sim,
    token_overlap,
    extract_street_number,
    extract_street_name_from_address,
    street_number_diff,
)


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


def _crm_address_raw(row: Any) -> str:
    street_number = _get(row, "street_number", "") or ""
    street_name = _get(row, "street_name", "") or ""
    return f"{street_number} {street_name}".strip()


def _crm_normalized_name(row: Any) -> str:
    return normalize_text(_get(row, "crm_name", "") or "")


def _crm_normalized_address(row: Any) -> str:
    return normalize_text(_crm_address_raw(row))


def _lookup_establishment(conn: sqlite3.Connection, siret: str) -> dict[str, Any] | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT siret, siren, denomination, address_full, postcode, city, insee_code, legal_nature "
        "FROM establishments WHERE siret = ?",
        (siret,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def _web_sources(candidate) -> set[str]:
    return {src for src in candidate.sources if src.startswith("WEB_")}


def _web_candidate_evidence(
    row: Any,
    candidate,
    siren_sources: set[str] | None = None,
) -> dict[str, Any]:
    crm_postcode = str(_get(row, "postcode", "") or "").strip()
    crm_city = str(_get(row, "city", "") or "").strip()
    crm_insee = str(_get(row, "insee_code", "") or "").strip()

    cand_postcode = str(candidate.postcode or "").strip()
    cand_city = str(candidate.city or "").strip()
    cand_insee = str(candidate.insee_code or "").strip()

    geo_ok = True
    if crm_insee and cand_insee and crm_insee != cand_insee:
        geo_ok = False
    if crm_postcode and cand_postcode and crm_postcode != cand_postcode:
        geo_ok = False
    if crm_city and cand_city and normalize_text(crm_city) != normalize_text(cand_city):
        geo_ok = False

    crm_addr_norm = _crm_normalized_address(row)
    cand_addr_norm = normalize_text(candidate.address or "")

    street_name_jaro = jaro_sim(
        extract_street_name_from_address(crm_addr_norm),
        extract_street_name_from_address(cand_addr_norm),
    )
    addr_overlap = token_overlap(crm_addr_norm, cand_addr_norm)
    crm_num = extract_street_number(crm_addr_norm)
    cand_num = extract_street_number(cand_addr_norm)
    num_diff = street_number_diff(crm_num, cand_num)

    num_ok_strict = True if crm_num is None or cand_num is None else num_diff <= 2
    num_ok_very = True if crm_num is None or cand_num is None else num_diff <= 1

    strict_addr = geo_ok and street_name_jaro >= 0.90 and addr_overlap >= 0.50 and num_ok_strict
    very_strict_addr = geo_ok and street_name_jaro >= 0.95 and addr_overlap >= 0.60 and num_ok_very

    web_sources = _web_sources(candidate)
    merged_sources = set(web_sources)
    if siren_sources:
        merged_sources.update(siren_sources)
    multi_site = len(merged_sources) >= 2
    has_annuaire = "WEB_ANNUAIRE" in merged_sources

    best_rank = None
    if getattr(candidate, "web_ranks", None):
        try:
            best_rank = min(int(v) for v in candidate.web_ranks.values())
        except Exception:
            best_rank = None

    return {
        "geo_ok": geo_ok,
        "street_name_jaro": street_name_jaro,
        "addr_overlap": addr_overlap,
        "num_diff": num_diff,
        "num_ok_strict": num_ok_strict,
        "num_ok_very": num_ok_very,
        "strict_addr": strict_addr,
        "very_strict_addr": very_strict_addr,
        "multi_site": multi_site,
        "has_annuaire": has_annuaire,
        "web_sources": sorted(web_sources),
        "web_sources_siren": sorted(siren_sources or []),
        "web_sources_merged": sorted(merged_sources),
        "best_rank": best_rank,
    }


def _select_web_candidate(
    row: Any,
    candidates: list,
    *,
    strict_for_no_match: bool,
    logger: logging.Logger,
) -> tuple[Any | None, dict[str, Any] | None, float, int]:
    eligible: list[tuple[Any, dict[str, Any], float]] = []

    siren_sources_map: dict[str, set[str]] = {}
    for cand in candidates:
        if not getattr(cand, "siren", None):
            continue
        sources = _web_sources(cand)
        if not sources:
            continue
        siren_sources_map.setdefault(cand.siren, set()).update(sources)

    for cand in candidates:
        if not cand.siret:
            continue
        evidence = _web_candidate_evidence(
            row,
            cand,
            siren_sources_map.get(cand.siren),
        )
        if not evidence["geo_ok"]:
            continue

        if strict_for_no_match:
            is_eligible = evidence["multi_site"] and evidence["very_strict_addr"]
        else:
            is_eligible = (evidence["multi_site"] and evidence["strict_addr"]) or (
                evidence["has_annuaire"] and evidence["very_strict_addr"]
            )

        if not is_eligible:
            continue

        score = evidence["street_name_jaro"] + evidence["addr_overlap"]
        if evidence["multi_site"]:
            score += 0.5
        if evidence["has_annuaire"]:
            score += 0.2
        if evidence["num_diff"] != 9999.0:
            score -= min(evidence["num_diff"], 5.0) * 0.05
        if evidence["best_rank"] is not None:
            score += max(0.0, 0.15 - float(evidence["best_rank"]) * 0.02)

        eligible.append((cand, evidence, score))

    if not eligible:
        return None, None, 0.0, 0

    eligible.sort(key=lambda x: x[2], reverse=True)
    chosen, evidence, score = eligible[0]

    if strict_for_no_match:
        confidence = 0.95 if evidence["very_strict_addr"] else 0.90
    else:
        if evidence["multi_site"] and evidence["very_strict_addr"]:
            confidence = 0.95
        elif evidence["multi_site"]:
            confidence = 0.90
        else:
            confidence = 0.88

    logger.debug(
        "Web candidate selected: siret=%s score=%.3f confidence=%.2f evidence=%s",
        chosen.siret,
        score,
        confidence,
        evidence,
    )
    return chosen, evidence, confidence, len(eligible)


def process_crm_row(
    row: Any,
    config: PipelineConfig,
    conn: sqlite3.Connection,
    logger: logging.Logger,
    *,
    rne_client: RneClient | None = None,
    llm_client=None,
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
            "xgb_status": None,
            "xgb_shap": None,
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

        # 3) Collecte des candidats externes (RNE, DataGouv, Web)
        all_candidates: list = []

        if getattr(config, "rne_enabled", True):
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
            all_candidates.extend(search_web_sites(row, config=config, logger=logger))
        except Exception as exc:
            logger.error("CRM %s: Web search failed: %s", crm_id, exc)

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
            "xgb_status": None,
            "xgb_shap": None,
        }


__all__.append("process_crm_row")


def process_crm_row_xgb(
    row: Any,
    config: PipelineConfig,
    conn: sqlite3.Connection,
    logger: logging.Logger,
) -> dict:
    """Process a single CRM row using XGBoost routing + web fallback."""

    crm_id = _get(row, "crm_id")
    xgb_status = (_get(row, "xgb_status") or "").strip().upper()
    chosen_siret_xgb = _get(row, "chosen_siret_xgb")
    score_top1 = float(_get(row, "score_top1") or 0.0)
    xgb_shap = _get(row, "xgb_shap_top1")

    def _base_result(
        *,
        status: str,
        chosen_siret: str | None,
        chosen_name: str | None,
        confidence: float,
        reason: str | None,
        sources: list[str] | None = None,
        candidate_count_total: int = 0,
        candidate_count_used: int = 0,
    ) -> dict:
        return {
            "crm_id": crm_id,
            "status": status,
            "chosen_siret": chosen_siret,
            "chosen_name": chosen_name,
            "confidence": confidence,
            "reason": reason,
            "crm_category": "INCONNU",
            "normalized_name": _crm_normalized_name(row),
            "normalized_address": _crm_normalized_address(row),
            "candidate_count_total": candidate_count_total,
            "candidate_count_used": candidate_count_used,
            "sources": sources or [],
            "xgb_status": xgb_status or None,
            "xgb_shap": xgb_shap,
        }

    if xgb_status == "AUTO" and chosen_siret_xgb:
        est = _lookup_establishment(conn, str(chosen_siret_xgb))
        chosen_name = est.get("denomination") if est else None
        return _base_result(
            status="MATCH",
            chosen_siret=str(chosen_siret_xgb),
            chosen_name=chosen_name,
            confidence=score_top1,
            reason="XGB_AUTO",
            sources=["XGB"],
        )

    if xgb_status not in {"REVIEW", "NO_MATCH"}:
        logger.warning("CRM %s: missing/invalid xgb_status=%s", crm_id, xgb_status)
        return _base_result(
            status="NO_MATCH",
            chosen_siret=None,
            chosen_name=None,
            confidence=0.0,
            reason="XGB_STATUS_MISSING",
            sources=[],
        )

    # Web fallback for REVIEW / NO_MATCH
    try:
        raw_candidates = search_web_sites(row, config=config, logger=logger)
    except Exception as exc:
        logger.error("CRM %s: Web search failed: %s", crm_id, exc)
        return _base_result(
            status=xgb_status,
            chosen_siret=None,
            chosen_name=None,
            confidence=score_top1,
            reason="WEB_SEARCH_ERROR",
            sources=[],
        )

    groups = group_raw_candidates(raw_candidates)
    candidates = enrich_candidates_from_sirene(groups, conn, logger)
    candidate_count_total = len(candidates)

    chosen, evidence, confidence, eligible_count = _select_web_candidate(
        row,
        candidates,
        strict_for_no_match=(xgb_status == "NO_MATCH"),
        logger=logger,
    )

    if not chosen:
        return _base_result(
            status=xgb_status,
            chosen_siret=None,
            chosen_name=None,
            confidence=score_top1,
            reason="WEB_NO_UPGRADE",
            sources=[],
            candidate_count_total=candidate_count_total,
            candidate_count_used=eligible_count,
        )

    reason = "WEB_UPGRADE"
    if evidence:
        reason = f"WEB_UPGRADE: {json.dumps(evidence, ensure_ascii=False)}"

    return _base_result(
        status="MATCH",
        chosen_siret=chosen.siret,
        chosen_name=chosen.name,
        confidence=confidence,
        reason=reason,
        sources=chosen.sources,
        candidate_count_total=candidate_count_total,
        candidate_count_used=eligible_count,
    )


__all__.append("process_crm_row_xgb")


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
    "xgb_status",
    "xgb_shap",
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

    def _should_parallelize() -> tuple[bool, int]:
        if str(getattr(config, "llm_provider", "")).lower() != "openrouter":
            return False, 0
        workers = max(
            1,
            min(
                int(getattr(config, "llm_max_concurrency", 1) or 1),
                os.cpu_count() or 4,
            ),
        )
        return workers > 1, workers

    df = load_crm(config.crm_path)
    if max_rows is not None:
        df = df.head(max_rows)

    if df.empty:
        log.warning("CRM input is empty; returning empty results DataFrame.")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    # Optional: merge XGBoost routed results (one row per crm_id)
    if config.xgb_routed_path is not None:
        xgb_path = config.xgb_routed_path
        if not xgb_path.exists():
            raise FileNotFoundError(f"XGB routed results not found: {xgb_path}")
        xgb_df = pd.read_csv(xgb_path)
        if "crm_id" not in xgb_df.columns:
            raise ValueError("XGB routed file missing required column: crm_id")
        df = df.merge(xgb_df, on="crm_id", how="left", suffixes=("", "_xgb"))
        missing = df["xgb_status"].isna().sum() if "xgb_status" in df.columns else len(df)
        if missing:
            log.warning("XGB routed merge: %d CRM rows missing xgb_status", missing)

    communes = extract_communes(df)

    # Initialize cache and preload SIRENE sequentially (I/O bound).
    conn = get_cache_connection(config)
    parallel_enabled, max_workers = _should_parallelize()
    llm_client = None
    rne_client: RneClient | None = None

    try:
        preload_sirene(communes, config, logger=log, conn=conn)

        if config.xgb_routed_path is not None:
            # XGBoost-first pipeline (no LLM/RNE/DataGouv).
            results: list[dict] = []
            iterable = _progress(
                df.itertuples(index=False, name="CRMRow"),
                total=len(df),
                description="Processing CRM (XGB)",
            )
            for row in iterable:
                results.append(process_crm_row_xgb(row, config, conn, log))
            return pd.DataFrame(results, columns=RESULT_COLUMNS)

        if parallel_enabled:
            log.info(
                "Running pipeline in parallel mode (provider=openrouter) with %d workers",
                max_workers,
            )

            def _task(row):
                local_conn = get_cache_connection(config, initialize=False)
                local_llm = create_llm_client(config=config, logger=log)
                local_rne = RneClient(config=config, logger=log)
                try:
                    return process_crm_row(
                        row,
                        config,
                        local_conn,
                        log,
                        rne_client=local_rne,
                        llm_client=local_llm,
                    )
                finally:
                    try:
                        local_llm.close()
                    except Exception:
                        log.exception("Failed to close LLM client")
                    try:
                        local_conn.close()
                    except Exception:
                        log.exception("Failed to close SQLite connection")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                rows = list(df.itertuples(index=False, name="CRMRow"))
                results = list(_progress(executor.map(_task, rows), total=len(rows), description="Processing CRM"))
            return pd.DataFrame(results, columns=RESULT_COLUMNS)

        # Sequential fallback (Ollama or single-worker)
        rne_client = RneClient(config=config, logger=log)
        llm_client = create_llm_client(config=config, logger=log)

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
