"""External source clients orchestrators (task 4.x).

Implements:
- RNE search (task 4.2)
- DataGouv Annuaire-entreprises search (task 4.3)

"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import requests

import pandas as pd

from .config import PipelineConfig
from .candidate_store import (
    RawCandidate,
    InvalidCandidateError,
    candidate_key,
    create_raw_candidate,
)
from .rne_client import RneClient, _map_company_to_candidates


LOGGER = logging.getLogger(__name__)






class DataGouvApiError(RuntimeError):
    """Raised when the DataGouv API fails after retries or returns invalid data."""


def search_rne(
    normalized_name: str,
    city: str,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
    postcode: str | None = None,
    client: RneClient | None = None,
) -> List[RawCandidate]:
    """Search RNE (INPI) and return RawCandidate list.

    Args:
        normalized_name: Normalized company name (LLM #1 output).
        city: CRM city (informational; not used by the RNE query).
        config: Pipeline configuration.
        logger: Optional logger.
        postcode: Optional postcode to refine the search via zipCodes[].
        client: Optional RneClient instance to reuse across calls.
    """

    log = logger or LOGGER
    close_client = False
    if client is None:
        client = RneClient(config=config, logger=log)
        close_client = True  # currently no close action, kept for symmetry

    query_text = f"{normalized_name} {postcode or ''}".strip()
    companies = client.search_companies(
        normalized_name,
        postcode,
        max_results=config.max_candidates_per_source,
    )

    candidates: list[RawCandidate] = []
    for company in companies:
        candidates.extend(
            _map_company_to_candidates(
                company,
                query=query_text,
                rank_offset=len(candidates),
                store_raw=config.store_source_raw,
            )
        )

    candidates = _deduplicate_candidates(
        candidates,
        max_candidates=config.max_candidates_per_source,
    )

    log.info(
        "RNE candidates: name=%s city=%s postcode=%s -> %s",
        normalized_name,
        city,
        postcode,
        len(candidates),
    )

    if close_client:
        # No persistent resources to close, placeholder for future.
        pass

    return candidates


__all__ = [
    "DataGouvApiError",
    "search_datagouv",
    "search_rne",
    ]


def _deduplicate_candidates(
    candidates: List[RawCandidate],
    *,
    max_candidates: int,
) -> List[RawCandidate]:
    """Keep at most `max_candidates`, deduplicating on SIRET/SIREN keys."""

    if max_candidates <= 0:
        return []

    deduped: list[RawCandidate] = []
    seen: set[tuple[str, str]] = set()

    for candidate in candidates:
        if len(deduped) >= max_candidates:
            break

        key = candidate_key(candidate)
        if key is None:
            deduped.append(candidate)
            continue

        if key in seen:
            continue

        seen.add(key)
        deduped.append(candidate)

    return deduped


# --------------------------------------------------------------------------- DataGouv (Annuaire-entreprises)


def _http_get_datagouv(
    url: str,
    params: Dict[str, Any],
    *,
    timeout: float = 20.0,
    connect_timeout: float = 5.0,
    max_retries: int = 3,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """GET JSON from DataGouv with retry on 429/5xx."""

    logger = logger or LOGGER
    _ = connect_timeout  # kept for signature parity / future fine-tuning
    query = urlencode(params, doseq=True)
    full_url = f"{url}?{query}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "Sireto-PipeV6/2.2",
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = Request(full_url, headers=headers, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except HTTPError as e:
            status = e.code
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""

            if status in (429, 500, 502, 503, 504):
                retry_after = e.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2.0 * attempt, 10.0)
                logger.warning(
                    "DataGouv HTTP %s (attempt %s/%s) – sleeping %.1fs",
                    status,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                last_exc = e
                continue
            raise DataGouvApiError(
                f"DataGouv request failed HTTP {status}: {body[:200]}"
            ) from e
        except URLError as e:
            delay = min(2.0 * attempt, 10.0)
            logger.warning(
                "DataGouv URLError (attempt %s/%s) – sleeping %.1fs",
                attempt,
                max_retries,
                delay,
            )
            time.sleep(delay)
            last_exc = e
            continue
        except json.JSONDecodeError as e:
            raise DataGouvApiError("DataGouv response is not valid JSON") from e

    assert last_exc is not None
    raise DataGouvApiError(f"DataGouv request failed after {max_retries} attempts: {last_exc}")


def _map_datagouv_result_to_candidates(
    result: Dict[str, Any],
    *,
    query: str,
    rank_offset: int = 0,
    store_raw: bool = True,
) -> List[RawCandidate]:
    """Map a DataGouv result to RawCandidate instances (siège + matching établissements)."""

    siren = result.get("siren")
    if not siren:
        return []

    label = result.get("nom_complet") or result.get("nom_raison_sociale")
    nature_juridique = result.get("nature_juridique")
    etat_admin = result.get("etat_administratif")
    activite = result.get("activite_principale")

    url = f"https://annuaire-entreprises.data.gouv.fr/entreprise/{siren}"

    extras_base: dict[str, Any] = {
        "query": query,
        "nature_juridique": nature_juridique,
        "etat_administratif": etat_admin,
        "activite_principale": activite,
    }
    if store_raw:
        extras_base["raw"] = result

    candidates: list[RawCandidate] = []
    rank = rank_offset

    siege = result.get("siege") or {}
    siret_siege = siege.get("siret")
    if siret_siege:
        extras_siege = {
            **extras_base,
            "rank": rank,
            "adresse": siege.get("adresse"),
            "date_creation": siege.get("date_creation"),
        }
        candidates.append(
            create_raw_candidate(
                source="DATAGOUV",
                siren=siren,
                siret=siret_siege,
                label=label,
                url=url,
                extra=extras_siege,
            )
        )
        rank += 1

    for etab in result.get("matching_etablissements") or []:
        siret_etab = etab.get("siret")
        if not siret_etab:
            continue
        extras_etab = {
            **extras_base,
            "rank": rank,
            "adresse": etab.get("adresse"),
            "date_creation": etab.get("date_creation"),
            "est_siege": etab.get("est_siege", False),
        }
        candidates.append(
            create_raw_candidate(
                source="DATAGOUV",
                siren=siren,
                siret=siret_etab,
                label=label,
                url=url,
                extra=extras_etab,
            )
        )
        rank += 1

    return candidates


def search_datagouv(
    normalized_name: str,
    city: str,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
    postcode: str | None = None,
) -> List[RawCandidate]:
    """Search DataGouv annuaire-entreprises API and return RawCandidate list."""

    log = logger or LOGGER
    url = f"{config.datagouv_api_url.rstrip('/')}/search"

    if postcode:
        query_text = normalized_name
        params = {"q": query_text, "code_postal": postcode}
    else:
        query_text = f"{normalized_name} {city}".strip()
        params = {"q": query_text}

    params["per_page"] = min(config.max_candidates_per_source, 25)
    params["page"] = 1

    log.debug("DataGouv API call: %s?%s", url, urlencode(params))

    response = _http_get_datagouv(
        url,
        params,
        timeout=float(config.llm_timeout_sec),
        connect_timeout=float(config.llm_connect_timeout_sec),
        max_retries=int(config.llm_max_retries),
        logger=log,
    )

    results = response.get("results")
    if not isinstance(results, list):
        raise DataGouvApiError("DataGouv response missing 'results' field")

    all_candidates: list[RawCandidate] = []
    for result in results:
        all_candidates.extend(
            _map_datagouv_result_to_candidates(
                result,
                query=query_text,
                rank_offset=len(all_candidates),
                store_raw=config.store_source_raw,
            )
        )

    candidates = _deduplicate_candidates(
        all_candidates,
        max_candidates=config.max_candidates_per_source,
    )

    total_results = response.get("total_results", len(results))
    total_pages = response.get("total_pages", 1)
    page = response.get("page", 1)
    log.debug(
        "DataGouv response: total_results=%s page=%s/%s",
        total_results,
        page,
        total_pages,
    )

    log.info(
        "DataGouv candidates: name=%s city=%s postcode=%s -> %d",
        normalized_name,
        city,
        postcode,
        len(candidates),
    )

    return candidates


