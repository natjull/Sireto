"""External source clients orchestrators (task 4.x).

Implements:
- RNE search (task 4.2)
- DataGouv Annuaire-entreprises search (task 4.3)
- Official web search (Google / Brave) for multisite lookups.
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

# Brave rate limiter: 1 request per second on the Free plan
_BRAVE_LAST_REQUEST_TS: float = 0.0
_BRAVE_RATE_LIMIT_SEC: float = 1.0


class DataGouvApiError(RuntimeError):
    """Raised when the DataGouv API fails after retries or returns invalid data."""


class WebSearchApiError(RuntimeError):
    """Raised when the web search API fails after retries or returns invalid data."""


class QwantApiError(WebSearchApiError):
    """Deprecated alias for legacy Qwant errors (kept for backward compatibility)."""


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
    "WebSearchApiError",
    "QwantApiError",
    "search_datagouv",
    "search_rne",
    "web_search",
    "search_web_sites",
    "extract_siren_siret_from_url",
    "extract_siren_siret_from_text",
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


# --------------------------------------------------------------------------- Web search (Google/Brave)


WEB_PROVIDER_GOOGLE = "google"
WEB_PROVIDER_BRAVE = "brave"
WEB_PROVIDERS = (WEB_PROVIDER_GOOGLE, WEB_PROVIDER_BRAVE)


@dataclass(frozen=True)
class WebSearchResult:
    provider: str
    items: list[dict]
    from_cache: bool = False


WEB_SITES = {
    "WEB_PAPPERS": "pappers.fr",
    "WEB_ANNUAIRE": "annuaire-entreprises.data.gouv.fr",
    "WEB_SOCIETE": "societe.com",
    "WEB_LEFIGARO": "entreprises.lefigaro.fr",
}


def _hash_query(query: str) -> str:
    return hashlib.sha1(query.encode("utf-8")).hexdigest()


def _open_web_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_web_cache(conn)
    conn.row_factory = sqlite3.Row
    return conn


def _init_web_cache(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS web_cache (
            cache_key TEXT PRIMARY KEY,
            provider TEXT,
            query_hash TEXT,
            query TEXT,
            response_json TEXT,
            created_at REAL,
            expires_at REAL,
            negative INTEGER DEFAULT 0,
            result_count INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS web_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT,
            ts REAL,
            query_hash TEXT,
            status_code INTEGER,
            from_cache INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_web_cache_provider_query ON web_cache(provider, query_hash)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_web_requests_provider_ts ON web_requests(provider, ts)"
    )
    conn.commit()


def _cache_key(provider: str, query_hash: str) -> str:
    return f"{provider}:{query_hash}"


def _cache_get(
    conn: sqlite3.Connection,
    provider: str,
    query_hash: str,
    now: float,
) -> list[dict] | None:
    key = _cache_key(provider, query_hash)
    row = conn.execute(
        "SELECT response_json, expires_at FROM web_cache WHERE cache_key = ?",
        (key,),
    ).fetchone()
    if not row:
        return None
    if row["expires_at"] is not None and row["expires_at"] < now:
        conn.execute("DELETE FROM web_cache WHERE cache_key = ?", (key,))
        conn.commit()
        return None
    try:
        return json.loads(row["response_json"])
    except Exception:
        return None


def _cache_set(
    conn: sqlite3.Connection,
    *,
    provider: str,
    query_hash: str,
    query: str,
    items: list[dict],
    ttl_seconds: float,
    negative: bool,
    now: float,
) -> None:
    key = _cache_key(provider, query_hash)
    expires_at = now + ttl_seconds if ttl_seconds > 0 else None
    conn.execute(
        """
        INSERT OR REPLACE INTO web_cache
        (cache_key, provider, query_hash, query, response_json, created_at, expires_at, negative, result_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            key,
            provider,
            query_hash,
            query,
            json.dumps(items, ensure_ascii=False),
            now,
            expires_at,
            1 if negative else 0,
            len(items),
        ),
    )
    conn.commit()


def _record_request(
    conn: sqlite3.Connection,
    provider: str,
    query_hash: str,
    *,
    status_code: int,
    from_cache: bool,
    now: float,
) -> None:
    conn.execute(
        """
        INSERT INTO web_requests (provider, ts, query_hash, status_code, from_cache)
        VALUES (?, ?, ?, ?, ?)
        """,
        (provider, now, query_hash, status_code, 1 if from_cache else 0),
    )
    conn.commit()


def _remaining_quota(
    conn: sqlite3.Connection,
    provider: str,
    limit: int,
    now: float,
) -> int:
    since = now - 24 * 3600
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM web_requests WHERE provider = ? AND ts >= ? AND from_cache = 0",
        (provider, since),
    ).fetchone()
    used = int(row["cnt"]) if row else 0
    return max(limit - used, 0)


def _provider_enabled(provider: str, config: PipelineConfig) -> bool:
    if provider == WEB_PROVIDER_GOOGLE:
        return bool(config.google_api_key and config.google_cse_id)
    if provider == WEB_PROVIDER_BRAVE:
        return bool(config.brave_api_key)
    return False


def _last_provider(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT provider FROM web_requests WHERE from_cache = 0 ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    return row["provider"] if row else None


def _select_provider(
    conn: sqlite3.Connection,
    config: PipelineConfig,
    now: float,
    logger: logging.Logger,
) -> str | None:
    google_remaining = (
        _remaining_quota(conn, WEB_PROVIDER_GOOGLE, config.google_daily_quota, now)
        if _provider_enabled(WEB_PROVIDER_GOOGLE, config)
        else 0
    )
    brave_remaining = (
        _remaining_quota(conn, WEB_PROVIDER_BRAVE, config.brave_daily_quota, now)
        if _provider_enabled(WEB_PROVIDER_BRAVE, config)
        else 0
    )

    if google_remaining <= 0 and brave_remaining <= 0:
        logger.warning(
            "Web search skipped: quotas exhausted (google=%s, brave=%s)",
            google_remaining,
            brave_remaining,
        )
        return None

    if google_remaining > 0 and brave_remaining > 0:
        if google_remaining != brave_remaining:
            return WEB_PROVIDER_GOOGLE if google_remaining > brave_remaining else WEB_PROVIDER_BRAVE
        last = _last_provider(conn)
        return WEB_PROVIDER_BRAVE if last == WEB_PROVIDER_GOOGLE else WEB_PROVIDER_GOOGLE

    return WEB_PROVIDER_GOOGLE if google_remaining > 0 else WEB_PROVIDER_BRAVE


def _google_search(
    query: str,
    config: PipelineConfig,
    logger: logging.Logger,
) -> tuple[int, list[dict]]:
    if not config.google_api_key or not config.google_cse_id:
        raise WebSearchApiError("Google API key or CSE id missing")
    params = {
        "key": config.google_api_key,
        "cx": config.google_cse_id,
        "q": query,
        "num": 10,
    }
    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params=params,
        timeout=float(config.llm_timeout_sec),
    )
    status = resp.status_code
    if status != 200:
        logger.warning("Google search HTTP %s: %s", status, resp.text[:200])
        return status, []
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise WebSearchApiError("Google response is not valid JSON") from exc
    items = []
    for idx, item in enumerate(data.get("items") or [], start=1):
        url = item.get("link")
        if not url:
            continue
        items.append(
            {
                "url": url,
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "rank": idx - 1,
            }
        )
    return status, items


def _brave_search(
    query: str,
    config: PipelineConfig,
    logger: logging.Logger,
) -> tuple[int, list[dict]]:
    global _BRAVE_LAST_REQUEST_TS
    
    if not config.brave_api_key:
        raise WebSearchApiError("Brave API key missing")
    
    # Rate limiting: ensure at least 1 second between Brave requests
    now = time.time()
    elapsed = now - _BRAVE_LAST_REQUEST_TS
    if elapsed < _BRAVE_RATE_LIMIT_SEC:
        sleep_time = _BRAVE_RATE_LIMIT_SEC - elapsed
        logger.debug("Brave rate limit: sleeping %.2fs", sleep_time)
        time.sleep(sleep_time)
    _BRAVE_LAST_REQUEST_TS = time.time()
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": config.brave_api_key,
    }
    params = {
        "q": query,
        "count": 10,
        "offset": 0,
        "search_lang": "fr",
    }
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params=params,
        headers=headers,
        timeout=float(config.llm_timeout_sec),
    )
    status = resp.status_code
    if status != 200:
        logger.warning("Brave search HTTP %s: %s", status, resp.text[:200])
        return status, []
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise WebSearchApiError("Brave response is not valid JSON") from exc
    results = data.get("web", {}).get("results") or []
    items = []
    for idx, item in enumerate(results, start=1):
        url = item.get("url")
        if not url:
            continue
        items.append(
            {
                "url": url,
                "title": item.get("title"),
                "snippet": item.get("description"),
                "rank": idx - 1,
            }
        )
    return status, items


def web_search(
    query: str,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
) -> WebSearchResult:
    """Search the web via Google or Brave, with cache and quotas.

    `query` is the base query (without provider-specific site filters). For
    Google PSE, site filters can be configured in the engine, so we keep the
    query clean.
    """

    log = logger or LOGGER
    if not config.web_search_enabled:
        log.info("Web search disabled via config; skipping search")
        return WebSearchResult(provider="disabled", items=[], from_cache=True)

    now = time.time()
    query_hash = _hash_query(query)

    conn = _open_web_cache(Path(config.web_cache_path))
    try:
        google_query = _build_query_for_provider(query, WEB_PROVIDER_GOOGLE, config)
        brave_query = _build_query_for_provider(query, WEB_PROVIDER_BRAVE, config)

        cached_google = _cache_get(conn, WEB_PROVIDER_GOOGLE, _hash_query(google_query), now)
        if cached_google is not None:
            _record_request(
                conn,
                WEB_PROVIDER_GOOGLE,
                _hash_query(google_query),
                status_code=200,
                from_cache=True,
                now=now,
            )
            return WebSearchResult(
                provider=WEB_PROVIDER_GOOGLE, items=cached_google, from_cache=True
            )

        cached_brave = _cache_get(conn, WEB_PROVIDER_BRAVE, _hash_query(brave_query), now)
        if cached_brave is not None:
            _record_request(
                conn,
                WEB_PROVIDER_BRAVE,
                _hash_query(brave_query),
                status_code=200,
                from_cache=True,
                now=now,
            )
            return WebSearchResult(
                provider=WEB_PROVIDER_BRAVE, items=cached_brave, from_cache=True
            )

        provider = _select_provider(conn, config, now, log)
        if provider is None:
            return WebSearchResult(provider="none", items=[], from_cache=False)

        status_code = 0
        items: list[dict] = []
        provider_query = _build_query_for_provider(query, provider, config)
        provider_hash = _hash_query(provider_query)
        try:
            if provider == WEB_PROVIDER_GOOGLE:
                status_code, items = _google_search(provider_query, config, log)
            else:
                status_code, items = _brave_search(provider_query, config, log)
        except WebSearchApiError as exc:
            log.error("Web search failed (%s): %s", provider, exc)
            status_code = 0
            items = []

        _record_request(
            conn,
            provider,
            provider_hash,
            status_code=status_code,
            from_cache=False,
            now=now,
        )

        if items:
            ttl = float(config.web_cache_ttl_days) * 24 * 3600
            _cache_set(
                conn,
                provider=provider,
                query_hash=provider_hash,
                query=provider_query,
                items=items,
                ttl_seconds=ttl,
                negative=False,
                now=now,
            )
        else:
            ttl = float(config.web_negative_cache_ttl_hours) * 3600
            _cache_set(
                conn,
                provider=provider,
                query_hash=provider_hash,
                query=provider_query,
                items=[],
                ttl_seconds=ttl,
                negative=True,
                now=now,
            )

        return WebSearchResult(provider=provider, items=items, from_cache=False)
    finally:
        conn.close()


def qwant_search(
    query: str,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Deprecated Qwant wrapper, mapped to web_search for backward compatibility."""

    result = web_search(query, config, logger)
    return result.items


def extract_siren_siret_from_url(url: str) -> tuple[str | None, str | None]:
    """Extract SIREN/SIRET from URL path + query string.

    Strategy: pick last 14-digit (SIRET) if present; else last 9-digit (SIREN).
    Returns (siren, siret) with siret possibly None.
    """

    parsed = urlparse(url)
    search_zone = parsed.path + ("?" + parsed.query if parsed.query else "")
    return extract_siren_siret_from_text(search_zone)


def extract_siren_siret_from_text(text: str) -> tuple[str | None, str | None]:
    """Extract SIREN/SIRET from free text (URL/title/snippet)."""

    patterns_14 = re.findall(r"\b(\d{14})\b", text)
    if patterns_14:
        siret = patterns_14[-1]
        siren = siret[:9]
        return siren, siret

    patterns_9 = re.findall(r"\b(\d{9})\b", text)
    if patterns_9:
        siren = patterns_9[-1]
        return siren, None

    return None, None


def _match_site(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    for source_name, domain in WEB_SITES.items():
        if host == domain or host.endswith(f".{domain}"):
            return source_name, domain
    return None


def _build_multisite_query(base_query: str) -> str:
    site_clause = " OR ".join(f"site:{domain}" for domain in WEB_SITES.values())
    return f"{base_query} ({site_clause})"


def _build_query_for_provider(base_query: str, provider: str, config: PipelineConfig) -> str:
    # Google PSE can enforce sites at the engine level; keep query clean.
    if provider == WEB_PROVIDER_GOOGLE and config.google_cse_id:
        return base_query
    return _build_multisite_query(base_query)


def search_web_sites(
    row: pd.Series,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
) -> List[RawCandidate]:
    """Search the web on multiple business sites with a single query."""

    log = logger or LOGGER

    def _value(key: str) -> str:
        # tolerate Series, dict-like, or itertuples rows
        if hasattr(row, "get"):
            try:
                val = row.get(key, "")  # type: ignore[call-arg]
            except Exception:
                val = ""
        elif hasattr(row, "_asdict"):
            val = row._asdict().get(key, "")  # type: ignore[call-arg]
        elif hasattr(row, key):
            val = getattr(row, key, "")
        else:
            try:
                val = row[key]  # type: ignore[index]
            except Exception:
                val = ""
        if pd.isna(val):
            return ""
        return str(val).strip()

    parts: list[str] = []
    for col in ["crm_name", "street_number", "street_name", "postcode"]:
        val = _value(col)
        if val:
            if col == "postcode":
                parts.append(f'"{val}"')
            else:
                parts.append(val)

    base_query = " ".join(parts)
    if not base_query:
        log.warning("Empty query for web search - skipping")
        return []

    query = base_query
    log.debug("Web search base query: %s", query[:200])

    try:
        result = web_search(query, config, log)
    except WebSearchApiError as exc:
        log.error("Web search failed: %s", exc)
        return []

    items = result.items
    provider = result.provider

    all_candidates: list[RawCandidate] = []
    counts: dict[str, int] = {key: 0 for key in WEB_SITES}
    seen_per_source: dict[str, set[tuple[str, str]]] = {key: set() for key in WEB_SITES}

    rank = 0
    for item in items:
        url = item.get("url")
        if not url:
            continue

        matched = _match_site(url)
        if matched is None:
            continue
        source_name, domain = matched
        if counts[source_name] >= config.max_candidates_per_source:
            continue

        siren, siret = extract_siren_siret_from_url(url)
        if not siren and not siret:
            text = " ".join(
                str(x)
                for x in [
                    item.get("title"),
                    item.get("snippet"),
                ]
                if x
            )
            siren, siret = extract_siren_siret_from_text(text)
        if not siren and not siret:
            log.debug("No SIREN/SIRET found in URL/snippet: %s", url)
            continue

        label = (item.get("title") or "").strip()

        key = ("siret", siret) if siret else ("siren", siren) if siren else None
        if key is not None:
            if key in seen_per_source[source_name]:
                continue
            seen_per_source[source_name].add(key)

        extra: dict[str, Any] = {
            "query": base_query,
            "site_domain": domain,
            "provider": provider,
            "rank": rank,
            "title": item.get("title"),
            "snippet": item.get("snippet"),
        }
        if config.store_source_raw:
            extra["raw"] = item

        try:
            candidate = create_raw_candidate(
                source=source_name,
                siren=siren,
                siret=siret,
                label=label,
                url=url,
                extra=extra,
            )
        except InvalidCandidateError as exc:
            log.debug("Invalid candidate from %s: %s", source_name, exc)
            continue

        all_candidates.append(candidate)
        counts[source_name] += 1
        rank += 1

    log.info(
        "Web candidates: crm_name=%s postcode=%s -> total=%d (PAPPERS=%d, ANNUAIRE=%d, SOCIETE=%d, LEFIGARO=%d) provider=%s",
        _value("crm_name"),
        _value("postcode"),
        len(all_candidates),
        counts.get("WEB_PAPPERS", 0),
        counts.get("WEB_ANNUAIRE", 0),
        counts.get("WEB_SOCIETE", 0),
        counts.get("WEB_LEFIGARO", 0),
        provider,
    )

    return all_candidates


def search_qwant_sites(
    row: pd.Series,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
) -> List[RawCandidate]:
    """Deprecated alias for backward compatibility."""

    return search_web_sites(row, config, logger)
