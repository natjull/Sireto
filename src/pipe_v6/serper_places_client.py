"""Serper.dev Google Places API client for V7 Places-guided candidate generation.

This module provides:
- search_places(): Main API call to Serper Places endpoint
- PlacesResult: Structured dataclass for API response
- Rate limiting and caching integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import requests

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)

# Rate limiter: Serper allows higher rates but we keep 1 req/sec for safety
_SERPER_LAST_REQUEST_TS: float = 0.0
_SERPER_RATE_LIMIT_SEC: float = 1.0

SERPER_PLACES_URL = "https://google.serper.dev/places"


class SerperApiError(RuntimeError):
    """Raised when the Serper API fails after retries or returns invalid data."""


@dataclass
class PlacesResult:
    """Structured result from Serper Places API."""

    position: int
    title: str
    address: str
    latitude: float | None = None
    longitude: float | None = None
    rating: float | None = None
    rating_count: int | None = None
    category: str | None = None
    phone_number: str | None = None
    website: str | None = None
    cid: str | None = None
    # Parsed address components
    street_number: str | None = None
    street_name: str | None = None
    postcode: str | None = None
    city: str | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_api_response(cls, item: dict, position: int) -> "PlacesResult":
        """Parse a single place from API response."""
        address = item.get("address", "")
        parsed = _parse_french_address(address)

        return cls(
            position=position,
            title=item.get("title", ""),
            address=address,
            latitude=item.get("latitude"),
            longitude=item.get("longitude"),
            rating=item.get("rating"),
            rating_count=item.get("ratingCount"),
            category=item.get("category"),
            phone_number=item.get("phoneNumber"),
            website=item.get("website"),
            cid=item.get("cid"),
            street_number=parsed.get("street_number"),
            street_name=parsed.get("street_name"),
            postcode=parsed.get("postcode"),
            city=parsed.get("city"),
            raw=item,
        )


@dataclass
class PlacesSearchResponse:
    """Full response from Places search."""

    query: str
    places: List[PlacesResult]
    from_cache: bool = False
    raw_response: dict | None = None


def _parse_french_address(address: str) -> dict:
    """Parse a French formatted address into components.

    Examples:
      - "33 Av. Franklin Roosevelt, 69150 Décines-Charpieu"
      - "15-17 Rue Emile Zola, 69150 Décines-Charpieu"

    Returns:
      {street_number, street_name, postcode, city}
    """
    result: dict = {
        "street_number": None,
        "street_name": None,
        "postcode": None,
        "city": None,
    }

    if not address:
        return result

    # Try to extract postcode (5 digits) and city
    postcode_match = re.search(r"\b(\d{5})\b\s*([^,]+)?$", address)
    if postcode_match:
        result["postcode"] = postcode_match.group(1)
        city_part = postcode_match.group(2)
        if city_part:
            result["city"] = city_part.strip()

    # Extract street part (before the postcode)
    street_part = address
    if postcode_match:
        street_part = address[: postcode_match.start()].rstrip(", ")

    # Try to extract street number at the beginning (supports ranges and BIS/TER)
    street_num_match = re.match(
        r"^\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*(?:[A-Za-z]|BIS|TER|QUATER))?)\s+(.+)",
        street_part,
        flags=re.IGNORECASE,
    )
    if street_num_match:
        raw_num = street_num_match.group(1).strip()
        # Normalize number: remove extra spaces around hyphens, uppercase suffix
        raw_num = re.sub(r"\s*[-–]\s*", "-", raw_num)
        raw_num = re.sub(r"\s+", " ", raw_num).upper()

        street_name = street_num_match.group(2).strip()
        # Remove duplicated leading number in street name (e.g., "107-115 107 RUE ...")
        street_name = re.sub(r"^\d+\s*(?:[-–]\s*\d+)?\s+", "", street_name).strip()

        result["street_number"] = raw_num
        result["street_name"] = street_name or None
    else:
        result["street_name"] = street_part.strip() if street_part else None

    return result


def _hash_query(query: str) -> str:
    """Generate a cache key hash for the query."""
    return hashlib.sha1(query.encode("utf-8")).hexdigest()


def _open_places_cache(path: Path) -> sqlite3.Connection:
    """Open or create the Places cache database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_places_cache(conn)
    conn.row_factory = sqlite3.Row
    return conn


def _init_places_cache(conn: sqlite3.Connection) -> None:
    """Create the cache table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS places_cache (
            cache_key TEXT PRIMARY KEY,
            query TEXT,
            query_hash TEXT,
            response_json TEXT,
            created_at REAL,
            expires_at REAL,
            result_count INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_places_cache_query ON places_cache(query_hash)"
    )
    conn.commit()


def _cache_get(
    conn: sqlite3.Connection,
    query_hash: str,
    now: float,
) -> List[dict] | None:
    """Retrieve cached results if valid."""
    row = conn.execute(
        "SELECT response_json, expires_at FROM places_cache WHERE cache_key = ?",
        (query_hash,),
    ).fetchone()
    if not row:
        return None
    if row["expires_at"] is not None and row["expires_at"] < now:
        conn.execute("DELETE FROM places_cache WHERE cache_key = ?", (query_hash,))
        conn.commit()
        return None
    try:
        return json.loads(row["response_json"])
    except Exception:
        return None


def _cache_set(
    conn: sqlite3.Connection,
    *,
    query_hash: str,
    query: str,
    places: List[dict],
    ttl_seconds: float,
    now: float,
) -> None:
    """Store results in cache."""
    expires_at = now + ttl_seconds if ttl_seconds > 0 else None
    conn.execute(
        """
        INSERT OR REPLACE INTO places_cache
        (cache_key, query, query_hash, response_json, created_at, expires_at, result_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query_hash,
            query,
            query_hash,
            json.dumps(places, ensure_ascii=False),
            now,
            expires_at,
            len(places),
        ),
    )
    conn.commit()


def _call_serper_places(
    query: str,
    config: PipelineConfig,
    logger: logging.Logger,
) -> tuple[int, List[dict]]:
    """Make the actual API call to Serper Places endpoint."""
    global _SERPER_LAST_REQUEST_TS

    if not config.serper_api_key:
        raise SerperApiError("Serper API key missing (set SIRETO_SERPER_API_KEY)")

    # Rate limiting
    now = time.time()
    elapsed = now - _SERPER_LAST_REQUEST_TS
    if elapsed < _SERPER_RATE_LIMIT_SEC:
        sleep_time = _SERPER_RATE_LIMIT_SEC - elapsed
        logger.debug("Serper rate limit: sleeping %.2fs", sleep_time)
        time.sleep(sleep_time)
    _SERPER_LAST_REQUEST_TS = time.time()

    headers = {
        "X-API-KEY": config.serper_api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "q": query,
        "gl": "fr",
        "hl": "fr",
    }

    try:
        resp = requests.post(
            SERPER_PLACES_URL,
            json=payload,
            headers=headers,
            timeout=float(config.llm_timeout_sec),
        )
    except requests.RequestException as exc:
        logger.error("Serper API request failed: %s", exc)
        return 0, []

    status = resp.status_code
    if status != 200:
        logger.warning("Serper Places HTTP %s: %s", status, resp.text[:200])
        return status, []

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise SerperApiError("Serper response is not valid JSON") from exc

    places = data.get("places") or []
    return status, places


def search_places(
    query: str,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
    *,
    use_cache: bool = True,
    cache_only: bool = False,
) -> PlacesSearchResponse:
    """Search Google Places via Serper.dev API.

    Args:
        query: Search query (e.g., "Stryker 16 rue de Lombardie 69150")
        config: Pipeline configuration with serper_api_key
        logger: Optional logger instance
    use_cache: Whether to use/store cached results
    cache_only: If True, never call the API (cache miss returns empty)

    Returns:
        PlacesSearchResponse with parsed PlacesResult objects
    """
    log = logger or LOGGER
    query_hash = _hash_query(query)
    now = time.time()

    if use_cache:
        conn = _open_places_cache(Path(config.web_cache_path))
        try:
            cached = _cache_get(conn, query_hash, now)
            if cached is not None:
                log.debug("Places cache hit for query: %s", query[:50])
                places = [
                    PlacesResult.from_api_response(item, idx)
                    for idx, item in enumerate(cached, start=1)
                ]
                return PlacesSearchResponse(
                    query=query,
                    places=places,
                    from_cache=True,
                    raw_response={"places": cached},
                )
        finally:
            conn.close()

    if cache_only:
        log.info("Places cache-only mode: miss for query '%s'", query[:50])
        return PlacesSearchResponse(query=query, places=[], from_cache=True, raw_response=None)

    # Make API call
    status, raw_places = _call_serper_places(query, config, log)

    places = [
        PlacesResult.from_api_response(item, idx)
        for idx, item in enumerate(raw_places, start=1)
    ]

    log.info(
        "Serper Places: query='%s' -> %d results (status=%d)",
        query[:50],
        len(places),
        status,
    )

    # Cache successful results
    if use_cache and raw_places:
        conn = _open_places_cache(Path(config.web_cache_path))
        try:
            ttl = float(config.web_cache_ttl_days) * 24 * 3600
            _cache_set(
                conn,
                query_hash=query_hash,
                query=query,
                places=raw_places,
                ttl_seconds=ttl,
                now=now,
            )
        finally:
            conn.close()

    return PlacesSearchResponse(
        query=query,
        places=places,
        from_cache=False,
        raw_response={"places": raw_places},
    )


def build_places_query(
    crm_name: str,
    street_number: str | None,
    street_name: str | None,
    postcode: str | None,
    city: str | None,
) -> str:
    """Build a Places API query from CRM components.

    Format: "{name} {address} {postcode} {city}"
    """
    parts: List[str] = []

    if crm_name:
        parts.append(crm_name.strip())

    if street_number:
        parts.append(street_number.strip())
    if street_name:
        parts.append(street_name.strip())
    if postcode:
        parts.append(postcode.strip())
    if city:
        parts.append(city.strip())

    return " ".join(parts)


__all__ = [
    "SerperApiError",
    "PlacesResult",
    "PlacesSearchResponse",
    "search_places",
    "build_places_query",
]
