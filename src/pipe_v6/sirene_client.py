"""SIRENE API client (task 2.2).

Implements fetching all establishments for a given commune and exposes
helper utilities used by the cache layer (task 2.4).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .config import PipelineConfig
from .commune_detection import CommuneKey


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Filter:
    key: str
    value: str


def _mask_key(k: str) -> str:
    if not k:
        return "<empty>"
    if len(k) <= 6:
        return "***"
    return f"{k[:4]}…{k[-4:]}"


def arrondissement_codes_for_parent(insee_code: str) -> List[str]:
    """Return arrondissement INSEE codes for Paris, Lyon, Marseille parent codes.

    - Paris parent: 75056 -> 75101..75120
    - Lyon parent: 69123 -> 69381..69389
    - Marseille parent: 13055 -> 13201..13216
    Otherwise, returns [insee_code].
    """

    if insee_code == "75056":
        return [f"75{100 + i:03d}" for i in range(1, 21)]  # 75101..75120
    if insee_code == "69123":
        return [f"6938{i}" for i in range(1, 10)]  # 69381..69389
    if insee_code == "13055":
        return [f"132{str(i).zfill(2)}" for i in range(1, 17)]  # 13201..13216
    return [insee_code]


def resolve_filters_for_commune(commune: CommuneKey) -> List[Filter]:
    """Compute the list of filters to fetch for this commune.

    Priority:
    - If INSEE is available: codeCommuneEtablissement for that code, expanding
      parent-city codes to arrondissement codes for P/L/M.
    - Else if postcode is available: codePostalEtablissement only (no city).
    - Else: raise ValueError (city-only fetch is not supported by the current rules).
    """

    if commune.insee_code:
        codes = arrondissement_codes_for_parent(commune.insee_code)
        return [Filter("codeCommuneEtablissement", c) for c in codes]
    if commune.postcode:
        return [Filter("codePostalEtablissement", commune.postcode)]
    raise ValueError("Cannot resolve filter: INSEE code or postcode required")


def _http_get_json(url: str, headers: Dict[str, str], *, timeout: float = 15.0, max_retries: int = 3, logger: logging.Logger | None = None) -> Dict:
    """HTTP GET with retries and basic 429 handling, returning parsed JSON."""

    logger = logger or LOGGER
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                return json.loads(data.decode("utf-8"))
        except HTTPError as e:
            status = e.code
            if status in (429, 500, 502, 503, 504):
                retry_after = e.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 2.0
                else:
                    delay = min(2.0 * attempt, 10.0) + (0.1 * (attempt - 1))
                logger.warning("HTTP %s on %s (attempt %s/%s) – sleeping %.1fs", status, url, attempt, max_retries, delay)
                time.sleep(delay)
                last_exc = e
                continue
            else:
                raise
        except URLError as e:
            delay = min(2.0 * attempt, 10.0)
            logger.warning("URLError on %s (attempt %s/%s) – sleeping %.1fs", url, attempt, max_retries, delay)
            time.sleep(delay)
            last_exc = e
            continue
    assert last_exc is not None
    raise last_exc


def _ensure_trailing_slash(base: str) -> str:
    return base if base.endswith('/') else base + '/'


def fetch_establishments_for_commune(
    commune: CommuneKey, config: PipelineConfig, logger: logging.Logger | None = None
) -> List[dict]:
    """Fetch all SIRENE establishments for a given commune.

    - Builds the appropriate filter(s) based on `CommuneKey`.
    - Paginates with `nombre=1000`, using `debut` offsets until all results are retrieved.
    - Returns a de-duplicated list of raw establishment dicts (complete objects).
    """

    logger = logger or LOGGER

    base = _ensure_trailing_slash(config.sirene_api_url)
    endpoint = urljoin(base, "siret")

    api_key = (getattr(config, "sirene_token", None) or getattr(config, "sirene_api_key", None) or "").strip()
    if not api_key:
        raise RuntimeError("SIRENE API key not configured (sirene_token or sirene_api_key)")

    headers = {
        "X-INSEE-Api-Key-Integration": api_key,
        "Accept": "application/json",
        "User-Agent": "Sireto-PipeV6/2.2",
    }

    filters = resolve_filters_for_commune(commune)
    logger.info(
        "SIRENE fetch: base=%s, key=%s, filters=%s",
        config.sirene_api_url,
        _mask_key(api_key),
        [f"{f.key}={f.value}" for f in filters],
    )

    all_records: List[dict] = []
    seen_siret: set[str] = set()
    per_page = 1000

    for flt in filters:
        # First page to get total
        params = {flt.key: flt.value, "nombre": per_page, "debut": 0}
        url = f"{endpoint}?{urlencode(params)}"
        payload = _http_get_json(url, headers, logger=logger)

        # API structures typically return header.total and etablissements[]
        header = payload.get("header", {})
        total = int(header.get("total", 0))
        results = payload.get("etablissements", []) or payload.get("etablissement", []) or []

        logger.info(
            "SIRENE page 1/?: %s=%s -> total=%s, returned=%s",
            flt.key,
            flt.value,
            total,
            len(results),
        )

        for rec in results:
            siret = str(rec.get("siret", "")).strip()
            if siret and siret not in seen_siret:
                all_records.append(rec)
                seen_siret.add(siret)

        # Remaining pages
        fetched = len(results)
        while fetched < total:
            params["debut"] = fetched
            url = f"{endpoint}?{urlencode(params)}"
            payload = _http_get_json(url, headers, logger=logger)
            results = payload.get("etablissements", []) or payload.get("etablissement", []) or []
            logger.debug(
                "SIRENE page next: %s=%s -> offset=%s, batch=%s",
                flt.key,
                flt.value,
                fetched,
                len(results),
            )
            for rec in results:
                siret = str(rec.get("siret", "")).strip()
                if siret and siret not in seen_siret:
                    all_records.append(rec)
                    seen_siret.add(siret)
            fetched += len(results)

    logger.info(
        "SIRENE fetch done: commune=%s (insee=%s, cp=%s), records=%s",
        commune.city or "",
        commune.insee_code or "",
        commune.postcode or "",
        len(all_records),
    )

    return all_records


__all__ = [
    "fetch_establishments_for_commune",
    "arrondissement_codes_for_parent",
    "resolve_filters_for_commune",
    "Filter",
]
