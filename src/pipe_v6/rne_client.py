"""INPI RNE API client (task 4.2).

Provides OAuth2 client-credentials authentication, pagination, retry logic,
and helpers to extract candidates from the `/api/companies` search endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from .config import PipelineConfig
from .candidate_store import RawCandidate, create_raw_candidate


LOGGER = logging.getLogger(__name__)


class RneApiError(RuntimeError):
    """Raised when the RNE API fails after retries or returns invalid data."""


def _mask_key(k: str) -> str:
    if not k:
        return "<empty>"
    if len(k) <= 6:
        return "***"
    return f"{k[:4]}…{k[-4:]}"


def _http_post_json(
    url: str,
    data: Dict[str, Any],
    *,
    timeout: float = 15.0,
    max_retries: int = 3,
    logger: logging.Logger | None = None,
    extra_headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """POST JSON and return parsed JSON with retry on 429/5xx."""

    logger = logger or LOGGER
    payload = json.dumps(data).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Sireto-PipeV6/2.2",
    }
    if extra_headers:
        headers.update(extra_headers)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = Request(url, data=payload, headers=headers, method="POST")
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
                    "HTTP %s on %s (attempt %s/%s) – sleeping %.1fs",
                    status,
                    url,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                last_exc = e
                continue
            raise RneApiError(f"Auth request failed HTTP {status}: {body[:200]}") from e
        except URLError as e:
            delay = min(2.0 * attempt, 10.0)
            logger.warning(
                "URLError on %s (attempt %s/%s) – sleeping %.1fs",
                url,
                attempt,
                max_retries,
                delay,
            )
            time.sleep(delay)
            last_exc = e
            continue
        except json.JSONDecodeError as e:
            raise RneApiError("Auth response is not valid JSON") from e

    assert last_exc is not None
    raise RneApiError(f"Auth failed after {max_retries} attempts: {last_exc}")


def _http_get_json(
    url: str,
    headers: Dict[str, str],
    *,
    timeout: float = 20.0,
    max_retries: int = 3,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """GET JSON with retries on rate-limit/server errors."""

    logger = logger or LOGGER
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = Request(url, headers=headers, method="GET")
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
                    "HTTP %s on %s (attempt %s/%s) – sleeping %.1fs",
                    status,
                    url,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                last_exc = e
                continue
            raise RneApiError(f"RNE request failed HTTP {status}: {body[:200]}") from e
        except URLError as e:
            delay = min(2.0 * attempt, 10.0)
            logger.warning(
                "URLError on %s (attempt %s/%s) – sleeping %.1fs",
                url,
                attempt,
                max_retries,
                delay,
            )
            time.sleep(delay)
            last_exc = e
            continue
        except json.JSONDecodeError as e:
            raise RneApiError("RNE response is not valid JSON") from e

    assert last_exc is not None
    raise RneApiError(f"RNE request failed after {max_retries} attempts: {last_exc}")


@dataclass
class RneClient:
    """Low-level RNE API client with token caching."""

    config: PipelineConfig
    logger: logging.Logger = LOGGER

    def __post_init__(self) -> None:
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._max_retries = max(1, int(getattr(self.config, "llm_max_retries", 3) or 3))

    # ------------------------------------------------------------------ Auth
    def _get_token(self) -> str:
        now = time.time()
        if self._token and now < self._expires_at:
            return self._token

        # Primary: username/password login (doc v4.0 June 2025)
        login_url = getattr(self.config, "rne_login_url", "").rstrip("/")
        username = getattr(self.config, "rne_username", "") or getattr(self.config, "rne_client_id", "")
        password = getattr(self.config, "rne_password", "") or getattr(self.config, "rne_client_secret", "")
        token = None
        expires_in = 3600.0

        def _auth_via_login() -> str | None:
            if not login_url or not username or not password:
                return None
            self.logger.info("RNE auth (login): username=%s", _mask_key(username))
            data = {"username": username, "password": password}
            # INPI login endpoint requires Origin/Referer headers (per doc v4.0)
            extra_headers = {}
            if "registre-national-entreprises" in login_url:
                base = login_url.split("/api/")[0]
                extra_headers = {
                    "Origin": base,
                    "Referer": f"{base}/",
                    "Accept": "application/json",
                }

            payload = _http_post_json(
                login_url,
                data,
                timeout=float(getattr(self.config, "llm_connect_timeout_sec", 5.0) or 5.0),
                max_retries=self._max_retries,
                logger=self.logger,
                extra_headers=extra_headers,
            )
            tok = payload.get("token") or payload.get("access_token")
            nonlocal expires_in
            expires_in = float(payload.get("expires_in", 3600.0))
            return tok

        def _auth_via_oauth() -> str | None:
            token_url = getattr(self.config, "rne_token_url", "").rstrip("/")
            client_id = getattr(self.config, "rne_client_id", "")
            client_secret = getattr(self.config, "rne_client_secret", "")
            if not token_url or not client_id or not client_secret:
                return None
            self.logger.info("RNE auth (oauth2 client_credentials): client_id=%s", _mask_key(client_id))
            data = {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            }
            payload = _http_post_json(
                token_url,
                data,
                timeout=float(getattr(self.config, "llm_connect_timeout_sec", 5.0) or 5.0),
                max_retries=self._max_retries,
                logger=self.logger,
            )
            nonlocal expires_in
            expires_in = float(payload.get("expires_in", 3600.0))
            return payload.get("access_token") or payload.get("token")

        # Try login first, then fallback to legacy oauth2 if needed.
        token = None
        try:
            token = _auth_via_login()
        except RneApiError as exc:
            self.logger.warning("RNE login auth failed (%s), trying legacy oauth2", exc)

        if not token:
            token = _auth_via_oauth()

        if not token:
            raise RneApiError("RNE auth response missing token")

        # Margin of 60 seconds to avoid edge expiration.
        self._token = token
        self._expires_at = now + max(60.0, expires_in - 60.0)
        self.logger.debug("RNE token refreshed, expires_in=%s", int(expires_in))
        return token

    # ------------------------------------------------------------------ Search
    def search_companies(
        self,
        name: str,
        postcode: str | None,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Search companies by name and optional postcode, handling pagination."""

        base = getattr(self.config, "rne_api_url", "").rstrip("/")
        if not base:
            raise RneApiError("rne_api_url is not configured")
        # Accept both the root API URL (".../api/") and a direct companies endpoint.
        if base.endswith("/companies"):
            endpoint = base
        else:
            endpoint = urljoin(base + "/", "companies")
        page_size = min(max_results, 100)
        collected: list[Dict[str, Any]] = []
        page = 1

        while len(collected) < max_results:
            params = {
                "companyName": name,
                "page": page,
                "pageSize": page_size,
            }
            if postcode:
                params["zipCodes[]"] = postcode

            query = urlencode(params, doseq=True)
            url = f"{endpoint}?{query}"
            token = self._get_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": "Sireto-PipeV6/2.2",
            }
            if "registre-national-entreprises" in base:
                origin = base.split("/api")[0]
                headers["Origin"] = origin
                headers["Referer"] = f"{origin}/"

            try:
                payload = _http_get_json(
                    url,
                    headers,
                    max_retries=self._max_retries,
                    logger=self.logger,
                )
            except RneApiError as exc:
                # If token expired mid-flight, force refresh once.
                if "401" in str(exc):
                    self._token = None
                    token = self._get_token()
                    headers["Authorization"] = f"Bearer {token}"
                    payload = _http_get_json(
                        url,
                        headers,
                        max_retries=self._max_retries,
                        logger=self.logger,
                    )
                else:
                    raise

            results = payload.get("results")
            if results is None:
                raise RneApiError("Invalid RNE response: missing 'results' key")

            if not isinstance(results, list):
                raise RneApiError("Invalid RNE response: 'results' is not a list")

            if not results and page == 1:
                self.logger.info(
                    "RNE search: name=%s postcode=%s -> 0 companies",
                    name,
                    postcode,
                )
                break

            collected.extend(results)

            self.logger.debug(
                "RNE page=%s size=%s -> %s companies (total collected=%s)",
                page,
                page_size,
                len(results),
                len(collected),
            )

            if len(results) < page_size:
                break  # last page

            if len(collected) >= max_results:
                break

            page += 1

        kept = collected[:max_results]
        self.logger.info(
            "RNE search: name=%s postcode=%s -> companies=%s kept=%s",
            name,
            postcode,
            len(collected),
            len(kept),
        )
        return kept


# --------------------------------------------------------------------------- Mapping helpers

def _pick_person(data: Dict[str, Any]) -> Dict[str, Any] | None:
    content = (
        data.get("formality", {})
        .get("content", {})
    )
    if "personneMorale" in content:
        return content.get("personneMorale") or None
    if "personnePhysique" in content:
        return content.get("personnePhysique") or None
    return None


def _extract_label(person: Dict[str, Any]) -> str | None:
    return (
        person.get("denomination")
        or person.get("nomCommercial")
        or person.get("sigle")
        or person.get("nom")
        or None
    )


def _map_company_to_candidates(
    company: Dict[str, Any],
    *,
    query: str,
    rank_offset: int = 0,
    store_raw: bool = True,
) -> List[RawCandidate]:
    """Extract RawCandidate objects (siège + établissements) from a company blob."""

    person = _pick_person(company)
    if not person:
        return []

    siren = person.get("siren")
    label = _extract_label(person)
    forme_juridique = person.get("formeJuridique")
    etat = person.get("etatEntreprise")
    date_immat = person.get("dateImmatriculation")

    extras_base = {
        "query": query,
        "forme_juridique": forme_juridique,
        "etat": etat,
        "date_immatriculation": date_immat,
    }
    if store_raw:
        extras_base["raw"] = company

    candidates: list[RawCandidate] = []
    rank = rank_offset

    def _append_candidate(siret: str | None, extra_override: dict[str, Any] | None = None) -> None:
        nonlocal rank
        if not siret and not siren:
            return  # skip entries without any identifier
        rank += 1
        extra = dict(extras_base)
        extra["rank"] = rank
        if extra_override:
            extra.update(extra_override)
        candidates.append(
            create_raw_candidate(
                source="RNE",
                siren=siren,
                siret=siret,
                label=label,
                url=None,
                extra=extra,
            )
        )

    siege = person.get("etablissementSiege") or {}
    _append_candidate(siege.get("siret"))

    autres = person.get("autresEtablissements") or []
    if isinstance(autres, list):
        for etab in autres:
            _append_candidate(etab.get("siret"))

    return candidates


__all__ = [
    "RneApiError",
    "RneClient",
    "_map_company_to_candidates",
]

# Backward compatibility for tests that still patch the old helper name.
_http_post_form = _http_post_json  # type: ignore
