"""External source clients orchestrators (task 4.x).

Currently implements RNE search (task 4.2). Other sources (DataGouv, Qwant)
will be added in subsequent tasks.
"""

from __future__ import annotations

import logging
from typing import List

from .config import PipelineConfig
from .candidate_store import RawCandidate, candidate_key
from .rne_client import RneClient, _map_company_to_candidates


LOGGER = logging.getLogger(__name__)


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
