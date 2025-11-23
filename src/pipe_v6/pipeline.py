"""Orchestration helpers for Pipe V6 (tasks 8.x)."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from .commune_detection import CommuneKey
from .config import PipelineConfig
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
