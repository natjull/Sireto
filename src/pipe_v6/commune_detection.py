"""Commune detection logic (task 1.2)."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List
import logging
import unicodedata

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommuneKey:
    """Minimal information required to fetch SIRENE data for a commune."""

    insee_code: str | None
    postcode: str | None
    city: str | None


def extract_communes(df: pd.DataFrame) -> List[CommuneKey]:
    """Return a deduplicated list of communes to pre-fetch from SIRENE."""

    if df.empty:
        return []

    working_df = df.copy()
    working_df["city_norm"] = working_df["city"].map(_normalize_city)

    insee_counts = defaultdict(Counter)
    combo_first_seen: dict[tuple[str, tuple[str | None, str | None]], int] = {}

    for idx, row in working_df.iterrows():
        insee = _safe_str(row.get("insee_code"))
        if not insee:
            continue
        combo = (_safe_str(row.get("postcode")), row.get("city_norm"))
        insee_counts[insee][combo] += 1
        combo_first_seen.setdefault((insee, combo), idx)

    insee_representative: dict[str, tuple[str | None, str | None]] = {}
    for insee, counts in insee_counts.items():
        best_combo = min(
            counts.items(),
            key=lambda item: (-item[1], combo_first_seen[(insee, item[0])]),
        )[0]
        insee_representative[insee] = best_combo

    keys: "OrderedDict[CommuneKey, None]" = OrderedDict()
    stats = Counter()

    for _, row in working_df.iterrows():
        key: CommuneKey | None = None
        category: str | None = None
        insee = _safe_str(row.get("insee_code"))

        if insee:
            postcode_mode, city_mode = insee_representative.get(insee, (None, None))
            key = CommuneKey(insee_code=insee, postcode=postcode_mode, city=city_mode)
            category = "insee"
        else:
            postcode = _safe_str(row.get("postcode"))
            if postcode:
                key = CommuneKey(insee_code=None, postcode=postcode, city=None)
                category = "postcode"
            else:
                city_norm = row.get("city_norm")
                if city_norm:
                    key = CommuneKey(insee_code=None, postcode=None, city=city_norm)
                    category = "city"
                else:
                    crm_id = row.get("crm_id", "<unknown>")
                    logger.warning(
                        "CRM entry %s has no commune information; skipping", crm_id
                    )

        if key and key not in keys:
            keys[key] = None
            if category:
                stats[category] += 1

    logger.info(
        "Commune extraction: %s INSEE, %s postcode-only, %s city-only",
        stats.get("insee", 0),
        stats.get("postcode", 0),
        stats.get("city", 0),
    )

    return list(keys.keys())


def _safe_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize_city(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper().replace("-", " ")
    text = " ".join(text.split())
    return text or None


__all__ = ["CommuneKey", "extract_communes"]
