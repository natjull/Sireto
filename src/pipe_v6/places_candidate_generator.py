"""Places-guided candidate generation for V7 pipeline.

Implements two strategies for generating SIRENE candidates from Places results:
- Arm A: Candidates by address (SQL + Python filtering)
- Arm B: Candidates by radius (bounding box + haversine distance)

The combined pool is then rescored by XGBoost using Places-normalized data.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from typing import Any, List, Set, Tuple

from .config import PipelineConfig
from .serper_places_client import PlacesResult

LOGGER = logging.getLogger(__name__)

# Geo enrichment guard (per process)
_GEO_ENRICH_DONE: bool = False

# Approximate degrees for bounding box (~200m at 45° latitude)
# 1° latitude ≈ 111km, 1° longitude ≈ 78km at 45°N
DEGREES_PER_METER_LAT = 1 / 111000
DEGREES_PER_METER_LON = 1 / 78000  # approximate for France


@dataclass
class SireneCandidate:
    """A candidate establishment from SIRENE cache."""

    siret: str
    siren: str
    denomination: str | None
    denomination_ci: str | None
    enseigne1: str | None
    enseigne2: str | None
    enseigne3: str | None
    denomination_unite_legale: str | None
    nom_unite_legale: str | None
    prenom1_unite_legale: str | None
    street_number: str | None
    street_type: str | None
    street_name: str | None
    address_full: str | None
    postcode: str | None
    city: str | None
    insee_code: str | None
    geo_latitude: float | None
    geo_longitude: float | None
    etat_administratif: str | None
    etablissement_siege: int | None
    activite_principale: str | None
    legal_nature: str | None
    # Metadata
    source: str = ""  # "arm_a", "arm_b", "topk"
    distance_m: float | None = None
    raw: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Best available name for display."""
        return (
            self.denomination
            or self.enseigne1
            or self.denomination_unite_legale
            or self.siren
        )

    @property
    def full_address(self) -> str:
        """Concatenated address string."""
        parts = [
            self.street_number,
            self.street_type,
            self.street_name,
            self.postcode,
            self.city,
        ]
        return " ".join(p for p in parts if p)


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate distance between two points in meters using Haversine formula."""
    R = 6371000  # Earth radius in meters
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _parse_street_number_range(street_number: str | None) -> tuple[int | None, int | None]:
    """Parse a street number string into a numeric range.

    Handles formats like:
      - "15"
      - "15B"
      - "15 BIS"
      - "15-17"
      - "107-115"

    Returns (min, max) or (None, None) if parsing fails.
    """
    if not street_number:
        return None, None
    import re

    text = str(street_number).strip().upper()
    if not text:
        return None, None

    # Normalize separators
    text = re.sub(r"\s*[-–]\s*", "-", text)
    text = re.sub(r"\s+", " ", text)

    # Range: 15-17
    m = re.match(r"^(\d+)\s*-\s*(\d+)", text)
    if m:
        try:
            start = int(m.group(1))
            end = int(m.group(2))
            return (min(start, end), max(start, end))
        except ValueError:
            return None, None

    # Single number (optionally with suffix)
    m = re.match(r"^(\d+)", text)
    if m:
        try:
            base = int(m.group(1))
            return base, base
        except ValueError:
            return None, None

    return None, None


def _ensure_geo_coords(
    conn: sqlite3.Connection,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
) -> None:
    """Populate geo_latitude/geo_longitude in the SIRENE cache from harvest_full.sqlite.

    This runs once per process and only if the cache currently has no coordinates.
    """
    global _GEO_ENRICH_DONE
    if _GEO_ENRICH_DONE:
        return
    log = logger or LOGGER

    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM establishments WHERE geo_latitude IS NOT NULL AND geo_longitude IS NOT NULL"
        )
        with_coords = cur.fetchone()[0]
    except Exception:
        with_coords = 0

    if with_coords > 0:
        _GEO_ENRICH_DONE = True
        return

    if not getattr(config, "places_geo_enrich_enabled", True):
        log.info("Geo enrichment disabled; skipping Arm B.")
        _GEO_ENRICH_DONE = True
        return

    harvest_path = getattr(config, "harvest_db_path", None)
    if not harvest_path or not Path(harvest_path).exists():
        log.warning("Harvest DB not found; Arm B disabled (no coords).")
        _GEO_ENRICH_DONE = True
        return

    log.info("Enriching SIRENE cache with geo coords from %s...", harvest_path)
    conn.execute(f"ATTACH DATABASE '{harvest_path}' AS harvest")
    try:
        conn.execute(
            """
            UPDATE establishments
            SET geo_latitude = (
                    SELECT CAST(NULLIF(latitude, '') AS REAL)
                    FROM harvest.etablissements h
                    WHERE h.siret = establishments.siret
                ),
                geo_longitude = (
                    SELECT CAST(NULLIF(longitude, '') AS REAL)
                    FROM harvest.etablissements h
                    WHERE h.siret = establishments.siret
                )
            WHERE geo_latitude IS NULL OR geo_longitude IS NULL
            """
        )
        # Optional indexes for faster geo queries (small DB but harmless)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_establishments_geo_lat ON establishments(geo_latitude)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_establishments_geo_lon ON establishments(geo_longitude)"
        )
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE harvest")
        conn.commit()

    _GEO_ENRICH_DONE = True
    log.info("Geo enrichment completed.")


def _row_to_candidate(row: sqlite3.Row, source: str) -> SireneCandidate:
    """Convert a SQLite row to SireneCandidate."""
    return SireneCandidate(
        siret=row["siret"],
        siren=row["siren"],
        denomination=row["denomination"],
        denomination_ci=row["denomination_ci"],
        enseigne1=row["enseigne1"],
        enseigne2=row["enseigne2"],
        enseigne3=row["enseigne3"],
        denomination_unite_legale=row["denomination_unite_legale"],
        nom_unite_legale=row["nom_unite_legale"],
        prenom1_unite_legale=row["prenom1_unite_legale"],
        street_number=row["street_number"],
        street_type=row["street_type"],
        street_name=row["street_name"],
        address_full=row["address_full"],
        postcode=row["postcode"],
        city=row["city"],
        insee_code=row["insee_code"],
        geo_latitude=row["geo_latitude"],
        geo_longitude=row["geo_longitude"],
        etat_administratif=row["etat_administratif"],
        etablissement_siege=row["etablissement_siege"],
        activite_principale=row["activite_principale"],
        legal_nature=row["legal_nature"],
        source=source,
        raw=dict(row),
    )


def generate_candidates_by_address(
    places_result: PlacesResult,
    conn: sqlite3.Connection,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
    *,
    street_name_jaro_min: float = 0.70,
) -> List[SireneCandidate]:
    """Arm A: Generate candidates by address matching.

    Strategy:
    1. SQL: cheap query on (insee_code OR postcode) + street_number range
    2. Python: filter by street_name Jaro-Winkler similarity

    Args:
        places_result: Parsed Places API result with address components
        conn: SQLite connection to sirene_cache
        config: Pipeline configuration
        logger: Optional logger
        street_name_jaro_min: Minimum Jaro-Winkler similarity for street name

    Returns:
        List of matching SireneCandidate objects
    """
    log = logger or LOGGER

    postcode = places_result.postcode
    street_num_str = places_result.street_number
    street_name = places_result.street_name

    if not postcode:
        log.debug("Arm A: No postcode in Places result, skipping")
        return []

    # Parse street number for range query
    num_min, num_max = _parse_street_number_range(street_num_str)
    if num_min is not None and num_max is not None:
        num_min = max(0, num_min - 2)
        num_max = num_max + 2

    # Build SQL query
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Optional: narrow down by a distinctive street token (e.g., ZOLA)
    street_like = None
    if street_name:
        norm = _normalize_street(street_name)
        stop_tokens = {
            "RUE",
            "AVENUE",
            "BOULEVARD",
            "PLACE",
            "ALLEE",
            "IMPASSE",
            "CHEMIN",
            "ROUTE",
            "SQUARE",
            "QUAI",
        }
        tokens = [t for t in norm.split() if t not in stop_tokens and len(t) >= 3]
        if tokens:
            street_like = max(tokens, key=len)

    if num_min is not None and num_max is not None:
        # Query with street number range
        query = """
            SELECT siret, siren, denomination, denomination_ci,
                   enseigne1, enseigne2, enseigne3,
                   denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
                   street_number, street_type, street_name, address_full, postcode, city, insee_code,
                   geo_latitude, geo_longitude, etat_administratif, etablissement_siege,
                   activite_principale, legal_nature
            FROM establishments
            WHERE postcode = ?
              AND street_number IS NOT NULL
              AND street_number GLOB '[0-9][0-9]*'
              AND CAST(street_number AS INTEGER) BETWEEN ? AND ?
        """
        params = [postcode, num_min, num_max]
        if street_like:
            query += " AND street_name LIKE ?"
            params.append(f"%{street_like}%")
        cursor.execute(query, params)
    else:
        # Query without street number constraint (broader)
        query = """
            SELECT siret, siren, denomination, denomination_ci,
                   enseigne1, enseigne2, enseigne3,
                   denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
                   street_number, street_type, street_name, address_full, postcode, city, insee_code,
                   geo_latitude, geo_longitude, etat_administratif, etablissement_siege,
                   activite_principale, legal_nature
            FROM establishments
            WHERE postcode = ?
            LIMIT ?
        """
        max_rows = int(getattr(config, "places_arm_a_max_rows", 5000))
        params = [postcode]
        if street_like:
            query = query.replace("WHERE postcode = ?", "WHERE postcode = ? AND street_name LIKE ?")
            params.append(f"%{street_like}%")
        params.append(max_rows)
        cursor.execute(query, params)

    rows = cursor.fetchall()
    cursor.close()

    log.debug(
        "Arm A SQL: postcode=%s num_range=%s-%s -> %d rows",
        postcode,
        num_min,
        num_max,
        len(rows),
    )

    if not rows:
        return []

    # Python filter: street name similarity
    candidates: List[SireneCandidate] = []
    places_street_norm = _normalize_street(street_name) if street_name else ""
    street_jaro_min = getattr(config, "places_arm_a_street_jaro_min", street_name_jaro_min)

    for row in rows:
        candidate = _row_to_candidate(row, source="arm_a")

        # If we have street names, check similarity
        if places_street_norm and (candidate.street_name or candidate.street_type):
            cand_street = " ".join(p for p in [candidate.street_type, candidate.street_name] if p)
            cand_street_norm = _normalize_street(cand_street)
            similarity = jaro_winkler(places_street_norm, cand_street_norm)
            if similarity < street_jaro_min:
                continue

        candidates.append(candidate)

    log.info(
        "Arm A: postcode=%s street='%s' -> %d candidates after filtering",
        postcode,
        street_name,
        len(candidates),
    )

    return candidates


def generate_candidates_by_radius(
    places_result: PlacesResult,
    conn: sqlite3.Connection,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
    *,
    radius_m: float | None = None,
) -> List[SireneCandidate]:
    """Arm B: Generate candidates within a radius of Places location.

    Strategy:
    1. SQL: bounding box query on geo_latitude/geo_longitude
    2. Python: exact haversine distance filter

    Args:
        places_result: Parsed Places API result with lat/lon
        conn: SQLite connection to sirene_cache
        config: Pipeline configuration
        logger: Optional logger
        radius_m: Search radius in meters (default from config)

    Returns:
        List of matching SireneCandidate objects with distance_m populated
    """
    log = logger or LOGGER

    lat = places_result.latitude
    lon = places_result.longitude

    if lat is None or lon is None:
        log.debug("Arm B: No coordinates in Places result, skipping")
        return []

    # Ensure geo coords exist in cache (one-time enrichment)
    _ensure_geo_coords(conn, config, logger=log)

    max_distance = radius_m if radius_m is not None else config.places_distance_max_m

    # Calculate bounding box
    lat_delta = max_distance * DEGREES_PER_METER_LAT
    lon_delta = max_distance * DEGREES_PER_METER_LON

    lat_min = lat - lat_delta
    lat_max = lat + lat_delta
    lon_min = lon - lon_delta
    lon_max = lon + lon_delta

    # SQL query with bounding box
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT siret, siren, denomination, denomination_ci,
               enseigne1, enseigne2, enseigne3,
               denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
               street_number, street_type, street_name, address_full, postcode, city, insee_code,
               geo_latitude, geo_longitude, etat_administratif, etablissement_siege,
               activite_principale, legal_nature
        FROM establishments
        WHERE geo_latitude BETWEEN ? AND ?
          AND geo_longitude BETWEEN ? AND ?
          AND geo_latitude IS NOT NULL
          AND geo_longitude IS NOT NULL
    """
    cursor.execute(query, (lat_min, lat_max, lon_min, lon_max))
    rows = cursor.fetchall()
    cursor.close()

    log.debug(
        "Arm B SQL: lat=%.4f lon=%.4f bbox=%.4f -> %d rows",
        lat,
        lon,
        lat_delta,
        len(rows),
    )

    if not rows:
        return []

    # Python filter: exact haversine distance
    candidates: List[SireneCandidate] = []

    for row in rows:
        cand_lat = row["geo_latitude"]
        cand_lon = row["geo_longitude"]

        if cand_lat is None or cand_lon is None:
            continue

        distance = haversine_distance(lat, lon, cand_lat, cand_lon)
        if distance > max_distance:
            continue

        candidate = _row_to_candidate(row, source="arm_b")
        candidate.distance_m = distance
        candidates.append(candidate)

    log.info(
        "Arm B: lat=%.4f lon=%.4f radius=%.0fm -> %d candidates",
        lat,
        lon,
        max_distance,
        len(candidates),
    )

    return candidates


def merge_candidate_pools(
    arm_a: List[SireneCandidate],
    arm_b: List[SireneCandidate],
    topk: List[SireneCandidate],
    logger: logging.Logger | None = None,
) -> List[SireneCandidate]:
    """Merge and deduplicate candidates from all sources.

    Priority order for duplicates: arm_a > arm_b > topk
    (preserves source attribution for candidates found by address first)

    Args:
        arm_a: Candidates from address matching
        arm_b: Candidates from radius search
        topk: Original XGB top-k candidates

    Returns:
        Deduplicated list of candidates
    """
    log = logger or LOGGER
    seen_sirets: Set[str] = set()
    merged: List[SireneCandidate] = []

    # Process in priority order
    for source_name, candidates in [("arm_a", arm_a), ("arm_b", arm_b), ("topk", topk)]:
        for cand in candidates:
            if cand.siret not in seen_sirets:
                seen_sirets.add(cand.siret)
                merged.append(cand)

    log.info(
        "Merged pools: arm_a=%d, arm_b=%d, topk=%d -> total=%d (deduped)",
        len(arm_a),
        len(arm_b),
        len(topk),
        len(merged),
    )

    return merged


def _normalize_street(name: str) -> str:
    """Normalize street name for comparison."""
    import unicodedata

    if not name:
        return ""

    # Remove accents
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Uppercase and normalize whitespace
    text = text.upper()
    text = " ".join(text.split())

    # Common abbreviation expansions
    abbreviations = {
        "AV.": "AVENUE",
        "AV": "AVENUE",
        "BD": "BOULEVARD",
        "BD.": "BOULEVARD",
        "RUE": "RUE",
        "R.": "RUE",
        "PL.": "PLACE",
        "PL": "PLACE",
        "ALL.": "ALLEE",
        "ALL": "ALLEE",
        "IMP.": "IMPASSE",
        "IMP": "IMPASSE",
        "CHE.": "CHEMIN",
        "CHE": "CHEMIN",
    }

    words = text.split()
    normalized_words = [abbreviations.get(w, w) for w in words]
    return " ".join(normalized_words)


def jaro_winkler(s1: str, s2: str) -> float:
    """Calculate Jaro-Winkler similarity between two strings.

    Returns a value between 0 (no similarity) and 1 (identical).
    """
    if not s1 or not s2:
        return 0.0

    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    # Find matches
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1 + matches / len2 + (matches - transpositions / 2) / matches
    ) / 3

    # Winkler modification: boost for common prefix
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


def count_candidates_at_address(
    conn: sqlite3.Connection,
    postcode: str,
    street_name: str | None,
    street_number: str | None,
) -> int:
    """Count establishments at a specific address (for multi-tenant detection)."""
    cursor = conn.cursor()

    if street_number and street_name:
        # Strict match
        cursor.execute(
            """
            SELECT COUNT(*) FROM establishments
            WHERE postcode = ?
              AND street_number = ?
              AND street_name LIKE ?
            """,
            (postcode, street_number, f"%{street_name}%"),
        )
    elif street_name:
        cursor.execute(
            """
            SELECT COUNT(*) FROM establishments
            WHERE postcode = ?
              AND street_name LIKE ?
            """,
            (postcode, f"%{street_name}%"),
        )
    else:
        cursor.execute(
            "SELECT COUNT(*) FROM establishments WHERE postcode = ?",
            (postcode,),
        )

    count = cursor.fetchone()[0]
    cursor.close()
    return count


__all__ = [
    "SireneCandidate",
    "haversine_distance",
    "generate_candidates_by_address",
    "generate_candidates_by_radius",
    "merge_candidate_pools",
    "jaro_winkler",
    "count_candidates_at_address",
]
