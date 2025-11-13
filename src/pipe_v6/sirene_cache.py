"""SQLite cache helpers for SIRENE establishments."""

from __future__ import annotations

import json
import logging
import sqlite3
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from .config import PipelineConfig
from .commune_detection import CommuneKey
from . import sirene_client

LOGGER = logging.getLogger(__name__)

_PRAGMAS: tuple[tuple[str, str], ...] = (
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("temp_store", "MEMORY"),
    ("foreign_keys", "ON"),
)


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply a default set of PRAGMA statements for better cache performance."""

    cursor = conn.cursor()
    for pragma, value in _PRAGMAS:
        cursor.execute(f"PRAGMA {pragma}={value};")
    cursor.close()


def _ensure_directory(path: Path) -> None:
    """Ensure parent directory exists before touching the SQLite file."""

    path.parent.mkdir(parents=True, exist_ok=True)


def init_db(path: Path, *, store_source_raw: bool = True) -> sqlite3.Connection:
    """Create (or open) the cache database and guarantee the schema is present."""

    resolved = Path(path).expanduser().resolve()
    _ensure_directory(resolved)

    conn = sqlite3.connect(resolved)
    _apply_pragmas(conn)
    _ensure_schema(conn)

    if not store_source_raw:
        LOGGER.debug("source_raw column present but will remain unused per configuration")

    return conn


def get_cache_connection(
    config: PipelineConfig, *, initialize: bool = True
) -> sqlite3.Connection:
    """Return a ready-to-use connection based on the pipeline configuration."""

    if initialize:
        return init_db(config.sqlite_path, store_source_raw=config.store_source_raw)

    resolved = config.sqlite_path.expanduser().resolve()
    conn = sqlite3.connect(resolved)
    _apply_pragmas(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the schema for establishments and cache status if missing."""

    cursor = conn.cursor()
    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS establishments (
            siret TEXT PRIMARY KEY,
            siren TEXT NOT NULL,
            nic TEXT NOT NULL,
            etablissement_siege INTEGER NOT NULL DEFAULT 0,
            denomination TEXT,
            denomination_ci TEXT,
            denomination_unite_legale TEXT,
            nom_unite_legale TEXT,
            prenom1_unite_legale TEXT,
            enseigne1 TEXT,
            enseigne2 TEXT,
            enseigne3 TEXT,
            street_number TEXT,
            street_type TEXT,
            street_name TEXT,
            address_full TEXT,
            postcode TEXT,
            city TEXT,
            commune_libelle_raw TEXT,
            insee_code TEXT,
            geo_latitude REAL,
            geo_longitude REAL,
            etat_administratif TEXT,
            date_debut TEXT,
            date_dernier_traitement TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            activite_principale TEXT,
            nomenclature_activite TEXT,
            tranche_effectifs TEXT,
            annee_effectifs TEXT,
            legal_nature TEXT,
            source_raw TEXT
        );

        CREATE TABLE IF NOT EXISTS commune_cache_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insee_code TEXT,
            postcode TEXT,
            city TEXT,
            fetch_filter TEXT NOT NULL,
            total_results INTEGER NOT NULL,
            fetched_at TEXT NOT NULL,
            status TEXT NOT NULL,
            api_version TEXT,
            UNIQUE(insee_code, postcode, city)
        );
        """
    )

    _create_indexes(cursor)
    conn.commit()
    cursor.close()


def _create_indexes(cursor: sqlite3.Cursor) -> None:
    """Create indexes that speed up candidate enrichment queries."""

    cursor.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_establishments_siren
            ON establishments(siren);

        CREATE INDEX IF NOT EXISTS idx_establishments_insee
            ON establishments(insee_code);

        CREATE INDEX IF NOT EXISTS idx_establishments_postcode_city
            ON establishments(postcode, city);

        CREATE INDEX IF NOT EXISTS idx_establishments_denomination_ci
            ON establishments(denomination_ci COLLATE NOCASE);

        CREATE INDEX IF NOT EXISTS idx_commune_cache_status_insee
            ON commune_cache_status(insee_code);

        CREATE INDEX IF NOT EXISTS idx_commune_cache_status_postcode_city
            ON commune_cache_status(postcode, city);
        """
    )


_ESTABLISHMENT_COLUMNS: Tuple[str, ...] = (
    "siret",
    "siren",
    "nic",
    "etablissement_siege",
    "denomination",
    "denomination_ci",
    "denomination_unite_legale",
    "nom_unite_legale",
    "prenom1_unite_legale",
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "street_number",
    "street_type",
    "street_name",
    "address_full",
    "postcode",
    "city",
    "commune_libelle_raw",
    "insee_code",
    "geo_latitude",
    "geo_longitude",
    "etat_administratif",
    "date_debut",
    "date_dernier_traitement",
    "activite_principale",
    "nomenclature_activite",
    "tranche_effectifs",
    "annee_effectifs",
    "legal_nature",
    "source_raw",
)


def upsert_establishments(
    records: Iterable[dict],
    conn: sqlite3.Connection,
    *,
    store_source_raw: bool | None = None,
    logger: logging.Logger | None = None,
) -> int:
    """Upsert a batch of SIRENE establishments into the cache."""

    logger = logger or LOGGER
    store_raw = True if store_source_raw is None else store_source_raw

    projected: list[tuple] = []
    skipped = 0

    for record in records:
        projected_row = _project_record(record, store_raw)
        if projected_row is None:
            skipped += 1
            continue
        projected.append(projected_row)

    if not projected:
        if skipped:
            logger.debug("All records skipped during upsert (count=%s)", skipped)
        return 0

    placeholders = ",".join(["?"] * len(_ESTABLISHMENT_COLUMNS))
    sql = (
        "INSERT OR REPLACE INTO establishments ("
        + ",".join(_ESTABLISHMENT_COLUMNS)
        + ") VALUES ("
        + placeholders
        + ")"
    )

    with conn:
        conn.executemany(sql, projected)

    logger.info(
        "Upserted %s SIRENE establishments (skipped=%s)", len(projected), skipped
    )
    return len(projected)


def _project_record(record: dict, store_source_raw: bool) -> tuple | None:
    siret = _safe_str(record.get("siret"))
    if not siret or len(siret) != 14:
        return None

    siren = _safe_str(record.get("siren")) or siret[:9]
    nic = _safe_str(record.get("nic")) or siret[-5:]

    ul = record.get("uniteLegale") or {}
    addr = record.get("adresseEtablissement") or {}

    denomination_unite_legale = _safe_str(ul.get("denominationUniteLegale"))
    nom_ul = _safe_str(ul.get("nomUniteLegale"))
    prenom_ul = _safe_str(ul.get("prenom1UniteLegale"))
    enseigne1 = _safe_str(record.get("enseigne1Etablissement"))
    enseigne2 = _safe_str(record.get("enseigne2Etablissement"))
    enseigne3 = _safe_str(record.get("enseigne3Etablissement"))

    denomination = _pick_denomination(
        denomination_unite_legale,
        [enseigne1, enseigne2, enseigne3],
        nom_ul,
        prenom_ul,
    )
    denomination_ci = _normalize_ci(denomination) if denomination else None

    number = _safe_str(addr.get("numeroVoieEtablissement"))
    repetition = _safe_str(addr.get("indiceRepetitionEtablissement"))
    street_number = _compose_street_number(number, repetition)
    street_type = _safe_str(addr.get("typeVoieEtablissement"))
    street_name = _safe_str(addr.get("libelleVoieEtablissement"))
    complement = _safe_str(addr.get("complementAdresseEtablissement"))
    address_full = _compose_address(complement, street_number, street_type, street_name)

    postcode = _pad_code(_safe_str(addr.get("codePostalEtablissement")))
    commune_raw = _safe_str(addr.get("libelleCommuneEtablissement"))
    city = _normalize_ci(commune_raw)
    insee_code = _pad_code(_safe_str(addr.get("codeCommuneEtablissement")))

    etat_admin = _safe_str(record.get("etatAdministratifEtablissement"))
    date_debut = _safe_str(record.get("dateCreationEtablissement")) or _safe_str(
        record.get("dateDebut"),
    )
    date_dernier_traitement = _safe_str(
        record.get("dateDernierTraitementEtablissement")
    )

    activite_principale = _safe_str(record.get("activitePrincipaleEtablissement")) or _safe_str(
        ul.get("activitePrincipaleUniteLegale"),
    )
    nomenclature_activite = _safe_str(
        record.get("nomenclatureActivitePrincipaleEtablissement")
    ) or _safe_str(ul.get("nomenclatureActivitePrincipaleUniteLegale"))
    tranche_effectifs = _safe_str(record.get("trancheEffectifsEtablissement")) or _safe_str(
        ul.get("trancheEffectifsUniteLegale"),
    )
    annee_effectifs = _safe_str(record.get("anneeEffectifsEtablissement")) or _safe_str(
        ul.get("anneeEffectifsUniteLegale"),
    )
    legal_nature = _safe_str(ul.get("categorieJuridiqueUniteLegale"))

    etablissement_siege = 1 if record.get("etablissementSiege") else 0

    source_raw = json.dumps(record, ensure_ascii=False) if store_source_raw else None

    row = (
        siret,
        siren,
        nic,
        etablissement_siege,
        denomination,
        denomination_ci,
        denomination_unite_legale,
        nom_ul,
        prenom_ul,
        enseigne1,
        enseigne2,
        enseigne3,
        street_number,
        street_type,
        street_name,
        address_full,
        postcode,
        city,
        commune_raw,
        insee_code,
        None,
        None,
        etat_admin,
        date_debut,
        date_dernier_traitement,
        activite_principale,
        nomenclature_activite,
        tranche_effectifs,
        annee_effectifs,
        legal_nature,
        source_raw,
    )

    return row


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_denomination(
    denomination_ul: str | None,
    enseignes: Sequence[str | None],
    nom_ul: str | None,
    prenom_ul: str | None,
) -> str | None:
    if denomination_ul:
        return denomination_ul
    for enseigne in enseignes:
        if enseigne:
            return enseigne
    if nom_ul and prenom_ul:
        return f"{nom_ul} {prenom_ul}".strip()
    return nom_ul or prenom_ul


def _normalize_ci(value: str | None) -> str | None:
    if not value:
        return None
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper().replace("-", " ")
    text = " ".join(text.split())
    return text or None


def _compose_street_number(number: str | None, repetition: str | None) -> str | None:
    if number and repetition:
        return f"{number} {repetition}"
    return number or repetition


def _compose_address(
    complement: str | None,
    street_number: str | None,
    street_type: str | None,
    street_name: str | None,
) -> str | None:
    parts = [complement, street_number, street_type, street_name]
    joined = " ".join(part for part in parts if part)
    joined = " ".join(joined.split())
    return joined or None


def _pad_code(value: str | None) -> str | None:
    if value is None:
        return None
    digits = value.strip()
    if digits.isdigit() and len(digits) < 5:
        digits = digits.zfill(5)
    return digits or None


def exists_in_cache(commune: CommuneKey, conn: sqlite3.Connection) -> tuple[bool, int]:
    """Return whether establishments for this commune are already cached."""

    filters = sirene_client.resolve_filters_for_commune(commune)
    if not filters:
        return False, 0

    cursor = conn.cursor()
    if filters[0].key == "codeCommuneEtablissement":
        insee_values = [flt.value for flt in filters]
        placeholders = ",".join(["?"] * len(insee_values))
        query = f"SELECT COUNT(*) FROM establishments WHERE insee_code IN ({placeholders})"
        cursor.execute(query, insee_values)
    else:
        # postcode only case (single filter)
        cursor.execute(
            "SELECT COUNT(*) FROM establishments WHERE postcode = ?",
            (filters[0].value,),
        )
    count = int(cursor.fetchone()[0])
    cursor.close()
    return (count > 0), count


def get_or_fetch_commune(
    commune: CommuneKey,
    config: PipelineConfig,
    conn: sqlite3.Connection | None = None,
    *,
    logger: logging.Logger | None = None,
    force_download: bool = False,
) -> dict:
    """Ensure the commune data exists in cache, downloading it otherwise."""

    logger = logger or LOGGER
    close_conn = False
    if conn is None:
        conn = get_cache_connection(config)
        close_conn = True

    try:
        cached = 0
        cached_exists = False
        if not force_download:
            cached_exists, cached = exists_in_cache(commune, conn)
        if cached_exists and cached > 0 and not force_download:
            logger.info(
                "Commune cache hit: insee=%s postcode=%s count=%s",
                commune.insee_code,
                commune.postcode,
                cached,
            )
            return {
                "status": "CACHED",
                "count": cached,
                "inserted": 0,
                "filters": sirene_client.resolve_filters_for_commune(commune),
            }

        records = sirene_client.fetch_establishments_for_commune(
            commune, config, logger=logger
        )
        inserted = upsert_establishments(
            records,
            conn,
            store_source_raw=config.store_source_raw,
            logger=logger,
        )
        _, count = exists_in_cache(commune, conn)
        status = "DOWNLOADED" if count else "EMPTY"
        logger.info(
            "Commune fetch: insee=%s postcode=%s total_api=%s inserted=%s count=%s",
            commune.insee_code,
            commune.postcode,
            len(records),
            inserted,
            count,
        )
        return {
            "status": status,
            "count": count,
            "inserted": inserted,
            "filters": sirene_client.resolve_filters_for_commune(commune),
        }
    finally:
        if close_conn:
            conn.close()


__all__ = [
    "init_db",
    "get_cache_connection",
    "upsert_establishments",
    "exists_in_cache",
    "get_or_fetch_commune",
]
