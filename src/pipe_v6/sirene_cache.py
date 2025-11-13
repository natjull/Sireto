"""SQLite cache helpers for SIRENE establishments."""

from __future__ import annotations

from pathlib import Path
import logging
import sqlite3
from typing import Iterable

from .config import PipelineConfig

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


__all__ = ["init_db", "get_cache_connection"]
