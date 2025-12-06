#!/usr/bin/env python3
"""
Integrity check and completion script for harvest_v2.

This script:
1. Loads all SIRENs from StockUniteLegale parquet (source of truth)
2. Compares with SIRENs in harvest_v2.sqlite
3. Identifies missing SIRENs
4. Fetches missing SIRENs via API and completes the database

Run this AFTER the main harvest_v2 script has finished.
"""

from __future__ import annotations

import sqlite3
import time
import logging
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm

# --------------------------- Logging --------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# --------------------------- Config ---------------------------------------- #
BASE_URL = "https://recherche-entreprises.api.gouv.fr/search"
DB_PATH = Path("data/harvest_v2.sqlite")
PARQUET_PATH = Path("data/StockUniteLegale_utf8.parquet")
BATCH_SIZE = 25  # SIRENs per API call (q parameter batching)
MAX_RETRIES = 4
BACKOFF_BASE = 0.5
REQUEST_INTERVAL = 1.0 / 7.0  # 7 req/s


# --------------------------- Rate Limiter ---------------------------------- #
class RateLimiter:
    def __init__(self, interval: float):
        self.interval = interval
        self.last_request = 0.0

    def acquire(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request = time.monotonic()


# --------------------------- DB helpers ------------------------------------ #
def get_harvested_sirens(conn: sqlite3.Connection) -> Set[str]:
    """Get all SIRENs already in the harvest database."""
    cur = conn.execute("SELECT DISTINCT siren FROM entreprises")
    return {row[0] for row in cur.fetchall()}


def upsert_entreprise(conn: sqlite3.Connection, ent: dict) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO entreprises
        (siren, nom_complet, nature_juridique, activite_principale, section_activite_principale,
         categorie_entreprise, tranche_effectif, etat_administratif, date_creation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ent.get("siren"),
        ent.get("nom_complet"),
        ent.get("nature_juridique"),
        ent.get("activite_principale"),
        ent.get("section_activite_principale"),
        ent.get("categorie_entreprise"),
        ent.get("tranche_effectif_salarie"),
        ent.get("etat_administratif"),
        ent.get("date_creation"),
    ))


def upsert_dirigeants(conn: sqlite3.Connection, siren_ent: str, dirigeants: list) -> int:
    count = 0
    for d in dirigeants or []:
        type_dir = d.get("type_dirigeant", "")
        try:
            if type_dir == "personne morale":
                conn.execute("""
                    INSERT OR IGNORE INTO dirigeants
                    (siren_entreprise, type_dirigeant, siren_dirigeant, denomination, qualite)
                    VALUES (?, ?, ?, ?, ?)
                """, (siren_ent, type_dir, d.get("siren"), d.get("denomination"), d.get("qualite")))
            else:
                conn.execute("""
                    INSERT OR IGNORE INTO dirigeants
                    (siren_entreprise, type_dirigeant, nom, prenoms, annee_naissance, nationalite, qualite)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (siren_ent, type_dir, d.get("nom"), d.get("prenoms"),
                      d.get("annee_de_naissance"), d.get("nationalite"), d.get("qualite")))
            count += 1
        except sqlite3.IntegrityError:
            pass
    return count


# --------------------------- API ------------------------------------------- #
def fetch_batch(session: requests.Session, limiter: RateLimiter, sirens: List[str]) -> Optional[dict]:
    """Fetch a batch of SIRENs using the q parameter."""
    q = " ".join(sirens)
    
    for attempt in range(MAX_RETRIES):
        limiter.acquire()
        try:
            params = {
                "q": q,
                "per_page": BATCH_SIZE,
                "page": 1,
                "minimal": True,
                "include": "dirigeants",
            }
            resp = session.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 429:
                delay = float(resp.headers.get("Retry-After", 3.0))
                time.sleep(delay)
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(BACKOFF_BASE * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                LOGGER.warning("Failed batch: %s", e)
                return None
            time.sleep(BACKOFF_BASE * (2 ** attempt))
    return None


# --------------------------- Main ------------------------------------------ #
def main() -> None:
    # Check files exist
    if not DB_PATH.exists():
        LOGGER.error("Database not found: %s", DB_PATH)
        return
    if not PARQUET_PATH.exists():
        LOGGER.error("Parquet not found: %s", PARQUET_PATH)
        return
    
    # Load source SIRENs
    LOGGER.info("Loading source SIRENs from parquet...")
    df = pd.read_parquet(PARQUET_PATH, columns=["siren"])
    source_sirens = set(df["siren"].dropna().drop_duplicates().astype(str).tolist())
    LOGGER.info("Source SIRENs: %d", len(source_sirens))
    
    # Load harvested SIRENs
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    LOGGER.info("Loading harvested SIRENs from database...")
    harvested_sirens = get_harvested_sirens(conn)
    LOGGER.info("Harvested SIRENs: %d", len(harvested_sirens))
    
    # Find missing
    missing_sirens = source_sirens - harvested_sirens
    LOGGER.info("Missing SIRENs: %d (%.2f%%)", 
                len(missing_sirens), 
                100 * len(missing_sirens) / len(source_sirens) if source_sirens else 0)
    
    if not missing_sirens:
        LOGGER.info("✅ All SIRENs are present! No completion needed.")
        conn.close()
        return
    
    # Fetch missing SIRENs in batches
    missing_list = sorted(missing_sirens)
    total_batches = (len(missing_list) + BATCH_SIZE - 1) // BATCH_SIZE
    
    LOGGER.info("Fetching %d missing SIRENs in %d batches...", len(missing_list), total_batches)
    
    limiter = RateLimiter(REQUEST_INTERVAL)
    session = requests.Session()
    session.headers.update({
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "Sireto-IntegrityCheck/1.0",
    })
    
    ent_count = 0
    dir_count = 0
    found_sirens = set()
    
    pbar = tqdm(range(0, len(missing_list), BATCH_SIZE), desc="Batches", unit="batch")
    for i in pbar:
        batch = missing_list[i:i + BATCH_SIZE]
        payload = fetch_batch(session, limiter, batch)
        
        if payload:
            for res in payload.get("results", []) or []:
                siren = res.get("siren")
                if siren:
                    upsert_entreprise(conn, res)
                    dir_count += upsert_dirigeants(conn, siren, res.get("dirigeants"))
                    ent_count += 1
                    found_sirens.add(siren)
            conn.commit()
        
        pbar.set_postfix({"Found": ent_count, "Dir": dir_count})
    
    pbar.close()
    
    # Final report
    not_found = missing_sirens - found_sirens
    LOGGER.info("=" * 60)
    LOGGER.info("Completion finished!")
    LOGGER.info("  Entreprises added: %d", ent_count)
    LOGGER.info("  Dirigeants added: %d", dir_count)
    LOGGER.info("  Still not found: %d (may be non-diffusible or inactive)", len(not_found))
    
    if not_found:
        LOGGER.info("  Not found SIRENs saved to: data/not_found_sirens.txt")
        with open("data/not_found_sirens.txt", "w") as f:
            for s in sorted(not_found):
                f.write(s + "\n")
    
    conn.close()


if __name__ == "__main__":
    main()
