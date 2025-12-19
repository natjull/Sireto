#!/usr/bin/env python3
"""
Recovery script for missing SIRENs.

Reads the list of diffusible missing SIRENs from data/reports/recoverable_sirens_all_france.txt
and fetches each one via the API, inserting into harvest_v2.sqlite.

Features:
- Rate limiting (7 req/s)
- Checkpoint every 100 SIRENs for resumability
- Logs non-found SIRENs
- Progress bar with ETA
"""

from __future__ import annotations

import sqlite3
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Set

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
SIRENS_FILE = Path("data/reports/recoverable_sirens_all_france.txt")
NOT_FOUND_FILE = Path("data/reports/sirens_not_found.txt")
PROGRESS_FILE = Path("data/reports/recovery_progress.txt")

MAX_RETRIES = 4
BACKOFF_BASE = 0.5
REQUEST_INTERVAL = 1.0 / 7.0  # 7 req/s
COMMIT_INTERVAL = 100  # Commit every N SIRENs


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
def fetch_siren(session: requests.Session, limiter: RateLimiter, siren: str) -> Optional[dict]:
    """Fetch a single SIREN via the q parameter."""
    for attempt in range(MAX_RETRIES):
        limiter.acquire()
        try:
            params = {
                "q": siren,
                "per_page": 1,
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
            data = resp.json()
            # Return first result if SIREN matches exactly
            for res in data.get("results", []) or []:
                if res.get("siren") == siren:
                    return res
            return None  # Not found
        except requests.exceptions.SSLError as e:
            LOGGER.warning("SSL error on %s (attempt %d): %s", siren, attempt + 1, e)
            time.sleep(5)
            continue
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                LOGGER.warning("Failed SIREN %s: %s", siren, e)
                return None
            time.sleep(BACKOFF_BASE * (2 ** attempt))
    return None


# --------------------------- Progress ------------------------------------ #
def load_progress() -> Set[str]:
    """Load already recovered SIRENs from progress file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_progress(siren: str) -> None:
    """Append a recovered SIREN to progress file."""
    with open(PROGRESS_FILE, "a") as f:
        f.write(siren + "\n")


def save_not_found(siren: str) -> None:
    """Append a not-found SIREN to not-found file."""
    with open(NOT_FOUND_FILE, "a") as f:
        f.write(siren + "\n")


# --------------------------- Main ------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description="Recover missing SIRENs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of SIRENs to process (for testing)")
    parser.add_argument("--start-from", type=int, default=0, help="Start from this index in the file")
    args = parser.parse_args()
    
    # Check files exist
    if not SIRENS_FILE.exists():
        LOGGER.error("SIRENs file not found: %s", SIRENS_FILE)
        LOGGER.info("Run analyze_missing_sirens.py first to generate the file.")
        return
    
    # Load SIRENs to recover
    LOGGER.info("Loading SIRENs to recover from %s...", SIRENS_FILE)
    with open(SIRENS_FILE, "r") as f:
        all_sirens = [line.strip() for line in f if line.strip()]
    LOGGER.info("Total SIRENs in file: %d", len(all_sirens))
    
    # Load already processed SIRENs
    already_done = load_progress()
    LOGGER.info("Already processed: %d", len(already_done))
    
    # Filter out already done
    sirens_to_process = [s for s in all_sirens if s not in already_done]
    LOGGER.info("Remaining to process: %d", len(sirens_to_process))
    
    # Apply start-from
    if args.start_from > 0:
        sirens_to_process = sirens_to_process[args.start_from:]
        LOGGER.info("Starting from index %d, remaining: %d", args.start_from, len(sirens_to_process))
    
    # Apply limit
    if args.limit:
        sirens_to_process = sirens_to_process[:args.limit]
        LOGGER.info("Limited to %d SIRENs", args.limit)
    
    if not sirens_to_process:
        LOGGER.info("No SIRENs to process!")
        return
    
    # Estimate time
    est_hours = len(sirens_to_process) / 7 / 3600
    LOGGER.info("Estimated time: %.1f hours (%.1f days)", est_hours, est_hours / 24)
    
    # Setup
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    limiter = RateLimiter(REQUEST_INTERVAL)
    session = requests.Session()
    session.headers.update({
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "Sireto-Recovery/1.0",
    })
    
    ent_count = 0
    dir_count = 0
    not_found_count = 0
    
    pbar = tqdm(sirens_to_process, desc="SIRENs", unit="siren")
    for i, siren in enumerate(pbar):
        res = fetch_siren(session, limiter, siren)
        
        if res:
            upsert_entreprise(conn, res)
            dir_count += upsert_dirigeants(conn, siren, res.get("dirigeants"))
            ent_count += 1
            save_progress(siren)
        else:
            not_found_count += 1
            save_not_found(siren)
            save_progress(siren)  # Mark as processed even if not found
        
        # Commit periodically
        if (i + 1) % COMMIT_INTERVAL == 0:
            conn.commit()
        
        pbar.set_postfix({"Found": ent_count, "Dir": dir_count, "NotFound": not_found_count})
    
    conn.commit()
    conn.close()
    pbar.close()
    
    # Final report
    LOGGER.info("=" * 60)
    LOGGER.info("Recovery complete!")
    LOGGER.info("  Entreprises added: %d", ent_count)
    LOGGER.info("  Dirigeants added: %d", dir_count)
    LOGGER.info("  Not found: %d", not_found_count)


if __name__ == "__main__":
    main()
