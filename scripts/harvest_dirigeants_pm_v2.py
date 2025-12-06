#!/usr/bin/env python3
"""
Harvest all enterprises whose dirigeants include at least one personne morale.

VERSION 2: SIREN Batching Approach
==================================
Instead of querying by commune (which has pagination/10k limit issues),
this version:
  1. Loads all SIRENs from StockUniteLegale parquet
  2. Batches 25 SIRENs at a time in the `q` parameter
  3. Queries: GET /search?q=SIREN1 SIREN2...&per_page=25&minimal=true&include=dirigeants
  4. Filters for type_dirigeant == "personne morale"

Advantages:
  - No pagination (always page 1)
  - No 10k limit issues
  - Maximum density (25 SIRENs per request)
  - ~175 entreprises/seconde at 7 req/s

Tables in data/dirigeants_pm_v2.sqlite:
  dirigeants_pm(siren_entreprise, siren_dirigeant, denomination_dirigeant,
               qualite, date_capture)
  harvest_progress(batch_id PRIMARY KEY, completed)
"""

from __future__ import annotations

import sqlite3
import threading
import time
import logging
from pathlib import Path
from typing import List, Optional, Iterator

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

# --------------------------- Tunables ------------------------------------- #
BASE_URL = "https://recherche-entreprises.api.gouv.fr/search"
BATCH_SIZE = 25  # Number of SIRENs per request (matches per_page max)
MAX_RETRIES = 4
BACKOFF_BASE = 0.5

DB_PATH = Path("data/dirigeants_pm_v2.sqlite")
PARQUET_PATH = Path("data/StockUniteLegale_utf8.parquet")

# Strict rate: 7 req/s = 1 request every 143ms
REQUEST_INTERVAL = 1.0 / 7.0  # ~0.143 seconds


# --------------------------- Rate Limiter ---------------------------------- #
class RateLimiter:
    """Simple strict rate limiter: enforces minimum interval between requests."""
    
    def __init__(self, interval: float):
        self.interval = interval
        self.last_request = 0.0
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_request
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_request = time.monotonic()


# --------------------------- DB helpers ------------------------------------ #
def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dirigeants_pm (
          siren_entreprise TEXT,
          siren_dirigeant TEXT,
          denomination_dirigeant TEXT,
          qualite TEXT,
          date_capture TEXT DEFAULT (datetime('now')),
          UNIQUE(siren_entreprise, siren_dirigeant, qualite)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS harvest_progress (
          batch_id INTEGER PRIMARY KEY,
          completed INTEGER DEFAULT 0
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_siren_ent ON dirigeants_pm(siren_entreprise);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_siren_dir ON dirigeants_pm(siren_dirigeant);")
    conn.commit()


def get_completed_batches(conn: sqlite3.Connection) -> set:
    """Get set of already completed batch IDs."""
    cur = conn.execute("SELECT batch_id FROM harvest_progress WHERE completed=1")
    return {row[0] for row in cur.fetchall()}


def mark_batch_completed(conn: sqlite3.Connection, batch_id: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO harvest_progress (batch_id, completed) VALUES (?, 1)",
        (batch_id,)
    )
    conn.commit()


def upsert_rows(conn: sqlite3.Connection, rows: List[tuple]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR IGNORE INTO dirigeants_pm
        (siren_entreprise, siren_dirigeant, denomination_dirigeant, qualite)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


# --------------------------- Load SIRENs ----------------------------------- #
def load_all_sirens() -> List[str]:
    """Load all unique SIRENs from the Unités Légales parquet."""
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")
    
    LOGGER.info("Loading SIRENs from parquet (this may take a moment)...")
    df = pd.read_parquet(PARQUET_PATH, columns=["siren"])
    sirens = df["siren"].dropna().drop_duplicates().astype(str).tolist()
    LOGGER.info("Loaded %d unique SIRENs", len(sirens))
    return sirens


def batch_sirens(sirens: List[str], batch_size: int = BATCH_SIZE) -> Iterator[tuple]:
    """Yield (batch_id, [list of sirens]) tuples."""
    for i in range(0, len(sirens), batch_size):
        batch = sirens[i:i + batch_size]
        batch_id = i // batch_size
        yield batch_id, batch


# --------------------------- HTTP ------------------------------------------ #
def backoff_sleep(attempt: int) -> None:
    time.sleep(BACKOFF_BASE * (2 ** attempt))


def fetch_batch(
    session: requests.Session,
    limiter: RateLimiter,
    sirens: List[str],
) -> Optional[dict]:
    """Fetch a batch of SIRENs using the q parameter."""
    # Build query string: "SIREN1 SIREN2 SIREN3..."
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
                retry_after = float(resp.headers.get("Retry-After", 3.0))
                delay = max(retry_after, BACKOFF_BASE * (2 ** attempt))
                LOGGER.warning("429, sleeping %.1fs", delay)
                time.sleep(delay)
                continue
            if resp.status_code == 400:
                LOGGER.warning("400 Bad Request - skipping batch")
                return None
            if resp.status_code in (500, 502, 503, 504):
                backoff_sleep(attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            LOGGER.warning("Timeout, retrying...")
            backoff_sleep(attempt)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                LOGGER.error("Failed to fetch batch: %s", e)
                return None
            backoff_sleep(attempt)
    return None


def extract_pm_dirigeants(payload: dict) -> List[tuple]:
    """Extract dirigeants PM rows from API response."""
    rows = []
    for res in payload.get("results", []) or []:
        siren_ent = res.get("siren")
        if not siren_ent:
            continue
        for d in res.get("dirigeants") or []:
            if d.get("type_dirigeant") != "personne morale":
                continue
            siren_dir = d.get("siren")
            if not siren_dir:
                continue
            rows.append((
                siren_ent,
                siren_dir,
                d.get("denomination"),
                d.get("qualite"),
            ))
    return rows


# --------------------------- Main ------------------------------------------ #
def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    ensure_db(conn)
    
    # Load all SIRENs
    all_sirens = load_all_sirens()
    total_batches = (len(all_sirens) + BATCH_SIZE - 1) // BATCH_SIZE
    LOGGER.info("Total batches: %d (SIREN count: %d)", total_batches, len(all_sirens))
    
    # Get completed batches
    completed = get_completed_batches(conn)
    LOGGER.info("Already completed: %d batches", len(completed))
    
    # Filter out completed batches
    batches_to_process = [
        (batch_id, batch) 
        for batch_id, batch in batch_sirens(all_sirens) 
        if batch_id not in completed
    ]
    LOGGER.info("Batches to process: %d", len(batches_to_process))
    
    if not batches_to_process:
        LOGGER.info("All batches already processed!")
        conn.close()
        return
    
    # Setup
    limiter = RateLimiter(REQUEST_INTERVAL)
    session = requests.Session()
    session.headers.update({
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "User-Agent": "Sireto-HarvestDirigeantsPM/2.0",
    })
    
    # Progress bar
    pbar = tqdm(batches_to_process, desc="Batches", unit="batch")
    
    total_pm = 0
    for batch_id, sirens in pbar:
        payload = fetch_batch(session, limiter, sirens)
        if payload:
            rows = extract_pm_dirigeants(payload)
            if rows:
                upsert_rows(conn, rows)
                total_pm += len(rows)
        
        mark_batch_completed(conn, batch_id)
        pbar.set_postfix({"PM": total_pm})
    
    pbar.close()
    
    # Final report
    cur = conn.execute("SELECT COUNT(*) FROM dirigeants_pm")
    total = cur.fetchone()[0]
    LOGGER.info("Harvest complete! Total dirigeants PM: %d", total)
    conn.close()


if __name__ == "__main__":
    main()
