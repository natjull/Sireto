#!/usr/bin/env python3
"""
Harvest all enterprises whose dirigeants include at least one personne morale.

Robust commune-based approach:
  - Load all communes from local parquet file
  - Expand Paris/Lyon/Marseille to arrondissements (like sirene_client.py)
  - Fetch each commune/arrondissement directly (no pre-scan)
  - Detect and warn if 10k limit reached (indicates potential data loss)
  - Parallel workers + rate limiting
  - Checkpoint per commune for resumability

Tables in data/dirigeants_pm.sqlite:
  dirigeants_pm(siren_entreprise, siren_dirigeant, denomination_dirigeant,
               qualite, code_commune, date_capture)
  harvest_progress(code_commune PRIMARY KEY, last_page, total_results, completed)
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

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
PER_PAGE = 25  # API maximum
MAX_PAGES = 400  # API maximum (400 * 25 = 10,000 results max)
WORKERS = 4
RATE_LIMIT_RPS = 6.5  # stay below 7 req/s
RATE_WINDOW = 10.0
MAX_RETRIES = 4
BACKOFF_BASE = 0.3

DB_PATH = Path("data/dirigeants_pm.sqlite")
PARQUET_PATH = Path("data/StockEtablissement_utf8.parquet")


# --------------------------- Rate Limiter ---------------------------------- #
class RateLimiter:
    def __init__(self, rps: float, window: float):
        self.capacity = int(rps * window)
        self.tokens = self.capacity
        self.refill_rate = rps
        self.last = time.monotonic()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def acquire(self) -> None:
        with self.cond:
            while True:
                now = time.monotonic()
                elapsed = now - self.last
                if elapsed > 0:
                    refill = elapsed * self.refill_rate
                    if refill >= 1:
                        self.tokens = min(self.capacity, self.tokens + int(refill))
                        self.last = now
                if self.tokens > 0:
                    self.tokens -= 1
                    return
                self.cond.wait(timeout=max(0.01, 1.0 / self.refill_rate))


# --------------------------- Arrondissement handling ----------------------- #
def expand_arrondissements(insee_code: str) -> List[str]:
    """Expand Paris/Lyon/Marseille parent codes to arrondissements.
    
    Like sirene_client.py:
    - Paris parent: 75056 -> 75101..75120
    - Lyon parent: 69123 -> 69381..69389
    - Marseille parent: 13055 -> 13201..13216
    Otherwise, returns [insee_code].
    """
    if insee_code == "75056":
        return [f"75{100 + i:03d}" for i in range(1, 21)]  # 75101..75120
    if insee_code == "69123":
        return [f"6938{i}" for i in range(1, 10)]  # 69381..69389
    if insee_code == "13055":
        return [f"132{str(i).zfill(2)}" for i in range(1, 17)]  # 13201..13216
    return [insee_code]


# --------------------------- DB helpers ------------------------------------ #
def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dirigeants_pm (
          siren_entreprise TEXT,
          siren_dirigeant TEXT,
          denomination_dirigeant TEXT,
          qualite TEXT,
          code_commune TEXT,
          date_capture TEXT DEFAULT (datetime('now')),
          UNIQUE(siren_entreprise, siren_dirigeant, qualite, code_commune)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS harvest_progress (
          code_commune TEXT PRIMARY KEY,
          last_page INTEGER DEFAULT 0,
          total_results INTEGER,
          completed INTEGER DEFAULT 0
        );
        """
    )
    conn.commit()


def load_progress(conn: sqlite3.Connection, code_commune: str) -> tuple[int, bool]:
    """Returns (last_page, completed)."""
    cur = conn.execute(
        "SELECT last_page, completed FROM harvest_progress WHERE code_commune=?",
        (code_commune,),
    )
    row = cur.fetchone()
    if row:
        return row[0], bool(row[1])
    return 0, False


def save_progress(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    code_commune: str,
    last_page: int,
    total_results: Optional[int] = None,
    completed: bool = False,
) -> None:
    with lock:
        conn.execute(
            """
            INSERT INTO harvest_progress (code_commune, last_page, total_results, completed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(code_commune)
            DO UPDATE SET last_page=excluded.last_page, 
                          total_results=COALESCE(excluded.total_results, total_results),
                          completed=excluded.completed
            """,
            (code_commune, last_page, total_results, int(completed)),
        )
        conn.commit()


def upsert_rows(conn: sqlite3.Connection, rows: Iterable[tuple], lock: threading.Lock) -> None:
    with lock:
        conn.executemany(
            """
            INSERT OR IGNORE INTO dirigeants_pm
            (siren_entreprise, siren_dirigeant, denomination_dirigeant, qualite, code_commune)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


# --------------------------- Commune list ---------------------------------- #
def load_all_communes() -> List[str]:
    """Load all unique commune codes from parquet, expanding Paris/Lyon/Marseille."""
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")
    
    df = pd.read_parquet(PARQUET_PATH, columns=["codeCommuneEtablissement"])
    raw_communes = (
        df["codeCommuneEtablissement"]
        .dropna()
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    
    # Expand Paris/Lyon/Marseille to arrondissements
    all_codes: List[str] = []
    for code in raw_communes:
        all_codes.extend(expand_arrondissements(code))
    
    return sorted(set(all_codes))


# --------------------------- HTTP ------------------------------------------ #
def backoff_sleep(attempt: int) -> None:
    time.sleep(BACKOFF_BASE * (2 ** attempt))


def fetch_page(
    session: requests.Session,
    limiter: RateLimiter,
    code_commune: str,
    page: int,
) -> Optional[dict]:
    """Fetch a single page for a commune. Returns None on failure."""
    for attempt in range(MAX_RETRIES):
        limiter.acquire()
        try:
            params = {
                "code_commune": code_commune,
                "per_page": PER_PAGE,
                "page": page,
                "minimal": True,
                "include": "dirigeants",
            }
            resp = session.get(BASE_URL, params=params, timeout=15)
            if resp.status_code == 429:
                delay = float(resp.headers.get("Retry-After", 2.0))
                LOGGER.warning("429 on commune %s page %d, sleeping %.1fs", code_commune, page, delay)
                time.sleep(delay)
                continue
            if resp.status_code in (500, 502, 503, 504):
                backoff_sleep(attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                LOGGER.error("Failed to fetch commune %s page %d: %s", code_commune, page, e)
                return None
            backoff_sleep(attempt)
    return None


def extract_rows(code_commune: str, payload: dict) -> List[tuple]:
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
            rows.append(
                (
                    siren_ent,
                    siren_dir,
                    d.get("denomination"),
                    d.get("qualite"),
                    code_commune,
                )
            )
    return rows


# --------------------------- Core fetch function --------------------------- #
def fetch_all_for_commune(
    session: requests.Session,
    limiter: RateLimiter,
    conn: sqlite3.Connection,
    db_lock: threading.Lock,
    code_commune: str,
    progress_bar: tqdm,
) -> None:
    """Robustly fetch ALL dirigeants PM for a single commune.
    
    Features:
    - Resume from last_page if interrupted
    - Detect 10k limit and log WARNING
    - Save progress after each page
    - Mark commune as completed when done
    """
    last_page, completed = load_progress(conn, code_commune)
    if completed:
        return  # Already done
    
    start_page = last_page + 1
    
    # First page to get total_results
    first_payload = fetch_page(session, limiter, code_commune, start_page)
    if not first_payload:
        return
    
    total_results = first_payload.get("total_results", 0)
    total_pages = first_payload.get("total_pages", 1)
    
    # CRITICAL: Detect 10k limit
    if total_results >= 10000:
        LOGGER.warning(
            "⚠️ COMMUNE %s HAS %d RESULTS (10k LIMIT) - POTENTIAL DATA LOSS!",
            code_commune, total_results
        )
    
    # Process first page
    rows = extract_rows(code_commune, first_payload)
    if rows:
        upsert_rows(conn, rows, db_lock)
    save_progress(conn, db_lock, code_commune, start_page, total_results)
    progress_bar.update(1)
    
    # Process remaining pages
    for page in range(start_page + 1, min(total_pages + 1, MAX_PAGES + 1)):
        payload = fetch_page(session, limiter, code_commune, page)
        if not payload:
            break
        
        rows = extract_rows(code_commune, payload)
        if rows:
            upsert_rows(conn, rows, db_lock)
        save_progress(conn, db_lock, code_commune, page, total_results)
        progress_bar.update(1)
        
        # Stop if no more results
        if not payload.get("results"):
            break
    
    # Mark as completed
    save_progress(conn, db_lock, code_commune, total_pages, total_results, completed=True)


# --------------------------- Worker ---------------------------------------- #
def worker(
    commune_queue: "queue.Queue[str]",
    limiter: RateLimiter,
    db_lock: threading.Lock,
    page_counter: tqdm,
    conn: sqlite3.Connection,
):
    session = requests.Session()
    session.headers.update({
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "User-Agent": "Sireto-HarvestDirigeantsPM/1.0",
    })

    while True:
        try:
            code_commune = commune_queue.get_nowait()
        except queue.Empty:
            break
        fetch_all_for_commune(session, limiter, conn, db_lock, code_commune, page_counter)
        commune_queue.task_done()


# --------------------------- Main ------------------------------------------ #
def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    ensure_db(conn)

    LOGGER.info("Loading communes from parquet...")
    all_communes = load_all_communes()
    LOGGER.info("Found %d communes (including arrondissements)", len(all_communes))

    # Filter out already completed communes
    completed_communes = set()
    cur = conn.execute("SELECT code_commune FROM harvest_progress WHERE completed=1")
    for (code,) in cur.fetchall():
        completed_communes.add(code)
    
    remaining = [c for c in all_communes if c not in completed_communes]
    LOGGER.info("Communes to process: %d (already completed: %d)", 
                len(remaining), len(completed_communes))

    if not remaining:
        LOGGER.info("All communes already processed!")
        conn.close()
        return

    commune_queue: queue.Queue[str] = queue.Queue()
    for c in remaining:
        commune_queue.put(c)

    limiter = RateLimiter(RATE_LIMIT_RPS, RATE_WINDOW)
    db_lock = threading.Lock()

    # Progress bar estimates ~5 pages per commune on average
    page_counter = tqdm(total=len(remaining) * 5, desc="Pages", smoothing=0.1)

    threads = []
    for _ in range(WORKERS):
        t = threading.Thread(
            target=worker,
            args=(commune_queue, limiter, db_lock, page_counter, conn),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    page_counter.close()
    
    # Final report
    cur = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN total_results >= 10000 THEN 1 ELSE 0 END) "
        "FROM harvest_progress WHERE completed=1"
    )
    completed, over_10k = cur.fetchone()
    LOGGER.info("Harvest complete: %d communes, %d with 10k+ results (check logs)", 
                completed, over_10k or 0)
    
    conn.close()


if __name__ == "__main__":
    main()
