#!/usr/bin/env python3
"""
Harvest all enterprises whose dirigeants include at least one personne morale.

Robust commune-based approach with AUTO-SHARDING:
  - Load all communes from local parquet file
  - Expand Paris/Lyon/Marseille to arrondissements
  - Fetch each commune directly
  - **AUTO-SHARD by NAF section if commune hits 10k limit**
  - Parallel workers + strict rate limiting
  - Checkpoint per scope for resumability

Tables in data/dirigeants_pm.sqlite:
  dirigeants_pm(siren_entreprise, siren_dirigeant, denomination_dirigeant,
               qualite, code_commune, date_capture)
  harvest_progress(scope_key PRIMARY KEY, last_page, total_results, completed)
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
WORKERS = 2  # Overlap network latency while respecting global rate limit
MAX_RETRIES = 4
BACKOFF_BASE = 0.5

DB_PATH = Path("data/dirigeants_pm.sqlite")
PARQUET_PATH = Path("data/StockEtablissement_utf8.parquet")

# Strict rate: 6.6 req/s = 1 request every 151ms (API documented limit)
REQUEST_INTERVAL = 1.0 / 6.6  # ~0.151 seconds; tune here if the provider changes the cap

# NAF sections for auto-sharding when commune hits 10k
NAF_SECTIONS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
    "N", "O", "P", "Q", "R", "S", "T", "U"
]


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


# --------------------------- Scope ----------------------------------------- #
@dataclass(frozen=True)
class Scope:
    """A scope defines what to fetch: commune + optional NAF filters.
    
    Multi-level sharding:
    - Level 0: commune only (section_naf=None, activite_principale=None)
    - Level 1: commune + section NAF (section_naf='A'-'U', activite_principale=None)
    - Level 2: commune + full NAF code (section_naf=None, activite_principale='01.11A')
    """
    code_commune: str
    section_naf: Optional[str] = None  # 'A'-'U' for section sharding
    activite_principale: Optional[str] = None  # Full NAF code for deep sharding
    
    @property
    def key(self) -> str:
        """Unique key for progress tracking."""
        if self.activite_principale:
            return f"{self.code_commune}:NAF:{self.activite_principale}"
        if self.section_naf:
            return f"{self.code_commune}:{self.section_naf}"
        return self.code_commune
    
    @property
    def shard_level(self) -> int:
        """Return the sharding depth: 0=commune, 1=section, 2=full NAF."""
        if self.activite_principale:
            return 2
        if self.section_naf:
            return 1
        return 0


# --------------------------- Arrondissement handling ----------------------- #
def expand_arrondissements(insee_code: str) -> List[str]:
    """Expand Paris/Lyon/Marseille parent codes to arrondissements."""
    if insee_code == "75056":
        return [f"75{100 + i:03d}" for i in range(1, 21)]
    if insee_code == "69123":
        return [f"6938{i}" for i in range(1, 10)]
    if insee_code == "13055":
        return [f"132{str(i).zfill(2)}" for i in range(1, 17)]
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
          scope_key TEXT PRIMARY KEY,
          last_page INTEGER DEFAULT 0,
          total_results INTEGER,
          completed INTEGER DEFAULT 0
        );
        """
    )
    conn.commit()


def load_progress(conn: sqlite3.Connection, scope: Scope) -> tuple[int, bool]:
    """Returns (last_page, completed)."""
    cur = conn.execute(
        "SELECT last_page, completed FROM harvest_progress WHERE scope_key=?",
        (scope.key,),
    )
    row = cur.fetchone()
    if row:
        return row[0], bool(row[1])
    return 0, False


def save_progress(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    scope: Scope,
    last_page: int,
    total_results: Optional[int] = None,
    completed: bool = False,
) -> None:
    with lock:
        conn.execute(
            """
            INSERT INTO harvest_progress (scope_key, last_page, total_results, completed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scope_key)
            DO UPDATE SET last_page=excluded.last_page, 
                          total_results=COALESCE(excluded.total_results, harvest_progress.total_results),
                          completed=excluded.completed
            """,
            (scope.key, last_page, total_results, int(completed)),
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
    scope: Scope,
    page: int,
) -> Optional[dict]:
    """Fetch a single page for a scope. Returns None on failure."""
    for attempt in range(MAX_RETRIES):
        limiter.acquire()
        try:
            params = {
                "code_commune": scope.code_commune,
                "per_page": PER_PAGE,
                "page": page,
                "minimal": True,
                "include": "dirigeants",
            }
            # Add NAF filters based on sharding level
            if scope.activite_principale:
                params["activite_principale"] = scope.activite_principale
            elif scope.section_naf:
                params["section_activite_principale"] = scope.section_naf
            
            resp = session.get(BASE_URL, params=params, timeout=15)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 3.0))
                delay = max(retry_after, BACKOFF_BASE * (2 ** attempt))
                LOGGER.warning("429 on %s page %d, sleeping %.1fs", scope.key, page, delay)
                time.sleep(delay)
                continue
            if resp.status_code == 400:
                # Invalid NAF code (old nomenclature) - skip gracefully
                LOGGER.warning("400 Bad Request on %s (invalid NAF code?) - skipping", scope.key)
                return None  # Don't retry, just skip this scope
            if resp.status_code in (500, 502, 503, 504):
                backoff_sleep(attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                LOGGER.error("Failed to fetch %s page %d: %s", scope.key, page, e)
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
def fetch_all_for_scope(
    session: requests.Session,
    limiter: RateLimiter,
    conn: sqlite3.Connection,
    db_lock: threading.Lock,
    scope: Scope,
    progress_bar: tqdm,
    scope_queue: "queue.Queue[Scope]",
) -> None:
    """Robustly fetch ALL dirigeants PM for a scope.
    
    If a commune (without NAF filter) hits 10k, we AUTO-SHARD by adding
    NAF-filtered scopes to the queue instead of losing data.
    """
    last_page, completed = load_progress(conn, scope)
    if completed:
        return
    
    start_page = last_page + 1
    
    # First page to get total_results
    first_payload = fetch_page(session, limiter, scope, start_page)
    if not first_payload:
        return
    
    total_results = first_payload.get("total_results", 0)
    total_pages = first_payload.get("total_pages", 1)
    
    # AUTO-SHARD LEVEL 1: commune hits 10k → split by NAF sections
    if total_results >= 10000 and scope.shard_level == 0:
        LOGGER.info(
            "🔀 AUTO-SHARD L1: Commune %s has %d results, splitting by NAF sections",
            scope.code_commune, total_results
        )
        # Process each section IMMEDIATELY (not queued for later)
        for section in NAF_SECTIONS:
            new_scope = Scope(scope.code_commune, section_naf=section)
            # PERSIST immediately so shards survive script interruption
            save_progress(conn, db_lock, new_scope, 0, None, completed=False)
            # Process the section NOW (recursive call)
            fetch_all_for_scope(session, limiter, conn, db_lock, new_scope, progress_bar, scope_queue)
        save_progress(conn, db_lock, scope, 0, total_results, completed=True)
        return
    
    # AUTO-SHARD LEVEL 2: section still hits 10k → split by full NAF codes
    if total_results >= 10000 and scope.shard_level == 1:
        LOGGER.info(
            "🔀 AUTO-SHARD L2: %s:%s has %d results, fetching NAF codes to split further",
            scope.code_commune, scope.section_naf, total_results
        )
        # Extract all unique NAF codes from results to shard by
        naf_codes_seen = set()
        # Process all pages to collect NAF codes (we'll re-fetch with specific NAF later)
        for res in first_payload.get("results", []) or []:
            naf = res.get("activite_principale")
            if naf:
                naf_codes_seen.add(naf)
        
        # Fetch remaining pages to get all NAF codes
        for page in range(2, min(total_pages + 1, MAX_PAGES + 1)):
            payload = fetch_page(session, limiter, scope, page)
            if not payload or not payload.get("results"):
                break
            for res in payload.get("results", []) or []:
                naf = res.get("activite_principale")
                if naf:
                    naf_codes_seen.add(naf)
        
        LOGGER.info("Found %d unique NAF codes in %s:%s", len(naf_codes_seen), scope.code_commune, scope.section_naf)
        
        # Process each NAF code IMMEDIATELY (not queued for later)
        for naf_code in naf_codes_seen:
            new_scope = Scope(scope.code_commune, activite_principale=naf_code)
            # PERSIST immediately so shards survive script interruption
            save_progress(conn, db_lock, new_scope, 0, None, completed=False)
            # Process the NAF code NOW (recursive call)
            fetch_all_for_scope(session, limiter, conn, db_lock, new_scope, progress_bar, scope_queue)
        
        save_progress(conn, db_lock, scope, 0, total_results, completed=True)
        return
    
    # LEVEL 2 still hits 10k? Log ERROR but continue (extremely rare edge case)
    if total_results >= 10000 and scope.shard_level == 2:
        LOGGER.error(
            "❌ CANNOT SHARD FURTHER: %s with NAF %s has %d results - POTENTIAL DATA LOSS!",
            scope.code_commune, scope.activite_principale, total_results
        )
    
    # Process first page
    rows = extract_rows(scope.code_commune, first_payload)
    if rows:
        upsert_rows(conn, rows, db_lock)
    save_progress(conn, db_lock, scope, start_page, total_results)
    progress_bar.update(1)
    
    # Process remaining pages
    for page in range(start_page + 1, min(total_pages + 1, MAX_PAGES + 1)):
        payload = fetch_page(session, limiter, scope, page)
        if not payload:
            break
        
        rows = extract_rows(scope.code_commune, payload)
        if rows:
            upsert_rows(conn, rows, db_lock)
        save_progress(conn, db_lock, scope, page, total_results)
        progress_bar.update(1)
        
        if not payload.get("results"):
            break
    
    # Mark as completed
    save_progress(conn, db_lock, scope, total_pages, total_results, completed=True)


# --------------------------- Worker ---------------------------------------- #
def worker(
    scope_queue: "queue.Queue[Scope]",
    limiter: RateLimiter,
    db_lock: threading.Lock,
    page_counter: tqdm,
    conn: sqlite3.Connection,
):
    session = requests.Session()
    session.headers.update({
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "User-Agent": "Sireto-HarvestDirigeantsPM/2.0",
    })
    adapter = HTTPAdapter(
        pool_connections=WORKERS * 4,
        pool_maxsize=WORKERS * 4,
        max_retries=Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        ),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    while True:
        try:
            scope = scope_queue.get(timeout=1.0)
        except queue.Empty:
            # Check if queue is really empty (other workers might add items)
            if scope_queue.empty():
                break
            continue
        fetch_all_for_scope(session, limiter, conn, db_lock, scope, page_counter, scope_queue)
        scope_queue.task_done()


# --------------------------- Main ------------------------------------------ #
def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    ensure_db(conn)

    LOGGER.info("Loading communes from parquet...")
    all_communes = load_all_communes()
    LOGGER.info("Found %d communes (including arrondissements)", len(all_communes))

    # Build initial scopes (just communes, no NAF filter)
    # NAF-sharded scopes will be added dynamically if needed
    all_scopes = [Scope(c) for c in all_communes]

    # Filter out already completed scopes
    completed_keys = set()
    cur = conn.execute("SELECT scope_key FROM harvest_progress WHERE completed=1")
    for (key,) in cur.fetchall():
        completed_keys.add(key)
    
    remaining = [s for s in all_scopes if s.key not in completed_keys]
    
    # ALSO load pending sharded scopes (persisted but not completed, e.g. after interruption)
    pending_shards = []
    cur = conn.execute("SELECT scope_key FROM harvest_progress WHERE completed=0")
    for (key,) in cur.fetchall():
        if ":" in key:  # It's a sharded scope (section or NAF code)
            parts = key.split(":")
            if parts[1] == "NAF":
                # Level 2: commune:NAF:code
                pending_shards.append(Scope(parts[0], activite_principale=parts[2]))
            else:
                # Level 1: commune:section
                pending_shards.append(Scope(parts[0], section_naf=parts[1]))
    
    LOGGER.info("Scopes to process: %d communes + %d pending shards (already completed: %d)", 
                len(remaining), len(pending_shards), len(completed_keys))

    all_remaining = remaining + pending_shards
    
    if not all_remaining:
        LOGGER.info("All scopes already processed!")
        conn.close()
        return

    scope_queue: queue.Queue[Scope] = queue.Queue()
    for s in all_remaining:
        scope_queue.put(s)

    limiter = RateLimiter(REQUEST_INTERVAL)
    db_lock = threading.Lock()

    # Progress bar (estimate ~5 pages per scope on average)
    page_counter = tqdm(total=len(all_remaining) * 5, desc="Pages", smoothing=0.1)

    threads = []
    for _ in range(WORKERS):
        t = threading.Thread(
            target=worker,
            args=(scope_queue, limiter, db_lock, page_counter, conn),
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
    completed_count, over_10k = cur.fetchone()
    LOGGER.info("Harvest complete: %d scopes, %d auto-sharded (10k+)", 
                completed_count, over_10k or 0)
    
    conn.close()


if __name__ == "__main__":
    main()
