#!/usr/bin/env python3
"""
Harvest all enterprises whose dirigeants include at least one personne morale,
with sharding to bypass the 10k-results-per-request cap of the API.

Features:
  - Pre-scan each département; if total_results < 10k, fetch by département.
    If total_results == 10k (cap reached), shard by communes of the département.
  - Commune list loaded from local parquet StockEtablissement_utf8.parquet
    (columns codeCommuneEtablissement, codeDepartementEtablissement) for accuracy.
  - Parallel workers (default 4) + shared token bucket limiter (<7 req/s global).
  - requests.Session keep-alive + gzip.
  - minimal=true + include=dirigeants to keep dirigeants while reducing payload.
  - Retries with backoff, re-try on empty pages, checkpoint per scope
    (departement or commune) for resumability.
  - Duplicate-safe INSERT OR IGNORE.
  - tqdm progress on pages.

Tables in data/dirigeants_pm.sqlite:
  dirigeants_pm(siren_entreprise, siren_dirigeant, denomination_dirigeant,
                qualite, source_departement, source_commune, date_capture)
  harvest_progress(scope_type, departement, commune, last_page)
    scope_type ∈ {'DEP','COM'}
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from tqdm import tqdm

# --------------------------- Tunables ------------------------------------- #
BASE_URL = "https://recherche-entreprises.api.gouv.fr/search"
PER_PAGE = 25  # API maximum
WORKERS = 4
RATE_LIMIT_RPS = 6.5  # stay below 7 req/s
RATE_WINDOW = 10.0
MAX_RETRIES = 4
BACKOFF_BASE = 0.3
EMPTY_PAGE_RETRIES = 2

DB_PATH = Path("data/dirigeants_pm.sqlite")
PARQUET_PATH = Path("data/StockEtablissement_utf8.parquet")

DEPARTEMENTS = (
    [f"{i:02d}" for i in range(1, 96)]
    + ["2A", "2B"]
    + ["971", "972", "973", "974", "975", "976", "978", "986", "987", "988"]
)


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


# --------------------------- Data models ----------------------------------- #
@dataclass(frozen=True)
class Scope:
    scope_type: str  # 'DEP' or 'COM'
    departement: str
    commune: Optional[str] = None


@dataclass
class PageState:
    scope: Scope
    last_page: int


# --------------------------- DB helpers ------------------------------------ #
def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dirigeants_pm (
          siren_entreprise TEXT,
          siren_dirigeant TEXT,
          denomination_dirigeant TEXT,
          qualite TEXT,
          source_departement TEXT,
          source_commune TEXT,
          date_capture TEXT DEFAULT (datetime('now')),
          UNIQUE(siren_entreprise, siren_dirigeant, qualite, source_commune)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS harvest_progress (
          scope_type TEXT,
          departement TEXT,
          commune TEXT,
          last_page INTEGER,
          PRIMARY KEY(scope_type, departement, commune)
        );
        """
    )
    conn.commit()


def load_progress(conn: sqlite3.Connection, scope: Scope) -> PageState:
    cur = conn.execute(
        """
        SELECT last_page FROM harvest_progress
        WHERE scope_type=? AND departement=? AND commune IS ?
        """,
        (scope.scope_type, scope.departement, scope.commune),
    )
    row = cur.fetchone()
    return PageState(scope, row[0] if row else 0)


def save_progress(conn: sqlite3.Connection, state: PageState, lock: threading.Lock) -> None:
    with lock:
        conn.execute(
            """
            INSERT INTO harvest_progress (scope_type, departement, commune, last_page)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scope_type, departement, commune)
            DO UPDATE SET last_page=excluded.last_page
            """,
            (state.scope.scope_type, state.scope.departement, state.scope.commune, state.last_page),
        )
        conn.commit()


def upsert_rows(conn: sqlite3.Connection, rows: Iterable[tuple], lock: threading.Lock) -> None:
    with lock:
        conn.executemany(
            """
            INSERT OR IGNORE INTO dirigeants_pm
            (siren_entreprise, siren_dirigeant, denomination_dirigeant,
             qualite, source_departement, source_commune)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


# --------------------------- Commune list ---------------------------------- #
def load_communes_for_departement(dep: str) -> list[str]:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")
    # Read only codeCommuneEtablissement (codeDepartementEtablissement doesn't exist)
    df = pd.read_parquet(PARQUET_PATH, columns=["codeCommuneEtablissement"])
    df = df.dropna()
    df["codeCommuneEtablissement"] = df["codeCommuneEtablissement"].astype(str)
    
    # Extract département from code commune
    # - 2A, 2B: Corse (starts with "2A" or "2B")
    # - 971-988: DOM-TOM (3 first chars)
    # - 01-95: Metropolitan France (2 first chars)
    def extract_dept(code: str) -> str:
        if code.startswith("2A") or code.startswith("2B"):
            return code[:2]
        elif len(code) >= 3 and code[:3] in ["971", "972", "973", "974", "975", "976", "978", "986", "987", "988"]:
            return code[:3]
        else:
            return code[:2]
    
    df["dept"] = df["codeCommuneEtablissement"].apply(extract_dept)
    communes = (
        df[df["dept"] == dep]["codeCommuneEtablissement"]
        .drop_duplicates()
        .tolist()
    )
    return sorted(communes)


# --------------------------- HTTP ------------------------------------------ #
def backoff_sleep(attempt: int) -> None:
    time.sleep(BACKOFF_BASE * (2 ** attempt))


def fetch_page(
    session: requests.Session,
    limiter: RateLimiter,
    scope: Scope,
    page: int,
) -> Optional[dict]:
    for attempt in range(MAX_RETRIES):
        limiter.acquire()
        try:
            params = {
                "per_page": PER_PAGE,
                "page": page,
                "minimal": True,
                "include": "dirigeants",
            }
            if scope.scope_type == "DEP":
                params["departement"] = scope.departement
            else:
                params["code_commune"] = scope.commune

            resp = session.get(BASE_URL, params=params, timeout=10)
            if resp.status_code == 429:
                backoff_sleep(attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            backoff_sleep(attempt)
    return None


def extract_rows(scope: Scope, payload: dict) -> list[tuple]:
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
                    scope.departement,
                    scope.commune,
                )
            )
    return rows


# --------------------------- Worker ---------------------------------------- #
def process_scope(
    session: requests.Session,
    limiter: RateLimiter,
    conn: sqlite3.Connection,
    db_lock: threading.Lock,
    page_counter: tqdm,
    scope: Scope,
) -> None:
    state = load_progress(conn, scope)
    start_page = state.last_page + 1

    first_payload = fetch_page(session, limiter, scope, start_page)
    if not first_payload:
        return
    total_pages = first_payload.get("total_pages", start_page)

    def handle_page(page: int, payload: dict) -> bool:
        rows = extract_rows(scope, payload)
        if rows:
            upsert_rows(conn, rows, db_lock)
        elif not payload.get("results"):
            return False
        return True

    def do_page(page: int, payload: dict) -> bool:
        if handle_page(page, payload):
            page_counter.update(1)
            state.last_page = page
            save_progress(conn, state, db_lock)
            return True
        # retry empty page
        for _ in range(EMPTY_PAGE_RETRIES):
            retry_payload = fetch_page(session, limiter, scope, page)
            if retry_payload and handle_page(page, retry_payload):
                page_counter.update(1)
                state.last_page = page
                save_progress(conn, state, db_lock)
                return True
        return False

    if not do_page(start_page, first_payload):
        return

    for page in range(start_page + 1, total_pages + 1):
        payload = fetch_page(session, limiter, scope, page)
        if not payload:
            break
        if not do_page(page, payload):
            break


def worker(
    scope_queue: "queue.Queue[Scope]",
    limiter: RateLimiter,
    db_lock: threading.Lock,
    page_counter: tqdm,
    conn: sqlite3.Connection,
):
    session = requests.Session()
    session.headers.update({"Accept-Encoding": "gzip, deflate", "Connection": "keep-alive"})

    while True:
        try:
            scope = scope_queue.get_nowait()
        except queue.Empty:
            break
        process_scope(session, limiter, conn, db_lock, page_counter, scope)
        scope_queue.task_done()


# --------------------------- Pre-scan -------------------------------------- #
def prescan_and_build_scopes() -> list[Scope]:
    scopes: list[Scope] = []
    session = requests.Session()
    limiter = RateLimiter(RATE_LIMIT_RPS, RATE_WINDOW)

    for dep in tqdm(DEPARTEMENTS, desc="Pré-scan départements"):
        limiter.acquire()
        r = session.get(
            BASE_URL,
            params={
                "departement": dep,
                "per_page": 1,
                "page": 1,
                "minimal": True,
                "include": "dirigeants",
            },
            timeout=10,
        )
        if r.status_code != 200:
            continue
        data = r.json()
        total = data.get("total_results", 0)
        if total < 10000:
            scopes.append(Scope("DEP", dep))
        else:
            # shard by communes
            communes = load_communes_for_departement(dep)
            for com in communes:
                scopes.append(Scope("COM", dep, com))
    return scopes


# --------------------------- Main ------------------------------------------ #
def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    ensure_db(conn)

    scopes = prescan_and_build_scopes()

    scope_queue: queue.Queue[Scope] = queue.Queue()
    for s in scopes:
        scope_queue.put(s)

    limiter = RateLimiter(RATE_LIMIT_RPS, RATE_WINDOW)
    db_lock = threading.Lock()

    # progress estimate: assume 400 pages per scope minus progress already done
    total_pages_est = 0
    cur = conn.execute("SELECT scope_type, departement, commune, last_page FROM harvest_progress")
    prog_map = {(t, d, c): lp for t, d, c, lp in cur.fetchall()}
    for s in scopes:
        key = (s.scope_type, s.departement, s.commune)
        already = prog_map.get(key, 0)
        total_pages_est += max(0, 400 - already)

    page_counter = tqdm(total=total_pages_est, desc="Pages", smoothing=0.1)

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
    conn.close()


if __name__ == "__main__":
    main()
