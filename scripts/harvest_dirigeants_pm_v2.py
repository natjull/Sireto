#!/usr/bin/env python3
"""
Harvest ALL enterprises from the API Recherche d'entreprises.

VERSION 2: Recursive Drill-Down Approach - FULL DATA
=====================================================
Instead of hardcoded sharding, this version uses intelligent recursive drill-down:
  1. Start with département
  2. If >10k results → drill down to code_postal
  3. If still >10k → drill down to section_activite_principale
  4. If still >10k → drill down to activite_principale  
  5. If still >10k → drill down to tranche_effectif_salarie

Captures:
  - ALL enterprises (not filtered by dirigeant type)
  - ALL dirigeants (both personne physique and personne morale)

Tables in data/harvest_v2.sqlite:
  entreprises(siren, nom_complet, nature_juridique, activite_principale,
              categorie_entreprise, tranche_effectif, etat_administratif, date_capture)
  dirigeants(siren_entreprise, type_dirigeant, siren_dirigeant, denomination,
             nom, prenoms, qualite, date_capture)
  fetch_progress(scope_key PRIMARY KEY, total_results, completed)
"""

from __future__ import annotations

import sqlite3
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

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
MAX_RESULTS = 10000  # API hard limit
MAX_PAGES = 400  # 400 * 25 = 10,000
MAX_RETRIES = 4
BACKOFF_BASE = 0.5

DB_PATH = Path("data/harvest_v2.sqlite")

# Strict rate: 7 req/s
REQUEST_INTERVAL = 1.0 / 7.0  # ~0.143 seconds

# All départements (including DOM-TOM)
DEPARTEMENTS = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "21",
    "22", "23", "24", "25", "26", "27", "28", "29", "2A", "2B",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
    "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
    "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
    "90", "91", "92", "93", "94", "95", "971", "972", "973", "974",
    "975", "976", "977", "978", "984", "986", "987", "988", "989"
]

# NAF Sections (A-U)
NAF_SECTIONS = list("ABCDEFGHIJKLMNOPQRSTU")

# Tranches d'effectifs (INSEE coding)
TRANCHES_EFFECTIF = ["NN", "00", "01", "02", "03", "11", "12", "21", "22", "31", "32", "41", "42", "51", "52", "53"]


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


# --------------------------- Scope ----------------------------------------- #
@dataclass
class Scope:
    """Represents a query scope with optional filters."""
    departement: Optional[str] = None
    code_postal: Optional[str] = None
    section_activite_principale: Optional[str] = None
    activite_principale: Optional[str] = None
    tranche_effectif_salarie: Optional[str] = None
    
    @property
    def key(self) -> str:
        """Unique key for progress tracking."""
        parts = []
        if self.departement:
            parts.append(f"D:{self.departement}")
        if self.code_postal:
            parts.append(f"CP:{self.code_postal}")
        if self.section_activite_principale:
            parts.append(f"S:{self.section_activite_principale}")
        if self.activite_principale:
            parts.append(f"NAF:{self.activite_principale}")
        if self.tranche_effectif_salarie:
            parts.append(f"TE:{self.tranche_effectif_salarie}")
        return "|".join(parts) if parts else "ALL"
    
    def to_params(self) -> Dict[str, Any]:
        """Convert to API query params."""
        params = {}
        if self.departement:
            params["departement"] = self.departement
        if self.code_postal:
            params["code_postal"] = self.code_postal
        if self.section_activite_principale:
            params["section_activite_principale"] = self.section_activite_principale
        if self.activite_principale:
            params["activite_principale"] = self.activite_principale
        if self.tranche_effectif_salarie:
            params["tranche_effectif_salarie"] = self.tranche_effectif_salarie
        return params
    
    @property
    def depth(self) -> int:
        """Return drilling depth (0-4)."""
        d = 0
        if self.departement: d += 1
        if self.code_postal: d += 1
        if self.section_activite_principale: d += 1
        if self.activite_principale: d += 1
        if self.tranche_effectif_salarie: d += 1
        return d


# --------------------------- DB helpers ------------------------------------ #
def ensure_db(conn: sqlite3.Connection) -> None:
    # Entreprises table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entreprises (
          siren TEXT PRIMARY KEY,
          nom_complet TEXT,
          nature_juridique TEXT,
          activite_principale TEXT,
          section_activite_principale TEXT,
          categorie_entreprise TEXT,
          tranche_effectif TEXT,
          etat_administratif TEXT,
          date_creation TEXT,
          date_capture TEXT DEFAULT (datetime('now'))
        );
    """)
    
    # Dirigeants table (both PP and PM)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dirigeants (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          siren_entreprise TEXT,
          type_dirigeant TEXT,
          -- PM fields
          siren_dirigeant TEXT,
          denomination TEXT,
          -- PP fields
          nom TEXT,
          prenoms TEXT,
          annee_naissance TEXT,
          nationalite TEXT,
          -- Common
          qualite TEXT,
          date_capture TEXT DEFAULT (datetime('now')),
          UNIQUE(siren_entreprise, type_dirigeant, siren_dirigeant, nom, prenoms, qualite)
        );
    """)
    
    # Progress table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_progress (
          scope_key TEXT PRIMARY KEY,
          total_results INTEGER,
          completed INTEGER DEFAULT 0
        );
    """)
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ent_siren ON entreprises(siren);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dir_siren_ent ON dirigeants(siren_entreprise);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dir_siren_dir ON dirigeants(siren_dirigeant);")
    conn.commit()


def is_completed(conn: sqlite3.Connection, scope: Scope) -> bool:
    cur = conn.execute("SELECT completed FROM fetch_progress WHERE scope_key=?", (scope.key,))
    row = cur.fetchone()
    return row is not None and row[0] == 1


def mark_completed(conn: sqlite3.Connection, scope: Scope, total_results: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO fetch_progress (scope_key, total_results, completed) VALUES (?, ?, 1)",
        (scope.key, total_results)
    )
    conn.commit()


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
            pass  # Duplicate, skip
    return count


# --------------------------- HTTP ------------------------------------------ #
class APIClient:
    def __init__(self, limiter: RateLimiter):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "Sireto-Harvest/2.0",
        })
        self.limiter = limiter
    
    def fetch(self, scope: Scope, page: int = 1, per_page: int = 1) -> Optional[dict]:
        """Fetch a page for the given scope."""
        params = scope.to_params()
        params["page"] = page
        params["per_page"] = per_page
        params["minimal"] = True
        params["include"] = "dirigeants"
        
        for attempt in range(MAX_RETRIES):
            self.limiter.acquire()
            try:
                resp = self.session.get(BASE_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    delay = float(resp.headers.get("Retry-After", 3.0))
                    LOGGER.warning("429, sleeping %.1fs", delay)
                    time.sleep(delay)
                    continue
                if resp.status_code == 400:
                    LOGGER.warning("400 Bad Request on %s - skipping", scope.key)
                    return None
                if resp.status_code in (500, 502, 503, 504):
                    time.sleep(BACKOFF_BASE * (2 ** attempt))
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                LOGGER.warning("Timeout on %s, retrying...", scope.key)
                time.sleep(BACKOFF_BASE * (2 ** attempt))
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    LOGGER.error("Failed %s: %s", scope.key, e)
                    return None
                time.sleep(BACKOFF_BASE * (2 ** attempt))
        return None
    
    def get_total_results(self, scope: Scope) -> int:
        """Get total_results count for a scope (quick probe with per_page=1)."""
        payload = self.fetch(scope, page=1, per_page=1)
        if payload:
            return payload.get("total_results", 0)
        return 0


# --------------------------- Core Logic ------------------------------------ #
def process_results(conn: sqlite3.Connection, payload: dict) -> tuple:
    """Extract and save all data from API response. Returns (ent_count, dir_count)."""
    ent_count = 0
    dir_count = 0
    
    for res in payload.get("results", []) or []:
        siren = res.get("siren")
        if not siren:
            continue
        
        # Save enterprise
        upsert_entreprise(conn, res)
        ent_count += 1
        
        # Save all dirigeants
        dir_count += upsert_dirigeants(conn, siren, res.get("dirigeants"))
    
    conn.commit()
    return ent_count, dir_count


def download_all_pages(client: APIClient, conn: sqlite3.Connection, scope: Scope, 
                       total_results: int, pbar: tqdm) -> tuple:
    """Download all pages for a scope that has <10k results."""
    total_pages = min((total_results + PER_PAGE - 1) // PER_PAGE, MAX_PAGES)
    ent_count = 0
    dir_count = 0
    
    for page in range(1, total_pages + 1):
        payload = client.fetch(scope, page=page, per_page=PER_PAGE)
        if not payload:
            break
        e, d = process_results(conn, payload)
        ent_count += e
        dir_count += d
        pbar.update(1)
        
        if not payload.get("results"):
            break
    
    return ent_count, dir_count


def get_codes_postaux_for_departement(dep: str) -> List[str]:
    """Generate plausible postal codes for a département."""
    codes = []
    if len(dep) == 2:
        for i in range(1000):
            codes.append(f"{dep}{i:03d}")
    elif len(dep) == 3:  # DOM-TOM
        for i in range(100):
            codes.append(f"{dep}{i:02d}")
    return codes


def get_naf_codes_from_results(client: APIClient, scope: Scope) -> List[str]:
    """Scan pages to discover all NAF codes present in results."""
    naf_codes = set()
    total = client.get_total_results(scope)
    pages_to_scan = min((total + PER_PAGE - 1) // PER_PAGE, MAX_PAGES)
    
    for page in range(1, pages_to_scan + 1):
        payload = client.fetch(scope, page=page, per_page=PER_PAGE)
        if not payload or not payload.get("results"):
            break
        for res in payload.get("results", []):
            naf = res.get("activite_principale")
            if naf:
                naf_codes.add(naf)
    
    return sorted(naf_codes)


def fetch_recursive(client: APIClient, conn: sqlite3.Connection, scope: Scope, 
                    pbar: tqdm, stats: dict) -> None:
    """Recursive drill-down fetch."""
    # Skip if already completed
    if is_completed(conn, scope):
        return
    
    # Get total results for this scope
    total = client.get_total_results(scope)
    
    if total == 0:
        mark_completed(conn, scope, 0)
        return
    
    # If under limit, download directly
    if total < MAX_RESULTS:
        ent, dir = download_all_pages(client, conn, scope, total, pbar)
        mark_completed(conn, scope, total)
        stats["ent_total"] += ent
        stats["dir_total"] += dir
        stats["scopes_completed"] += 1
        pbar.set_postfix({"Ent": stats["ent_total"], "Dir": stats["dir_total"]})
        return
    
    # Need to drill down
    # Level 1 → 2: département → code_postal
    if scope.departement and not scope.code_postal:
        LOGGER.info("🔀 DRILL-DOWN: %s has %d results, splitting by code_postal", scope.key, total)
        for cp in get_codes_postaux_for_departement(scope.departement):
            new_scope = Scope(
                departement=scope.departement,
                code_postal=cp,
                section_activite_principale=scope.section_activite_principale,
                activite_principale=scope.activite_principale,
                tranche_effectif_salarie=scope.tranche_effectif_salarie,
            )
            fetch_recursive(client, conn, new_scope, pbar, stats)
        mark_completed(conn, scope, total)
        return
    
    # Level 2 → 3: code_postal → section NAF
    if not scope.section_activite_principale and not scope.activite_principale:
        LOGGER.info("🔀 DRILL-DOWN: %s has %d results, splitting by NAF section", scope.key, total)
        for section in NAF_SECTIONS:
            new_scope = Scope(
                departement=scope.departement,
                code_postal=scope.code_postal,
                section_activite_principale=section,
                activite_principale=scope.activite_principale,
                tranche_effectif_salarie=scope.tranche_effectif_salarie,
            )
            fetch_recursive(client, conn, new_scope, pbar, stats)
        mark_completed(conn, scope, total)
        return
    
    # Level 3 → 4: section NAF → activite_principale
    if scope.section_activite_principale and not scope.activite_principale:
        LOGGER.info("🔀 DRILL-DOWN: %s has %d results, discovering NAF codes", scope.key, total)
        naf_codes = get_naf_codes_from_results(client, scope)
        LOGGER.info("Found %d NAF codes in %s", len(naf_codes), scope.key)
        for naf in naf_codes:
            new_scope = Scope(
                departement=scope.departement,
                code_postal=scope.code_postal,
                section_activite_principale=None,
                activite_principale=naf,
                tranche_effectif_salarie=scope.tranche_effectif_salarie,
            )
            fetch_recursive(client, conn, new_scope, pbar, stats)
        mark_completed(conn, scope, total)
        return
    
    # Level 4 → 5: activite_principale → tranche_effectif
    if scope.activite_principale and not scope.tranche_effectif_salarie:
        LOGGER.info("🔀 DRILL-DOWN: %s has %d results, splitting by tranche_effectif", scope.key, total)
        for tranche in TRANCHES_EFFECTIF:
            new_scope = Scope(
                departement=scope.departement,
                code_postal=scope.code_postal,
                section_activite_principale=scope.section_activite_principale,
                activite_principale=scope.activite_principale,
                tranche_effectif_salarie=tranche,
            )
            fetch_recursive(client, conn, new_scope, pbar, stats)
        mark_completed(conn, scope, total)
        return
    
    # If we get here, we've exhausted all levels and still have >10k
    LOGGER.error("❌ CANNOT DRILL FURTHER: %s has %d results - DATA LOSS!", scope.key, total)
    ent, dir = download_all_pages(client, conn, scope, total, pbar)
    mark_completed(conn, scope, total)
    stats["ent_total"] += ent
    stats["dir_total"] += dir
    stats["scopes_completed"] += 1


# --------------------------- Main ------------------------------------------ #
def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    ensure_db(conn)
    
    limiter = RateLimiter(REQUEST_INTERVAL)
    client = APIClient(limiter)
    
    LOGGER.info("Starting recursive drill-down harvest (FULL DATA)...")
    LOGGER.info("Départements to process: %d", len(DEPARTEMENTS))
    
    stats = {"ent_total": 0, "dir_total": 0, "scopes_completed": 0}
    
    pbar = tqdm(total=len(DEPARTEMENTS) * 100, desc="Pages", smoothing=0.1)
    
    for dep in DEPARTEMENTS:
        scope = Scope(departement=dep)
        fetch_recursive(client, conn, scope, pbar, stats)
    
    pbar.close()
    
    # Final report
    cur = conn.execute("SELECT COUNT(*) FROM entreprises")
    total_ent = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM dirigeants")
    total_dir = cur.fetchone()[0]
    LOGGER.info("Harvest complete! Entreprises: %d, Dirigeants: %d", total_ent, total_dir)
    conn.close()


if __name__ == "__main__":
    main()
