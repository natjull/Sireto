#!/usr/bin/env python3
"""
Recovery script for `scripts/harvest_full.py` output.

Goal: recover *everything that is recoverable via the API* but missing from
`data/harvest_full.sqlite`, excluding non-diffusible entities ("P") which do not
exist in the API (per OpenAPI docs).

Workflow:
  1) Prepare a list of diffusible missing SIRENs ("O") by intersecting:
       - `data/reports/missing_sirens.txt` (Parquet - SQLite)
       - `data/StockUniteLegale_utf8.parquet` diffusion status
     Output: `data/reports/missing_sirens_diffusible_O.txt`
  2) Bulk recovery for what `harvest_full.py` can miss in practice:
       - resume `scopes_in_progress`
       - re-run "dense" parent scopes deterministically (avoid NAF discovery based on
         the first 10k results)
  3) Final safety net: for remaining SIRENs, query `/search?q=<siren>` (per_page=1)
     and upsert the full payload into the same SQLite schema as `harvest_full.py`.

This is slower than scope-based harvesting, but it is the most robust way to
guarantee we fill whatever `harvest_full.py` could not retrieve (truncated
scopes, transient errors, etc.) without missing any individual SIREN.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from tqdm import tqdm


LOG = logging.getLogger(__name__)


API_URL = "https://recherche-entreprises.api.gouv.fr/search"


def load_harvest_full_module(path: Path):
    spec = importlib.util.spec_from_file_location("harvest_full", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure the module is registered before executing it.
    # dataclasses (Py3.12+) may consult sys.modules during decoration.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def prepare_diffusible_missing_list(
    missing_sirens_path: Path,
    ul_parquet_path: Path,
    output_path: Path,
) -> int:
    """
    Create `output_path` containing only missing SIRENs with diffusion == 'O'.

    Assumptions:
      - `missing_sirens_path` is sorted (it is written sorted by compare script)
      - `ul_parquet_path` is sorted by `siren` (observed in this dataset)
    """
    if not missing_sirens_path.exists():
        raise FileNotFoundError(missing_sirens_path)
    if not ul_parquet_path.exists():
        raise FileNotFoundError(ul_parquet_path)

    pf = pq.ParquetFile(str(ul_parquet_path))

    missing_f = missing_sirens_path.open("r", encoding="utf-8")

    def next_missing() -> Optional[str]:
        while True:
            line = missing_f.readline()
            if not line:
                return None
            s = line.strip()
            if s:
                return s

    current = next_missing()
    count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for rg in tqdm(range(pf.num_row_groups), desc="Prepare list (row groups)"):
            if current is None:
                break

            t = pf.read_row_group(
                rg,
                columns=["siren", "statutDiffusionUniteLegale"],
            )
            siren_arr = t.column("siren")
            if len(siren_arr) == 0:
                continue
            min_s = siren_arr[0].as_py()
            max_s = siren_arr[len(siren_arr) - 1].as_py()

            # Advance current up to min_s (should be uncommon)
            if current < min_s:
                while current is not None and current < min_s:
                    current = next_missing()
                if current is None:
                    break

            # Skip row groups that are entirely below current
            if current > max_s:
                continue

            group_missing: list[str] = []
            while current is not None and current <= max_s:
                if current >= min_s:
                    group_missing.append(current)
                current = next_missing()

            if not group_missing:
                continue

            value_set = pa.array(group_missing, type=pa.string())
            mask = pc.is_in(siren_arr, value_set=value_set)
            if pc.sum(mask).as_py() == 0:
                continue

            sirens_hit = pc.filter(siren_arr, mask)
            diffusion_hit = pc.filter(t.column("statutDiffusionUniteLegale"), mask)
            mask_o = pc.equal(diffusion_hit, "O")
            if pc.sum(mask_o).as_py() == 0:
                continue

            sirens_o = pc.filter(sirens_hit, mask_o).to_pylist()
            for s in sirens_o:
                out.write(str(s) + "\n")
            count += len(sirens_o)

    missing_f.close()
    return count


@dataclass
class RateLimiter:
    interval: float
    last_request: float = 0.0

    def acquire(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request = time.monotonic()


def init_progress_table(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            siren TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def load_done_set(conn: sqlite3.Connection, table: str, statuses: tuple[str, ...]) -> set[str]:
    placeholders = ",".join("?" for _ in statuses)
    rows = conn.execute(
        f"SELECT siren FROM {table} WHERE status IN ({placeholders})",
        statuses,
    ).fetchall()
    return {r[0] for r in rows}


def upsert_progress(
    conn: sqlite3.Connection,
    table: str,
    siren: str,
    status: str,
    attempts: int,
    last_error: Optional[str] = None,
) -> None:
    conn.execute(
        f"""
        INSERT INTO {table} (siren, status, attempts, last_error, updated_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(siren) DO UPDATE SET
            status=excluded.status,
            attempts=excluded.attempts,
            last_error=excluded.last_error,
            updated_at=excluded.updated_at
        """,
        (siren, status, attempts, last_error),
    )


def init_scope_progress_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recovery_scopes_in_progress (
            scope_key TEXT PRIMARY KEY,
            total_results INTEGER NOT NULL,
            last_page INTEGER NOT NULL,
            started_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recovery_scopes_completed (
            scope_key TEXT PRIMARY KEY,
            total_results INTEGER NOT NULL,
            pages_fetched INTEGER NOT NULL,
            completed_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def mark_scope_in_progress(
    conn: sqlite3.Connection, scope_key: str, total_results: int, last_page: int
) -> None:
    conn.execute(
        """
        INSERT INTO recovery_scopes_in_progress (scope_key, total_results, last_page, started_at, updated_at)
        VALUES (?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(scope_key) DO UPDATE SET
            total_results=excluded.total_results,
            last_page=excluded.last_page,
            updated_at=excluded.updated_at
        """,
        (scope_key, total_results, last_page),
    )


def mark_scope_completed(conn: sqlite3.Connection, scope_key: str, total_results: int, pages_fetched: int) -> None:
    conn.execute("DELETE FROM recovery_scopes_in_progress WHERE scope_key=?", (scope_key,))
    conn.execute(
        """
        INSERT INTO recovery_scopes_completed (scope_key, total_results, pages_fetched, completed_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(scope_key) DO UPDATE SET
            total_results=excluded.total_results,
            pages_fetched=excluded.pages_fetched,
            completed_at=excluded.completed_at
        """,
        (scope_key, total_results, pages_fetched),
    )


def get_scope_progress(conn: sqlite3.Connection, scope_key: str) -> Optional[tuple[int, int]]:
    row = conn.execute(
        "SELECT total_results, last_page FROM recovery_scopes_in_progress WHERE scope_key=?",
        (scope_key,),
    ).fetchone()
    if row:
        return int(row[0]), int(row[1])
    return None


def is_scope_completed(conn: sqlite3.Connection, scope_key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM recovery_scopes_completed WHERE scope_key=?",
        (scope_key,),
    ).fetchone()
    return row is not None


def parse_scope_key(scope_key: str) -> Dict[str, Any]:
    """
    Parse harvest_full.py-style keys into API query params.

    Supports:
      D: departement
      CP: code_postal
      S: section_activite_principale
      NAF: activite_principale
      TE: tranche_effectif_salarie
      EA: etat_administratif
      NJ: nature_juridique
      EI: est_entrepreneur_individuel (recovery-only)
    """
    params: Dict[str, Any] = {}
    for part in scope_key.split("|"):
        if part.startswith("D:"):
            params["departement"] = part[2:]
        elif part.startswith("CP:"):
            params["code_postal"] = part[3:]
        elif part.startswith("S:"):
            params["section_activite_principale"] = part[2:]
        elif part.startswith("NAF:"):
            params["activite_principale"] = part[4:]
        elif part.startswith("TE:"):
            params["tranche_effectif_salarie"] = part[3:]
        elif part.startswith("EA:"):
            params["etat_administratif"] = part[3:]
        elif part.startswith("NJ:"):
            params["nature_juridique"] = part[3:]
        elif part.startswith("EI:"):
            v = part[3:].strip().lower()
            params["est_entrepreneur_individuel"] = v in ("1", "true", "yes", "o")
    return params


def build_scope_key(base_key: str, extra: str) -> str:
    return f"{base_key}|{extra}" if base_key else extra


def api_get_total(
    session: requests.Session,
    limiter: RateLimiter,
    params: Dict[str, Any],
    *,
    timeout_s: float,
    max_retries: int,
) -> int:
    probe = dict(params)
    probe.update({"page": 1, "per_page": 1, "minimal": True})

    for attempt in range(max_retries):
        limiter.acquire()
        try:
            resp = session.get(API_URL, params=probe, timeout=timeout_s)
            if resp.status_code == 429:
                delay = float(resp.headers.get("Retry-After", 5.0))
                time.sleep(delay)
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(1.0 * (2**attempt))
                continue
            if resp.status_code == 400:
                return 0
            resp.raise_for_status()
            return int(resp.json().get("total_results") or 0)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(1.0 * (2**attempt))
            continue
        except requests.exceptions.RequestException:
            time.sleep(1.0 * (2**attempt))
            continue
        except Exception:
            time.sleep(1.0 * (2**attempt))
            continue
    return 0


def api_fetch_page(
    session: requests.Session,
    limiter: RateLimiter,
    params: Dict[str, Any],
    page: int,
    *,
    timeout_s: float,
    max_retries: int,
) -> Optional[dict]:
    query = dict(params)
    query.update(
        {
            "page": page,
            "per_page": 25,
            "limite_matching_etablissements": 100,
            # no minimal=True: we want the full payload like harvest_full.py
        }
    )
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            resp = session.get(API_URL, params=query, timeout=timeout_s)
            if resp.status_code == 429:
                delay = float(resp.headers.get("Retry-After", 5.0))
                time.sleep(delay)
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(1.0 * (2**attempt))
                continue
            if resp.status_code == 400:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            time.sleep(1.0 * (2**attempt))
        except requests.exceptions.ConnectionError:
            time.sleep(1.0 * (2**attempt))
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(1.0 * (2**attempt))
    return None


def process_scope_pages(
    *,
    conn: sqlite3.Connection,
    save_entreprise,
    session: requests.Session,
    limiter: RateLimiter,
    scope_key: str,
    params: Dict[str, Any],
    total_results: int,
    timeout_s: float,
    max_retries: int,
    commit_every_pages: int,
) -> None:
    total_pages = min((total_results + 24) // 25, 400)
    progress = get_scope_progress(conn, scope_key)
    start_page = (progress[1] + 1) if progress else 1

    pages_fetched = start_page - 1 if start_page > 1 else 0
    for page in range(start_page, total_pages + 1):
        payload = api_fetch_page(
            session,
            limiter,
            params,
            page,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        if not payload or not payload.get("results"):
            mark_scope_in_progress(conn, scope_key, total_results, page - 1)
            conn.commit()
            raise RuntimeError(f"Failed to fetch page {page} for {scope_key}")

        for ent in payload.get("results") or []:
            save_entreprise(conn, ent)

        pages_fetched = page
        mark_scope_in_progress(conn, scope_key, total_results, page)
        if page % commit_every_pages == 0:
            conn.commit()

    mark_scope_completed(conn, scope_key, total_results, pages_fetched)
    conn.commit()


def load_unique_naf_codes(ul_parquet_path: Path) -> list[str]:
    table = pq.read_table(str(ul_parquet_path), columns=["activitePrincipaleUniteLegale"])
    unique = pc.unique(table.column(0))
    values = [v.as_py() for v in unique if v is not None]
    return sorted({str(v).strip() for v in values if str(v).strip() and str(v).strip() != "None"})


def naf_section_from_code(naf: str) -> Optional[str]:
    """
    Approximate NAF Rev.2 section (A..U) from a NAF code like '70.22Z'.
    Mapping is based on the 2-digit division (01..99):
      A: 01-03, B: 05-09, C: 10-33, D: 35, E: 36-39, F: 41-43,
      G: 45-47, H: 49-53, I: 55-56, J: 58-63, K: 64-66, L: 68,
      M: 69-75, N: 77-82, O: 84, P: 85, Q: 86-88, R: 90-93,
      S: 94-96, T: 97-98, U: 99
    """
    m = re.match(r"^(\d{2})", str(naf))
    if not m:
        return None
    div = int(m.group(1))
    if 1 <= div <= 3:
        return "A"
    if 5 <= div <= 9:
        return "B"
    if 10 <= div <= 33:
        return "C"
    if div == 35:
        return "D"
    if 36 <= div <= 39:
        return "E"
    if 41 <= div <= 43:
        return "F"
    if 45 <= div <= 47:
        return "G"
    if 49 <= div <= 53:
        return "H"
    if 55 <= div <= 56:
        return "I"
    if 58 <= div <= 63:
        return "J"
    if 64 <= div <= 66:
        return "K"
    if div == 68:
        return "L"
    if 69 <= div <= 75:
        return "M"
    if 77 <= div <= 82:
        return "N"
    if div == 84:
        return "O"
    if div == 85:
        return "P"
    if 86 <= div <= 88:
        return "Q"
    if 90 <= div <= 93:
        return "R"
    if 94 <= div <= 96:
        return "S"
    if 97 <= div <= 98:
        return "T"
    if div == 99:
        return "U"
    return None


def group_naf_by_section(naf_codes: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for code in naf_codes:
        section = naf_section_from_code(code)
        if not section:
            continue
        grouped.setdefault(section, []).append(code)
    for section in grouped:
        grouped[section].sort()
    return grouped


def load_unique_nature_juridique_codes(ul_parquet_path: Path) -> list[str]:
    table = pq.read_table(str(ul_parquet_path), columns=["categorieJuridiqueUniteLegale"])
    unique = pc.unique(table.column(0))
    values = [v.as_py() for v in unique if v is not None]
    # Stored as int64 in parquet
    return sorted({str(int(v)) for v in values})


def bulk_recover_dense_scopes(
    *,
    conn: sqlite3.Connection,
    save_entreprise,
    session: requests.Session,
    limiter: RateLimiter,
    ul_parquet_path: Path,
    timeout_s: float,
    total_retries: int,
    max_retries: int,
    commit_every_pages: int,
    max_scopes: Optional[int],
) -> None:
    init_scope_progress_tables(conn)

    # Resume original in-progress scopes (best effort).
    in_prog = conn.execute(
        "SELECT scope_key, total_results, last_page FROM scopes_in_progress"
    ).fetchall()
    if in_prog:
        LOG.info("Resuming %d original scopes_in_progress...", len(in_prog))
    for scope_key, total_results, last_page in in_prog:
        scope_key = str(scope_key)
        params = parse_scope_key(scope_key)
        total = (
            int(total_results)
            if int(total_results) > 0
            else api_get_total(
                session,
                limiter,
                params,
                timeout_s=timeout_s,
                max_retries=total_retries,
            )
        )
        mark_scope_in_progress(conn, scope_key, total, int(last_page))
        try:
            process_scope_pages(
                conn=conn,
                save_entreprise=save_entreprise,
                session=session,
                limiter=limiter,
                scope_key=scope_key,
                params=params,
                total_results=total,
                timeout_s=timeout_s,
                max_retries=max_retries,
                commit_every_pages=commit_every_pages,
            )
        except Exception as e:
            LOG.warning("Failed to resume %s: %s", scope_key, e)

    parent_scopes = conn.execute(
        """
        SELECT scope_key
        FROM scopes_completed
        WHERE total_results >= 10000
          AND pages_fetched = 0
          AND scope_key LIKE '%EA:%'
          AND scope_key NOT LIKE '%NAF:%'
        ORDER BY scope_key
        """
    ).fetchall()
    parent_keys = [str(r[0]) for r in parent_scopes]
    if max_scopes is not None:
        parent_keys = parent_keys[:max_scopes]

    if parent_keys:
        LOG.info("Dense parent scopes to re-run with deterministic NAF enumeration: %d", len(parent_keys))

    LOG.info("Loading NAF and legal nature dictionaries from Parquet (one-time)...")
    naf_codes = load_unique_naf_codes(ul_parquet_path)
    naf_by_section = group_naf_by_section(naf_codes)
    nj_codes = load_unique_nature_juridique_codes(ul_parquet_path)
    LOG.info("NAF unique codes: %d; CJ unique codes: %d", len(naf_codes), len(nj_codes))

    unshardable: list[dict[str, Any]] = []

    for base_scope_key in tqdm(parent_keys, desc="Dense scopes"):
        base_params = parse_scope_key(base_scope_key)
        section = base_params.get("section_activite_principale")
        naf_iter = naf_by_section.get(str(section), []) if section else naf_codes
        if not naf_iter:
            # Fallback to full list if section mapping failed.
            naf_iter = naf_codes
        tqdm.write(f"[bulk] {base_scope_key} -> trying {len(naf_iter)} NAF codes")
        naf_checked = 0

        for naf in naf_iter:
            child_key = build_scope_key(base_scope_key, f"NAF:{naf}")
            if is_scope_completed(conn, child_key):
                continue

            child_params = dict(base_params)
            child_params["activite_principale"] = naf

            naf_checked += 1
            if naf_checked % 50 == 0:
                tqdm.write(f"[bulk] {base_scope_key} totals checked: {naf_checked}/{len(naf_iter)}")

            total = api_get_total(
                session,
                limiter,
                child_params,
                timeout_s=timeout_s,
                max_retries=total_retries,
            )
            if total == 0:
                continue

            if total < 10000:
                try:
                    process_scope_pages(
                        conn=conn,
                        save_entreprise=save_entreprise,
                        session=session,
                        limiter=limiter,
                        scope_key=child_key,
                        params=child_params,
                        total_results=total,
                        timeout_s=timeout_s,
                        max_retries=max_retries,
                        commit_every_pages=commit_every_pages,
                    )
                except Exception as e:
                    LOG.warning("Scope failed %s: %s", child_key, e)
                continue

            # Still too dense: EI split.
            for ei in (True, False):
                ei_key = build_scope_key(child_key, f"EI:{1 if ei else 0}")
                if is_scope_completed(conn, ei_key):
                    continue
                ei_params = dict(child_params)
                ei_params["est_entrepreneur_individuel"] = ei
                total_ei = api_get_total(
                    session,
                    limiter,
                    ei_params,
                    timeout_s=timeout_s,
                    max_retries=total_retries,
                )
                if total_ei == 0:
                    mark_scope_completed(conn, ei_key, 0, 0)
                    conn.commit()
                    continue
                if total_ei < 10000:
                    try:
                        process_scope_pages(
                            conn=conn,
                            save_entreprise=save_entreprise,
                            session=session,
                            limiter=limiter,
                            scope_key=ei_key,
                            params=ei_params,
                            total_results=total_ei,
                            timeout_s=timeout_s,
                            max_retries=max_retries,
                            commit_every_pages=commit_every_pages,
                        )
                    except Exception as e:
                        LOG.warning("Scope failed %s: %s", ei_key, e)
                    continue

                # Still too dense: shard by nature_juridique.
                any_nj = False
                for nj in nj_codes:
                    nj_key = build_scope_key(ei_key, f"NJ:{nj}")
                    if is_scope_completed(conn, nj_key):
                        continue
                    nj_params = dict(ei_params)
                    nj_params["nature_juridique"] = nj
                    total_nj = api_get_total(
                        session,
                        limiter,
                        nj_params,
                        timeout_s=timeout_s,
                        max_retries=total_retries,
                    )
                    if total_nj == 0:
                        continue
                    any_nj = True
                    if total_nj < 10000:
                        try:
                            process_scope_pages(
                                conn=conn,
                                save_entreprise=save_entreprise,
                                session=session,
                                limiter=limiter,
                                scope_key=nj_key,
                                params=nj_params,
                                total_results=total_nj,
                                timeout_s=timeout_s,
                                max_retries=max_retries,
                                commit_every_pages=commit_every_pages,
                            )
                        except Exception as e:
                            LOG.warning("Scope failed %s: %s", nj_key, e)
                    else:
                        unshardable.append({"scope_key": nj_key, "total_results": total_nj})

                if not any_nj:
                    unshardable.append({"scope_key": ei_key, "total_results": total_ei})

    if unshardable:
        out = Path("data/reports/recovery_unshardable_scopes.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for row in unshardable:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        LOG.warning("Unshardable scopes encountered: %d (logged to %s)", len(unshardable), out)


def fetch_siren_payload(
    session: requests.Session,
    limiter: RateLimiter,
    siren: str,
    *,
    max_retries: int,
) -> Optional[dict]:
    """
    Fetch one SIREN from the API. Returns the matching result dict (enterprise),
    or None if not found / unrecoverable.
    """
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            params = {
                "q": siren,
                "page": 1,
                "per_page": 1,
                "limite_matching_etablissements": 100,
                # no "minimal": we want the full payload like harvest_full.py
            }
            resp = session.get(API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                delay = float(resp.headers.get("Retry-After", 5.0))
                time.sleep(delay)
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(1.0 * (2**attempt))
                continue
            if resp.status_code == 400:
                return None
            resp.raise_for_status()
            data = resp.json()
            for res in data.get("results") or []:
                if res.get("siren") == siren:
                    return res
            return None
        except requests.exceptions.Timeout:
            time.sleep(1.0 * (2**attempt))
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(1.0 * (2**attempt))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover missing (diffusible) SIRENs into harvest_full.sqlite"
    )
    parser.add_argument("--db-path", type=Path, default=Path("data/harvest_full.sqlite"))
    parser.add_argument(
        "--harvest-full-script",
        type=Path,
        default=Path("scripts/harvest_full.py"),
        help="Path to harvest_full.py (used to reuse its DB insert logic).",
    )
    parser.add_argument(
        "--ul-parquet",
        type=Path,
        default=Path("data/StockUniteLegale_utf8.parquet"),
    )
    parser.add_argument(
        "--missing-sirens",
        type=Path,
        default=Path("data/reports/missing_sirens.txt"),
        help="Input list produced by scripts/compare_sirens.py (sorted).",
    )
    parser.add_argument(
        "--recoverable-sirens",
        type=Path,
        default=Path("data/reports/missing_sirens_diffusible_O.txt"),
        help="Generated list of missing diffusible SIRENs (O).",
    )
    parser.add_argument(
        "--mode",
        choices=("prepare", "bulk", "fallback", "run", "all"),
        default="all",
        help="prepare=list generation, bulk=scope recovery, fallback=per-siren safety net, run=bulk+fallback, all=prepare+run",
    )
    parser.add_argument("--rate", type=float, default=7.0, help="Max requests per second.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--total-retries", type=int, default=4, help="Retries for total_results probes in bulk mode.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout (seconds).")
    parser.add_argument("--commit-every", type=int, default=100, help="Commit every N SIRENs in fallback mode.")
    parser.add_argument("--commit-every-pages", type=int, default=5, help="Commit every N pages in bulk mode.")
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-bulk-scopes", type=int, default=None, help="Limit number of dense scopes (for testing).")
    parser.add_argument(
        "--progress-table",
        type=str,
        default="recovery_progress_full",
        help="SQLite table to store recovery progress.",
    )
    parser.add_argument(
        "--retry-status",
        type=str,
        default="",
        help="Comma list of statuses to retry (e.g. failed,not_found).",
    )
    parser.add_argument(
        "--not-found-file",
        type=Path,
        default=Path("data/reports/recovery_harvest_full_not_found.txt"),
    )
    parser.add_argument(
        "--failed-file",
        type=Path,
        default=Path("data/reports/recovery_harvest_full_failed.txt"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.mode in ("prepare", "all"):
        if args.recoverable_sirens.exists():
            LOG.info("Recoverable list already exists: %s", args.recoverable_sirens)
        else:
            LOG.info("Preparing recoverable diffusible SIRENs list...")
            n = prepare_diffusible_missing_list(
                missing_sirens_path=args.missing_sirens,
                ul_parquet_path=args.ul_parquet,
                output_path=args.recoverable_sirens,
            )
            LOG.info("Prepared %d diffusible missing SIRENs: %s", n, args.recoverable_sirens)

    if args.mode in ("bulk", "run", "all"):
        if not args.db_path.exists():
            raise FileNotFoundError(args.db_path)

        harvest_full = load_harvest_full_module(args.harvest_full_script)
        conn = sqlite3.connect(str(args.db_path))
        harvest_full.init_db(conn)
        init_scope_progress_tables(conn)

        session = requests.Session()
        session.headers.update(
            {
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": "Sireto-RecoverHarvestFull-Bulk/1.0",
            }
        )
        limiter = RateLimiter(interval=1.0 / max(args.rate, 0.1))

        LOG.info("Starting bulk recovery (scopes)...")
        bulk_recover_dense_scopes(
            conn=conn,
            save_entreprise=harvest_full.save_entreprise,
            session=session,
            limiter=limiter,
            ul_parquet_path=args.ul_parquet,
            timeout_s=max(5.0, float(args.timeout)),
            total_retries=max(1, int(args.total_retries)),
            max_retries=args.max_retries,
            commit_every_pages=max(1, args.commit_every_pages),
            max_scopes=args.max_bulk_scopes,
        )
        conn.close()

    if args.mode in ("fallback", "run", "all"):
        if not args.db_path.exists():
            raise FileNotFoundError(args.db_path)
        if not args.recoverable_sirens.exists():
            raise FileNotFoundError(args.recoverable_sirens)

        harvest_full = load_harvest_full_module(args.harvest_full_script)

        conn = sqlite3.connect(str(args.db_path))
        harvest_full.init_db(conn)
        init_progress_table(conn, args.progress_table)

        retry_statuses = tuple(s.strip() for s in args.retry_status.split(",") if s.strip())
        done = load_done_set(conn, args.progress_table, ("done",))

        session = requests.Session()
        session.headers.update(
            {
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": "Sireto-RecoverHarvestFull-Fallback/1.0",
            }
        )
        limiter = RateLimiter(interval=1.0 / max(args.rate, 0.1))

        sirens = list(iter_lines(args.recoverable_sirens))
        if args.start_from:
            sirens = sirens[args.start_from :]
        if args.limit is not None:
            sirens = sirens[: args.limit]

        args.not_found_file.parent.mkdir(parents=True, exist_ok=True)
        args.failed_file.parent.mkdir(parents=True, exist_ok=True)

        not_found_f = args.not_found_file.open("a", encoding="utf-8")
        failed_f = args.failed_file.open("a", encoding="utf-8")

        processed = 0
        saved = 0

        try:
            for siren in tqdm(sirens, desc="Fallback per-SIREN"):
                if siren in done:
                    continue

                # If retry mode is enabled, skip SIRENs already processed with a non-retriable status.
                if retry_statuses:
                    row = conn.execute(
                        f"SELECT status FROM {args.progress_table} WHERE siren=?",
                        (siren,),
                    ).fetchone()
                    if row:
                        status = row[0]
                        if status == "done":
                            done.add(siren)
                            continue
                        if status not in retry_statuses:
                            continue

                # If already present in the DB (bulk recovery may have inserted it),
                # mark as done without an API call.
                if conn.execute(
                    "SELECT 1 FROM entreprises WHERE siren=? LIMIT 1",
                    (siren,),
                ).fetchone():
                    attempts_row = conn.execute(
                        f"SELECT attempts FROM {args.progress_table} WHERE siren=?",
                        (siren,),
                    ).fetchone()
                    attempts = int(attempts_row[0]) if attempts_row else 0
                    upsert_progress(conn, args.progress_table, siren, "done", attempts, None)
                    done.add(siren)
                    processed += 1
                    if processed % args.commit_every == 0:
                        conn.commit()
                    continue

                attempts_row = conn.execute(
                    f"SELECT attempts FROM {args.progress_table} WHERE siren=?",
                    (siren,),
                ).fetchone()
                attempts = int(attempts_row[0]) if attempts_row else 0
                attempts += 1

                ent = fetch_siren_payload(
                    session,
                    limiter,
                    siren,
                    max_retries=args.max_retries,
                )
                if ent is None:
                    upsert_progress(conn, args.progress_table, siren, "not_found", attempts, None)
                    not_found_f.write(siren + "\n")
                else:
                    try:
                        saved += int(harvest_full.save_entreprise(conn, ent) or 0)
                        upsert_progress(conn, args.progress_table, siren, "done", attempts, None)
                        done.add(siren)
                    except Exception as e:
                        err = str(e)
                        upsert_progress(conn, args.progress_table, siren, "failed", attempts, err[:1000])
                        failed_f.write(json.dumps({"siren": siren, "error": err}, ensure_ascii=False) + "\n")
                processed += 1

                if processed % args.commit_every == 0:
                    conn.commit()

        except KeyboardInterrupt:
            LOG.warning("Interrupted by user, committing progress...")
        finally:
            conn.commit()
            not_found_f.close()
            failed_f.close()
            conn.close()

        LOG.info("Fallback run complete: processed=%d saved=%d", processed, saved)


if __name__ == "__main__":
    main()
