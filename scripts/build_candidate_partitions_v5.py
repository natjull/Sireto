#!/usr/bin/env python3
"""
Build partitioned candidate store (v5) for fast per-commune loading.

v5 changes from v4:
  - Include closed establishments by default (--exclude-closed to disable)
  - Accept --scope-csv for production-scope (all CRM codes, not just training)
  - Output to data/candidates_v5_all/ by default

Outputs:
  data/candidates_v5_all/
    insee/   (partitioned by insee code)
    cp/      (partitioned by postcode)

This keeps only the columns needed by the matcher, with normalized names.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_TRAINING = Path("data/entrainements.csv")
DEFAULT_SCOPE = None  # Optional: use a different CSV for scope (all CRM codes)
DEFAULT_PARQUET = Path("data/StockEtablissement_utf8.parquet")
DEFAULT_UL = Path("data/StockUniteLegale_utf8.parquet")
DEFAULT_HARVEST_DB = Path("data/harvest_full.sqlite")
DEFAULT_OUT_DIR = Path("data/candidates_v5_all")
CODE_BATCH = 200


def load_scope_codes(training_path: Path, scope_path: Path | None) -> Tuple[Set[str], Set[str]]:
    """Load INSEE and postcode sets from training + optional scope CSV.
    
    If scope_path is provided, codes from both files are combined.
    This allows building partitions for all production codes, not just training.
    """
    def _load_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
        # Accept both raw CRM exports and audit_gt outputs
        df = df.rename(columns={
            "CODE_POSTAL": "postcode",
            "CODE_INSEE": "insee",
            "crm_cp": "postcode",
            "crm_insee": "insee",
        })
        if "postcode" not in df.columns or "insee" not in df.columns:
            raise ValueError(
                f"Missing postcode/insee columns in {path}. Found columns: {list(df.columns)}"
            )
        return df

    df = _load_csv(training_path)
    df["postcode"] = _normalize_codes(df["postcode"])
    df["insee"] = _normalize_codes(df["insee"])
    
    insee_codes: Set[str] = set(df["insee"].dropna().unique())
    postcodes: Set[str] = set(df["postcode"].dropna().unique())
    
    if scope_path and scope_path.exists():
        print(f"Loading additional scope from {scope_path}...")
        scope_df = _load_csv(scope_path)
        scope_df["postcode"] = _normalize_codes(scope_df["postcode"])
        scope_df["insee"] = _normalize_codes(scope_df["insee"])
        insee_codes |= set(scope_df["insee"].dropna().unique())
        postcodes |= set(scope_df["postcode"].dropna().unique())
    
    return insee_codes, postcodes


def load_training_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    df = df.rename(columns={
        "CODE_POSTAL": "postcode",
        "CODE_INSEE": "insee",
    })
    return df


def _normalize_codes(series: pd.Series) -> pd.Series:
    def _safe_convert(x):
        if pd.isna(x) or x in ["", "nan"]:
            return x
        try:
            return str(int(float(x)))
        except (ValueError, TypeError):
            return None  # Invalid code, will be filtered out
    return series.apply(_safe_convert)


def _parse_siege(val) -> bool:
    if isinstance(val, str):
        return val.strip().upper() in {"TRUE", "VRAI", "1", "OUI", "YES"}
    return bool(val)


def load_pm_dirigeant_names(
    sirens: Set[str],
    db_path: Path,
) -> Dict[str, List[str]]:
    if not db_path.exists() or not sirens:
        return {}
    out: Dict[str, Set[str]] = {s: set() for s in sirens}
    con = sqlite3.connect(str(db_path))
    siren_list = list(sirens)
    for i in range(0, len(siren_list), 900):
        chunk = siren_list[i : i + 900]
        placeholders = ",".join(["?"] * len(chunk))
        cur = con.execute(
            f"""
            SELECT siren, denomination
            FROM dirigeants
            WHERE siren IN ({placeholders})
              AND type_dirigeant = 'personne morale'
              AND denomination IS NOT NULL
            """,
            chunk,
        )
        for siren, denom in cur.fetchall():
            if denom and siren in out:
                out[siren].add(denom)
    con.close()
    return {k: sorted(v) for k, v in out.items() if v}


def _normalize_output(pdf: pd.DataFrame, pm_map: Dict[str, List[str]]) -> pd.DataFrame:
    pdf["siren"] = pdf["siren"].astype(str)
    pdf["is_siege"] = pdf["etablissementSiege"].apply(_parse_siege)
    pdf["pm_dirigeant_names"] = pdf["siren"].map(
        lambda s: "|".join(pm_map.get(str(s), [])) if s in pm_map else None
    )
    pdf = pdf.rename(
        columns={
            "etablissementSiege": "etablissementSiege",
        }
    )
    return pdf


def _query_etab_ul(
    con: duckdb.DuckDBPyConnection,
    etab_path: Path,
    ul_path: Path,
    codes: List[str],
    code_col: str,
    include_closed_establishments: bool,
) -> pd.DataFrame:
    code_df = pd.DataFrame({"code": codes})
    try:
        con.unregister("code_filter")
    except Exception:
        pass
    con.register("code_filter", code_df)
    closed_clause = ""
    if not include_closed_establishments:
        closed_clause = "AND (e.etatAdministratifEtablissement IS NULL OR e.etatAdministratifEtablissement != 'F')"
    query = f"""
        SELECT
            e.siret AS siret,
            e.siren AS siren,
            e.denominationUsuelleEtablissement AS denomination,
            e.enseigne1Etablissement AS enseigne1,
            e.enseigne2Etablissement AS enseigne2,
            e.enseigne3Etablissement AS enseigne3,
            e.etablissementSiege AS etablissementSiege,
            e.numeroVoieEtablissement AS numeroVoie,
            e.typeVoieEtablissement AS typeVoie,
            e.libelleVoieEtablissement AS libelleVoie,
            e.complementAdresseEtablissement AS complementAdresse,
            e.codePostalEtablissement AS postcode,
            e.libelleCommuneEtablissement AS city,
            e.codeCommuneEtablissement AS insee,
            ul.categorieJuridiqueUniteLegale AS cj_ul,
            e.etatAdministratifEtablissement AS etat_admin,
            e.dateDernierTraitementEtablissement AS last_treatment_date,
            ul.sigleUniteLegale AS sigle_ul,
            ul.denominationUniteLegale AS denomination_ul,
            TRIM(CONCAT_WS(' ',
                ul.denominationUsuelle1UniteLegale,
                ul.denominationUsuelle2UniteLegale,
                ul.denominationUsuelle3UniteLegale
            )) AS denomination_usuelle_ul,
            ul.nomUniteLegale AS nom_ul,
            ul.nomUsageUniteLegale AS nom_usage_ul,
            ul.prenomUsuelUniteLegale AS prenom_usuel_ul,
            ul.pseudonymeUniteLegale AS pseudonyme_ul
        FROM read_parquet('{etab_path.as_posix()}') e
        LEFT JOIN read_parquet('{ul_path.as_posix()}') ul
          ON e.siren = ul.siren
        WHERE e.{code_col} IN (SELECT code FROM code_filter)
          {closed_clause}
    """
    return con.execute(query).df()


def build_partitions(
    training_csv: Path,
    scope_csv: Path | None,
    parquet_path: Path,
    ul_path: Path,
    harvest_db: Path,
    output_dir: Path,
    code_batch: int,
    include_closed_establishments: bool,
) -> None:
    insee_codes, postcodes = load_scope_codes(training_csv, scope_csv)

    print(f"INSEE codes: {len(insee_codes)} | Postcodes: {len(postcodes)}")
    output_insee = output_dir / "insee"
    output_cp = output_dir / "cp"
    output_insee.mkdir(parents=True, exist_ok=True)
    output_cp.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    insee_list = sorted(insee_codes)
    cp_list = sorted(postcodes)

    for i in tqdm(range(0, len(insee_list), code_batch), desc="INSEE partitions"):
        batch = insee_list[i : i + code_batch]
        if not batch:
            continue
        pdf = _query_etab_ul(
            con,
            parquet_path,
            ul_path,
            batch,
            "codeCommuneEtablissement",
            include_closed_establishments,
        )
        if pdf.empty:
            continue
        pdf["postcode"] = _normalize_codes(pdf["postcode"])
        pdf["insee"] = _normalize_codes(pdf["insee"])
        sirens = set(pdf["siren"].dropna().astype(str).unique())
        pm_map = load_pm_dirigeant_names(sirens, harvest_db)
        pdf = _normalize_output(pdf, pm_map)
        table = pa.Table.from_pandas(pdf, preserve_index=False)
        pq.write_to_dataset(table, root_path=output_insee, partition_cols=["insee"])

    for i in tqdm(range(0, len(cp_list), code_batch), desc="CP partitions"):
        batch = cp_list[i : i + code_batch]
        if not batch:
            continue
        pdf = _query_etab_ul(
            con,
            parquet_path,
            ul_path,
            batch,
            "codePostalEtablissement",
            include_closed_establishments,
        )
        if pdf.empty:
            continue
        pdf["postcode"] = _normalize_codes(pdf["postcode"])
        pdf["insee"] = _normalize_codes(pdf["insee"])
        sirens = set(pdf["siren"].dropna().astype(str).unique())
        pm_map = load_pm_dirigeant_names(sirens, harvest_db)
        pdf = _normalize_output(pdf, pm_map)
        table = pa.Table.from_pandas(pdf, preserve_index=False)
        pq.write_to_dataset(table, root_path=output_cp, partition_cols=["postcode"])

    print(f"✅ Partitions written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate partitions v5 (includes closed by default).")
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING,
                        help="Training CSV for scope (required)")
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE,
                        help="Optional: additional CSV for production scope (combines codes)")
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--ul-path", type=Path, default=DEFAULT_UL)
    parser.add_argument("--harvest-db", type=Path, default=DEFAULT_HARVEST_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--code-batch", type=int, default=CODE_BATCH)
    parser.add_argument(
        "--exclude-closed",
        action="store_true",
        help="Exclude establishments with etatAdministratifEtablissement == 'F' (default: include all).",
    )
    args = parser.parse_args()

    build_partitions(
        training_csv=args.training_csv,
        scope_csv=args.scope_csv,
        parquet_path=args.parquet_path,
        ul_path=args.ul_path,
        harvest_db=args.harvest_db,
        output_dir=args.output_dir,
        code_batch=args.code_batch,
        include_closed_establishments=not args.exclude_closed,
    )


if __name__ == "__main__":
    main()
