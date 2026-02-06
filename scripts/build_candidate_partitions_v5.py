#!/usr/bin/env python3
"""
Build partitioned candidate store (v5) for fast per-commune loading.

v5 changes from v4:
  - Include closed establishments by default (--exclude-closed to disable)
  - Accept --scope-csv for production-scope (all CRM codes, not just training)
  - Output to data/candidates_v6_all/ by default

Outputs:
  data/candidates_v6_all/
    insee/   (partitioned by insee code)
    cp/      (partitioned by postcode)

This keeps only the columns needed by the matcher, with normalized names.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
import shutil
from typing import Dict, List, Set, Tuple, cast

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
DEFAULT_OUT_DIR = Path("data/candidates_v6_all")
CODE_BATCH = 200

OUTPUT_COLUMNS = [
    "siret",
    "siren",
    "denomination",
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "etablissementSiege",
    "is_siege",
    "numeroVoie",
    "typeVoie",
    "libelleVoie",
    "complementAdresse",
    "postcode",
    "city",
    "insee",
    "cj_ul",
    "etat_admin",
    "last_treatment_date",
    "sigle_ul",
    "denomination_ul",
    "denomination_usuelle_ul",
    "nom_ul",
    "prenom_usuel_ul",
    "pm_dirigeant_names",
]

OUTPUT_SCHEMA = pa.schema([
    ("siret", pa.string()),
    ("siren", pa.string()),
    ("denomination", pa.string()),
    ("enseigne1", pa.string()),
    ("enseigne2", pa.string()),
    ("enseigne3", pa.string()),
    ("etablissementSiege", pa.bool_()),
    ("is_siege", pa.bool_()),
    ("numeroVoie", pa.string()),
    ("typeVoie", pa.string()),
    ("libelleVoie", pa.string()),
    ("complementAdresse", pa.string()),
    ("postcode", pa.string()),
    ("city", pa.string()),
    ("insee", pa.string()),
    ("cj_ul", pa.string()),
    ("etat_admin", pa.string()),
    ("last_treatment_date", pa.timestamp("us")),
    ("sigle_ul", pa.string()),
    ("denomination_ul", pa.string()),
    ("denomination_usuelle_ul", pa.string()),
    ("nom_ul", pa.string()),
    ("prenom_usuel_ul", pa.string()),
    ("pm_dirigeant_names", pa.string()),
])


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


def _normalize_codes(series):
    def _safe_convert(x):
        if pd.isna(x) or x in ["", "nan"]:
            return None
        try:
            s = str(x).strip()
        except (ValueError, TypeError):
            return None
        if not s or s.lower() == "nan":
            return None
        if re.fullmatch(r"\d+\.0+", s):
            s = s.split(".")[0]
        return s or None
    return series.apply(_safe_convert)


def _normalize_siret(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    if len(digits) < 14:
        return digits.zfill(14)
    return digits if len(digits) == 14 else None


def _normalize_siren(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    if len(digits) < 9:
        return digits.zfill(9)
    return digits if len(digits) == 9 else None


def _normalize_cj_ul(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def _ensure_columns(pdf: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col not in pdf.columns:
            pdf[col] = None
    return pdf


def _clean_pdf(pdf: pd.DataFrame, pm_map: Dict[str, List[str]], *, partition_col: str) -> pd.DataFrame:
    if pdf.empty:
        return pdf

    pdf = _ensure_columns(pdf, OUTPUT_COLUMNS)
    pdf["siren"] = pdf["siren"].apply(_normalize_siren)
    pdf = _normalize_output(pdf, pm_map)
    pdf["siret"] = pdf["siret"].apply(_normalize_siret)
    missing_siren = pdf["siren"].isna() & pdf["siret"].notna()
    pdf.loc[missing_siren, "siren"] = pdf.loc[missing_siren, "siret"].str[:9]
    pdf["cj_ul"] = pdf["cj_ul"].apply(_normalize_cj_ul)

    insee_series = cast(pd.Series, _normalize_codes(cast(pd.Series, pdf["insee"])))
    postcode_series = cast(pd.Series, _normalize_codes(cast(pd.Series, pdf["postcode"])))
    pdf["insee"] = insee_series.astype("string")
    pdf["postcode"] = postcode_series.astype("string")

    pdf["last_treatment_date"] = pd.to_datetime(pdf["last_treatment_date"], errors="coerce")

    pdf = cast(pd.DataFrame, pdf[pdf["siret"].notna() & (pdf["siret"].str.len() == 14)])
    if partition_col == "insee":
        pdf = cast(pd.DataFrame, pdf[pdf["insee"].notna() & (pdf["insee"] != "")])
    elif partition_col == "postcode":
        pdf = cast(pd.DataFrame, pdf[pdf["postcode"].notna() & (pdf["postcode"] != "")])

    pdf = pdf.drop_duplicates(subset=["siret"], keep="first")
    pdf = cast(pd.DataFrame, pdf[OUTPUT_COLUMNS])
    return pdf


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
    pdf["is_siege"] = pdf["etablissementSiege"].apply(_parse_siege)
    pdf["pm_dirigeant_names"] = pdf["siren"].map(
        lambda s: "|".join(pm_map.get(str(s), [])) if s else None
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
    
    # Nettoyage des partitions existantes pour éviter l'accumulation de doublons
    # (PyArrow write_to_dataset ajoute des fichiers au lieu d'écraser)
    for subdir in [output_insee, output_cp]:
        if subdir.exists():
            file_count = sum(1 for _ in subdir.rglob("*.parquet"))
            if file_count > 0:
                print(f"⚠️  Nettoyage de {subdir} ({file_count} fichiers parquet existants)")
                shutil.rmtree(subdir)
    
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
        sirens = set(pdf["siren"].apply(_normalize_siren).dropna().unique())
        pm_map = load_pm_dirigeant_names(sirens, harvest_db)
        pdf = _clean_pdf(pdf, pm_map, partition_col="insee")
        if pdf.empty:
            continue
        table = pa.Table.from_pandas(pdf, schema=OUTPUT_SCHEMA, preserve_index=False)
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
        sirens = set(pdf["siren"].apply(_normalize_siren).dropna().unique())
        pm_map = load_pm_dirigeant_names(sirens, harvest_db)
        pdf = _clean_pdf(pdf, pm_map, partition_col="postcode")
        if pdf.empty:
            continue
        table = pa.Table.from_pandas(pdf, schema=OUTPUT_SCHEMA, preserve_index=False)
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
