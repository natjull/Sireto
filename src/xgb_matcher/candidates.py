"""
Shared candidate pooling logic for XGBoost matcher.

This module provides consistent candidate loading for both training and inference,
ensuring alignment between train and serve environments.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set

import pyarrow.parquet as pq

# Paths (configurable, with defaults)
DEFAULT_PARQUET_PATH = Path("data/StockEtablissement_utf8.parquet")
DEFAULT_UL_PATH = Path("data/StockUniteLegale_utf8.parquet")
DEFAULT_HARVEST_DB = Path("data/harvest_full.sqlite")
BATCH_SIZE = 100_000


def load_pm_dirigeant_names(
    sirens: Set[str],
    harvest_db: Path = DEFAULT_HARVEST_DB,
) -> Dict[str, List[str]]:
    """
    Load PM dirigeant names (personne morale) from harvest database.
    
    Args:
        sirens: Set of SIREN codes to lookup
        harvest_db: Path to harvest SQLite database
        
    Returns:
        Dict mapping SIREN to list of PM dirigeant names
    """
    if not harvest_db.exists() or not sirens:
        return {}

    out: Dict[str, Set[str]] = {s: set() for s in sirens}
    con = sqlite3.connect(str(harvest_db))

    # Chunk SIRENs for SQLite parameter limit
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


def load_candidates_for_locations(
    postcodes: Set[str],
    insee_codes: Set[str],
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    ul_path: Path = DEFAULT_UL_PATH,
    harvest_db: Path = DEFAULT_HARVEST_DB,
    load_pm_names: bool = True,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Load candidate establishments from parquet, filtered by location.
    
    This is the SHARED function used by both training and inference to ensure
    consistency in candidate pooling.
    
    Args:
        postcodes: Set of postal codes to filter
        insee_codes: Set of INSEE commune codes to filter
        parquet_path: Path to establishment parquet file
        ul_path: Path to UniteLegale parquet file
        harvest_db: Path to harvest SQLite database for PM dirigeants
        load_pm_names: Whether to load PM dirigeant names
        verbose: Print progress messages
        
    Returns:
        Dict mapping SIRET to candidate dict with all relevant fields
    """
    cols = [
        "siret",
        "enseigne1Etablissement",
        "enseigne2Etablissement",
        "enseigne3Etablissement",
        "denominationUsuelleEtablissement",
        "siren",
        "etablissementSiege",
        "numeroVoieEtablissement",
        "typeVoieEtablissement",
        "libelleVoieEtablissement",
        "complementAdresseEtablissement",
        "codePostalEtablissement",
        "libelleCommuneEtablissement",
        "codeCommuneEtablissement",
        "categorieJuridiqueUniteLegale",
        "etatAdministratifEtablissement",
    ]
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    pf = pq.ParquetFile(parquet_path)
    mapping: Dict[str, dict] = {}
    sirens: Set[str] = set()

    # First pass: collect all rows for CP/INSEE
    for batch in pf.iter_batches(columns=cols, batch_size=BATCH_SIZE):
        pdf = batch.to_pandas()
        mask = (
            pdf["codePostalEtablissement"].isin(postcodes) | 
            pdf["codeCommuneEtablissement"].isin(insee_codes)
        )
        pdf = pdf[mask]
        
        # Filter only active establishments
        pdf = pdf[pdf["etatAdministratifEtablissement"] != "F"]

        for _, r in pdf.iterrows():
            # Parse siege boolean from various formats
            siege_val = r.get("etablissementSiege")
            if isinstance(siege_val, str):
                siege_norm = siege_val.strip().upper()
                is_siege = siege_norm in {"TRUE", "VRAI", "1", "OUI", "YES"}
            else:
                is_siege = bool(siege_val)
            
            cand = {
                "siret": r["siret"],
                "siren": r.get("siren"),
                "denomination": r.get("denominationUsuelleEtablissement"),
                "enseigne1": r.get("enseigne1Etablissement"),
                "enseigne2": r.get("enseigne2Etablissement"),
                "enseigne3": r.get("enseigne3Etablissement"),
                "is_siege": is_siege,
                "numeroVoie": r.get("numeroVoieEtablissement"),
                "typeVoie": r.get("typeVoieEtablissement"),
                "libelleVoie": r.get("libelleVoieEtablissement"),
                "complementAdresse": r.get("complementAdresseEtablissement"),
                "postcode": r.get("codePostalEtablissement"),
                "city": r.get("libelleCommuneEtablissement"),
                "insee": r.get("codeCommuneEtablissement"),
                "cj_ul": r.get("categorieJuridiqueUniteLegale"),
            }
            if cand.get("siren"):
                sirens.add(str(cand["siren"]))
            mapping[r["siret"]] = cand

    # Enrichment with UniteLegale names
    if ul_path.exists() and sirens:
        if verbose:
            print("  Loading UniteLegale names for candidates...")
        ul_cols = [
            "siren",
            "sigleUniteLegale",
            "denominationUniteLegale",
            "denominationUsuelle1UniteLegale",
            "denominationUsuelle2UniteLegale",
            "denominationUsuelle3UniteLegale",
            "nomUniteLegale",
            "nomUsageUniteLegale",
            "prenomUsuelUniteLegale",
            "pseudonymeUniteLegale",
        ]
        pf_ul = pq.ParquetFile(ul_path)
        ul_map: Dict[str, dict] = {}
        for batch in pf_ul.iter_batches(columns=ul_cols, batch_size=BATCH_SIZE):
            pdf = batch.to_pandas()
            pdf = pdf[pdf["siren"].isin(sirens)]
            for _, r in pdf.iterrows():
                ul_map[r["siren"]] = {
                    "sigle_ul": r.get("sigleUniteLegale"),
                    "denomination_ul": r.get("denominationUniteLegale"),
                    "denomination_usuelle_ul": " ".join(
                        filter(
                            None,
                            [
                                r.get("denominationUsuelle1UniteLegale"),
                                r.get("denominationUsuelle2UniteLegale"),
                                r.get("denominationUsuelle3UniteLegale"),
                            ],
                        )
                    ),
                    "nom_ul": r.get("nomUniteLegale"),
                    "nom_usage_ul": r.get("nomUsageUniteLegale"),
                    "prenom_usuel_ul": r.get("prenomUsuelUniteLegale"),
                    "pseudonyme_ul": r.get("pseudonymeUniteLegale"),
                }
        # Merge into mapping
        for siret, cand in mapping.items():
            siren = cand.get("siren")
            if siren and siren in ul_map:
                cand.update(ul_map[siren])

    # Enrich with PM dirigeant names
    if load_pm_names and harvest_db.exists():
        pm_names = load_pm_dirigeant_names(sirens, harvest_db)
        if pm_names:
            for siret, cand in mapping.items():
                siren = cand.get("siren")
                if siren and siren in pm_names:
                    cand["pm_dirigeant_names"] = pm_names[siren]

    return mapping


def get_candidates_for_query(
    postcode: Optional[str],
    insee: Optional[str],
    all_candidates: Dict[str, dict],
) -> List[tuple]:
    """
    Filter candidates for a specific query based on location.
    
    Args:
        postcode: Query postal code
        insee: Query INSEE code
        all_candidates: Full candidate pool from load_candidates_for_locations
        
    Returns:
        List of (siret, candidate_dict) tuples matching the location
    """
    results = []
    for siret, cand in all_candidates.items():
        cand_pc = cand.get("postcode", "")
        cand_insee = cand.get("insee", "")
        if (postcode and cand_pc == postcode) or (insee and cand_insee == insee):
            results.append((siret, cand))
    return results
