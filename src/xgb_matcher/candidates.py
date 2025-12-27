"""
Shared candidate pooling logic for XGBoost matcher.

This module provides consistent candidate loading for both training and inference,
ensuring alignment between train and serve environments.
"""

from __future__ import annotations

import sqlite3
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq

from .naming import build_candidate_names, normalize_text, primary_name
from .features import NAME_STOPWORDS, build_address, set_global_name_idf_map

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
    from tqdm import tqdm
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


def _tokenize_name_for_idf(text: str | None) -> Set[str]:
    """Tokenize a name for IDF computation (uppercased, stopwords removed)."""
    if not text:
        return set()
    tokens = normalize_text(text).split()
    return {t for t in tokens if t and t not in NAME_STOPWORDS and len(t) >= 2}


def compute_name_idf_map(candidates: Dict[str, dict]) -> Tuple[Dict[str, float], float]:
    """Compute IDF map for candidate name tokens over the loaded pool."""
    doc_freq: Dict[str, int] = defaultdict(int)
    doc_count = 0
    for cand in candidates.values():
        name = primary_name(cand)
        tokens = _tokenize_name_for_idf(name)
        if not tokens:
            continue
        doc_count += 1
        for tok in tokens:
            doc_freq[tok] += 1
    if doc_count == 0:
        return {}, 0.0
    idf_map = {
        tok: math.log((doc_count + 1) / (df + 1)) + 1.0
        for tok, df in doc_freq.items()
    }
    default_idf = math.log((doc_count + 1) / 1) + 1.0
    return idf_map, default_idf


def _attach_address_density(
    candidates: Dict[str, dict],
    *,
    key: str,
    target_field: str,
) -> None:
    """Attach address density per location key (insee or postcode)."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for cand in candidates.values():
        loc = cand.get(key)
        if loc:
            groups[loc].append(cand)
    for _, group in groups.items():
        counts: Dict[str, int] = defaultdict(int)
        for cand in group:
            addr = build_address(cand)
            if addr:
                counts[addr] += 1
        for cand in group:
            addr = build_address(cand)
            cand[target_field] = counts.get(addr, 1) if addr else 1


def load_candidates_for_locations(
    postcodes: Set[str],
    insee_codes: Set[str],
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    ul_path: Path = DEFAULT_UL_PATH,
    harvest_db: Path = DEFAULT_HARVEST_DB,
    load_pm_names: bool = True,
    include_closed_establishments: bool = False,
    drop_unnamed_candidates: bool = True,
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
        include_closed_establishments: If True, include establishments with etatAdministratifEtablissement == 'F'.
        drop_unnamed_candidates: If True, drop candidates with no usable name fields.
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
        "dateDernierTraitementEtablissement",
    ]
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    pf = pq.ParquetFile(parquet_path)
    mapping: Dict[str, dict] = {}
    sirens: Set[str] = set()

    # First pass: collect all rows for CP/INSEE
    if verbose:
        from tqdm import tqdm
        desc = "  [1/3] Reading establishments"
        total = math.ceil(pf.metadata.num_rows / BATCH_SIZE) if pf.metadata else None
    else:
        desc = None
        total = None

    batches = pf.iter_batches(columns=cols, batch_size=BATCH_SIZE)
    for batch in (tqdm(batches, desc=desc, total=total) if desc else batches):
        pdf = batch.to_pandas()
        mask = (
            pdf["codePostalEtablissement"].isin(postcodes) | 
            pdf["codeCommuneEtablissement"].isin(insee_codes)
        )
        pdf = pdf[mask]
        
        # By default we filter only active establishments; when enabled we keep closed ones too.
        if not include_closed_establishments:
            pdf = pdf[pdf["etatAdministratifEtablissement"] != "F"]
        
        for r in pdf.to_dict("records"):
            # Parse siege boolean from various formats
            siege_val = r.get("etablissementSiege")
            if isinstance(siege_val, str):
                siege_norm = siege_val.strip().upper()
                is_siege = siege_norm in {"TRUE", "VRAI", "1", "OUI", "YES"}
            else:
                is_siege = bool(siege_val)
            
            cand = {
                "siret": r.get("siret"),
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
                "etat_admin": r.get("etatAdministratifEtablissement"),
                "last_treatment_date": r.get("dateDernierTraitementEtablissement"),
            }
            if cand.get("siren"):
                sirens.add(str(cand["siren"]))
            mapping[r.get("siret")] = cand

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
        
        if verbose:
            from tqdm import tqdm
            desc = "  [2/3] Loading UniteLegale info"
            total = math.ceil(pf_ul.metadata.num_rows / BATCH_SIZE) if pf_ul.metadata else None
        else:
            desc = None
            total = None

        batches = pf_ul.iter_batches(columns=ul_cols, batch_size=BATCH_SIZE)
        for batch in (tqdm(batches, desc=desc, total=total) if desc else batches):
            pdf = batch.to_pandas()
            matches = pdf[pdf["siren"].isin(sirens)]
            
            for r in matches.to_dict("records"):
                ul_map[r.get("siren")] = {
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
        if verbose:
            print("  [3/3] Loading PM dirigeant names from SQLite...")
        pm_names = load_pm_dirigeant_names(sirens, harvest_db)
        if pm_names:
            for siret, cand in mapping.items():
                siren = cand.get("siren")
                if siren and siren in pm_names:
                    cand["pm_dirigeant_names"] = pm_names[siren]

    # Drop candidates with no usable names (these tend to pollute top-k with address-only matches).
    if drop_unnamed_candidates:
        before = len(mapping)
        mapping = {siret: cand for siret, cand in mapping.items() if build_candidate_names(cand)}
        if verbose:
            dropped = before - len(mapping)
            print(f"  Dropped {dropped} / {before} candidates without names")

    # Compute global name IDF map and register it for feature computation.
    if mapping:
        idf_map, default_idf = compute_name_idf_map(mapping)
        set_global_name_idf_map(idf_map, default_idf)

        # Address density per INSEE and per postcode (used by address_density feature)
        _attach_address_density(mapping, key="insee", target_field="_xgb_addr_density_insee")
        _attach_address_density(mapping, key="postcode", target_field="_xgb_addr_density_cp")

    # Return dict sorted by SIRET for deterministic iteration order
    return dict(sorted(mapping.items()))


def build_location_index(
    all_candidates: Dict[str, dict],
) -> Dict[str, Dict[str, List[Tuple[str, dict]]]]:
    """
    Build indices to retrieve candidates by INSEE or postcode in O(1).
    
    Returns:
        {"insee": {code: [(siret, cand), ...]}, "postcode": {cp: [(siret, cand), ...]}}
    """
    by_insee: Dict[str, List[Tuple[str, dict]]] = {}
    by_postcode: Dict[str, List[Tuple[str, dict]]] = {}
    for siret, cand in all_candidates.items():
        insee = cand.get("insee") or ""
        if insee:
            by_insee.setdefault(insee, []).append((siret, cand))
        postcode = cand.get("postcode") or ""
        if postcode:
            by_postcode.setdefault(postcode, []).append((siret, cand))
    return {"insee": by_insee, "postcode": by_postcode}


def get_candidates_for_query(
    postcode: Optional[str],
    insee: Optional[str],
    all_candidates: Dict[str, dict],
    index: Optional[Dict[str, Dict[str, List[Tuple[str, dict]]]]] = None,
) -> List[Tuple[str, dict]]:
    """
    Filter candidates for a specific query based on location.
    
    Args:
        postcode: Query postal code
        insee: Query INSEE code
        all_candidates: Full candidate pool from load_candidates_for_locations
        index: Optional index from build_location_index for fast lookup
        
    Returns:
        List of (siret, candidate_dict) tuples matching the location
    """
    if index:
        results: List[Tuple[str, dict]] = []
        seen: Set[str] = set()
        if insee and insee in index["insee"]:
            for siret, cand in index["insee"][insee]:
                if siret not in seen:
                    results.append((siret, cand))
                    seen.add(siret)
        if postcode and postcode in index["postcode"]:
            for siret, cand in index["postcode"][postcode]:
                if siret not in seen:
                    results.append((siret, cand))
                    seen.add(siret)
        return results

    results = []
    for siret, cand in all_candidates.items():
        cand_pc = cand.get("postcode", "")
        cand_insee = cand.get("insee", "")
        if (postcode and cand_pc == postcode) or (insee and cand_insee == insee):
            results.append((siret, cand))
    return results
