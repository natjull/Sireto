"""
Offline scoring of the trained XGBoost matcher on a CRM CSV, producing top-k candidates per row.

Inputs:
  - CRM file: data/testcrm/data_56_subset_corbas_decines.csv
  - Parquet SIRENE: data/StockEtablissement_utf8.parquet
  - Trained artifacts: models/xgb_matcher.json, models/xgb_matcher_scaler.pkl, models/xgb_matcher_features.json

Strategy:
  1) Read CRM, collect the set of codePostal and codeCommune present.
  2) Stream the parquet once, keeping only rows whose codePostal or codeCommune is in those sets.
     This builds an in-memory candidate pool with name + address for the relevant communes.
  3) For each CRM row, score all candidates sharing the same codeCommune if available,
     otherwise the same codePostal. Return top-k (default 5).

Features are computed using the shared module: src.xgb_matcher.features
"""

from __future__ import annotations

import argparse
import os
import json
import re
import pickle
import sqlite3
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from xgboost import XGBClassifier, XGBRanker
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    build_address,
    build_semantic_name_pool,
    jaro_sim,
    make_features_from_preprocessed,
    normalize_text,
    preprocess_crm_row,
)
from src.xgb_matcher.naming import build_candidate_names, primary_name
from src.xgb_matcher.candidates import (
    load_candidates_for_locations as load_candidates_shared,
    get_candidates_for_query,
)
from src.xgb_matcher.semantic import batch_encode_texts, get_cache_stats, top2_semantic_similarities_batch

# Config
CRM_PATH = Path("data/testcrm/data_aligned.csv")
MODEL_DIR = Path("models")
TOP_K = 5
BATCH_SIZE = 100_000
USE_RERANK_RULES = False  # Set via --no-rerank flag


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer XGB matcher top-k (optionally with TreeSHAP contribs).")
    p.add_argument("--crm-path", type=Path, default=CRM_PATH)
    p.add_argument("--output-path", type=Path, default=Path("reports/xgb_infer_topk.csv"))
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--write-csv", action="store_true", default=True)
    p.add_argument("--with-shap", action="store_true", help="Add TreeSHAP explanations for each top-k row.")
    p.add_argument("--shap-top-n", type=int, default=int(os.getenv("XGB_SHAP_TOP_N", "10")))
    p.add_argument(
        "--include-closed-candidates",
        action="store_true",
        help="Include closed establishments (etatAdministratifEtablissement == 'F') in the candidate pool.",
    )
    p.add_argument(
        "--keep-unnamed-candidates",
        action="store_true",
        help="Disable filtering of candidates without names (may increase noise).",
    )
    return p.parse_args()


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid (xgboost margins can be large).
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def find_latest_models():
    """Find the most recent ranker and classifier models."""
    # Look for timestamped metadata (format: xgb_matcher_features_YYYYMMDD_HHMMSS.json)
    pattern = list(MODEL_DIR.glob("xgb_matcher_features_[0-9]*_[0-9]*.json"))
    if pattern:
        latest_meta = sorted(pattern, reverse=True)[0]
        timestamp = latest_meta.stem.replace("xgb_matcher_features_", "")
        print(f"Using latest model cascade: {timestamp}")
        return {
            "ranker": MODEL_DIR / f"xgbranker_{timestamp}.json",
            "classifier": MODEL_DIR / f"xgbclassifier_{timestamp}.json",
            "scaler": MODEL_DIR / f"xgb_matcher_scaler_{timestamp}.pkl",
            "meta": latest_meta,
            "timestamp": timestamp,
        }
    else:
        print("Using default model (no timestamp)")
        return {
            "ranker": MODEL_DIR / "xgbranker.json",
            "classifier": MODEL_DIR / "xgbclassifier.json",
            "scaler": MODEL_DIR / "xgb_matcher_scaler.pkl",
            "meta": MODEL_DIR / "xgb_matcher_features.json",
            "timestamp": "default",
        }


MODEL_PATHS = find_latest_models()

# Rerank coefficients
BOOST_PERFECT_ADDR = 0.15    # D1
PENALTY_WRONG_ADDR = 0.10    # D1
BOOST_PUBLIC_SCHOOL = 0.20   # B1
PENALTY_ASSOCIATION = 0.25   # B1
PENALTY_HOLDING_MISMATCH = 0.20  # A1

# ---------- Reranking helpers ----------

logger = logging.getLogger(__name__)

SCHOOL_TOKENS = {
    "COLLEGE",
    "COLLGE",
    "CLG",
    "LYCEE",
    "LYCE",
    "ECOLE",
    "ACADEMY",
}

ASSOCIATION_TOKENS = {
    "ASSOCIATION",
    "ASSO",
    "FOYER",
    "AMICALE",
    "CONSEIL",
    "PARENTS",
    "PARENT",
    "ELEVES",
    "SPORTIVE",
    "FSE",
    "SOCIO",
}
HOLDING_TOKENS = {"HOLDING", "GROUPE", "CORPORATION", "GIE"}


def extract_significant_tokens(text: str, min_len: int = 3) -> Set[str]:
    """Extract significant tokens (min_len+ chars) from normalized text."""
    if not text:
        return set()
    norm = normalize_text(text)
    return {tok for tok in norm.split() if len(tok) >= min_len}


def compute_ul_token_bonus(crm_name: str, cand: dict) -> float:
    """
    Compute bonus for partial keyword matches in UL names.
    
    Helps portmanteau names like 'DigitBoxing' match 'SPORT BOXING & CO'.
    Returns a score in [0, 1] based on how many CRM tokens appear in UL names.
    """
    crm_tokens = extract_significant_tokens(crm_name)
    if not crm_tokens:
        return 0.0
    
    # Collect all UL name tokens
    ul_fields = ["denomination_ul", "denomination_usuelle_ul", "sigle_ul"]
    ul_tokens: Set[str] = set()
    for field in ul_fields:
        val = cand.get(field)
        if val:
            ul_tokens.update(extract_significant_tokens(str(val)))
    
    if not ul_tokens:
        return 0.0
    
    # Check for substring matches (e.g., 'BOXING' in 'DIGITBOXING' or vice versa)
    matches = 0
    for crm_tok in crm_tokens:
        for ul_tok in ul_tokens:
            # Direct token match
            if crm_tok == ul_tok:
                matches += 1
                break
            # Substring in either direction (for portmanteaux)
            if len(crm_tok) >= 4 and len(ul_tok) >= 4:
                if crm_tok in ul_tok or ul_tok in crm_tok:
                    matches += 0.7
                    break
    
    return min(1.0, matches / len(crm_tokens))


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return set(normalize_text(text).split())


def _is_school_crm(crm_row: pd.Series) -> bool:
    name = crm_row.get("crm_name", "")
    tokens = _tokenize(name)
    return any(tok in SCHOOL_TOKENS for tok in tokens)


def _is_public_school_candidate(cand_name: str) -> bool:
    tokens = _tokenize(cand_name)
    return any(tok in SCHOOL_TOKENS for tok in tokens)


def _is_association_candidate(cand_name: str) -> bool:
    tokens = _tokenize(cand_name)
    return any(tok in ASSOCIATION_TOKENS for tok in tokens)


def _association_token_count(cand_name: str) -> int:
    tokens = _tokenize(cand_name)
    return len(ASSOCIATION_TOKENS & tokens)


def _holding_mismatch(crm_name: str, cand_name: str) -> bool:
    crm_tokens = _tokenize(crm_name)
    cand_tokens = _tokenize(cand_name)
    if not cand_tokens:
        return False
    return not (HOLDING_TOKENS & crm_tokens) and bool(HOLDING_TOKENS & cand_tokens)


def apply_rerank_rules(scores: np.ndarray, features: List[dict], crm_row: pd.Series) -> np.ndarray:
    """
    Apply post-inference rerank rules on scores.

    Args:
        scores: raw model scores (per candidate)
        features: list of feature dicts (with extra metadata keys)
        crm_row: CRM row (for school detection)
    """
    adjusted = scores.astype(float).copy()
    is_school = _is_school_crm(crm_row)

    for i, feat in enumerate(features):
        delta = 0.0
        siret = feat.get("_siret", "")
        cand_name = feat.get("_cand_name", "") or ""

        # D1: address-based rerank
        addr_jaro = float(feat.get("addr_jaro", 0.0))
        street_number_diff = float(feat.get("street_number_diff", 9999))
        name_jaro_max = float(feat.get("name_jaro_max", 0.0))
        acronym_match_max = float(feat.get("acronym_match_max", 0.0))
        
        # Expert logic: boost if perfect address AND (solid name OR acronym match)
        if addr_jaro >= 0.97 and (street_number_diff == 0 or street_number_diff == 9999):
            if name_jaro_max >= 0.8 or (acronym_match_max > 0.5 and name_jaro_max >= 0.4):
                delta += BOOST_PERFECT_ADDR
                logger.info("rerank_adjust|%s|%s|D1_PERFECT_ADDR|+%.3f", crm_row.get("crm_name"), siret, BOOST_PERFECT_ADDR)

        # Penalty logic: tougher on short names (like ASL) which match easily by accident
        query_len = len(str(crm_row.get("crm_name", "")).replace(" ", ""))
        penalty_name_threshold = 0.6 if query_len > 4 else 0.95
        
        if addr_jaro <= 0.70 and street_number_diff >= 5 and name_jaro_max < penalty_name_threshold:
            delta -= PENALTY_WRONG_ADDR
            logger.debug("rerank_adjust|%s|%s|D1_WRONG_ADDR|-%.3f", crm_row.get("crm_name"), siret, PENALTY_WRONG_ADDR)

        # D2: Semantic boost for weak lexical match but strong semantic signal
        # Only apply when semantic is DOMINANT over lexical (prevents false boosting)
        # Example: "DigitBoxing" <-> "SPORT BOXING" (Jaro=0.63, Sem=0.65)
        name_semantic_max = float(feat.get("name_semantic_max", 0.0))
        if (addr_jaro >= 0.97 
            and name_semantic_max >= 0.55  # Strong semantic signal
            and name_semantic_max > name_jaro_max  # Semantic must be better than lexical
            and name_jaro_max < 0.65):  # Lexical is genuinely weak
            semantic_boost = 0.20
            delta += semantic_boost
            logger.info("rerank_adjust|%s|%s|D2_SEMANTIC_BOOST|+%.3f", crm_row.get("crm_name"), siret, semantic_boost)

        # D3: Penalize candidates with perfect address but NO name signal at all
        # These are likely false positives that score high only due to address match
        if addr_jaro >= 0.97 and name_jaro_max < 0.3 and name_semantic_max < 0.1 and acronym_match_max < 0.5:
            empty_name_penalty = 0.15
            delta -= empty_name_penalty
            logger.debug("rerank_adjust|%s|%s|D3_EMPTY_NAME|-%.3f", crm_row.get("crm_name"), siret, empty_name_penalty)

        # B1: public school vs association
        if is_school:
            assoc_count = _association_token_count(cand_name)
            if assoc_count:
                assoc_penalty = PENALTY_ASSOCIATION * min(3, assoc_count)
                delta -= assoc_penalty
                logger.info("rerank_adjust|%s|%s|B1_ASSOCIATION|-%.3f", crm_row.get("crm_name"), siret, assoc_penalty)
            elif _is_public_school_candidate(cand_name):
                delta += BOOST_PUBLIC_SCHOOL
                logger.info("rerank_adjust|%s|%s|B1_PUBLIC_SCHOOL|+%.3f", crm_row.get("crm_name"), siret, BOOST_PUBLIC_SCHOOL)

        # A1: penalize holding/group/corporation when absent from CRM (address must be close)
        if addr_jaro > 0.8 and _holding_mismatch(crm_row.get("crm_name", ""), cand_name):
            delta -= PENALTY_HOLDING_MISMATCH
            logger.info("rerank_adjust|%s|%s|A1_HOLDING_MISMATCH|-%.3f", crm_row.get("crm_name"), siret, PENALTY_HOLDING_MISMATCH)

        if delta != 0.0:
            adjusted[i] = adjusted[i] + delta

    # D4: Dominant Name Uniqueness Boost
    # Identify unique brand-like matches in the commune that should win even if address is wrong
    tier1_indices = [] # Super Dominant (Exact substring or perfect acronym)
    tier2_indices = [] # Strong Match (High Jaro or high semantic)
    
    for i, feat in enumerate(features):
        name_jaro_max = float(feat.get("name_jaro_max", 0.0))
        name_contains_crm_max = float(feat.get("name_contains_crm_max", 0.0))
        name_semantic_max = float(feat.get("name_semantic_max", 0.0))
        acronym_match_max = float(feat.get("acronym_match_max", 0.0))
        
        # Tier 1: Exceptional evidence
        # For acronyms, require at least modest address match to avoid false positives
        addr_jaro = float(feat.get("addr_jaro", 0.0))
        is_tier1 = (
            (name_contains_crm_max == 1.0 and name_jaro_max >= 0.7) or  # Brand found
            (acronym_match_max >= 1.0 and addr_jaro >= 0.70)  # Perfect acronym BUT with reasonable address
        )
        if is_tier1:
            tier1_indices.append(i)
            continue
            
        # Tier 2: Very high similarity
        is_tier2 = (
            name_jaro_max >= 0.95 or
            name_semantic_max >= 0.85
        )
        if is_tier2:
            tier2_indices.append(i)

    # Strategy: Boost only if a single hero emerges from the best available tier
    boost_idx = None
    if len(tier1_indices) == 1:
        boost_idx = tier1_indices[0]
    elif len(tier1_indices) == 0 and len(tier2_indices) == 1:
        boost_idx = tier2_indices[0]

    if boost_idx is not None:
        idx = boost_idx
        siret = features[idx].get("_siret", "")
        boost_val = 0.75 # Massive boost to ensure victory over address confusers
        adjusted[idx] += boost_val
        logger.info("rerank_adjust|%s|%s|D4_DOMINANT_UNIQUE|+%.3f", crm_row.get("crm_name"), siret, boost_val)

    # D5: Strong name match boost
    for i, feat in enumerate(features):
        name_jaro = float(feat.get("name_jaro_max", 0.0))
        name_len = float(feat.get("name_length_max", 0.0))
        addr = float(feat.get("addr_jaro", 0.0))
        if name_len < 6 or addr < 0.85:
            continue
        if name_jaro >= 0.95:
            adjusted[i] += 0.40
        elif name_jaro >= 0.90:
            adjusted[i] += 0.25

    # D7: Source-aware tie-break (PM downranked when UL equivalent exists)
    ul_denoms_pool = set()
    for feat in features:
        for d in feat.get("_ul_denoms", []):
            if d:
                ul_denoms_pool.add(d)
    crm_norm = normalize_text(crm_row.get("crm_name", ""))
    for i, feat in enumerate(features):
        type_max = float(feat.get("type_of_max_name", 0.0))
        is_ul = float(feat.get("is_ul_name_max", 0.0))
        addr = float(feat.get("addr_jaro", 0.0))
        if type_max == 4 and is_ul == 0.0 and addr >= 0.90:
            for ul_d in ul_denoms_pool:
                if jaro_sim(crm_norm, ul_d) >= 0.85:
                    adjusted[i] -= 0.15
                    break

    # D8: Contains-boost for strong address matches
    for i, feat in enumerate(features):
        addr = float(feat.get("addr_jaro", 0.0))
        contains = float(feat.get("name_contains_crm_max", 0.0))
        jaro = float(feat.get("name_jaro_max", 0.0))
        if addr >= 0.97 and contains == 1.0 and jaro >= 0.80:
            adjusted[i] += 0.20

    # D6: Alias boost (parentheses) - rerank only
    crm_raw = crm_row.get("crm_name", "")
    aliases = [normalize_text(m) for m in re.findall(r"\(([^)]+)\)", crm_raw)]
    blacklist = {"SIEGE", "BAT", "BATIMENT", "ETAGE", "LOCAL", "AGENCE", "RDC"}
    aliases = [a for a in aliases if len(a) >= 3 and a not in blacklist and not a.isdigit()]
    alias_match = [False] * len(features)
    if aliases:
        for i, feat in enumerate(features):
            cand_norm = normalize_text(feat.get("_cand_name", ""))
            for alias in aliases:
                if jaro_sim(alias, cand_norm) >= 0.80 or alias in cand_norm:
                    alias_match[i] = True
                    addr = float(feat.get("addr_jaro", 0.0))
                    if addr >= 0.95:
                        adjusted[i] += 0.25
                    break

        # D10: Boost unique alias match with near-perfect address → force rank 1
        alias_perfect_addr_matches = []
        for i, feat in enumerate(features):
            if not alias_match[i]:
                continue
            addr = float(feat.get("addr_jaro", 0.0))
            street_name_jaro = float(feat.get("street_name_jaro", 0.0))
            street_number_diff = float(feat.get("street_number_diff", 9999))
            # Check alias Jaro quality
            cand_norm = normalize_text(feat.get("_cand_name", ""))
            best_alias_jaro = max(jaro_sim(alias, cand_norm) for alias in aliases) if aliases else 0.0
            # Guard rails: alias_jaro >= 0.85, addr >= 0.96, street >= 0.95, diff <= 2
            if best_alias_jaro >= 0.85 and addr >= 0.96 and street_name_jaro >= 0.95 and street_number_diff <= 2:
                alias_perfect_addr_matches.append(i)
        if len(alias_perfect_addr_matches) == 1:
            idx = alias_perfect_addr_matches[0]
            # Force rank 1 by setting score to max + margin
            adjusted[idx] = max(adjusted) + 0.10
            siret = features[idx].get("_siret", "")
            logger.info("rerank|%s|%s|D10_ALIAS_FORCE_TOP|score=%.3f", crm_row.get("crm_name"), siret, adjusted[idx])

    # D9: Establishment head office tie-break
    for i, feat in enumerate(features):
        addr = float(feat.get("addr_jaro", 0.0))
        if addr < 0.97:
            continue
        if feat.get("_is_siege"):
            adjusted[i] += 0.10

    # D11: Token-match boost - when a significant CRM token appears in candidate name + perfect address
    crm_name_norm = normalize_text(crm_row.get("crm_name", ""))
    crm_tokens = {tok for tok in crm_name_norm.split() if len(tok) >= 3}
    # Exclude common stopwords
    stopwords = {"AND", "THE", "LES", "DES", "DU", "DE", "LA", "LE", "ET", "GROUPE", "SOCIETE", "HOLDING"}
    crm_tokens = crm_tokens - stopwords
    
    if crm_tokens:
        for i, feat in enumerate(features):
            addr = float(feat.get("addr_jaro", 0.0))
            if addr < 0.97:
                continue
            
            # Skip D11 for associations when CRM is a school - they shouldn't get token boost
            # (e.g., "FOYER SOCIO EDUCATIF DU COLLEGE" should not beat the actual "COLLEGE")
            cand_name = feat.get("_cand_name", "")
            if is_school and _is_association_candidate(cand_name):
                continue
            
            # Collect tokens from _cand_name AND all _ul_denoms
            all_cand_names = [normalize_text(cand_name)]
            for ul_name in feat.get("_ul_denoms", []):
                if ul_name:
                    all_cand_names.append(ul_name)
            
            cand_tokens = set()
            for name in all_cand_names:
                cand_tokens.update(name.split())
            
            # Direct token match
            common_tokens = crm_tokens & cand_tokens
            
            # Substring match: check if candidate token is IN a CRM token (for CamelCase like DIGITBOXING)
            # or vice-versa - only for tokens >= 4 chars to avoid false positives
            substring_matches = set()
            for cand_tok in cand_tokens:
                if len(cand_tok) >= 4:
                    for crm_tok in crm_tokens:
                        if len(crm_tok) >= 6 and cand_tok in crm_tok:
                            substring_matches.add(cand_tok)
                            break
            
            all_matches = common_tokens | substring_matches
            if all_matches:
                # Boost proportional to number of matching tokens
                boost = min(0.30 * len(all_matches), 0.60)
                adjusted[i] += boost
                siret = feat.get("_siret", "")
                logger.info("rerank|%s|%s|D11_TOKEN_MATCH(%s)|+%.2f", 
                            crm_row.get("crm_name"), siret, all_matches, boost)

    return adjusted


# ---------- Data loading ----------


def load_crm(path: Path) -> pd.DataFrame:
    """Load CRM test data."""
    df = pd.read_csv(path, sep=";", dtype=str)
    # Map available columns from this test file
    df = df.rename(columns={
        "Client final": "crm_name",
        "Adresse": "crm_address",
        "Commune": "crm_city",
        "Code Postal": "postcode",
        "Code INSEE": "insee",
    })
    # Clean up postcode/insee values (remove .0 suffix from float-like strings)
    for col in ["postcode", "insee"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(int(float(x))) if pd.notna(x) and x not in ["", "nan"] else x)
    # This dataset has no SIRET ground truth; we generate synthetic ids for bookkeeping
    df["crm_id"] = df.index
    return df


def _chunked(seq: List[str], size: int = 900) -> List[List[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def load_pm_dirigeant_names(sirens: set[str]) -> Dict[str, List[str]]:
    """Load PM dirigeant names (personne morale) from harvest_full.sqlite."""
    if not HARVEST_DB.exists() or not sirens:
        return {}
    con = sqlite3.connect(HARVEST_DB)
    cur = con.cursor()
    out: Dict[str, set[str]] = {s: set() for s in sirens}
    for chunk in _chunked(sorted(sirens)):
        q = ",".join("?" for _ in chunk)
        cur.execute(
            f"""
            SELECT siren, denomination
            FROM dirigeants
            WHERE siren IN ({q})
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


def load_candidates_for_locations(postcodes: set[str], insee: set[str]) -> Dict[str, dict]:
    """
    Load candidates from parquet, filtered by location.

    Excludes candidates without any name field (matching would be meaningless).
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
    ]
    pf = pq.ParquetFile(PARQUET_PATH)
    mapping: Dict[str, dict] = {}

    # First pass: collect all rows for CP/INSEE and the set of siren
    sirens: set[str] = set()

    for batch in pf.iter_batches(columns=cols, batch_size=BATCH_SIZE):
        pdf = batch.to_pandas()
        mask = pdf["codePostalEtablissement"].isin(postcodes) | pdf["codeCommuneEtablissement"].isin(insee)
        pdf = pdf[mask]

        for _, r in pdf.iterrows():
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

    # Optional enrichment with UniteLegale names if file exists
    if UNITE_LEGALE_PATH.exists() and sirens:
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
        pf_ul = pq.ParquetFile(UNITE_LEGALE_PATH)
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

    # Enrich with PM dirigeant names (from harvest_full.sqlite)
    pm_names = load_pm_dirigeant_names(sirens)
    if pm_names:
        for siret, cand in mapping.items():
            siren = cand.get("siren")
            if siren and siren in pm_names:
                cand["pm_dirigeant_names"] = pm_names[siren]

    return mapping


# ---------- Main inference ----------


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    print("Loading CRM subset...")
    crm = load_crm(args.crm_path)
    crm_records = crm.to_dict("records")
    crm_pre = [preprocess_crm_row(r) for r in crm_records]
    postcodes = set(crm["postcode"].dropna().unique()) if "postcode" in crm else set()
    insee = set(crm["insee"].dropna().unique()) if "insee" in crm else set()
    print(f"Postcodes of interest: {len(postcodes)} | INSEE codes: {len(insee)}")

    semantic_enabled = os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"
    with_shap = args.with_shap or (os.getenv("XGB_SHAP_ENABLED", "0") == "1")
    if semantic_enabled:
        crm_semantic_names = [
            c.get("crm_name_semantic", "") for c in crm_pre if c.get("crm_name_semantic")
        ]
        unique_crm_semantic = list(dict.fromkeys(crm_semantic_names))
        if unique_crm_semantic:
            print(f"[Semantic] Warmup CRM embeddings: {len(unique_crm_semantic)} unique names...")
            batch_encode_texts(unique_crm_semantic)
            cache_size, cache_mb = get_cache_stats()
            print(f"[Semantic] Cache: {cache_size} embeddings (~{cache_mb:.1f} MB)")

    print("Building candidate pool from shared module (filtered by CP/INSEE)...")
    include_closed = args.include_closed_candidates or (os.getenv("XGB_INCLUDE_CLOSED", "0") == "1")
    candidates = load_candidates_shared(
        postcodes,
        insee,
        include_closed_establishments=include_closed,
        drop_unnamed_candidates=not args.keep_unnamed_candidates,
    )
    print(f"Candidates loaded: {len(candidates)}")

    print("Loading model cascade (Ranker + Classifier)...")
    ranker = XGBRanker()
    ranker.load_model(str(MODEL_PATHS["ranker"]))
    
    classifier = XGBClassifier()
    classifier.load_model(str(MODEL_PATHS["classifier"]))
    
    scaler = None
    scaler_path = MODEL_PATHS.get("scaler")
    if scaler_path and scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    with open(MODEL_PATHS["meta"]) as f:
        meta = json.load(f)
    feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
    use_scaler = scaler is not None

    print(f"Using {len(feature_order)} features: {feature_order}")
    print(f"Scaler: {'enabled' if use_scaler else 'disabled'}")

    # Build index for fast candidate lookup by postcode and INSEE
    print("Building candidate index by postcode/INSEE...")
    cand_by_postcode: Dict[str, List[tuple]] = {}
    cand_by_insee: Dict[str, List[tuple]] = {}
    for siret, c in candidates.items():
        cp = c.get("postcode")
        insee = c.get("insee")
        if cp:
            cand_by_postcode.setdefault(cp, []).append((siret, c))
        if insee:
            cand_by_insee.setdefault(insee, []).append((siret, c))
    print(f"  Index: {len(cand_by_postcode)} postcodes, {len(cand_by_insee)} INSEE codes")

    rows_out: List[dict] = []
    for r, crm_ctx in tqdm(zip(crm_records, crm_pre), total=len(crm_records), desc="Infer"):
        # Pool: use indexed lookup instead of iterating all candidates
        query_insee = r.get("insee")
        query_cp = r.get("postcode")
        
        # Prefer INSEE match, fallback to postcode
        if query_insee and query_insee in cand_by_insee:
            cand_list = cand_by_insee[query_insee]
        elif query_cp and query_cp in cand_by_postcode:
            cand_list = cand_by_postcode[query_cp]
        else:
            continue
        if not cand_list:
            continue

        # --- STAGE 1: Fast ranking (skip semantic features for speed) ---
        feats_stage1 = []
        for siret, c in cand_list:
            feat = make_features_from_preprocessed(crm_ctx, c, skip_semantic=True)
            feats_stage1.append(feat)
        
        X1 = pd.DataFrame(feats_stage1)[feature_order]
        Xs1 = scaler.transform(X1) if use_scaler else X1.values
        
        # Get raw ranking scores for all candidates
        rank_scores = ranker.predict(Xs1)
        
        # Pick top candidates for re-scoring
        stage1_top_n = min(200, len(rank_scores))
        top_n_idx = np.argsort(rank_scores)[::-1][:stage1_top_n]

        # Address-rescue: recover near-perfect address matches outside top-N
        rescue_indices = []
        top_n_set = set(top_n_idx)
        for i, feat in enumerate(feats_stage1):
            if i in top_n_set:
                continue
            addr_jaro = float(feat.get("addr_jaro", 0.0))
            street_name_jaro = float(feat.get("street_name_jaro", 0.0))
            street_number_diff = float(feat.get("street_number_diff", 9999))
            if addr_jaro >= 0.96 and street_name_jaro >= 0.95 and street_number_diff <= 2:
                rescue_indices.append(i)
        rescue_indices = rescue_indices[:50]
        if rescue_indices:
            top_n_idx = np.unique(np.concatenate([top_n_idx, np.array(rescue_indices, dtype=int)]))
        
        # --- STAGE 2: Full scoring with semantic features on top-N only ---
        feats_n = []
        cand_list_n = []
        semantic_pools = []
        for idx in top_n_idx:
            siret, c = cand_list[idx]
            # Reuse stage-1 features; only semantic differs between stage-1 and stage-2.
            feat = feats_stage1[idx]

            if semantic_enabled:
                cand_city_norm = c.get("_xgb_cached_city_norm") or normalize_text(c.get("city"))
                semantic_pools.append(
                    build_semantic_name_pool(
                        build_candidate_names(c),
                        crm_city_norm=crm_ctx.get("crm_city_norm", ""),
                        cand_city_norm=cand_city_norm,
                    )
                )

            feat["_siret"] = siret
            feat["_cand_name"] = primary_name(c) or f"SIRET {siret}"
            feat["_ul_denoms"] = [
                normalize_text(c.get("denomination_ul") or ""),
                normalize_text(c.get("denomination_usuelle_ul") or ""),
                normalize_text(c.get("sigle_ul") or ""),
            ]
            feat["_is_siege"] = bool(c.get("is_siege"))
            feats_n.append(feat)
            cand_list_n.append((siret, c))

        if semantic_enabled:
            sem = top2_semantic_similarities_batch(crm_ctx.get("crm_name_semantic", ""), semantic_pools)
            for feat, (sem_max, sem_second, sem_gap) in zip(feats_n, sem, strict=True):
                feat["name_semantic_max"] = sem_max
                feat["name_semantic_second"] = sem_second
                feat["name_semantic_gap"] = sem_gap
        
        X_n = pd.DataFrame(feats_n)[feature_order]
        Xs_n = scaler.transform(X_n) if use_scaler else X_n.values
        
        # predict_proba returns [P_neg, P_pos]
        class_probs = classifier.predict_proba(Xs_n)[:, 1]
        
        # Apply rerank rules on classifier probabilities (if enabled)
        scores = np.array(class_probs)
        if USE_RERANK_RULES:
            scores = apply_rerank_rules(scores, feats_n, r)
        
        # Final Top-K selection
        topk_idx = np.argsort(scores)[::-1][: args.top_k]
        shap_contribs = None
        X_n_shap = None
        Xs_n_shap = None
        if with_shap and len(topk_idx) > 0:
            booster = classifier.get_booster()
            X_n_shap = X_n.iloc[topk_idx]
            Xs_n_shap = Xs_n[topk_idx]
            dm = xgb.DMatrix(Xs_n_shap, feature_names=feature_order)
            shap_contribs = booster.predict(dm, pred_contribs=True)

        if len(topk_idx) > 0:
            top0 = topk_idx[0]
            siret0, cand0 = cand_list_n[top0]
            cand_name0 = primary_name(cand0) or f"SIRET {siret0}"
            print(
                f"[infer] crm_id={r.get('crm_id')} | {r.get('crm_name','')}"
                f" -> TOP1 {siret0} | {cand_name0}"
            )

        for rank, idx_k in enumerate(topk_idx, start=1):
            siret_k, cand_k = cand_list_n[idx_k]
            cand_name = primary_name(cand_k) or f"SIRET {siret_k}"
            etat_admin = str(cand_k.get("etat_admin") or "").strip().upper() or None
            candidate_state = None
            if etat_admin is not None:
                candidate_state = "FERME" if etat_admin == "F" else "OUVERT"
            shap_json = None
            if shap_contribs is not None and X_n_shap is not None and Xs_n_shap is not None:
                contrib_row = shap_contribs[rank - 1]
                bias = float(contrib_row[-1])
                vals = contrib_row[:-1]
                margin = float(contrib_row.sum())
                prob_from_margin = _sigmoid(margin)

                top_n = max(0, int(args.shap_top_n))
                top_feats = []
                if top_n:
                    order = np.argsort(np.abs(vals))[::-1][:top_n]
                    raw_row = X_n_shap.iloc[rank - 1]
                    scaled_row = Xs_n_shap[rank - 1]
                    for fi in order:
                        feat_name = feature_order[int(fi)]
                        top_feats.append(
                            {
                                "feature": feat_name,
                                "shap": float(vals[int(fi)]),
                                "value": float(raw_row[feat_name]),
                                "model_input": float(scaled_row[int(fi)]),
                            }
                        )
                shap_json = json.dumps(
                    {
                        "bias": bias,
                        "margin": margin,
                        "prob_from_margin": prob_from_margin,
                        "top": top_feats,
                    },
                    ensure_ascii=False,
                )
            rows_out.append({
                "crm_id": r.get("crm_id"),
                "crm_name": r["crm_name"],
                "crm_address": r.get("crm_address"),
                "crm_postcode": r.get("postcode"),
                "crm_city": r.get("crm_city"),
                "siret_candidate": siret_k,
                "score": float(scores[idx_k]),
                "candidate_name": cand_name,
                "candidate_addr": build_address(cand_k),
                "candidate_city": cand_k.get("city"),
                "candidate_postcode": cand_k.get("postcode"),
                "candidate_insee": cand_k.get("insee"),
                "candidate_state": candidate_state,
                "candidate_last_treatment_date": cand_k.get("last_treatment_date"),
                "rank": rank,
                "shap": shap_json,
            })

    out_df = pd.DataFrame(rows_out).sort_values(["crm_name", "rank", "score"], ascending=[True, True, False])
    out_path = args.output_path
    if args.write_csv:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved top-{args.top_k} results to {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
