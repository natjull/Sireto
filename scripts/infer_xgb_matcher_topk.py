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

import json
import pickle
import sqlite3
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from xgboost import XGBClassifier, XGBRanker

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    build_address,
    make_features,
    normalize_text,
)
from src.xgb_matcher.naming import primary_name

# Config
CRM_PATH = Path("data/testcrm/data_56_subset_corbas_decines.csv")
PARQUET_PATH = Path("data/StockEtablissement_utf8.parquet")
UNITE_LEGALE_PATH = Path("data/StockUniteLegale_utf8.parquet")  # optional
HARVEST_DB = Path("data/harvest_full.sqlite")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_matcher.json"
SCALER_PATH = MODEL_DIR / "xgb_matcher_scaler.pkl"
FEATURE_META_PATH = MODEL_DIR / "xgb_matcher_features.json"
TOP_K = 5
BATCH_SIZE = 100_000

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
        if addr_jaro >= 0.97 and (street_number_diff == 0 or street_number_diff == 9999) and name_jaro_max >= 0.8:
            delta += BOOST_PERFECT_ADDR
            logger.info("rerank_adjust|%s|%s|D1_PERFECT_ADDR|+%.3f", crm_row.get("crm_name"), siret, BOOST_PERFECT_ADDR)
        elif addr_jaro <= 0.70 and street_number_diff >= 5 and name_jaro_max < 0.6:
            delta -= PENALTY_WRONG_ADDR
            logger.info("rerank_adjust|%s|%s|D1_WRONG_ADDR|-%.3f", crm_row.get("crm_name"), siret, PENALTY_WRONG_ADDR)

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
            cand = {
                "siret": r["siret"],
                "siren": r.get("siren"),
                "denomination": r.get("denominationUsuelleEtablissement"),
                "enseigne1": r.get("enseigne1Etablissement"),
                "enseigne2": r.get("enseigne2Etablissement"),
                "enseigne3": r.get("enseigne3Etablissement"),
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    print("Loading CRM subset...")
    crm = load_crm(CRM_PATH)
    postcodes = set(crm["postcode"].dropna().unique()) if "postcode" in crm else set()
    insee = set(crm["insee"].dropna().unique()) if "insee" in crm else set()
    print(f"Postcodes of interest: {len(postcodes)} | INSEE codes: {len(insee)}")

    print("Building candidate pool from parquet (filtered by CP/INSEE)...")
    candidates = load_candidates_for_locations(postcodes, insee)
    print(f"Candidates loaded: {len(candidates)}")

    print("Loading model artifacts...")
    # The saved default model may be a classifier or a ranker depending on training.
    try:
        clf = XGBClassifier()
        clf.load_model(MODEL_PATH)
        is_ranker = False
    except TypeError:
        clf = XGBRanker()
        clf.load_model(MODEL_PATH)
        is_ranker = True
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_META_PATH) as f:
        meta = json.load(f)
    feature_order = meta["feature_order"]

    print(f"Using {len(feature_order)} features: {feature_order}")

    rows_out: List[dict] = []
    for _, r in crm.iterrows():
        # Pool: prefer same INSEE, else same postcode
        cand_list = []
        for siret, c in candidates.items():
            same_insee = c.get("insee") and r.get("insee") and c["insee"] == r["insee"]
            same_cp = c.get("postcode") and r.get("postcode") and c["postcode"] == r["postcode"]
            if same_insee or same_cp:
                cand_list.append((siret, c))
        if not cand_list:
            continue

        # Compute features using shared module
        feats = []
        for siret, c in cand_list:
            feat = make_features(r, c)
            feat["_siret"] = siret
            feat["_cand_name"] = primary_name(c) or f"SIRET {siret}"
            feats.append(feat)
        X = pd.DataFrame(feats)[feature_order]
        Xs = scaler.transform(X)
        raw_scores = clf.predict(Xs) if is_ranker else clf.predict_proba(Xs)[:, 1]

        # Pure model scores - no post-adjustments
        scores = np.array(raw_scores)
        # Apply rerank rules
        scores = apply_rerank_rules(scores, feats, r)

        topk_idx = np.argsort(scores)[::-1][:TOP_K]
        for rank, idx_k in enumerate(topk_idx, start=1):
            siret_k, cand_k = cand_list[idx_k]
            cand_name = primary_name(cand_k) or f"SIRET {siret_k}"
            rows_out.append({
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
                "rank": rank,
            })

    out_df = pd.DataFrame(rows_out).sort_values(["crm_name", "rank", "score"], ascending=[True, True, False])
    out_path = Path("reports/xgb_infer_topk.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved top-{TOP_K} results to {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
