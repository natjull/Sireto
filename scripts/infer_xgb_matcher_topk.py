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
import sys
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
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_matcher.json"
SCALER_PATH = MODEL_DIR / "xgb_matcher_scaler.pkl"
FEATURE_META_PATH = MODEL_DIR / "xgb_matcher_features.json"
TOP_K = 5
BATCH_SIZE = 100_000


# ---------- Reranking helpers ----------


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

    return mapping


# ---------- Main inference ----------


def main():
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
        feats = [make_features(r, c) for _, c in cand_list]
        X = pd.DataFrame(feats)[feature_order]
        Xs = scaler.transform(X)
        raw_scores = clf.predict(Xs) if is_ranker else clf.predict_proba(Xs)[:, 1]

        # Post-adjustments (inference-only):
        # - pénalise les candidats sans nom
        # - bonifie légèrement les adresses parfaitement identiques selon le nom
        # - bonifie les candidats avec nom UL quand le nom établissement est absent
        # - bonus token pour les portmanteaux (ex: BOXING dans DigitBoxing)
        crm_name_raw = r.get("crm_name", "")
        
        # First pass: compute base adjustments
        adjusted_scores = []
        for idx, (sc, f) in enumerate(zip(raw_scores, feats)):
            adj = sc
            _, cand = cand_list[idx]
            
            # Penalty for candidates without any name
            if f.get("has_any_name", 1.0) == 0.0:
                adj *= 0.9
            
            # Bonus for perfect address match scaled by name similarity
            addr_quality = f.get("addr_jaro", 0.0)
            street_match = f.get("street_number_diff", 9999) == 0
            if addr_quality == 1.0 and street_match:
                adj += 0.05 * f.get("name_jaro_max", 0.0)
            
            # UL name boost: when etablissement name is weak/absent but UL name matches
            name_sim_etab = f.get("name_sim_max_etab", 0.0)
            name_sim_ul = f.get("name_sim_max_ul", 0.0)
            
            if name_sim_etab < 0.35 and name_sim_ul > 0.3:
                if addr_quality == 1.0 and street_match:
                    # Perfect address + UL name = very strong signal
                    ul_boost = name_sim_ul * 0.80 + 0.30
                else:
                    # Good address + UL name = moderate boost
                    ul_boost = name_sim_ul * addr_quality * 0.40
                adj += ul_boost
            
            # Token substring bonus for portmanteau names
            if addr_quality >= 0.85:
                token_bonus = compute_ul_token_bonus(crm_name_raw, cand)
                if token_bonus > 0:
                    adj += token_bonus * 0.30
            
            adjusted_scores.append((adj, addr_quality, street_match, name_sim_ul))
        
        # Second pass: apply address-relative anchor bonus
        # If there's a candidate with perfect address and meaningful UL name,
        # heavily penalize candidates with much worse addresses
        perfect_addr_candidates = [
            (i, s, ul) for i, (s, aq, sm, ul) in enumerate(adjusted_scores) 
            if aq == 1.0 and sm and ul > 0.3
        ]
        
        scores = []
        if perfect_addr_candidates:
            # We have at least one perfect-address candidate with UL name
            best_perfect_score = max(s for _, s, _ in perfect_addr_candidates)
            
            for idx, (adj, addr_quality, street_match, name_sim_ul) in enumerate(adjusted_scores):
                if addr_quality == 1.0 and street_match:
                    # Perfect address candidates get a boost relative to best imperfect scorer
                    # This ensures they compete favorably
                    if name_sim_ul > 0.3:
                        # Has UL name - ensure it's competitive with wrong-address high scorers
                        bonus = max(0, (best_perfect_score - adj) * 0.3 + 0.5)
                        adj += bonus
                else:
                    # Non-perfect address candidates get penalized if there are good
                    # perfect-address alternatives
                    address_penalty = (1.0 - addr_quality) * 1.2
                    adj -= address_penalty
                
                scores.append(adj)
        else:
            scores = [adj for adj, _, _, _ in adjusted_scores]
        
        scores = np.array(scores)

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
