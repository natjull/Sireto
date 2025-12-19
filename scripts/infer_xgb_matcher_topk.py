"""
Offline scoring of the trained XGBoost matcher on a CRM CSV, producing top-k candidates per row.

Inputs:
  - CRM file: data/testcrm/data_56_subset_corbas_decines.csv
  - Harvest DB: data/harvest_full.sqlite
  - Trained artifacts: models/xgb_matcher.json, models/xgb_matcher_scaler.pkl, models/xgb_matcher_features.json

Strategy:
  1) Read CRM, collect the set of codePostal and codeCommune present.
  2) Query harvest_full.sqlite, keeping only rows whose codePostal or codeCommune is in those sets.
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
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
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
HARVEST_DB = Path("data/harvest_full.sqlite")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_matcher.json"
SCALER_PATH = MODEL_DIR / "xgb_matcher_scaler.pkl"
FEATURE_META_PATH = MODEL_DIR / "xgb_matcher_features.json"
TOP_K = 5


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


def _chunked(seq: List[str], size: int = 900) -> List[List[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _parse_enseignes(raw: str | None) -> List[str]:
    if not raw:
        return []
    s = str(raw).strip()
    if not s or s.upper() == "[ND]":
        return []
    if s.startswith("["):
        try:
            values = json.loads(s)
            return [v for v in values if v and str(v).upper() != "[ND]"]
        except json.JSONDecodeError:
            return [s]
    return [s]


def _load_pm_dirigeant_names(con: sqlite3.Connection, sirens: set[str]) -> Dict[str, List[str]]:
    if not sirens:
        return {}
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
    return {k: sorted(v) for k, v in out.items() if v}


def _load_siren_etab_names(con: sqlite3.Connection, sirens: set[str]) -> Dict[str, List[str]]:
    if not sirens:
        return {}
    cur = con.cursor()
    out: Dict[str, set[str]] = {s: set() for s in sirens}
    for chunk in _chunked(sorted(sirens)):
        q = ",".join("?" for _ in chunk)
        cur.execute(
            f"""
            SELECT siren, nom_commercial, liste_enseignes
            FROM etablissements
            WHERE siren IN ({q})
            """,
            chunk,
        )
        for siren, nom_commercial, liste_enseignes in cur.fetchall():
            if siren not in out:
                continue
            names = []
            if nom_commercial:
                names.append(nom_commercial)
            names.extend(_parse_enseignes(liste_enseignes))
            for n in names:
                if n:
                    out[siren].add(n)
    return {k: sorted(v) for k, v in out.items() if v}


def load_candidates_for_locations(postcodes: set[str], insee: set[str]) -> Dict[str, dict]:
    """
    Load candidates from harvest_full.sqlite, filtered by location.

    Excludes candidates without any name field (matching would be meaningless).
    """
    if not HARVEST_DB.exists() or (not postcodes and not insee):
        return {}
    con = sqlite3.connect(HARVEST_DB)
    cur = con.cursor()
    mapping: Dict[str, dict] = {}
    sirens: set[str] = set()

    where_clauses = []
    params: List[str] = []
    if postcodes:
        q = ",".join("?" for _ in postcodes)
        where_clauses.append(f"e.code_postal IN ({q})")
        params.extend(sorted(postcodes))
    if insee:
        q = ",".join("?" for _ in insee)
        where_clauses.append(f"e.commune IN ({q})")
        params.extend(sorted(insee))

    where_sql = " OR ".join(where_clauses)
    cur.execute(
        f"""
        SELECT
            e.siret,
            e.siren,
            e.adresse,
            e.code_postal,
            e.libelle_commune,
            e.commune,
            e.nom_commercial,
            e.liste_enseignes,
            s.numero_voie,
            s.type_voie,
            s.libelle_voie,
            s.complement_adresse,
            s.code_postal AS s_code_postal,
            s.libelle_commune AS s_libelle_commune,
            s.commune AS s_commune,
            ent.nom_complet,
            ent.nom_raison_sociale,
            ent.sigle,
            ent.nature_juridique
        FROM etablissements e
        LEFT JOIN sieges s ON s.siret = e.siret
        LEFT JOIN entreprises ent ON ent.siren = e.siren
        WHERE {where_sql}
        """,
        params,
    )
    for r in cur.fetchall():
        (
            siret,
            siren,
            adresse,
            e_cp,
            e_city,
            e_insee,
            nom_commercial,
            liste_enseignes,
            numero_voie,
            type_voie,
            libelle_voie,
            complement_adresse,
            s_cp,
            s_city,
            s_insee,
            nom_complet,
            nom_raison_sociale,
            sigle,
            nature_juridique,
        ) = r

        names = []
        if nom_commercial:
            names.append(nom_commercial)
        names.extend(_parse_enseignes(liste_enseignes))
        seen = set()
        enseignes = []
        for n in names:
            if n and n not in seen:
                enseignes.append(n)
                seen.add(n)
        enseigne1 = enseignes[0] if len(enseignes) > 0 else None
        enseigne2 = enseignes[1] if len(enseignes) > 1 else None
        enseigne3 = enseignes[2] if len(enseignes) > 2 else None

        cand = {
            "siret": siret,
            "siren": siren,
            "denomination": None,
            "enseigne1": enseigne1,
            "enseigne2": enseigne2,
            "enseigne3": enseigne3,
            "numeroVoie": numero_voie,
            "typeVoie": type_voie,
            "libelleVoie": libelle_voie,
            "complementAdresse": complement_adresse or adresse,
            "postcode": s_cp or e_cp,
            "city": s_city or e_city,
            "insee": s_insee or e_insee,
            "cj_ul": nature_juridique,
            "sigle_ul": sigle,
            "denomination_ul": nom_raison_sociale or nom_complet,
            "denomination_usuelle_ul": nom_complet if nom_complet and nom_complet != nom_raison_sociale else None,
        }
        if siren:
            sirens.add(str(siren))
        mapping[str(siret)] = cand

    pm_map = _load_pm_dirigeant_names(con, sirens)
    siren_names_map = _load_siren_etab_names(con, sirens)
    for siret, cand in mapping.items():
        siren = cand.get("siren")
        if siren:
            if siren in pm_map:
                cand["pm_dirigeant_names"] = pm_map[siren]
            if siren in siren_names_map:
                cand["siren_etab_names"] = siren_names_map[siren]

    con.close()
    return mapping


# ---------- Main inference ----------


def main():
    print("Loading CRM subset...")
    crm = load_crm(CRM_PATH)
    postcodes = set(crm["postcode"].dropna().unique()) if "postcode" in crm else set()
    insee = set(crm["insee"].dropna().unique()) if "insee" in crm else set()
    print(f"Postcodes of interest: {len(postcodes)} | INSEE codes: {len(insee)}")

    print("Building candidate pool from harvest_full.sqlite (filtered by CP/INSEE)...")
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
