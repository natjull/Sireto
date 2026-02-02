#!/usr/bin/env python3
"""
Two-stage XGBoost inference with multi-blocking and partitioned candidates.

Stage 1: Ranker (fast, no semantic)
Stage 2: Decider (full features + optional calibration)
"""
from __future__ import annotations

# CRITICAL: Enable semantic BEFORE any xgb_matcher imports (train/serve alignment)
import os
os.environ.setdefault("XGB_SEMANTIC_ENABLED", "1")

import argparse
import hashlib
import json
import logging
import pickle
import sys
import multiprocessing as mp
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.blocking import (
    address_hash,
    attach_address_density,
    dedupe_candidates,
    department_from_code,
    extract_numeric_tokens,
    build_tfidf_index,
    build_char_tfidf_index,
    prefilter_candidates_tfidf,
    build_address_hash_index,
    build_numeric_token_index,
    build_address_tfidf_index,
    prefilter_candidates_address_tfidf,
)
from src.xgb_matcher.features import (
    FEATURE_NAMES,
    FAST_RANKER_FEATURE_NAMES,
    build_address,
    build_semantic_name_pool,
    make_features_from_preprocessed,
    normalize_text,
    preprocess_crm_row,
    semantic_gate_allows,
    set_global_name_idf_map,
)
from src.xgb_matcher.candidates import compute_name_idf_map
from src.xgb_matcher.naming import build_candidate_names, primary_name
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.semantic import (
    batch_encode_texts,
    get_cache_stats,
    top2_semantic_similarities_batch,
    is_semantic_available,
    set_semantic_client,
    _model_name,
    _device,
)
from src.xgb_matcher.semantic_server import semantic_server_process, SemanticClient

# LRU dept_cache: keeps at most MAX_DEPT_CACHE departments in memory
MAX_DEPT_CACHE = 5
# Maximum number of parallel workers for multi-process inference
MAX_WORKERS = 4


# Calibrator wrapper classes (pickle compatibility)
class IsotonicCalibrator:
    def __init__(self, base_estimator, iso_reg):
        self.base_estimator = base_estimator
        self.iso_reg = iso_reg

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.iso_reg.predict(proba)
        calibrated = np.clip(calibrated, 0, 1)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SigmoidCalibrator:
    def __init__(self, base_estimator, lr):
        self.base_estimator = base_estimator
        self.lr = lr

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.lr.predict_proba(proba.reshape(-1, 1))
        return calibrated

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)



@dataclass
class DeptCacheItem:
    candidates: List[dict]
    tfidf_vec: Any  # TfidfVectorizer
    tfidf_mat: Any  # sparse matrix
    tfidf_names: List[str]
    addr_tfidf_vec: Any = None # New: Address TF-IDF
    addr_tfidf_mat: Any = None
    char_vec: Any = None
    char_mat: Any = None
    addr_index: Dict[str, List[int]] | None = None
    num_index: Dict[str, List[int]] | None = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer XGB two-stage matcher (partitioned + multi-blocking).")
    p.add_argument("--crm-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, default=Path("reports/xgb_two_stage_topk.csv"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v4_active"))
    p.add_argument("--pool-mode", choices=["insee_then_postcode", "union", "multi"], default="insee_then_postcode")
    p.add_argument("--prefilter-k", type=int, default=500)
    p.add_argument("--min-candidates", type=int, default=150)
    p.add_argument("--stage1-top-n", type=int, default=200)
    p.add_argument("--char-top-k", type=int, default=200)
    p.add_argument("--tfidf-name-mode", choices=["primary", "bag"], default="bag")
    p.add_argument("--siren-siblings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dept-prefilter-k", type=int, default=200)
    p.add_argument("--max-dept-candidates", type=int, default=50000)
    p.add_argument(
        "--semantic-retrieval-k",
        type=int,
        default=int(os.getenv("XGB_SEMANTIC_RETRIEVAL_K", "0")),
        help="Top-K candidates to add from semantic retrieval (department pool).",
    )
    p.add_argument(
        "--semantic-retrieval-min-sim",
        type=float,
        default=float(os.getenv("XGB_SEMANTIC_RETRIEVAL_MIN_SIM", "0.55")),
        help="Minimum cosine similarity for semantic retrieval candidates.",
    )
    p.add_argument("--drop-unnamed", action="store_true", default=True)
    p.add_argument("--exclude-closed", action="store_true", default=False)
    p.add_argument("--export-routing-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--allow-no-semantic",
        action="store_true",
        default=False,
        help="Allow running without semantic embeddings (semantic features will be zero).",
    )
    p.add_argument("--meta-path", type=Path, default=None)
    p.add_argument("--ranker-model", type=Path, default=None)
    p.add_argument("--ranker-fast-model", type=Path, default=None)
    p.add_argument("--decider-model", type=Path, default=None)
    p.add_argument("--calibrator-path", type=Path, default=None)
    # Debug GT mode: trace where GT is lost in the pipeline
    p.add_argument("--debug-gt", action="store_true", default=False,
                   help="Enable GT debug mode: track where ground truth is pruned.")
    p.add_argument("--gt-column", type=str, default="ground_truth_siret",
                   help="Column name for ground truth SIRET in CRM file.")
    p.add_argument("--debug-gt-output", type=Path, default=None,
                   help="Path for GT debug CSV output (default: <output>_gt_debug.csv).")
    return p.parse_args()


def load_crm(path: Path) -> pd.DataFrame:
    import csv

    sample = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ";"

    df = pd.read_csv(path, sep=delimiter, dtype=str)

    # If file already has standardized columns, keep as-is
    
    df = df.rename(
        columns={
            "Client final": "crm_name",
            "Adresse": "crm_address",
            "Commune": "crm_city",
            "Code Postal": "postcode",
            "Code INSEE": "insee", "crm_insee": "insee", "crm_cp": "postcode", "crm_adresse": "crm_address", "crm_commune": "crm_city",
        }
    )
    for col in ["postcode", "insee"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and x not in ["", "nan"] else x
            )
    if "crm_id" not in df.columns:
        df["crm_id"] = df.index
    print(f"DEBUG: df columns: {df.columns.tolist()}")
    print(df.head())

    return df


def _find_latest_meta(model_dir: Path) -> Path | None:
    candidates = sorted(model_dir.glob("xgb_two_stage_meta_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _load_models_from_meta(meta_path: Path) -> Dict:
    with open(meta_path) as f:
        meta = json.load(f)
    return meta


def has_name_evidence(feat_row: dict) -> int:
    return int(
        (feat_row.get("name_jaro_max", 0.0) >= 0.60)
        or (feat_row.get("name_token_overlap_max", 0.0) >= 0.30)
        or (feat_row.get("name_sim_max_etab", 0.0) >= 0.60)
        or (feat_row.get("name_crm_contains_cand_max", 0.0) >= 0.60)
        or (feat_row.get("numeric_token_match", 0.0) >= 0.50)
    )


def _semantic_top_k_candidates(
    crm_name: str,
    candidates: List[dict],
    top_k: int,
    min_sim: float,
) -> List[dict]:
    if top_k <= 0:
        return []
    if not os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1":
        return []
    if not crm_name or not candidates:
        return []
    names = [primary_name(c) or "" for c in candidates]
    texts = [crm_name] + [n for n in names if n]
    embeddings = batch_encode_texts(texts)
    crm_emb = embeddings.get(crm_name)
    if crm_emb is None:
        return []
    scored: List[Tuple[float, int]] = []
    for idx, name in enumerate(names):
        if not name:
            continue
        emb = embeddings.get(name)
        if emb is None:
            continue
        sim = float(np.dot(crm_emb, emb))
        if sim >= min_sim:
            scored.append((sim, idx))
    if not scored:
        return []
    scored.sort(reverse=True)
    top_idx = [i for _, i in scored[:top_k]]
    return [candidates[i] for i in top_idx]


def _apply_candidate_filters(
    candidates: List[dict],
    drop_unnamed: bool,
    exclude_closed: bool,
) -> List[dict]:
    out = candidates
    if drop_unnamed:
        out = [c for c in out if build_candidate_names(c)]
    if exclude_closed:
        out = [c for c in out if str(c.get("etat_admin") or "").strip().upper() != "F"]
    return out


def _stable_seed(value: str) -> int:
    base = value or ""
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()
    return int(digest, 16) & 0x7FFFFFFF


def _deterministic_sample(items: List[dict], k: int, seed_value: int) -> List[dict]:
    if k <= 0 or not items:
        return []
    import random
    rng = random.Random(seed_value)
    return rng.sample(items, min(k, len(items)))


def _is_closed_candidate(cand: dict) -> bool:
    return str(cand.get("etat_admin") or "").strip().upper() == "F"


def _promote_open_over_closed(topk_idx: List[int], cand_list_n: List[Tuple[str, dict]]) -> List[int]:
    """Promote open establishments when same SIREN has open/closed in top-k."""
    if not topk_idx:
        return topk_idx

    order = list(topk_idx)
    open_pos: Dict[str, int] = {}
    closed_pos: Dict[str, int] = {}

    for pos, idx in enumerate(order):
        cand = cand_list_n[idx][1]
        siren = str(cand.get("siren") or "").strip()
        if not siren:
            continue
        if _is_closed_candidate(cand):
            closed_pos.setdefault(siren, pos)
        else:
            open_pos.setdefault(siren, pos)

    for siren, c_pos in closed_pos.items():
        o_pos = open_pos.get(siren)
        if o_pos is None or c_pos >= o_pos:
            continue
        order[c_pos], order[o_pos] = order[o_pos], order[c_pos]

    return order


@dataclass
class GTDebugInfo:
    """Debug info tracking where GT SIRET was lost in the pipeline."""
    gt_siret: str
    in_base_pool: bool = False
    in_filtered_pool: bool = False
    in_tfidf_pool: bool = False
    in_stage1_topn: bool = False
    in_stage2_topk: bool = False
    stage1_rank: int = -1
    stage2_rank: int = -1
    base_pool_size: int = 0
    filtered_pool_size: int = 0
    tfidf_pool_size: int = 0
    stage1_topn_size: int = 0
    loss_reason: str = ""


def _check_gt_in_list(candidates: List[dict], gt_siret: str | None) -> bool:
    """Check if GT SIRET is in candidate list."""
    if not gt_siret:
        return False
    gt_norm = str(gt_siret).zfill(14)
    for c in candidates:
        if str(c.get("siret") or "").zfill(14) == gt_norm:
            return True
    return False


def _get_dept_cache_item(
    *,
    dept_cache: OrderedDict[str, DeptCacheItem],
    dept_key: str | None,
    store: PartitionedCandidateStore,
    insee: str | None,
    postcode: str | None,
    drop_unnamed: bool,
    exclude_closed: bool,
    max_dept_candidates: int,
    tfidf_name_mode: str,
    siren_siblings: bool,
) -> DeptCacheItem | None:
    if not dept_key:
        return None
    cached = dept_cache.get(dept_key)
    if cached is not None:
        dept_cache.move_to_end(dept_key)
        return cached

    dept_candidates = store.load_by_department(insee, postcode)
    if not dept_candidates:
        return None
    dept_candidates = _apply_candidate_filters(dept_candidates, drop_unnamed, exclude_closed)
    if max_dept_candidates and len(dept_candidates) > max_dept_candidates:
        dept_candidates = dept_candidates[:max_dept_candidates]

    vec, mat, names = build_tfidf_index(
        dept_candidates,
        name_mode=tfidf_name_mode,
        siren_siblings=siren_siblings,
    )
    # New: Build address TF-IDF index
    addr_vec, addr_mat = build_address_tfidf_index(dept_candidates)
    
    char_vec, char_mat = (None, None)
    if names:
        char_vec, char_mat = build_char_tfidf_index(names)

    addr_index = build_address_hash_index(dept_candidates)
    num_index = build_numeric_token_index(dept_candidates)

    item = DeptCacheItem(
        candidates=dept_candidates,
        tfidf_vec=vec,
        tfidf_mat=mat,
        tfidf_names=names,
        addr_tfidf_vec=addr_vec,
        addr_tfidf_mat=addr_mat,
        char_vec=char_vec,
        char_mat=char_mat,
        addr_index=addr_index,
        num_index=num_index,
    )
    dept_cache[dept_key] = item
    # LRU eviction
    while len(dept_cache) > MAX_DEPT_CACHE:
        dept_cache.popitem(last=False)
    return item


def _build_candidate_pool_with_debug(
    store: PartitionedCandidateStore,
    crm_row: dict,
    crm_pre: dict,
    pool_mode: str,
    prefilter_k: int,
    dept_prefilter_k: int,
    max_dept_candidates: int,
    semantic_retrieval_k: int,
    semantic_retrieval_min_sim: float,
    drop_unnamed: bool,
    exclude_closed: bool,
    dept_cache: OrderedDict[str, DeptCacheItem],
    tfidf_name_mode: str = "bag",
    siren_siblings: bool = True,
    min_candidates: int = 150,
    char_top_k: int = 200,
    gt_siret: str | None = None,
) -> Tuple[List[dict], GTDebugInfo | None]:
    """Build candidate pool with optional GT debug tracking."""
    debug = GTDebugInfo(gt_siret=gt_siret or "") if gt_siret else None
    
    insee = crm_row.get("insee")
    postcode = crm_row.get("postcode")
    crm_name = crm_row.get("crm_name", "")

    # Step 1: Load base candidates
    base_candidates: List[dict] = []
    if pool_mode == "insee_then_postcode":
        if insee:
            base_candidates = store.load_by_insee(insee)
        if not base_candidates and postcode:
            base_candidates = store.load_by_postcode(postcode)
    else:
        base_candidates = store.load_by_insee(insee) + store.load_by_postcode(postcode)

    if debug:
        debug.base_pool_size = len(base_candidates)
        debug.in_base_pool = _check_gt_in_list(base_candidates, gt_siret)

    # Step 2: Apply filters (drop_unnamed, exclude_closed)
    filtered = _apply_candidate_filters(base_candidates, drop_unnamed, exclude_closed)
    pool = dedupe_candidates(filtered)

    if debug:
        debug.filtered_pool_size = len(pool)
        debug.in_filtered_pool = bool(gt_siret and str(gt_siret).zfill(14) in {str(s).zfill(14) for s in pool.keys()})
        if debug.in_base_pool and not debug.in_filtered_pool:
            if exclude_closed:
                debug.loss_reason = "FILTERED_CLOSED"
            elif drop_unnamed:
                debug.loss_reason = "FILTERED_UNNAMED"
            else:
                debug.loss_reason = "FILTERED_OTHER"

    whitelisted_sirets = set()
    
    # ── Universal Rescue: Apply Whitelist Logic (Address Hash + Numeric Tokens) ──
    # Previously only active in pool_mode='multi', now active everywhere to catch 'bad name / good address'.
    
    # Load or build indices for the current pool (optimization: only build if needed)
    dept_candidates = list(pool.values())  # Use filtered pool as base for rescue
    addr_index = None
    num_index = None
    
    # Check if we are in multi mode with a cache hit, otherwise build ad-hoc
    cache_item = None
    if pool_mode == "multi":
        dept_key = department_from_code(insee, postcode)
        cache_item = _get_dept_cache_item(
            dept_cache=dept_cache,
            dept_key=dept_key,
            store=store,
            insee=insee,
            postcode=postcode,
            drop_unnamed=drop_unnamed,
            exclude_closed=exclude_closed,
            max_dept_candidates=max_dept_candidates,
            tfidf_name_mode=tfidf_name_mode,
            siren_siblings=siren_siblings,
        )
    
    # Use cached indices if available (multi mode), otherwise build on-the-fly (insee mode)
    if cache_item:
        dept_candidates = cache_item.candidates
        addr_index = cache_item.addr_index
        num_index = cache_item.num_index
    else:
        # For insee mode, indices are cheap to build on the small local pool
        if not addr_index:
            addr_index = build_address_hash_index(dept_candidates)
        if not num_index:
            num_index = build_numeric_token_index(dept_candidates)

    extra_rescue: List[dict] = []

    # 1. Address Hash Rescue
    addr_h = address_hash(crm_pre.get("crm_street_num"), crm_pre.get("crm_street_name"))
    if addr_h and addr_index:
        addr_indices = addr_index.get(addr_h, [])
        for idx in addr_indices:
            if idx < len(dept_candidates):
                cand = dept_candidates[idx]
                extra_rescue.append(cand)
                siret = str(cand.get("siret") or "")
                if siret:
                    whitelisted_sirets.add(siret)

    # 2. Numeric Token Rescue
    numeric_tokens = extract_numeric_tokens(crm_name)
    if numeric_tokens and num_index:
        num_indices: set[int] = set()
        for token in numeric_tokens:
            num_indices.update(num_index.get(token, []))
        for idx in sorted(num_indices):
            if idx < len(dept_candidates):
                cand = dept_candidates[idx]
                extra_rescue.append(cand)
                siret = str(cand.get("siret") or "")
                if siret:
                    whitelisted_sirets.add(siret)

    # Add rescued candidates to the pool
    for cand in extra_rescue:
        siret = str(cand.get("siret") or "")
        if siret:
            pool[siret] = cand

    # ── End Rescue ──

    # Additional Multi-mode specific steps (Dept-wide TF-IDF + Semantic)
    if pool_mode == "multi" and cache_item:
        extra: List[dict] = []
        if dept_candidates and (dept_prefilter_k > 0):
            vec = cache_item.tfidf_vec
            mat = cache_item.tfidf_mat
            names = cache_item.tfidf_names
            char_vec = cache_item.char_vec
            char_mat = cache_item.char_mat

            if vec is not None and mat is not None:
                idx = prefilter_candidates_tfidf(
                    crm_name,
                    vec,
                    mat,
                    min(dept_prefilter_k, len(dept_candidates)),
                    cand_names=names,
                    char_top_k=min(char_top_k, dept_prefilter_k),
                    char_vectorizer=char_vec,
                    char_matrix=char_mat,
                )
                extra.extend([dept_candidates[i] for i in idx])

        if dept_candidates and semantic_retrieval_k > 0:
            sem_hits = _semantic_top_k_candidates(
                crm_name,
                dept_candidates,
                semantic_retrieval_k,
                semantic_retrieval_min_sim,
            )
            extra.extend(sem_hits)

        if max_dept_candidates and len(extra) > max_dept_candidates:
            extra = extra[:max_dept_candidates]

        for cand in extra:
            siret = str(cand.get("siret") or "")
            if siret:
                pool[siret] = cand


    pool_candidates = list(pool.values())
    candidates_dict = {str(c.get("siret") or ""): c for c in pool_candidates if c.get("siret")}
    if candidates_dict:
        idf_map, default_idf = compute_name_idf_map(candidates_dict)
    else:
        idf_map, default_idf = {}, 0.0
    set_global_name_idf_map(idf_map, default_idf)
    attach_address_density(pool_candidates)

    candidates = pool_candidates
    
    # Step 3: TF-IDF prefilter (Issue 1: use Variant C settings)
    pre_tfidf_size = len(candidates)
    effective_prefilter_k = prefilter_k
    if pool_mode == "insee_then_postcode":
        effective_prefilter_k = 0
    if effective_prefilter_k and len(candidates) > effective_prefilter_k:
        vec, mat, names = build_tfidf_index(candidates, name_mode=tfidf_name_mode, siren_siblings=siren_siblings)
        if vec is not None and mat is not None:
            idx = prefilter_candidates_tfidf(
                crm_name,
                vec,
                mat,
                effective_prefilter_k,
                cand_names=names,
                char_top_k=min(char_top_k, effective_prefilter_k),
            )
            tfidf_cands = [candidates[i] for i in idx]
            tfidf_sirets = {str(c.get("siret")) for c in tfidf_cands if c.get("siret")}

            # Merge TF-IDF results with whitelisted extras
            final_cands = list(tfidf_cands)
            for cand in pool.values():
                siret = str(cand.get("siret") or "")
                if siret in whitelisted_sirets and siret not in tfidf_sirets:
                    final_cands.append(cand)

            if len(final_cands) >= min_candidates:
                candidates = final_cands
            else:
                final_sirets = {str(c.get("siret") or "") for c in final_cands if c.get("siret")}
                remaining = [c for c in candidates if str(c.get("siret") or "") not in final_sirets]
                needed = min(min_candidates, effective_prefilter_k) - len(final_cands)
                if needed > 0 and remaining:
                    seed = _stable_seed(str(crm_row.get("crm_id") or crm_row.get("crm_name") or ""))
                    random_extra = _deterministic_sample(remaining, needed, seed)
                    candidates = final_cands + random_extra
                else:
                    candidates = final_cands

    if debug:
        debug.tfidf_pool_size = len(candidates)
        debug.in_tfidf_pool = _check_gt_in_list(candidates, gt_siret)
        if debug.in_filtered_pool and not debug.in_tfidf_pool:
            debug.loss_reason = "PRUNED_BY_TFIDF"

    return candidates, debug


def _build_candidate_pool(
    store: PartitionedCandidateStore,
    crm_row: dict,
    crm_pre: dict,
    pool_mode: str,
    prefilter_k: int,
    dept_prefilter_k: int,
    max_dept_candidates: int,
    semantic_retrieval_k: int,
    semantic_retrieval_min_sim: float,
    drop_unnamed: bool,
    exclude_closed: bool,
    dept_cache: OrderedDict[str, DeptCacheItem],
    tfidf_name_mode: str = "bag",
    siren_siblings: bool = True,
    min_candidates: int = 150,
    char_top_k: int = 200,
) -> List[dict]:
    insee = crm_row.get("insee")
    postcode = crm_row.get("postcode")
    crm_name = crm_row.get("crm_name", "")

    base_candidates: List[dict] = []
    if pool_mode == "insee_then_postcode":
        if insee:
            base_candidates = store.load_by_insee(insee)
        if not base_candidates and postcode:
            base_candidates = store.load_by_postcode(postcode)
    else:
        base_candidates = store.load_by_insee(insee) + store.load_by_postcode(postcode)

    pool = dedupe_candidates(_apply_candidate_filters(base_candidates, drop_unnamed, exclude_closed))

    whitelisted_sirets = set()
    if pool_mode == "multi":
        dept_key = department_from_code(insee, postcode)
        cache_item = _get_dept_cache_item(
            dept_cache=dept_cache,
            dept_key=dept_key,
            store=store,
            insee=insee,
            postcode=postcode,
            drop_unnamed=drop_unnamed,
            exclude_closed=exclude_closed,
            max_dept_candidates=max_dept_candidates,
            tfidf_name_mode=tfidf_name_mode,
            siren_siblings=siren_siblings,
        )
        if cache_item:
            dept_candidates = cache_item.candidates
            extra: List[dict] = []

            addr_h = address_hash(crm_pre.get("crm_street_num"), crm_pre.get("crm_street_name"))
            if addr_h and cache_item.addr_index:
                addr_indices = cache_item.addr_index.get(addr_h, [])
                for idx in addr_indices:
                    cand = dept_candidates[idx]
                    extra.append(cand)
                    siret = str(cand.get("siret") or "")
                    if siret:
                        whitelisted_sirets.add(siret)

            numeric_tokens = extract_numeric_tokens(crm_name)
            if numeric_tokens and cache_item.num_index:
                num_indices: set[int] = set()
                for token in numeric_tokens:
                    num_indices.update(cache_item.num_index.get(token, []))
                for idx in sorted(num_indices):
                    cand = dept_candidates[idx]
                    extra.append(cand)
                    siret = str(cand.get("siret") or "")
                    if siret:
                        whitelisted_sirets.add(siret)

            if dept_candidates and (dept_prefilter_k > 0):
                vec = cache_item.tfidf_vec
                mat = cache_item.tfidf_mat
                names = cache_item.tfidf_names
                char_vec = cache_item.char_vec
                char_mat = cache_item.char_mat
                
                addr_vec = cache_item.addr_tfidf_vec
                addr_mat = cache_item.addr_tfidf_mat

                name_idx = []
                if vec is not None and mat is not None:
                    name_idx = prefilter_candidates_tfidf(
                        crm_name,
                        vec,
                        mat,
                        min(dept_prefilter_k, len(dept_candidates)),
                        cand_names=names,
                        char_top_k=min(char_top_k, dept_prefilter_k),
                        char_vectorizer=char_vec,
                        char_matrix=char_mat,
                    )
                
                addr_idx = []
                if addr_vec is not None and addr_mat is not None:
                    addr_idx = prefilter_candidates_address_tfidf(
                        crm_pre.get("crm_address"),
                        addr_vec,
                        addr_mat,
                        min(dept_prefilter_k, len(dept_candidates)),
                    )
                
                # Merge indices
                combined_idx = sorted(list(set(name_idx) | set(addr_idx)))
                extra.extend([dept_candidates[i] for i in combined_idx])

            if dept_candidates and semantic_retrieval_k > 0:
                sem_hits = _semantic_top_k_candidates(
                    crm_name,
                    dept_candidates,
                    semantic_retrieval_k,
                    semantic_retrieval_min_sim,
                )
                extra.extend(sem_hits)

            if max_dept_candidates and len(extra) > max_dept_candidates:
                extra = extra[:max_dept_candidates]

            for cand in extra:
                siret = str(cand.get("siret") or "")
                if siret:
                    pool[siret] = cand


    pool_candidates = list(pool.values())
    candidates_dict = {str(c.get("siret") or ""): c for c in pool_candidates if c.get("siret")}
    if candidates_dict:
        idf_map, default_idf = compute_name_idf_map(candidates_dict)
    else:
        idf_map, default_idf = {}, 0.0
    set_global_name_idf_map(idf_map, default_idf)
    attach_address_density(pool_candidates)

    candidates = pool_candidates
    # Issue 1: TF-IDF prefilter with Variant C settings for better recall
    effective_prefilter_k = prefilter_k
    if pool_mode == "insee_then_postcode":
        effective_prefilter_k = 0
    if effective_prefilter_k and len(candidates) > effective_prefilter_k:
        # Build Name Index
        vec, mat, names = build_tfidf_index(candidates, name_mode=tfidf_name_mode, siren_siblings=siren_siblings)
        
        # Build Address Index
        addr_vec, addr_mat = build_address_tfidf_index(candidates)
        
        name_idx = []
        if vec is not None and mat is not None:
            name_idx = prefilter_candidates_tfidf(
                crm_name,
                vec,
                mat,
                effective_prefilter_k,
                cand_names=names,
                char_top_k=min(char_top_k, effective_prefilter_k),
            )
            
        addr_idx = []
        if addr_vec is not None and addr_mat is not None:
            addr_idx = prefilter_candidates_address_tfidf(
                crm_pre.get("crm_address"),
                addr_vec,
                addr_mat,
                effective_prefilter_k,
            )
            
        # Merge results (Union of Name and Address retrieval)
        combined_idx = sorted(list(set(name_idx) | set(addr_idx)))
        
        if combined_idx:
            tfidf_cands = [candidates[i] for i in combined_idx]
            tfidf_sirets = {str(c.get("siret")) for c in tfidf_cands if c.get("siret")}

            # Merge TF-IDF results with whitelisted extras
            final_cands = list(tfidf_cands)
            for cand in pool.values():
                siret = str(cand.get("siret") or "")
                if siret in whitelisted_sirets and siret not in tfidf_sirets:
                    final_cands.append(cand)

            # Guarantee minimum candidates by combining TF-IDF + random padding
            if len(final_cands) >= min_candidates:
                candidates = final_cands
            else:
                tfidf_set = set(combined_idx)
                remaining = [c for i, c in enumerate(candidates) if i not in tfidf_set]
                needed = min(min_candidates, effective_prefilter_k) - len(tfidf_cands)
                if needed > 0 and remaining:
                    seed = _stable_seed(str(crm_row.get("crm_id") or crm_row.get("crm_name") or ""))
                    random_extra = _deterministic_sample(remaining, needed, seed)
                    candidates = tfidf_cands + random_extra
                else:
                    candidates = tfidf_cands

    return candidates


def _infer_query(
    r: dict,
    crm_ctx: dict,
    store: PartitionedCandidateStore,
    dept_cache: "OrderedDict[str, DeptCacheItem]",
    ranker: "xgb.Booster",
    classifier: "XGBClassifier",
    calibrator: Any,
    ranker_feature_order: List[str],
    decider_feature_order: List[str],
    semantic_enabled: bool,
    cfg: dict,
) -> List[dict]:
    """Run two-stage inference on a single CRM query. Returns output row dicts."""
    candidates = _build_candidate_pool(
        store, r, crm_ctx, cfg["pool_mode"], cfg["prefilter_k"],
        cfg["dept_prefilter_k"], cfg["max_dept_candidates"],
        cfg["semantic_retrieval_k"], cfg["semantic_retrieval_min_sim"],
        cfg["drop_unnamed"], cfg["exclude_closed"], dept_cache,
        tfidf_name_mode=cfg["tfidf_name_mode"],
        siren_siblings=cfg["siren_siblings"],
        min_candidates=cfg["min_candidates"],
        char_top_k=cfg["char_top_k"],
    )
    if not candidates:
        return []

    cand_list = [(c.get("siret"), c) for c in candidates if c.get("siret")]
    if not cand_list:
        return []

    # Stage 1 features (no semantic)
    feats_stage1 = [make_features_from_preprocessed(crm_ctx, c, skip_semantic=True) for _, c in cand_list]
    X1 = pd.DataFrame(feats_stage1)[ranker_feature_order]
    scores_stage1 = ranker.predict(xgb.DMatrix(X1.values, feature_names=ranker_feature_order))

    # Stage 1 top-N selection
    stage1_top_n = min(cfg.get("stage1_top_n", 200), len(scores_stage1))
    top_n_idx = np.argsort(scores_stage1)[::-1][:stage1_top_n]

    # Address rescue
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

    # Stage 2 features on top-N
    feats_n = []
    cand_list_n = []
    semantic_pools = []
    semantic_allowed_flags: List[bool] = []
    for idx in top_n_idx:
        siret, c = cand_list[idx]
        feat = feats_stage1[idx]
        semantic_allowed = False
        if semantic_enabled:
            semantic_allowed = semantic_gate_allows(
                feat.get("name_jaro_max", 0.0),
                feat.get("name_token_overlap_max", 0.0),
            )
            if semantic_allowed:
                cand_city_norm = c.get("_xgb_cached_city_norm") or normalize_text(c.get("city"))
                semantic_pools.append(
                    build_semantic_name_pool(
                        build_candidate_names(c),
                        crm_city_norm=crm_ctx.get("crm_city_norm", ""),
                        cand_city_norm=cand_city_norm,
                    )
                )
            else:
                semantic_pools.append([])
        semantic_allowed_flags.append(semantic_allowed)
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
        for feat, (sem_max, sem_second, sem_gap), allowed in zip(
            feats_n, sem, semantic_allowed_flags, strict=True
        ):
            if allowed:
                feat["name_semantic_max"] = sem_max
                feat["name_semantic_second"] = sem_second
                feat["name_semantic_gap"] = sem_gap
            else:
                feat["name_semantic_max"] = 0.0
                feat["name_semantic_second"] = 0.0
                feat["name_semantic_gap"] = 0.0

    X_n = pd.DataFrame(feats_n)[decider_feature_order]
    probs = classifier.predict_proba(X_n.values)[:, 1]
    if calibrator is not None:
        probs = calibrator.predict_proba(X_n.values)[:, 1]

    scores = np.array(probs)
    pool_size_stage1 = len(cand_list)
    pool_size_stage2 = len(scores)
    scores_sorted = np.sort(scores)[::-1]
    top3_avg = float(np.mean(scores_sorted[:3])) if len(scores_sorted) >= 3 else float(np.mean(scores_sorted)) if len(scores_sorted) else 0.0

    top_k = cfg["top_k"]
    topk_idx = np.argsort(scores)[::-1][:top_k].tolist()
    topk_idx = _promote_open_over_closed(topk_idx, cand_list_n)
    top1_score = float(scores[topk_idx[0]]) if topk_idx else 0.0
    top2_score = float(scores[topk_idx[1]]) if len(topk_idx) > 1 else 0.0
    score_gap = top1_score - top2_score
    denom = top2_score if top2_score > 1e-6 else 0.001
    score_ratio = top1_score / denom

    orig_index = r.get("_orig_index")
    rows_out: List[dict] = []
    for rank, idx_k in enumerate(topk_idx, start=1):
        siret_k, cand_k = cand_list_n[idx_k]
        cand_name = primary_name(cand_k) or f"SIRET {siret_k}"
        etat_admin = str(cand_k.get("etat_admin") or "").strip().upper() or None
        candidate_state = None
        if etat_admin is not None:
            candidate_state = "FERME" if etat_admin == "F" else "OUVERT"
        feat_row = feats_n[idx_k]
        name_semantic_max = float(feat_row.get("name_semantic_max", 0.0) or 0.0)
        name_semantic_second = float(feat_row.get("name_semantic_second", 0.0) or 0.0)

        row_out = {
            "_orig_index": orig_index,
            "crm_id": r.get("crm_id"),
            "crm_name": r.get("crm_name"),
            "crm_address": r.get("crm_address"),
            "crm_postcode": r.get("postcode"),
            "crm_city": r.get("crm_city"),
            "siret_candidate": siret_k,
            "score": float(scores[idx_k]),
            "score_top1": top1_score,
            "score_top2": top2_score,
            "score_gap": score_gap,
            "score_ratio": score_ratio,
            "top3_avg": top3_avg,
            "pool_size": pool_size_stage2,
            "pool_size_stage1": pool_size_stage1,
            "name_semantic_max": name_semantic_max,
            "name_semantic_second": name_semantic_second,
            "candidate_name": cand_name,
            "candidate_addr": build_address(cand_k),
            "candidate_city": cand_k.get("city"),
            "candidate_postcode": cand_k.get("postcode"),
            "candidate_insee": cand_k.get("insee"),
            "candidate_state": candidate_state,
            "candidate_last_treatment_date": cand_k.get("last_treatment_date"),
            "rank": rank,
            "has_name_evidence": has_name_evidence(feat_row),
        }

        if cfg["export_routing_features"]:
            row_out.update(
                {
                    "name_jaro_max": float(feat_row.get("name_jaro_max", 0.0)),
                    "name_token_overlap_max": float(feat_row.get("name_token_overlap_max", 0.0)),
                    "name_sim_max_etab": float(feat_row.get("name_sim_max_etab", 0.0)),
                    "name_crm_contains_cand_max": float(feat_row.get("name_crm_contains_cand_max", 0.0)),
                    "name_sim_max_pm_dirigeant": float(feat_row.get("name_sim_max_pm_dirigeant", 0.0)),
                    "idf_name": float(feat_row.get("idf_name", 0.0)),
                    "numeric_token_match": float(feat_row.get("numeric_token_match", 0.0)),
                    "addr_jaro": float(feat_row.get("addr_jaro", 0.0)),
                    "addr_token_overlap": float(feat_row.get("addr_token_overlap", 0.0)),
                    "address_density": float(feat_row.get("address_density", 1.0)),
                    "street_number_diff": float(feat_row.get("street_number_diff", 9999)),
                    "name_semantic_gap": float(feat_row.get("name_semantic_gap", 0.0)),
                    "name_semantic_second": float(feat_row.get("name_semantic_second", 0.0)),
                    "name_contains_crm_max": float(feat_row.get("name_contains_crm_max", 0.0)),
                    "postcode_match": float(feat_row.get("postcode_match", 0.0)),
                    "city_match": float(feat_row.get("city_match", 0.0)),
                    "street_name_jaro": float(feat_row.get("street_name_jaro", 0.0)),
                    "name_addr_consistency": float(feat_row.get("name_addr_consistency", 0.0)),
                    "name_length_max": float(feat_row.get("name_length_max", 0.0)),
                    "legal_form_category": float(feat_row.get("legal_form_category", 0.0)),
                }
            )

        rows_out.append(row_out)

    return rows_out


def _infer_dept_batch_worker(worker_args: dict) -> List[dict]:
    """Worker function for ProcessPoolExecutor: processes all queries for a department group.

    Loads its own models and store to avoid shared-state issues across processes.
    """
    records_batch = worker_args["records"]
    pre_batch = worker_args["pre"]
    ranker_path = worker_args["ranker_path"]
    decider_path = worker_args["decider_path"]
    calibrator_path = worker_args["calibrator_path"]
    partitions_dir = worker_args["partitions_dir"]
    ranker_feature_order = worker_args["ranker_feature_order"]
    decider_feature_order = worker_args["decider_feature_order"]
    cfg = worker_args["cfg"]
    semantic_queue = worker_args.get("semantic_queue")
    semantic_responses = worker_args.get("semantic_responses")

    # Each worker loads its own models (avoids pickle issues with xgb objects)
    ranker = xgb.Booster()
    ranker.load_model(str(ranker_path))

    classifier = XGBClassifier()
    classifier.load_model(str(decider_path))

    calibrator = None
    if calibrator_path:
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)

    store = PartitionedCandidateStore(Path(partitions_dir))
    dept_cache: OrderedDict[str, DeptCacheItem] = OrderedDict()

    semantic_enabled = cfg["semantic_enabled"]
    if semantic_enabled and semantic_queue is not None and semantic_responses is not None:
        client = SemanticClient(semantic_queue, semantic_responses)
        set_semantic_client(client)

    # Warmup semantic for this batch's CRM names
    if semantic_enabled:
        sem_names = list(dict.fromkeys(
            p.get("crm_name_semantic", "") for p in pre_batch if p.get("crm_name_semantic")
        ))
        if sem_names:
            batch_encode_texts(sem_names)

    rows_out: List[dict] = []
    for r, crm_ctx in zip(records_batch, pre_batch):
        rows_out.extend(
            _infer_query(
                r, crm_ctx, store, dept_cache,
                ranker, classifier, calibrator,
                ranker_feature_order, decider_feature_order, semantic_enabled, cfg,
            )
        )
    return rows_out


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger(__name__)

    if args.allow_no_semantic:
        os.environ["XGB_ALLOW_NO_SEMANTIC"] = "1"

    crm = load_crm(args.crm_path)
    crm_records = crm.to_dict("records")
    for idx, r in enumerate(crm_records):
        r["_orig_index"] = idx
    crm_pre = [preprocess_crm_row(r) for r in crm_records]

    # Sort CRM by department only when using pool_mode=multi (dept cache locality).
    if args.pool_mode == "multi":
        _dept_keys = [
            department_from_code(r.get("insee"), r.get("postcode")) or "ZZZ"
            for r in crm_records
        ]
        _sort_order = sorted(range(len(crm_records)), key=lambda i: _dept_keys[i])
        crm_records = [crm_records[i] for i in _sort_order]
        crm_pre = [crm_pre[i] for i in _sort_order]

    # Issue 2: Check if semantic is actually available (vs just enabled in env)
    semantic_available = is_semantic_available()
    if not semantic_available:
        if args.allow_no_semantic:
            logger.warning(
                "⚠️  SEMANTIC UNAVAILABLE: sentence_transformers not installed or model failed to load. "
                "Features (name_semantic_max, name_semantic_gap, etc.) will be ZERO. "
                "This WILL degrade routing precision. Install: pip install sentence-transformers"
            )
        else:
            raise RuntimeError(
                "SEMANTIC UNAVAILABLE: sentence_transformers not installed or model failed to load. "
                "Set --allow-no-semantic to run anyway (CAUTION: degraded features). "
                "Install: pip install sentence-transformers"
            )
    semantic_enabled = semantic_available  # Only use semantic if actually available
    
    if os.getenv("XGB_SEMANTIC_ENABLED", "0") != "1":
        logger.info("Semantic is forced ON in inference (XGB_SEMANTIC_ENABLED=0).")
    if args.semantic_retrieval_k > 0 and args.pool_mode != "multi":
        logger.warning("semantic_retrieval_k is set but pool_mode=%s; semantic retrieval runs only in pool_mode=multi.", args.pool_mode)
    crm_semantic_names = [c.get("crm_name_semantic", "") for c in crm_pre if c.get("crm_name_semantic")]
    unique_crm_semantic = list(dict.fromkeys(crm_semantic_names))
    if unique_crm_semantic and semantic_enabled:
        logger.info("[Semantic] Warmup CRM embeddings: %d unique names", len(unique_crm_semantic))
        batch_encode_texts(unique_crm_semantic)
        cache_size, cache_mb = get_cache_stats()
        logger.info("[Semantic] Cache: %d embeddings (~%.1f MB)", cache_size, cache_mb)

    model_dir = Path("models")
    meta_path = args.meta_path or _find_latest_meta(model_dir)
    meta = {}
    if meta_path and meta_path.exists():
        meta = _load_models_from_meta(meta_path)
        logger.info("Using meta: %s", meta_path)

    decider_feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
    ranker_feature_order = meta.get("ranker_feature_order") or decider_feature_order
    ranker_fast_feature_order = meta.get("ranker_fast_feature_order") or []
    meta_semantic = meta.get("semantic_enabled_samples")
    semantic_env = int(os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1")
    if meta_semantic is not None and int(meta_semantic) != semantic_env:
        logger.warning(
            "Semantic enabled mismatch: meta=%s vs env=%s (train/serve skew risk)",
            meta_semantic,
            semantic_env,
    )

    ranker_path = args.ranker_fast_model or meta.get("ranker_fast_model") or args.ranker_model or meta.get("ranker_model")
    if not ranker_path:
        raise FileNotFoundError("Ranker model path not provided and not found in meta.")
    ranker_is_fast = False
    if args.ranker_fast_model:
        ranker_is_fast = True
    elif meta.get("ranker_fast_model") and str(ranker_path) == str(meta.get("ranker_fast_model")):
        ranker_is_fast = True
    elif "fast" in str(ranker_path).lower():
        ranker_is_fast = True
    if ranker_is_fast:
        stage1_feature_order = ranker_fast_feature_order or FAST_RANKER_FEATURE_NAMES or ranker_feature_order
    else:
        stage1_feature_order = ranker_feature_order
    ranker = xgb.Booster()
    ranker.load_model(str(ranker_path))
    logger.info("Ranker: %s", ranker_path)

    decider_path = args.decider_model or meta.get("decider_model")
    if not decider_path:
        raise FileNotFoundError("Decider model path not provided and not found in meta.")
    classifier = XGBClassifier()
    classifier.load_model(str(decider_path))
    logger.info("Decider: %s", decider_path)

    calibrator = None
    calibrator_path = args.calibrator_path or meta.get("decider_calibrator")
    if calibrator_path:
        calib_path = Path(calibrator_path)
        if not calib_path.exists():
            calib_path = model_dir / calib_path.name
        if calib_path.exists():
            with open(calib_path, "rb") as f:
                calibrator = pickle.load(f)
            logger.info("Calibrator: %s", calib_path)

    store = PartitionedCandidateStore(args.partitions_dir)
    dept_cache: OrderedDict[str, DeptCacheItem] = OrderedDict()

    # Dept cache is now LRU (max MAX_DEPT_CACHE depts) and loaded on-demand.
    # CRM is sorted by department below to maximise cache hits.
    if args.pool_mode == "multi":
        unique_depts = set()
        for r in crm_records:
            dept = department_from_code(r.get("insee"), r.get("postcode"))
            if dept:
                unique_depts.add(dept)
        logger.info("LRU dept_cache (max %d). %d unique departments in CRM.", MAX_DEPT_CACHE, len(unique_depts))

    # Debug GT mode setup
    debug_gt_mode = args.debug_gt
    gt_column = args.gt_column
    if debug_gt_mode:
        if gt_column not in crm.columns:
            logger.warning("GT column '%s' not found in CRM. Available: %s", gt_column, list(crm.columns))
            debug_gt_mode = False
        else:
            logger.info("Debug GT mode enabled with column '%s'", gt_column)
    debug_rows: List[dict] = []

    stage1_top_n = args.stage1_top_n
    if args.pool_mode != "multi":
        stage1_top_n = max(stage1_top_n, 500)

    # Build shared config dict for _infer_query / worker
    infer_cfg = {
        "pool_mode": args.pool_mode,
        "prefilter_k": args.prefilter_k,
        "dept_prefilter_k": args.dept_prefilter_k,
        "max_dept_candidates": args.max_dept_candidates,
        "semantic_retrieval_k": args.semantic_retrieval_k,
        "semantic_retrieval_min_sim": args.semantic_retrieval_min_sim,
        "drop_unnamed": args.drop_unnamed,
        "exclude_closed": args.exclude_closed,
        "tfidf_name_mode": args.tfidf_name_mode,
        "siren_siblings": args.siren_siblings,
        "min_candidates": args.min_candidates,
        "char_top_k": args.char_top_k,
        "top_k": args.top_k,
        "stage1_top_n": stage1_top_n,
        "export_routing_features": args.export_routing_features,
        "semantic_enabled": semantic_enabled,
    }

    rows_out: List[dict] = []

    if debug_gt_mode:
        # ── Sequential path (debug GT mode) ──────────────────────────────
        # Debug GT mode needs mutable debug_info and shared debug_rows,
        # which don't serialize well across processes.
        crm_pairs = list(zip(crm_records, crm_pre))
        progress = tqdm(total=len(crm_pairs), desc="Infer")
        for dept_key, group_iter in groupby(
            crm_pairs,
            key=lambda rp: department_from_code(rp[0].get("insee"), rp[0].get("postcode")) or "ZZZ",
        ):
            group = list(group_iter)
            if args.pool_mode == "multi" and group:
                first_r = group[0][0]
                _get_dept_cache_item(
                    dept_cache=dept_cache,
                    dept_key=dept_key if dept_key != "ZZZ" else None,
                    store=store,
                    insee=first_r.get("insee"),
                    postcode=first_r.get("postcode"),
                    drop_unnamed=args.drop_unnamed,
                    exclude_closed=args.exclude_closed,
                    max_dept_candidates=args.max_dept_candidates,
                    tfidf_name_mode=args.tfidf_name_mode,
                    siren_siblings=args.siren_siblings,
                )

            for r, crm_ctx in group:
                orig_index = r.get("_orig_index")
                gt_siret = str(r.get(gt_column) or "").strip()
                gt_siret = gt_siret if gt_siret and gt_siret not in ("", "nan", "None") else None

                candidates, debug_info = _build_candidate_pool_with_debug(
                    store, r, crm_ctx, args.pool_mode, args.prefilter_k,
                    args.dept_prefilter_k, args.max_dept_candidates,
                    args.semantic_retrieval_k, args.semantic_retrieval_min_sim,
                    args.drop_unnamed, args.exclude_closed, dept_cache,
                    tfidf_name_mode=args.tfidf_name_mode,
                    siren_siblings=args.siren_siblings,
                    min_candidates=args.min_candidates,
                    char_top_k=args.char_top_k,
                    gt_siret=gt_siret,
                )
                if not candidates:
                    progress.update(1)
                    continue

                cand_list = [(c.get("siret"), c) for c in candidates if c.get("siret")]
                if not cand_list:
                    progress.update(1)
                    continue

                feats_stage1 = [make_features_from_preprocessed(crm_ctx, c, skip_semantic=True) for _, c in cand_list]
                X1 = pd.DataFrame(feats_stage1)[stage1_feature_order]
                scores_stage1 = ranker.predict(xgb.DMatrix(X1.values, feature_names=stage1_feature_order))

                stage1_top_n = min(infer_cfg.get("stage1_top_n", 200), len(scores_stage1))
                top_n_idx = np.argsort(scores_stage1)[::-1][:stage1_top_n]

                rescue_indices = []
                top_n_set = set(top_n_idx)
                for i, feat in enumerate(feats_stage1):
                    if i in top_n_set:
                        continue
                    if (float(feat.get("addr_jaro", 0.0)) >= 0.96
                            and float(feat.get("street_name_jaro", 0.0)) >= 0.95
                            and float(feat.get("street_number_diff", 9999)) <= 2):
                        rescue_indices.append(i)
                rescue_indices = rescue_indices[:50]
                if rescue_indices:
                    top_n_idx = np.unique(np.concatenate([top_n_idx, np.array(rescue_indices, dtype=int)]))

                if debug_info and gt_siret:
                    gt_norm = str(gt_siret).zfill(14)
                    stage1_sirets = {str(cand_list[i][0]).zfill(14) for i in top_n_idx}
                    debug_info.in_stage1_topn = gt_norm in stage1_sirets
                    debug_info.stage1_topn_size = len(top_n_idx)
                    if debug_info.in_tfidf_pool and not debug_info.in_stage1_topn:
                        for rank_s1, idx_s1 in enumerate(np.argsort(scores_stage1)[::-1], start=1):
                            if str(cand_list[idx_s1][0]).zfill(14) == gt_norm:
                                debug_info.stage1_rank = rank_s1
                                break
                        debug_info.loss_reason = f"PRUNED_BY_STAGE1_RANK_{debug_info.stage1_rank}"

                feats_n = []
                cand_list_n = []
                semantic_pools = []
                semantic_allowed_flags: List[bool] = []
                for idx in top_n_idx:
                    siret, c = cand_list[idx]
                    feat = feats_stage1[idx]
                    semantic_allowed = False
                    if semantic_enabled:
                        semantic_allowed = semantic_gate_allows(
                            feat.get("name_jaro_max", 0.0),
                            feat.get("name_token_overlap_max", 0.0),
                        )
                        if semantic_allowed:
                            cand_city_norm = c.get("_xgb_cached_city_norm") or normalize_text(c.get("city"))
                            semantic_pools.append(
                                build_semantic_name_pool(
                                    build_candidate_names(c),
                                    crm_city_norm=crm_ctx.get("crm_city_norm", ""),
                                    cand_city_norm=cand_city_norm,
                                )
                            )
                        else:
                            semantic_pools.append([])
                    semantic_allowed_flags.append(semantic_allowed)
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
                    for feat, (sem_max, sem_second, sem_gap), allowed in zip(
                        feats_n, sem, semantic_allowed_flags, strict=True
                    ):
                        if allowed:
                            feat["name_semantic_max"] = sem_max
                            feat["name_semantic_second"] = sem_second
                            feat["name_semantic_gap"] = sem_gap
                        else:
                            feat["name_semantic_max"] = 0.0
                            feat["name_semantic_second"] = 0.0
                            feat["name_semantic_gap"] = 0.0

                X_n = pd.DataFrame(feats_n)[decider_feature_order]
                probs = classifier.predict_proba(X_n.values)[:, 1]
                if calibrator is not None:
                    probs = calibrator.predict_proba(X_n.values)[:, 1]

                scores = np.array(probs)
                pool_size_stage1 = len(cand_list)
                pool_size_stage2 = len(scores)
                scores_sorted = np.sort(scores)[::-1]
                top3_avg = float(np.mean(scores_sorted[:3])) if len(scores_sorted) >= 3 else float(np.mean(scores_sorted)) if len(scores_sorted) else 0.0

                topk_idx = np.argsort(scores)[::-1][: args.top_k].tolist()
                topk_idx = _promote_open_over_closed(topk_idx, cand_list_n)
                top1_score = float(scores[topk_idx[0]]) if topk_idx else 0.0
                top2_score = float(scores[topk_idx[1]]) if len(topk_idx) > 1 else 0.0
                score_gap = top1_score - top2_score
                denom = top2_score if top2_score > 1e-6 else 0.001
                score_ratio = top1_score / denom

                if debug_info and gt_siret:
                    gt_norm = str(gt_siret).zfill(14)
                    topk_sirets = [str(cand_list_n[i][0]).zfill(14) for i in topk_idx]
                    debug_info.in_stage2_topk = gt_norm in topk_sirets
                    for rank_s2, idx_s2 in enumerate(np.argsort(scores)[::-1], start=1):
                        if str(cand_list_n[idx_s2][0]).zfill(14) == gt_norm:
                            debug_info.stage2_rank = rank_s2
                            break
                    if debug_info.in_stage1_topn and not debug_info.in_stage2_topk and not debug_info.loss_reason:
                        debug_info.loss_reason = f"PRUNED_BY_STAGE2_TOPK_{debug_info.stage2_rank}"
                    if not debug_info.loss_reason and not debug_info.in_base_pool:
                        debug_info.loss_reason = "NOT_IN_PARTITION"
                    debug_rows.append({
                        "_orig_index": orig_index,
                        "crm_id": r.get("crm_id"),
                        "crm_name": r.get("crm_name"),
                        "gt_siret": gt_siret,
                        "in_base_pool": debug_info.in_base_pool,
                        "in_filtered_pool": debug_info.in_filtered_pool,
                        "in_tfidf_pool": debug_info.in_tfidf_pool,
                        "in_stage1_topn": debug_info.in_stage1_topn,
                        "in_stage2_topk": debug_info.in_stage2_topk,
                        "base_pool_size": debug_info.base_pool_size,
                        "filtered_pool_size": debug_info.filtered_pool_size,
                        "tfidf_pool_size": debug_info.tfidf_pool_size,
                        "stage1_topn_size": debug_info.stage1_topn_size,
                        "stage1_rank": debug_info.stage1_rank,
                        "stage2_rank": debug_info.stage2_rank,
                        "loss_reason": debug_info.loss_reason,
                    })

                # Emit output rows for this query (debug path — reuse already computed data)
                for rank, idx_k in enumerate(topk_idx, start=1):
                    siret_k, cand_k = cand_list_n[idx_k]
                    cand_name = primary_name(cand_k) or f"SIRET {siret_k}"
                    etat_admin = str(cand_k.get("etat_admin") or "").strip().upper() or None
                    candidate_state = None
                    if etat_admin is not None:
                        candidate_state = "FERME" if etat_admin == "F" else "OUVERT"
                    feat_row = feats_n[idx_k]
                    row_out = {
                        "_orig_index": orig_index,
                        "crm_id": r.get("crm_id"),
                        "crm_name": r.get("crm_name"),
                        "crm_address": r.get("crm_address"),
                        "crm_postcode": r.get("postcode"),
                        "crm_city": r.get("crm_city"),
                        "siret_candidate": siret_k,
                        "score": float(scores[idx_k]),
                        "score_top1": top1_score,
                        "score_top2": top2_score,
                        "score_gap": score_gap,
                        "score_ratio": score_ratio,
                        "top3_avg": top3_avg,
                        "pool_size": pool_size_stage2,
                        "pool_size_stage1": pool_size_stage1,
                        "name_semantic_max": float(feat_row.get("name_semantic_max", 0.0) or 0.0),
                        "name_semantic_second": float(feat_row.get("name_semantic_second", 0.0) or 0.0),
                        "candidate_name": cand_name,
                        "candidate_addr": build_address(cand_k),
                        "candidate_city": cand_k.get("city"),
                        "candidate_postcode": cand_k.get("postcode"),
                        "candidate_insee": cand_k.get("insee"),
                        "candidate_state": candidate_state,
                        "candidate_last_treatment_date": cand_k.get("last_treatment_date"),
                        "rank": rank,
                        "has_name_evidence": has_name_evidence(feat_row),
                    }
                    if args.export_routing_features:
                        row_out.update({
                            "name_jaro_max": float(feat_row.get("name_jaro_max", 0.0)),
                            "name_token_overlap_max": float(feat_row.get("name_token_overlap_max", 0.0)),
                            "name_sim_max_etab": float(feat_row.get("name_sim_max_etab", 0.0)),
                            "name_crm_contains_cand_max": float(feat_row.get("name_crm_contains_cand_max", 0.0)),
                            "name_sim_max_pm_dirigeant": float(feat_row.get("name_sim_max_pm_dirigeant", 0.0)),
                            "idf_name": float(feat_row.get("idf_name", 0.0)),
                            "numeric_token_match": float(feat_row.get("numeric_token_match", 0.0)),
                            "addr_jaro": float(feat_row.get("addr_jaro", 0.0)),
                            "addr_token_overlap": float(feat_row.get("addr_token_overlap", 0.0)),
                            "address_density": float(feat_row.get("address_density", 1.0)),
                            "street_number_diff": float(feat_row.get("street_number_diff", 9999)),
                            "name_semantic_gap": float(feat_row.get("name_semantic_gap", 0.0)),
                            "name_semantic_second": float(feat_row.get("name_semantic_second", 0.0)),
                            "name_contains_crm_max": float(feat_row.get("name_contains_crm_max", 0.0)),
                            "postcode_match": float(feat_row.get("postcode_match", 0.0)),
                            "city_match": float(feat_row.get("city_match", 0.0)),
                            "street_name_jaro": float(feat_row.get("street_name_jaro", 0.0)),
                            "name_addr_consistency": float(feat_row.get("name_addr_consistency", 0.0)),
                            "name_length_max": float(feat_row.get("name_length_max", 0.0)),
                            "legal_form_category": float(feat_row.get("legal_form_category", 0.0)),
                        })
                    rows_out.append(row_out)
                progress.update(1)
        progress.close()
    else:
        if args.pool_mode != "multi":
            # ── Sequential path for INSEE/CP pools ────────────────────────
            for r, crm_ctx in tqdm(zip(crm_records, crm_pre), total=len(crm_records), desc="Infer"):
                rows_out.extend(
                    _infer_query(
                        r, crm_ctx, store, dept_cache,
                        ranker, classifier, calibrator,
                        stage1_feature_order, decider_feature_order, semantic_enabled, infer_cfg,
                    )
                )
        else:
            # ── Parallel path (dept batches) ──────────────────────────────
            # Group queries by department for ProcessPoolExecutor.
            dept_groups: Dict[str, Tuple[List[dict], List[dict]]] = {}
            for r, crm_ctx in zip(crm_records, crm_pre):
                dept = department_from_code(r.get("insee"), r.get("postcode")) or "ZZZ"
                if dept not in dept_groups:
                    dept_groups[dept] = ([], [])
                dept_groups[dept][0].append(r)
                dept_groups[dept][1].append(crm_ctx)

            # Resolve model paths for workers (they load their own instances)
            ranker_path_str = str(ranker_path)
            decider_path_str = str(decider_path)
            calibrator_path_str = str(calibrator_path) if calibrator_path else None
            # Check if calibrator file actually exists
            if calibrator_path_str:
                _cp = Path(calibrator_path_str)
                if not _cp.exists():
                    _cp = Path("models") / _cp.name
                    calibrator_path_str = str(_cp) if _cp.exists() else None
            partitions_dir_str = str(args.partitions_dir)

            worker_tasks = []
            ctx = mp.get_context("spawn")
            semantic_queue = None
            semantic_responses = None
            semantic_server = None
            manager = None
            if semantic_enabled:
                manager = ctx.Manager()
                semantic_responses = manager.dict()
                semantic_queue = manager.Queue(maxsize=200)
                semantic_server = ctx.Process(
                    target=semantic_server_process,
                    args=(semantic_queue, semantic_responses, _model_name(), _device()),
                )
                semantic_server.start()
            for dept, (recs, pres) in dept_groups.items():
                worker_tasks.append({
                    "records": recs,
                    "pre": pres,
                    "ranker_path": ranker_path_str,
                    "decider_path": decider_path_str,
                    "calibrator_path": calibrator_path_str,
                    "partitions_dir": partitions_dir_str,
                    "ranker_feature_order": stage1_feature_order,
                    "decider_feature_order": decider_feature_order,
                    "cfg": infer_cfg,
                    "semantic_queue": semantic_queue,
                    "semantic_responses": semantic_responses,
                })

            logger.info(
                "Parallel inference: %d dept groups across %d workers",
                len(worker_tasks), min(MAX_WORKERS, len(worker_tasks)),
            )

            try:
                with ProcessPoolExecutor(
                    max_workers=min(MAX_WORKERS, len(worker_tasks)),
                    mp_context=ctx,
                ) as executor:
                    futures = {
                        executor.submit(_infer_dept_batch_worker, task): dept_key
                        for task, dept_key in zip(worker_tasks, dept_groups.keys())
                    }
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Dept batches"):
                        dept_key = futures[future]
                        try:
                            batch_rows = future.result()
                            rows_out.extend(batch_rows)
                        except Exception:
                            logger.exception("Worker failed for dept %s", dept_key)
            finally:
                if semantic_queue is not None:
                    semantic_queue.put(None)
                if semantic_server is not None:
                    semantic_server.join()
                if manager is not None:
                    manager.shutdown()

    # Issue 4: Handle empty output gracefully (avoid KeyError on sort)
    if not rows_out:
        logger.warning("No inference results produced. Output will be empty.")
        out_df = pd.DataFrame(columns=["crm_id", "crm_name", "siret_candidate", "score", "rank"])
    else:
        out_df = pd.DataFrame(rows_out)
        if "_orig_index" in out_df.columns:
            out_df = out_df.sort_values(["_orig_index", "rank"], ascending=[True, True])
            out_df = out_df.drop(columns=["_orig_index"])
        else:
            out_df = out_df.sort_values(["crm_name", "rank", "score"], ascending=[True, True, False])
    
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    logger.info("Saved top-%d results to %s (%d rows)", args.top_k, args.output_path, len(out_df))

    # Export GT debug CSV if enabled
    if debug_gt_mode and debug_rows:
        debug_output = args.debug_gt_output or args.output_path.with_name(args.output_path.stem + "_gt_debug.csv")
        debug_df = pd.DataFrame(debug_rows)
        if "_orig_index" in debug_df.columns:
            debug_df = debug_df.sort_values(["_orig_index"], ascending=[True])
            debug_df = debug_df.drop(columns=["_orig_index"])
        debug_df.to_csv(debug_output, index=False)
        # Summary stats
        n_total = len(debug_df)
        n_in_base = debug_df["in_base_pool"].sum()
        n_in_filtered = debug_df["in_filtered_pool"].sum()
        n_in_tfidf = debug_df["in_tfidf_pool"].sum()
        n_in_stage1 = debug_df["in_stage1_topn"].sum()
        n_in_topk = debug_df["in_stage2_topk"].sum()
        logger.info("GT Debug summary (%d queries with GT):", n_total)
        logger.info("  in_base_pool: %d (%.1f%%)", n_in_base, 100 * n_in_base / n_total if n_total else 0)
        logger.info("  in_filtered_pool: %d (%.1f%%)", n_in_filtered, 100 * n_in_filtered / n_total if n_total else 0)
        logger.info("  in_tfidf_pool: %d (%.1f%%)", n_in_tfidf, 100 * n_in_tfidf / n_total if n_total else 0)
        logger.info("  in_stage1_topn: %d (%.1f%%)", n_in_stage1, 100 * n_in_stage1 / n_total if n_total else 0)
        logger.info("  in_stage2_topk: %d (%.1f%%)", n_in_topk, 100 * n_in_topk / n_total if n_total else 0)
        # Loss reason breakdown
        loss_counts = debug_df["loss_reason"].value_counts()
        logger.info("  Loss reasons:")
        for reason, count in loss_counts.items():
            if reason:
                logger.info("    %s: %d", reason, count)
        logger.info("Saved GT debug to %s", debug_output)


if __name__ == "__main__":
    main()
