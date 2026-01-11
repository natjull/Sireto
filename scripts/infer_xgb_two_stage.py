#!/usr/bin/env python3
"""
Two-stage XGBoost inference with multi-blocking and partitioned candidates.

Stage 1: Ranker (fast, no semantic)
Stage 2: Decider (full features + optional calibration)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    filter_candidates_by_address_hash,
    filter_candidates_by_numeric_tokens,
    build_tfidf_index,
    prefilter_candidates_tfidf,
)
from src.xgb_matcher.features import (
    FEATURE_NAMES,
    build_address,
    build_semantic_name_pool,
    make_features_from_preprocessed,
    normalize_text,
    preprocess_crm_row,
    semantic_gate_allows,
)
from src.xgb_matcher.naming import build_candidate_names, primary_name
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.semantic import batch_encode_texts, get_cache_stats, top2_semantic_similarities_batch


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer XGB two-stage matcher (partitioned + multi-blocking).")
    p.add_argument("--crm-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, default=Path("reports/xgb_two_stage_topk.csv"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v4_active"))
    p.add_argument("--pool-mode", choices=["insee_then_postcode", "union", "multi"], default="insee_then_postcode")
    p.add_argument("--prefilter-k", type=int, default=500)
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
    p.add_argument("--meta-path", type=Path, default=None)
    p.add_argument("--ranker-model", type=Path, default=None)
    p.add_argument("--ranker-fast-model", type=Path, default=None)
    p.add_argument("--decider-model", type=Path, default=None)
    p.add_argument("--calibrator-path", type=Path, default=None)
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
    if "crm_name" not in df.columns:
        df = df.rename(
            columns={
                "Client final": "crm_name",
                "Adresse": "crm_address",
                "Commune": "crm_city",
                "Code Postal": "postcode",
                "Code INSEE": "insee",
            }
        )
    for col in ["postcode", "insee"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and x not in ["", "nan"] else x
            )
    if "crm_id" not in df.columns:
        df["crm_id"] = df.index
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
    tfidf_cache: Dict[Tuple[str, str], tuple],
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

    if pool_mode == "multi":
        dept_candidates = store.load_by_department(insee, postcode)
        dept_candidates = _apply_candidate_filters(dept_candidates, drop_unnamed, exclude_closed)
        if max_dept_candidates and len(dept_candidates) > max_dept_candidates:
            dept_candidates = dept_candidates[:max_dept_candidates]
        extra: List[dict] = []

        addr_hash = address_hash(crm_pre.get("crm_street_num"), crm_pre.get("crm_street_name"))
        extra.extend(filter_candidates_by_address_hash(dept_candidates, addr_hash))

        numeric_tokens = extract_numeric_tokens(crm_name)
        if numeric_tokens:
            extra.extend(filter_candidates_by_numeric_tokens(dept_candidates, numeric_tokens))

        if dept_candidates and (dept_prefilter_k > 0):
            dept_key = department_from_code(insee, postcode) or "unknown"
            key = ("dept", dept_key)
            vec, mat, names = tfidf_cache.get(key, (None, None, None))
            if vec is None:
                vec, mat, names = build_tfidf_index(dept_candidates)
                tfidf_cache[key] = (vec, mat, names)
            if vec is not None and mat is not None:
                idx = prefilter_candidates_tfidf(
                    crm_name,
                    vec,
                    mat,
                    min(dept_prefilter_k, len(dept_candidates)),
                    cand_names=names,
                    char_top_k=min(200, dept_prefilter_k),
                )
                extra.extend([dept_candidates[i] for i in idx])

        # Semantic retrieval (ANN-style) to recover CRM_LOC_MISMATCH
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

    candidates = list(pool.values())
    MIN_CANDIDATES_INFER = 100  # Guarantee at least this many candidates after TF-IDF prefilter
    if prefilter_k and len(candidates) > prefilter_k:
        vec, mat, names = build_tfidf_index(candidates)
        if vec is not None and mat is not None:
            idx = prefilter_candidates_tfidf(
                crm_name,
                vec,
                mat,
                prefilter_k,
                cand_names=names,
                char_top_k=min(200, prefilter_k),
            )
            # FIX: Guarantee minimum candidates by combining TF-IDF + random
            if len(idx) >= MIN_CANDIDATES_INFER:
                candidates = [candidates[i] for i in idx]
            else:
                # Combine TF-IDF matches with random samples
                tfidf_cands = [candidates[i] for i in idx]
                tfidf_set = set(idx)
                remaining = [c for i, c in enumerate(candidates) if i not in tfidf_set]
                needed = min(MIN_CANDIDATES_INFER, prefilter_k) - len(tfidf_cands)
                if needed > 0 and remaining:
                    import random
                    random_extra = random.sample(remaining, min(needed, len(remaining)))
                    candidates = tfidf_cands + random_extra
                else:
                    candidates = tfidf_cands

    attach_address_density(candidates)
    return candidates


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger(__name__)

    crm = load_crm(args.crm_path)
    crm_records = crm.to_dict("records")
    crm_pre = [preprocess_crm_row(r) for r in crm_records]

    semantic_enabled = os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"
    if args.semantic_retrieval_k > 0 and not semantic_enabled:
        logger.warning("semantic_retrieval_k=%d but XGB_SEMANTIC_ENABLED=0 (no semantic retrieval).", args.semantic_retrieval_k)
    if args.semantic_retrieval_k > 0 and args.pool_mode != "multi":
        logger.warning("semantic_retrieval_k is set but pool_mode=%s; semantic retrieval runs only in pool_mode=multi.", args.pool_mode)
    if semantic_enabled:
        crm_semantic_names = [c.get("crm_name_semantic", "") for c in crm_pre if c.get("crm_name_semantic")]
        unique_crm_semantic = list(dict.fromkeys(crm_semantic_names))
        if unique_crm_semantic:
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

    feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
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
    tfidf_cache: Dict[Tuple[str, str], tuple] = {}

    rows_out: List[dict] = []
    for r, crm_ctx in tqdm(zip(crm_records, crm_pre), total=len(crm_records), desc="Infer"):
        candidates = _build_candidate_pool(
            store,
            r,
            crm_ctx,
            args.pool_mode,
            args.prefilter_k,
            args.dept_prefilter_k,
            args.max_dept_candidates,
            args.semantic_retrieval_k,
            args.semantic_retrieval_min_sim,
            args.drop_unnamed,
            args.exclude_closed,
            tfidf_cache,
        )
        if not candidates:
            continue

        cand_list = [(c.get("siret"), c) for c in candidates if c.get("siret")]
        if not cand_list:
            continue

        # Stage 1 features (no semantic)
        feats_stage1 = [make_features_from_preprocessed(crm_ctx, c, skip_semantic=True) for _, c in cand_list]
        X1 = pd.DataFrame(feats_stage1)[feature_order]
        scores_stage1 = ranker.predict(xgb.DMatrix(X1.values, feature_names=feature_order))

        # Stage 1 top-N selection
        stage1_top_n = min(200, len(scores_stage1))
        top_n_idx = np.argsort(scores_stage1)[::-1][:stage1_top_n]

        # Address rescue: include near-perfect address matches outside top-N
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
        for idx in top_n_idx:
            siret, c = cand_list[idx]
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
                if semantic_gate_allows(
                    feat.get("name_jaro_max", 0.0),
                    feat.get("name_token_overlap_max", 0.0),
                ):
                    feat["name_semantic_max"] = sem_max
                    feat["name_semantic_second"] = sem_second
                    feat["name_semantic_gap"] = sem_gap
                else:
                    feat["name_semantic_max"] = 0.0
                    feat["name_semantic_second"] = 0.0
                    feat["name_semantic_gap"] = 0.0

        X_n = pd.DataFrame(feats_n)[feature_order]
        probs = classifier.predict_proba(X_n.values)[:, 1]
        if calibrator is not None:
            probs = calibrator.predict_proba(X_n.values)[:, 1]

        scores = np.array(probs)
        pool_size_stage1 = len(cand_list)
        pool_size_stage2 = len(scores)
        scores_sorted = np.sort(scores)[::-1]
        top1_score = float(scores_sorted[0]) if len(scores_sorted) else 0.0
        top2_score = float(scores_sorted[1]) if len(scores_sorted) > 1 else 0.0
        top3_avg = float(np.mean(scores_sorted[:3])) if len(scores_sorted) >= 3 else float(np.mean(scores_sorted)) if len(scores_sorted) else 0.0
        score_gap = top1_score - top2_score
        score_ratio = top1_score / (top2_score + 1e-9) if top2_score > 0 else 1.0

        topk_idx = np.argsort(scores)[::-1][: args.top_k]
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

            if args.export_routing_features:
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
                        "name_length_max": float(feat_row.get("name_length_max", 0.0)),
                        "legal_form_category": float(feat_row.get("legal_form_category", 0.0)),
                    }
                )

            rows_out.append(row_out)

    out_df = pd.DataFrame(rows_out).sort_values(["crm_name", "rank", "score"], ascending=[True, True, False])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    logger.info("Saved top-%d results to %s (%d rows)", args.top_k, args.output_path, len(out_df))


if __name__ == "__main__":
    main()
