"""Importable XGBoost two-stage inference logic.

This module extracts the core inference logic from scripts/infer_xgb_two_stage.py
so it can be reused by the Places fallback without code duplication.

Usage:
    from xgb_matcher.infer import XgbInferenceEngine, InferenceResult

    engine = XgbInferenceEngine.from_models(
        ranker_path="models/xgbranker_*.json",
        decider_path="models/xgb_decider_*.json",
        partitions_dir="data/candidates_v4_active",
    )
    result = engine.infer_single(crm_row)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

if TYPE_CHECKING:
    from .profile import InferenceProfile
from .blocking import (
    build_tfidf_index,
    build_char_tfidf_index,
    prefilter_candidates_tfidf,
    dedupe_candidates,
    attach_address_density,
    address_hash,
    department_from_code,
    extract_numeric_tokens,
    build_address_hash_index,
    build_numeric_token_index,
)
from .features import (
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
from .candidates import compute_name_idf_map
from .naming import build_candidate_names, primary_name
from .partitioned_store import PartitionedCandidateStore
from .semantic import top2_semantic_similarities_batch, is_semantic_available
from .calibration import load_calibrator


@dataclass
class DeptCacheItem:
    candidates: List[dict]
    tfidf_vec: Any
    tfidf_mat: Any
    tfidf_names: List[str]
    char_vec: Any = None
    char_mat: Any = None
    addr_index: Dict[str, List[int]] | None = None
    num_index: Dict[str, List[int]] | None = None

@dataclass
class InferenceResult:
    """Result from XGB inference for a single CRM row."""
    
    crm_id: str
    status: str  # "AUTO" or "REVIEW" (from risk model routing)
    top1_siret: str | None
    top1_score: float
    top2_score: float
    score_gap: float
    score_ratio: float
    pool_size: int
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    risk_score: float | None = None
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class CrmInput:
    """Minimal CRM input for inference."""
    
    crm_id: str
    crm_name: str
    crm_address: str | None = None
    postcode: str | None = None
    insee: str | None = None
    crm_city: str | None = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CrmInput":
        return cls(
            crm_id=str(d.get("crm_id", "")),
            crm_name=str(d.get("crm_name", "") or ""),
            crm_address=str(d.get("crm_address", "") or "") or None,
            postcode=str(d.get("postcode", "") or "") or None,
            insee=str(d.get("insee", "") or "") or None,
            crm_city=str(d.get("crm_city", "") or "") or None,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "crm_id": self.crm_id,
            "crm_name": self.crm_name,
            "crm_address": self.crm_address,
            "postcode": self.postcode,
            "insee": self.insee,
            "crm_city": self.crm_city,
        }


class XgbInferenceEngine:
    """Reusable XGB two-stage inference engine."""
    
    def __init__(
        self,
        ranker: xgb.Booster,
        decider: XGBClassifier,
        store: PartitionedCandidateStore,
        feature_order: List[str],
        ranker_feature_order: List[str] | None = None,
        risk_model: Any | None = None,
        risk_calibrator: Any | None = None,
        risk_features: List[str] | None = None,
        risk_threshold: float = 0.835,
        calibrator: Any | None = None,
        # Profile-driven configuration
        tfidf_name_mode: str = "bag",
        siren_siblings: bool = True,
        prefilter_k: int = 500,
        char_top_k: int = 200,
        min_candidates: int = 150,
        stage1_top_n: int = 200,
        drop_unnamed: bool = True,
        exclude_closed: bool = False,
    ):
        self.ranker = ranker
        self.decider = decider
        self.store = store
        self.feature_order = feature_order
        self.ranker_feature_order = ranker_feature_order or feature_order
        self.risk_model = risk_model
        self.risk_calibrator = risk_calibrator
        self.risk_features = risk_features or []
        self.risk_threshold = risk_threshold
        self.calibrator = calibrator
        # Profile-driven config
        self.tfidf_name_mode = tfidf_name_mode
        self.siren_siblings = siren_siblings
        self.prefilter_k = prefilter_k
        self.char_top_k = char_top_k
        self.min_candidates = min_candidates
        self.stage1_top_n = stage1_top_n
        self.drop_unnamed = drop_unnamed
        self.exclude_closed = exclude_closed
        self._dept_cache: OrderedDict[str, DeptCacheItem] = OrderedDict()
        self._max_dept_cache = 5
        
        # Fail-fast: semantic must be enabled
        if os.getenv("XGB_SEMANTIC_ENABLED", "0") != "1":
            raise RuntimeError(
                "SEMANTIC SKEW RISK: XGB_SEMANTIC_ENABLED != '1'. "
                "Set os.environ['XGB_SEMANTIC_ENABLED'] = '1' BEFORE importing xgb_matcher modules."
            )

        if not is_semantic_available():
            if os.getenv("XGB_ALLOW_NO_SEMANTIC", "0") == "1":
                logging.getLogger(__name__).warning(
                    "Semantic embeddings unavailable. Proceeding with semantic disabled; "
                    "name_semantic_* features will be ZERO."
                )
            else:
                raise RuntimeError(
                    "SEMANTIC UNAVAILABLE: sentence_transformers not installed or model failed to load. "
                    "Install with: pip install sentence-transformers or set XGB_ALLOW_NO_SEMANTIC=1."
                )
    
    @classmethod
    def from_models(
        cls,
        ranker_path: Path | str,
        decider_path: Path | str,
        partitions_dir: Path | str,
        meta_path: Path | str | None = None,
        risk_model_path: Path | str | None = None,
        risk_meta_path: Path | str | None = None,
        risk_calibrator_path: Path | str | None = None,
        calibrator_path: Path | str | None = None,
    ) -> "XgbInferenceEngine":
        """Load models and create engine."""
        # Load meta for feature order
        feature_order = FEATURE_NAMES
        ranker_feature_order = None
        if meta_path:
            meta_path = Path(meta_path)
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
                ranker_feature_order = meta.get("ranker_feature_order") or None
                ranker_fast_feature_order = meta.get("ranker_fast_feature_order") or None
                if ranker_fast_feature_order and "fast" in str(ranker_path).lower():
                    ranker_feature_order = ranker_fast_feature_order
        if "fast" in str(ranker_path).lower() and not ranker_feature_order:
            ranker_feature_order = FAST_RANKER_FEATURE_NAMES
        
        # Load ranker
        ranker = xgb.Booster()
        ranker.load_model(str(ranker_path))
        
        # Load decider
        decider = XGBClassifier()
        decider.load_model(str(decider_path))
        
        # Load store
        store = PartitionedCandidateStore(Path(partitions_dir))
        
        # Load risk model if provided
        risk_model = None
        risk_threshold = 0.835
        risk_features: List[str] = []
        if risk_model_path:
            risk_path = Path(risk_model_path)
            if risk_path.exists():
                with open(risk_path, "rb") as f:
                    risk_model = pickle.load(f)
        
        if risk_meta_path:
            risk_meta = Path(risk_meta_path)
            if risk_meta.exists():
                with open(risk_meta) as f:
                    meta = json.load(f)
                risk_threshold = meta.get("threshold", 0.835)
                risk_features = meta.get("features") or []
                if not risk_calibrator_path:
                    risk_calibrator_path = meta.get("calibrator_path")

        risk_calibrator = None
        if risk_calibrator_path:
            calib_path = Path(risk_calibrator_path)
            if calib_path.exists():
                risk_calibrator = load_calibrator(calib_path)
        
        # Load calibrator if provided
        calibrator = None
        if calibrator_path:
            calib_path = Path(calibrator_path)
            if calib_path.exists():
                with open(calib_path, "rb") as f:
                    calibrator = pickle.load(f)
        
        return cls(
            ranker=ranker,
            decider=decider,
            store=store,
            feature_order=feature_order,
            ranker_feature_order=ranker_feature_order,
            risk_model=risk_model,
            risk_calibrator=risk_calibrator,
            risk_features=risk_features,
            risk_threshold=risk_threshold,
            calibrator=calibrator,
        )
    
    @classmethod
    def from_profile(cls, profile: "InferenceProfile") -> "XgbInferenceEngine":
        """Create engine from InferenceProfile (ensures train/serve alignment).
        
        This is the recommended way to instantiate the engine as it guarantees
        configuration matches training knobs.
        """
        # Validate profile (raises if semantic mismatch)
        profile.validate(strict=True)
        
        # Load ranker (use ranker_fast for Stage 1 if available)
        ranker_path = profile.get_stage1_ranker_path()
        if not ranker_path or not ranker_path.exists():
            raise FileNotFoundError(f"Ranker not found: {ranker_path}")
        ranker = xgb.Booster()
        ranker.load_model(str(ranker_path))
        
        # Load decider
        if not profile.decider_path or not profile.decider_path.exists():
            raise FileNotFoundError(f"Decider not found: {profile.decider_path}")
        decider = XGBClassifier()
        decider.load_model(str(profile.decider_path))
        
        # Load store
        store = PartitionedCandidateStore(profile.partitions_dir)
        
        # Load risk model
        risk_model = None
        risk_calibrator = None
        risk_features: List[str] = []
        risk_threshold = profile.risk_threshold
        if profile.risk_model_path and profile.risk_model_path.exists():
            with open(profile.risk_model_path, "rb") as f:
                risk_model = pickle.load(f)
        risk_calibrator_path = None
        if profile.risk_meta_path and profile.risk_meta_path.exists():
            with open(profile.risk_meta_path) as f:
                meta = json.load(f)
            risk_features = meta.get("features") or []
            risk_threshold = meta.get("threshold", profile.risk_threshold)
            risk_calibrator_path = meta.get("calibrator_path")
        
        # Load risk calibrator (if available)
        if risk_calibrator_path:
            calib_path = Path(risk_calibrator_path)
            if calib_path.exists():
                risk_calibrator = load_calibrator(calib_path)

        # Load decider calibrator
        calibrator = None
        if profile.calibrator_path and profile.calibrator_path.exists():
            with open(profile.calibrator_path, "rb") as f:
                calibrator = pickle.load(f)
        
        return cls(
            ranker=ranker,
            decider=decider,
            store=store,
            feature_order=profile.feature_order,
            ranker_feature_order=profile.get_stage1_feature_order(),
            risk_model=risk_model,
            risk_calibrator=risk_calibrator,
            risk_features=risk_features,
            risk_threshold=risk_threshold,
            calibrator=calibrator,
            tfidf_name_mode=profile.tfidf_name_mode,
            siren_siblings=profile.siren_siblings,
            prefilter_k=profile.prefilter_k,
            char_top_k=profile.char_top_k,
            min_candidates=profile.min_candidates,
            stage1_top_n=profile.stage1_top_n,
            drop_unnamed=profile.drop_unnamed,
            exclude_closed=profile.exclude_closed,
        )
    
    def infer_single(
        self,
        crm_input: CrmInput,
        top_k: int = 5,
        pool_mode: str = "insee_then_postcode",
        drop_unnamed: bool | None = None,
        exclude_closed: bool | None = None,
    ) -> InferenceResult:
        """Run full two-stage inference on a single CRM row.
        
        Uses profile-driven config for prefilter_k, min_candidates, etc.
        """
        crm_row = crm_input.to_dict()
        crm_pre = preprocess_crm_row(crm_row)
        
        if drop_unnamed is None:
            drop_unnamed = self.drop_unnamed
        if exclude_closed is None:
            exclude_closed = self.exclude_closed

        # Build candidate pool (uses instance config from profile)
        candidates, idf_map, default_idf = self._build_candidate_pool(
            crm_row=crm_row,
            crm_id=crm_input.crm_id,
            crm_pre=crm_pre,
            pool_mode=pool_mode,
            drop_unnamed=drop_unnamed,
            exclude_closed=exclude_closed,
        )
        
        if not candidates:
            return InferenceResult(
                crm_id=crm_input.crm_id,
                status="REVIEW",
                top1_siret=None,
                top1_score=0.0,
                top2_score=0.0,
                score_gap=0.0,
                score_ratio=1.0,
                pool_size=0,
            )
        
        cand_list = [(c.get("siret"), c) for c in candidates if c.get("siret")]
        if not cand_list:
            return InferenceResult(
                crm_id=crm_input.crm_id,
                status="REVIEW",
                top1_siret=None,
                top1_score=0.0,
                top2_score=0.0,
                score_gap=0.0,
                score_ratio=1.0,
                pool_size=0,
            )
        
        # Compute IDF map from full candidate pool (pre-TF-IDF)
        set_global_name_idf_map(idf_map, default_idf)
        
        # Stage 1: Ranker (no semantic)
        feats_stage1 = [
            make_features_from_preprocessed(crm_pre, c, skip_semantic=True)
            for _, c in cand_list
        ]
        ranker_feature_order = self.ranker_feature_order or self.feature_order
        X1 = pd.DataFrame(feats_stage1)[ranker_feature_order]
        scores_stage1 = self.ranker.predict(
            xgb.DMatrix(X1.values, feature_names=ranker_feature_order)
        )
        
        # Top-N selection (use instance config)
        stage1_top_n = min(self.stage1_top_n, len(scores_stage1))
        top_n_idx = np.argsort(scores_stage1)[::-1][:stage1_top_n]
        
        # Address rescue
        rescue_indices = self._rescue_by_address(feats_stage1, set(top_n_idx))
        if rescue_indices:
            top_n_idx = np.unique(np.concatenate([top_n_idx, np.array(rescue_indices, dtype=int)]))
        
        # Stage 2: Decider (with semantic)
        feats_n = []
        cand_list_n = []
        semantic_pools = []
        
        for idx in top_n_idx:
            siret, c = cand_list[idx]
            feat = feats_stage1[idx].copy()
            cand_city_norm = c.get("_xgb_cached_city_norm") or normalize_text(c.get("city"))
            semantic_pools.append(
                build_semantic_name_pool(
                    build_candidate_names(c),
                    crm_city_norm=crm_pre.get("crm_city_norm", ""),
                    cand_city_norm=cand_city_norm,
                )
            )
            feat["_siret"] = siret
            feat["_cand_name"] = primary_name(c) or f"SIRET {siret}"
            feats_n.append(feat)
            cand_list_n.append((siret, c))
        
        # Add semantic features
        sem = top2_semantic_similarities_batch(crm_pre.get("crm_name_semantic", ""), semantic_pools)
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
        
        # Stage 2 scoring
        X_n = pd.DataFrame(feats_n)[self.feature_order]
        probs = self.decider.predict_proba(X_n.values)[:, 1]
        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(X_n.values)[:, 1]
        
        scores = np.array(probs)
        sorted_idx = np.argsort(scores)[::-1]
        topk_idx = sorted_idx[:top_k].tolist()
        topk_idx = self._promote_open_over_closed(topk_idx, cand_list_n)

        top1_idx = topk_idx[0]
        top1_score = float(scores[top1_idx]) if topk_idx else 0.0
        top2_score = float(scores[topk_idx[1]]) if len(topk_idx) > 1 else 0.0
        score_gap = top1_score - top2_score
        denom = top2_score if top2_score > 1e-6 else 0.001
        score_ratio = top1_score / denom
        top1_siret = str(cand_list_n[top1_idx][0])
        top1_feat = feats_n[top1_idx]
        
        # Determine status via risk model if available
        status = "REVIEW"
        risk_score = None
        
        if self.risk_model is not None:
            from .routing_risk import build_feature_row, default_feature_columns
            
            # Build risk features from top1 and top2
            top2_row = None
            if len(topk_idx) > 1:
                top2_idx = topk_idx[1]
                top2_row = pd.Series(feats_n[top2_idx])
                top2_row["score"] = float(scores[top2_idx])
            
            top1_row = pd.Series(top1_feat)
            top1_row["score"] = top1_score
            top1_row["score_gap"] = score_gap
            top1_row["score_ratio"] = score_ratio
            
            feature_list = self.risk_features or default_feature_columns()
            risk_features = build_feature_row(top1_row, top2_row, feature_list)
            if self.risk_calibrator is not None:
                risk_score = float(self.risk_calibrator.predict_proba([risk_features])[:, 1][0])
            else:
                risk_score = float(self.risk_model.predict_proba([risk_features])[:, 1][0])
            
            if risk_score >= self.risk_threshold:
                status = "AUTO"
        
        # Build output candidates
        candidates_out = []
        for rank, idx_k in enumerate(topk_idx, start=1):
            siret_k, cand_k = cand_list_n[idx_k]
            candidates_out.append({
                "siret": siret_k,
                "score": float(scores[idx_k]),
                "rank": rank,
                "name": primary_name(cand_k) or "",
                "address": build_address(cand_k),
                "city": cand_k.get("city"),
                "postcode": cand_k.get("postcode"),
            })
        
        return InferenceResult(
            crm_id=crm_input.crm_id,
            status=status,
            top1_siret=top1_siret,
            top1_score=top1_score,
            top2_score=top2_score,
            score_gap=score_gap,
            score_ratio=score_ratio,
            pool_size=len(cand_list_n),
            candidates=candidates_out,
            risk_score=risk_score,
            features=top1_feat,
        )
    
    def _build_candidate_pool(
        self,
        crm_row: Dict[str, Any],
        crm_id: str,
        crm_pre: Dict[str, Any],
        pool_mode: str,
        drop_unnamed: bool,
        exclude_closed: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], float]:
        """Build candidate pool from partitioned store.
        
        Uses instance config (prefilter_k, min_candidates, tfidf_name_mode, siren_siblings).
        crm_id is used as seed for deterministic padding.
        """
        insee = crm_row.get("insee")
        postcode = crm_row.get("postcode")
        crm_name = crm_row.get("crm_name", "")
        
        base_candidates: List[Dict] = []
        whitelisted_sirets: set[str] = set()

        # Load base candidates
        if pool_mode == "insee_then_postcode":
            if insee:
                base_candidates = self.store.load_by_insee(insee)
            if not base_candidates and postcode:
                base_candidates = self.store.load_by_postcode(postcode)
        else:
            base_candidates = self.store.load_by_insee(insee) + self.store.load_by_postcode(postcode)

        base_candidates = self._apply_candidate_filters(
            base_candidates,
            drop_unnamed=drop_unnamed,
            exclude_closed=exclude_closed,
        )
        pool = dedupe_candidates(base_candidates)

        if pool_mode == "multi":
            dept_key = department_from_code(insee, postcode)
            cache_item = self._get_dept_cache_item(
                dept_key=dept_key,
                insee=insee,
                postcode=postcode,
                drop_unnamed=drop_unnamed,
                exclude_closed=exclude_closed,
            )
            if cache_item:
                dept_candidates = cache_item.candidates
                extra: List[Dict[str, Any]] = []

                addr_h = address_hash(
                    crm_pre.get("crm_street_num"),
                    crm_pre.get("crm_street_name"),
                )
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

                for cand in extra:
                    siret = str(cand.get("siret") or "")
                    if siret:
                        pool[siret] = cand

        pool_candidates = list(pool.values())
        candidates_dict = {
            str(c.get("siret") or ""): c for c in pool_candidates if c.get("siret")
        }
        if candidates_dict:
            idf_map, default_idf = compute_name_idf_map(candidates_dict)
        else:
            idf_map, default_idf = {}, 0.0
        attach_address_density(pool_candidates)

        candidates = pool_candidates
        
        # TF-IDF prefilter (uses instance config)
        prefilter_k = self.prefilter_k
        min_candidates = self.min_candidates
        char_top_k = self.char_top_k
        
        if prefilter_k and len(candidates) > prefilter_k:
            # Build TF-IDF index with profile-aligned settings
            vec, mat, names = build_tfidf_index(
                candidates,
                name_mode=self.tfidf_name_mode,
                siren_siblings=self.siren_siblings,
            )
            char_vec, char_mat = (None, None)
            if names:
                char_vec, char_mat = build_char_tfidf_index(names)
            if vec is not None and mat is not None:
                idx = prefilter_candidates_tfidf(
                    crm_name,
                    vec,
                    mat,
                    prefilter_k,
                    cand_names=names,
                    char_top_k=char_top_k,
                    char_vectorizer=char_vec,
                    char_matrix=char_mat,
                )
                
                # Identify TF-IDF winners
                tfidf_cands = [candidates[i] for i in idx]
                tfidf_sirets = {str(c.get("siret")) for c in tfidf_cands if c.get("siret")}
                
                # Whitelist rescue: ensure strong address matches survive TF-IDF
                final_cands = list(tfidf_cands)
                for cand in pool.values():
                    siret = str(cand.get("siret") or "")
                    if siret in whitelisted_sirets and siret not in tfidf_sirets:
                        final_cands.append(cand)
                
                if len(final_cands) >= min_candidates:
                    candidates = final_cands
                else:
                    # Deterministic padding with crm_id as seed
                    import random
                    final_sirets = {str(c.get("siret") or "") for c in final_cands if c.get("siret")}
                    remaining = [c for c in candidates if str(c.get("siret") or "") not in final_sirets]
                    needed = min(min_candidates, prefilter_k) - len(final_cands)
                    if needed > 0 and remaining:
                        # Deterministic seed aligned with legacy script (crm_id or crm_name fallback)
                        seed_val = str(crm_id) if crm_id else ""
                        if not seed_val:
                            seed_val = crm_row.get("crm_name", "")

                        seed = self._stable_seed(seed_val)
                        rng = random.Random(seed)
                        random_extra = rng.sample(remaining, min(needed, len(remaining)))
                        candidates = final_cands + random_extra
                    else:
                        candidates = final_cands
        
        return candidates, idf_map, default_idf

    def _get_dept_cache_item(
        self,
        *,
        dept_key: str | None,
        insee: str | None,
        postcode: str | None,
        drop_unnamed: bool,
        exclude_closed: bool,
    ) -> DeptCacheItem | None:
        if not dept_key:
            return None
        cached = self._dept_cache.get(dept_key)
        if cached is not None:
            self._dept_cache.move_to_end(dept_key)
            return cached

        dept_candidates = self.store.load_by_department(insee, postcode)
        if not dept_candidates:
            return None
        dept_candidates = self._apply_candidate_filters(
            dept_candidates,
            drop_unnamed=drop_unnamed,
            exclude_closed=exclude_closed,
        )

        vec, mat, names = build_tfidf_index(
            dept_candidates,
            name_mode=self.tfidf_name_mode,
            siren_siblings=self.siren_siblings,
        )
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
            char_vec=char_vec,
            char_mat=char_mat,
            addr_index=addr_index,
            num_index=num_index,
        )
        self._dept_cache[dept_key] = item
        while len(self._dept_cache) > self._max_dept_cache:
            self._dept_cache.popitem(last=False)
        return item

    def _apply_candidate_filters(
        self,
        candidates: List[Dict[str, Any]],
        *,
        drop_unnamed: bool,
        exclude_closed: bool,
    ) -> List[Dict[str, Any]]:
        out = candidates
        if drop_unnamed:
            out = [c for c in out if build_candidate_names(c)]
        if exclude_closed:
            out = [c for c in out if str(c.get("etat_admin") or "").strip().upper() != "F"]
        return out
    
    def _rescue_by_address(
        self,
        feats_stage1: List[Dict],
        top_n_set: set,
    ) -> List[int]:
        """Rescue candidates with near-perfect address match."""
        rescue = []
        for i, feat in enumerate(feats_stage1):
            if i in top_n_set:
                continue
            addr_jaro = float(feat.get("addr_jaro", 0.0))
            street_name_jaro = float(feat.get("street_name_jaro", 0.0))
            street_number_diff = float(feat.get("street_number_diff", 9999))
            if addr_jaro >= 0.96 and street_name_jaro >= 0.95 and street_number_diff <= 2:
                rescue.append(i)
        return rescue[:50]

    @staticmethod
    def _stable_seed(value: str) -> int:
        base = value or ""
        digest = hashlib.md5(base.encode("utf-8")).hexdigest()
        return int(digest, 16) & 0x7FFFFFFF

    @staticmethod
    def _is_closed(cand: Dict[str, Any]) -> bool:
        return str(cand.get("etat_admin") or "").strip().upper() == "F"

    def _promote_open_over_closed(
        self,
        topk_idx: List[int],
        cand_list_n: List[Tuple[str, Dict[str, Any]]],
    ) -> List[int]:
        """Promote open establishments when same SIREN has open/closed in top-k."""
        if not topk_idx:
            return topk_idx

        order = list(topk_idx)
        pos_by_idx = {idx: pos for pos, idx in enumerate(order)}
        open_pos: Dict[str, int] = {}
        closed_pos: Dict[str, int] = {}

        for pos, idx in enumerate(order):
            cand = cand_list_n[idx][1]
            siren = str(cand.get("siren") or "").strip()
            if not siren:
                continue
            if self._is_closed(cand):
                closed_pos.setdefault(siren, pos)
            else:
                open_pos.setdefault(siren, pos)

        for siren, c_pos in closed_pos.items():
            o_pos = open_pos.get(siren)
            if o_pos is None or c_pos >= o_pos:
                continue
            idx_closed = order[c_pos]
            idx_open = order[o_pos]
            order[c_pos], order[o_pos] = idx_open, idx_closed
            pos_by_idx[idx_closed], pos_by_idx[idx_open] = o_pos, c_pos

        return order


__all__ = [
    "XgbInferenceEngine",
    "InferenceResult",
    "CrmInput",
]
