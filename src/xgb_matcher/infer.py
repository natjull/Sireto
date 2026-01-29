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

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from .blocking import (
    attach_address_density,
    dedupe_candidates,
    build_tfidf_index,
    prefilter_candidates_tfidf,
)
from .features import (
    FEATURE_NAMES,
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
from .semantic import top2_semantic_similarities_batch

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
        risk_model: Any | None = None,
        risk_calibrator: Any | None = None,
        risk_features: List[str] | None = None,
        risk_threshold: float = 0.835,
        calibrator: Any | None = None,
    ):
        self.ranker = ranker
        self.decider = decider
        self.store = store
        self.feature_order = feature_order
        self.risk_model = risk_model
        self.risk_calibrator = risk_calibrator
        self.risk_features = risk_features or []
        self.risk_threshold = risk_threshold
        self.calibrator = calibrator
        self._tfidf_cache: Dict[Tuple[str, str], tuple] = {}
    
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
        if meta_path:
            meta_path = Path(meta_path)
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
        
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

        risk_calibrator = None
        if risk_calibrator_path:
            calib_path = Path(risk_calibrator_path)
            if calib_path.exists():
                with open(calib_path, "rb") as f:
                    risk_calibrator = pickle.load(f)
        
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
            risk_model=risk_model,
            risk_calibrator=risk_calibrator,
            risk_features=risk_features,
            risk_threshold=risk_threshold,
            calibrator=calibrator,
        )
    
    def infer_single(
        self,
        crm_input: CrmInput,
        top_k: int = 5,
        prefilter_k: int = 500,
        pool_mode: str = "insee_then_postcode",
        drop_unnamed: bool = True,
        exclude_closed: bool = False,
    ) -> InferenceResult:
        """Run full two-stage inference on a single CRM row.
        
        This is the core logic extracted from infer_xgb_two_stage.py.
        """
        crm_row = crm_input.to_dict()
        crm_pre = preprocess_crm_row(crm_row)
        
        # Build candidate pool
        candidates = self._build_candidate_pool(
            crm_row=crm_row,
            pool_mode=pool_mode,
            prefilter_k=prefilter_k,
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
        
        # Compute IDF map
        candidates_dict = {str(siret): c for siret, c in cand_list if siret}
        idf_map, default_idf = compute_name_idf_map(candidates_dict)
        set_global_name_idf_map(idf_map, default_idf)
        
        # Stage 1: Ranker (no semantic)
        feats_stage1 = [
            make_features_from_preprocessed(crm_pre, c, skip_semantic=True)
            for _, c in cand_list
        ]
        X1 = pd.DataFrame(feats_stage1)[self.feature_order]
        scores_stage1 = self.ranker.predict(
            xgb.DMatrix(X1.values, feature_names=self.feature_order)
        )
        
        # Top-N selection
        stage1_top_n = min(200, len(scores_stage1))
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
        scores_sorted = np.sort(scores)[::-1]
        top1_score = float(scores_sorted[0]) if len(scores_sorted) else 0.0
        top2_score = float(scores_sorted[1]) if len(scores_sorted) > 1 else 0.0
        score_gap = top1_score - top2_score
        score_ratio = top1_score / (top2_score + 1e-9) if top2_score > 0 else 1.0
        
        # Get top-k
        topk_idx = np.argsort(scores)[::-1][:top_k]
        top1_idx = topk_idx[0]
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
        pool_mode: str,
        prefilter_k: int,
        drop_unnamed: bool,
        exclude_closed: bool,
    ) -> List[Dict[str, Any]]:
        """Build candidate pool from partitioned store."""
        insee = crm_row.get("insee")
        postcode = crm_row.get("postcode")
        crm_name = crm_row.get("crm_name", "")
        
        base_candidates: List[Dict] = []
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
        candidates = list(pool.values())
        
        # TF-IDF prefilter
        MIN_CANDIDATES = 100
        if prefilter_k and len(candidates) > prefilter_k:
            vec, mat, names = build_tfidf_index(candidates)
            if vec is not None and mat is not None:
                idx = prefilter_candidates_tfidf(
                    crm_name, vec, mat, prefilter_k,
                    cand_names=names, char_top_k=min(200, prefilter_k),
                )
                if len(idx) >= MIN_CANDIDATES:
                    candidates = [candidates[i] for i in idx]
                else:
                    tfidf_cands = [candidates[i] for i in idx]
                    tfidf_set = set(idx)
                    remaining = [c for i, c in enumerate(candidates) if i not in tfidf_set]
                    needed = MIN_CANDIDATES - len(tfidf_cands)
                    if needed > 0 and remaining:
                        import random
                        random_extra = random.sample(remaining, min(needed, len(remaining)))
                        candidates = tfidf_cands + random_extra
                    else:
                        candidates = tfidf_cands
        
        attach_address_density(candidates)
        return candidates

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


__all__ = [
    "XgbInferenceEngine",
    "InferenceResult",
    "CrmInput",
]
