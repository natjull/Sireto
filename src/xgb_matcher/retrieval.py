"""
Unified retrieval module for train/serve alignment.

This module provides a single code-path for candidate pool construction,
used by both generate_training_samples and infer_xgb_two_stage.

Key features:
- Configurable via RetrievalKnobs dataclass
- GT tracking for train/eval without injection bias
- Multi-strategy rescue (addr hash, numeric tokens, semantic)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .partitioned_store import PartitionedCandidateStore
from .blocking import (
    build_tfidf_index,
    prefilter_candidates_tfidf,
    dedupe_candidates,
    address_hash,
    filter_candidates_by_address_hash,
    extract_numeric_tokens,
    filter_candidates_by_numeric_tokens,
    department_from_code,
    attach_address_density,
)


@dataclass
class RetrievalKnobs:
    """Configuration for retrieval pipeline."""
    
    # Pool construction
    include_closed: bool = True
    drop_unnamed: bool = True
    pool_mode: str = "multi"  # "insee_then_postcode" | "multi"
    
    # TF-IDF prefilter
    tfidf_name_mode: str = "bag"  # "bag" | "primary"
    prefilter_k: int = 500
    min_candidates: int = 100
    char_top_k: int = 200
    
    # Department fallback (for pool_mode="multi")
    dept_prefilter_k: int = 100
    max_dept_candidates: int = 2000
    
    # Rescue strategies
    rescue_addr_hash: bool = True
    rescue_numeric_tokens: bool = True
    rescue_semantic_k: int = 50
    semantic_min_sim: float = 0.3


@dataclass
class CandidatePoolResult:
    """Result of candidate pool construction with GT tracking."""
    
    candidates: List[dict]
    
    # GT tracking (for train/eval)
    gt_in_base_pool: bool = False
    gt_in_filtered_pool: bool = False
    gt_in_tfidf_pool: bool = False
    gt_was_injected: bool = False  # True if GT was added because it was missing
    
    # Pool sizes at each stage
    pool_sizes: Dict[str, int] = field(default_factory=dict)
    
    # Loss reason if GT not in final pool
    loss_reason: str = ""


def _check_siret_in_list(candidates: List[dict], siret: str | None) -> bool:
    """Check if a SIRET is in candidate list."""
    if not siret:
        return False
    siret_norm = str(siret).zfill(14)
    return any(str(c.get("siret", "")).zfill(14) == siret_norm for c in candidates)


def _get_siret_set(candidates: List[dict]) -> Set[str]:
    """Get set of normalized SIRETs from candidates."""
    return {str(c.get("siret", "")).zfill(14) for c in candidates if c.get("siret")}


def _apply_filters(
    candidates: List[dict],
    drop_unnamed: bool,
    include_closed: bool,
) -> List[dict]:
    """Apply drop_unnamed and closed filters."""
    out = []
    for c in candidates:
        # Drop unnamed filter
        if drop_unnamed:
            has_name = any([
                c.get("denomination"),
                c.get("enseigne1"),
                c.get("denomination_ul"),
                c.get("sigle_ul"),
            ])
            if not has_name:
                continue
        
        # Closed filter
        if not include_closed:
            if c.get("etat_admin") == "F":
                continue
        
        out.append(c)
    return out


def build_candidate_pool(
    store: PartitionedCandidateStore,
    crm_row: dict,
    crm_pre: dict,
    knobs: RetrievalKnobs,
    tfidf_cache: Dict[Tuple[str, str], tuple],
    gt_siret: str | None = None,
    semantic_search_fn: Optional[callable] = None,
) -> CandidatePoolResult:
    """
    Build candidate pool with unified code-path for train and inference.
    
    Args:
        store: Partitioned candidate store
        crm_row: Raw CRM row dict
        crm_pre: Preprocessed CRM row (from preprocess_crm_row)
        knobs: Retrieval configuration
        tfidf_cache: Cache for TF-IDF indices (by partition key)
        gt_siret: Ground truth SIRET (for tracking, NOT for injection)
        semantic_search_fn: Optional function for semantic rescue
    
    Returns:
        CandidatePoolResult with candidates and GT tracking info
    """
    result = CandidatePoolResult(candidates=[])
    
    insee = crm_row.get("insee") or crm_row.get("crm_insee")
    postcode = crm_row.get("postcode") or crm_row.get("crm_cp")
    crm_name = crm_row.get("crm_name", "")
    
    gt_norm = str(gt_siret).zfill(14) if gt_siret else None
    
    # Step 1: Load base candidates from partitions
    base_candidates: List[dict] = []
    if knobs.pool_mode == "insee_then_postcode":
        if insee:
            base_candidates = store.load_by_insee(insee)
        if not base_candidates and postcode:
            base_candidates = store.load_by_postcode(postcode)
    else:
        # Union mode
        insee_cands = store.load_by_insee(insee) if insee else []
        cp_cands = store.load_by_postcode(postcode) if postcode else []
        base_candidates = insee_cands + cp_cands
    
    result.pool_sizes["base"] = len(base_candidates)
    result.gt_in_base_pool = _check_siret_in_list(base_candidates, gt_norm)
    
    # Step 2: Apply filters
    filtered = _apply_filters(base_candidates, knobs.drop_unnamed, knobs.include_closed)
    pool = dedupe_candidates(filtered)
    
    result.pool_sizes["filtered"] = len(pool)
    result.gt_in_filtered_pool = gt_norm in _get_siret_set(list(pool.values())) if gt_norm else False
    
    if result.gt_in_base_pool and not result.gt_in_filtered_pool:
        if not knobs.include_closed:
            result.loss_reason = "FILTERED_CLOSED"
        elif knobs.drop_unnamed:
            result.loss_reason = "FILTERED_UNNAMED"
        else:
            result.loss_reason = "FILTERED_OTHER"
    
    # Step 3: Department fallback and rescues (for pool_mode="multi")
    if knobs.pool_mode == "multi":
        dept_candidates = store.load_by_department(insee, postcode)
        dept_candidates = _apply_filters(dept_candidates, knobs.drop_unnamed, knobs.include_closed)
        if knobs.max_dept_candidates and len(dept_candidates) > knobs.max_dept_candidates:
            dept_candidates = dept_candidates[:knobs.max_dept_candidates]
        
        extra: List[dict] = []
        
        # Rescue: address hash
        if knobs.rescue_addr_hash:
            crm_addr_hash = address_hash(
                crm_pre.get("crm_street_num"),
                crm_pre.get("crm_street_name")
            )
            if crm_addr_hash:
                extra.extend(filter_candidates_by_address_hash(dept_candidates, crm_addr_hash))
        
        # Rescue: numeric tokens
        if knobs.rescue_numeric_tokens:
            numeric_tokens = extract_numeric_tokens(crm_name)
            if numeric_tokens:
                extra.extend(filter_candidates_by_numeric_tokens(dept_candidates, numeric_tokens))
        
        # Rescue: TF-IDF on department
        if dept_candidates and knobs.dept_prefilter_k > 0:
            dept_key = department_from_code(insee, postcode) or "unknown"
            cache_key = ("dept", dept_key)
            vec, mat, names = tfidf_cache.get(cache_key, (None, None, None))
            if vec is None:
                vec, mat, names = build_tfidf_index(dept_candidates, name_mode=knobs.tfidf_name_mode)
                tfidf_cache[cache_key] = (vec, mat, names)
            if vec is not None and mat is not None:
                idx = prefilter_candidates_tfidf(
                    crm_name,
                    vec,
                    mat,
                    min(knobs.dept_prefilter_k, len(dept_candidates)),
                    cand_names=names,
                    char_top_k=min(200, knobs.char_top_k),
                )
                extra.extend([dept_candidates[i] for i in idx])
        
        # Rescue: semantic
        if semantic_search_fn and knobs.rescue_semantic_k > 0:
            sem_hits = semantic_search_fn(
                crm_name,
                dept_candidates,
                knobs.rescue_semantic_k,
                knobs.semantic_min_sim,
            )
            extra.extend(sem_hits)
        
        # Add extra candidates to pool
        if knobs.max_dept_candidates and len(extra) > knobs.max_dept_candidates:
            extra = extra[:knobs.max_dept_candidates]
        
        for cand in extra:
            siret = str(cand.get("siret") or "")
            if siret:
                pool[siret] = cand
    
    candidates = list(pool.values())
    result.pool_sizes["after_rescues"] = len(candidates)
    
    # Step 4: TF-IDF prefilter
    pre_tfidf_size = len(candidates)
    if knobs.prefilter_k and len(candidates) > knobs.prefilter_k:
        cache_key = ("main", f"{insee}_{postcode}")
        vec, mat, names = tfidf_cache.get(cache_key, (None, None, None))
        if vec is None:
            vec, mat, names = build_tfidf_index(candidates, name_mode=knobs.tfidf_name_mode)
            tfidf_cache[cache_key] = (vec, mat, names)
        
        if vec is not None and mat is not None:
            idx = prefilter_candidates_tfidf(
                crm_name,
                vec,
                mat,
                knobs.prefilter_k,
                cand_names=names,
                char_top_k=knobs.char_top_k,
            )
            
            # Ensure minimum candidates
            if len(idx) >= knobs.min_candidates:
                candidates = [candidates[i] for i in idx]
            else:
                tfidf_cands = [candidates[i] for i in idx]
                tfidf_set = set(idx)
                remaining = [c for i, c in enumerate(candidates) if i not in tfidf_set]
                needed = min(knobs.min_candidates, knobs.prefilter_k) - len(tfidf_cands)
                if needed > 0 and remaining:
                    import random
                    random_extra = random.sample(remaining, min(needed, len(remaining)))
                    candidates = tfidf_cands + random_extra
                else:
                    candidates = tfidf_cands
    
    result.pool_sizes["tfidf"] = len(candidates)
    result.gt_in_tfidf_pool = _check_siret_in_list(candidates, gt_norm)
    
    if result.gt_in_filtered_pool and not result.gt_in_tfidf_pool and not result.loss_reason:
        result.loss_reason = "PRUNED_BY_TFIDF"
    
    if not result.gt_in_base_pool and not result.loss_reason:
        result.loss_reason = "NOT_IN_PARTITION"
    
    # Attach address density
    attach_address_density(candidates)
    
    result.candidates = candidates
    return result


def inject_gt_if_missing(
    result: CandidatePoolResult,
    gt_candidate: dict | None,
) -> CandidatePoolResult:
    """
    Inject GT candidate into pool if missing (for training only).
    
    Sets gt_was_injected=True if injection occurred.
    """
    if gt_candidate is None:
        return result
    
    gt_siret = str(gt_candidate.get("siret", "")).zfill(14)
    if not gt_siret:
        return result
    
    existing_sirets = _get_siret_set(result.candidates)
    if gt_siret not in existing_sirets:
        result.candidates.append(gt_candidate)
        result.gt_was_injected = True
    
    return result
