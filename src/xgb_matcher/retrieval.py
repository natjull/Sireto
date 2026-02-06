"""Unified retrieval module (Variant B / SSOT).

Single code-path for candidate pool construction (train + serve).
Strict insee_then_postcode, no department fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
from typing import Dict, List, Optional, Set, Tuple

from .partitioned_store import PartitionedCandidateStore
from .retrieval_config import RetrievalConfigV1
from .candidates import compute_name_idf_map
from .blocking import (
    build_tfidf_index,
    build_address_tfidf_index,
    build_char_tfidf_index,
    prefilter_candidates_tfidf,
    prefilter_candidates_address_tfidf,
    dedupe_candidates,
    address_hash,
    extract_numeric_tokens,
    attach_address_density,
    build_address_hash_index,
    build_numeric_token_index,
)


@dataclass
class CandidatePoolResult:
    """Result of candidate pool construction with GT tracking."""

    candidates: List[dict]

    # Prefilter order (TF-IDF/union) for cheap sampling
    prefilter_sirets: List[str] = field(default_factory=list)

    # GT tracking (for train/eval)
    gt_in_base_pool: bool = False
    gt_in_filtered_pool: bool = False
    gt_in_tfidf_pool: bool = False
    gt_was_injected: bool = False

    # Pool sizes at each stage
    pool_sizes: Dict[str, int] = field(default_factory=dict)

    # IDF map (from full pool, pre-TF-IDF)
    idf_map: Dict[str, float] = field(default_factory=dict)
    default_idf: float = 0.0

    # Loss reason if GT not in final pool
    loss_reason: str = ""


def _check_siret_in_list(candidates: List[dict], siret: str | None) -> bool:
    if not siret:
        return False
    siret_norm = str(siret).zfill(14)
    return any(str(c.get("siret", "")).zfill(14) == siret_norm for c in candidates)


def _get_siret_set(candidates: List[dict]) -> Set[str]:
    return {str(c.get("siret", "")).zfill(14) for c in candidates if c.get("siret")}


def _apply_filters(
    candidates: List[dict],
    drop_unnamed: bool,
    include_closed: bool,
) -> List[dict]:
    out = []
    for c in candidates:
        if drop_unnamed:
            has_name = any([
                c.get("denomination"),
                c.get("denomination_usuelle_ul"),
                c.get("enseigne1"),
                c.get("enseigne2"),
                c.get("enseigne3"),
                c.get("denomination_ul"),
                c.get("sigle_ul"),
                c.get("nom_ul"),
                c.get("prenom_usuel_ul"),
            ])
            if not has_name:
                continue
        if not include_closed:
            if c.get("etat_admin") == "F":
                continue
        out.append(c)
    return out


def _stable_seed(value: str) -> int:
    base = value or ""
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()
    return int(digest, 16) & 0x7FFFFFFF


def build_candidate_pool(
    store: PartitionedCandidateStore,
    crm_row: dict,
    crm_pre: dict,
    config: RetrievalConfigV1,
    tfidf_cache: Dict[Tuple[str, str], tuple],
    gt_siret: str | None = None,
) -> CandidatePoolResult:
    """Build candidate pool with unified code-path for train and inference."""
    result = CandidatePoolResult(candidates=[])

    insee = crm_row.get("insee") or crm_row.get("crm_insee")
    postcode = crm_row.get("postcode") or crm_row.get("crm_cp")
    crm_name = crm_row.get("crm_name", "")
    crm_address = crm_pre.get("crm_addr", "")

    gt_norm = str(gt_siret).zfill(14) if gt_siret else None

    # Step 1: Load base candidates (strict insee_then_postcode + mega policy)
    base_candidates = store.load_by_insee_then_postcode(
        insee,
        postcode,
        mega_insee_max_rows=config.mega_insee_max_rows,
        mega_insee_policy=config.mega_insee_policy,
    )

    result.pool_sizes["base"] = len(base_candidates)
    result.gt_in_base_pool = _check_siret_in_list(base_candidates, gt_norm)

    # Step 2: Apply filters + dedupe
    filtered = _apply_filters(base_candidates, config.drop_unnamed, config.include_closed)
    pool = dedupe_candidates(filtered)
    pool_list = list(pool.values())

    result.pool_sizes["filtered"] = len(pool_list)
    result.gt_in_filtered_pool = gt_norm in _get_siret_set(pool_list) if gt_norm else False

    candidates_dict = {str(c.get("siret") or ""): c for c in pool_list if c.get("siret")}
    if candidates_dict:
        idf_map, default_idf = compute_name_idf_map(candidates_dict)
        result.idf_map = idf_map
        result.default_idf = float(default_idf)

    if result.gt_in_base_pool and not result.gt_in_filtered_pool:
        if not config.include_closed:
            result.loss_reason = "FILTERED_CLOSED"
        elif config.drop_unnamed:
            result.loss_reason = "FILTERED_UNNAMED"
        else:
            result.loss_reason = "FILTERED_OTHER"

    # Step 3: Universal rescue whitelist (addr_hash + numeric tokens)
    whitelisted_sirets: set[str] = set()
    if pool_list:
        addr_index = build_address_hash_index(pool_list)
        num_index = build_numeric_token_index(pool_list)

        addr_h = address_hash(
            crm_pre.get("crm_street_num"),
            crm_pre.get("crm_street_name"),
        )
        if config.rescue_addr_hash and addr_h and addr_index:
            for idx in addr_index.get(addr_h, []):
                if idx < len(pool_list):
                    siret = str(pool_list[idx].get("siret") or "")
                    if siret:
                        whitelisted_sirets.add(siret)

        numeric_tokens = extract_numeric_tokens(crm_name)
        if config.rescue_numeric_tokens and numeric_tokens and num_index:
            for token in numeric_tokens:
                for idx in num_index.get(token, []):
                    if idx < len(pool_list):
                        siret = str(pool_list[idx].get("siret") or "")
                        if siret:
                            whitelisted_sirets.add(siret)

    # Step 4: TF-IDF prefilter (Name + Address union)
    candidates = pool_list
    if config.prefilter_k and len(candidates) > config.prefilter_k:
        cache_key = ("main", f"{insee}_{postcode}")
        name_vec, name_mat, names = tfidf_cache.get(cache_key, (None, None, None))
        if name_vec is None:
            name_vec, name_mat, names = build_tfidf_index(
                candidates,
                name_mode=config.tfidf_name_mode,
                siren_siblings=config.siren_siblings,
            )
            tfidf_cache[cache_key] = (name_vec, name_mat, names)

        char_vec = None
        char_mat = None
        if names:
            char_vec, char_mat = build_char_tfidf_index(names)

        addr_vec, addr_mat = build_address_tfidf_index(candidates)

        name_idx: List[int] = []
        if name_vec is not None and name_mat is not None:
            name_idx = prefilter_candidates_tfidf(
                crm_name,
                name_vec,
                name_mat,
                config.prefilter_k,
                cand_names=names,
                char_top_k=config.char_top_k,
                char_vectorizer=char_vec,
                char_matrix=char_mat,
            )

        addr_idx: List[int] = []
        if addr_vec is not None and addr_mat is not None:
            addr_idx = prefilter_candidates_address_tfidf(
                crm_address,
                addr_vec,
                addr_mat,
                config.prefilter_k,
            )

        combined_idx = list(dict.fromkeys(name_idx + addr_idx))
        if config.prefilter_union_cap:
            combined_idx = combined_idx[: config.prefilter_union_cap]

        if combined_idx:
            tfidf_cands = [candidates[i] for i in combined_idx if i < len(candidates)]
            tfidf_sirets = {str(c.get("siret")) for c in tfidf_cands if c.get("siret")}

            final_cands = list(tfidf_cands)
            for cand in pool.values():
                siret = str(cand.get("siret") or "")
                if siret in whitelisted_sirets and siret not in tfidf_sirets:
                    final_cands.append(cand)

            if len(final_cands) >= config.min_candidates:
                candidates = final_cands
            else:
                final_sirets = {str(c.get("siret") or "") for c in final_cands if c.get("siret")}
                remaining = [c for c in candidates if str(c.get("siret") or "") not in final_sirets]
                needed = min(config.min_candidates, config.prefilter_k) - len(final_cands)
                if needed > 0 and remaining:
                    seed_val = str(crm_row.get("crm_id") or "") or crm_name
                    rng = random.Random(_stable_seed(seed_val))
                    random_extra = rng.sample(remaining, min(needed, len(remaining)))
                    candidates = final_cands + random_extra
                else:
                    candidates = final_cands

    result.pool_sizes["tfidf"] = len(candidates)
    result.gt_in_tfidf_pool = _check_siret_in_list(candidates, gt_norm)

    if result.gt_in_filtered_pool and not result.gt_in_tfidf_pool and not result.loss_reason:
        result.loss_reason = "PRUNED_BY_TFIDF"

    if not result.gt_in_base_pool and not result.loss_reason:
        result.loss_reason = "NOT_IN_PARTITION"

    attach_address_density(candidates)
    result.prefilter_sirets = [str(c.get("siret") or "") for c in candidates if c.get("siret")]
    result.candidates = candidates
    return result


def inject_gt_if_missing(
    result: CandidatePoolResult,
    gt_candidate: dict | None,
) -> CandidatePoolResult:
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


__all__ = [
    "CandidatePoolResult",
    "build_candidate_pool",
    "inject_gt_if_missing",
]
