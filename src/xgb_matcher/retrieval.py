"""Unified retrieval module (Variant B / SSOT) — with hybrid dense+sparse support.

Single code-path for candidate pool construction (train + serve).
Strict insee_then_postcode, no department fallback.

P0: Persistent TF-IDF cache + timing instrumentation.
P1: Optional hybrid dense (FAISS) + sparse (TF-IDF) retrieval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import hashlib
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .partitioned_store import PartitionedCandidateStore
from .retrieval_config import RetrievalConfigV1
from .candidates import compute_name_idf_map
from .fusion import annotate_fused_candidate, reciprocal_rank_fusion
from .blocking import (
    build_tfidf_index,
    build_address_tfidf_index,
    build_char_tfidf_index,
    prefilter_candidates_tfidf,
    prefilter_candidates_tfidf_scored,
    prefilter_candidates_address_tfidf,
    prefilter_candidates_address_tfidf_scored,
    dedupe_candidates,
    address_hash,
    extract_numeric_tokens,
    attach_address_density,
    build_address_hash_index,
    build_numeric_token_index,
)

if TYPE_CHECKING:
    from .tfidf_cache import TfidfPersistentCache
    from .dense_retrieval import PartitionEmbeddingStore
    from .timing import PipelineTimer

_logger = logging.getLogger(__name__)


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


def _expand_ranked_sirens(
    *,
    ranked_sirens: List[Tuple[str, float]],
    store: PartitionedCandidateStore,
    siren_to_geo: Any,
    crm_insee: str,
    crm_postcode: str,
    config: RetrievalConfigV1,
    siren_candidate_store: Any = None,
) -> tuple[List[dict], Dict[str, int]]:
    """Expand ranked SIRENs to filtered SIRETs, preserving SIREN rank."""
    candidates: List[dict] = []
    siren_rank_by_siret: Dict[str, int] = {}
    seen_sirets: Set[str] = set()
    loaded_partitions: Dict[str, List[dict]] = {}

    def load_partition(insee_code: str) -> List[dict]:
        if insee_code not in loaded_partitions:
            loaded_partitions[insee_code] = store.load_by_insee(insee_code)
        return loaded_partitions[insee_code]

    direct_candidates = (
        siren_candidate_store.get_candidates(
            [siren for siren, _score in ranked_sirens],
            crm_insee=str(crm_insee),
            crm_postcode=str(crm_postcode),
            max_per_siren=max(1, config.max_sirets_per_siren * 2),
        )
        if siren_candidate_store is not None
        else None
    )
    for siren_rank, (siren, _score) in enumerate(ranked_sirens, start=1):
        locations = siren_to_geo.get_locations(siren)
        locations = sorted(
            locations,
            key=lambda location: (
                str(location[0]) != str(crm_insee),
                str(location[1]) != str(crm_postcode),
            ),
        )
        selected_for_siren = 0
        if direct_candidates is not None:
            location_counts = dict(
                (
                    (str(insee), str(postcode)),
                    int(count),
                )
                for insee, postcode, count
                in siren_to_geo.get_location_counts(siren)
            )
            candidates_for_siren = sorted(
                direct_candidates.get(str(siren), []),
                key=lambda candidate: (
                    str(candidate.get("insee") or "") != str(crm_insee),
                    str(candidate.get("postcode") or "") != str(crm_postcode),
                    -location_counts.get(
                        (
                            str(candidate.get("insee") or ""),
                            str(candidate.get("postcode") or ""),
                        ),
                        0,
                    ),
                    str(candidate.get("siret") or ""),
                ),
            )
            for candidate in candidates_for_siren:
                siret = str(candidate.get("siret") or "")
                if (
                    not siret
                    or siret in seen_sirets
                    or not _apply_filters(
                        [candidate],
                        config.drop_unnamed,
                        config.include_closed,
                    )
                ):
                    continue
                candidates.append(candidate)
                siren_rank_by_siret[siret] = siren_rank
                seen_sirets.add(siret)
                selected_for_siren += 1
                if selected_for_siren >= config.max_sirets_per_siren:
                    break
            continue
        for insee_code, _postcode in locations:
            if selected_for_siren >= config.max_sirets_per_siren:
                break
            for candidate in load_partition(str(insee_code)):
                siret = str(candidate.get("siret") or "")
                if (
                    str(candidate.get("siren") or "") != str(siren)
                    or not siret
                    or siret in seen_sirets
                ):
                    continue
                if not _apply_filters(
                    [candidate],
                    config.drop_unnamed,
                    config.include_closed,
                ):
                    continue
                candidates.append(candidate)
                siren_rank_by_siret[siret] = siren_rank
                seen_sirets.add(siret)
                selected_for_siren += 1
                if selected_for_siren >= config.max_sirets_per_siren:
                    break
    return candidates, siren_rank_by_siret


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


# ---------------------------------------------------------------------------
# TF-IDF artifact building (extracted for caching)
# ---------------------------------------------------------------------------

def _build_tfidf_artifacts(
    candidates: List[dict],
    config: RetrievalConfigV1,
) -> tuple:
    """Build all TF-IDF indexes and return as a cacheable artifact bundle."""
    name_vec, name_mat, names = build_tfidf_index(
        candidates,
        name_mode=config.tfidf_name_mode,
        siren_siblings=config.siren_siblings,
    )
    char_vec, char_mat = (None, None)
    if names:
        char_vec, char_mat = build_char_tfidf_index(names)
    addr_vec, addr_mat = build_address_tfidf_index(candidates)
    return (name_vec, name_mat, names, char_vec, char_mat, addr_vec, addr_mat)


def _get_tfidf_artifacts(
    candidates: List[dict],
    config: RetrievalConfigV1,
    tfidf_cache: Dict[Tuple[str, str], tuple],
    cache_key: Tuple[str, str],
    persistent_cache: Optional["TfidfPersistentCache"] = None,
    partition_key: str = "",
    timer: Optional["PipelineTimer"] = None,
) -> tuple:
    """Get TF-IDF artifacts: in-memory cache -> persistent cache -> build."""
    # 1. In-memory cache (current run, shared within loc_key group)
    cached = tfidf_cache.get(cache_key)
    if cached and len(cached) >= 7:
        return cached

    # 2. Persistent cache (cross-run, on disk)
    if persistent_cache and partition_key:
        disk_cached = persistent_cache.get(partition_key)
        if disk_cached and len(disk_cached) >= 7:
            tfidf_cache[cache_key] = disk_cached
            return disk_cached

    # 3. Build from scratch
    if timer:
        with timer.stage("tfidf_fit"):
            artifacts = _build_tfidf_artifacts(candidates, config)
    else:
        artifacts = _build_tfidf_artifacts(candidates, config)

    # Populate both caches
    tfidf_cache[cache_key] = artifacts
    if persistent_cache and partition_key:
        persistent_cache.put(partition_key, artifacts)
    return artifacts


# ---------------------------------------------------------------------------
# Dense retrieval integration
# ---------------------------------------------------------------------------

def _dense_retrieval_hits(
    crm_name: str,
    candidates: List[dict],
    partition_key: str,
    dense_store: Optional["PartitionEmbeddingStore"],
    top_k: int,
    timer: Optional["PipelineTimer"] = None,
) -> List[Tuple[int, float]]:
    """Return ranked ``(candidate index, score)`` dense hits."""
    if dense_store is None:
        return []
    if not dense_store.has_embeddings(partition_key):
        return []

    try:
        from .dense_retrieval import encode_query
    except ImportError:
        return []

    if timer:
        with timer.stage("dense_encode"):
            query_vec = encode_query(crm_name)
    else:
        query_vec = encode_query(crm_name)

    if query_vec is None:
        return []

    if timer:
        with timer.stage("dense_search"):
            idx = dense_store.get_index(partition_key)
    else:
        idx = dense_store.get_index(partition_key)

    if idx is None:
        return []
    if not dense_store.validates_candidate_order(
        partition_key,
        candidates,
        idx,
    ):
        return []

    scores, indices = idx.search(query_vec, top_k)
    # Filter out -1 padding indices from FAISS
    return [
        (int(i), float(score))
        for i, score in zip(indices, scores, strict=True)
        if i >= 0 and i < len(candidates)
    ]


def _dense_retrieval_indices(
    crm_name: str,
    candidates: List[dict],
    partition_key: str,
    dense_store: Optional["PartitionEmbeddingStore"],
    top_k: int,
    timer: Optional["PipelineTimer"] = None,
) -> List[int]:
    """Backward-compatible index-only dense retrieval helper."""
    return [
        index
        for index, _ in _dense_retrieval_hits(
            crm_name,
            candidates,
            partition_key,
            dense_store,
            top_k,
            timer,
        )
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_candidate_pool(
    store: PartitionedCandidateStore,
    crm_row: dict,
    crm_pre: dict,
    config: RetrievalConfigV1,
    tfidf_cache: Dict[Tuple[str, str], tuple],
    gt_siret: str | None = None,
    *,
    persistent_cache: Optional["TfidfPersistentCache"] = None,
    dense_store: Optional["PartitionEmbeddingStore"] = None,
    timer: Optional["PipelineTimer"] = None,
    siren_global_index: Optional[Any] = None,
    siren_to_geo: Optional[Any] = None,
    dense_siren_index: Optional[Any] = None,
    siren_candidate_store: Optional[Any] = None,
) -> CandidatePoolResult:
    """Build candidate pool with unified code-path for train and inference.

    New optional parameters (backward-compatible):
        persistent_cache: Disk-backed TF-IDF cache (P0a).
        dense_store: Pre-computed FAISS embeddings for hybrid retrieval (P1).
        timer: Pipeline timing instrumentation (P0c).
        siren_global_index: Route B SIREN global index (optional).
        siren_to_geo: Route B SIREN → geo mapping (optional).
    """
    result = CandidatePoolResult(candidates=[])

    insee = crm_row.get("insee") or crm_row.get("crm_insee")
    postcode = crm_row.get("postcode") or crm_row.get("crm_cp")
    crm_name = crm_row.get("crm_name", "")
    crm_address = crm_pre.get("crm_addr", "")
    partition_key: Optional[str] = None

    gt_norm = str(gt_siret).zfill(14) if gt_siret else None

    # === ROUTE B: SIREN-first retrieval (only if global SIREN index present, expansion mode excluded) ===
    if siren_global_index is not None and not config.siren_expansion_enabled:
        crm_name_bag = crm_pre.get("crm_name", "")

        # Phase 1: Query SIREN global index
        top_sirens = siren_global_index.query(crm_name_bag, top_k=config.siren_top_k)

        # Phase 2: Retrieve SIRETs for top SIRENs, geo-aware
        candidates: List[Dict[str, Any]] = []
        seen_sirets: set[str] = set()

        for siren, _ in top_sirens:
            siren_locs = siren_to_geo.get_locations(siren)
            if not siren_locs:
                continue

            # Prioritize locations: INSEE match > postcode match > all others
            sorted_locs = sorted(
                siren_locs,
                key=lambda loc: (loc[0] != insee, loc[1] != postcode)
            )

            loaded_for_siren = 0
            for loc_insee, loc_postcode in sorted_locs:
                if loaded_for_siren >= config.max_sirets_per_siren:
                    break

                try:
                    partition = store.load_by_insee(loc_insee)
                except Exception:
                    continue

                for cand in partition:
                    if (str(cand.get("siren") or "") == siren and
                        str(cand.get("siret") or "") not in seen_sirets):

                        # Apply filters
                        if config.drop_unnamed and not any([
                            cand.get("denomination"),
                            cand.get("denomination_usuelle_ul"),
                            cand.get("enseigne1"),
                            cand.get("enseigne2"),
                            cand.get("enseigne3"),
                            cand.get("denomination_ul"),
                            cand.get("sigle_ul"),
                            cand.get("nom_ul"),
                            cand.get("prenom_usuel_ul"),
                        ]):
                            continue

                        if not config.include_closed and cand.get("etat_admin") == "F":
                            continue

                        candidates.append(cand)
                        seen_sirets.add(str(cand.get("siret") or ""))
                        loaded_for_siren += 1

        # Phase 3: Compute IDF map
        candidates_dict = {
            str(c.get("siret") or ""): c
            for c in candidates
            if c.get("siret")
        }
        idf_map, default_idf = compute_name_idf_map(candidates_dict)

        result.pool_sizes["base"] = len(candidates)
        result.pool_sizes["filtered"] = len(candidates)
        result.pool_sizes["prefilter"] = len(candidates)
        result.pool_sizes["tfidf"] = len(candidates)
        result.candidates = candidates
        result.idf_map = idf_map
        result.default_idf = float(default_idf)
        result.gt_in_base_pool = _check_siret_in_list(candidates, gt_norm)
        result.gt_in_filtered_pool = _check_siret_in_list(candidates, gt_norm)
        result.gt_in_tfidf_pool = _check_siret_in_list(candidates, gt_norm)

        return result

    # === V7: Standard geo-partitioned retrieval (default) ===
    # Step 1: Load base candidates (strict insee_then_postcode + mega policy)
    if timer:
        with timer.stage("partition_load"):
            base_candidates, partition_key = store.load_by_insee_then_postcode_with_key(
                insee,
                postcode,
                mega_insee_max_rows=config.mega_insee_max_rows,
                mega_insee_policy=config.mega_insee_policy,
            )
    else:
        base_candidates, partition_key = store.load_by_insee_then_postcode_with_key(
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

    # Step 4: Prefilter — hybrid sparse (TF-IDF) + dense (FAISS) + rescue
    candidates = pool_list
    prune_threshold = (
        config.retrieval_budget
        if config.fusion_mode == "rrf" and config.retrieval_budget is not None
        else config.prefilter_k
    )
    if prune_threshold and len(candidates) > prune_threshold:
        cache_key = ("main", f"{insee}_{postcode}")

        # 4a. Sparse retrieval (TF-IDF) — with persistent cache
        sparse_idx: List[int] = []
        sparse_scores: Dict[int, float] = {}
        if config.sparse_retrieval_enabled:
            artifacts = _get_tfidf_artifacts(
                candidates, config, tfidf_cache, cache_key,
                persistent_cache=persistent_cache,
                partition_key=partition_key,
                timer=timer,
            )
            name_vec, name_mat, names, char_vec, char_mat, addr_vec, addr_mat = artifacts

            name_idx: List[int] = []
            if name_vec is not None and name_mat is not None:
                if config.fusion_mode == "rrf":
                    name_hits = prefilter_candidates_tfidf_scored(
                        crm_name,
                        name_vec,
                        name_mat,
                        config.prefilter_k,
                        char_vectorizer=char_vec,
                        char_matrix=char_mat,
                    )
                    name_idx = [index for index, _ in name_hits]
                    sparse_scores.update(name_hits)
                elif timer:
                    with timer.stage("tfidf_query"):
                        name_idx = prefilter_candidates_tfidf(
                            crm_name, name_vec, name_mat, config.prefilter_k,
                            cand_names=names, char_top_k=config.char_top_k,
                            char_vectorizer=char_vec, char_matrix=char_mat,
                        )
                else:
                    name_idx = prefilter_candidates_tfidf(
                        crm_name, name_vec, name_mat, config.prefilter_k,
                        cand_names=names, char_top_k=config.char_top_k,
                        char_vectorizer=char_vec, char_matrix=char_mat,
                    )

            addr_idx: List[int] = []
            if addr_vec is not None and addr_mat is not None:
                if config.fusion_mode == "rrf":
                    addr_hits = prefilter_candidates_address_tfidf_scored(
                        crm_address, addr_vec, addr_mat, config.prefilter_k,
                    )
                    addr_idx = [index for index, _ in addr_hits]
                    for index, score in addr_hits:
                        sparse_scores[index] = max(
                            sparse_scores.get(index, 0.0),
                            score,
                        )
                else:
                    addr_idx = prefilter_candidates_address_tfidf(
                        crm_address, addr_vec, addr_mat, config.prefilter_k,
                    )

            if config.fusion_mode == "rrf":
                sparse_idx = [
                    index
                    for index, _ in sorted(
                        sparse_scores.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ]
            else:
                sparse_idx = list(dict.fromkeys(name_idx + addr_idx))

        # 4b. Dense retrieval (FAISS) — if enabled and available
        dense_idx: List[int] = []
        dense_scores: Dict[int, float] = {}
        if (
            config.dense_retrieval_enabled
            and dense_store is not None
            and partition_key is not None
        ):
            dense_hits = _dense_retrieval_hits(
                crm_name, candidates, partition_key,
                dense_store, config.dense_top_k, timer,
            )
            dense_idx = [index for index, _ in dense_hits]
            dense_scores = dict(dense_hits)

        if config.fusion_mode == "rrf":
            index_by_siret = {
                str(candidate.get("siret") or ""): index
                for index, candidate in enumerate(candidates)
                if candidate.get("siret")
            }
            rescue_idx = [
                index_by_siret[siret]
                for siret in sorted(whitelisted_sirets)
                if siret in index_by_siret
            ]
            budget = config.retrieval_budget or config.prefilter_k
            fused_hits = reciprocal_rank_fusion(
                {
                    "sparse": sparse_idx,
                    "dense": dense_idx,
                    "rescue": rescue_idx,
                },
                budget=budget,
                rrf_k=config.rrf_k,
            )
            combined_idx = [int(hit.key) for hit in fused_hits]
            if len(combined_idx) < min(budget, len(candidates)):
                selected = set(combined_idx)
                for index in range(len(candidates)):
                    if index not in selected:
                        combined_idx.append(index)
                    if len(combined_idx) >= min(budget, len(candidates)):
                        break
            fused_by_index = {int(hit.key): hit for hit in fused_hits}
        else:
            # Legacy order-preserving union.
            combined_idx = list(dict.fromkeys(sparse_idx + dense_idx))
            if config.prefilter_union_cap:
                combined_idx = combined_idx[: config.prefilter_union_cap]
            fused_by_index = {}

        if combined_idx:
            prefilter_cands = []
            for retrieval_rank, index in enumerate(combined_idx, start=1):
                if index >= len(candidates):
                    continue
                candidate = candidates[index]
                hit = fused_by_index.get(index)
                if hit is not None:
                    candidate = annotate_fused_candidate(candidate, hit)
                elif config.fusion_mode == "rrf":
                    candidate = dict(candidate)
                    candidate["rrf_score"] = 0.0
                    candidate["retrieval_rank"] = retrieval_rank
                    candidate["retrieval_source"] = "padding"
                    candidate["retrieval_channel_count"] = 0
                    candidate["retrieval_agreement"] = 0
                if index in dense_scores:
                    candidate = dict(candidate)
                    candidate["dense_score"] = dense_scores[index]
                if index in sparse_scores:
                    candidate = dict(candidate)
                    candidate["sparse_score"] = sparse_scores[index]
                prefilter_cands.append(candidate)
            prefilter_sirets = {str(c.get("siret")) for c in prefilter_cands if c.get("siret")}

            # Legacy mode appends rescue candidates. RRF already treats rescue as
            # a channel and therefore preserves its exact candidate budget.
            final_cands = list(prefilter_cands)
            if config.fusion_mode != "rrf":
                for cand in pool.values():
                    siret = str(cand.get("siret") or "")
                    if siret in whitelisted_sirets and siret not in prefilter_sirets:
                        final_cands.append(cand)

            if config.fusion_mode == "rrf":
                candidates = final_cands
            elif len(final_cands) >= config.min_candidates:
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

    result.pool_sizes["prefilter"] = len(candidates)
    # Backward compatibility for existing reports/scripts expecting "tfidf" key
    result.pool_sizes["tfidf"] = len(candidates)
    result.gt_in_tfidf_pool = _check_siret_in_list(candidates, gt_norm)

    if result.gt_in_filtered_pool and not result.gt_in_tfidf_pool and not result.loss_reason:
        result.loss_reason = (
            "PRUNED_BY_RRF"
            if config.fusion_mode == "rrf"
            else ("PRUNED_BY_TFIDF" if config.sparse_retrieval_enabled else "PRUNED_BY_PREFILTER")
        )

    # V9 global dense SIREN rescue. This is a second fixed-budget fusion,
    # not Route B: local SIRET retrieval remains the primary channel.
    if (
        config.global_dense_siren_enabled
        and dense_siren_index is not None
        and siren_to_geo is not None
    ):
        ranked_sirens = dense_siren_index.query(
            crm_pre.get("crm_name", "") or crm_name,
            top_k=config.siren_top_k,
        )
        global_candidates, global_rank_by_siret = _expand_ranked_sirens(
            ranked_sirens=ranked_sirens,
            store=store,
            siren_to_geo=siren_to_geo,
            crm_insee=str(insee or ""),
            crm_postcode=str(postcode or ""),
            config=config,
            siren_candidate_store=siren_candidate_store,
        )
        candidate_by_siret = {
            str(candidate.get("siret") or ""): candidate
            for candidate in global_candidates
            if candidate.get("siret")
        }
        candidate_by_siret.update(
            {
                str(candidate.get("siret") or ""): candidate
                for candidate in candidates
                if candidate.get("siret")
            }
        )
        local_order = [
            str(candidate.get("siret") or "")
            for candidate in candidates
            if candidate.get("siret")
        ]
        global_order = [
            str(candidate.get("siret") or "")
            for candidate in global_candidates
            if candidate.get("siret")
        ]
        budget = config.retrieval_budget or config.prefilter_k
        global_fused = reciprocal_rank_fusion(
            {"local": local_order, "global_siren": global_order},
            budget=budget,
            rrf_k=config.rrf_k,
        )
        fused_candidates: List[dict] = []
        for hit in global_fused:
            siret = str(hit.key)
            candidate = dict(candidate_by_siret[siret])
            candidate["rrf_score"] = float(hit.rrf_score)
            candidate["retrieval_rank"] = int(hit.rank)
            candidate["global_siren_rank"] = global_rank_by_siret.get(siret)
            previous_source = str(candidate.get("retrieval_source") or "")
            sources = set(filter(None, previous_source.split("+")))
            sources.update(hit.channel_ranks)
            candidate["retrieval_source"] = "+".join(sorted(sources))
            candidate["retrieval_channel_count"] = len(sources)
            candidate["retrieval_agreement"] = int(
                "sparse" in sources and "dense" in sources
            )
            fused_candidates.append(candidate)
        candidates = fused_candidates
        result.pool_sizes["global_siren_candidates"] = len(global_candidates)
        result.pool_sizes["global_fused"] = len(candidates)
        result.pool_sizes["prefilter"] = len(candidates)
        result.pool_sizes["tfidf"] = len(candidates)
        result.gt_in_tfidf_pool = _check_siret_in_list(candidates, gt_norm)
        if result.gt_in_tfidf_pool:
            result.loss_reason = ""
        final_candidates = {
            str(candidate.get("siret") or ""): candidate
            for candidate in candidates
            if candidate.get("siret")
        }
        if final_candidates:
            result.idf_map, result.default_idf = compute_name_idf_map(final_candidates)

    if (
        not result.gt_in_base_pool
        and not result.gt_in_tfidf_pool
        and not result.loss_reason
    ):
        result.loss_reason = "NOT_IN_PARTITION"

    # === Step 5: SIREN expansion (local + cross-partition) ===
    if config.siren_expansion_enabled and siren_to_geo is not None and candidates:
        pool_before = len(candidates)
        seed_sirens = sorted({c.get("siren") for c in candidates if c.get("siren")})
        seen_sirets = {c["siret"] for c in candidates if c.get("siret")}
        crm_insee = crm_pre.get("insee") or ""
        crm_cp = crm_pre.get("postcode") or ""

        exp_insee: list[dict] = []
        exp_cp: list[dict] = []
        exp_other: list[dict] = []
        loaded_partitions: dict[str, list] = {}  # cache local per INSEE

        def _load(insee_code: str) -> list:
            if insee_code not in loaded_partitions:
                loaded_partitions[insee_code] = store.load_by_insee(insee_code)
            return loaded_partitions[insee_code]

        for siren in seed_sirens:
            locs = siren_to_geo.get_locations(siren)
            sorted_locs = sorted(locs, key=lambda loc: (loc[0] != crm_insee, loc[1] != crm_cp))
            added_siren = 0

            for insee_code, _ in sorted_locs:
                if added_siren >= config.max_sirets_per_siren:
                    break
                for c in _load(insee_code):
                    siret = c.get("siret")
                    if c.get("siren") != siren or siret in seen_sirets:
                        continue

                    # Apply business filters (consistent with Route B)
                    if config.drop_unnamed and not any([
                        c.get("denomination"),
                        c.get("denomination_usuelle_ul"),
                        c.get("enseigne1"),
                        c.get("enseigne2"),
                        c.get("enseigne3"),
                        c.get("denomination_ul"),
                        c.get("sigle_ul"),
                        c.get("nom_ul"),
                        c.get("prenom_usuel_ul"),
                    ]):
                        continue

                    if not config.include_closed and c.get("etat_admin") == "F":
                        continue

                    if insee_code == crm_insee:
                        exp_insee.append(c)
                    elif c.get("postcode") == crm_cp:
                        exp_cp.append(c)
                    else:
                        exp_other.append(c)
                    seen_sirets.add(siret)
                    added_siren += 1

        def _sort_key(c):
            return (c.get("etat_admin") == "F", not c.get("is_siege", False), c.get("siret", ""))

        expansion = (
            sorted(exp_insee, key=_sort_key)
            + sorted(exp_cp, key=_sort_key)
            + sorted(exp_other, key=_sort_key)
        )

        if expansion:
            candidates = candidates + expansion
            if len(candidates) > config.siren_expansion_pool_cap:
                candidates = candidates[:config.siren_expansion_pool_cap]

            # Recalculer IDF sur pool élargi — retourner dans CandidatePoolResult
            # NE PAS appeler set_global_name_idf_map() ici (le caller le fait)
            tmp = {c["siret"]: c for c in candidates if c.get("siret")}
            result.idf_map, result.default_idf = compute_name_idf_map(tmp)

        # Telémétrie
        result.pool_sizes.update({
            "prefilter_before_expansion": pool_before,
            "expansion_added_insee": len(exp_insee),
            "expansion_added_postcode": len(exp_cp),
            "expansion_added_cross": len(exp_other),
            "expanded_pool_final": len(candidates),
        })

        # Recalculate GT metrics after expansion
        result.gt_in_tfidf_pool = _check_siret_in_list(candidates, gt_norm)
        if result.gt_in_filtered_pool and not result.gt_in_tfidf_pool and result.loss_reason == "PRUNED_BY_TFIDF":
            # GT was recovered by expansion, clear loss reason
            result.loss_reason = None
    elif config.siren_expansion_enabled and siren_to_geo is None:
        logger = logging.getLogger(__name__)
        logger.warning("siren_expansion_enabled=True but siren_to_geo not loaded — skipping expansion")

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
