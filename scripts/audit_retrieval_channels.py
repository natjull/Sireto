#!/usr/bin/env python3
"""Audit sparse retrieval channels independently on a frozen benchmark split."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import (  # noqa: E402
    _binary_metric,
    _git_commit,
    load_benchmark,
    retrieval_config,
)
from src.xgb_matcher.blocking import (  # noqa: E402
    address_hash,
    build_address_hash_index,
    build_numeric_token_index,
    dedupe_candidates,
    extract_numeric_tokens,
    prefilter_candidates_address_tfidf_scored,
    prefilter_candidates_char_tfidf_scored,
    prefilter_candidates_tfidf_scored,
    prefilter_candidates_word_tfidf_scored,
)
from src.xgb_matcher.features import preprocess_crm_row  # noqa: E402
from src.xgb_matcher.fusion import reciprocal_rank_fusion  # noqa: E402
from src.xgb_matcher.naming import build_candidate_names, normalize_name  # noqa: E402
from src.xgb_matcher.partitioned_store import (  # noqa: E402
    PartitionedCandidateStore,
)
from src.xgb_matcher.retrieval import (  # noqa: E402
    _apply_filters,
    _get_tfidf_artifacts,
)
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-retrieval-channel-audit-1"
RANKED_CHANNELS = (
    "name_word",
    "name_char",
    "address_word",
    "siren_head",
    "siren_sites",
    "current_sparse",
)
EXACT_CHANNELS = ("name_exact", "address_exact", "numeric_name")
ALL_CHANNELS = RANKED_CHANNELS + EXACT_CHANNELS


def _normalize_siret(candidate: dict) -> str:
    value = str(candidate.get("siret") or "")
    return value.zfill(14) if value else ""


def _indices_to_sirets(
    candidates: list[dict],
    indices: Iterable[int],
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for index in indices:
        if index < 0 or index >= len(candidates):
            continue
        siret = _normalize_siret(candidates[index])
        if siret and siret not in seen:
            seen.add(siret)
            output.append(siret)
    return output


def _rank(values: list[str], target: str) -> int | None:
    try:
        return values.index(target) + 1
    except ValueError:
        return None


def _unique_sirens(sirets: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for siret in sirets:
        siren = siret[:9]
        if siren and siren not in seen:
            seen.add(siren)
            output.append(siren)
    return output


def _current_sparse_indices(
    *,
    candidates: list[dict],
    crm_name: str,
    crm_address: str,
    name_vectorizer: Any,
    name_matrix: Any,
    char_vectorizer: Any,
    char_matrix: Any,
    address_vectorizer: Any,
    address_matrix: Any,
    rescue_indices: list[int],
    per_channel_k: int,
    budget: int,
    rrf_k: int,
    prefilter_trigger_size: int,
) -> list[int]:
    """Reproduce the current sparse+rescue ordering without calling the core."""
    if len(candidates) <= prefilter_trigger_size:
        return list(range(min(budget, len(candidates))))

    sparse_scores: dict[int, float] = {}
    if name_vectorizer is not None and name_matrix is not None:
        name_hits = prefilter_candidates_tfidf_scored(
            crm_name,
            name_vectorizer,
            name_matrix,
            per_channel_k,
            char_vectorizer=char_vectorizer,
            char_matrix=char_matrix,
        )
        sparse_scores.update(name_hits)
    if address_vectorizer is not None and address_matrix is not None:
        address_hits = prefilter_candidates_address_tfidf_scored(
            crm_address,
            address_vectorizer,
            address_matrix,
            per_channel_k,
        )
        for index, score in address_hits:
            sparse_scores[index] = max(sparse_scores.get(index, 0.0), score)

    sparse_indices = [
        index
        for index, _score in sorted(
            sparse_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    fused = reciprocal_rank_fusion(
        {
            "sparse": sparse_indices,
            "rescue": rescue_indices,
        },
        budget=budget,
        rrf_k=rrf_k,
    )
    combined = [int(hit.key) for hit in fused]
    if len(combined) < min(budget, len(candidates)):
        selected = set(combined)
        for index in range(len(candidates)):
            if index not in selected:
                combined.append(index)
            if len(combined) >= min(budget, len(candidates)):
                break
    return combined


def _exact_indexes(candidates: list[dict]) -> dict[str, Any]:
    name_index: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        seen_names: set[str] = set()
        for candidate_name in build_candidate_names(candidate):
            if candidate_name.text and candidate_name.text not in seen_names:
                name_index[candidate_name.text].append(index)
                seen_names.add(candidate_name.text)
    return {
        "name": dict(name_index),
        "address": build_address_hash_index(candidates),
        "numeric": build_numeric_token_index(candidates),
    }


def _ordered_union(index_groups: Iterable[Iterable[int]]) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for group in index_groups:
        for index in group:
            if index not in seen:
                seen.add(index)
                output.append(index)
    return output


def _unique_sirens_from_indices(
    candidates: list[dict],
    indices: Iterable[int],
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for index in indices:
        if index < 0 or index >= len(candidates):
            continue
        siren = str(candidates[index].get("siren") or "")
        if siren and siren not in seen:
            seen.add(siren)
            output.append(siren)
    return output


def _siren_sibling_channels(
    *,
    candidates: list[dict],
    word_indices: list[int],
    char_indices: list[int],
    current_indices: list[int],
    address_indices: list[int],
    address_exact_indices: list[int],
    max_output: int,
    max_sites_per_siren: int = 20,
) -> tuple[list[int], list[int]]:
    """Transfer entity-name evidence to local SIRET siblings.

    ``siren_head`` emits the best local site for each ranked SIREN. The
    round-robin ``siren_sites`` list then exposes additional sites without
    letting one large public entity consume the whole candidate budget.
    """
    siren_lists = {
        "word": _unique_sirens_from_indices(candidates, word_indices),
        "char": _unique_sirens_from_indices(candidates, char_indices),
        "current": _unique_sirens_from_indices(candidates, current_indices),
    }
    ranked_sirens = [
        str(hit.key)
        for hit in reciprocal_rank_fusion(
            siren_lists,
            budget=min(max_output, len(candidates)),
            rrf_k=60,
        )
    ]
    if not ranked_sirens:
        return [], []

    siblings: dict[str, list[int]] = defaultdict(list)
    ranked_siren_set = set(ranked_sirens)
    for index, candidate in enumerate(candidates):
        siren = str(candidate.get("siren") or "")
        if siren in ranked_siren_set:
            siblings[siren].append(index)

    address_rank = {
        index: rank
        for rank, index in enumerate(address_indices, start=1)
    }
    current_rank = {
        index: rank
        for rank, index in enumerate(current_indices, start=1)
    }
    exact_rank = {
        index: rank
        for rank, index in enumerate(address_exact_indices, start=1)
    }

    def site_key(index: int) -> tuple[Any, ...]:
        candidate = candidates[index]
        return (
            exact_rank.get(index, 10**9),
            not bool(candidate.get("is_siege")),
            address_rank.get(index, 10**9),
            current_rank.get(index, 10**9),
            candidate.get("etat_admin") == "F",
            _normalize_siret(candidate),
        )

    ranked_sites: list[list[int]] = []
    for siren in ranked_sirens:
        ordered = sorted(siblings.get(siren, []), key=site_key)
        if ordered:
            ranked_sites.append(ordered[:max_sites_per_siren])
    heads = [sites[0] for sites in ranked_sites[:max_output]]
    round_robin: list[int] = []
    for site_position in range(max_sites_per_siren):
        for sites in ranked_sites:
            if site_position < len(sites):
                round_robin.append(sites[site_position])
                if len(round_robin) >= max_output:
                    return heads, round_robin
    return heads, round_robin


def _segment_masks(raw: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=raw.index),
        "gt_active": raw["ground_truth_state"].fillna("").eq("A"),
        "gt_closed": raw["ground_truth_state"].fillna("").eq("F"),
        "mega_base_pool": raw["mega_base_pool"].astype(bool),
        "multi_site_siren": raw["multi_site_siren"].astype(bool),
        "location_match_type=cp_only": (
            raw["location_match_type"].astype(str).eq("cp_only")
        ),
        "location_match_type=insee": (
            raw["location_match_type"].astype(str).eq("insee")
        ),
    }


def summarize_channel_audit(
    raw: pd.DataFrame,
    *,
    cutoffs: list[int],
) -> dict[str, Any]:
    """Summarize individual channels and diagnostic channel oracles."""
    masks = _segment_masks(raw)
    channels: dict[str, Any] = {}
    for channel in ALL_CHANNELS:
        rank_col = f"{channel}_rank"
        channel_summary: dict[str, Any] = {"recall": {}, "segments_at_100": {}}
        for cutoff in cutoffs:
            hit = raw[rank_col].notna() & raw[rank_col].le(cutoff)
            channel_summary["recall"][str(cutoff)] = _binary_metric(hit)
        hit_100 = raw[rank_col].notna() & raw[rank_col].le(100)
        for segment, mask in masks.items():
            if mask.sum():
                channel_summary["segments_at_100"][segment] = _binary_metric(
                    hit_100[mask]
                )
        channel_summary["candidate_count"] = {
            "mean": float(raw[f"{channel}_count"].mean()),
            "p95": float(raw[f"{channel}_count"].quantile(0.95)),
            "max": int(raw[f"{channel}_count"].max()),
        }
        channels[channel] = channel_summary

    oracle: dict[str, Any] = {}
    current_hit_100 = (
        raw["current_sparse_rank"].notna()
        & raw["current_sparse_rank"].le(100)
    )
    for cutoff in cutoffs:
        hit_columns = [
            raw[f"{channel}_rank"].notna()
            & raw[f"{channel}_rank"].le(cutoff)
            for channel in ALL_CHANNELS
            if channel != "current_sparse"
        ]
        any_hit = pd.concat(hit_columns, axis=1).any(axis=1)
        oracle[str(cutoff)] = {
            "any_individual_channel": _binary_metric(any_hit),
            "additional_vs_current_sparse_at_100": int(
                (any_hit & ~current_hit_100).sum()
            ),
        }

    recoveries = {}
    for channel in ALL_CHANNELS:
        if channel == "current_sparse":
            continue
        hit = raw[f"{channel}_rank"].notna() & raw[f"{channel}_rank"].le(100)
        recoveries[channel] = {
            "recovers_current_sparse_misses": int(
                (hit & ~current_hit_100).sum()
            ),
            "misses_current_sparse_hits": int(
                (~hit & current_hit_100).sum()
            ),
            "both_hit": int((hit & current_hit_100).sum()),
        }

    return {
        "query_count": int(len(raw)),
        "channels": channels,
        "diagnostic_oracle": oracle,
        "paired_at_100": recoveries,
        "geography": {
            "base_pool_siret": _binary_metric(raw["ground_truth_in_base"]),
            "base_pool_siren": _binary_metric(raw["ground_truth_siren_in_base"]),
        },
        "siren": {
            "current_sparse_siren_recall_at_100": _binary_metric(
                raw["current_sparse_siren_rank"].notna()
                & raw["current_sparse_siren_rank"].le(100)
            ),
            "correct_siren_but_wrong_siret_at_100": int(
                (
                    raw["current_sparse_siren_rank"].notna()
                    & raw["current_sparse_siren_rank"].le(100)
                    & ~current_hit_100
                ).sum()
            ),
        },
        "latency_ms": {
            "p50": float(raw["latency_ms"].quantile(0.50)),
            "p95": float(raw["latency_ms"].quantile(0.95)),
            "p99": float(raw["latency_ms"].quantile(0.99)),
            "mean": float(raw["latency_ms"].mean()),
        },
    }


def _markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    rows = [
        "# Audit unitaire des canaux de retrieval",
        "",
        f"- Benchmark : `{manifest['benchmark_build_id']}` / `{manifest['split']}`",
        f"- Requêtes : {manifest['query_count']}",
        f"- Commit : `{manifest['git_commit'][:12]}`",
        (
            "- Le canal `current_sparse` est vérifié candidat par candidat "
            "contre l’artefact baseline gelé."
            if manifest["verification"]["baseline_verified"]
            else "- Audit autonome de la source : aucune baseline externe "
            "n’était attendue."
        ),
        "",
        "## Recall SIRET par canal",
        "",
        "| Canal | @50 | @100 | @200 | @500 |",
        "|---|---:|---:|---:|---:|",
    ]
    for channel, values in summary["channels"].items():
        cells = []
        for cutoff in (50, 100, 200, 500):
            metric = values["recall"].get(str(cutoff))
            cells.append(
                f"{metric['successes']}/{metric['total']} = {metric['rate']:.2%}"
                if metric
                else "n/a"
            )
        rows.append(f"| {channel} | " + " | ".join(cells) + " |")
    rows.extend(
        [
            "",
            "## Complémentarité à 100",
            "",
            "| Canal | Misses sparse récupérées | Hits sparse perdus | Hits communs |",
            "|---|---:|---:|---:|",
        ]
    )
    for channel, values in summary["paired_at_100"].items():
        rows.append(
            f"| {channel} | "
            f"{values['recovers_current_sparse_misses']} | "
            f"{values['misses_current_sparse_hits']} | "
            f"{values['both_hit']} |"
        )
    geography = summary["geography"]["base_pool_siret"]
    siren = summary["siren"]
    rows.extend(
        [
            "",
            "## Plafonds diagnostiques",
            "",
            f"- Vérité SIRET dans le pool géographique : "
            f"{geography['successes']}/{geography['total']} = "
            f"{geography['rate']:.2%}.",
            f"- Bon SIREN dans les 100 premiers sparse : "
            f"{siren['current_sparse_siren_recall_at_100']['rate']:.2%}.",
            f"- Bon SIREN mais mauvais SIRET à 100 : "
            f"{siren['correct_siren_but_wrong_siret_at_100']} requêtes.",
            "",
            "L’oracle `any_individual_channel` mesure seulement si au moins un "
            "canal voit la vérité à son propre cutoff. Ce n’est pas une union "
            "éligible : sa cardinalité cumulée peut dépasser 100.",
            "",
        ]
    )
    return "\n".join(rows)


def _load_exact_cache(
    cache: OrderedDict[str, dict[str, Any]],
    *,
    partition_key: str,
    candidates: list[dict],
) -> dict[str, Any]:
    cached = cache.get(partition_key)
    if cached is not None:
        cache.move_to_end(partition_key)
        return cached
    indexes = _exact_indexes(candidates)
    cache[partition_key] = indexes
    while len(cache) > 5:
        cache.popitem(last=False)
    return indexes


def run_audit(
    *,
    benchmark: pd.DataFrame,
    partitions_dir: Path,
    cache_root: Path,
    baseline_raw: Path | None,
    per_channel_k: int,
    cutoffs: list[int],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    max_k = max(cutoffs)
    config = retrieval_config(
        "sparse",
        per_channel_k=per_channel_k,
        budget=max_k,
        prefilter_trigger_size=min(cutoffs),
    )
    legacy_signatures = [config.signature().hash, config.legacy_signature_hash()]
    config_without_trigger = replace(config, prefilter_trigger_size=None)
    legacy_signatures.extend(
        [
            config_without_trigger.signature().hash,
            config_without_trigger.legacy_signature_hash(),
        ]
    )
    persistent_cache = TfidfPersistentCache(
        config.tfidf_artifact_hash(),
        cache_dir=cache_root,
        fallback_config_hashes=legacy_signatures,
    )
    store = PartitionedCandidateStore(partitions_dir)
    tfidf_memory: OrderedDict[tuple[str, str], tuple] = OrderedDict()
    exact_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
    baseline_by_query: dict[str, list[str]] | None = None
    if baseline_raw is not None:
        baseline = pd.read_parquet(
            baseline_raw,
            columns=["query_id", "candidate_sirets_json"],
        )
        benchmark_query_ids = set(benchmark["query_id"].astype(str))
        baseline = baseline[
            baseline["query_id"].astype(str).isin(benchmark_query_ids)
        ].copy()
        baseline_by_query = dict(
            zip(
                baseline["query_id"].astype(str),
                baseline["candidate_sirets_json"].map(json.loads),
                strict=True,
            )
        )
        if set(baseline_by_query) != benchmark_query_ids:
            raise ValueError(
                "Baseline and benchmark query IDs are not identical"
            )

    siren_siret_counts = (
        benchmark.groupby("ground_truth_siren")["ground_truth_siret"]
        .nunique()
        .to_dict()
    )
    processing = benchmark.copy()
    processing["_original_order"] = np.arange(len(processing))
    processing["_insee_sort"] = processing["insee"].fillna("").astype(str)
    processing["_postcode_sort"] = processing["postcode"].fillna("").astype(str)
    processing = processing.sort_values(
        ["_insee_sort", "_postcode_sort", "_original_order"],
        kind="stable",
    )

    records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    started = time.perf_counter()
    for processed_count, (_index, row) in enumerate(
        processing.iterrows(),
        start=1,
    ):
        query_started = time.perf_counter()
        crm_row = {
            "query_id": str(row["query_id"]),
            "crm_id": str(row["query_id"]),
            "crm_name": row.get("crm_name") or "",
            "crm_address": row.get("crm_address") or "",
            "crm_city": row.get("crm_city") or "",
            "crm_city_addr": row.get("crm_city") or "",
            "postcode": row.get("postcode") or "",
            "insee": row.get("insee") or "",
        }
        crm_pre = preprocess_crm_row(crm_row)
        base_candidates, partition_key = (
            store.load_by_insee_then_postcode_with_key(
                crm_row["insee"],
                crm_row["postcode"],
                mega_insee_max_rows=config.mega_insee_max_rows,
                mega_insee_policy=config.mega_insee_policy,
            )
        )
        filtered = _apply_filters(
            base_candidates,
            config.drop_unnamed,
            config.include_closed,
        )
        candidates = list(dedupe_candidates(filtered).values())
        ground_truth_siret = str(row["ground_truth_siret"])
        ground_truth_siren = str(row["ground_truth_siren"])
        base_sirets = {_normalize_siret(candidate) for candidate in candidates}
        base_sirens = {siret[:9] for siret in base_sirets if siret}

        channel_sirets: dict[str, list[str]] = {
            channel: [] for channel in ALL_CHANNELS
        }
        if candidates and partition_key:
            cache_key = (
                "main",
                f"{crm_row['insee']}_{crm_row['postcode']}",
            )
            artifacts = _get_tfidf_artifacts(
                candidates,
                config,
                tfidf_memory,
                cache_key,
                persistent_cache=persistent_cache,
                partition_key=partition_key,
            )
            (
                name_vec,
                name_mat,
                _names,
                char_vec,
                char_mat,
                address_vec,
                address_mat,
            ) = artifacts
            word_hits = (
                prefilter_candidates_word_tfidf_scored(
                    crm_row["crm_name"],
                    name_vec,
                    name_mat,
                    per_channel_k,
                )
                if name_vec is not None and name_mat is not None
                else []
            )
            char_hits = (
                prefilter_candidates_char_tfidf_scored(
                    crm_row["crm_name"],
                    char_vec,
                    char_mat,
                    per_channel_k,
                )
                if char_vec is not None and char_mat is not None
                else []
            )
            address_hits = (
                prefilter_candidates_address_tfidf_scored(
                    crm_pre["crm_addr"],
                    address_vec,
                    address_mat,
                    per_channel_k,
                )
                if address_vec is not None and address_mat is not None
                else []
            )
            exact_indexes = _load_exact_cache(
                exact_cache,
                partition_key=partition_key,
                candidates=candidates,
            )
            # Audit the literal normalized CRM key. Location-stripped names are
            # a distinct heuristic and must not be conflated with exactness.
            name_key = normalize_name(crm_row["crm_name"])
            name_exact_indices = exact_indexes["name"].get(name_key, [])
            address_key = address_hash(
                crm_pre.get("crm_street_num"),
                crm_pre.get("crm_street_name"),
            )
            address_exact_indices = (
                exact_indexes["address"].get(address_key, [])
                if address_key
                else []
            )
            numeric_tokens = sorted(
                extract_numeric_tokens(crm_row["crm_name"])
            )
            numeric_indices = _ordered_union(
                exact_indexes["numeric"].get(token, [])
                for token in numeric_tokens
            )
            whitelisted_sirets = {
                _normalize_siret(candidates[index])
                for index in address_exact_indices + numeric_indices
                if 0 <= index < len(candidates)
            }
            index_by_siret = {
                _normalize_siret(candidate): index
                for index, candidate in enumerate(candidates)
                if _normalize_siret(candidate)
            }
            rescue_indices = [
                index_by_siret[siret]
                for siret in sorted(whitelisted_sirets)
                if siret in index_by_siret
            ]
            current_indices = _current_sparse_indices(
                candidates=candidates,
                crm_name=crm_row["crm_name"],
                crm_address=crm_pre["crm_addr"],
                name_vectorizer=name_vec,
                name_matrix=name_mat,
                char_vectorizer=char_vec,
                char_matrix=char_mat,
                address_vectorizer=address_vec,
                address_matrix=address_mat,
                rescue_indices=rescue_indices,
                per_channel_k=per_channel_k,
                budget=max_k,
                rrf_k=config.rrf_k,
                prefilter_trigger_size=min(cutoffs),
            )
            word_indices = [index for index, _score in word_hits]
            char_indices = [index for index, _score in char_hits]
            address_indices = [index for index, _score in address_hits]
            siren_head_indices, siren_site_indices = _siren_sibling_channels(
                candidates=candidates,
                word_indices=word_indices,
                char_indices=char_indices,
                current_indices=current_indices,
                address_indices=address_indices,
                address_exact_indices=address_exact_indices,
                max_output=max_k,
            )
            channel_sirets = {
                "name_word": _indices_to_sirets(
                    candidates, word_indices
                ),
                "name_char": _indices_to_sirets(
                    candidates, char_indices
                ),
                "address_word": _indices_to_sirets(
                    candidates, address_indices
                ),
                "siren_head": _indices_to_sirets(
                    candidates, siren_head_indices
                ),
                "siren_sites": _indices_to_sirets(
                    candidates, siren_site_indices
                ),
                "name_exact": _indices_to_sirets(
                    candidates, name_exact_indices
                ),
                "address_exact": _indices_to_sirets(
                    candidates, address_exact_indices
                ),
                "numeric_name": _indices_to_sirets(
                    candidates, numeric_indices
                ),
                "current_sparse": _indices_to_sirets(
                    candidates, current_indices
                ),
            }

        if baseline_by_query is not None:
            baseline_values = baseline_by_query[str(row["query_id"])]
            if channel_sirets["current_sparse"] != baseline_values:
                mismatches.append(str(row["query_id"]))
                if len(mismatches) <= 3:
                    print(
                        f"[mismatch] query={row['query_id']} "
                        f"audit={channel_sirets['current_sparse'][:5]} "
                        f"baseline={baseline_values[:5]}",
                        flush=True,
                    )

        current_sirens = _unique_sirens(channel_sirets["current_sparse"])
        record: dict[str, Any] = {
            "_original_order": int(row["_original_order"]),
            "query_id": str(row["query_id"]),
            "ground_truth_siret": ground_truth_siret,
            "ground_truth_siren": ground_truth_siren,
            "ground_truth_state": row.get("ground_truth_state"),
            "location_match_type": row.get("location_match_type"),
            "multi_site_siren": siren_siret_counts.get(
                ground_truth_siren, 0
            ) > 1,
            "base_pool_size": len(candidates),
            "mega_base_pool": len(candidates) > 100_000,
            "partition_key": partition_key or "",
            "ground_truth_in_base": ground_truth_siret in base_sirets,
            "ground_truth_siren_in_base": ground_truth_siren in base_sirens,
            "current_sparse_siren_rank": _rank(
                current_sirens, ground_truth_siren
            ),
            "latency_ms": (time.perf_counter() - query_started) * 1000,
        }
        for channel, sirets in channel_sirets.items():
            record[f"{channel}_rank"] = _rank(sirets, ground_truth_siret)
            record[f"{channel}_count"] = len(sirets)
            record[f"{channel}_sirets_json"] = json.dumps(
                sirets,
                separators=(",", ":"),
            )
        records.append(record)
        while len(tfidf_memory) > 20:
            tfidf_memory.popitem(last=False)
        if processed_count % 250 == 0:
            elapsed = time.perf_counter() - started
            current_hits = sum(
                item["current_sparse_rank"] is not None
                and item["current_sparse_rank"] <= 100
                for item in records
            )
            print(
                f"[channels] {processed_count}/{len(processing)} "
                f"recall100={current_hits / len(records):.4f} "
                f"mismatches={len(mismatches)} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    if mismatches:
        raise RuntimeError(
            "Current sparse reproduction diverges from frozen baseline for "
            f"{len(mismatches)} queries; first={mismatches[:10]}"
        )
    raw = (
        pd.DataFrame(records)
        .sort_values("_original_order", kind="stable")
        .drop(columns=["_original_order"])
        .reset_index(drop=True)
    )
    summary = summarize_channel_audit(raw, cutoffs=cutoffs)
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["tfidf_cache"] = persistent_cache.stats()
    return raw, summary, {
        "baseline_verified": baseline_by_query is not None,
        "current_sparse_mismatches": len(mismatches),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--partitions-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-raw",
        type=Path,
        default=None,
        help="Optional frozen raw result used for exact list reproduction.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=[50, 100, 200, 500],
    )
    parser.add_argument("--per-channel-k", type=int, default=500)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    cutoffs = sorted(set(args.cutoffs))
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("Cutoffs must be positive")
    if cutoffs[-1] > args.per_channel_k:
        raise ValueError("Largest cutoff cannot exceed per-channel-k")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Immutable output directory exists: {args.output_dir}"
        )
    benchmark_manifest = json.loads(
        args.benchmark_manifest.read_text(encoding="utf-8")
    )
    benchmark_hash = file_sha256(args.benchmark)
    expected_hash = benchmark_manifest.get("output_sha256", {}).get(
        args.benchmark.name
    )
    if benchmark_hash != expected_hash:
        raise ValueError("Benchmark hash does not match frozen manifest")
    benchmark = load_benchmark(args.benchmark, args.split)
    if args.max_rows:
        benchmark = benchmark.head(args.max_rows).copy()

    raw, summary, verification = run_audit(
        benchmark=benchmark,
        partitions_dir=args.partitions_dir,
        cache_root=args.cache_dir,
        baseline_raw=args.baseline_raw,
        per_channel_k=args.per_channel_k,
        cutoffs=cutoffs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = args.output_dir / "raw_results.parquet"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    raw.to_parquet(raw_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "benchmark_build_id": benchmark_manifest["build_id"],
        "benchmark_manifest_sha256": file_sha256(args.benchmark_manifest),
        "benchmark_sha256": benchmark_hash,
        "partitions_sha256": benchmark_manifest["partitions_sha256"],
        "baseline_raw_sha256": (
            file_sha256(args.baseline_raw)
            if args.baseline_raw is not None
            else None
        ),
        "split": args.split,
        "query_count": int(len(benchmark)),
        "cutoffs": cutoffs,
        "per_channel_k": args.per_channel_k,
        "verification": verification,
    }
    report_path.write_text(_markdown(summary, manifest), encoding="utf-8")
    manifest["outputs"] = {
        "raw_results.parquet": file_sha256(raw_path),
        "summary.json": file_sha256(summary_path),
        "report.md": file_sha256(report_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
