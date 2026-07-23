#!/usr/bin/env python3
"""Benchmark retrieval recall: sparse-only vs dense-only vs hybrid.

Runs retrieval on all GT rows and reports recall@K for each mode.
Helps decide if TF-IDF can be phased out in favor of dense-only.

Usage:
    python scripts/benchmark_retrieval.py \
        --crm-gt data/crm_ok_gt.csv \
        --partitions-dir data/candidates_v7_all \
        --dense-dir data/dense_index \
        --prefilter-k 500

Output: recall table printed to stdout + saved as CSV.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.features import preprocess_crm_row
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache
from src.xgb_matcher.timing import PipelineTimer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _load_gt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    renames = {
        "CODE_POSTAL": "crm_cp", "CODE_INSEE": "crm_insee",
        "Client final": "crm_name", "Adresse": "crm_address",
        "Ville": "crm_city", "ground_truth_siret": "gt_siret",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    if "gt_siret" not in df.columns:
        for col in df.columns:
            if "siret" in col.lower() and "gt" in col.lower():
                df = df.rename(columns={col: "gt_siret"})
                break
    return df


def run_benchmark(
    df: pd.DataFrame,
    partitions_dir: Path,
    dense_dir: Path | None,
    prefilter_k: int,
    candidate_budget: int | None = None,
    global_siren_dense_dir: Path | None = None,
    siren_geo_index: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    """Run retrieval in 3 modes and compute recall@K."""
    budget = candidate_budget or prefilter_k
    common = {
        "prefilter_k": prefilter_k,
        "fusion_mode": "rrf",
        "retrieval_budget": budget,
        "prefilter_union_cap": None,
        "min_candidates": min(50, budget),
    }

    modes = {
        "sparse_only": RetrievalConfigV1(
            **common,
            dense_retrieval_enabled=False,
        ),
        "hybrid": RetrievalConfigV1(
            **common,
            dense_retrieval_enabled=True,
            dense_top_k=prefilter_k,
        ),
    }

    # Dense-only: disable sparse branch entirely and keep only dense ANN retrieval.
    modes["dense_only"] = RetrievalConfigV1(
        **common,
        sparse_retrieval_enabled=False,
        dense_retrieval_enabled=True,
        dense_top_k=prefilter_k,
    )

    dense_store = None
    if dense_dir and dense_dir.exists():
        from src.xgb_matcher.dense_retrieval import PartitionEmbeddingStore
        dense_store = PartitionEmbeddingStore(dense_dir)
        _logger.info("Dense store loaded from %s", dense_dir)
    else:
        _logger.warning("No dense store at %s — running sparse_only only to avoid misleading dense metrics", dense_dir)
        modes = {"sparse_only": modes["sparse_only"]}

    dense_siren_index = None
    siren_to_geo = None
    if global_siren_dense_dir and siren_geo_index:
        from src.xgb_matcher.dense_retrieval import GlobalDenseSirenIndex
        from src.xgb_matcher.siren_retrieval import SirenToGeoIndex

        dense_siren_index = GlobalDenseSirenIndex(global_siren_dense_dir)
        siren_to_geo = SirenToGeoIndex(siren_geo_index)
        modes["sparse_local_dense_global_siren"] = RetrievalConfigV1(
            **common,
            dense_retrieval_enabled=False,
            global_dense_siren_enabled=True,
        )

    results: Dict[str, Dict[str, float]] = {}

    for mode_name, config in modes.items():
        _logger.info("=== Running mode: %s ===", mode_name)
        store = PartitionedCandidateStore(partitions_dir)
        cache = TfidfPersistentCache(config.signature().hash)
        timer = PipelineTimer()

        total = 0
        gt_in_pool = 0
        gt_in_base = 0
        loss_reasons: Counter = Counter()
        query_latencies: List[float] = []

        t0 = time.perf_counter()

        for _, row in df.iterrows():
            gt_siret = row.get("gt_siret") or row.get("ground_truth_siret")
            if not gt_siret or pd.isna(gt_siret):
                continue

            crm_row = {
                "crm_name": row.get("crm_name", ""),
                "crm_address": row.get("crm_address", ""),
                "crm_city": row.get("crm_city", ""),
                "crm_city_addr": row.get("crm_city_addr", row.get("crm_city", "")),
                "postcode": row.get("crm_cp", ""),
                "insee": row.get("crm_insee", ""),
                "crm_id": row.get("crm_id", ""),
            }
            crm_pre = preprocess_crm_row(crm_row)

            tfidf_cache: Dict = {}
            query_started = time.perf_counter()
            res = build_candidate_pool(
                store, crm_row, crm_pre, config, tfidf_cache,
                gt_siret=str(gt_siret),
                persistent_cache=cache,
                dense_store=dense_store if config.dense_retrieval_enabled else None,
                dense_siren_index=(
                    dense_siren_index if config.global_dense_siren_enabled else None
                ),
                siren_to_geo=(
                    siren_to_geo if config.global_dense_siren_enabled else None
                ),
                timer=timer,
            )
            query_latencies.append(time.perf_counter() - query_started)

            total += 1
            if res.gt_in_base_pool:
                gt_in_base += 1
            if res.gt_in_tfidf_pool:
                gt_in_pool += 1
            if res.loss_reason:
                loss_reasons[res.loss_reason] += 1

        elapsed = time.perf_counter() - t0

        recall_base = gt_in_base / total * 100 if total else 0
        recall_prefilter = gt_in_pool / total * 100 if total else 0
        recall_prefilter_given_base = gt_in_pool / gt_in_base * 100 if gt_in_base else 0

        results[mode_name] = {
            "total": total,
            "gt_in_base": gt_in_base,
            "gt_in_prefilter": gt_in_pool,
            "recall@base_%": recall_base,
            f"recall@{budget}_%": recall_prefilter,
            f"recall@{budget}_given_base_%": recall_prefilter_given_base,
            "elapsed_s": elapsed,
            "queries_per_sec": total / max(elapsed, 0.001),
            "latency_p50_ms": (
                float(pd.Series(query_latencies).quantile(0.50) * 1000)
                if query_latencies else 0.0
            ),
            "latency_p95_ms": (
                float(pd.Series(query_latencies).quantile(0.95) * 1000)
                if query_latencies else 0.0
            ),
        }

        _logger.info(
            "[%s] recall@base=%.2f%%  recall@%d=%.2f%%  (given base: %.2f%%)  %.1fs (%.1f q/s)",
            mode_name, recall_base, budget, recall_prefilter,
            recall_prefilter_given_base, elapsed, total / max(elapsed, 0.001),
        )
        for reason, count in loss_reasons.most_common():
            _logger.info("  loss: %-25s  %d (%.1f%%)", reason, count, count / total * 100)

        timer.log_summary(prefix=mode_name)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval recall ablation.")
    parser.add_argument("--crm-gt", type=Path, default=Path("data/crm_ok_gt.csv"))
    parser.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v7_all"))
    parser.add_argument("--dense-dir", type=Path, default=Path("data/dense_index"))
    parser.add_argument("--global-siren-dense-dir", type=Path)
    parser.add_argument("--siren-geo-index", type=Path)
    parser.add_argument("--prefilter-k", type=int, default=500)
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=50,
        help="Exact post-fusion candidate budget (default: 50).",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for quick test (0=all)")
    args = parser.parse_args()

    df = _load_gt(args.crm_gt)
    _logger.info("Loaded %d GT rows", len(df))

    if args.max_rows > 0:
        df = df.head(args.max_rows)
        _logger.info("Limited to %d rows", len(df))

    results = run_benchmark(
        df,
        args.partitions_dir,
        args.dense_dir,
        args.prefilter_k,
        args.candidate_budget,
        args.global_siren_dense_dir,
        args.siren_geo_index,
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("RETRIEVAL RECALL BENCHMARK")
    print("=" * 80)
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())

    if args.output_csv:
        summary_df.to_csv(args.output_csv)
        _logger.info("Saved to %s", args.output_csv)


if __name__ == "__main__":
    main()
