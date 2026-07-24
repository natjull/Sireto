#!/usr/bin/env python3
"""Run immutable fixed-budget V9 retrieval experiments with raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.features import preprocess_crm_row
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache
from src.xgb_matcher.timing import PipelineTimer
from src.xgb_matcher.v9_dataset import file_sha256


EXPERIMENT_SCHEMA = "v9-retrieval-experiment-1"
SUPPORTED_MODES = {
    "sparse",
    "hybrid_local",
    "dense_only",
    "hybrid_global_siren",
}


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z_by_confidence = {
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    z = z_by_confidence[confidence]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def load_benchmark(path: Path, split: str) -> pd.DataFrame:
    frame = (
        pd.read_parquet(path)
        if path.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(path, sep=";", dtype=str)
    )
    required = {
        "query_id",
        "crm_name",
        "crm_address",
        "crm_city",
        "postcode",
        "insee",
        "split",
        "ground_truth_siret",
        "ground_truth_siren",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Benchmark is missing columns: {missing}")
    frame = frame[frame["split"].eq(split)].copy()
    if frame.empty:
        raise ValueError(f"Benchmark split is empty: {split}")
    frame["query_id"] = frame["query_id"].astype(str)
    frame["ground_truth_siret"] = (
        frame["ground_truth_siret"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(14)
    )
    frame["ground_truth_siren"] = frame["ground_truth_siret"].str[:9]
    if frame["query_id"].duplicated().any():
        raise ValueError("Benchmark query_id values must be unique within split")
    return frame.reset_index(drop=True)


def retrieval_config(mode: str, *, per_channel_k: int, budget: int) -> RetrievalConfigV1:
    common = {
        "prefilter_k": per_channel_k,
        "fusion_mode": "rrf",
        "retrieval_budget": budget,
        "prefilter_union_cap": None,
        "min_candidates": min(50, budget),
        "mega_insee_policy": "full_insee",
    }
    if mode == "sparse":
        return RetrievalConfigV1(
            **common,
            sparse_retrieval_enabled=True,
            dense_retrieval_enabled=False,
        )
    if mode == "hybrid_local":
        return RetrievalConfigV1(
            **common,
            sparse_retrieval_enabled=True,
            dense_retrieval_enabled=True,
            dense_top_k=per_channel_k,
        )
    if mode == "dense_only":
        return RetrievalConfigV1(
            **common,
            sparse_retrieval_enabled=False,
            dense_retrieval_enabled=True,
            dense_top_k=per_channel_k,
        )
    if mode == "hybrid_global_siren":
        return RetrievalConfigV1(
            **common,
            sparse_retrieval_enabled=True,
            dense_retrieval_enabled=False,
            global_dense_siren_enabled=True,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def _rank_of(values: list[str], target: str) -> int | None:
    try:
        return values.index(target) + 1
    except ValueError:
        return None


def _budget_compliant(
    candidate_count: int,
    *,
    expected_minimum: int,
    budget: int,
) -> bool:
    """Enforce the cutoff without rejecting candidates added by a new channel."""
    return expected_minimum <= candidate_count <= budget


def run_mode(
    *,
    mode: str,
    benchmark: pd.DataFrame,
    partitions_dir: Path,
    cache_root: Path,
    per_channel_k: int,
    budget: int,
    dense_store: Any = None,
    dense_siren_index: Any = None,
    siren_to_geo: Any = None,
    siren_candidate_store: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = retrieval_config(mode, per_channel_k=per_channel_k, budget=budget)
    store = PartitionedCandidateStore(partitions_dir)
    persistent_cache = TfidfPersistentCache(
        config.signature().hash,
        cache_dir=cache_root,
    )
    in_memory_tfidf: OrderedDict = OrderedDict()
    timer = PipelineTimer()
    records: list[dict[str, Any]] = []
    siren_siret_counts = (
        benchmark.groupby("ground_truth_siren")["ground_truth_siret"]
        .nunique()
        .to_dict()
    )
    started = time.perf_counter()
    for position, row in benchmark.iterrows():
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
        query_started = time.perf_counter()
        result = build_candidate_pool(
            store,
            crm_row,
            preprocess_crm_row(crm_row),
            config,
            in_memory_tfidf,
            gt_siret=str(row["ground_truth_siret"]),
            persistent_cache=persistent_cache,
            dense_store=(
                dense_store if config.dense_retrieval_enabled else None
            ),
            dense_siren_index=(
                dense_siren_index
                if config.global_dense_siren_enabled
                else None
            ),
            siren_to_geo=(
                siren_to_geo if config.global_dense_siren_enabled else None
            ),
            siren_candidate_store=(
                siren_candidate_store
                if config.global_dense_siren_enabled
                else None
            ),
            timer=timer,
        )
        latency_ms = (time.perf_counter() - query_started) * 1000
        candidate_sirets = [
            str(candidate.get("siret") or "").zfill(14)
            for candidate in result.candidates
            if candidate.get("siret")
        ]
        candidate_sirens = [siret[:9] for siret in candidate_sirets]
        ground_truth_siret = str(row["ground_truth_siret"])
        ground_truth_siren = str(row["ground_truth_siren"])
        hit_at_1_siret = bool(
            candidate_sirets and candidate_sirets[0] == ground_truth_siret
        )
        hit_at_1_siren = bool(
            candidate_sirens and candidate_sirens[0] == ground_truth_siren
        )
        filtered_count = int(result.pool_sizes.get("filtered", 0))
        expected_count = min(budget, filtered_count)
        record = {
            "mode": mode,
            "query_id": str(row["query_id"]),
            "crm_record_id": row.get("crm_record_id"),
            "ground_truth_siret": ground_truth_siret,
            "ground_truth_siren": ground_truth_siren,
            "hit_at_1_siret": hit_at_1_siret,
            "hit_at_1_siren": hit_at_1_siren,
            "hit_at_budget_siret": ground_truth_siret in candidate_sirets,
            "hit_at_budget_siren": ground_truth_siren in candidate_sirens,
            "ground_truth_rank": _rank_of(candidate_sirets, ground_truth_siret),
            "candidate_count": len(candidate_sirets),
            "expected_candidate_count": expected_count,
            "budget_compliant": _budget_compliant(
                len(candidate_sirets),
                expected_minimum=expected_count,
                budget=budget,
            ),
            "base_pool_size": int(result.pool_sizes.get("base", 0)),
            "filtered_pool_size": filtered_count,
            "ground_truth_in_base": bool(result.gt_in_base_pool),
            "ground_truth_in_filtered": bool(result.gt_in_filtered_pool),
            "loss_reason": result.loss_reason or "",
            "latency_ms": latency_ms,
            "ground_truth_state": row.get("ground_truth_state"),
            "location_match_type": row.get("location_match_type"),
            "missing_insee": not bool(str(row.get("insee") or "").strip()),
            "mega_base_pool": int(result.pool_sizes.get("base", 0)) > 100_000,
            "multi_site_siren": siren_siret_counts.get(ground_truth_siren, 0) > 1,
            "candidate_sirets_json": json.dumps(candidate_sirets),
        }
        records.append(record)
        # Keep RAM bounded on the 24 GB Mac. Evicted partitions remain
        # available in the snapshot-scoped persistent cache on SSD.
        while len(in_memory_tfidf) > 20:
            in_memory_tfidf.popitem(last=False)
        if (position + 1) % 250 == 0:
            print(
                f"[{mode}] {position + 1}/{len(benchmark)} "
                f"recall={sum(item['hit_at_budget_siret'] for item in records) / len(records):.4f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    raw = pd.DataFrame(records)
    summary = summarize_mode(raw, budget=budget)
    summary.update(
        {
            "mode": mode,
            "elapsed_seconds": elapsed,
            "queries_per_second": len(raw) / max(elapsed, 1e-9),
            "retrieval_config": config.to_dict(),
            "retrieval_signature": config.signature().hash,
            "tfidf_cache": persistent_cache.stats(),
            "timer": timer.summary(),
        }
    )
    return raw, summary


def _binary_metric(values: pd.Series) -> dict[str, Any]:
    total = int(len(values))
    successes = int(values.astype(bool).sum())
    lower_95, upper_95 = wilson_interval(successes, total, confidence=0.95)
    lower_99, upper_99 = wilson_interval(successes, total, confidence=0.99)
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else 0.0,
        "wilson_95": [lower_95, upper_95],
        "wilson_99": [lower_99, upper_99],
    }


def _segment_summary(raw: pd.DataFrame) -> dict[str, Any]:
    segments: dict[str, pd.Series] = {
        "all": pd.Series(True, index=raw.index),
        "gt_active": raw["ground_truth_state"].fillna("").eq("A"),
        "gt_closed": raw["ground_truth_state"].fillna("").eq("F"),
        "missing_insee": raw["missing_insee"].astype(bool),
        "mega_base_pool": raw["mega_base_pool"].astype(bool),
        "multi_site_siren": raw["multi_site_siren"].astype(bool),
    }
    for value in sorted(
        str(item)
        for item in raw["location_match_type"].dropna().unique()
        if str(item)
    ):
        segments[f"location_match_type={value}"] = (
            raw["location_match_type"].astype(str).eq(value)
        )
    output = {}
    for name, mask in segments.items():
        subset = raw[mask]
        if subset.empty:
            continue
        output[name] = {
            "recall_at_budget_siret": _binary_metric(
                subset["hit_at_budget_siret"]
            ),
            "recall_at_budget_siren": _binary_metric(
                subset["hit_at_budget_siren"]
            ),
        }
    return output


def summarize_mode(raw: pd.DataFrame, *, budget: int) -> dict[str, Any]:
    summary = {
        "query_count": int(len(raw)),
        "candidate_budget": budget,
        "recall_at_budget_siret": _binary_metric(raw["hit_at_budget_siret"]),
        "recall_at_budget_siren": _binary_metric(raw["hit_at_budget_siren"]),
        "base_recall_siret": _binary_metric(raw["ground_truth_in_base"]),
        "budget_violations": int((~raw["budget_compliant"]).sum()),
        "latency_ms": {
            "p50": float(raw["latency_ms"].quantile(0.50)),
            "p95": float(raw["latency_ms"].quantile(0.95)),
            "p99": float(raw["latency_ms"].quantile(0.99)),
            "mean": float(raw["latency_ms"].mean()),
        },
        "loss_reasons": {
            str(key): int(value)
            for key, value in raw["loss_reason"].value_counts().items()
        },
        "segments": _segment_summary(raw),
    }
    if {"hit_at_1_siret", "hit_at_1_siren"} <= set(raw.columns):
        summary["hit_at_1_siret"] = _binary_metric(raw["hit_at_1_siret"])
        summary["hit_at_1_siren"] = _binary_metric(raw["hit_at_1_siren"])
    return summary


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _artifact_contract(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if manifest_path.is_file():
        digest = file_sha256(manifest_path)
        contract_type = "manifest_or_file"
    elif path.is_dir():
        manifests = sorted(path.glob("*_manifest.json"))
        if not manifests:
            raise FileNotFoundError(
                f"Artifact contract is missing: {manifest_path}"
            )
        hasher = hashlib.sha256()
        for item in manifests:
            hasher.update(item.name.encode("utf-8"))
            hasher.update(item.read_bytes())
        digest = hasher.hexdigest()
        contract_type = "partition_manifests"
    else:
        raise FileNotFoundError(f"Artifact is missing: {path}")
    return {
        "path": str(path),
        "contract_type": contract_type,
        "contract_sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument(
        "--partitions-dir",
        type=Path,
        default=Path("data/candidates_v7_all"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--modes", nargs="+", choices=sorted(SUPPORTED_MODES), required=True)
    parser.add_argument("--per-channel-k", type=int, default=500)
    parser.add_argument("--candidate-budget", type=int, default=50)
    parser.add_argument("--dense-dir", type=Path)
    parser.add_argument("--global-siren-dense-dir", type=Path)
    parser.add_argument("--siren-geo-index", type=Path)
    parser.add_argument("--siren-candidate-store", type=Path)
    parser.add_argument(
        "--semantic-model",
        type=Path,
        default=Path("models/semantic/siret-bert-deploy"),
    )
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Immutable output directory exists: {args.output_dir}")
    if args.candidate_budget <= 0 or args.per_channel_k <= 0:
        raise ValueError("Candidate budgets must be positive")

    benchmark_manifest = json.loads(
        args.benchmark_manifest.read_text(encoding="utf-8")
    )
    expected_benchmark_hash = benchmark_manifest.get("output_sha256", {}).get(
        args.benchmark.name
    )
    observed_benchmark_hash = file_sha256(args.benchmark)
    if expected_benchmark_hash != observed_benchmark_hash:
        raise ValueError("Benchmark hash does not match its frozen manifest")
    benchmark = load_benchmark(args.benchmark, args.split)
    if args.max_rows:
        benchmark = benchmark.head(args.max_rows).copy()

    needs_local_dense = bool(
        {"hybrid_local", "dense_only"} & set(args.modes)
    )
    needs_global_dense = "hybrid_global_siren" in args.modes
    if needs_local_dense and args.dense_dir is None:
        raise ValueError("Local dense modes require --dense-dir")
    if needs_global_dense and (
        args.global_siren_dense_dir is None or args.siren_geo_index is None
    ):
        raise ValueError(
            "Global dense mode requires --global-siren-dense-dir and "
            "--siren-geo-index"
        )

    dense_store = None
    dense_siren_index = None
    siren_to_geo = None
    siren_candidate_store = None
    semantic_client = None
    semantic_model_fingerprint = None
    if needs_local_dense or needs_global_dense:
        os.environ["XGB_SEMANTIC_ENABLED"] = "1"
        os.environ["XGB_SEMANTIC_MODEL"] = str(args.semantic_model)
        os.environ["XGB_SEMANTIC_DEVICE"] = "cpu"
        from src.xgb_matcher.semantic import (
            semantic_artifact_fingerprint,
            set_semantic_client,
        )
        from src.xgb_matcher.semantic_process import SemanticProcessClient

        semantic_client = SemanticProcessClient(args.semantic_model, device="cpu")
        set_semantic_client(semantic_client)
        semantic_model_fingerprint = semantic_artifact_fingerprint(
            args.semantic_model
        )
    if needs_local_dense:
        from src.xgb_matcher.dense_retrieval import PartitionEmbeddingStore

        dense_store = PartitionEmbeddingStore(
            args.dense_dir,
            expected_model_fingerprint=semantic_model_fingerprint,
        )
    if needs_global_dense:
        from src.xgb_matcher.dense_retrieval import GlobalDenseSirenIndex
        from src.xgb_matcher.siren_retrieval import SirenToGeoIndex
        from src.xgb_matcher.siren_retrieval import SirenCandidateStore

        dense_siren_index = GlobalDenseSirenIndex(
            args.global_siren_dense_dir,
            expected_model_fingerprint=semantic_model_fingerprint,
        )
        siren_to_geo = SirenToGeoIndex(args.siren_geo_index)
        if args.siren_candidate_store is not None:
            siren_candidate_store = SirenCandidateStore(
                args.siren_candidate_store
            )

    all_raw = []
    summaries = {}
    try:
        for mode in args.modes:
            raw, summary = run_mode(
                mode=mode,
                benchmark=benchmark,
                partitions_dir=args.partitions_dir,
                cache_root=args.cache_dir,
                per_channel_k=args.per_channel_k,
                budget=args.candidate_budget,
                dense_store=dense_store,
                dense_siren_index=dense_siren_index,
                siren_to_geo=siren_to_geo,
                siren_candidate_store=siren_candidate_store,
            )
            all_raw.append(raw)
            summaries[mode] = summary
    finally:
        if dense_store is not None:
            dense_store.close()
        if dense_siren_index is not None:
            dense_siren_index.close()
        if semantic_client is not None:
            from src.xgb_matcher.semantic import set_semantic_client

            semantic_client.close()
            set_semantic_client(None)
        if siren_candidate_store is not None:
            siren_candidate_store.close()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = args.output_dir / "raw_results.parquet"
    summary_path = args.output_dir / "summary.json"
    pd.concat(all_raw, ignore_index=True).to_parquet(raw_path, index=False)
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "benchmark_build_id": benchmark_manifest["build_id"],
        "benchmark_manifest_sha256": file_sha256(args.benchmark_manifest),
        "benchmark_sha256": observed_benchmark_hash,
        "partitions_sha256": benchmark_manifest["partitions_sha256"],
        "split": args.split,
        "query_count": int(len(benchmark)),
        "modes": args.modes,
        "per_channel_k": args.per_channel_k,
        "candidate_budget": args.candidate_budget,
        "semantic_model": (
            str(args.semantic_model) if needs_local_dense or needs_global_dense else None
        ),
        "semantic_model_fingerprint": semantic_model_fingerprint,
        "dense_artifacts": {
            "local": _artifact_contract(args.dense_dir),
            "global_siren": _artifact_contract(
                args.global_siren_dense_dir
            ),
            "siren_geo": _artifact_contract(args.siren_geo_index),
            "siren_candidates": _artifact_contract(
                args.siren_candidate_store
            ),
        },
        "outputs": {
            "raw_results.parquet": file_sha256(raw_path),
            "summary.json": file_sha256(summary_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
