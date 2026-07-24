#!/usr/bin/env python3
"""Measure an immutable sparse retrieval recall curve in one max-K pass."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import (  # noqa: E402
    _binary_metric,
    _git_commit,
    load_benchmark,
    run_mode,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-retrieval-recall-curve-1"
CRITICAL_SEGMENTS = (
    "all",
    "gt_active",
    "gt_closed",
    "mega_base_pool",
    "multi_site_siren",
    "location_match_type=cp_only",
    "location_match_type=insee",
)


def _segment_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=frame.index),
        "gt_active": frame["ground_truth_state"].fillna("").eq("A"),
        "gt_closed": frame["ground_truth_state"].fillna("").eq("F"),
        "mega_base_pool": frame["mega_base_pool"].astype(bool),
        "multi_site_siren": frame["multi_site_siren"].astype(bool),
        "location_match_type=cp_only": (
            frame["location_match_type"].astype(str).eq("cp_only")
        ),
        "location_match_type=insee": (
            frame["location_match_type"].astype(str).eq("insee")
        ),
    }


def _loss_bucket(row: pd.Series, *, max_cutoff: int) -> str:
    if not bool(row["ground_truth_in_base"]):
        return "PARTITION_MISS"
    if not bool(row["ground_truth_in_filtered_pre_dedupe"]):
        return "FILTER_MISS"
    if not bool(row["ground_truth_in_deduped"]):
        return "DEDUPE_MISS"
    rank_value = row.get("ground_truth_rank")
    if pd.isna(rank_value) or int(rank_value) > max_cutoff:
        return f"PRUNED_AT_{max_cutoff}"
    rank = int(rank_value)
    if rank <= 50:
        return "HIT_AT_50"
    if rank <= 100:
        return "PRUNED_AT_50_RECOVERED_BY_100"
    if rank <= 200:
        return "PRUNED_AT_100_RECOVERED_BY_200"
    return "PRUNED_AT_200_RECOVERED_BY_500"


def build_recall_curve(
    raw: pd.DataFrame,
    *,
    cutoffs: list[int],
) -> dict[str, Any]:
    candidate_lists = raw["candidate_sirets_json"].map(json.loads)
    ground_truth_sirets = raw["ground_truth_siret"].astype(str)
    ground_truth_sirens = raw["ground_truth_siren"].astype(str)
    masks = _segment_masks(raw)
    curve: dict[str, Any] = {}

    for cutoff in cutoffs:
        hit_siret = pd.Series(
            [
                ground_truth in candidates[:cutoff]
                for ground_truth, candidates in zip(
                    ground_truth_sirets,
                    candidate_lists,
                    strict=True,
                )
            ],
            index=raw.index,
        )
        hit_siren = pd.Series(
            [
                ground_truth in {siret[:9] for siret in candidates[:cutoff]}
                for ground_truth, candidates in zip(
                    ground_truth_sirens,
                    candidate_lists,
                    strict=True,
                )
            ],
            index=raw.index,
        )
        segments = {}
        for name, mask in masks.items():
            subset_siret = hit_siret[mask]
            subset_siren = hit_siren[mask]
            if subset_siret.empty:
                continue
            segments[name] = {
                "siret": _binary_metric(subset_siret),
                "siren": _binary_metric(subset_siren),
            }
        curve[str(cutoff)] = {
            "siret": _binary_metric(hit_siret),
            "siren": _binary_metric(hit_siren),
            "segments": segments,
        }
    return curve


def build_stage_audit(raw: pd.DataFrame, *, max_cutoff: int) -> dict[str, Any]:
    buckets = raw.apply(
        _loss_bucket,
        axis=1,
        max_cutoff=max_cutoff,
    )
    return {
        "stage_recall": {
            "partition": _binary_metric(raw["ground_truth_in_base"]),
            "filtered_pre_dedupe": _binary_metric(
                raw["ground_truth_in_filtered_pre_dedupe"]
            ),
            "deduped": _binary_metric(raw["ground_truth_in_deduped"]),
        },
        "loss_buckets": {
            str(name): int(count)
            for name, count in buckets.value_counts().sort_index().items()
        },
        "loss_bucket_by_query": buckets,
    }


def _markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    rows = [
        "# Retrieval SIRET — courbe sparse Recall@K",
        "",
        f"- Benchmark : `{manifest['benchmark_build_id']}` / `{manifest['split']}`",
        f"- Requêtes : {manifest['query_count']}",
        f"- Commit : `{manifest['git_commit'][:12]}`",
        "",
        "## Courbe globale",
        "",
        "| K | Recall SIRET | IC95 | Recall SIREN | Maximum candidats |",
        "|---:|---:|---:|---:|---:|",
    ]
    for cutoff, values in summary["recall_curve"].items():
        siret = values["siret"]
        siren = values["siren"]
        rows.append(
            f"| {cutoff} | {siret['successes']}/{siret['total']} = "
            f"{siret['rate']:.2%} | "
            f"[{siret['wilson_95'][0]:.2%}; {siret['wilson_95'][1]:.2%}] | "
            f"{siren['rate']:.2%} | {cutoff} |"
        )
    rows.extend(
        [
            "",
            "## Rappel par stage",
            "",
            "| Stage | Vérités présentes | Recall |",
            "|---|---:|---:|",
        ]
    )
    for name, values in summary["stage_audit"]["stage_recall"].items():
        rows.append(
            f"| {name} | {values['successes']}/{values['total']} | "
            f"{values['rate']:.2%} |"
        )
    rows.extend(
        [
            "",
            "## Attribution des requêtes",
            "",
            "| Bucket | Nombre |",
            "|---|---:|",
        ]
    )
    for name, count in summary["stage_audit"]["loss_buckets"].items():
        rows.append(f"| {name} | {count} |")
    rows.extend(
        [
            "",
            "## Latence et cardinalité",
            "",
            f"- p95 : {summary['latency_ms']['p95']:.0f} ms",
            f"- maximum retourné : {summary['candidate_counts']['max']}",
            f"- moyenne retournée : {summary['candidate_counts']['mean']:.1f}",
            f"- p95 retourné : {summary['candidate_counts']['p95']:.1f}",
            "",
        ]
    )
    return "\n".join(rows)


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
    observed_benchmark_hash = file_sha256(args.benchmark)
    expected_benchmark_hash = benchmark_manifest.get(
        "output_sha256", {}
    ).get(args.benchmark.name)
    if observed_benchmark_hash != expected_benchmark_hash:
        raise ValueError("Benchmark hash does not match frozen manifest")
    benchmark = load_benchmark(args.benchmark, args.split)
    if args.max_rows:
        benchmark = benchmark.head(args.max_rows).copy()

    raw, run_summary = run_mode(
        mode="sparse",
        benchmark=benchmark,
        partitions_dir=args.partitions_dir,
        cache_root=args.cache_dir,
        per_channel_k=args.per_channel_k,
        budget=cutoffs[-1],
    )
    stage_audit = build_stage_audit(raw, max_cutoff=cutoffs[-1])
    raw["loss_bucket"] = stage_audit.pop("loss_bucket_by_query")
    candidate_counts = raw["candidate_count"].astype(int)
    summary = {
        "recall_curve": build_recall_curve(raw, cutoffs=cutoffs),
        "stage_audit": stage_audit,
        "candidate_counts": {
            "max": int(candidate_counts.max()),
            "mean": float(candidate_counts.mean()),
            "p95": float(np.quantile(candidate_counts, 0.95)),
            "over_100": int(candidate_counts.gt(100).sum()),
        },
        "latency_ms": run_summary["latency_ms"],
        "elapsed_seconds": run_summary["elapsed_seconds"],
        "tfidf_cache": run_summary["tfidf_cache"],
        "retrieval_config": run_summary["retrieval_config"],
        "retrieval_signature": run_summary["retrieval_signature"],
    }

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
        "benchmark_sha256": observed_benchmark_hash,
        "partitions_sha256": benchmark_manifest["partitions_sha256"],
        "split": args.split,
        "query_count": int(len(benchmark)),
        "cutoffs": cutoffs,
        "per_channel_k": args.per_channel_k,
    }
    report_path.write_text(
        _markdown(summary, manifest),
        encoding="utf-8",
    )
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
