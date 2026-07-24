#!/usr/bin/env python3
"""Evaluate the strongest frozen deterministic admission policy on dev."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import (  # noqa: E402
    _binary_metric,
    _git_commit,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-retrieval-admission-diagnostic-1"
V7_WEIGHTS = {
    "current_sparse": 2.0,
    "name_word": 1.0,
    "name_char": 1.0,
    "address_word": 0.5,
    "siren_head": 1.0,
    "name_exact": 2.0,
    "address_exact": 2.0,
}
OVERLAY_QUOTAS = {
    "name_word": 1,
    "name_char": 10,
}
ORACLE_CHANNELS = (
    *V7_WEIGHTS,
    "siren_sites",
    "numeric_name",
)


def _load_and_validate(
    raw_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("outputs", {}).get(raw_path.name)
    if expected != file_sha256(raw_path):
        raise ValueError(f"Raw artifact hash mismatch: {raw_path}")
    return pd.read_parquet(raw_path), manifest


def select_candidates(
    *,
    v7_channels: dict[str, list[str]],
    overlay_channels: dict[str, list[str]],
    budget: int,
    internal_k: int,
) -> list[str]:
    """Apply weighted rank fusion plus small source-preserving overlay quotas."""
    scores: dict[str, float] = {}
    for channel, weight in V7_WEIGHTS.items():
        for rank, siret in enumerate(
            v7_channels.get(channel, [])[:internal_k],
            start=1,
        ):
            scores[siret] = scores.get(siret, 0.0) + weight / rank

    selected: list[str] = []
    seen: set[str] = set()
    for channel, quota in OVERLAY_QUOTAS.items():
        for siret in overlay_channels.get(channel, [])[:quota]:
            if siret not in seen:
                seen.add(siret)
                selected.append(siret)
                if len(selected) >= budget:
                    return selected

    for siret in sorted(scores, key=lambda item: (-scores[item], item)):
        if siret in seen:
            continue
        seen.add(siret)
        selected.append(siret)
        if len(selected) >= budget:
            break
    return selected


def _oracle_candidates(
    *,
    v7_channels: dict[str, list[str]],
    overlay_channels: dict[str, list[str]],
    internal_k: int,
) -> set[str]:
    output: set[str] = set()
    for channel in ORACLE_CHANNELS:
        output.update(v7_channels.get(channel, [])[:internal_k])
        output.update(overlay_channels.get(channel, [])[:internal_k])
    return output


def _segment_summary(raw: pd.DataFrame) -> dict[str, Any]:
    masks = {
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
    return {
        name: _binary_metric(raw.loc[mask, "hit_at_100"])
        for name, mask in masks.items()
        if mask.sum()
    }


def evaluate(
    *,
    v7_raw: pd.DataFrame,
    overlay_raw: pd.DataFrame,
    frozen_baseline_raw: pd.DataFrame,
    budget: int,
    internal_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if list(v7_raw["query_id"].astype(str)) != list(
        overlay_raw["query_id"].astype(str)
    ):
        raise ValueError("V7 and overlay query order differs")
    baseline_by_query = frozen_baseline_raw.set_index(
        frozen_baseline_raw["query_id"].astype(str)
    )
    if set(baseline_by_query.index) != set(v7_raw["query_id"].astype(str)):
        raise ValueError("Frozen baseline query IDs differ")

    records: list[dict[str, Any]] = []
    selection_latencies: list[float] = []
    for position, (_, v7_row) in enumerate(v7_raw.iterrows()):
        overlay_row = overlay_raw.iloc[position]
        v7_channels = {
            channel: json.loads(v7_row[f"{channel}_sirets_json"])
            for channel in ORACLE_CHANNELS
        }
        overlay_channels = {
            channel: json.loads(overlay_row[f"{channel}_sirets_json"])
            for channel in ORACLE_CHANNELS
        }
        started = time.perf_counter()
        selected = select_candidates(
            v7_channels=v7_channels,
            overlay_channels=overlay_channels,
            budget=budget,
            internal_k=internal_k,
        )
        selection_latencies.append((time.perf_counter() - started) * 1000)
        oracle = _oracle_candidates(
            v7_channels=v7_channels,
            overlay_channels=overlay_channels,
            internal_k=internal_k,
        )
        query_id = str(v7_row["query_id"])
        ground_truth = str(v7_row["ground_truth_siret"])
        baseline_candidates = json.loads(
            baseline_by_query.loc[query_id, "candidate_sirets_json"]
        )
        records.append(
            {
                "query_id": query_id,
                "ground_truth_siret": ground_truth,
                "ground_truth_siren": str(v7_row["ground_truth_siren"]),
                "ground_truth_state": v7_row.get("ground_truth_state"),
                "location_match_type": v7_row.get("location_match_type"),
                "mega_base_pool": bool(v7_row.get("mega_base_pool")),
                "multi_site_siren": bool(v7_row.get("multi_site_siren")),
                "hit_at_100": ground_truth in selected,
                "oracle_hit": ground_truth in oracle,
                "baseline_hit_at_100": ground_truth in baseline_candidates[:100],
                "candidate_count": len(selected),
                "ground_truth_rank": (
                    selected.index(ground_truth) + 1
                    if ground_truth in selected
                    else None
                ),
                "candidate_sirets_json": json.dumps(
                    selected,
                    separators=(",", ":"),
                ),
            }
        )
    raw = pd.DataFrame(records)
    candidate_counts = raw["candidate_count"].astype(int)
    target_successes = int(np.ceil(0.99 * len(raw)))
    summary = {
        "policy_status": "EXPLORATORY_DEV_SELECTED_NOT_PROMOTABLE",
        "candidate_budget": budget,
        "internal_channel_k": internal_k,
        "policy": {
            "fusion": "weighted_reciprocal_rank_k0",
            "v7_weights": V7_WEIGHTS,
            "overlay_quotas": OVERLAY_QUOTAS,
            "tie_break": "siret_ascending",
        },
        "recall_at_100_siret": _binary_metric(raw["hit_at_100"]),
        "frozen_sparse_at_100_siret": _binary_metric(
            raw["baseline_hit_at_100"]
        ),
        "internal_oracle_siret": _binary_metric(raw["oracle_hit"]),
        "target": {
            "rate": 0.99,
            "required_successes": target_successes,
            "gap_successes": max(
                0,
                target_successes - int(raw["hit_at_100"].sum()),
            ),
        },
        "losses": {
            "not_seen_by_any_internal_channel": int(
                (~raw["oracle_hit"]).sum()
            ),
            "pruned_by_admission": int(
                (raw["oracle_hit"] & ~raw["hit_at_100"]).sum()
            ),
        },
        "candidate_counts": {
            "max": int(candidate_counts.max()),
            "mean": float(candidate_counts.mean()),
            "p95": float(candidate_counts.quantile(0.95)),
            "over_100": int(candidate_counts.gt(100).sum()),
        },
        "selection_latency_ms": {
            "p50": float(np.quantile(selection_latencies, 0.50)),
            "p95": float(np.quantile(selection_latencies, 0.95)),
            "p99": float(np.quantile(selection_latencies, 0.99)),
            "mean": float(np.mean(selection_latencies)),
        },
        "segments": _segment_summary(raw),
    }
    return raw, summary


def _markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    observed = summary["recall_at_100_siret"]
    baseline = summary["frozen_sparse_at_100_siret"]
    oracle = summary["internal_oracle_siret"]
    return "\n".join(
        [
            "# Admission déterministe — diagnostic dev",
            "",
            f"- Benchmark : `{manifest['benchmark_build_id']}` / dev",
            f"- Requêtes : {observed['total']}",
            f"- Commit : `{manifest['git_commit'][:12]}`",
            "- Statut : exploratoire, sélectionné sur dev, non promouvable.",
            "",
            "## Résultats",
            "",
            f"- Sparse gelé @100 : {baseline['successes']}/{baseline['total']} "
            f"= {baseline['rate']:.2%}.",
            f"- Meilleure admission déterministe observée @100 : "
            f"{observed['successes']}/{observed['total']} = "
            f"{observed['rate']:.2%}.",
            f"- Oracle des canaux internes @"
            f"{summary['internal_channel_k']} : "
            f"{oracle['successes']}/{oracle['total']} = {oracle['rate']:.2%}.",
            f"- Écart au gate : {summary['target']['gap_successes']} requêtes.",
            f"- Pertes admission malgré candidat visible : "
            f"{summary['losses']['pruned_by_admission']}.",
            f"- Candidats >100 : {summary['candidate_counts']['over_100']}.",
            "",
            "## Conclusion",
            "",
            "Le sourcing et les canaux internes rendent la cible théoriquement "
            "accessible, mais une fusion déterministe ne compresse pas ce signal "
            "en 100 candidats. Le gate retrieval indépendant du ranker échoue.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-raw", type=Path, required=True)
    parser.add_argument("--v7-manifest", type=Path, required=True)
    parser.add_argument("--overlay-raw", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--frozen-baseline-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=100)
    parser.add_argument("--internal-k", type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Immutable output already exists: {args.output_dir}"
        )
    if args.candidate_budget > 100:
        raise ValueError("Candidate budget cannot exceed 100")

    v7_raw, v7_manifest = _load_and_validate(
        args.v7_raw,
        args.v7_manifest,
    )
    overlay_raw, overlay_manifest = _load_and_validate(
        args.overlay_raw,
        args.overlay_manifest,
    )
    if (
        v7_manifest["benchmark_build_id"]
        != overlay_manifest["benchmark_build_id"]
        or v7_manifest["split"] != overlay_manifest["split"]
    ):
        raise ValueError("Input benchmark contracts differ")
    frozen_baseline = pd.read_parquet(
        args.frozen_baseline_raw,
        columns=["query_id", "candidate_sirets_json"],
    )
    raw, summary = evaluate(
        v7_raw=v7_raw,
        overlay_raw=overlay_raw,
        frozen_baseline_raw=frozen_baseline,
        budget=args.candidate_budget,
        internal_k=args.internal_k,
    )

    args.output_dir.mkdir(parents=True)
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
        "benchmark_build_id": v7_manifest["benchmark_build_id"],
        "split": v7_manifest["split"],
        "query_count": int(len(raw)),
        "inputs": {
            "v7_manifest_sha256": file_sha256(args.v7_manifest),
            "overlay_manifest_sha256": file_sha256(args.overlay_manifest),
            "frozen_baseline_raw_sha256": file_sha256(
                args.frozen_baseline_raw
            ),
        },
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
