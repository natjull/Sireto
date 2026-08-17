#!/usr/bin/env python3
"""Evaluate frozen top-100 admission on the prospective certified CRM DEV."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_retrieval_admission import ORACLE_CHANNELS, select_candidates
from src.xgb_matcher.v9_dataset import file_sha256


def _load(root: Path, source_manifest: Path, split: str) -> pd.DataFrame:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["benchmark_manifest_sha256"] != file_sha256(source_manifest):
        raise ValueError("Channel benchmark mismatch")
    if manifest["split"] != split or int(manifest["per_channel_k"]) != 5000:
        raise ValueError("Channel policy mismatch")
    if file_sha256(root / "raw_results.parquet") != manifest["outputs"]["raw_results.parquet"]:
        raise ValueError("Channel raw hash mismatch")
    return pd.read_parquet(root / "raw_results.parquet")


def _metric(series: pd.Series) -> dict[str, float | int]:
    return {"successes": int(series.sum()), "total": len(series), "rate": float(series.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v7", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--preliminary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="crm_prospective_dev")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    benchmark = pd.read_parquet(args.source / "benchmark.parquet")
    benchmark = benchmark[benchmark["split"].eq(args.split)].copy()
    benchmark_by_query = benchmark.assign(
        query_id=benchmark["query_id"].astype(str)
    ).set_index("query_id")
    v7 = _load(args.v7, args.source / "manifest.json", args.split)
    overlay = _load(args.overlay, args.source / "manifest.json", args.split)
    if list(v7["query_id"].astype(str)) != list(overlay["query_id"].astype(str)):
        raise ValueError("Channel order differs")
    records = []
    for idx, row in v7.iterrows():
        other = overlay.iloc[idx]
        v7_lists = {name: json.loads(row[f"{name}_sirets_json"]) for name in ORACLE_CHANNELS}
        overlay_lists = {name: json.loads(other[f"{name}_sirets_json"]) for name in ORACLE_CHANNELS}
        selected = select_candidates(v7_channels=v7_lists, overlay_channels=overlay_lists, budget=100, internal_k=5000)
        oracle = set()
        for name in ORACLE_CHANNELS:
            oracle.update(v7_lists[name][:5000])
            oracle.update(overlay_lists[name][:5000])
        truth = str(row["ground_truth_siret"])
        acceptable = set(
            json.loads(
                str(
                    benchmark_by_query.loc[
                        str(row["query_id"]), "acceptable_sirets_operational"
                    ]
                )
            )
        )
        acceptable.add(truth)
        records.append({
            "query_id": str(row["query_id"]), "ground_truth_siret": truth,
            "ground_truth_state": str(row["ground_truth_state"]),
            "candidate_count": len(selected), "oracle_hit": truth in oracle,
            "hit_at_100": truth in selected,
            "oracle_hit_operational": bool(acceptable.intersection(oracle)),
            "hit_at_100_operational": bool(acceptable.intersection(selected)),
            "acceptable_sirets_operational_json": json.dumps(
                sorted(acceptable), separators=(",", ":")
            ),
            "candidate_sirets_json": json.dumps(selected, separators=(",", ":")),
        })
    raw = pd.DataFrame(records)
    preliminary_labels = pd.read_parquet(args.preliminary / "labels.parquet")
    preliminary_dev = preliminary_labels[
        preliminary_labels["data_origin"].eq("REAL_CRM_20260817")
        & preliminary_labels["split_role"].eq("PROSPECTIVE_DEV")
    ]
    qualification_coverage = len(raw) / len(preliminary_dev)
    summary = {
        "schema_version": "sireto-crm-gt-v2-retrieval-eval-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "qualification_coverage_over_geographic_source": {
            "successes": len(raw), "total": len(preliminary_dev), "rate": qualification_coverage,
        },
        "recall_at_100_exact": _metric(raw["hit_at_100"]),
        "recall_at_100_operational": _metric(raw["hit_at_100_operational"]),
        "oracle_recall_at_5000": _metric(raw["oracle_hit"]),
        "oracle_recall_at_5000_operational": _metric(raw["oracle_hit_operational"]),
        "recall_by_state": {
            state: _metric(group["hit_at_100"])
            for state, group in raw.groupby("ground_truth_state")
        },
        "recall_operational_by_state": {
            state: _metric(group["hit_at_100_operational"])
            for state, group in raw.groupby("ground_truth_state")
        },
        "candidate_count_max": int(raw["candidate_count"].max()),
        "positive_injection": False,
    }
    summary["gates"] = {
        "qualification_coverage_at_least_80pct": qualification_coverage >= 0.80,
        "recall_at_100_at_least_99pct": summary["recall_at_100_exact"]["rate"] >= 0.99,
        "oracle_complete": bool(raw["oracle_hit"].all()),
        "candidate_ceiling_100": int(raw["candidate_count"].max()) <= 100,
    }
    summary["verdict"] = "GO" if all(summary["gates"].values()) else "PIVOT"
    args.output_dir.mkdir(parents=True)
    raw.to_parquet(args.output_dir / "raw_results.parquet", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": summary["schema_version"],
        "source_manifest_sha256": file_sha256(args.source / "manifest.json"),
        "v7_manifest_sha256": file_sha256(args.v7 / "manifest.json"),
        "overlay_manifest_sha256": file_sha256(args.overlay / "manifest.json"),
        "outputs": {
            "raw_results.parquet": file_sha256(args.output_dir / "raw_results.parquet"),
            "summary.json": file_sha256(args.output_dir / "summary.json"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
