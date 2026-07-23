#!/usr/bin/env python3
"""Freeze the exact local dense partitions required by benchmark splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.blocking import normalize_code
from src.xgb_matcher.v9_dataset import file_sha256


def available_codes(partitions_dir: Path, kind: str) -> set[str]:
    column = "insee" if kind == "insee" else "postcode"
    return {
        path.name.split("=", 1)[1]
        for path in (partitions_dir / kind).iterdir()
        if path.is_dir() and path.name.startswith(f"{column}=")
    }


def build_partition_plan(
    benchmark: pd.DataFrame,
    *,
    partitions_dir: Path,
    splits: set[str],
) -> dict:
    selected = benchmark[benchmark["split"].isin(splits)].copy()
    if selected.empty:
        raise ValueError(f"No benchmark rows for splits: {sorted(splits)}")
    insee_available = available_codes(partitions_dir, "insee")
    postcode_available = available_codes(partitions_dir, "cp")
    insee_codes: set[str] = set()
    postcode_codes: set[str] = set()
    missing_queries: list[str] = []
    for row in selected.to_dict("records"):
        insee = normalize_code(row.get("insee"))
        postcode = normalize_code(row.get("postcode"))
        if insee and insee in insee_available:
            insee_codes.add(insee)
        elif postcode and postcode in postcode_available:
            postcode_codes.add(postcode)
        else:
            missing_queries.append(str(row.get("query_id") or ""))
    return {
        "schema_version": "v9-dense-partition-plan-1",
        "splits": sorted(splits),
        "query_count": int(len(selected)),
        "insee_codes": sorted(insee_codes),
        "postcode_codes": sorted(postcode_codes),
        "missing_query_ids": missing_queries,
        "counts": {
            "insee_partitions": len(insee_codes),
            "postcode_partitions": len(postcode_codes),
            "missing_queries": len(missing_queries),
        },
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
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Immutable partition plan exists: {args.output}")
    manifest = json.loads(args.benchmark_manifest.read_text(encoding="utf-8"))
    benchmark_hash = file_sha256(args.benchmark)
    if manifest["output_sha256"].get(args.benchmark.name) != benchmark_hash:
        raise ValueError("Benchmark hash mismatch")
    benchmark = pd.read_parquet(args.benchmark)
    plan = build_partition_plan(
        benchmark,
        partitions_dir=args.partitions_dir,
        splits=set(args.splits),
    )
    plan.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_build_id": manifest["build_id"],
            "benchmark_sha256": benchmark_hash,
            "partitions_sha256": manifest["partitions_sha256"],
        }
    )
    payload = json.dumps(plan, indent=2, sort_keys=True)
    plan["plan_id"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(plan["counts"], indent=2))


if __name__ == "__main__":
    main()
