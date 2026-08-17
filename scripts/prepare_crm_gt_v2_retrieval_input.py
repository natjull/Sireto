#!/usr/bin/env python3
"""Freeze certified CRM GT v2 rows as a non-injected retrieval benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256
from scripts.freeze_v9_closed_benchmark import directory_tree_sha256


SCHEMA_VERSION = "sireto-crm-gt-v2-retrieval-input-1"
SPLITS = {
    "TRAIN": "crm_train",
    "PROSPECTIVE_DEV": "crm_prospective_dev",
    "PROSPECTIVE_TEST": "crm_prospective_test",
}


def build(args: argparse.Namespace) -> Path:
    manifest = json.loads((args.population / "manifest.json").read_text())
    if manifest.get("audit_gate", {}).get("status") != "PASS":
        raise ValueError("CRM GT population is not certified")
    for name, expected in manifest["outputs"].items():
        if file_sha256(args.population / name) != expected:
            raise ValueError(f"Population hash mismatch: {name}")
    queries = pd.read_parquet(args.population / "queries.parquet")
    labels = pd.read_parquet(args.population / "labels.parquet")
    labels = labels[
        labels["data_origin"].eq("REAL_CRM_20260817")
        & labels["exact_metric_eligible"].astype(bool)
    ].copy()
    frame = queries.merge(
        labels[
            ["query_id", "ground_truth_siret", "ground_truth_siren", "ground_truth_state"]
        ],
        on="query_id",
        validate="one_to_one",
    )
    frame["split"] = frame["split_role"].map(SPLITS)
    frame["postcode"] = frame["crm_postcode"]
    frame["insee"] = frame["crm_insee"]
    frame["location_match_type"] = "insee"
    benchmark = frame[
        [
            "query_id", "crm_name", "crm_address", "crm_city", "postcode", "insee",
            "reference_date", "split", "ground_truth_siret", "ground_truth_siren",
            "ground_truth_state", "location_match_type", "oof_fold",
        ]
    ].sort_values("query_id", kind="mergesort")
    partition_fingerprint = directory_tree_sha256(args.partitions_dir)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "population_manifest_sha256": file_sha256(args.population / "manifest.json"),
        "retrieval_contract_sha256": file_sha256(args.retrieval_contract),
        "partitions_sha256": partition_fingerprint["sha256"],
        "partitions_file_count": partition_fingerprint["file_count"],
        "partitions_total_bytes": partition_fingerprint["total_bytes"],
        "positive_injection": False,
        "qualification_uses_retrieval_or_model_scores": False,
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        return destination
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    benchmark.to_parquet(temporary / "benchmark.parquet", index=False)
    output_hash = file_sha256(temporary / "benchmark.parquet")
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_identity": identity,
        "query_count": len(benchmark),
        "split_counts": {str(k): int(v) for k, v in benchmark["split"].value_counts().items()},
        "partitions_sha256": partition_fingerprint["sha256"],
        "partitions": partition_fingerprint,
        "positive_injection": False,
        "qualification_uses_retrieval_or_model_scores": False,
        "output_sha256": {"benchmark.parquet": output_hash},
        "outputs": {"benchmark.parquet": output_hash},
    }
    (temporary / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--retrieval-contract", type=Path, required=True)
    parser.add_argument("--partitions-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(build(parser.parse_args()))


if __name__ == "__main__":
    main()
