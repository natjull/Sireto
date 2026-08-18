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


SCHEMA_VERSION = "sireto-crm-gt-v2-retrieval-input-3"
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
    scope = str(getattr(args, "scope", "fresh_commercial"))
    if scope == "fresh_commercial":
        labels = labels[
            labels["data_origin"].eq("REAL_CRM_20260817")
            & labels["exact_metric_eligible"].astype(bool)
        ].copy()
    elif scope == "all_human":
        if "label_is_human_validated" not in labels:
            raise ValueError("Full-human scope requires explicit human provenance")
        labels = labels[labels["label_is_human_validated"].astype(bool)].copy()
    else:
        raise ValueError(f"Unsupported CRM retrieval scope: {scope}")
    label_columns = [
        "query_id", "ground_truth_siret", "ground_truth_siren",
        "ground_truth_state", "acceptable_sirets_operational",
    ]
    for optional in (
        "historical_ground_truth_siret", "historical_ground_truth_siren",
        "label_kind", "exact_metric_eligible", "data_origin",
    ):
        if optional in labels and optional not in queries:
            label_columns.append(optional)
    frame = queries.merge(
        labels[label_columns],
        on="query_id",
        validate="one_to_one",
    )
    if scope == "all_human":
        frame["split"] = frame["oof_fold"].map(
            {0: "crm_prospective_dev", 1: "crm_prospective_test",
             2: "crm_train", 3: "crm_train", 4: "crm_train"}
        )
    else:
        frame["split"] = frame["split_role"].map(SPLITS)
    frame["postcode"] = frame["crm_postcode"]
    frame["insee"] = frame["crm_insee"]
    frame["location_match_type"] = frame["insee"].fillna("").astype(str).str.strip().map(
        lambda value: "insee" if value else "cp_only"
    )
    if scope == "all_human":
        frame["label_kind"] = frame["label_kind"].fillna("UNRESOLVED")
        frame["identifiable_exact"] = frame["exact_metric_eligible"].astype(bool)
        # Both policy views are explicit and retrieval-independent.  The
        # prospective commercial labels are human exact labels guarded by
        # SIRENE geography; historical rows retain their already-published
        # MATCH_EXACT/AMBIGUOUS/UNRESOLVED classification.
        frame["v2_label_kind"] = frame["label_kind"]
        frame["v3_label_kind"] = frame["label_kind"]
        frame["v2_exact"] = frame["label_kind"].eq("MATCH_EXACT")
        frame["v3_exact"] = frame["label_kind"].eq("MATCH_EXACT")
        frame["qualification_v2"] = frame["v2_label_kind"]
        frame["qualification_v3"] = frame["v3_label_kind"]
        frame["source_kind"] = "REAL_HUMAN_CRM"
        frame["is_synthetic"] = False
        train = frame[frame["oof_fold"].isin([2, 3, 4])]
        train_sirens = set(train["ground_truth_siren"].dropna().astype(str))
        train_sirets = set(train["ground_truth_siret"].dropna().astype(str))
        frame["unseen_siren"] = ~frame["ground_truth_siren"].fillna("").astype(str).isin(
            train_sirens
        )
        frame["new_site_known_siren"] = (
            frame["ground_truth_siren"].fillna("").astype(str).isin(train_sirens)
            & ~frame["ground_truth_siret"].fillna("").astype(str).isin(train_sirets)
        )
    columns = [
        "query_id", "crm_name", "crm_address", "crm_city", "postcode", "insee",
        "reference_date", "split", "ground_truth_siret", "ground_truth_siren",
        "ground_truth_state", "acceptable_sirets_operational",
        "location_match_type", "oof_fold",
    ]
    if scope == "all_human":
        columns.extend(
            [
                "historical_ground_truth_siret", "historical_ground_truth_siren",
                "label_kind", "identifiable_exact", "v2_label_kind", "v3_label_kind",
                "v2_exact", "v3_exact", "qualification_v2", "qualification_v3",
                "source_kind", "is_synthetic", "unseen_siren", "new_site_known_siren",
                "data_origin",
            ]
        )
    benchmark = frame[columns].sort_values("query_id", kind="mergesort")
    partition_fingerprint = directory_tree_sha256(args.partitions_dir)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "population_manifest_sha256": file_sha256(args.population / "manifest.json"),
        "retrieval_contract_sha256": file_sha256(args.retrieval_contract),
        "partitions_sha256": partition_fingerprint["sha256"],
        "partitions_file_count": partition_fingerprint["file_count"],
        "partitions_total_bytes": partition_fingerprint["total_bytes"],
        "population_scope": scope,
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
    output_hashes = {"benchmark.parquet": file_sha256(temporary / "benchmark.parquet")}
    if scope == "all_human":
        # The development input physically excludes fold 1.  The locked test
        # file is sealed now but can only be consumed through the one-shot test
        # authorization implemented by the admission runner.
        development = benchmark[benchmark["oof_fold"].isin([0, 2, 3, 4])]
        test_locked = benchmark[benchmark["oof_fold"].eq(1)]
        development.to_parquet(temporary / "development.parquet", index=False)
        test_locked.to_parquet(temporary / "test_locked.parquet", index=False)
        output_hashes.update(
            {
                name: file_sha256(temporary / name)
                for name in ("development.parquet", "test_locked.parquet")
            }
        )
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_identity": identity,
        "query_count": len(benchmark),
        "identifiable_exact_count": int(
            benchmark.get("identifiable_exact", pd.Series([True] * len(benchmark))).sum()
        ),
        "split_counts": {str(k): int(v) for k, v in benchmark["split"].value_counts().items()},
        "partitions_sha256": partition_fingerprint["sha256"],
        "partitions": partition_fingerprint,
        "positive_injection": False,
        "qualification_uses_retrieval_or_model_scores": False,
        "development_opened_folds": [0, 2, 3, 4] if scope == "all_human" else None,
        "test_locked": scope == "all_human",
        "output_sha256": output_hashes,
        "outputs": output_hashes,
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
    parser.add_argument(
        "--scope", choices=["fresh_commercial", "all_human"], default="fresh_commercial"
    )
    print(build(parser.parse_args()))


if __name__ == "__main__":
    main()
