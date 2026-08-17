#!/usr/bin/env python3
"""Certify a pending sanitized CRM GT population after its frozen review gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_crm_gt_v2_population import sha256


SCHEMA_VERSION = "sireto-crm-gt-v2-certified-population-1"


def build(args: argparse.Namespace) -> Path:
    pending_manifest = json.loads((args.pending / "manifest.json").read_text())
    for name, expected in pending_manifest["outputs"].items():
        if sha256(args.pending / name) != expected:
            raise ValueError(f"Pending manifest mismatch: {name}")
    review_manifest = json.loads((args.review / "manifest.json").read_text())
    if (
        review_manifest.get("rows") != 400
        or review_manifest.get("counts", {}).get("CERTAIN_FALSE_LABEL") != 0
        or not review_manifest.get("gate_pass")
    ):
        raise ValueError("Independent review gate did not pass")
    review_path = args.review / "reviews.jsonl"
    if sha256(review_path) != review_manifest["outputs"]["reviews.jsonl"]:
        raise ValueError("Independent review hash mismatch")
    reviews = pd.DataFrame(json.loads(line) for line in review_path.read_text().splitlines())
    borderline_ids = set(reviews.loc[reviews["verdict"].eq("BORDERLINE"), "query_id"])

    queries = pd.read_parquet(args.pending / "queries.parquet")
    labels = pd.read_parquet(args.pending / "labels.parquet")
    folds = pd.read_parquet(args.pending / "fold_assignments.parquet")
    new = labels["data_origin"].eq("REAL_CRM_20260817")
    certified = new & ~labels["query_id"].astype(str).isin(borderline_ids)
    labels.loc[certified, "validator"] = "DIRECT_IDENTITY_EXACT_SITE_V1_CERTIFIED"
    labels.loc[certified, "reliability"] = "HIGH_AUTOMATED_RULE_CERTIFIED"
    labels.loc[certified, "label_audit_status"] = "RULE_CERTIFIED_FRESH_400_ZERO_FALSE"
    labels.loc[certified, "exact_metric_eligible"] = True
    labels.loc[certified, "identity_training_eligible"] = True
    labels.loc[certified, "operational_training_eligible"] = True
    labels.loc[certified, "ranker_weight"] = labels.loc[certified, "ground_truth_state"].map(
        {"A": 1.0, "F": 0.5}
    )
    borderline = new & labels["query_id"].astype(str).isin(borderline_ids)
    labels.loc[borderline, "label_kind"] = "UNRESOLVED"
    labels.loc[borderline, "reliability"] = "QUARANTINED"
    labels.loc[borderline, "label_audit_status"] = "SAMPLED_BORDERLINE_QUARANTINE"
    labels.loc[borderline, "exact_metric_eligible"] = False
    labels.loc[borderline, "identity_training_eligible"] = False
    labels.loc[borderline, "operational_training_eligible"] = False
    labels.loc[borderline, "ranker_weight"] = 0.0
    labels.loc[borderline, "acceptor_weight"] = 0.0
    labels.loc[borderline, "ground_truth_siret"] = ""
    labels.loc[borderline, "ground_truth_siren"] = ""
    labels.loc[borderline, "ground_truth_siret_exact"] = ""
    labels.loc[borderline, "acceptable_sirets_operational"] = "[]"
    if labels.loc[new, "label_audit_status"].str.startswith("PENDING").any():
        raise AssertionError("No pending new label may be published")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "pending_manifest": sha256(args.pending / "manifest.json"),
            "review_manifest": sha256(args.review / "manifest.json"),
            "reviews": sha256(review_path),
        },
        "retrieval_inputs_used": False,
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        return destination
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    queries.to_parquet(temporary / "queries.parquet", index=False)
    labels.to_parquet(temporary / "labels.parquet", index=False)
    folds.to_parquet(temporary / "fold_assignments.parquet", index=False)
    shutil.copy2(args.pending / "crm_ok_gt_v2_sanitized.csv", temporary / "crm_ok_gt_v2_sanitized.csv")
    shutil.copy2(args.pending / "independent_audit_sample_400.csv", temporary / "independent_audit_sample_400.csv")
    shutil.copy2(review_path, temporary / "independent_reviews_400.jsonl")
    counts = {
        "population_rows": len(labels),
        "historical_rows": int((~new).sum()),
        "new_exact_certified_rows": int(certified.sum()),
        "new_sampled_borderline_quarantined_rows": int(borderline.sum()),
        "new_train_rows": int((certified & labels["split_role"].eq("TRAIN")).sum()),
        "new_prospective_dev_rows": int((certified & labels["split_role"].eq("PROSPECTIVE_DEV")).sum()),
        "new_prospective_test_rows": int((certified & labels["split_role"].eq("PROSPECTIVE_TEST")).sum()),
    }
    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "audit_gate": {
            "status": "PASS",
            "rows": 400,
            "pass": int(review_manifest["counts"]["PASS"]),
            "borderline": int(review_manifest["counts"]["BORDERLINE"]),
            "certain_false_labels": 0,
        },
        "consumer_gate": {"new_rows_training_eligible": True},
    }
    output_names = [
        "queries.parquet", "labels.parquet", "fold_assignments.parquet",
        "crm_ok_gt_v2_sanitized.csv", "independent_audit_sample_400.csv",
        "independent_reviews_400.jsonl",
    ]
    manifest["outputs"] = {name: sha256(temporary / name) for name in output_names}
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(build(parser.parse_args()))


if __name__ == "__main__":
    main()
