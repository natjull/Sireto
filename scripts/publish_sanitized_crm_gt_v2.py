#!/usr/bin/env python3
"""Publish a component-stable CRM GT population from strictly admitted rows."""
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

from scripts.build_crm_gt_v2_population import audit_sample, sha256


SCHEMA_VERSION = "sireto-crm-gt-v2-sanitized-population-1"


def _verified_population(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    frames = []
    for name in ("queries.parquet", "labels.parquet", "fold_assignments.parquet"):
        path = root / name
        if sha256(path) != manifest["outputs"][name]:
            raise ValueError(f"Manifest mismatch: {path}")
        frames.append(pd.read_parquet(path))
    return (*frames, manifest)


def _fresh_audit_rows(
    admitted: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    excluded: set[str],
    seed: int,
) -> pd.DataFrame:
    rows = admitted.copy()
    rows["query_id"] = rows["crm_gt_fingerprint"].map(lambda value: f"NEWCRM:{value[:32]}")
    meta = folds[["query_id", "siren_component_id", "oof_fold", "split_role"]]
    label_meta = labels[["query_id", "acceptable_sirets_operational"]]
    rows = rows.merge(meta, on="query_id", validate="one_to_one")
    rows = rows.merge(label_meta, on="query_id", validate="one_to_one")
    return audit_sample(rows, seed, excluded)


def build(args: argparse.Namespace) -> Path:
    queries, labels, folds, preliminary_manifest = _verified_population(args.preliminary)
    admitted = pd.read_csv(args.admitted, sep=";", dtype=str, keep_default_na=False)
    admitted_ids = set(
        admitted["crm_gt_fingerprint"].map(lambda value: f"NEWCRM:{value[:32]}")
    )
    new_mask = labels["data_origin"].eq("REAL_CRM_20260817")
    historical_ids = set(labels.loc[~new_mask, "query_id"].astype(str))
    preliminary_ids = set(labels["query_id"].astype(str))
    model_admitted_ids = admitted_ids & preliminary_ids
    retained_ids = historical_ids | model_admitted_ids
    admitted = admitted[
        admitted["crm_gt_fingerprint"].map(lambda value: f"NEWCRM:{value[:32]}").isin(model_admitted_ids)
    ].copy()

    out_queries = queries[queries["query_id"].astype(str).isin(retained_ids)].copy()
    out_labels = labels[labels["query_id"].astype(str).isin(retained_ids)].copy()
    out_folds = folds[folds["query_id"].astype(str).isin(retained_ids)].copy()
    admitted_mask = out_labels["query_id"].astype(str).isin(model_admitted_ids)
    out_labels.loc[admitted_mask, "validator"] = "DIRECT_IDENTITY_EXACT_SITE_V1"
    out_labels.loc[admitted_mask, "reliability"] = "PENDING_INDEPENDENT_AUDIT"
    out_labels.loc[admitted_mask, "label_audit_status"] = "PENDING_FRESH_INDEPENDENT_REVIEW"
    out_labels.loc[admitted_mask, "exact_metric_eligible"] = False
    out_labels.loc[admitted_mask, "identity_training_eligible"] = False
    out_labels.loc[admitted_mask, "operational_training_eligible"] = False
    out_labels.loc[admitted_mask, "ranker_weight"] = 0.0
    out_labels.loc[admitted_mask, "acceptor_weight"] = 0.0
    if out_folds.groupby("siren_component_id")["oof_fold"].nunique().max() != 1:
        raise AssertionError("Component leakage after sanitization")

    excluded = set(
        pd.read_csv(args.prior_audit_sample, dtype=str, keep_default_na=False)["query_id"]
    )
    audit = _fresh_audit_rows(admitted, out_labels, out_folds, excluded, args.audit_seed)
    if set(audit["query_id"]) & excluded:
        raise AssertionError("Fresh audit overlaps the failed preregistered sample")

    existing = pd.read_csv(args.existing_crm, sep=";", dtype=str, keep_default_na=False)
    core = [
        "crm_name", "crm_cp", "crm_insee", "crm_id", "crm_commune", "gt_siret",
        "crm_adresse", "SITE_CLI_COMMUNE", "sirene_insee", "sirene_cp", "sirene_etat",
        "loc_match_type",
    ]
    combined = pd.concat([existing[core], admitted[core]], ignore_index=True)

    identity = {
        "schema_version": SCHEMA_VERSION,
        "audit_seed": args.audit_seed,
        "inputs": {
            "preliminary_manifest": sha256(args.preliminary / "manifest.json"),
            "admitted": sha256(args.admitted),
            "existing_crm": sha256(args.existing_crm),
            "prior_audit_sample": sha256(args.prior_audit_sample),
        },
        "retrieval_inputs_used": False,
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        return destination
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    out_queries.to_parquet(temporary / "queries.parquet", index=False)
    out_labels.to_parquet(temporary / "labels.parquet", index=False)
    out_folds.to_parquet(temporary / "fold_assignments.parquet", index=False)
    combined.to_csv(temporary / "crm_ok_gt_v2_sanitized.csv", sep=";", index=False)
    audit.to_csv(temporary / "independent_audit_sample_400.csv", index=False)
    counts = {
        "historical_model_rows": len(historical_ids),
        "admitted_new_rows": len(admitted_ids),
        "model_population_rows": len(out_labels),
        "crm_rows": len(combined),
        "audit_rows": len(audit),
        "new_split_roles": {
            str(key): int(value)
            for key, value in out_folds[
                out_folds["query_id"].astype(str).isin(model_admitted_ids)
            ]["split_role"].value_counts().items()
        },
        "strict_rows_excluded_by_preliminary_component_quarantine": len(admitted_ids - model_admitted_ids),
    }
    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "audit_gate": {
            "status": "PENDING_FRESH_INDEPENDENT_REVIEW",
            "required_rows": 400,
            "required_certain_false_labels": 0,
        },
        "consumer_gate": {
            "new_rows_training_eligible": False,
            "reason": "FRESH_INDEPENDENT_REVIEW_REQUIRED",
        },
    }
    outputs = [
        "queries.parquet", "labels.parquet", "fold_assignments.parquet",
        "crm_ok_gt_v2_sanitized.csv", "independent_audit_sample_400.csv",
    ]
    manifest["outputs"] = {name: sha256(temporary / name) for name in outputs}
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preliminary", type=Path, required=True)
    parser.add_argument("--admitted", type=Path, required=True)
    parser.add_argument("--existing-crm", type=Path, required=True)
    parser.add_argument("--prior-audit-sample", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-seed", type=int, default=20260818)
    print(build(parser.parse_args()))


if __name__ == "__main__":
    main()
