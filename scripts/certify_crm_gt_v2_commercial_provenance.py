#!/usr/bin/env python3
"""Certify fresh CRM labels from their commercial-entry provenance.

The SIRET was entered by a commercial assistant when the CRM site was created.
The upstream population builder already verifies that the SIRET exists in the
current SIRENE snapshot and that the CRM location agrees through INSEE, with
postcode fallback.  This publisher records that contract without attempting to
re-adjudicate identity from current SIRENE names or addresses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_crm_gt_v2_population import sha256


SCHEMA_VERSION = "sireto-crm-gt-v2-commercial-provenance-1"
ARTIFACTS = (
    "queries.parquet",
    "labels.parquet",
    "fold_assignments.parquet",
    "crm_ok_gt_v2.csv",
    "independent_audit_sample_400.csv",
)


def build(preliminary: Path, output_root: Path) -> Path:
    source_manifest_path = preliminary / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    for name, expected in source_manifest["outputs"].items():
        path = preliminary / name
        if not path.exists() or sha256(path) != expected:
            raise ValueError(f"Preliminary population hash mismatch: {name}")

    labels = pd.read_parquet(preliminary / "labels.parquet")
    fresh = labels["data_origin"].eq("REAL_CRM_20260817")
    historical = labels["data_origin"].eq("REAL_CRM_HISTORICAL")
    if not fresh.any():
        raise ValueError("No REAL_CRM_20260817 labels")
    if not labels.loc[fresh, "exact_metric_eligible"].astype(bool).all():
        raise ValueError("Fresh population contains non-eligible labels")

    labels.loc[fresh, "label_source"] = "COMMERCIAL_ASSISTANT_ENTERED_CRM_20260817"
    labels.loc[fresh, "validator"] = "COMMERCIAL_ENTRY_PLUS_SIRENE_INSEE_CP_V1"
    labels.loc[fresh, "reliability"] = "HIGH_HUMAN_ENTERED_GEO_GUARDED"
    labels.loc[fresh, "label_is_human_validated"] = True
    labels.loc[fresh, "label_audit_status"] = (
        "PROVENANCE_CONFIRMED_COMMERCIAL_ENTRY_GEO_GUARDED"
    )
    labels.loc[historical, "label_source"] = "HISTORICAL_HUMAN_ENTERED_CRM"
    labels.loc[historical, "validator"] = "HISTORICAL_CRM_LABEL_CONTRACT"
    labels.loc[historical, "reliability"] = "HIGH_HUMAN_ENTERED_CRM"
    labels.loc[historical, "label_is_human_validated"] = True
    labels.loc[historical, "label_audit_status"] = (
        "PROVENANCE_CONFIRMED_HISTORICAL_HUMAN_CRM"
    )
    labels["human_label_provenance"] = labels["data_origin"].map(
        {
            "REAL_CRM_20260817": "COMMERCIAL_ASSISTANT_SITE_CREATION_20260817",
            "REAL_CRM_HISTORICAL": "HISTORICAL_CRM_SIRET_ENTRY",
        }
    )
    if labels["human_label_provenance"].isna().any():
        raise ValueError("Unexpected non-CRM label in commercial population")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": sha256(Path(__file__)),
        "preliminary_manifest_sha256": sha256(source_manifest_path),
        "label_contract": {
            "source": "SIRET_ENTERED_BY_COMMERCIAL_ASSISTANT_AT_CRM_SITE_CREATION",
            "deterministic_guards": [
                "SIRET_EXISTS_IN_PINNED_SIRENE",
                "CRM_AND_SIRENE_INSEE_MATCH",
                "POSTCODE_FALLBACK_ONLY_WHEN_INSEE_UNAVAILABLE",
                "SIREN_COMPONENT_SPLITS_PRESERVED",
            ],
            "llm_review_role": "DIAGNOSTIC_SUSPICION_ONLY_NOT_LABEL_ORACLE",
        },
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:16]
    destination = output_root / build_id
    if destination.exists():
        return destination
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=output_root))

    for name in ARTIFACTS:
        if name == "labels.parquet":
            labels.to_parquet(temporary / name, index=False)
        else:
            shutil.copy2(preliminary / name, temporary / name)

    fresh_labels = labels[fresh]
    counts = {
        "model_population_rows": len(labels),
        "historical_human_gt_rows": int(historical.sum()),
        "fresh_commercial_gt_rows": len(fresh_labels),
        "fresh_train_rows": int(fresh_labels["split_role"].eq("TRAIN").sum()),
        "fresh_prospective_dev_rows": int(
            fresh_labels["split_role"].eq("PROSPECTIVE_DEV").sum()
        ),
        "fresh_prospective_test_rows": int(
            fresh_labels["split_role"].eq("PROSPECTIVE_TEST").sum()
        ),
    }
    report = (
        "# CRM GT v2 — provenance commerciale\n\n"
        f"- Nouveaux labels CRM admis : **{len(fresh_labels)}**\n"
        "- Source du label : SIRET saisi par l'assistant commercial lors de "
        "la création du site CRM.\n"
        "- Gardes : existence SIRENE et cohérence commune INSEE/CP.\n"
        "- Les revues LLM restent des signaux de suspicion et ne peuvent pas "
        "annuler seules un label commercial.\n"
        "- Les conflits de composantes SIREN restent exclus en amont pour "
        "éviter toute fuite entre folds.\n"
    )
    (temporary / "report.md").write_text(report)
    outputs = {
        name: sha256(temporary / name)
        for name in (*ARTIFACTS, "report.md")
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_identity": identity,
        "counts": counts,
        "audit_gate": {
            "status": "PASS",
            "basis": "HUMAN_COMMERCIAL_ENTRY_PLUS_SIRENE_GEO_GUARDS",
            "known_label_errors": "EXPECTED_TO_BE_MARGINAL_AND_AUDITED_SEPARATELY",
        },
        "qualification": {
            "retrieval_inputs_used": False,
            "model_scores_used": False,
            "llm_decisions_used_for_admission": False,
        },
        "outputs": outputs,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preliminary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.preliminary, args.output_root))


if __name__ == "__main__":
    main()
