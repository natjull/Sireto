#!/usr/bin/env python3
"""Freeze the V4.12 trusted-label ranker/acceptor bundle before a new holdout."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v411_ranker_c_development import RANKER_C_FEATURE_ORDER
from src.xgb_matcher.v411_scene import V411_ACCEPTOR_FEATURE_NAMES


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
RANKER = BASE / "experiments/v4_12_trusted_label_ranker/2f57628196fefce0/ranker_candidate.json"
ACCEPTOR_ROOT = BASE / "experiments/v4_12_acceptor_conservative/88e50a879d7fcc2b"
ACCEPTOR = ACCEPTOR_ROOT / "acceptor_candidate.joblib"
EVALUATION = ACCEPTOR_ROOT / "evaluation.json"
DATASET_MANIFEST = BASE / "datasets/v4_11_input_blind/ec4326ec57e4411d/manifest.json"
TRUSTED_LABELS = Path("reports/v412_review_trusted_labels_279.csv")
TAXONOMY = Path("config/v4_9_site_function_taxonomy.json")
OUTPUT_ROOT = BASE / "bundles/v4_12_trusted"
SCHEMA_VERSION = "sireto-v4.12-trusted-model-bundle-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> Path:
    evaluation = json.loads(EVALUATION.read_text())
    if evaluation.get("final_test_opened") is not False:
        raise ValueError("Final test state is not closed")
    policy = evaluation["fixed_policy"]
    if policy != {
        "family": "MONOTONIC_XGB",
        "threshold": 0.9886879324913025,
        "trusted_weight": 10.0,
    }:
        raise ValueError("Conservative policy changed")
    ranker = xgb.XGBRanker()
    ranker.load_model(RANKER)
    acceptor = joblib.load(ACCEPTOR)
    if not hasattr(acceptor, "predict_proba"):
        raise ValueError("Acceptor cannot score probabilities")

    inputs = {
        "ranker_model": sha256(RANKER),
        "acceptor_model": sha256(ACCEPTOR),
        "acceptor_evaluation": sha256(EVALUATION),
        "dataset_manifest": sha256(DATASET_MANIFEST),
        "trusted_labels": sha256(TRUSTED_LABELS),
        "taxonomy": sha256(TAXONOMY),
        "ranker_source": sha256(Path("scripts/evaluate_v412_trusted_label_ranker.py")),
        "acceptor_source": sha256(Path("scripts/evaluate_v412_acceptor_conservative.py")),
        "scene_source": sha256(Path("src/xgb_matcher/v411_scene.py")),
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "inputs": inputs,
                "policy": policy,
                "ranker_features": RANKER_C_FEATURE_ORDER,
                "acceptor_features": V411_ACCEPTOR_FEATURE_NAMES,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = OUTPUT_ROOT / identity
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(RANKER, output / "ranker.json")
    shutil.copy2(ACCEPTOR, output / "acceptor.joblib")
    shutil.copy2(TAXONOMY, output / "site_function_taxonomy.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_policy": {
            "auto_decision": "AUTO_MATCH",
            "fallback_decision": "REVIEW",
            "acceptor_threshold": policy["threshold"],
            "auto_no_match_enabled": False,
        },
        "retrieval_contract": {
            "version": "V4.2",
            "max_candidates": 100,
            "positive_injection": False,
        },
        "ranker_feature_order": RANKER_C_FEATURE_ORDER,
        "acceptor_feature_order": V411_ACCEPTOR_FEATURE_NAMES,
        "inputs": inputs,
        "artifacts": {
            "ranker.json": sha256(output / "ranker.json"),
            "acceptor.joblib": sha256(output / "acceptor.joblib"),
            "site_function_taxonomy.json": sha256(output / "site_function_taxonomy.json"),
        },
        "development_gate": evaluation["combined_dev_projection"],
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "next_required_evidence": "NEW_INDEPENDENT_CRM_HOLDOUT",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return output


if __name__ == "__main__":
    print(main())
