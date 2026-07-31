#!/usr/bin/env python3
"""Conservative final dev calibration of the fixed trusted-label acceptor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_acceptor_hard_weight import _fit, _scores
from scripts.run_v411_acceptor_development import decision_metrics, select_threshold
from src.xgb_matcher.v411_acceptor import MONOTONIC_XGB


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_SCENES = (
    BASE
    / "experiments/v4_12_trusted_acceptor/7bde8fd021ec1915/acceptor_scenes.parquet"
)
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_acceptor_conservative"
DEFAULT_TRUSTED = Path("reports/v412_review_trusted_labels_279.csv")
SCHEMA_VERSION = "sireto-v4.12-acceptor-conservative-development-1"
TRUSTED_WEIGHT = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    scenes = pd.read_parquet(args.scenes)
    scenes["query_id"] = scenes["query_id"].astype(str)
    trusted_labels = pd.read_csv(args.trusted_labels, dtype=str).fillna("")
    trusted_ids = set(trusted_labels["query_id"].astype(str))
    base = scenes[
        scenes["split"].eq("fit") & scenes["label_kind"].eq("MATCH_EXACT")
    ].copy()
    trusted = scenes[scenes["query_id"].isin(trusted_ids)].copy()
    if (
        len(base) != 4666
        or len(trusted) != 279
        or trusted["acceptor_target"].value_counts().to_dict() != {1: 216, 0: 63}
    ):
        raise ValueError("Conservative calibration populations changed")

    parts: list[pd.DataFrame] = []
    score_parts: list[np.ndarray] = []
    for fold in range(5):
        fold_base = base[base["oof_fold"].astype(int).ne(fold)]
        fold_trusted = trusted[trusted["oof_fold"].astype(int).ne(fold)]
        model = _fit(
            pd.concat([fold_base, fold_trusted], ignore_index=True),
            set(fold_trusted["query_id"].astype(str)),
            TRUSTED_WEIGHT,
            MONOTONIC_XGB,
        )
        held = trusted[trusted["oof_fold"].astype(int).eq(fold)].copy()
        parts.append(held)
        score_parts.append(_scores(model, held))
    trusted_oof = pd.concat(parts, ignore_index=True)
    trusted_oof["acceptor_score"] = np.concatenate(score_parts)
    selected = select_threshold(
        trusted_oof["acceptor_score"].to_numpy(),
        trusted_oof["acceptor_target"].astype(int).to_numpy(),
        trusted_oof["label_kind"].astype(str).to_numpy(),
    )
    if selected is None:
        raise ValueError("No conservative OOF threshold")
    threshold, trusted_metrics, _ = selected
    if trusted_metrics["error_auto"] or trusted_metrics["ambiguous_auto"]:
        raise ValueError("Conservative threshold is not safe on consumed OOF")

    final_model = _fit(
        pd.concat([base, trusted], ignore_index=True),
        trusted_ids,
        TRUSTED_WEIGHT,
        MONOTONIC_XGB,
    )
    trusted_components = set(trusted["siren_component_id"].astype(str))
    controls = scenes[
        scenes["split"].eq("dev")
        & scenes["label_kind"].eq("MATCH_EXACT")
        & ~scenes["query_id"].isin(trusted_ids)
        & ~scenes["siren_component_id"].astype(str).isin(trusted_components)
    ].copy()
    if len(controls) != 1127 or not controls["acceptor_target"].eq(1).all():
        raise ValueError("Independent positive control population changed")
    controls["acceptor_score"] = _scores(final_model, controls)
    control_metrics = decision_metrics(
        controls["acceptor_score"].to_numpy(),
        controls["acceptor_target"].astype(int).to_numpy(),
        controls["label_kind"].astype(str).to_numpy(),
        threshold,
    )

    detail = pd.concat(
        [
            trusted_oof.assign(evaluation_role="TRUSTED_OOF_CONSUMED"),
            controls.assign(evaluation_role="NON_TRUSTED_DEV_POSITIVE_CONTROL"),
        ],
        ignore_index=True,
    )
    detail["decision"] = np.where(
        detail["acceptor_score"].ge(threshold), "AUTO_MATCH", "REVIEW"
    )
    combined_metrics = decision_metrics(
        detail["acceptor_score"].to_numpy(),
        detail["acceptor_target"].astype(int).to_numpy(),
        detail["label_kind"].astype(str).to_numpy(),
        threshold,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_CALIBRATION_PLUS_POSITIVE_CONTROL",
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "inputs": {
            "scenes_sha256": _sha256(args.scenes),
            "trusted_labels_sha256": _sha256(args.trusted_labels),
        },
        "fixed_policy": {
            "family": MONOTONIC_XGB,
            "trusted_weight": TRUSTED_WEIGHT,
            "threshold": threshold,
        },
        "trusted_oof_consumed": trusted_metrics,
        "non_trusted_dev_positive_control": control_metrics,
        "combined_dev_projection": combined_metrics,
        "limitations": {
            "trusted_oof_used_to_select_threshold": True,
            "positive_control_contains_no_negative_cases": True,
            "precision_99_8_certified": False,
        },
    }
    result["verdict"] = (
        "GO_FREEZE_FOR_ONE_FINAL_TEST"
        if combined_metrics["coverage"] >= 0.80
        and combined_metrics["error_auto"] == 0
        and combined_metrics["ambiguous_auto"] == 0
        else "PIVOT_COVERAGE"
    )
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "scenes": result["inputs"]["scenes_sha256"],
                "trusted": result["inputs"]["trusted_labels_sha256"],
                "family": MONOTONIC_XGB,
                "weight": TRUSTED_WEIGHT,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    detail[
        [
            "query_id",
            "evaluation_role",
            "label_kind",
            "ground_truth_siret",
            "predicted_siret",
            "acceptor_target",
            "acceptor_score",
            "decision",
        ]
    ].to_parquet(output / "development_decisions.parquet", index=False)
    joblib.dump(final_model, output / "acceptor_candidate.joblib")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--trusted-labels", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
