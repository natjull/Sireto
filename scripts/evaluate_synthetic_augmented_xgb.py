#!/usr/bin/env python3
"""Paired fold-0 evaluation of real-only versus real+synthetic XGBoost."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v412_learned_business_features import BUSINESS_FEATURE_ORDER  # noqa: E402
from scripts.evaluate_v412_bge_operational_secondary import _same_site_evidence  # noqa: E402
from scripts.train_v412_learned_oof_rankers import (  # noqa: E402
    RANKER_PARAMS,
    _fit,
    _score,
    _training_targets,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_REAL_BUSINESS = BASE / "datasets/v4_12_learned_business_features/8800ef53f6927215"
DEFAULT_REAL_TEXT = BASE / "datasets/v4_12_neural_text_corpus/02b8668f8050c5e9"
DEFAULT_PLAN = Path("config/synthetic_augmented_model_eval_v1.json")
DEFAULT_OUTPUT_ROOT = BASE / "experiments/synthetic_augmented_xgb_v1"
SCHEMA_VERSION = "sireto-synthetic-augmented-xgb-evaluation-1"
TRAIN_FOLDS = (2, 3, 4)
TARGET_FOLD = 0


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verified(root: Path, names: tuple[str, ...]) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name in names:
        if manifest.get("outputs", {}).get(name) != file_sha256(root / name):
            raise ValueError(f"Manifest mismatch: {root / name}")
    return manifest


def _metrics(detail: pd.DataFrame, correct_column: str) -> list[dict[str, Any]]:
    masks = {
        "exact": pd.Series(True, index=detail.index),
        "difficult": detail["label_is_human_validated"].astype(bool),
        "active": detail["ground_truth_state"].eq("A"),
        "closed": detail["ground_truth_state"].eq("F"),
    }
    return [
        {
            "segment": name,
            "correct": int(detail.loc[mask, correct_column].sum()),
            "total": int(mask.sum()),
            "hit_at_1": float(detail.loc[mask, correct_column].mean()) if mask.any() else None,
        }
        for name, mask in masks.items()
    ]


def _operational_correct(
    *,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    real_business: Path,
    real_text: Path,
) -> pd.Series:
    query_ids = set(labels["query_id"].astype(str))
    queries = pd.read_parquet(real_business / "queries.parquet")
    queries["query_id"] = queries["query_id"].astype(str)
    query_by_id = queries[queries["query_id"].isin(query_ids)].set_index("query_id")
    texts = pd.read_parquet(
        real_text / "candidates_text.parquet",
        columns=["query_id", "candidate_siret", "candidate_siren", "candidate_text"],
    )
    texts["query_id"] = texts["query_id"].astype(str)
    texts["candidate_siret"] = texts["candidate_siret"].astype(str).str.zfill(14)
    texts["candidate_siren"] = texts["candidate_siren"].astype(str).str.zfill(9)
    texts = texts[texts["query_id"].isin(query_ids)]
    text_by_key = texts.set_index(["query_id", "candidate_siret"])
    label_by_id = labels.set_index("query_id")
    output: list[bool] = []
    for row in predictions.itertuples(index=False):
        query_id = str(row.query_id)
        predicted = str(row.candidate_siret).zfill(14)
        truth = label_by_id.loc[query_id]
        if predicted == str(truth.ground_truth_siret).zfill(14):
            output.append(True)
            continue
        key = (query_id, predicted)
        if key not in text_by_key.index:
            output.append(False)
            continue
        candidate = text_by_key.loc[key]
        same_siren = str(candidate.candidate_siren).zfill(9) == str(truth.ground_truth_siren).zfill(9)
        evidence = _same_site_evidence(query_by_id.loc[query_id], candidate.candidate_text)
        output.append(bool(same_siren and evidence is not None))
    return pd.Series(output, index=predictions.index, dtype=bool)


def run(args: argparse.Namespace) -> Path:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("fold_roles", {}).get("train") != list(TRAIN_FOLDS):
        raise ValueError("Experiment plan does not freeze train folds 2/3/4")
    if plan.get("fold_roles", {}).get("development") != TARGET_FOLD:
        raise ValueError("Experiment plan does not freeze development fold 0")
    real_manifest = _verified(
        args.real_business, ("candidates_business.parquet", "labels.parquet", "queries.parquet")
    )
    _verified(args.real_text, ("candidates_text.parquet", "labels.parquet"))
    mix_manifest = _verified(
        args.mix,
        ("synthetic_labels_selected.parquet", "synthetic_candidates_business_selected.parquet"),
    )
    if mix_manifest.get("schema_version") != "sireto-synthetic-augmented-model-mix-1":
        raise ValueError("Unexpected synthetic mix schema")
    if mix_manifest.get("positive_injection") is not False:
        raise ValueError("Synthetic mix is not certified non-injected")
    if any(
        mix_manifest.get(key) is not False
        for key in ("risk_model_allowed", "calibration_allowed", "auto_threshold_selection_allowed")
    ):
        raise ValueError("Synthetic mix authorizes a forbidden downstream consumer")
    if list(real_manifest.get("business_feature_order", [])) != BUSINESS_FEATURE_ORDER:
        raise ValueError("Real BUSINESS feature order changed")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "plan_sha256": file_sha256(args.plan),
        "real_business_manifest_sha256": file_sha256(args.real_business / "manifest.json"),
        "real_text_manifest_sha256": file_sha256(args.real_text / "manifest.json"),
        "mix_manifest_sha256": file_sha256(args.mix / "manifest.json"),
        "features": BUSINESS_FEATURE_ORDER,
        "ranker_params": RANKER_PARAMS,
        "train_folds": list(TRAIN_FOLDS),
        "development_fold": TARGET_FOLD,
        "arms": ["REAL_ONLY", "REAL_PLUS_SYNTHETIC"],
        "positive_injection": False,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    real_candidates = pd.read_parquet(args.real_business / "candidates_business.parquet")
    real_labels = pd.read_parquet(args.real_business / "labels.parquet")
    for frame in (real_candidates, real_labels):
        frame["query_id"] = frame["query_id"].astype(str)
    real_candidates["candidate_siret"] = real_candidates["candidate_siret"].astype(str).str.zfill(14)
    real_labels["ground_truth_siret"] = real_labels["ground_truth_siret"].astype("string").str.zfill(14)
    real_targets, diagnostics = _training_targets(
        real_candidates,
        real_labels,
        include_weak_open_labels=False,
        human_weight_multiplier=1.0,
    )
    real_train = real_targets[real_targets["oof_fold"].astype(int).isin(TRAIN_FOLDS)].copy()

    synthetic_candidates = pd.read_parquet(
        args.mix / "synthetic_candidates_business_selected.parquet"
    )
    synthetic_labels = pd.read_parquet(args.mix / "synthetic_labels_selected.parquet")
    for frame in (synthetic_candidates, synthetic_labels):
        frame["query_id"] = frame["query_id"].astype(str)
    synthetic_candidates["candidate_siret"] = synthetic_candidates["candidate_siret"].astype(str).str.zfill(14)
    synthetic_labels["ground_truth_siret"] = synthetic_labels["ground_truth_siret"].astype(str).str.zfill(14)
    synthetic_train = synthetic_candidates.merge(
        synthetic_labels[
            ["query_id", "ground_truth_siret", "oof_fold", "scene_weight"]
        ],
        on="query_id",
        validate="many_to_one",
    )
    synthetic_train["training_positive"] = synthetic_train["candidate_siret"].eq(
        synthetic_train["ground_truth_siret"]
    ).astype(np.int8)
    synthetic_train["query_weight"] = synthetic_train["scene_weight"].astype(np.float32)
    if synthetic_train.groupby("query_id")["training_positive"].sum().ne(1).any():
        raise ValueError("A selected synthetic XGBoost scene lacks one natural positive")
    if not set(synthetic_train["oof_fold"].astype(int)).issubset(TRAIN_FOLDS):
        raise ValueError("Synthetic XGBoost training escaped folds 2/3/4")
    augmented_train = pd.concat([real_train, synthetic_train], ignore_index=True, sort=False)
    if not np.isfinite(augmented_train[BUSINESS_FEATURE_ORDER].to_numpy(dtype=np.float32)).all():
        raise ValueError("Augmented XGBoost matrix contains non-finite values")

    target_ids = set(
        real_labels.loc[
            real_labels["oof_fold"].astype(int).eq(TARGET_FOLD)
            & real_labels["label_kind"].eq("MATCH_EXACT"),
            "query_id",
        ]
    )
    target_candidates = real_candidates[real_candidates["query_id"].isin(target_ids)].copy()
    target_labels = real_labels[real_labels["query_id"].isin(target_ids)].copy()
    if set(target_candidates["query_id"]) != target_ids:
        raise ValueError("A development exact query has no candidate pool")

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    arm_results: dict[str, Any] = {}
    arm_details: list[pd.DataFrame] = []
    try:
        for arm, train in (("REAL_ONLY", real_train), ("REAL_PLUS_SYNTHETIC", augmented_train)):
            started = time.perf_counter()
            model, training = _fit(train, BUSINESS_FEATURE_ORDER, negative_limit=0)
            fit_seconds = time.perf_counter() - started
            scored = _score(
                model,
                target_candidates,
                BUSINESS_FEATURE_ORDER,
                fold=TARGET_FOLD,
                variant=arm,
            )
            top1 = scored[scored["ranker_rank"].eq(1)].copy()
            detail = target_labels.merge(
                top1[["query_id", "candidate_siret", "candidate_siren", "ranker_score"]],
                on="query_id", how="left", validate="one_to_one",
            )
            detail["exact_siret_correct"] = detail["candidate_siret"].eq(
                detail["ground_truth_siret"]
            ).fillna(False)
            detail["operational_siret_correct"] = _operational_correct(
                predictions=detail[["query_id", "candidate_siret"]],
                labels=target_labels,
                real_business=args.real_business,
                real_text=args.real_text,
            )
            detail["arm"] = arm
            exact_metrics = _metrics(detail, "exact_siret_correct")
            operational_metrics = _metrics(detail, "operational_siret_correct")
            arm_results[arm] = {
                "training": training,
                "fit_seconds": fit_seconds,
                "exact_metrics": exact_metrics,
                "operational_metrics_secondary": operational_metrics,
            }
            model.save_model(temporary / f"{arm.lower()}_model.json")
            scored.to_parquet(temporary / f"{arm.lower()}_fold0_candidates.parquet", index=False)
            arm_details.append(detail)

        details = pd.concat(arm_details, ignore_index=True)
        pivot = details.pivot(index="query_id", columns="arm", values="exact_siret_correct")
        matrix = {
            "both_correct": int((pivot["REAL_ONLY"] & pivot["REAL_PLUS_SYNTHETIC"]).sum()),
            "real_only_correct": int((pivot["REAL_ONLY"] & ~pivot["REAL_PLUS_SYNTHETIC"]).sum()),
            "synthetic_only_correct": int((~pivot["REAL_ONLY"] & pivot["REAL_PLUS_SYNTHETIC"]).sum()),
            "both_wrong": int((~pivot["REAL_ONLY"] & ~pivot["REAL_PLUS_SYNTHETIC"]).sum()),
        }
        base = {row["segment"]: row for row in arm_results["REAL_ONLY"]["exact_metrics"]}
        candidate = {
            row["segment"]: row for row in arm_results["REAL_PLUS_SYNTHETIC"]["exact_metrics"]
        }
        frozen_gate = plan["development_gate"]
        gate = {
            "exact_gain_at_least_10": candidate["exact"]["correct"] - base["exact"]["correct"]
            >= int(frozen_gate["minimum_exact_gain_over_paired_control"]),
            "exact_at_least_2452": candidate["exact"]["correct"] >= int(frozen_gate["minimum_exact_correct"]),
            "difficult_at_least_33": candidate["difficult"]["correct"] >= int(frozen_gate["minimum_difficult_correct"]),
            "active_at_least_2164": candidate["active"]["correct"] >= int(frozen_gate["minimum_active_correct"]),
            "closed_at_least_246": candidate["closed"]["correct"] >= int(frozen_gate["minimum_closed_correct"]),
            "evaluation_real_fold0_only": True,
            "confirmation_fold_closed": True,
            "final_test_closed": True,
            "positive_injection": False,
        }
        passed = all(gate.values())
        details.to_parquet(temporary / "fold0_top1_detail.parquet", index=False)
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "DEVELOPMENT_FOLD_0_PAIRED",
            "real_training_diagnostics": diagnostics,
            "arms": arm_results,
            "paired_exact_matrix": matrix,
            "gate": gate,
            "gate_passed": passed,
            "verdict": "GO_SYNTHETIC_AUGMENTATION_XGB" if passed else "STOP_SYNTHETIC_AUGMENTATION_XGB",
            "primary_metric": "exact_siret_hit_at_1",
            "operational_metric_is_secondary": True,
            "risk_model_trained": False,
            "calibration_trained": False,
            "auto_threshold_selected": False,
            "confirmation_fold_opened": False,
            "final_test_opened": False,
        }
        _json_dump(temporary / "evaluation.json", evaluation)
        outputs = {
            str(path.relative_to(temporary)): file_sha256(path)
            for path in temporary.rglob("*") if path.is_file()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "positive_injection": False,
            "confirmation_fold_opened": False,
            "final_test_opened": False,
            "outputs": outputs,
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--real-business", type=Path, default=DEFAULT_REAL_BUSINESS)
    parser.add_argument("--real-text", type=Path, default=DEFAULT_REAL_TEXT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
