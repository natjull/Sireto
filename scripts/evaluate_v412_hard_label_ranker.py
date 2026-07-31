#!/usr/bin/env python3
"""Evaluate a ranker augmented with the V4.12 hard labels.

The experiment is development-only.  It trains each hard query out of fold
using the already frozen SIREN-component folds, counts missing positives as
end-to-end errors, and screens the full augmented model for regressions on
the remaining consumed V4.11 development population.  It never reads the
sealed final holdout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v411_ranker_c_development import (
    RANKER_C_FEATURE_ORDER,
    RANKER_PARAMS,
    eligible_ranker_rows,
)


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_DATASET = BASE / "datasets/v4_11_input_blind/ec4326ec57e4411d"
DEFAULT_REFERENCE = BASE / "references/v4_12_service_parity/b4b7fef24c5e7036"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_hard_label_ranker"
SCHEMA_VERSION = "sireto-v4.12-hard-label-ranker-development-2"
EXPECTED_R30_EXACT = 27
EXPECTED_R30_AMBIGUOUS = 3
EXPECTED_R53_EXACT = 50
EXPECTED_R53_AMBIGUOUS = 3
EXPECTED_HARD_EXACT = EXPECTED_R30_EXACT + EXPECTED_R53_EXACT
EXPECTED_HARD_RETRIEVAL_MISSES = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_hard_labels(
    r30_path: Path,
    r53_path: Path,
    corrected_overlay_path: Path | None = None,
) -> tuple[pd.DataFrame, set[str], dict[str, int]]:
    r30 = pd.read_csv(r30_path, dtype=str).fillna("")
    r30_exact = r30[r30["ranking_label_usable"].eq("true")][
        ["query_id", "validated_siret"]
    ].rename(columns={"validated_siret": "ground_truth_siret"})
    r30_ambiguous = int(r30["label"].eq("AMBIGUOUS").sum())
    if len(r30_exact) != EXPECTED_R30_EXACT or r30_ambiguous != EXPECTED_R30_AMBIGUOUS:
        raise ValueError("R30 labels differ from the adjudicated milestone")

    r53 = pd.read_csv(r53_path, dtype=str).fillna("")
    r53_exact = r53[r53["label"].eq("MATCH_EXACT")][
        ["query_id", "adjudication"]
    ].rename(columns={"adjudication": "ground_truth_siret"})
    r53_ambiguous = int(r53["label"].eq("AMBIGUOUS").sum())
    if len(r53_exact) != EXPECTED_R53_EXACT or r53_ambiguous != EXPECTED_R53_AMBIGUOUS:
        raise ValueError("R53 labels differ from the adjudicated milestone")

    exact_parts = [r30_exact, r53_exact]
    all_id_parts = [r30["query_id"].astype(str), r53["query_id"].astype(str)]
    counts = {
        "r30_exact": len(r30_exact),
        "r30_ambiguous": r30_ambiguous,
        "r53_exact": len(r53_exact),
        "r53_ambiguous": r53_ambiguous,
    }
    expected_exact = EXPECTED_HARD_EXACT
    expected_all = 83
    if corrected_overlay_path is not None:
        corrected = pd.read_csv(corrected_overlay_path, dtype=str).fillna("")
        corrected_counts = corrected["label_kind"].value_counts().to_dict()
        if len(corrected) != 60 or corrected_counts != {
            "MATCH_EXACT": 56,
            "AMBIGUOUS": 4,
        }:
            raise ValueError("Corrected REVIEW overlay must contain 56 exact and 4 ambiguous labels")
        corrected_exact = corrected[corrected["label_kind"].eq("MATCH_EXACT")][
            ["query_id", "ground_truth_siret"]
        ]
        exact_parts.append(corrected_exact)
        all_id_parts.append(corrected["query_id"].astype(str))
        counts.update({"corrected_exact": 56, "corrected_ambiguous": 4})
        expected_exact += 56
        expected_all += 60

    labels = pd.concat(exact_parts, ignore_index=True)
    if len(labels) != expected_exact or labels["query_id"].duplicated().any():
        raise ValueError(f"Hard labels must contain {expected_exact} unique exact queries")
    labels["ground_truth_siret"] = labels["ground_truth_siret"].astype(str)
    if not labels["ground_truth_siret"].str.fullmatch(r"\d{14}").all():
        raise ValueError("Every hard exact label must be a 14-digit SIRET")
    labels["ground_truth_siren"] = labels["ground_truth_siret"].str[:9]
    if corrected_overlay_path is None and labels["ground_truth_siren"].duplicated().any():
        raise ValueError("Hard exact labels unexpectedly share a SIREN")
    all_adjudicated_ids = set(pd.concat(all_id_parts, ignore_index=True))
    if len(all_adjudicated_ids) != expected_all:
        raise ValueError(f"Expected {expected_all} distinct adjudicated REVIEW queries")
    return labels, all_adjudicated_ids, counts


def fit_weighted_ranker(
    rows: pd.DataFrame,
    *,
    hard_query_ids: set[str],
    hard_weight: float,
) -> xgb.XGBRanker:
    if not 0.0 < hard_weight <= 1.0:
        raise ValueError("hard_weight must be in ]0, 1]")
    ordered = rows.sort_values(["query_id", "candidate_siret"], kind="mergesort")
    positives = ordered.groupby("query_id", sort=False)["is_ground_truth"].sum()
    if ordered.empty or not positives.eq(1).all():
        raise ValueError("Every weighted ranker group must contain one positive")
    grouped = ordered.groupby("query_id", sort=False)
    groups = grouped.size().to_numpy()
    query_order = list(grouped.indices)
    group_weights = np.asarray(
        [hard_weight if query_id in hard_query_ids else 1.0 for query_id in query_order],
        dtype=np.float32,
    )
    model = xgb.XGBRanker(**RANKER_PARAMS)
    model.fit(
        ordered[RANKER_C_FEATURE_ORDER].to_numpy(dtype=np.float32),
        ordered["is_ground_truth"].to_numpy(dtype=np.int8),
        group=groups,
        sample_weight=group_weights,
        verbose=False,
    )
    return model


def _rank_predictions(model: xgb.XGBRanker, rows: pd.DataFrame) -> pd.DataFrame:
    output = rows[["query_id", "candidate_siret", "candidate_siren", "retrieval_rank"]].copy()
    output["ranker_score"] = model.predict(
        rows[RANKER_C_FEATURE_ORDER].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    output = output.sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    output["ranker_rank"] = output.groupby("query_id", sort=False).cumcount() + 1
    return output.reset_index(drop=True)


def _evaluate_predictions(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    top1 = predictions[predictions["ranker_rank"].eq(1)][
        ["query_id", "candidate_siret", "ranker_score"]
    ].rename(columns={"candidate_siret": "predicted_siret"})
    present = predictions.merge(
        truth[["query_id", "ground_truth_siret"]], on="query_id", how="inner"
    )
    present_ids = set(
        present.loc[
            present["candidate_siret"].eq(present["ground_truth_siret"]), "query_id"
        ].astype(str)
    )
    detail = truth.merge(top1, on="query_id", how="left", validate="one_to_one")
    detail["truth_in_pool"] = detail["query_id"].isin(present_ids)
    detail["top1_correct"] = detail["predicted_siret"].eq(detail["ground_truth_siret"])
    count = len(detail)
    correct = int(detail["top1_correct"].sum())
    return {
        "query_count": count,
        "truth_present_count": int(detail["truth_in_pool"].sum()),
        "retrieval_miss_count": int((~detail["truth_in_pool"]).sum()),
        "top1_correct_count": correct,
        "hit_at_1": correct / count if count else None,
    }, detail


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset.resolve()
    reference = args.reference.resolve()
    hard_labels, all_adjudicated_ids, label_counts = load_hard_labels(
        args.r30_labels, args.r53_labels, args.corrected_overlay
    )
    expected_hard_exact = len(hard_labels)

    assignments = pd.read_parquet(dataset / "split_assignments.parquet")
    labels = pd.read_parquet(dataset / "labels.parquet")
    candidates = pd.read_parquet(dataset / "candidates_sparse_top100.parquet")
    baseline_predictions = pd.read_parquet(reference / "ranker_reference.parquet")
    for frame in (assignments, labels, candidates, baseline_predictions):
        frame["query_id"] = frame["query_id"].astype(str)

    hard_assignments = hard_labels.merge(
        assignments[["query_id", "siren_component_id", "split", "oof_fold"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    if hard_assignments["split"].value_counts().to_dict() != {"dev": expected_hard_exact}:
        raise ValueError("Every hard label must belong to the consumed V4.11 dev split")
    hard_assignments["oof_fold"] = hard_assignments["oof_fold"].astype(int)

    fit_labels = labels.merge(assignments, on="query_id", validate="one_to_one")
    fit_labels = fit_labels[fit_labels["split"].eq("fit")].copy()
    fit_sirens = set(
        fit_labels.loc[fit_labels["label_kind"].eq("MATCH_EXACT"), "ground_truth_siren"]
        .fillna("")
        .astype(str)
    )
    overlap = sorted(set(hard_labels["ground_truth_siren"]) & fit_sirens)
    if overlap:
        raise ValueError(f"Hard-label SIRENs overlap base fit: {overlap[:5]}")

    fit_candidates = candidates[candidates["query_id"].isin(fit_labels["query_id"])].copy()
    base_rows = eligible_ranker_rows(fit_candidates, fit_labels)

    hard_candidates = candidates[candidates["query_id"].isin(hard_labels["query_id"])].copy()
    hard_candidates = hard_candidates.drop(columns=["is_ground_truth"]).merge(
        hard_assignments[
            ["query_id", "ground_truth_siret", "ground_truth_siren", "oof_fold"]
        ],
        on="query_id",
        how="inner",
        validate="many_to_one",
    )
    hard_candidates["is_ground_truth"] = hard_candidates["candidate_siret"].eq(
        hard_candidates["ground_truth_siret"]
    ).astype(np.int8)
    positive_counts = hard_candidates.groupby("query_id")["is_ground_truth"].sum()
    eligible_hard_ids = set(positive_counts[positive_counts.eq(1)].index.astype(str))
    missing_hard_ids = sorted(set(hard_labels["query_id"]) - eligible_hard_ids)
    if len(missing_hard_ids) != EXPECTED_HARD_RETRIEVAL_MISSES:
        raise ValueError(f"Expected two hard retrieval misses, got {missing_hard_ids}")

    oof_parts: list[pd.DataFrame] = []
    fold_training: dict[str, Any] = {}
    for fold in range(5):
        augmentation = hard_candidates[
            hard_candidates["query_id"].isin(eligible_hard_ids)
            & hard_candidates["oof_fold"].ne(fold)
        ].copy()
        train_rows = pd.concat([base_rows, augmentation], ignore_index=True)
        model = fit_weighted_ranker(
            train_rows,
            hard_query_ids=set(augmentation["query_id"].astype(str)),
            hard_weight=args.hard_weight,
        )
        held_out = hard_candidates[hard_candidates["oof_fold"].eq(fold)].copy()
        ranked = _rank_predictions(model, held_out)
        ranked["oof_fold"] = fold
        oof_parts.append(ranked)
        fold_training[str(fold)] = {
            "base_query_count": int(base_rows["query_id"].nunique()),
            "augmented_query_count": int(augmentation["query_id"].nunique()),
            "held_out_query_count": int(held_out["query_id"].nunique()),
        }
    oof_predictions = pd.concat(oof_parts, ignore_index=True)
    if oof_predictions["query_id"].nunique() != expected_hard_exact:
        raise ValueError(f"OOF predictions do not cover all {expected_hard_exact} hard queries")

    hard_truth = hard_assignments[
        ["query_id", "ground_truth_siret", "ground_truth_siren", "oof_fold"]
    ]
    candidate_hard_metrics, candidate_hard_detail = _evaluate_predictions(
        oof_predictions, hard_truth
    )
    baseline_hard_metrics, baseline_hard_detail = _evaluate_predictions(
        baseline_predictions[baseline_predictions["query_id"].isin(hard_truth["query_id"])],
        hard_truth,
    )
    comparison = baseline_hard_detail[
        ["query_id", "predicted_siret", "top1_correct"]
    ].merge(
        candidate_hard_detail[
            ["query_id", "predicted_siret", "top1_correct", "truth_in_pool"]
        ],
        on="query_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    hard_fixed = int(
        ((~comparison["top1_correct_baseline"]) & comparison["top1_correct_candidate"]).sum()
    )
    hard_regressed = int(
        (comparison["top1_correct_baseline"] & (~comparison["top1_correct_candidate"])).sum()
    )

    all_augmentation = hard_candidates[
        hard_candidates["query_id"].isin(eligible_hard_ids)
    ].copy()
    full_model = fit_weighted_ranker(
        pd.concat([base_rows, all_augmentation], ignore_index=True),
        hard_query_ids=set(all_augmentation["query_id"].astype(str)),
        hard_weight=args.hard_weight,
    )

    hard_components = set(hard_assignments["siren_component_id"].astype(str))
    regression_labels = labels.merge(assignments, on="query_id", validate="one_to_one")
    regression_labels = regression_labels[
        regression_labels["split"].eq("dev")
        & regression_labels["label_kind"].eq("MATCH_EXACT")
        & ~regression_labels["query_id"].isin(all_adjudicated_ids)
        & ~regression_labels["siren_component_id"].astype(str).isin(hard_components)
    ][["query_id", "ground_truth_siret", "ground_truth_siren"]]
    regression_candidates = candidates[
        candidates["query_id"].isin(regression_labels["query_id"])
    ].copy()
    full_predictions = _rank_predictions(full_model, regression_candidates)
    candidate_regression_metrics, candidate_regression_detail = _evaluate_predictions(
        full_predictions, regression_labels
    )
    baseline_regression_metrics, baseline_regression_detail = _evaluate_predictions(
        baseline_predictions[
            baseline_predictions["query_id"].isin(regression_labels["query_id"])
        ],
        regression_labels,
    )
    regression_comparison = baseline_regression_detail[
        ["query_id", "predicted_siret", "top1_correct"]
    ].merge(
        candidate_regression_detail[
            ["query_id", "predicted_siret", "top1_correct", "truth_in_pool"]
        ],
        on="query_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    screen_fixed = int(
        ((~regression_comparison["top1_correct_baseline"]) & regression_comparison["top1_correct_candidate"]).sum()
    )
    screen_regressed = int(
        (regression_comparison["top1_correct_baseline"] & (~regression_comparison["top1_correct_candidate"])).sum()
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "model_training_performed": True,
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "inputs": {
            "dataset": str(dataset),
            "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
            "reference": str(reference),
            "r30_labels_sha256": _sha256(args.r30_labels),
            "r53_labels_sha256": _sha256(args.r53_labels),
            "corrected_overlay_sha256": (
                _sha256(args.corrected_overlay)
                if args.corrected_overlay is not None
                else None
            ),
            "label_counts": label_counts,
            "all_adjudicated_query_count": len(all_adjudicated_ids),
        },
        "training": {
            "ranker_params": RANKER_PARAMS,
            "feature_order": RANKER_C_FEATURE_ORDER,
            "base_fit_query_count": int(base_rows["query_id"].nunique()),
            "eligible_hard_query_count": len(eligible_hard_ids),
            "hard_retrieval_miss_query_ids": missing_hard_ids,
            "hard_query_group_weight": args.hard_weight,
            "folds": fold_training,
        },
        "hard_oof": {
            "baseline": baseline_hard_metrics,
            "candidate": candidate_hard_metrics,
            "fixed_count": hard_fixed,
            "regressed_count": hard_regressed,
        },
        "non_hard_dev_regression_screen": {
            "baseline": baseline_regression_metrics,
            "candidate": candidate_regression_metrics,
            "fixed_count": screen_fixed,
            "regressed_count": screen_regressed,
            "excluded_hard_components": len(hard_components),
        },
    }
    result["verdict"] = (
        "GO_NEW_INDEPENDENT_VALIDATION"
        if candidate_hard_metrics["top1_correct_count"] > baseline_hard_metrics["top1_correct_count"]
        and hard_fixed > hard_regressed
        and screen_fixed >= screen_regressed
        else "PIVOT_RANKER_TRAINING"
    )

    identity = hashlib.sha256(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": result["inputs"]["dataset_manifest_sha256"],
                "r30": result["inputs"]["r30_labels_sha256"],
                "r53": result["inputs"]["r53_labels_sha256"],
                "corrected_overlay": result["inputs"]["corrected_overlay_sha256"],
                "params": RANKER_PARAMS,
                "hard_weight": args.hard_weight,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    _json_dump(output / "evaluation.json", result)
    comparison.to_parquet(output / "hard_oof_comparison.parquet", index=False)
    regression_comparison.to_parquet(
        output / "non_hard_dev_regression_comparison.parquet", index=False
    )
    full_model.save_model(output / "ranker_candidate.json")
    _json_dump(
        output / "artifact_hashes.json",
        {
            name: _sha256(output / name)
            for name in (
                "evaluation.json",
                "hard_oof_comparison.parquet",
                "non_hard_dev_regression_comparison.parquet",
                "ranker_candidate.json",
            )
        },
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--r30-labels",
        type=Path,
        default=Path("reports/v412_review_adjudication_labels.csv"),
    )
    parser.add_argument(
        "--r53-labels",
        type=Path,
        default=Path("reports/v412_review_rerank_counteraudit_53.csv"),
    )
    parser.add_argument("--corrected-overlay", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
