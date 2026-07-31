#!/usr/bin/env python3
"""OOF ranker retraining on all 279 trusted REVIEW adjudications."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_hard_label_ranker import (
    _evaluate_predictions,
    _rank_predictions,
    fit_weighted_ranker,
)
from scripts.run_v411_ranker_c_development import eligible_ranker_rows


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_DATASET = BASE / "datasets/v4_11_input_blind/ec4326ec57e4411d"
DEFAULT_REFERENCE = BASE / "references/v4_12_service_parity/b4b7fef24c5e7036"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_trusted_label_ranker"
DEFAULT_TRUSTED = Path("reports/v412_review_trusted_labels_279.csv")
SCHEMA_VERSION = "sireto-v4.12-trusted-label-ranker-development-2"
WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, int, int]:
    baseline_metrics, baseline_detail = _evaluate_predictions(
        baseline_predictions, truth
    )
    candidate_metrics, candidate_detail = _evaluate_predictions(
        candidate_predictions, truth
    )
    detail = baseline_detail[
        ["query_id", "predicted_siret", "top1_correct"]
    ].merge(
        candidate_detail[
            ["query_id", "predicted_siret", "top1_correct", "truth_in_pool"]
        ],
        on="query_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    fixed = int(
        ((~detail["top1_correct_baseline"]) & detail["top1_correct_candidate"]).sum()
    )
    regressed = int(
        (detail["top1_correct_baseline"] & (~detail["top1_correct_candidate"])).sum()
    )
    return baseline_metrics, candidate_metrics, detail, fixed, regressed


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset.resolve()
    trusted = pd.read_csv(args.trusted_labels, dtype=str).fillna("")
    if len(trusted) != 279 or trusted["label_kind"].value_counts().to_dict() != {
        "MATCH_EXACT": 254,
        "AMBIGUOUS": 25,
    }:
        raise ValueError("Trusted REVIEW overlay changed")
    trusted_exact = trusted[trusted["label_kind"].eq("MATCH_EXACT")][
        ["query_id", "ground_truth_siret"]
    ].copy()
    trusted_exact["ground_truth_siren"] = trusted_exact["ground_truth_siret"].str[:9]
    trusted_ids = set(trusted["query_id"].astype(str))

    assignments = pd.read_parquet(dataset / "split_assignments.parquet")
    labels = pd.read_parquet(dataset / "labels.parquet")
    candidates = pd.read_parquet(dataset / "candidates_sparse_top100.parquet")
    baseline = pd.read_parquet(args.reference.resolve() / "ranker_reference.parquet")
    for frame in (assignments, labels, candidates, baseline):
        frame["query_id"] = frame["query_id"].astype(str)

    trusted_truth = trusted_exact.merge(
        assignments[["query_id", "oof_fold", "siren_component_id", "split"]],
        on="query_id",
        validate="one_to_one",
    )
    if trusted_truth["split"].value_counts().to_dict() != {"dev": 254}:
        raise ValueError("Trusted exact labels must all belong to dev")
    trusted_truth["oof_fold"] = trusted_truth["oof_fold"].astype(int)

    population = labels.merge(assignments, on="query_id", validate="one_to_one")
    fit_population = population[population["split"].eq("fit")].copy()
    fit_candidates = candidates[candidates["query_id"].isin(fit_population["query_id"])].copy()
    base_rows = eligible_ranker_rows(fit_candidates, fit_population)

    trusted_candidates = candidates[
        candidates["query_id"].isin(trusted_truth["query_id"])
    ].drop(columns=["is_ground_truth"]).merge(
        trusted_truth[
            ["query_id", "ground_truth_siret", "ground_truth_siren", "oof_fold"]
        ],
        on="query_id",
        validate="many_to_one",
    )
    trusted_candidates["is_ground_truth"] = trusted_candidates["candidate_siret"].eq(
        trusted_candidates["ground_truth_siret"]
    ).astype("int8")
    positive_counts = trusted_candidates.groupby("query_id")["is_ground_truth"].sum()
    eligible_ids = set(positive_counts[positive_counts.eq(1)].index.astype(str))
    retrieval_misses = sorted(set(trusted_truth["query_id"]) - eligible_ids)
    if len(eligible_ids) != 251 or len(retrieval_misses) != 3:
        raise ValueError("Trusted retrieval presence changed")

    variants: list[dict[str, Any]] = []
    details: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for weight in WEIGHTS:
        trusted_parts: list[pd.DataFrame] = []
        base_parts: list[pd.DataFrame] = []
        for fold in range(5):
            base_train = base_rows[
                ~base_rows["query_id"].isin(
                    set(
                        fit_population.loc[
                            fit_population["oof_fold"].astype(int).eq(fold), "query_id"
                        ].astype(str)
                    )
                )
            ]
            trusted_train = trusted_candidates[
                trusted_candidates["query_id"].isin(eligible_ids)
                & trusted_candidates["oof_fold"].ne(fold)
            ]
            model = fit_weighted_ranker(
                pd.concat([base_train, trusted_train], ignore_index=True),
                hard_query_ids=set(trusted_train["query_id"].astype(str)),
                hard_weight=weight,
            )
            trusted_parts.append(
                _rank_predictions(
                    model,
                    trusted_candidates[trusted_candidates["oof_fold"].eq(fold)],
                )
            )
            base_held_ids = set(
                fit_population.loc[
                    fit_population["oof_fold"].astype(int).eq(fold), "query_id"
                ].astype(str)
            )
            base_parts.append(
                _rank_predictions(
                    model, candidates[candidates["query_id"].isin(base_held_ids)]
                )
            )
        trusted_oof = pd.concat(trusted_parts, ignore_index=True)
        base_oof = pd.concat(base_parts, ignore_index=True)
        trusted_base, trusted_candidate, trusted_detail, fixed, regressed = _comparison(
            baseline[baseline["query_id"].isin(trusted_truth["query_id"])],
            trusted_oof,
            trusted_truth,
        )
        base_truth = fit_population[
            fit_population["label_kind"].eq("MATCH_EXACT")
        ][["query_id", "ground_truth_siret", "ground_truth_siren"]]
        base_candidate, base_detail = _evaluate_predictions(base_oof, base_truth)
        eligible = (
            fixed > regressed
            and float(base_candidate["hit_at_1"] or 0.0) >= 0.995
        )
        variants.append(
            {
                "hard_weight": weight,
                "eligible": eligible,
                "trusted_oof": {
                    "baseline": trusted_base,
                    "candidate": trusted_candidate,
                    "fixed_count": fixed,
                    "regressed_count": regressed,
                },
                "base_fit_oof_screen": {
                    "candidate": base_candidate,
                    "minimum_hit_at_1": 0.995,
                },
            }
        )
        details[weight] = (trusted_detail, base_detail)

    eligible_variants = [item for item in variants if item["eligible"]]
    winner = (
        sorted(
            eligible_variants,
            key=lambda item: (
                -item["trusted_oof"]["candidate"]["top1_correct_count"],
                -item["base_fit_oof_screen"]["candidate"]["top1_correct_count"],
                item["hard_weight"],
            ),
        )[0]
        if eligible_variants
        else None
    )
    if winner is None:
        raise ValueError("No eligible trusted-label ranker variant")

    selected_weight = float(winner["hard_weight"])
    full_augmentation = trusted_candidates[
        trusted_candidates["query_id"].isin(eligible_ids)
    ]
    model = fit_weighted_ranker(
        pd.concat([base_rows, full_augmentation], ignore_index=True),
        hard_query_ids=set(full_augmentation["query_id"].astype(str)),
        hard_weight=selected_weight,
    )

    trusted_components = set(trusted_truth["siren_component_id"].astype(str))
    regression_truth = population[
        population["split"].eq("dev")
        & population["label_kind"].eq("MATCH_EXACT")
        & ~population["query_id"].isin(trusted_ids)
        & ~population["siren_component_id"].astype(str).isin(trusted_components)
    ][["query_id", "ground_truth_siret", "ground_truth_siren"]]
    regression_predictions = _rank_predictions(
        model, candidates[candidates["query_id"].isin(regression_truth["query_id"])]
    )
    reg_base, reg_candidate, reg_detail, reg_fixed, reg_regressed = _comparison(
        baseline[baseline["query_id"].isin(regression_truth["query_id"])],
        regression_predictions,
        regression_truth,
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "inputs": {
            "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
            "trusted_labels_sha256": _sha256(args.trusted_labels),
            "trusted_query_count": 279,
            "trusted_exact_count": 254,
            "trusted_ambiguous_count": 25,
        },
        "retrieval": {
            "truth_present_count": 251,
            "retrieval_miss_count": 3,
            "retrieval_miss_query_ids": retrieval_misses,
        },
        "variants": variants,
        "selected": winner,
        "non_trusted_dev_regression_screen": {
            "baseline": reg_base,
            "candidate": reg_candidate,
            "fixed_count": reg_fixed,
            "regressed_count": reg_regressed,
        },
    }
    result["verdict"] = (
        "GO_BUILD_TRUSTED_OOF_SCENES"
        if reg_fixed >= reg_regressed
        else "PIVOT_RANKER_REGRESSION"
    )
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "dataset": result["inputs"]["dataset_manifest_sha256"],
                "trusted": result["inputs"]["trusted_labels_sha256"],
                "weights": WEIGHTS,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    trusted_detail, base_detail = details[selected_weight]
    trusted_detail.to_parquet(output / "trusted_oof_comparison.parquet", index=False)
    base_detail.to_parquet(output / "base_fit_oof_comparison.parquet", index=False)
    reg_detail.to_parquet(output / "non_trusted_dev_comparison.parquet", index=False)
    model.save_model(output / "ranker_candidate.json")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--trusted-labels", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
