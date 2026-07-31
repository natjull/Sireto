#!/usr/bin/env python3
"""Evaluate a bounded REVIEW-only reranking hypothesis on consumed data.

This script does not train a model and does not open any final test.  It
compares the frozen V4.12 ranker with a simple conditional score adjustment,
first on the 30 newly adjudicated REVIEW cases, then on the older reliable
TOP1_CORRECT development scenes as a regression check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_REFERENCE = BASE / "references/v4_12_service_parity/b4b7fef24c5e7036"
DEFAULT_PILOT = (
    BASE
    / "audits/v4_12_review_adjudication_pilot/c7a9feecaf2d3c2a"
)
DEFAULT_HISTORICAL_LABELS = (
    BASE
    / "audits/v4_7_current_adjudications/4cc5420fb5da0683/current_labels.parquet"
)
DEFAULT_HISTORICAL_CANDIDATES = (
    BASE
    / "datasets/v4_5_hard_scenes/21f8c0b0b172b907/candidates.parquet"
)


def _predictions(
    frame: pd.DataFrame,
    *,
    key: str,
    beta: float,
    siege_alpha: float,
    minimum_top1_score: float,
) -> tuple[pd.Series, pd.Series]:
    original = frame.loc[frame.groupby(key)["ranker_score"].idxmax()].set_index(key)
    rescored = frame.copy()
    rescored["spike_score"] = (
        rescored["ranker_score"]
        + beta * rescored["name_sim_max_ul"]
        + siege_alpha * rescored["is_siege"]
    )
    alternative = rescored.loc[
        rescored.groupby(key)["spike_score"].idxmax()
    ].set_index(key)
    prediction = original["candidate_siret"].copy()
    eligible = original["ranker_score"] >= minimum_top1_score
    prediction.loc[eligible] = alternative.loc[eligible, "candidate_siret"]
    return original["candidate_siret"], prediction


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    labels = pd.read_csv(args.labels, dtype=str)
    labels["selection_ordinal"] = labels["selection_ordinal"].astype(int)
    labels["ranking_label_usable"] = labels["ranking_label_usable"].eq("true")
    exact = labels[labels["ranking_label_usable"]].copy()
    ambiguous = labels[labels["label"].eq("AMBIGUOUS")].copy()

    docket = pd.read_parquet(args.pilot / "comparison/docket.parquet")
    if list(docket.sort_values("selection_ordinal")["query_id"]) != list(
        labels.sort_values("selection_ordinal")["query_id"]
    ):
        raise ValueError("label order or query identities differ from the frozen docket")
    if len(exact) != 27 or len(ambiguous) != 3:
        raise ValueError("expected 27 exact labels and 3 ambiguous labels")

    features = pd.read_parquet(args.reference / "candidates_features.parquet")
    ranker = pd.read_parquet(args.reference / "ranker_reference.parquet")
    selected_ids = set(labels["query_id"])
    selected = features.merge(
        ranker[["query_id", "candidate_siret", "ranker_score", "ranker_rank"]],
        on=["query_id", "candidate_siret"],
        validate="one_to_one",
    )
    selected = selected[selected["query_id"].isin(selected_ids)].copy()
    if selected["query_id"].nunique() != 30:
        raise ValueError("the reference does not contain all 30 selected scenes")
    baseline, spike = _predictions(
        selected,
        key="query_id",
        beta=args.beta,
        siege_alpha=args.siege_alpha,
        minimum_top1_score=args.minimum_top1_score,
    )
    truth = dict(zip(exact["query_id"], exact["validated_siret"]))
    baseline_correct = {query for query, siret in truth.items() if baseline[query] == siret}
    spike_correct = {query for query, siret in truth.items() if spike[query] == siret}
    fixed = sorted(spike_correct - baseline_correct)
    regressed_r30 = sorted(baseline_correct - spike_correct)
    remaining_wrong = sorted(set(truth) - spike_correct)

    historical_labels = pd.read_parquet(args.historical_labels)
    historical_labels = historical_labels[
        historical_labels["current_evidence_validated"].eq(True)
        & historical_labels["current_adjudication_label"].eq("TOP1_CORRECT")
    ][["audit_case_id", "current_top1_siret"]]
    historical_truth = dict(
        zip(
            historical_labels["audit_case_id"],
            historical_labels["current_top1_siret"],
        )
    )
    historical_candidates = pd.read_parquet(args.historical_candidates)
    historical_candidates = historical_candidates[
        historical_candidates["audit_case_id"].isin(historical_truth)
    ].copy()
    if historical_candidates["audit_case_id"].nunique() != len(historical_truth):
        raise ValueError("historical regression candidates are incomplete")
    historical_baseline, historical_spike = _predictions(
        historical_candidates,
        key="audit_case_id",
        beta=args.beta,
        siege_alpha=args.siege_alpha,
        minimum_top1_score=args.minimum_top1_score,
    )
    invalid_baseline = sorted(
        query
        for query, siret in historical_truth.items()
        if historical_baseline[query] != siret
    )
    if invalid_baseline:
        raise ValueError("historical TOP1_CORRECT labels do not reproduce the baseline")
    historical_regressions = sorted(
        query
        for query, siret in historical_truth.items()
        if historical_spike[query] != siret
    )

    return {
        "schema_version": "sireto-v4.12-review-rerank-spike-1",
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "model_training_performed": False,
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "hypothesis": {
            "formula": "ranker_score + beta*name_sim_max_ul + siege_alpha*is_siege",
            "beta": args.beta,
            "siege_alpha": args.siege_alpha,
            "applied_only_when_original_top1_score_at_least": args.minimum_top1_score,
        },
        "r30": {
            "exact_label_count": len(truth),
            "ambiguous_count": len(ambiguous),
            "baseline_correct": len(baseline_correct),
            "spike_correct": len(spike_correct),
            "fixed_count": len(fixed),
            "fixed_query_ids": fixed,
            "regression_count": len(regressed_r30),
            "regressed_query_ids": regressed_r30,
            "remaining_wrong_count": len(remaining_wrong),
            "remaining_wrong_query_ids": remaining_wrong,
            "ambiguous_forced_to_exact_label": 0,
        },
        "historical_regression_check": {
            "reliable_top1_correct_count": len(historical_truth),
            "regression_count": len(historical_regressions),
            "regressed_audit_case_ids": historical_regressions,
        },
        "verdict": (
            "GO_EXPAND_LABELS_BEFORE_TRAINING"
            if len(fixed) >= 8
            and not regressed_r30
            and not historical_regressions
            else "STOP_RERANK_HYPOTHESIS"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("reports/v412_review_adjudication_labels.csv"),
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument(
        "--historical-labels", type=Path, default=DEFAULT_HISTORICAL_LABELS
    )
    parser.add_argument(
        "--historical-candidates", type=Path, default=DEFAULT_HISTORICAL_CANDIDATES
    )
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--siege-alpha", type=float, default=2.0)
    parser.add_argument("--minimum-top1-score", type=float, default=2.5)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
