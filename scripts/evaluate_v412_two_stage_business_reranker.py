#!/usr/bin/env python3
"""Cross-fitted V4.12 ranker followed by a top-20 business reranker."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_hard_label_ranker import _evaluate_predictions
from scripts.evaluate_v412_ranker_business_features import (
    BASE,
    DEFAULT_DATASET,
    DEFAULT_ETABLISSEMENTS,
    DEFAULT_REFERENCE,
    DEFAULT_TRUSTED,
    DEFAULT_UNITES_LEGALES,
    VARIANTS,
    _compare,
    _fit_ranker,
    _read_enriched_sources,
    _relational_features,
    _source_features,
)
from scripts.run_v411_ranker_c_development import (
    RANKER_C_FEATURE_ORDER,
    RANKER_PARAMS,
    eligible_ranker_rows,
)


DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_two_stage_business_reranker"
STAGE2_FEATURES = VARIANTS["targeted"] + ["_stage1_score", "_stage1_rank"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(model: xgb.XGBRanker, rows: pd.DataFrame, features: list[str], score_name: str) -> pd.DataFrame:
    output = rows.copy()
    output[score_name] = model.predict(
        output[features].to_numpy(dtype=np.float32)
    ).astype("float32")
    return output


def _rank_column(rows: pd.DataFrame, score_name: str, rank_name: str) -> pd.DataFrame:
    output = rows.sort_values(
        ["query_id", score_name, "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()
    output[rank_name] = output.groupby("query_id", sort=False).cumcount() + 1
    return output.sort_index()


def _top_stage1(rows: pd.DataFrame, limit: int, *, include_truth: bool) -> pd.DataFrame:
    keep = rows["_stage1_rank"].le(limit)
    if include_truth:
        keep |= rows["is_ground_truth"].eq(1)
    return rows[keep].copy()


def _fit_stage2(
    rows: pd.DataFrame,
    hard_ids: set[str],
    hard_weight: float,
) -> xgb.XGBRanker:
    ordered = rows.sort_values(["query_id", "candidate_siret"], kind="mergesort")
    grouped = ordered.groupby("query_id", sort=False)
    query_order = list(grouped.indices)
    params = dict(RANKER_PARAMS)
    params.update(
        {
            "max_depth": 4,
            "min_child_weight": 2,
            "n_estimators": 600,
            "learning_rate": 0.03,
        }
    )
    model = xgb.XGBRanker(**params)
    model.fit(
        ordered[STAGE2_FEATURES].to_numpy(dtype=np.float32),
        ordered["is_ground_truth"].to_numpy(dtype=np.int8),
        group=grouped.size().to_numpy(),
        sample_weight=np.asarray(
            [hard_weight if query_id in hard_ids else 1.0 for query_id in query_order],
            dtype=np.float32,
        ),
        verbose=False,
    )
    return model


def _stage2_predictions(model: xgb.XGBRanker, rows: pd.DataFrame) -> pd.DataFrame:
    scored = _score(model, rows, STAGE2_FEATURES, "ranker_score")
    ranked = _rank_column(scored, "ranker_score", "ranker_rank")
    return ranked[
        ["query_id", "candidate_siret", "candidate_siren", "retrieval_rank", "ranker_score", "ranker_rank"]
    ].reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset.resolve()
    frame = _relational_features(
        _source_features(
            _read_enriched_sources(dataset, args.etablissements, args.unites_legales)
        )
    )
    labels = pd.read_parquet(dataset / "labels.parquet")
    assignments = pd.read_parquet(dataset / "split_assignments.parquet")
    reference = pd.read_parquet(args.reference.resolve() / "ranker_reference.parquet")
    trusted = pd.read_csv(args.trusted_labels, dtype=str).fillna("")
    for value in (frame, labels, assignments, reference, trusted):
        value["query_id"] = value["query_id"].astype(str)
    population = labels.merge(assignments, on="query_id", validate="one_to_one")
    fit_population = population[population["split"].eq("fit")].copy()
    base_rows = eligible_ranker_rows(
        frame[frame["query_id"].isin(fit_population["query_id"])], fit_population
    )

    trusted_truth = trusted[trusted["label_kind"].eq("MATCH_EXACT")][
        ["query_id", "ground_truth_siret"]
    ].merge(
        assignments[["query_id", "oof_fold", "siren_component_id", "split"]],
        on="query_id",
        validate="one_to_one",
    )
    trusted_truth["ground_truth_siren"] = trusted_truth["ground_truth_siret"].str[:9]
    trusted_truth["oof_fold"] = trusted_truth["oof_fold"].astype(int)
    trusted_rows = frame[frame["query_id"].isin(trusted_truth["query_id"])].drop(
        columns=["is_ground_truth"]
    ).merge(
        trusted_truth[["query_id", "ground_truth_siret", "oof_fold"]],
        on="query_id",
        validate="many_to_one",
    )
    trusted_rows["is_ground_truth"] = trusted_rows["candidate_siret"].eq(
        trusted_rows["ground_truth_siret"]
    ).astype("int8")
    present = trusted_rows.groupby("query_id")["is_ground_truth"].sum()
    eligible_trusted = set(present[present.eq(1)].index.astype(str))

    fold_map = assignments.set_index("query_id")["oof_fold"].astype(int)
    frame["_fold"] = frame["query_id"].map(fold_map)
    base_rows["_fold"] = base_rows["query_id"].map(fold_map)
    trusted_rows["_fold"] = trusted_rows["oof_fold"].astype(int)

    stage1_scored_parts: list[pd.DataFrame] = []
    stage1_models: dict[int, xgb.XGBRanker] = {}
    for fold in range(5):
        stage1_train = pd.concat(
            [
                base_rows[base_rows["_fold"].ne(fold)],
                trusted_rows[
                    trusted_rows["query_id"].isin(eligible_trusted)
                    & trusted_rows["_fold"].ne(fold)
                ],
            ],
            ignore_index=True,
        )
        hard_ids = set(
            trusted_rows.loc[
                trusted_rows["query_id"].isin(eligible_trusted)
                & trusted_rows["_fold"].ne(fold),
                "query_id",
            ].astype(str)
        )
        model = _fit_ranker(
            stage1_train,
            RANKER_C_FEATURE_ORDER,
            hard_ids,
            args.hard_weight,
            0,
        )
        stage1_models[fold] = model
        stage1_scored_parts.append(
            _score(
                model,
                frame[frame["_fold"].eq(fold)],
                RANKER_C_FEATURE_ORDER,
                "_stage1_score",
            )
        )
    oof_frame = pd.concat(stage1_scored_parts).sort_index()
    oof_frame = _rank_column(oof_frame, "_stage1_score", "_stage1_rank")

    truth_lookup = trusted_truth.set_index("query_id")["ground_truth_siret"]
    oof_frame["is_ground_truth"] = oof_frame["candidate_siret"].eq(
        oof_frame["query_id"].map(truth_lookup)
    ).astype("int8")
    base_truth_lookup = fit_population.set_index("query_id")["ground_truth_siret"]
    base_mask = oof_frame["query_id"].isin(fit_population["query_id"])
    oof_frame.loc[base_mask, "is_ground_truth"] = oof_frame.loc[base_mask, "candidate_siret"].eq(
        oof_frame.loc[base_mask, "query_id"].map(base_truth_lookup)
    ).astype("int8")

    trusted_predictions: list[pd.DataFrame] = []
    base_predictions: list[pd.DataFrame] = []
    stage2_models: dict[int, xgb.XGBRanker] = {}
    train_ids = set(base_rows["query_id"]) | eligible_trusted
    for fold in range(5):
        stage2_train = oof_frame[
            oof_frame["query_id"].isin(train_ids) & oof_frame["_fold"].ne(fold)
        ]
        stage2_train = _top_stage1(stage2_train, args.top_n, include_truth=True)
        positive_counts = stage2_train.groupby("query_id")["is_ground_truth"].sum()
        stage2_train = stage2_train[
            stage2_train["query_id"].isin(set(positive_counts[positive_counts.eq(1)].index))
        ]
        hard_ids = set(
            trusted_rows.loc[
                trusted_rows["query_id"].isin(eligible_trusted)
                & trusted_rows["_fold"].ne(fold),
                "query_id",
            ].astype(str)
        )
        model = _fit_stage2(stage2_train, hard_ids, args.stage2_hard_weight)
        stage2_models[fold] = model
        trusted_test = _top_stage1(
            oof_frame[
                oof_frame["query_id"].isin(trusted_truth["query_id"])
                & oof_frame["_fold"].eq(fold)
            ],
            args.top_n,
            include_truth=False,
        )
        base_test = _top_stage1(
            oof_frame[
                oof_frame["query_id"].isin(fit_population["query_id"])
                & oof_frame["_fold"].eq(fold)
            ],
            args.top_n,
            include_truth=False,
        )
        trusted_predictions.append(_stage2_predictions(model, trusted_test))
        base_predictions.append(_stage2_predictions(model, base_test))

    trusted_oof = pd.concat(trusted_predictions, ignore_index=True)
    base_oof = pd.concat(base_predictions, ignore_index=True)
    trusted_metrics, trusted_detail = _compare(
        reference[reference["query_id"].isin(trusted_truth["query_id"])],
        trusted_oof,
        trusted_truth,
    )
    base_truth = fit_population[fit_population["label_kind"].eq("MATCH_EXACT")][
        ["query_id", "ground_truth_siret", "ground_truth_siren"]
    ]
    base_metrics, _ = _evaluate_predictions(base_oof, base_truth)

    full_stage1_train = pd.concat(
        [base_rows, trusted_rows[trusted_rows["query_id"].isin(eligible_trusted)]],
        ignore_index=True,
    )
    full_stage1 = _fit_ranker(
        full_stage1_train,
        RANKER_C_FEATURE_ORDER,
        eligible_trusted,
        args.hard_weight,
        0,
    )
    trusted_ids = set(trusted["query_id"])
    trusted_components = set(trusted_truth["siren_component_id"].astype(str))
    regression_truth = population[
        population["split"].eq("dev")
        & population["label_kind"].eq("MATCH_EXACT")
        & ~population["query_id"].isin(trusted_ids)
        & ~population["siren_component_id"].astype(str).isin(trusted_components)
    ][["query_id", "ground_truth_siret", "ground_truth_siren"]]
    regression_rows = _score(
        full_stage1,
        frame[frame["query_id"].isin(regression_truth["query_id"])],
        RANKER_C_FEATURE_ORDER,
        "_stage1_score",
    )
    regression_rows = _rank_column(regression_rows, "_stage1_score", "_stage1_rank")
    full_stage2_train = _top_stage1(
        oof_frame[oof_frame["query_id"].isin(train_ids)], args.top_n, include_truth=True
    )
    positive_counts = full_stage2_train.groupby("query_id")["is_ground_truth"].sum()
    full_stage2_train = full_stage2_train[
        full_stage2_train["query_id"].isin(set(positive_counts[positive_counts.eq(1)].index))
    ]
    full_stage2 = _fit_stage2(full_stage2_train, eligible_trusted, args.stage2_hard_weight)
    regression_predictions = _stage2_predictions(
        full_stage2, _top_stage1(regression_rows, args.top_n, include_truth=False)
    )
    regression_metrics, regression_detail = _compare(
        reference[reference["query_id"].isin(regression_truth["query_id"])],
        regression_predictions,
        regression_truth,
    )

    result = {
        "schema_version": "sireto-v4.12-two-stage-business-reranker-development-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "final_test_opened": False,
        "top_n": args.top_n,
        "stage1_hard_weight": args.hard_weight,
        "stage2_hard_weight": args.stage2_hard_weight,
        "stage2_feature_count": len(STAGE2_FEATURES),
        "trusted_oof": trusted_metrics,
        "base_fit_oof": base_metrics,
        "non_trusted_dev": regression_metrics,
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": result["schema_version"],
                "dataset": _sha256(dataset / "manifest.json"),
                "trusted": _sha256(args.trusted_labels),
                "top_n": args.top_n,
                "stage1_weight": args.hard_weight,
                "stage2_weight": args.stage2_hard_weight,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    trusted_detail.to_parquet(output / "trusted_oof_comparison.parquet", index=False)
    regression_detail.to_parquet(output / "non_trusted_dev_comparison.parquet", index=False)
    full_stage1.save_model(output / "ranker_stage1.json")
    full_stage2.save_model(output / "ranker_stage2.json")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--trusted-labels", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--etablissements", type=Path, default=DEFAULT_ETABLISSEMENTS)
    parser.add_argument("--unites-legales", type=Path, default=DEFAULT_UNITES_LEGALES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--stage2-hard-weight", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
