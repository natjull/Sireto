#!/usr/bin/env python3
"""Train the single V9 candidate ranker and emit OOF/holdout predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.contracts import GroundTruthKind
from src.xgb_matcher.features import SEMANTIC_FEATURE_NAMES
from src.xgb_matcher.v9_dataset import V9DatasetManifest
from src.xgb_matcher.v9_scene import V9_ACCEPTOR_EVIDENCE_BASE_FEATURE_NAMES


def fold_for_query(query_id: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}:fold:{query_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % folds


def fold_for_entity(
    query_id: str,
    ground_truth_siren: str | None,
    seed: int,
    folds: int,
) -> int:
    """Keep every labelled occurrence of a SIREN in the same OOF fold."""
    entity = str(ground_truth_siren or "").strip()
    key = f"siren:{entity}" if entity else f"query:{query_id}"
    return fold_for_query(key, seed, folds)


def eligible_ranker_rows(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Keep exact-match queries whose retrieved pool genuinely contains the positive."""
    exact_ids = set(
        labels.loc[
            labels["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value),
            "query_id",
        ].astype(str)
    )
    rows = candidates[candidates["query_id"].astype(str).isin(exact_ids)].copy()
    has_positive = rows.groupby("query_id")["is_ground_truth"].max()
    return rows[rows["query_id"].isin(has_positive[has_positive.eq(1)].index)].copy()


def validate_training_positive_rows(
    injected: pd.DataFrame,
    labels: pd.DataFrame,
    feature_order: list[str],
) -> pd.DataFrame:
    """Allow positives only as ranker fit rows, never as evaluation candidates."""
    required = {"query_id", "candidate_siret"} | set(feature_order)
    missing = required - set(injected.columns)
    if missing:
        raise ValueError(f"Missing injected-positive columns: {sorted(missing)}")
    label_index = labels.set_index("query_id")
    unknown = set(injected["query_id"]) - set(label_index.index)
    if unknown:
        raise ValueError("Injected positives reference unknown queries")
    if not injected["query_id"].map(label_index["split"]).eq("train").all():
        raise ValueError("Positive injection is forbidden outside the train split")
    expected = injected["query_id"].map(label_index["ground_truth_siret"])
    if not injected["candidate_siret"].astype(str).eq(expected.astype(str)).all():
        raise ValueError("Injected row must be the exact labelled SIRET")
    output = injected.copy()
    output["candidate_siren"] = output["candidate_siret"].astype(str).str[:9]
    output["split"] = "train"
    output["is_ground_truth"] = 1
    output["retrieval_source"] = "injected_training_positive"
    return output


def train_ranker(rows: pd.DataFrame, feature_order: list[str], seed: int) -> xgb.XGBRanker:
    ordered = rows.sort_values(["query_id", "candidate_siret"]).copy()
    groups = ordered.groupby("query_id", sort=False).size().to_numpy()
    model = xgb.XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@1",
        n_estimators=800,
        max_depth=6,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(
        ordered[feature_order].astype(float).to_numpy(),
        ordered["is_ground_truth"].astype(int).to_numpy(),
        group=groups,
        verbose=False,
    )
    return model


def score_rows(
    model: xgb.XGBRanker,
    rows: pd.DataFrame,
    feature_order: list[str],
    *,
    origin: str,
    fold: int | None,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    output = rows[
        [
            column
            for column in (
                "query_id",
                "candidate_siret",
                "candidate_siren",
                "sparse_rank",
                "dense_rank",
                "rrf_score",
                "retrieval_channel_count",
                "retrieval_agreement",
                *V9_ACCEPTOR_EVIDENCE_BASE_FEATURE_NAMES,
            )
            if column in rows.columns
        ]
    ].copy()
    output["score"] = model.predict(rows[feature_order].astype(float).to_numpy())
    output["prediction_origin"] = origin
    output["fold"] = fold
    output["rank"] = (
        output.groupby("query_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return output


def ranking_metrics(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict:
    exact = labels[labels["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)].copy()
    top1 = (
        predictions.sort_values(["query_id", "rank"])
        .groupby("query_id", as_index=False)
        .first()
    )
    evaluated = exact.merge(top1, on="query_id", how="left")
    candidate_hits = exact[["query_id", "ground_truth_siret"]].merge(
        predictions[["query_id", "candidate_siret"]],
        on="query_id",
        how="left",
    )
    candidate_hits["hit"] = (
        candidate_hits["candidate_siret"].fillna("")
        == candidate_hits["ground_truth_siret"].fillna("")
    )
    recall_by_query = candidate_hits.groupby("query_id")["hit"].max()
    siret_correct = (
        evaluated["candidate_siret"].fillna("")
        == evaluated["ground_truth_siret"].fillna("")
    )
    siren_correct = (
        evaluated["candidate_siren"].fillna("")
        == evaluated["ground_truth_siren"].fillna("")
    )
    return {
        "exact_query_count": int(len(evaluated)),
        "candidate_recall": (
            float(
                exact["query_id"]
                .map(recall_by_query)
                .fillna(False)
                .astype(bool)
                .mean()
            )
            if len(exact)
            else 0.0
        ),
        "hit_at_1_siret": float(siret_correct.mean()) if len(evaluated) else 0.0,
        "hit_at_1_siren": float(siren_correct.mean()) if len(evaluated) else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--training-positive-rows",
        type=Path,
        help="Optional query-specific positive feature rows; train-only and never scored.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-semantic",
        action="store_true",
        help="Ablation only: include the three repaired semantic features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {args.output_dir}")

    manifest = V9DatasetManifest.load(args.dataset / "manifest.json")
    manifest.validate(feature_order=manifest.feature_order)
    labels = pd.read_parquet(args.dataset / "labels.parquet")
    candidates = pd.read_parquet(args.dataset / "candidates.parquet")
    features = (
        list(manifest.feature_order)
        if args.include_semantic
        else [
            feature
            for feature in manifest.feature_order
            if feature not in SEMANTIC_FEATURE_NAMES
        ]
    )
    missing_features = set(features) - set(manifest.feature_order)
    if missing_features:
        raise ValueError(
            f"Dataset manifest is missing ranker features: {sorted(missing_features)}"
        )
    if args.include_semantic:
        missing_semantic = set(SEMANTIC_FEATURE_NAMES) - set(features)
        if missing_semantic:
            raise ValueError(
                "Semantic ablation requested but dataset manifest is missing "
                f"features: {sorted(missing_semantic)}"
            )

    train_labels = labels[labels["split"].eq("train")].copy()
    train_candidates = candidates[candidates["split"].eq("train")].copy()
    fit_candidates = train_candidates
    if args.training_positive_rows:
        injected = pd.read_parquet(args.training_positive_rows)
        injected = validate_training_positive_rows(injected, labels, features)
        fit_candidates = pd.concat(
            [train_candidates, injected],
            ignore_index=True,
        ).drop_duplicates(["query_id", "candidate_siret"], keep="first")
    train_labels["fold"] = [
        fold_for_entity(
            str(query_id),
            str(siren) if pd.notna(siren) else None,
            args.seed,
            args.folds,
        )
        for query_id, siren in zip(
            train_labels["query_id"],
            train_labels["ground_truth_siren"],
            strict=True,
        )
    ]

    prediction_parts: list[pd.DataFrame] = []
    for fold in range(args.folds):
        fold_train_ids = set(
            train_labels.loc[train_labels["fold"].ne(fold), "query_id"].astype(str)
        )
        fold_valid_ids = set(
            train_labels.loc[train_labels["fold"].eq(fold), "query_id"].astype(str)
        )
        fit_rows = eligible_ranker_rows(
            fit_candidates[fit_candidates["query_id"].isin(fold_train_ids)],
            train_labels[train_labels["query_id"].isin(fold_train_ids)],
        )
        if fit_rows.empty:
            raise ValueError(f"Fold {fold} has no eligible ranker training rows")
        model = train_ranker(fit_rows, features, args.seed + fold)
        valid_rows = train_candidates[
            train_candidates["query_id"].isin(fold_valid_ids)
        ]
        prediction_parts.append(
            score_rows(model, valid_rows, features, origin="oof", fold=fold)
        )

    final_fit_rows = eligible_ranker_rows(fit_candidates, train_labels)
    final_model = train_ranker(final_fit_rows, features, args.seed)
    holdout = candidates[candidates["split"].isin(["dev", "test"])]
    prediction_parts.append(
        score_rows(final_model, holdout, features, origin="holdout", fold=None)
    )
    predictions = pd.concat(prediction_parts, ignore_index=True)

    # Zero-candidate queries are explicit sentinel scenes, never silently dropped.
    predicted_ids = set(predictions["query_id"].astype(str))
    missing = labels[~labels["query_id"].astype(str).isin(predicted_ids)]
    if not missing.empty:
        sentinels = pd.DataFrame(
            {
                "query_id": missing["query_id"].astype(str),
                "candidate_siret": None,
                "candidate_siren": None,
                "score": 0.0,
                "prediction_origin": missing["split"].map(
                    {"train": "oof", "dev": "holdout", "test": "holdout"}
                ),
                "fold": [
                    (
                        fold_for_entity(
                            str(query_id),
                            (
                                str(ground_truth_siren)
                                if pd.notna(ground_truth_siren)
                                else None
                            ),
                            args.seed,
                            args.folds,
                        )
                        if split == "train"
                        else None
                    )
                    for query_id, ground_truth_siren, split in zip(
                        missing["query_id"],
                        missing["ground_truth_siren"],
                        missing["split"],
                        strict=True,
                    )
                ],
                "rank": 1,
            }
        )
        predictions = pd.concat([predictions, sentinels], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    final_model.save_model(args.output_dir / "ranker.json")
    predictions.to_parquet(args.output_dir / "ranker_predictions.parquet", index=False)
    metrics = {
        split: ranking_metrics(
            predictions[predictions["query_id"].isin(
                set(labels.loc[labels["split"].eq(split), "query_id"])
            )],
            labels[labels["split"].eq(split)],
        )
        for split in ("train", "dev", "test")
    }
    metadata = {
        "schema_version": "v9-ranker-1",
        "dataset_manifest_id": manifest.build_id,
        "retrieval_signature": manifest.retrieval_signature,
        "tokenizer_fingerprint": manifest.tokenizer_fingerprint,
        "feature_order": features,
        "folds": args.folds,
        "fold_group": "ground_truth_siren_else_query_id",
        "seed": args.seed,
        "semantic_features_included": bool(args.include_semantic),
        "metrics": metrics,
        "positive_injection": bool(args.training_positive_rows),
        "positive_injection_for_ranker_fit_only": bool(args.training_positive_rows),
        "positive_injection_in_predictions": False,
        "train_scene_prediction_origin": "oof",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
