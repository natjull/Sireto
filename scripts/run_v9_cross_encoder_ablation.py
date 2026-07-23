#!/usr/bin/env python3
"""Fine-tune and score the optional pinned multilingual V9 cross-encoder."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.semantic import assert_tokenizer_healthy
from src.xgb_matcher.v9_cross_encoder import serialize_cross_encoder_pair
from src.xgb_matcher.v9_dataset import V9DatasetManifest, read_table


def paired_top20(
    dataset_dir: Path,
    prediction_path: Path,
) -> pd.DataFrame:
    queries = pd.read_parquet(dataset_dir / "queries.parquet")
    candidates = pd.read_parquet(dataset_dir / "candidates.parquet")
    predictions = read_table(prediction_path)
    top20 = predictions[
        pd.to_numeric(predictions["rank"], errors="coerce").le(20)
        & predictions["candidate_siret"].notna()
    ].copy()
    paired = top20.merge(
        candidates,
        on=["query_id", "candidate_siret"],
        how="left",
        suffixes=("", "_candidate"),
        validate="one_to_one",
    ).merge(
        queries[
            [
                "query_id",
                "crm_name",
                "crm_address",
                "crm_postcode",
                "crm_city",
            ]
        ],
        on="query_id",
        how="left",
        validate="many_to_one",
    )
    pairs = paired.apply(serialize_cross_encoder_pair, axis=1)
    paired["cross_text_crm"] = [pair[0] for pair in pairs]
    paired["cross_text_candidate"] = [pair[1] for pair in pairs]
    return paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ranker-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {args.output_dir}")
    manifest = V9DatasetManifest.load(args.dataset / "manifest.json")
    manifest.validate(feature_order=manifest.feature_order)
    paired = paired_top20(args.dataset, args.ranker_predictions)
    original_predictions = read_table(args.ranker_predictions)
    sentinels = original_predictions[
        original_predictions["candidate_siret"].isna()
    ].copy()

    def with_sentinels(predictions: pd.DataFrame) -> pd.DataFrame:
        if sentinels.empty:
            return predictions
        aligned = sentinels.reindex(columns=predictions.columns)
        return pd.concat([predictions, aligned], ignore_index=True)

    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    pinned_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(pinned_path)
    assert_tokenizer_healthy(tokenizer)
    def fit_cross_encoder(training_rows: pd.DataFrame):
        cross_model = CrossEncoder(
            pinned_path,
            num_labels=1,
            device=args.device,
        )
        examples = [
            InputExample(
                texts=[row.cross_text_crm, row.cross_text_candidate],
                label=float(row.is_ground_truth),
            )
            for row in training_rows.itertuples()
        ]
        loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
        cross_model.fit(
            train_dataloader=loader,
            epochs=args.epochs,
            warmup_steps=max(1, len(loader) // 10),
            show_progress_bar=True,
        )
        return cross_model

    def score_cross_encoder(cross_model, scoring_rows: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            cross_model.predict(
                scoring_rows[["cross_text_crm", "cross_text_candidate"]]
                .apply(tuple, axis=1)
                .tolist(),
                batch_size=args.batch_size,
                show_progress_bar=True,
            )
        ).reshape(-1)

    # Cross-encoder train scores are themselves OOF. This is deliberately more
    # expensive than scoring all train rows with one in-sample model, because
    # the downstream acceptor is forbidden from learning on optimistic scenes.
    paired["cross_encoder_score"] = np.nan
    train = paired[paired["split"].eq("train")].copy()
    folds = sorted(
        pd.to_numeric(train["fold"], errors="coerce").dropna().astype(int).unique()
    )
    if len(folds) < 2:
        raise ValueError("Cross-encoder ablation requires OOF ranker folds")
    for fold in folds:
        fit_rows = train[pd.to_numeric(train["fold"], errors="coerce").ne(fold)]
        valid_mask = paired["split"].eq("train") & pd.to_numeric(
            paired["fold"], errors="coerce"
        ).eq(fold)
        cross_model = fit_cross_encoder(fit_rows)
        paired.loc[valid_mask, "cross_encoder_score"] = score_cross_encoder(
            cross_model,
            paired.loc[valid_mask],
        )
        del cross_model
        gc.collect()

    final_cross_model = fit_cross_encoder(train)
    holdout_mask = paired["split"].isin(["dev", "test"])
    paired.loc[holdout_mask, "cross_encoder_score"] = score_cross_encoder(
        final_cross_model,
        paired.loc[holdout_mask],
    )
    if paired["cross_encoder_score"].isna().any():
        raise ValueError("Cross-encoder OOF/holdout scoring is incomplete")
    paired["cross_encoder_rank"] = (
        paired.groupby("query_id")["cross_encoder_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    final_cross_model.save(str(args.output_dir / "model"))
    paired.to_parquet(args.output_dir / "top20_cross_scores.parquet", index=False)

    # Variant 1: no cross-encoder (copy the exact paired baseline predictions).
    base_columns = [
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "score",
        "rank",
        "prediction_origin",
        "fold",
    ]
    baseline_predictions = with_sentinels(paired[base_columns].copy())
    baseline_predictions.to_parquet(
        args.output_dir / "predictions_without_cross_encoder.parquet",
        index=False,
    )

    # Variant 2: cross-encoder alone on the same top-20.
    cross_predictions = paired[base_columns].copy()
    cross_predictions["score"] = paired["cross_encoder_score"].to_numpy()
    cross_predictions["rank"] = paired["cross_encoder_rank"].to_numpy()
    cross_predictions = with_sentinels(cross_predictions)
    cross_predictions.to_parquet(
        args.output_dir / "predictions_cross_encoder_only.parquet",
        index=False,
    )

    # Variant 3: cross-encoder score injected into a newly trained XGB ranker.
    from scripts.train_v9_ranker import (
        eligible_ranker_rows,
        score_rows,
        train_ranker,
    )

    labels = pd.read_parquet(args.dataset / "labels.parquet")
    injected_features = list(manifest.feature_order) + ["cross_encoder_score"]
    injected_parts = []
    for fold in folds:
        fold_train = train[pd.to_numeric(train["fold"], errors="coerce").ne(fold)]
        fit_rows = eligible_ranker_rows(fold_train, labels)
        ranker = train_ranker(fit_rows, injected_features, seed=42 + fold)
        fold_valid = paired[
            paired["split"].eq("train")
            & pd.to_numeric(paired["fold"], errors="coerce").eq(fold)
        ]
        injected_parts.append(
            score_rows(
                ranker,
                fold_valid,
                injected_features,
                origin="oof",
                fold=fold,
            )
        )
    final_ranker = train_ranker(
        eligible_ranker_rows(train, labels),
        injected_features,
        seed=42,
    )
    injected_parts.append(
        score_rows(
            final_ranker,
            paired[holdout_mask],
            injected_features,
            origin="holdout",
            fold=None,
        )
    )
    injected_predictions = with_sentinels(
        pd.concat(injected_parts, ignore_index=True)
    )
    injected_predictions.to_parquet(
        args.output_dir / "predictions_cross_encoder_injected.parquet",
        index=False,
    )
    final_ranker.save_model(args.output_dir / "ranker_with_cross_encoder.json")
    metadata = {
        "schema_version": "v9-cross-encoder-ablation-1",
        "dataset_manifest_id": manifest.build_id,
        "model_id": args.model_id,
        "revision": args.revision,
        "top_k": 20,
        "epochs": args.epochs,
        "cross_encoder_train_prediction_origin": "oof",
        "variants": {
            "without_cross_encoder": "predictions_without_cross_encoder.parquet",
            "cross_encoder_only": "predictions_cross_encoder_only.parquet",
            "cross_encoder_injected": "predictions_cross_encoder_injected.parquet",
        },
        "next_step": (
            "Run train_v9_acceptor.py independently on each predictions file, "
            "then apply evaluate_v9_gates.py to paired coverage/segment/latency."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
