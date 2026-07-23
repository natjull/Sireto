#!/usr/bin/env python3
"""Fine-tune and score the optional pinned multilingual V9 cross-encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
    ).merge(queries, on="query_id", how="left", validate="many_to_one")
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
    model = CrossEncoder(
        pinned_path,
        num_labels=1,
        device=args.device,
    )

    train = paired[paired["split"].eq("train")].copy()
    examples = [
        InputExample(
            texts=[row.cross_text_crm, row.cross_text_candidate],
            label=float(row.is_ground_truth),
        )
        for row in train.itertuples()
    ]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=max(1, len(loader) // 10),
        show_progress_bar=True,
    )
    scores = model.predict(
        paired[["cross_text_crm", "cross_text_candidate"]]
        .apply(tuple, axis=1)
        .tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    paired["cross_encoder_score"] = np.asarray(scores).reshape(-1)
    paired["cross_encoder_rank"] = (
        paired.groupby("query_id")["cross_encoder_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    model.save(str(args.output_dir / "model"))
    paired.to_parquet(args.output_dir / "top20_cross_scores.parquet", index=False)
    metadata = {
        "schema_version": "v9-cross-encoder-ablation-1",
        "dataset_manifest_id": manifest.build_id,
        "model_id": args.model_id,
        "revision": args.revision,
        "top_k": 20,
        "epochs": args.epochs,
        "output_contract": (
            "Use cross_encoder_rank for the CE-only variant; append "
            "cross_encoder_score to the canonical ranker feature order and "
            "retrain train_v9_ranker.py for the injected variant."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
