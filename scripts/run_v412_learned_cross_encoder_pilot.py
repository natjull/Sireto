#!/usr/bin/env python3
"""One-fold CPU/MPS pilot for a learned CRM/SIRET cross-encoder reranker."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_cross_encoder_reranker import _serialise_pairs  # noqa: E402
from scripts.evaluate_v412_ranker_business_features import _read_enriched_sources  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_RAW = BASE / "datasets/v4_12_learned_candidate_features/e22aa96feb6ac16f"
DEFAULT_BUSINESS = BASE / "datasets/v4_12_learned_business_features/8800ef53f6927215"
DEFAULT_RANKER = BASE / "experiments/v4_12_learned_oof_rankers/839ef55308d5077e"
DEFAULT_MODEL = BASE / (
    "models/huggingface/hub/models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1/"
    "snapshots/1427fd652930e4ba29e8149678df786c240d8825"
)
SCHEMA_VERSION = "sireto-v4.12-learned-cross-encoder-pilot-1"
SEED = 42


class PairwiseTextDataset(Dataset):
    def __init__(self, positive: list[tuple[str, str]], negative: list[tuple[str, str]]):
        if len(positive) != len(negative):
            raise ValueError("Positive and negative pair lists differ")
        self.positive = positive
        self.negative = negative

    def __len__(self) -> int:
        return len(self.positive)

    def __getitem__(self, index: int) -> tuple[tuple[str, str], tuple[str, str]]:
        return self.positive[index], self.negative[index]


def _collate(tokenizer: Any, max_length: int):
    def collate(batch: list[tuple[tuple[str, str], tuple[str, str]]]) -> dict[str, torch.Tensor]:
        pairs = [item[0] for item in batch] + [item[1] for item in batch]
        return tokenizer(
            [item[0] for item in pairs],
            [item[1] for item in pairs],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    return collate


def _score_pairs(
    model: Any,
    tokenizer: Any,
    pairs: list[tuple[str, str]],
    *,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(
                [item[0] for item in batch],
                [item[1] for item in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            output.append(model(**encoded).logits.squeeze(-1).detach().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def run(args: argparse.Namespace) -> Path:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    labels = pd.read_parquet(args.business / "labels.parquet")
    predictions_path = args.ranker / "business_learned_oof_candidates.parquet"
    predictions = pd.read_parquet(predictions_path)
    for frame in (labels, predictions):
        frame["query_id"] = frame["query_id"].astype(str)
    labels["ground_truth_siret"] = labels["ground_truth_siret"].astype("string")
    predictions["candidate_siret"] = predictions["candidate_siret"].astype(str)

    exact = labels[labels["label_kind"].eq("MATCH_EXACT")].copy()
    truth = exact.set_index("query_id")["ground_truth_siret"].astype(str)
    ranked = predictions[predictions["query_id"].isin(set(exact["query_id"]))].copy()
    ranked["truth"] = ranked["query_id"].map(truth)
    ranked["is_truth"] = ranked["candidate_siret"].eq(ranked["truth"])
    positive_counts = ranked.groupby("query_id")["is_truth"].sum()
    eligible = set(positive_counts[positive_counts.eq(1)].index)
    train_ids = set(
        exact.loc[exact["oof_fold"].astype(int).ne(args.fold), "query_id"].astype(str)
    ) & eligible
    validation_ids = set(
        labels.loc[labels["oof_fold"].astype(int).eq(args.fold), "query_id"].astype(str)
    )
    train_rows = ranked[ranked["query_id"].isin(train_ids)]
    positive = train_rows[train_rows["is_truth"]][["query_id", "candidate_siret"]]
    negative = (
        train_rows[~train_rows["is_truth"]]
        .sort_values(["query_id", "ranker_rank"], kind="mergesort")
        .drop_duplicates("query_id")[["query_id", "candidate_siret"]]
    )
    triples = positive.merge(negative, on="query_id", suffixes=("_positive", "_negative"), validate="one_to_one")
    validation_top = predictions[
        predictions["query_id"].isin(validation_ids) & predictions["ranker_rank"].le(args.top_n)
    ][["query_id", "candidate_siret", "candidate_siren", "retrieval_rank", "ranker_score", "ranker_rank"]]
    selected_pairs = pd.concat(
        [
            triples[["query_id", "candidate_siret_positive"]].rename(columns={"candidate_siret_positive": "candidate_siret"}),
            triples[["query_id", "candidate_siret_negative"]].rename(columns={"candidate_siret_negative": "candidate_siret"}),
            validation_top[["query_id", "candidate_siret"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["query_id", "candidate_siret"])

    source = _read_enriched_sources(
        args.raw,
        args.etablissements,
        args.unites_legales,
        candidate_filename="candidates.parquet",
    )
    source = source.merge(selected_pairs, on=["query_id", "candidate_siret"], validate="one_to_one")
    query_fields = pd.read_parquet(
        args.raw / "queries.parquet",
        columns=["query_id", "crm_address", "crm_postcode", "crm_city"],
    )
    source = source.merge(query_fields, on="query_id", validate="many_to_one")
    pair_text = _serialise_pairs(source)
    text_by_key = dict(zip(zip(source["query_id"], source["candidate_siret"]), pair_text))
    positive_text = [text_by_key[(row.query_id, row.candidate_siret_positive)] for row in triples.itertuples(index=False)]
    negative_text = [text_by_key[(row.query_id, row.candidate_siret_negative)] for row in triples.itertuples(index=False)]
    validation_text = [text_by_key[(row.query_id, row.candidate_siret)] for row in validation_top.itertuples(index=False)]
    train_query_count = len(triples)
    del source, selected_pairs, pair_text, text_by_key, ranked, train_rows
    del positive, negative, predictions
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()

    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "raw_manifest_sha256": file_sha256(args.raw / "manifest.json"),
        "business_manifest_sha256": file_sha256(args.business / "manifest.json"),
        "ranker_manifest_sha256": file_sha256(args.ranker / "manifest.json"),
        "model_config_sha256": file_sha256(args.model / "config.json"),
        "fold": args.fold,
        "top_n": args.top_n,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": args.device,
        "seed": SEED,
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        return destination
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True).to(args.device)
        dataset = PairwiseTextDataset(positive_text, negative_text)
        generator = torch.Generator().manual_seed(SEED)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=_collate(tokenizer, args.max_length),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        total_steps = max(1, len(loader) * args.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps,
        )
        losses: list[float] = []
        model.train()
        for epoch in range(args.epochs):
            for step, encoded in enumerate(loader, start=1):
                encoded = encoded.to(args.device)
                batch = encoded["input_ids"].shape[0] // 2
                logits = model(**encoded).logits.squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(logits[:batch] - logits[batch:]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
                if step % 100 == 0:
                    print(f"[ce] epoch={epoch + 1} step={step}/{len(loader)} loss={np.mean(losses[-100:]):.4f}", flush=True)
                    if args.device == "mps":
                        torch.mps.empty_cache()

        ce_scores = _score_pairs(
            model,
            tokenizer,
            validation_text,
            device=args.device,
            batch_size=args.score_batch_size,
            max_length=args.max_length,
        )
        scored = validation_top.copy()
        scored["cross_encoder_score"] = ce_scores
        for column in ("ranker_score", "cross_encoder_score"):
            grouped = scored.groupby("query_id")[column]
            scored[f"{column}_z"] = (
                (scored[column] - grouped.transform("mean"))
                / grouped.transform("std").replace(0.0, np.nan)
            ).fillna(0.0)
        local = pd.read_csv(args.local_labels, dtype=str, keep_default_na=False)
        difficult = set(local.loc[local["label_kind"].eq("MATCH_EXACT"), "query_id"].astype(str))
        fold_truth = exact[exact["oof_fold"].astype(int).eq(args.fold)][["query_id", "ground_truth_siret"]]
        variants: list[dict[str, object]] = []
        for alpha in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            scored["combined"] = scored["ranker_score_z"] + alpha * scored["cross_encoder_score_z"]
            top1 = scored.sort_values(
                ["query_id", "combined", "ranker_rank"],
                ascending=[True, False, True],
                kind="mergesort",
            ).drop_duplicates("query_id")
            detail = fold_truth.merge(top1[["query_id", "candidate_siret"]], on="query_id", how="left")
            detail["correct"] = detail["candidate_siret"].astype(str).eq(detail["ground_truth_siret"].astype(str))
            hard = detail["query_id"].isin(difficult)
            variants.append(
                {
                    "alpha": alpha,
                    "fold_exact_correct": int(detail["correct"].sum()),
                    "fold_exact_total": len(detail),
                    "fold_difficult_correct": int(detail.loc[hard, "correct"].sum()),
                    "fold_difficult_total": int(hard.sum()),
                }
            )
        scored.to_parquet(temporary / "validation_top20_scores.parquet", index=False)
        model.save_pretrained(temporary / "model")
        tokenizer.save_pretrained(temporary / "model")
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "CONSUMED_DEVELOPMENT_PILOT_FOLD",
            "fold": args.fold,
            "train_query_count": train_query_count,
            "validation_query_count": len(validation_ids),
            "mean_last_100_loss": float(np.mean(losses[-100:])),
            "elapsed_seconds": time.perf_counter() - started,
            "variants": variants,
            "full_oof": False,
            "final_test_opened": False,
        }
        (temporary / "evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "build_identity": identity,
            "outputs": {
                str(path.relative_to(temporary)): file_sha256(path)
                for path in temporary.rglob("*") if path.is_file()
            },
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--business", type=Path, default=DEFAULT_BUSINESS)
    parser.add_argument("--ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--etablissements", type=Path, default=Path("data/StockEtablissement_utf8.parquet"))
    parser.add_argument("--unites-legales", type=Path, default=Path("data/StockUniteLegale_utf8.parquet"))
    parser.add_argument("--local-labels", type=Path, default=Path("reports/v412_review_local_identifiable_labels_279.csv"))
    parser.add_argument("--output-root", type=Path, default=BASE / "experiments/v4_12_learned_cross_encoder_pilot")
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
