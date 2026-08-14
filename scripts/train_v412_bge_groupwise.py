#!/usr/bin/env python3
"""Deterministic groupwise BGE training and OOF target scoring for V4.12-BGE."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_neural_reranker import (  # noqa: E402
    DEFAULT_CORPUS,
    MODEL_SPECS,
    _evaluate,
    _load_fold,
    _model_fingerprint,
)
from scripts.train_v412_neural_groupwise_cross_encoder import (  # noqa: E402
    _encode_scenes,
    _freeze_for_top_layer_training,
    _scene_batches,
    _score_candidates,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_GROUPS = BASE / "datasets/v4_12_bge_training_groups/114b407f2ccf7b40"
DEFAULT_RANKER = BASE / "experiments/v4_12_learned_oof_rankers/839ef55308d5077e"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_bge_groupwise"
SCHEMA_VERSION = "sireto-v4.12-bge-groupwise-1"
MODEL_KEY = "bge_ref"
SEED = 42


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_folds(value: str) -> tuple[int, ...]:
    folds = tuple(sorted({int(item) for item in value.split(",") if item != ""}))
    if not folds or any(fold not in {2, 3, 4} for fold in folds):
        raise argparse.ArgumentTypeError("Training folds must be a subset of 2,3,4")
    return folds


def _validate_fold_roles(train_folds: tuple[int, ...], target_fold: int) -> None:
    if target_fold not in range(5):
        raise ValueError("Target fold must be between 0 and 4")
    if target_fold in train_folds:
        raise ValueError("Target fold cannot be used to train BGE")
    if target_fold == 1:
        raise ValueError("Confirmation fold 1 is closed before the ranker gate")


def _load_target(
    corpus: Path,
    ranker: Path,
    fold: int,
    query_limit: int | None,
    business_top_k: int | None,
) -> pd.DataFrame:
    _, candidates = _load_fold(corpus, fold, query_limit)
    ranker_scores = pd.read_parquet(
        ranker / "business_learned_oof_candidates.parquet",
        columns=[
            "query_id",
            "candidate_siret",
            "candidate_siren",
            "ranker_score",
            "ranker_rank",
            "oof_fold",
        ],
    )
    ranker_scores["query_id"] = ranker_scores["query_id"].astype(str)
    ranker_scores["candidate_siret"] = ranker_scores["candidate_siret"].astype(str)
    ranker_scores = ranker_scores[ranker_scores["oof_fold"].astype(int).eq(fold)]
    candidates = candidates.merge(
        ranker_scores.drop(columns="oof_fold"),
        on=["query_id", "candidate_siret", "candidate_siren"],
        validate="one_to_one",
    )
    if business_top_k is not None:
        candidates = candidates[candidates["ranker_rank"].astype(int).le(business_top_k)].copy()
    return candidates.sort_values(
        ["query_id", "ranker_rank", "retrieval_rank", "candidate_siret"],
        kind="mergesort",
    ).reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    _validate_fold_roles(args.train_folds, args.target_fold)
    group_manifest = json.loads((args.groups / "manifest.json").read_text(encoding="utf-8"))
    if group_manifest.get("positive_injection") is not False:
        raise ValueError("BGE groups contain positive injection")
    if group_manifest.get("build_identity", {}).get("train_folds") != [2, 3, 4]:
        raise ValueError("BGE source groups are not restricted to folds 2/3/4")
    corpus_manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    ranker_manifest = json.loads((args.ranker / "manifest.json").read_text(encoding="utf-8"))
    for root, manifest, name in (
        (args.groups, group_manifest, "training_groups.parquet"),
        (args.corpus, corpus_manifest, "candidates_text.parquet"),
        (args.ranker, ranker_manifest, "business_learned_oof_candidates.parquet"),
    ):
        if manifest.get("outputs", {}).get(name) != file_sha256(root / name):
            raise ValueError(f"Input hash mismatch: {root / name}")

    spec = MODEL_SPECS[MODEL_KEY]
    model_files = _model_fingerprint(spec.path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "groups_manifest_sha256": file_sha256(args.groups / "manifest.json"),
        "corpus_manifest_sha256": file_sha256(args.corpus / "manifest.json"),
        "ranker_manifest_sha256": file_sha256(args.ranker / "manifest.json"),
        "model_key": MODEL_KEY,
        "model_repo": spec.repo_id,
        "model_revision": spec.revision,
        "model_files": model_files,
        "train_folds": list(args.train_folds),
        "target_fold": args.target_fold,
        "business_top_k": args.business_top_k,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "scenes_per_batch": args.scenes_per_batch,
        "trainable_layers": args.trainable_layers,
        "max_length": args.max_length,
        "score_batch_size": args.score_batch_size,
        "max_train_scenes": args.max_train_scenes,
        "target_query_limit": args.target_query_limit,
        "device": args.device,
        "seed": SEED,
        "loss": "group_softmax_cross_entropy",
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

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    groups = pd.read_parquet(args.groups / "training_groups.parquet")
    groups["query_id"] = groups["query_id"].astype(str)
    groups = groups[groups["oof_fold"].astype(int).isin(args.train_folds)].copy()
    groups = groups.sort_values(["query_id", "group_position"], kind="mergesort")
    if args.max_train_scenes is not None:
        keep = groups["query_id"].drop_duplicates().head(args.max_train_scenes)
        groups = groups[groups["query_id"].isin(set(keep))].copy()
    frame_by_query = {
        str(query_id): frame.reset_index(drop=True)
        for query_id, frame in groups.groupby("query_id", sort=False)
    }
    if not frame_by_query:
        raise ValueError("No BGE training scenes selected")

    tokenizer = AutoTokenizer.from_pretrained(spec.path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.path,
        local_files_only=True,
    ).to(args.device)
    parameter_counts = _freeze_for_top_layer_training(model, args.trainable_layers)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    total_steps = max(
        1,
        int(np.ceil(len(frame_by_query) / args.scenes_per_batch)) * args.epochs,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=max(1, total_steps // 10),
    )
    rng = random.Random(SEED)
    losses: list[float] = []
    training_started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        batches = _scene_batches(groups, args.scenes_per_batch, rng)
        for step, scene_ids in enumerate(batches, start=1):
            encoded, group_size = _encode_scenes(
                frame_by_query,
                scene_ids,
                tokenizer,
                args.max_length,
                args.device,
            )
            logits = model(**encoded).logits.squeeze(-1).reshape(len(scene_ids), group_size)
            target = torch.zeros(len(scene_ids), dtype=torch.long, device=args.device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0 or step == len(batches):
                print(
                    f"[train] folds={args.train_folds} epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{len(batches)} loss={np.mean(losses[-100:]):.5f}",
                    flush=True,
                )
                if args.device == "mps":
                    torch.mps.empty_cache()
    training_seconds = time.perf_counter() - training_started

    target = _load_target(
        args.corpus,
        args.ranker,
        args.target_fold,
        args.target_query_limit,
        args.business_top_k,
    )
    scoring_started = time.perf_counter()
    target["bge_score"] = _score_candidates(
        model,
        tokenizer,
        target,
        batch_size=args.score_batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    scoring_seconds = time.perf_counter() - scoring_started
    scores = target[
        [
            "query_id",
            "candidate_siret",
            "candidate_siren",
            "retrieval_rank",
            "ranker_score",
            "ranker_rank",
            "bge_score",
        ]
    ].copy()
    metrics, detail = _evaluate(
        args.corpus,
        args.target_fold,
        scores.rename(columns={"bge_score": "neural_score"}),
        set(target["query_id"]),
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        model.save_pretrained(temporary / "model")
        tokenizer.save_pretrained(temporary / "model")
        scores.to_parquet(temporary / "target_scores.parquet", index=False)
        detail.to_parquet(temporary / "target_top1_detail.parquet", index=False)
        metrics.to_csv(temporary / "target_metrics.csv", index=False)
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "RESOURCE_SMOKE" if args.max_train_scenes else "OOF_TARGET",
            "train_folds": list(args.train_folds),
            "target_fold": args.target_fold,
            "business_top_k": args.business_top_k,
            "train_scene_count": len(frame_by_query),
            "train_pair_count": len(groups),
            "target_query_count": int(target["query_id"].nunique()),
            "target_candidate_count": len(target),
            "mean_last_100_loss": float(np.mean(losses[-100:])),
            "parameter_counts": parameter_counts,
            "training_seconds": training_seconds,
            "scoring_seconds": scoring_seconds,
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "metrics": metrics.to_dict("records"),
            "positive_injection": False,
            "confirmation_fold_opened": False,
            "final_test_opened": False,
        }
        _json_dump(temporary / "evaluation.json", evaluation)
        output_names = [
            str(path.relative_to(temporary))
            for path in temporary.rglob("*")
            if path.is_file()
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "build_identity": identity,
            "outputs": {
                name: file_sha256(temporary / name) for name in sorted(output_names)
            },
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    del model, tokenizer
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--train-folds", type=_parse_folds, required=True)
    parser.add_argument("--target-fold", type=int, required=True)
    parser.add_argument("--business-top-k", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--scenes-per-batch", type=int, default=1)
    parser.add_argument("--trainable-layers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--target-query-limit", type=int)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
