#!/usr/bin/env python3
"""Groupwise fine-tuning of encoder rerankers for V4.12-N."""

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
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_GROUPS = BASE / "datasets/v4_12_neural_training_groups/55b5fa545d29fd26"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_neural_groupwise_cross_encoder"
SCHEMA_VERSION = "sireto-v4.12-neural-groupwise-cross-encoder-1"
SEED = 42
SUPPORTED_MODELS = {"gte_reranker", "camembert_fr"}


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _encoder_layers(model: Any) -> tuple[list[Any], list[Any]]:
    if hasattr(model, "new"):
        return list(model.new.encoder.layer), [model.new.pooler, model.classifier]
    if hasattr(model, "roberta"):
        extras = [model.classifier]
        if getattr(model.roberta, "pooler", None) is not None:
            extras.append(model.roberta.pooler)
        return list(model.roberta.encoder.layer), extras
    raise TypeError(f"Unsupported sequence-classifier architecture: {type(model).__name__}")


def _freeze_for_top_layer_training(model: Any, trainable_layers: int) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers, extras = _encoder_layers(model)
    if not 1 <= trainable_layers <= len(layers):
        raise ValueError("trainable_layers is outside the encoder depth")
    for layer in layers[-trainable_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for module in extras:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def _scene_batches(
    frame: pd.DataFrame, scenes_per_batch: int, rng: random.Random
) -> list[list[str]]:
    scene_ids = frame["query_id"].drop_duplicates().astype(str).tolist()
    rng.shuffle(scene_ids)
    return [scene_ids[index : index + scenes_per_batch] for index in range(0, len(scene_ids), scenes_per_batch)]


def _encode_scenes(
    frame_by_query: dict[str, pd.DataFrame],
    scene_ids: list[str],
    tokenizer: Any,
    max_length: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], int]:
    rows = [frame_by_query[query_id] for query_id in scene_ids]
    group_sizes = {len(row) for row in rows}
    if len(group_sizes) != 1:
        raise ValueError("A minibatch mixes different group sizes")
    group_size = group_sizes.pop()
    for row in rows:
        if int(row.iloc[0]["is_positive"]) != 1 or int(row["is_positive"].sum()) != 1:
            raise ValueError("Positive must be first and unique in every scene")
    flattened = pd.concat(rows, ignore_index=True)
    encoded = tokenizer(
        flattened["query_text"].tolist(),
        flattened["candidate_text"].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}, group_size


def _score_candidates(
    model: Any,
    tokenizer: Any,
    candidates: pd.DataFrame,
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            batch = candidates.iloc[start : start + batch_size]
            encoded = tokenizer(
                batch["query_text"].tolist(),
                batch["candidate_text"].tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1)
            output.append(logits.detach().float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def run(args: argparse.Namespace) -> Path:
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(f"Groupwise encoder trainer supports {sorted(SUPPORTED_MODELS)}")
    spec = MODEL_SPECS[args.model]
    group_manifest = json.loads((args.groups / "manifest.json").read_text())
    if group_manifest.get("positive_injection") is not False:
        raise ValueError("Training groups contain positive injection")
    if group_manifest.get("build_identity", {}).get("train_folds") != [2, 3, 4]:
        raise ValueError("Training groups are not restricted to folds 2/3/4")
    model_files = _model_fingerprint(spec.path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "groups_manifest_sha256": file_sha256(args.groups / "manifest.json"),
        "corpus_manifest_sha256": file_sha256(args.corpus / "manifest.json"),
        "model_key": spec.key,
        "model_repo": spec.repo_id,
        "model_revision": spec.revision,
        "code_revision": spec.code_revision,
        "model_files": model_files,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "scenes_per_batch": args.scenes_per_batch,
        "trainable_layers": args.trainable_layers,
        "max_length": args.max_length,
        "score_batch_size": args.score_batch_size,
        "max_train_scenes": args.max_train_scenes,
        "selection_query_limit": args.selection_query_limit,
        "device": args.device,
        "seed": SEED,
        "loss": "group_softmax_cross_entropy",
        "xgboost_used_for_decision": False,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / spec.key / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text())
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    groups = pd.read_parquet(args.groups / "training_groups.parquet")
    groups["query_id"] = groups["query_id"].astype(str)
    groups = groups.sort_values(["query_id", "group_position"], kind="mergesort")
    if args.max_train_scenes:
        keep = groups["query_id"].drop_duplicates().head(args.max_train_scenes)
        groups = groups[groups["query_id"].isin(set(keep))].copy()
    frame_by_query = {
        str(query_id): frame.reset_index(drop=True)
        for query_id, frame in groups.groupby("query_id", sort=False)
    }

    tokenizer = AutoTokenizer.from_pretrained(spec.path, local_files_only=True)
    model_kwargs: dict[str, Any] = {}
    if spec.code_revision:
        model_kwargs["code_revision"] = spec.code_revision
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.path,
        local_files_only=True,
        trust_remote_code=bool(spec.code_revision),
        **model_kwargs,
    ).to(args.device)
    parameter_counts = _freeze_for_top_layer_training(model, args.trainable_layers)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    total_steps = max(1, int(np.ceil(len(frame_by_query) / args.scenes_per_batch)) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(1, total_steps // 10)
    )
    rng = random.Random(SEED)
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        batches = _scene_batches(groups, args.scenes_per_batch, rng)
        for step, scene_ids in enumerate(batches, start=1):
            encoded, group_size = _encode_scenes(
                frame_by_query, scene_ids, tokenizer, args.max_length, args.device
            )
            logits = model(**encoded).logits.squeeze(-1).reshape(len(scene_ids), group_size)
            target = torch.zeros(len(scene_ids), dtype=torch.long, device=args.device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0 or step == len(batches):
                print(
                    f"[train] epoch={epoch + 1}/{args.epochs} step={step}/{len(batches)} "
                    f"loss={np.mean(losses[-100:]):.5f}",
                    flush=True,
                )
                if args.device == "mps":
                    torch.mps.empty_cache()
    training_seconds = time.perf_counter() - started

    _, selection = _load_fold(args.corpus, 0, args.selection_query_limit)
    score_started = time.perf_counter()
    selection["neural_score"] = _score_candidates(
        model,
        tokenizer,
        selection,
        batch_size=args.score_batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    scoring_seconds = time.perf_counter() - score_started
    scores = selection[
        ["query_id", "candidate_siret", "candidate_siren", "retrieval_rank", "neural_score"]
    ].copy()
    metrics, detail = _evaluate(args.corpus, 0, scores, set(selection["query_id"]))
    if args.selection_query_limit is None:
        baseline = pd.read_csv(args.corpus / "baseline_by_fold.csv")
        baseline = baseline[baseline["fold"].astype(str).eq("0")]
        comparison = metrics.merge(
            baseline[["segment", "correct", "total", "hit_at_1"]],
            on="segment",
            suffixes=("_neural", "_baseline"),
            validate="one_to_one",
        )
        comparison["delta_correct"] = comparison["correct_neural"] - comparison["correct_baseline"]
        comparison["delta_hit_at_1"] = comparison["hit_at_1_neural"] - comparison["hit_at_1_baseline"]
    else:
        comparison = metrics

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=destination.parent))
    try:
        model.save_pretrained(temporary / "model")
        tokenizer.save_pretrained(temporary / "model")
        scores.to_parquet(temporary / "selection_scores.parquet", index=False)
        detail.to_parquet(temporary / "selection_top1_detail.parquet", index=False)
        comparison.to_csv(temporary / "comparison.csv", index=False)
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "RESOURCE_SMOKE" if args.max_train_scenes else "SELECTION_FOLD",
            "model": spec.key,
            "train_scene_count": len(frame_by_query),
            "train_pair_count": len(groups),
            "selection_query_count": selection["query_id"].nunique(),
            "selection_candidate_count": len(selection),
            "mean_last_100_loss": float(np.mean(losses[-100:])),
            "parameter_counts": parameter_counts,
            "training_seconds": training_seconds,
            "scoring_seconds": scoring_seconds,
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "metrics": comparison.to_dict("records"),
            "positive_injection": False,
            "xgboost_used_for_decision": False,
            "confirmation_fold_opened": False,
            "final_test_opened": False,
        }
        _json_dump(temporary / "evaluation.json", evaluation)
        output_names = [
            str(path.relative_to(temporary))
            for path in temporary.rglob("*")
            if path.is_file()
        ]
        output_manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "build_identity": identity,
            "outputs": {
                name: file_sha256(temporary / name) for name in sorted(output_names)
            },
        }
        _json_dump(temporary / "manifest.json", output_manifest)
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
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--scenes-per-batch", type=int, default=1)
    parser.add_argument("--trainable-layers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--selection-query-limit", type=int)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
