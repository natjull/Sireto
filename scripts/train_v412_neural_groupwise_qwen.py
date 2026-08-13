#!/usr/bin/env python3
"""MLX LoRA groupwise fine-tuning of Qwen3-Reranker-0.6B for V4.12-N."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load as mlx_load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx.utils import tree_flatten
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_neural_reranker import (  # noqa: E402
    DEFAULT_CORPUS,
    MODEL_SPECS,
    TASK,
    _evaluate,
    _load_fold,
    _model_fingerprint,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_GROUPS = BASE / "datasets/v4_12_neural_training_groups/55b5fa545d29fd26"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_neural_groupwise_qwen"
SCHEMA_VERSION = "sireto-v4.12-neural-groupwise-qwen-1"
SEED = 42
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. The answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_token_ids(
    tokenizer: Any,
    query_texts: list[str],
    candidate_texts: list[str],
    max_length: int,
) -> tuple[mx.array, mx.array]:
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    body_limit = max_length - len(prefix) - len(suffix)
    if body_limit < 64:
        raise ValueError("max_length leaves too little room for CRM/SIRENE evidence")
    bodies = [
        f"<Instruct>: {TASK}\n<Query>: {query}\n<Document>: {candidate}"
        for query, candidate in zip(query_texts, candidate_texts)
    ]
    encoded = tokenizer._tokenizer(
        bodies,
        padding=False,
        truncation=True,
        max_length=body_limit,
        add_special_tokens=False,
    )["input_ids"]
    encoded = [prefix + ids + suffix for ids in encoded]
    lengths = [len(ids) for ids in encoded]
    width = max(lengths)
    padded = mx.array(
        [ids + [tokenizer.eos_token_id] * (width - len(ids)) for ids in encoded]
    )
    return padded, mx.array(lengths)


def _group_loss(
    model: Any,
    inputs: mx.array,
    lengths: mx.array,
    group_count: int,
    group_size: int,
    true_id: int,
    false_id: int,
) -> mx.array:
    hidden = model.model(inputs)
    final_hidden = hidden[mx.arange(inputs.shape[0]), lengths - 1]
    weights = model.model.embed_tokens.weight
    scores = final_hidden @ (weights[true_id] - weights[false_id])
    scores = scores.reshape(group_count, group_size)
    return mx.mean(mx.logsumexp(scores, axis=1) - scores[:, 0])


def _score(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    *,
    batch_size: int,
    max_length: int,
    true_id: int,
    false_id: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        inputs, lengths = _prepare_token_ids(
            tokenizer,
            batch["query_text"].tolist(),
            batch["candidate_text"].tolist(),
            max_length,
        )
        hidden = model.model(inputs)
        final_hidden = hidden[mx.arange(inputs.shape[0]), lengths - 1]
        weights = model.model.embed_tokens.weight
        values = final_hidden @ (weights[true_id] - weights[false_id])
        mx.eval(values)
        output.append(np.asarray(values.tolist(), dtype=np.float32))
        if (start // batch_size + 1) % 100 == 0:
            print(f"[score] candidates={min(start + batch_size, len(frame))}/{len(frame)}", flush=True)
    return np.concatenate(output)


def run(args: argparse.Namespace) -> Path:
    spec = MODEL_SPECS["qwen_reranker"]
    group_manifest = json.loads((args.groups / "manifest.json").read_text())
    if group_manifest.get("positive_injection") is not False:
        raise ValueError("Training groups contain positive injection")
    if group_manifest.get("build_identity", {}).get("train_folds") != [2, 3, 4]:
        raise ValueError("Training groups are not restricted to folds 2/3/4")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "groups_manifest_sha256": file_sha256(args.groups / "manifest.json"),
        "corpus_manifest_sha256": file_sha256(args.corpus / "manifest.json"),
        "model_repo": spec.repo_id,
        "model_revision": spec.revision,
        "model_files": _model_fingerprint(spec.path),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "scenes_per_batch": args.scenes_per_batch,
        "lora_layers": args.lora_layers,
        "lora_rank": args.lora_rank,
        "lora_scale": args.lora_scale,
        "max_length": args.max_length,
        "score_batch_size": args.score_batch_size,
        "max_train_scenes": args.max_train_scenes,
        "selection_query_limit": args.selection_query_limit,
        "seed": SEED,
        "loss": "group_softmax_cross_entropy",
        "backend": "mlx",
        "xgboost_used_for_decision": False,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text())
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    groups = pd.read_parquet(args.groups / "training_groups.parquet").sort_values(
        ["query_id", "group_position"], kind="mergesort"
    )
    groups["query_id"] = groups["query_id"].astype(str)
    scene_ids = groups["query_id"].drop_duplicates().tolist()
    if args.max_train_scenes:
        scene_ids = scene_ids[: args.max_train_scenes]
        groups = groups[groups["query_id"].isin(set(scene_ids))].copy()
    by_query = {
        str(query_id): frame.reset_index(drop=True)
        for query_id, frame in groups.groupby("query_id", sort=False)
    }
    group_sizes = groups.groupby("query_id").size().unique()
    if len(group_sizes) != 1:
        raise ValueError("Qwen training groups have inconsistent sizes")
    group_size = int(group_sizes[0])
    if not groups.groupby("query_id")["is_positive"].sum().eq(1).all():
        raise ValueError("Qwen training groups do not have exactly one positive")

    model, tokenizer = mlx_load(str(spec.path), lazy=False)
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=args.lora_layers,
        config={
            "rank": args.lora_rank,
            "scale": args.lora_scale,
            "dropout": 0.0,
            "keys": {"self_attn.q_proj", "self_attn.v_proj"},
        },
    )
    true_id = tokenizer.convert_tokens_to_ids("yes")
    false_id = tokenizer.convert_tokens_to_ids("no")
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(model, _group_loss)
    rng = random.Random(SEED)
    losses: list[float] = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        order = list(scene_ids)
        rng.shuffle(order)
        for start in range(0, len(order), args.scenes_per_batch):
            selected = order[start : start + args.scenes_per_batch]
            frame = pd.concat([by_query[query_id] for query_id in selected], ignore_index=True)
            if not all(
                int(by_query[query_id].iloc[0]["is_positive"]) == 1
                for query_id in selected
            ):
                raise ValueError("Positive is not first in a Qwen training scene")
            inputs, lengths = _prepare_token_ids(
                tokenizer,
                frame["query_text"].tolist(),
                frame["candidate_text"].tolist(),
                args.max_length,
            )
            loss, gradients = loss_and_grad(
                model,
                inputs,
                lengths,
                len(selected),
                group_size,
                true_id,
                false_id,
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(float(loss.item()))
            step = start // args.scenes_per_batch + 1
            if step % 100 == 0 or start + args.scenes_per_batch >= len(order):
                print(
                    f"[train-qwen] epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{int(np.ceil(len(order) / args.scenes_per_batch))} "
                    f"loss={np.mean(losses[-100:]):.5f}",
                    flush=True,
                )
    training_seconds = time.perf_counter() - started

    _, selection = _load_fold(args.corpus, 0, args.selection_query_limit)
    score_started = time.perf_counter()
    selection["neural_score"] = _score(
        model,
        tokenizer,
        selection,
        batch_size=args.score_batch_size,
        max_length=args.max_length,
        true_id=true_id,
        false_id=false_id,
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
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(temporary / "adapters.safetensors"), adapter_weights)
        _json_dump(
            temporary / "adapter_config.json",
            {
                "model": spec.repo_id,
                "revision": spec.revision,
                "num_layers": args.lora_layers,
                "lora_parameters": {
                    "rank": args.lora_rank,
                    "scale": args.lora_scale,
                    "dropout": 0.0,
                    "keys": ["self_attn.q_proj", "self_attn.v_proj"],
                },
            },
        )
        scores.to_parquet(temporary / "selection_scores.parquet", index=False)
        detail.to_parquet(temporary / "selection_top1_detail.parquet", index=False)
        comparison.to_csv(temporary / "comparison.csv", index=False)
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "RESOURCE_SMOKE" if args.max_train_scenes else "SELECTION_FOLD",
            "model": "qwen_reranker",
            "train_scene_count": len(scene_ids),
            "train_pair_count": len(groups),
            "selection_query_count": selection["query_id"].nunique(),
            "selection_candidate_count": len(selection),
            "mean_last_100_loss": float(np.mean(losses[-100:])),
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
            "adapters.safetensors",
            "adapter_config.json",
            "selection_scores.parquet",
            "selection_top1_detail.parquet",
            "comparison.csv",
            "evaluation.json",
        ]
        output_manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "build_identity": identity,
            "outputs": {name: file_sha256(temporary / name) for name in output_names},
        }
        _json_dump(temporary / "manifest.json", output_manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--scenes-per-batch", type=int, default=2)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=16.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--selection-query-limit", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
