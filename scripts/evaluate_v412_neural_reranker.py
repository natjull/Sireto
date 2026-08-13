#!/usr/bin/env python3
"""Zero-shot top-100 benchmark for the preregistered V4.12-N rerankers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
from typing import Any, Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import duckdb
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
CACHE = BASE / "models/huggingface/hub"
DEFAULT_CORPUS = BASE / "datasets/v4_12_neural_text_corpus/02b8668f8050c5e9"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_neural_zero_shot"
SCHEMA_VERSION = "sireto-v4.12-neural-zero-shot-1"
TASK = (
    "Given a French CRM record and a French SIRENE establishment, determine "
    "whether they refer to the exact same legal establishment (the same SIRET "
    "site). Consider legal and trade names, address, municipality, activity, "
    "head-office status and operating status. Do not accept merely the same "
    "legal unit when the physical site differs."
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    revision: str
    relative_path: str
    kind: str
    batch_size: int
    code_revision: str | None = None

    @property
    def path(self) -> Path:
        return CACHE / self.relative_path / "snapshots" / self.revision


MODEL_SPECS = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "qwen_reranker",
            "Qwen/Qwen3-Reranker-0.6B",
            "e61197ed45024b0ed8a2d74b80b4d909f1255473",
            "models--Qwen--Qwen3-Reranker-0.6B",
            "qwen_causal",
            16,
        ),
        ModelSpec(
            "gte_reranker",
            "Alibaba-NLP/gte-multilingual-reranker-base",
            "8215cf04918ba6f7b6a62bb44238ce2953d8831c",
            "models--Alibaba-NLP--gte-multilingual-reranker-base",
            "cross_encoder",
            32,
            "40ced75c3017eb27626c9d4ea981bde21a2662f4",
        ),
        ModelSpec(
            "camembert_fr",
            "antoinelouis/crossencoder-camembert-large-mmarcoFR",
            "8636e2f548bfce7576808c40b454606c7a881d31",
            "models--antoinelouis--crossencoder-camembert-large-mmarcoFR",
            "cross_encoder",
            16,
        ),
        ModelSpec(
            "mminilm_ref",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "1427fd652930e4ba29e8149678df786c240d8825",
            "models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1",
            "cross_encoder",
            64,
        ),
        ModelSpec(
            "bge_ref",
            "BAAI/bge-reranker-v2-m3",
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "models--BAAI--bge-reranker-v2-m3",
            "cross_encoder",
            16,
        ),
    )
}


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _model_fingerprint(path: Path) -> dict[str, str]:
    names = [
        item
        for item in path.rglob("*")
        if item.is_file() and item.name not in {"README.md", ".gitattributes"}
    ]
    return {str(item.relative_to(path)): file_sha256(item) for item in sorted(names)}


def _load_fold(corpus: Path, fold: int, query_limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    with duckdb.connect() as connection:
        queries = connection.execute(
            """
            SELECT q.query_id, q.query_text
            FROM read_parquet(?) q
            INNER JOIN read_parquet(?) l USING (query_id)
            WHERE l.label_kind = 'MATCH_EXACT' AND l.oof_fold = ?
            ORDER BY q.query_id
            """,
            [str(corpus / "queries_text.parquet"), str(corpus / "labels.parquet"), fold],
        ).fetchdf()
        queries["query_id"] = queries["query_id"].astype(str)
        if query_limit is not None:
            queries = queries.head(query_limit).copy()
        ids = queries[["query_id"]]
        connection.register("selected_queries", ids)
        candidates = connection.execute(
            """
            SELECT c.query_id, c.candidate_siret, c.candidate_siren,
                   c.retrieval_rank, c.candidate_text
            FROM read_parquet(?) c
            INNER JOIN selected_queries s USING (query_id)
            ORDER BY c.query_id, c.retrieval_rank, c.candidate_siret
            """,
            [str(corpus / "candidates_text.parquet")],
        ).fetchdf()
    candidates["query_id"] = candidates["query_id"].astype(str)
    candidates["candidate_siret"] = candidates["candidate_siret"].astype(str)
    candidates = candidates.merge(queries, on="query_id", validate="many_to_one")
    return queries, candidates


def _qwen_scorer(
    spec: ModelSpec, device: str, max_length: int, batch_size: int
) -> tuple[Callable[[list[tuple[str, str]]], np.ndarray], Callable[[], None]]:
    # MLX executes the native Qwen decoder substantially faster than PyTorch
    # MPS on Apple Silicon. It uses the exact pinned safetensors, not a
    # quantized or converted checkpoint.
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(str(spec.path), lazy=False)
    true_id = tokenizer.convert_tokens_to_ids("yes")
    false_id = tokenizer.convert_tokens_to_ids("no")
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements based on '
        'the Query and the Instruct provided. The answer can only be "yes" or "no".'
        '<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    body_limit = max_length - len(prefix_ids) - len(suffix_ids)
    if body_limit < 64:
        raise ValueError("Qwen max_length leaves too little room for evidence")

    def score(pairs: list[tuple[str, str]]) -> np.ndarray:
        output: list[np.ndarray] = []
        bodies = [
            f"<Instruct>: {TASK}\n<Query>: {query}\n<Document>: {document}"
            for query, document in pairs
        ]
        for start in range(0, len(bodies), batch_size):
            encoded = tokenizer._tokenizer(
                bodies[start : start + batch_size],
                padding=False,
                truncation=True,
                max_length=body_limit,
                add_special_tokens=False,
            )["input_ids"]
            encoded = [prefix_ids + ids + suffix_ids for ids in encoded]
            lengths = [len(ids) for ids in encoded]
            width = max(lengths)
            padded = mx.array(
                [ids + [tokenizer.eos_token_id] * (width - len(ids)) for ids in encoded]
            )
            hidden = model.model(padded)
            final_hidden = hidden[mx.arange(len(encoded)), mx.array(lengths) - 1]
            weights = model.model.embed_tokens.weight
            values = final_hidden @ (weights[true_id] - weights[false_id])
            mx.eval(values)
            output.append(np.asarray(values.tolist(), dtype=np.float32))
        return np.concatenate(output).astype(np.float32)

    def close() -> None:
        nonlocal model, tokenizer
        del model, tokenizer
        gc.collect()
        mx.clear_cache()

    return score, close


def _cross_encoder_scorer(
    spec: ModelSpec, device: str, max_length: int, batch_size: int
) -> tuple[Callable[[list[tuple[str, str]]], np.ndarray], Callable[[], None]]:
    model_kwargs: dict[str, Any] = {}
    if spec.code_revision:
        model_kwargs["code_revision"] = spec.code_revision
    model = CrossEncoder(
        str(spec.path),
        device=device,
        max_length=max_length,
        trust_remote_code=bool(spec.code_revision),
        local_files_only=True,
        model_kwargs=model_kwargs,
    )

    def score(pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray(
            model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        ).reshape(-1)

    def close() -> None:
        nonlocal model
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    return score, close


def _score_by_query_chunks(
    candidates: pd.DataFrame,
    scorer: Callable[[list[tuple[str, str]]], np.ndarray],
    query_chunk_size: int,
) -> tuple[np.ndarray, list[float]]:
    query_ids = candidates["query_id"].drop_duplicates().tolist()
    scores = np.empty(len(candidates), dtype=np.float32)
    chunk_latencies: list[float] = []
    row_start = 0
    for chunk_start in range(0, len(query_ids), query_chunk_size):
        chunk_ids = set(query_ids[chunk_start : chunk_start + query_chunk_size])
        row_end = row_start
        while row_end < len(candidates) and candidates.iloc[row_end]["query_id"] in chunk_ids:
            row_end += 1
        chunk = candidates.iloc[row_start:row_end]
        pairs = list(zip(chunk["query_text"].tolist(), chunk["candidate_text"].tolist()))
        started = time.perf_counter()
        values = scorer(pairs)
        elapsed = time.perf_counter() - started
        if len(values) != len(chunk):
            raise ValueError("Reranker returned a different score count")
        scores[row_start:row_end] = values
        chunk_latencies.append(elapsed / max(1, len(chunk_ids)))
        row_start = row_end
        done = min(chunk_start + query_chunk_size, len(query_ids))
        if done % 100 == 0 or done == len(query_ids):
            print(f"[reranker] queries={done}/{len(query_ids)}", flush=True)
    if row_start != len(candidates):
        raise ValueError("Candidate ordering is not contiguous by query")
    return scores, chunk_latencies


def _evaluate(
    corpus: Path, fold: int, scored: pd.DataFrame, query_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(corpus / "labels.parquet")
    labels["query_id"] = labels["query_id"].astype(str)
    truth = labels[
        labels["query_id"].isin(query_ids) & labels["label_kind"].eq("MATCH_EXACT")
    ].copy()
    top1 = (
        scored.sort_values(
            ["query_id", "neural_score", "retrieval_rank", "candidate_siret"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates("query_id")
        [["query_id", "candidate_siret", "neural_score", "retrieval_rank"]]
    )
    detail = truth.merge(top1, on="query_id", how="left", validate="one_to_one")
    detail["correct"] = detail["candidate_siret"].astype("string").eq(
        detail["ground_truth_siret"].astype("string")
    ).fillna(False)
    masks = {
        "exact": pd.Series(True, index=detail.index),
        "difficult": detail["label_is_human_validated"].astype(bool),
        "active": detail["ground_truth_state"].eq("A"),
        "closed": detail["ground_truth_state"].eq("F"),
    }
    rows = []
    for segment, mask in masks.items():
        selected = detail[mask]
        rows.append(
            {
                "fold": fold,
                "segment": segment,
                "correct": int(selected["correct"].sum()),
                "total": len(selected),
                "hit_at_1": float(selected["correct"].mean()) if len(selected) else None,
            }
        )
    return pd.DataFrame(rows), detail


def run(args: argparse.Namespace) -> Path:
    spec = MODEL_SPECS[args.model]
    if not spec.path.exists():
        raise FileNotFoundError(f"Pinned model is not local: {spec.path}")
    corpus_manifest = json.loads((args.corpus / "manifest.json").read_text())
    if corpus_manifest.get("positive_injection") is not False:
        raise ValueError("Benchmark corpus contains positive injection")
    if int(corpus_manifest.get("candidate_ceiling", 0)) != 100:
        raise ValueError("Benchmark corpus is not top-100")
    model_files = _model_fingerprint(spec.path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "corpus_manifest_sha256": file_sha256(args.corpus / "manifest.json"),
        "model_key": spec.key,
        "model_repo": spec.repo_id,
        "model_revision": spec.revision,
        "code_revision": spec.code_revision,
        "model_files": model_files,
        "fold": args.fold,
        "query_limit": args.query_limit,
        "max_length": args.max_length,
        "batch_size": args.batch_size or spec.batch_size,
        "query_chunk_size": args.query_chunk_size,
        "device": args.device,
        "task": TASK,
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

    queries, candidates = _load_fold(args.corpus, args.fold, args.query_limit)
    if candidates.groupby("query_id").size().max() > 100:
        raise ValueError("Scoring input exceeds 100 candidates")
    if set(candidates["query_id"]) != set(queries["query_id"]):
        raise ValueError("A selected query has no candidate")
    batch_size = args.batch_size or spec.batch_size
    load_started = time.perf_counter()
    if spec.kind == "qwen_causal":
        scorer, close = _qwen_scorer(spec, args.device, args.max_length, batch_size)
    else:
        scorer, close = _cross_encoder_scorer(spec, args.device, args.max_length, batch_size)
    model_load_seconds = time.perf_counter() - load_started
    started = time.perf_counter()
    try:
        scores, latencies = _score_by_query_chunks(
            candidates, scorer, args.query_chunk_size
        )
    finally:
        close()
    scoring_seconds = time.perf_counter() - started
    candidates["neural_score"] = scores
    scored = candidates[
        ["query_id", "candidate_siret", "candidate_siren", "retrieval_rank", "neural_score"]
    ].copy()
    metrics, detail = _evaluate(
        args.corpus, args.fold, scored, set(queries["query_id"])
    )
    if args.query_limit is None:
        baseline = pd.read_csv(args.corpus / "baseline_by_fold.csv")
        baseline = baseline[baseline["fold"].astype(str).eq(str(args.fold))].copy()
        comparison = metrics.merge(
            baseline[["segment", "correct", "total", "hit_at_1"]],
            on="segment",
            suffixes=("_neural", "_baseline"),
            validate="one_to_one",
        )
        comparison["delta_correct"] = comparison["correct_neural"] - comparison["correct_baseline"]
        comparison["delta_hit_at_1"] = comparison["hit_at_1_neural"] - comparison["hit_at_1_baseline"]
    else:
        comparison = metrics.rename(
            columns={
                "correct": "correct_neural",
                "total": "total_neural",
                "hit_at_1": "hit_at_1_neural",
            }
        )
        comparison["correct_baseline"] = None
        comparison["total_baseline"] = None
        comparison["hit_at_1_baseline"] = None
        comparison["delta_correct"] = None
        comparison["delta_hit_at_1"] = None

    args.output_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=destination.parent))
    try:
        scored.to_parquet(temporary / "scores.parquet", index=False)
        detail.to_parquet(temporary / "top1_detail.parquet", index=False)
        comparison.to_csv(temporary / "comparison.csv", index=False)
        resources = {
            "model_load_seconds": model_load_seconds,
            "scoring_seconds": scoring_seconds,
            "candidates_per_second": len(scored) / scoring_seconds,
            "mean_seconds_per_query": scoring_seconds / len(queries),
            "p95_chunk_seconds_per_query": float(np.quantile(latencies, 0.95)),
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "mps_driver_allocated_bytes_after_close": (
                int(torch.mps.driver_allocated_memory()) if args.device == "mps" else None
            ),
        }
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "RESOURCE_SMOKE" if args.query_limit else "SELECTION_FOLD",
            "model": spec.key,
            "fold": args.fold,
            "query_count": len(queries),
            "candidate_count": len(scored),
            "metrics": comparison.to_dict("records"),
            "resources": resources,
            "positive_injection": False,
            "candidate_ceiling": int(candidates.groupby("query_id").size().max()),
            "xgboost_used_for_decision": False,
            "final_test_opened": False,
        }
        _json_dump(temporary / "evaluation.json", evaluation)
        output_names = ["scores.parquet", "top1_detail.parquet", "comparison.csv", "evaluation.json"]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "build_identity": identity,
            "outputs": {name: file_sha256(temporary / name) for name in output_names},
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--query-chunk-size", type=int, default=10)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
