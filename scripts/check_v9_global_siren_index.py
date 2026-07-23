#!/usr/bin/env python3
"""Check integrity and sampled self-recall of a V9 global SIREN ANN index."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v9_siren_dense_index import NAME_COLUMNS, siren_text
from src.xgb_matcher.v9_dataset import file_sha256


def sample_entities(
    source: Path,
    *,
    sample_count: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Sample valid named entities across evenly spaced parquet row groups."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    parquet = pq.ParquetFile(source)
    available = set(parquet.schema.names)
    columns = [column for column in NAME_COLUMNS if column in available]
    if "siren" not in columns:
        raise ValueError("Source parquet must contain a siren column")
    group_count = parquet.metadata.num_row_groups
    selected_groups = np.unique(
        np.linspace(
            0,
            group_count - 1,
            num=min(group_count, sample_count),
            dtype=int,
        )
    )
    rng = np.random.default_rng(seed)
    samples: list[tuple[str, str]] = []
    target_per_group = max(
        1,
        int(np.ceil(sample_count / len(selected_groups))),
    )
    for group_index in selected_groups:
        rows = parquet.read_row_group(
            int(group_index),
            columns=columns,
        ).to_pylist()
        order = rng.permutation(len(rows))
        selected = 0
        for row_index in order:
            row = rows[int(row_index)]
            siren = "".join(
                char for char in str(row.get("siren") or "") if char.isdigit()
            )
            text = siren_text(row)
            if not siren or not text:
                continue
            samples.append((siren.zfill(9), text))
            selected += 1
            if selected >= target_per_group or len(samples) >= sample_count:
                break
        if len(samples) >= sample_count:
            break
    if len(samples) < sample_count:
        raise ValueError(
            f"Only {len(samples)} valid entities sampled, expected {sample_count}"
        )
    return samples


def verify_output_hashes(index_dir: Path) -> dict[str, str]:
    manifest = json.loads(
        (index_dir / "manifest.json").read_text(encoding="utf-8")
    )
    observed = {
        name: file_sha256(index_dir / name)
        for name in manifest.get("outputs", {})
    }
    if observed != manifest.get("outputs"):
        raise ValueError("Global SIREN index output hash mismatch")
    return observed


def evaluate_self_recall(
    *,
    source: Path,
    index_dir: Path,
    model: Path,
    sample_count: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    output_hashes = verify_output_hashes(index_dir)
    samples = sample_entities(source, sample_count=sample_count, seed=seed)

    os.environ["XGB_SEMANTIC_ENABLED"] = "1"
    os.environ["XGB_SEMANTIC_MODEL"] = str(model)
    os.environ["XGB_SEMANTIC_DEVICE"] = "cpu"
    from src.xgb_matcher.dense_retrieval import GlobalDenseSirenIndex
    from src.xgb_matcher.semantic import (
        batch_encode_texts,
        semantic_artifact_fingerprint,
        set_semantic_client,
    )
    from src.xgb_matcher.semantic_process import SemanticProcessClient

    client = SemanticProcessClient(model, device="cpu")
    set_semantic_client(client)
    index = None
    try:
        model_fingerprint = semantic_artifact_fingerprint(model)
        index = GlobalDenseSirenIndex(
            index_dir,
            expected_model_fingerprint=model_fingerprint,
        )
        # Populate the query cache in one semantic IPC call so measured
        # latency reflects ANN search rather than per-query model startup.
        batch_encode_texts([text for _, text in samples])
        ranks: list[int | None] = []
        latencies_ms: list[float] = []
        for expected_siren, text in samples:
            started = time.perf_counter()
            hits = index.query(text, top_k=top_k)
            latencies_ms.append((time.perf_counter() - started) * 1000)
            returned = [siren for siren, _ in hits]
            ranks.append(
                returned.index(expected_siren) + 1
                if expected_siren in returned
                else None
            )
    finally:
        if index is not None:
            index.close()
        client.close()
        set_semantic_client(None)

    rank_values = np.asarray(
        [rank if rank is not None else top_k + 1 for rank in ranks],
        dtype=int,
    )
    latency_values = np.asarray(latencies_ms, dtype=float)
    return {
        "source": str(source),
        "index_dir": str(index_dir),
        "model": str(model),
        "sample_count": sample_count,
        "seed": seed,
        "top_k": top_k,
        "self_recall_at_1": float((rank_values == 1).mean()),
        "self_recall_at_k": float((rank_values <= top_k).mean()),
        "misses_at_k": int((rank_values > top_k).sum()),
        "latency_ms": {
            "p50": float(np.quantile(latency_values, 0.50)),
            "p95": float(np.quantile(latency_values, 0.95)),
            "mean": float(latency_values.mean()),
        },
        "index_output_hashes": output_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_self_recall(
        source=args.source,
        index_dir=args.index_dir,
        model=args.model,
        sample_count=args.sample_count,
        top_k=args.top_k,
        seed=args.seed,
    )
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"Immutable output exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
