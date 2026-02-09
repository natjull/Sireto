#!/usr/bin/env python3
"""Pre-compute dense embeddings per partition for hybrid FAISS retrieval.

For each partition (insee or cp), encodes all candidate names using the
semantic model (MiniLM or finetuned siret-bert) and stores:
  - <store_dir>/<partition_key>_embeddings.npy  (N x dim, float32)
  - <store_dir>/<partition_key>_faiss.index      (serialized FAISS)

Usage:
    python scripts/precompute_embeddings.py \
        --partitions-dir data/candidates_v7_all \
        --output-dir data/dense_index \
        --batch-size 256

Requires: faiss-cpu, sentence-transformers, torch
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.naming import candidate_tfidf_text
from src.xgb_matcher.dense_retrieval import DenseIndex, PartitionEmbeddingStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _load_partition_candidates(partitions_dir: Path, partition_type: str, code: str) -> List[dict]:
    """Load candidates for a single partition key."""
    import pyarrow.dataset as ds
    import pyarrow as pa

    part_dir = partitions_dir / partition_type
    if not part_dir.exists():
        return []

    col_name = "insee" if partition_type == "insee" else "postcode"
    partitioning = ds.partitioning(pa.schema([(col_name, pa.string())]), flavor="hive")
    dataset = ds.dataset(part_dir, format="parquet", partitioning=partitioning)

    try:
        table = dataset.to_table(filter=ds.field(col_name) == code)
        return table.to_pylist()
    except Exception as exc:
        _logger.warning("Failed to load %s=%s: %s", partition_type, code, exc)
        return []


def _list_partition_codes(partitions_dir: Path, partition_type: str) -> List[str]:
    """List all partition codes from the directory structure."""
    part_dir = partitions_dir / partition_type
    if not part_dir.exists():
        return []
    codes = []
    col_name = "insee" if partition_type == "insee" else "postcode"
    for subdir in sorted(part_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith(f"{col_name}="):
            code = subdir.name.split("=", 1)[1]
            if code:
                codes.append(code)
    return codes


def encode_candidates(
    candidates: List[dict],
    model_name: str,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Encode all candidate names into dense embeddings.

    Each candidate's bag-of-names (via candidate_tfidf_text) is encoded.
    Returns (N, dim) float32 array with L2-normalized vectors.
    """
    from sentence_transformers import SentenceTransformer
    from src.xgb_matcher.semantic import _normalize_for_embedding

    if device is None:
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    encoder = SentenceTransformer(model_name, device=device)

    texts = [_normalize_for_embedding(candidate_tfidf_text(c)) or " " for c in candidates]

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute dense embeddings per partition.")
    parser.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v7_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dense_index"))
    parser.add_argument("--partition-type", choices=["insee", "cp", "both"], default="both")
    parser.add_argument("--model", type=str, default=None,
                        help="Sentence transformer model (default: auto-detect finetuned or paraphrase-multilingual-MiniLM)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None, help="torch device (auto: mps > cpu)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip partitions with existing embeddings")
    args = parser.parse_args()

    # Auto-detect model
    model_name = args.model
    if model_name is None:
        local_model = Path("models/semantic/siret-bert-deploy")
        if local_model.exists():
            model_name = str(local_model)
            _logger.info("Using finetuned model: %s", model_name)
        else:
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            _logger.info("Using default model: %s", model_name)

    store = PartitionEmbeddingStore(args.output_dir)

    partition_types = ["insee", "cp"] if args.partition_type == "both" else [args.partition_type]

    for ptype in partition_types:
        codes = _list_partition_codes(args.partitions_dir, ptype)
        _logger.info("[%s] Found %d partitions", ptype, len(codes))

        for i, code in enumerate(codes):
            partition_key = f"{code}_" if ptype == "insee" else f"_{code}"

            if args.skip_existing and store.has_embeddings(partition_key):
                continue

            candidates = _load_partition_candidates(args.partitions_dir, ptype, code)
            if not candidates:
                continue

            t0 = time.perf_counter()
            embeddings = encode_candidates(candidates, model_name, args.batch_size, args.device)
            elapsed = time.perf_counter() - t0

            store.save_embeddings(partition_key, embeddings)

            # Also build and save the FAISS index
            idx = DenseIndex(embeddings)
            idx.save(store._index_path(partition_key))

            if (i + 1) % 50 == 0 or (i + 1) == len(codes):
                _logger.info(
                    "[%s] %d/%d  %s: %d candidates, %.1fs (%.0f vec/s)",
                    ptype, i + 1, len(codes), code,
                    len(candidates), elapsed,
                    len(candidates) / max(elapsed, 0.001),
                )

    _logger.info("Done. Embeddings stored in %s", args.output_dir)


if __name__ == "__main__":
    main()
