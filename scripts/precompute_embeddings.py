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
        --partition-type insee \
        --mega-first \
        --mega-threshold 100000 \
        --skip-existing

    python scripts/precompute_embeddings.py \
        --partitions-dir data/candidates_v7_all \
        --output-dir data/dense_index \
        --partition-type insee \
        --dry-run \
        --sort-by-size \
        --min-rows 50000

Requires: faiss-cpu, sentence-transformers, torch
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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


def _load_insee_manifest_counts(partitions_dir: Path) -> Dict[str, int]:
    """Load optional V7 INSEE row counts manifest."""
    import pyarrow.parquet as pq

    manifest_path = partitions_dir / "manifest" / "insee_counts.parquet"
    if not manifest_path.exists():
        return {}
    try:
        table = pq.read_table(manifest_path, columns=["insee", "row_count"])
    except Exception as exc:
        _logger.warning("Could not read manifest %s: %s", manifest_path, exc)
        return {}

    counts: Dict[str, int] = {}
    for row in table.to_pylist():
        code = str(row.get("insee") or "").strip()
        if not code:
            continue
        try:
            counts[code] = int(row.get("row_count") or 0)
        except Exception:
            continue
    return counts


def _order_partition_codes(
    codes: List[str],
    partition_type: str,
    insee_counts: Dict[str, int],
    *,
    sort_by_size: bool,
    mega_first: bool,
    mega_threshold: int,
    min_rows: int,
) -> List[str]:
    """Apply ordering/filtering strategy to partition codes."""
    if partition_type != "insee":
        return sorted(codes)

    if not insee_counts:
        if sort_by_size or mega_first or min_rows > 0:
            _logger.warning(
                "Manifest counts unavailable; running INSEE partitions in lexical order without size filtering"
            )
        return sorted(codes)

    selected: List[Tuple[str, int]] = []
    for code in codes:
        count = insee_counts.get(code)
        if count is None:
            continue
        if min_rows > 0 and count < min_rows:
            continue
        selected.append((code, count))

    if mega_first and sort_by_size:
        selected.sort(key=lambda item: (item[1] < mega_threshold, -item[1], item[0]))
    elif mega_first:
        selected.sort(key=lambda item: (item[1] < mega_threshold, item[0]))
    elif sort_by_size:
        selected.sort(key=lambda item: (-item[1], item[0]))
    else:
        selected.sort(key=lambda item: item[0])

    return [code for code, _ in selected]


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
    parser.add_argument("--sort-by-size", action="store_true", help="Sort INSEE partitions by descending row_count from manifest")
    parser.add_argument("--mega-first", action="store_true", help="Process INSEE mega-communes first using manifest row_count")
    parser.add_argument("--mega-threshold", type=int, default=100000, help="Row threshold to classify INSEE as mega for --mega-first")
    parser.add_argument("--min-rows", type=int, default=0, help="Filter INSEE partitions with row_count >= min-rows (manifest required)")
    parser.add_argument("--max-partitions", type=int, default=0, help="Limit number of partitions processed per partition type")
    parser.add_argument("--log-every", type=int, default=50, help="Progress logging interval (in partitions)")
    parser.add_argument("--dry-run", action="store_true", help="Print selected partitions and exit without encoding")
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
    insee_counts = _load_insee_manifest_counts(args.partitions_dir)

    partition_types = ["insee", "cp"] if args.partition_type == "both" else [args.partition_type]

    for ptype in partition_types:
        codes = _list_partition_codes(args.partitions_dir, ptype)
        codes = _order_partition_codes(
            codes,
            ptype,
            insee_counts,
            sort_by_size=args.sort_by_size,
            mega_first=args.mega_first,
            mega_threshold=args.mega_threshold,
            min_rows=args.min_rows,
        )
        if args.max_partitions > 0:
            codes = codes[: args.max_partitions]

        _logger.info("[%s] Selected %d partitions", ptype, len(codes))
        if ptype == "insee" and insee_counts:
            est_rows = sum(insee_counts.get(code, 0) for code in codes)
            _logger.info("[%s] Estimated vectors to encode: %d", ptype, est_rows)

        if args.dry_run:
            preview = codes[:20]
            _logger.info("[%s] Dry-run preview (%d shown)", ptype, len(preview))
            for code in preview:
                if ptype == "insee" and insee_counts:
                    _logger.info("  - %s (rows=%d)", code, insee_counts.get(code, 0))
                else:
                    _logger.info("  - %s", code)
            continue

        processed = 0
        skipped_existing = 0
        skipped_empty = 0
        total_vectors = 0
        total_elapsed = 0.0

        for i, code in enumerate(codes):
            partition_key = f"{code}_" if ptype == "insee" else f"_{code}"

            if args.skip_existing and store.has_embeddings(partition_key):
                skipped_existing += 1
                continue

            candidates = _load_partition_candidates(args.partitions_dir, ptype, code)
            if not candidates:
                skipped_empty += 1
                continue

            t0 = time.perf_counter()
            embeddings = encode_candidates(candidates, model_name, args.batch_size, args.device)
            elapsed = time.perf_counter() - t0

            store.save_embeddings(partition_key, embeddings)

            # Also build and save the FAISS index
            idx = DenseIndex(embeddings)
            idx.save(store._index_path(partition_key))

            processed += 1
            total_vectors += len(candidates)
            total_elapsed += elapsed

            if (i + 1) % max(args.log_every, 1) == 0 or (i + 1) == len(codes):
                vec_rate = total_vectors / max(total_elapsed, 0.001)
                _logger.info(
                    "[%s] %d/%d  %s: %d candidates, %.1fs (%.0f vec/s) | processed=%d skipped_existing=%d skipped_empty=%d cum_vec=%d cum_rate=%.0f vec/s",
                    ptype, i + 1, len(codes), code,
                    len(candidates), elapsed,
                    len(candidates) / max(elapsed, 0.001),
                    processed,
                    skipped_existing,
                    skipped_empty,
                    total_vectors,
                    vec_rate,
                )

        _logger.info(
            "[%s] Completed: processed=%d skipped_existing=%d skipped_empty=%d total_vectors=%d",
            ptype,
            processed,
            skipped_existing,
            skipped_empty,
            total_vectors,
        )

    if args.dry_run:
        _logger.info("Dry-run completed. No embeddings were generated.")
    else:
        _logger.info("Done. Embeddings stored in %s", args.output_dir)


if __name__ == "__main__":
    main()
