#!/usr/bin/env python3
"""Pre-compute dense embeddings per partition for hybrid FAISS retrieval.

For each partition (insee or cp), encodes all canonical retrieval candidates
using the semantic model (MiniLM or finetuned siret-bert) and stores:
  - <store_dir>/<partition_key>_faiss.index   (serialized FAISS)
  - <store_dir>/<partition_key>_manifest.json (row-order contract)

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
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.naming import candidate_tfidf_text

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
    encoder: Any,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode all candidate names into dense embeddings.

    Each candidate's bag-of-names (via candidate_tfidf_text) is encoded.
    Returns (N, dim) float32 array with L2-normalized vectors.
    """
    from src.xgb_matcher.semantic import _normalize_for_embedding

    # The semantic encoder truncates at its token limit. Capping before IPC
    # avoids pathological 70k-character bags caused by aggregated director
    # names, without removing information the model could consume.
    texts = [
        (_normalize_for_embedding(candidate_tfidf_text(c)) or " ")[:2048]
        for c in candidates
    ]

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def prepare_candidates_for_index(
    candidates: List[dict],
    *,
    drop_unnamed: bool,
    include_closed: bool,
) -> List[dict]:
    """Apply the exact canonical retrieval filtering and ordering contract."""
    from src.xgb_matcher.blocking import dedupe_candidates
    from src.xgb_matcher.retrieval import _apply_filters

    filtered = _apply_filters(candidates, drop_unnamed, include_closed)
    return list(dedupe_candidates(filtered).values())


def _siret_order_sha256(candidates: List[dict]) -> str:
    payload = "\n".join(str(candidate.get("siret") or "") for candidate in candidates)
    return hashlib.sha256(payload.encode("ascii", errors="ignore")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute dense embeddings per partition.")
    parser.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v7_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dense_index"))
    parser.add_argument("--partition-type", choices=["insee", "cp", "both"], default="both")
    parser.add_argument("--model", type=str, default=None,
                        help="Sentence transformer model (default: auto-detect finetuned or paraphrase-multilingual-MiniLM)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None, help="torch device (default: cpu)")
    parser.add_argument(
        "--drop-unnamed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror RetrievalConfigV1.drop_unnamed (default: true)",
    )
    parser.add_argument(
        "--include-closed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror RetrievalConfigV1.include_closed (default: true)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip partitions with existing embeddings")
    parser.add_argument("--sort-by-size", action="store_true", help="Sort INSEE partitions by descending row_count from manifest")
    parser.add_argument("--mega-first", action="store_true", help="Process INSEE mega-communes first using manifest row_count")
    parser.add_argument("--mega-threshold", type=int, default=100000, help="Row threshold to classify INSEE as mega for --mega-first")
    parser.add_argument("--min-rows", type=int, default=0, help="Filter INSEE partitions with row_count >= min-rows (manifest required)")
    parser.add_argument("--max-partitions", type=int, default=0, help="Limit number of partitions processed per partition type")
    parser.add_argument(
        "--partition-plan",
        type=Path,
        help="Frozen JSON plan with insee_codes/postcode_codes",
    )
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

    from src.xgb_matcher.semantic import semantic_artifact_fingerprint

    model_fingerprint = semantic_artifact_fingerprint(model_name)
    partition_plan = None
    partition_plan_hash = None
    if args.partition_plan is not None:
        partition_plan = json.loads(args.partition_plan.read_text(encoding="utf-8"))
        partition_plan_hash = hashlib.sha256(
            args.partition_plan.read_bytes()
        ).hexdigest()

    encoder = None
    if not args.dry_run:
        from src.xgb_matcher.semantic_process import SemanticProcessClient

        # V9's no-GPU contract deliberately defaults to CPU. Importing Torch
        # in this FAISS parent process would defeat OpenMP isolation.
        device = args.device or "cpu"

        _logger.info("Starting isolated SentenceTransformer worker on device=%s", device)
        encoder = SemanticProcessClient(model_name, device=device)

    insee_counts = _load_insee_manifest_counts(args.partitions_dir)
    partition_types = ["insee", "cp"] if args.partition_type == "both" else [args.partition_type]

    for ptype in partition_types:
        codes = _list_partition_codes(args.partitions_dir, ptype)
        if partition_plan is not None:
            planned = set(
                str(code)
                for code in partition_plan.get(f"{ptype}_codes", [])
            )
            available = set(codes)
            missing_codes = sorted(planned - available)
            if missing_codes:
                raise ValueError(
                    f"Partition plan references missing {ptype} codes: "
                    f"{missing_codes[:20]}"
                )
            codes = [code for code in codes if code in planned]
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
            safe_key = (
                partition_key.replace("|", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )
            index_path = args.output_dir / f"{safe_key}_faiss.index"
            manifest_path = args.output_dir / f"{safe_key}_manifest.json"

            if (
                args.skip_existing
                and index_path.exists()
                and manifest_path.exists()
            ):
                skipped_existing += 1
                continue

            candidates = _load_partition_candidates(args.partitions_dir, ptype, code)
            if not candidates:
                skipped_empty += 1
                continue
            raw_candidate_count = len(candidates)
            candidates = prepare_candidates_for_index(
                candidates,
                drop_unnamed=args.drop_unnamed,
                include_closed=args.include_closed,
            )
            if not candidates:
                skipped_empty += 1
                continue

            t0 = time.perf_counter()
            embeddings = encode_candidates(candidates, encoder, args.batch_size)
            elapsed = time.perf_counter() - t0

            # Build and save only the FAISS index. FAISS runs in its own
            # interpreter because executing FAISS and PyArrow kernels in the
            # same macOS process aborts on duplicate libomp runtimes.
            from src.xgb_matcher.faiss_process import build_faiss_index_isolated

            build_faiss_index_isolated(embeddings, index_path)
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "v9-partition-dense-1",
                        "partition_key": partition_key,
                        "partition_type": ptype,
                        "partition_code": code,
                        "raw_candidate_count": raw_candidate_count,
                        "candidate_count": len(candidates),
                        "siret_order_sha256": _siret_order_sha256(candidates),
                        "drop_unnamed": args.drop_unnamed,
                        "include_closed": args.include_closed,
                        "model": str(model_name),
                        "semantic_model_fingerprint": model_fingerprint,
                        "partition_plan_sha256": partition_plan_hash,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

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
    if encoder is not None:
        encoder.close()

    if args.dry_run:
        _logger.info("Dry-run completed. No embeddings were generated.")
    else:
        _logger.info("Done. Embeddings stored in %s", args.output_dir)


if __name__ == "__main__":
    main()
