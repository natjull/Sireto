#!/usr/bin/env python3
"""Gate-0 smoke test for the local CPU semantic/FAISS V9 runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/semantic/siret-bert-deploy"),
    )
    parser.add_argument("--rows", type=int, default=256)
    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    os.environ["XGB_SEMANTIC_ENABLED"] = "1"
    os.environ["XGB_SEMANTIC_MODEL"] = str(args.model)
    os.environ["XGB_SEMANTIC_DEVICE"] = "cpu"

    import numpy as np
    import pyarrow
    import xgboost

    from src.xgb_matcher.faiss_process import (
        FaissSearchProcessClient,
        build_faiss_index_isolated,
    )
    from src.xgb_matcher.semantic_process import SemanticProcessClient

    texts = [
        f"Entreprise française numéro {index} 12 rue de Paris"
        for index in range(args.rows)
    ]
    with SemanticProcessClient(args.model, device="cpu") as encoder:
        started = time.perf_counter()
        vectors = encoder.encode(
            texts,
            batch_size=min(128, args.rows),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        elapsed = time.perf_counter() - started
        runtime_info = encoder.runtime_info
    # Execute a PyArrow kernel in the parent before FAISS. This is the order
    # that used to abort on duplicate libomp runtimes.
    arrow_rows = pyarrow.table({"row": list(range(len(vectors)))})
    if arrow_rows.num_rows != len(vectors):
        raise RuntimeError("PyArrow execution smoke test failed")
    with tempfile.TemporaryDirectory(prefix="sireto-v9-runtime-") as temp_dir:
        index_path = Path(temp_dir) / "runtime_faiss.index"
        build_faiss_index_isolated(vectors, index_path)
        faiss_client = FaissSearchProcessClient(max_cache=1)
        try:
            scores, indices = faiss_client.search(
                index_path,
                vectors[:1],
                min(5, len(vectors)),
            )
            faiss_version = faiss_client.runtime_info["faiss"]
        finally:
            faiss_client.close()
    if int(indices[0]) != 0:
        raise RuntimeError("FAISS self-neighbour smoke test failed")

    report = {
        "status": "PASS",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": runtime_info["torch"],
        "mps_built": runtime_info["mps_built"],
        "mps_available": runtime_info["mps_available"],
        "xgboost": xgboost.__version__,
        "pyarrow": pyarrow.__version__,
        "faiss": faiss_version,
        "device": "cpu",
        "semantic_runtime": "isolated_subprocess",
        "faiss_runtime": "isolated_subprocess",
        "rows": len(vectors),
        "dimension": int(vectors.shape[1]),
        "encode_seconds": elapsed,
        "rows_per_second": len(vectors) / max(elapsed, 1e-9),
        "top1_score": float(scores[0]),
        "model": str(args.model),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
