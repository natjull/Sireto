#!/usr/bin/env python3
"""Project dossier-v3 RNE/BODACC history and optionally build its overlay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.dossier_temporal_evidence import (  # noqa: E402
    materialize_dossier_temporal_layers,
)
from src.xgb_matcher.official_evidence_tantivy import (  # noqa: E402
    build_official_evidence_tantivy_overlay_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, required=True)
    parser.add_argument("--layer-output-root", type=Path, required=True)
    parser.add_argument("--name-portfolio", type=Path)
    parser.add_argument("--index-output-root", type=Path)
    parser.add_argument("--max-names", type=int, default=16)
    parser.add_argument("--max-addresses", type=int, default=16)
    parser.add_argument("--duckdb-temp-directory", type=Path)
    parser.add_argument("--duckdb-memory-limit", default="8GB")
    parser.add_argument("--duckdb-threads", type=int, default=8)
    parser.add_argument("--writer-heap-bytes", type=int, default=256_000_000)
    parser.add_argument("--writer-threads", type=int, default=4)
    parser.add_argument("--commit-every", type=int, default=250_000)
    parser.add_argument("--partition-prefix-digits", type=int, choices=(0, 1, 2), default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = materialize_dossier_temporal_layers(
        args.dossier,
        args.layer_output_root,
        name_portfolio_path=args.name_portfolio,
        max_names=args.max_names,
        max_addresses=args.max_addresses,
        duckdb_temp_directory=args.duckdb_temp_directory,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_threads=args.duckdb_threads,
    )
    print(layers)
    if args.index_output_root is None:
        return
    canonical_layers = [
        (
            layers / source / "official_evidence.parquet",
            layers / source / "official_relation.parquet",
        )
        for source in ("rne", "bodacc")
    ]
    index = build_official_evidence_tantivy_overlay_layers(
        canonical_layers,
        args.index_output_root,
        writer_heap_bytes=args.writer_heap_bytes,
        writer_threads=args.writer_threads,
        commit_every=args.commit_every,
        duckdb_temp_directory=args.duckdb_temp_directory,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_threads=args.duckdb_threads,
        duckdb_partition_prefix_digits=args.partition_prefix_digits,
    )
    print(index)


if __name__ == "__main__":
    main()
