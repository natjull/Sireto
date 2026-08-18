#!/usr/bin/env python3
"""Build the content-addressed Tantivy official-evidence overlay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.official_evidence_tantivy import (  # noqa: E402
    build_official_evidence_tantivy_overlay,
    build_official_evidence_tantivy_overlay_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-evidence", type=Path)
    parser.add_argument("--official-relations", type=Path)
    parser.add_argument("--canonical-layer", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--writer-heap-bytes", type=int, default=256_000_000)
    parser.add_argument("--writer-threads", type=int, default=4)
    parser.add_argument("--commit-every", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--duckdb-temp-directory", type=Path)
    parser.add_argument("--duckdb-memory-limit")
    parser.add_argument("--duckdb-threads", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = dict(
        writer_heap_bytes=args.writer_heap_bytes,
        writer_threads=args.writer_threads,
        commit_every=args.commit_every,
        batch_size=args.batch_size,
        duckdb_temp_directory=args.duckdb_temp_directory,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_threads=args.duckdb_threads,
    )
    if args.canonical_layer:
        if args.official_evidence is not None or args.official_relations is not None:
            raise SystemExit("use either --canonical-layer or the single-layer arguments")
        layers = [
            (layer / "official_evidence.parquet", layer / "official_relation.parquet")
            for layer in args.canonical_layer
        ]
        output = build_official_evidence_tantivy_overlay_layers(
            layers, args.output_root, **kwargs
        )
    else:
        if args.official_evidence is None or args.official_relations is None:
            raise SystemExit("single-layer mode requires evidence and relations")
        output = build_official_evidence_tantivy_overlay(
            args.official_evidence,
            args.official_relations,
            args.output_root,
            **kwargs,
        )
    print(output)


if __name__ == "__main__":
    main()
