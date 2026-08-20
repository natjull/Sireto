#!/usr/bin/env python3
"""Build the separate SIRENE history/succession Tantivy rescue overlay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.sirene_temporal_tantivy import build_temporal_tantivy_overlay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", type=Path, action="append", default=[])
    parser.add_argument(
        "--projection-root", type=Path,
        help="Root produced by build-all; all prefix artifacts are loaded from its manifest.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--writer-heap-bytes", type=int, default=256_000_000)
    parser.add_argument("--writer-threads", type=int, default=4)
    args = parser.parse_args()
    projections = list(args.projection)
    if args.projection_root:
        import json
        manifest = json.loads((args.projection_root / "prefix_set_manifest.json").read_text())
        projections.extend(args.projection_root / item["output_dir"] for item in manifest["prefixes"])
    if not projections:
        parser.error("--projection or --projection-root is required")
    print(build_temporal_tantivy_overlay(
        projections, args.output_root,
        writer_heap_bytes=args.writer_heap_bytes, writer_threads=args.writer_threads,
    ))


if __name__ == "__main__":
    main()
