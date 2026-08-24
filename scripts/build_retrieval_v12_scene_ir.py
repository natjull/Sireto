#!/usr/bin/env python3
"""Build the compact label-blind SceneIR index on an external output root."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_v12_scene_ir import build_scene_ir_cache  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="absolute external-volume directory; workspace paths are rejected",
    )
    parser.add_argument("--batch-size", type=int, default=262_144)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build = build_scene_ir_cache(
        source=args.source,
        source_manifest=args.source_manifest,
        output_root=args.output_root,
        batch_size=args.batch_size,
    )
    print(build.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
