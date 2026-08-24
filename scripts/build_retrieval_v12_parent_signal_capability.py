#!/usr/bin/env python3
"""Build the exhaustive capability for the v12 parent-only Arrow scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_v12_parent_signal_capability import (  # noqa: E402
    build_parent_signal_capability_v12,
)
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively prove that all v11 channels/non-null rank columns "
            "are represented in the sealed signals list"
        )
    )
    parser.add_argument("--scene-ir", type=Path, required=True)
    parser.add_argument("--source-override", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache = SceneIRCache(
        args.scene_ir,
        source_override=args.source_override,
        verify_source_hash=True,
    )
    result = build_parent_signal_capability_v12(
        cache,
        output_root=args.output_root,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "build_id": result.build_id,
                "candidate_rows": result.candidate_rows,
                "projection_columns": len(result.projection),
                "fold0_opened": False,
                "fold1_opened": False,
                "human_labels_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
