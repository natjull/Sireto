#!/usr/bin/env python3
"""Build content-addressed retrieval v11 geographic site sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_sidecars_v11 import build_retrieval_sidecars  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="4GB")
    args = parser.parse_args()
    result = build_retrieval_sidecars(
        dossier_dir=args.dossier_dir,
        output_root=args.output_root,
        temp_directory=args.temp_directory,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
