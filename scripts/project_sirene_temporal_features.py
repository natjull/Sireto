#!/usr/bin/env python3
"""Project SIRENE temporal features onto retrieval candidate pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.sirene_temporal import project_temporal_candidate_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--projection", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(project_temporal_candidate_features(args.candidates, args.projection, args.output))


if __name__ == "__main__":
    main()
