#!/usr/bin/env python3
"""Project shared official dossier features onto query/candidate pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.siren_dossier import project_dossier_candidate_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = project_dossier_candidate_features(
        dossier_dir=args.dossier,
        candidates_path=args.candidates,
        output_path=args.output,
    )
    print(f"{args.output}\t{rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
