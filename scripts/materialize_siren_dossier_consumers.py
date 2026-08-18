#!/usr/bin/env python3
"""Materialize retrieval or neural-text views from a SIREN dossier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.siren_dossier import (  # noqa: E402
    materialize_dossier_retrieval_documents,
    project_dossier_fusion_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    retrieval = sub.add_parser("retrieval")
    retrieval.add_argument("--dossier", type=Path, required=True)
    retrieval.add_argument("--output-dir", type=Path, required=True)
    retrieval.add_argument("--name-portfolio-policy", type=Path)
    fusion = sub.add_parser("fusion-text")
    fusion.add_argument("--dossier", type=Path, required=True)
    fusion.add_argument("--candidates", type=Path, required=True)
    fusion.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "retrieval":
        print(materialize_dossier_retrieval_documents(
            dossier_dir=args.dossier,
            output_dir=args.output_dir,
            name_portfolio_policy=args.name_portfolio_policy,
        ))
    else:
        print(project_dossier_fusion_text(dossier_dir=args.dossier, candidates_path=args.candidates, output_path=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
