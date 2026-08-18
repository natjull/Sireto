#!/usr/bin/env python3
"""Build the shared Parquet/DuckDB official dossier store."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.siren_dossier import SirenDossierInputs, build_siren_dossier  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sirene-establishments", type=Path, required=True)
    parser.add_argument("--sirene-legal-units", type=Path, required=True)
    parser.add_argument("--official-evidence", type=Path, action="append", required=True)
    parser.add_argument("--official-relations", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="12GB")
    args = parser.parse_args()
    result = build_siren_dossier(
        SirenDossierInputs(
            args.sirene_establishments,
            args.sirene_legal_units,
            tuple(args.official_evidence),
            tuple(args.official_relations),
        ),
        output_root=args.output_root,
        temp_directory=args.temp_directory,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
