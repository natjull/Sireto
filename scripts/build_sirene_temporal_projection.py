#!/usr/bin/env python3
"""Preflight or build the truthful SIRENE temporal retrieval sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.sirene_temporal import (  # noqa: E402
    SireneTemporalInputs,
    build_temporal_projection,
    build_all_temporal_projections,
    temporal_preflight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "build", "build-all"))
    parser.add_argument("--establishment-history", type=Path, required=True)
    parser.add_argument("--legal-unit-history", type=Path, required=True)
    parser.add_argument("--successions", type=Path, required=True)
    parser.add_argument("--establishments", type=Path, required=True)
    parser.add_argument("--legal-units", type=Path, required=True)
    parser.add_argument("--siren-prefix", default="00")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = SireneTemporalInputs(
        establishment_history=args.establishment_history,
        legal_unit_history=args.legal_unit_history,
        successions=args.successions,
        establishments=args.establishments,
        legal_units=args.legal_units,
    )
    if args.command == "preflight":
        report = temporal_preflight(inputs, siren_prefix=args.siren_prefix)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        if not report["gate_pass"]:
            raise SystemExit(2)
        return
    if args.output_root is None:
        raise SystemExit("--output-root is required for build")
    if args.command == "build-all":
        builds = build_all_temporal_projections(
            inputs, args.output_root,
            duckdb_memory_limit=args.memory_limit, duckdb_threads=args.threads,
        )
        print(args.output_root.resolve())
        print(f"prefixes={len(builds)}")
        return
    build = build_temporal_projection(
        inputs,
        args.output_root,
        siren_prefix=args.siren_prefix or None,
        duckdb_memory_limit=args.memory_limit,
        duckdb_threads=args.threads,
    )
    print(build.output_dir)


if __name__ == "__main__":
    main()
