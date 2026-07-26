#!/usr/bin/env python3
"""Validate and atomically publish precomputed V4.1 shadow artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v41_shadow import write_shadow_run  # noqa: E402
from src.xgb_matcher.v9_dataset import read_table  # noqa: E402


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--candidates-top10", type=Path, required=True)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--model-bundle-id", required=True)
    parser.add_argument("--dataset-manifest-id", required=True)
    args = parser.parse_args()

    inputs = {
        "inventory": args.inventory,
        "decisions": args.decisions,
        "candidates_top10": args.candidates_top10,
        "evidence": args.evidence_jsonl,
        "panel": args.panel,
    }
    result = write_shadow_run(
        output_root=args.output_root,
        run_id=args.run_id,
        inventory=read_table(args.inventory),
        decisions=read_table(args.decisions),
        candidates_top10=read_table(args.candidates_top10),
        evidence=_read_jsonl(args.evidence_jsonl),
        panel=read_table(args.panel),
        input_artifacts=inputs,
        run_metadata={
            "model_bundle_id": args.model_bundle_id,
            "dataset_manifest_id": args.dataset_manifest_id,
        },
    )
    print(result)


if __name__ == "__main__":
    main()
