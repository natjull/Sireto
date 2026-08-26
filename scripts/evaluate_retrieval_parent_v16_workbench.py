#!/usr/bin/env python3
"""Evaluate a sealed v16 native-parent generation on the truth side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_parent_v16 import sha256_file
from src.xgb_matcher.retrieval_parent_v16_evaluation import evaluate_parent_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sealed v16 parent workbench")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--baseline-parents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    path = evaluate_parent_generation(
        generation_dir=args.generation_dir,
        cohort_dir=args.cohort_dir,
        baseline_parents_path=args.baseline_parents,
        output_dir=args.output_dir,
        config=config,
        config_sha256=sha256_file(config_path),
    )
    payload = json.loads(path.read_text())
    print(json.dumps({"manifest": str(path), "sha256": sha256_file(path), "decision": payload["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
