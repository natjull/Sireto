#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_master_v15 import sha256_file
from src.xgb_matcher.retrieval_master_v15_evaluation import evaluate_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the sealed v15 Master feasibility generation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = evaluate_generation(
        generation_dir=args.generation_dir, cohort_dir=args.cohort_dir,
        output_dir=args.output_dir, config_sha256=sha256_file(args.config),
    )
    payload = json.loads(manifest.read_text())
    print(json.dumps({"manifest": str(manifest), "sha256": sha256_file(manifest), "decision": payload["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
