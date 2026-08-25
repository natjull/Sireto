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
from src.xgb_matcher.retrieval_master_v15_evaluation import build_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the truth-side v15 Master feasibility cohort")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text())
    authorities = raw["authorities"]
    manifest = build_cohort(
        raw_union_path=Path(authorities["raw_union"]["path"]),
        label_ledger_path=Path(authorities["label_ledger"]["path"]),
        output_dir=args.output_dir,
        expected_raw_sha256=authorities["raw_union"]["sha256"],
        expected_label_sha256=authorities["label_ledger"]["sha256"],
    )
    print(json.dumps({"manifest": str(manifest), "sha256": sha256_file(manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
