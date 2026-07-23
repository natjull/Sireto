#!/usr/bin/env python3
"""Evaluate retrieval, cross-encoder and deployment gates from measured JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_cross_encoder import cross_encoder_gate
from src.xgb_matcher.v9_evaluation import (
    retrieval_promotion_gate,
    v9_deployment_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(args.measurements.read_text(encoding="utf-8"))
    report = {
        "retrieval": retrieval_promotion_gate(**values["retrieval"]),
        "cross_encoder": cross_encoder_gate(**values["cross_encoder"]),
        "deployment": v9_deployment_gate(**values["deployment"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["deployment"]["deploy"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
