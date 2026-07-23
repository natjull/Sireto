#!/usr/bin/env python3
"""Validate and freeze human adjudications for the V9 open-set benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_adjudication import freeze_adjudications
from src.xgb_matcher.v9_dataset import read_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=500)
    args = parser.parse_args()
    rows = read_table(args.source)
    if len(rows) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} adjudications, received {len(rows)}"
        )
    target = freeze_adjudications(
        rows,
        output_dir=args.output_root,
        source_path=args.source,
    )
    print(target)


if __name__ == "__main__":
    main()
