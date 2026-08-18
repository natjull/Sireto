#!/usr/bin/env python3
"""Build privacy-minimal RNE annual-account deposit metadata Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.rne_accounts import build_rne_account_deposits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--payload-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    result = build_rne_account_deposits(
        manifest_path=args.manifest,
        payload_name=args.payload_name,
        output_root=args.output_root,
        batch_size=args.batch_size,
    )
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
