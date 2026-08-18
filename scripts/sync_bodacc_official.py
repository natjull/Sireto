#!/usr/bin/env python3
"""Synchronize official BODACC backfills and Opendatasoft v2.1 increments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.official_source_sync import (  # noqa: E402
    BodaccSyncConfig,
    OfficialSyncError,
    load_json_config,
    sync_bodacc,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = BodaccSyncConfig.from_dict(load_json_config(args.config))
        output = sync_bodacc(config=config, output_root=args.output_root)
    except OfficialSyncError as exc:
        print(f"BODACC sync refused: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
