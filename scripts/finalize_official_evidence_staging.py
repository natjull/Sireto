#!/usr/bin/env python3
"""Finalize a completed official-evidence staging scan atomically."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.official_evidence_builder import (  # noqa: E402
    SnapshotRole,
    finalize_official_evidence_staging,
    snapshot_specs_from_sync_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--payload-name", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    specs = snapshot_specs_from_sync_manifest(
        args.source_manifest,
        role=SnapshotRole.RNE_RECORDS,
        batch_size=args.batch_size,
        payload_names=set(args.payload_name),
    )
    result = finalize_official_evidence_staging(
        args.staging_dir,
        args.output_dir,
        specs=specs,
        work_dir=args.work_dir,
        batch_size=args.batch_size,
    )
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
