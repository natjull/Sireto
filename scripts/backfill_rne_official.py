#!/usr/bin/env python3
"""Run or resume the complete partitioned INPI RNE HTTPS backfill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.rne_backfill import (  # noqa: E402
    RneBackfillConfig,
    run_rne_backfill,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    config = RneBackfillConfig.from_dict(raw)
    output = run_rne_backfill(
        config,
        output_root=args.output_root,
        receipt_path=args.receipt,
        progress=lambda message: print(message, flush=True),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
