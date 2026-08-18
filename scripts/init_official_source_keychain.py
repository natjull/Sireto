#!/usr/bin/env python3
"""Initialize an official-source password via the macOS secure prompt."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.official_source_sync import (  # noqa: E402
    KeychainLocator,
    OfficialSyncError,
    initialize_keychain_secret,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True, help="Keychain service locator (not secret)")
    parser.add_argument("--account", required=True, help="Transfer username / Keychain account")
    args = parser.parse_args()
    try:
        initialize_keychain_secret(KeychainLocator(args.service, args.account))
    except OfficialSyncError as exc:
        print(f"Keychain initialization refused: {exc}", file=sys.stderr)
        return 2
    print("Keychain password stored; no secret was passed in argv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
