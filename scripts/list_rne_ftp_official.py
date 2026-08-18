#!/usr/bin/env python3
"""List an explicitly authorized RNE plaintext FTP directory safely."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.official_source_sync import (  # noqa: E402
    Credentials,
    KeychainLocator,
    OfficialSyncError,
    PlainFtpTransport,
    read_keychain_secret,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=21)
    parser.add_argument("--path", default="/")
    parser.add_argument("--keychain-service", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument(
        "--allow-insecure-plaintext",
        action="store_true",
        help="Required acknowledgement that FTP credentials and payload are plaintext.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.allow_insecure_plaintext:
        print("FTP listing refused: explicit plaintext acknowledgement is required", file=sys.stderr)
        return 2

    secret = bytearray()
    try:
        locator = KeychainLocator(args.keychain_service, args.account)
        secret = read_keychain_secret(locator)
        credentials = Credentials.from_keychain_payload(secret, username=args.account)
        entries = PlainFtpTransport().list_directory(
            host=args.host,
            port=args.port,
            remote_path=args.path,
            credentials=credentials,
        )
        result = {
            "schema_version": "sireto-rne-ftp-listing-v1",
            "host": args.host,
            "path": args.path,
            "plaintext_explicitly_authorized": True,
            "credential_material_recorded": False,
            "entries": entries,
        }
        payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
    except OfficialSyncError as exc:
        print(f"FTP listing failed: {exc}", file=sys.stderr)
        return 2
    finally:
        for index in range(len(secret)):
            secret[index] = 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
