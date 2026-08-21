#!/usr/bin/env python3
"""Extract RNE person-morale roles and optionally build their SIREN overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.rne_legal_entity_roles import (  # noqa: E402
    build_legal_entity_role_index,
    iter_rne_zip_records,
    sha256_file,
    write_legal_entity_roles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--rne-formalities-zip", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--batch-size", type=int, default=8192)
    build = sub.add_parser("build-index")
    build.add_argument("--roles", type=Path, required=True)
    build.add_argument("--siren-documents", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "extract":
        counts = write_legal_entity_roles(
            iter_rne_zip_records(args.rne_formalities_zip),
            args.output,
            batch_size=args.batch_size,
        )
        manifest = {
            "schema_version": "sireto-rne-legal-entity-role-extract-manifest-v1",
            "source_sha256": sha256_file(args.rne_formalities_zip),
            "output_sha256": sha256_file(args.output),
            **counts,
        }
        path = args.output.with_suffix(args.output.suffix + ".manifest.json")
        path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
        print(args.output)
        return
    print(
        build_legal_entity_role_index(
            roles_path=args.roles,
            siren_documents_path=args.siren_documents,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()

