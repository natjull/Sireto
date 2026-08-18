#!/usr/bin/env python3
"""Build canonical official_evidence/official_relation Parquets in streaming mode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.official_evidence import OfficialSource  # noqa: E402
from src.xgb_matcher.official_evidence_builder import (  # noqa: E402
    SnapshotRole,
    SnapshotSpec,
    build_official_evidence_layer,
    snapshot_specs_from_sync_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sirene-establishments", type=Path)
    parser.add_argument("--sirene-establishment-history", type=Path)
    parser.add_argument("--sirene-legal-units", type=Path)
    parser.add_argument("--sirene-legal-unit-history", type=Path)
    parser.add_argument("--sirene-successions", type=Path)
    parser.add_argument("--rne-manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--rne-payload-name",
        action="append",
        default=[],
        help="Select named RNE payloads (e.g. formalities but not annual accounts).",
    )
    parser.add_argument("--bodacc-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def _sirene_specs(args: argparse.Namespace) -> list[SnapshotSpec]:
    values = [
        (
            args.sirene_establishments,
            OfficialSource.SIRENE_CURRENT,
            SnapshotRole.SIRENE_ESTABLISHMENTS,
        ),
        (
            args.sirene_establishment_history,
            OfficialSource.SIRENE_HISTORY,
            SnapshotRole.SIRENE_ESTABLISHMENT_HISTORY,
        ),
        (
            args.sirene_legal_units,
            OfficialSource.SIRENE_CURRENT,
            SnapshotRole.SIRENE_LEGAL_UNITS,
        ),
        (
            args.sirene_legal_unit_history,
            OfficialSource.SIRENE_HISTORY,
            SnapshotRole.SIRENE_LEGAL_UNIT_HISTORY,
        ),
        (
            args.sirene_successions,
            OfficialSource.SIRENE_SUCCESSION,
            SnapshotRole.SIRENE_SUCCESSIONS,
        ),
    ]
    return [
        SnapshotSpec(path, source, role, batch_size=args.batch_size)
        for path, source, role in values
        if path is not None
    ]


def main() -> None:
    args = parse_args()
    specs = _sirene_specs(args)
    for manifest in args.rne_manifest:
        specs.extend(
            snapshot_specs_from_sync_manifest(
                manifest,
                role=SnapshotRole.RNE_RECORDS,
                batch_size=args.batch_size,
                payload_names=(set(args.rne_payload_name) or None),
            )
        )
    for manifest in args.bodacc_manifest:
        specs.extend(
            snapshot_specs_from_sync_manifest(
                manifest,
                role=SnapshotRole.BODACC_ANNOUNCEMENTS,
                batch_size=args.batch_size,
            )
        )
    result = build_official_evidence_layer(
        specs,
        args.output_dir,
        work_dir=args.work_dir,
        batch_size=args.batch_size,
    )
    print(result.output_dir)


if __name__ == "__main__":
    main()
