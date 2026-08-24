#!/usr/bin/env python3
"""Build the immutable W1 projection or compare one fresh W3 against it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PHYSICAL_CONFIG = (
    ROOT / "config" / "retrieval_all_human_v13_physical_execution.json"
)
DEFAULT_PHYSICAL_CONFIG_SHA256 = (
    "be3445c9ef377e9090043856f2c886e3005eac38b9b665542fc6cfeac4429f46"
)

from src.xgb_matcher.retrieval_fold_fused_logical_io_v13 import (  # noqa: E402
    build_immutable_w1_io_projection_v13,
    compare_w3_to_immutable_w1_io_v13,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Cache-neutral logical-I/O bridge: project pinned v12 W1 without "
            "executing it, then compare one fresh v13 W3-only run"
        )
    )
    commands = value.add_subparsers(dest="command", required=True)
    project = commands.add_parser("project-w1")
    project.add_argument(
        "--physical-config",
        type=Path,
        default=DEFAULT_PHYSICAL_CONFIG,
        help="pinned v13 physical config",
    )
    project.add_argument(
        "--physical-config-sha256",
        default=DEFAULT_PHYSICAL_CONFIG_SHA256,
        help="expected SHA-256 of the pinned v13 physical config",
    )
    project.add_argument("--output-root", type=Path, required=True)

    compare = commands.add_parser("compare-w3")
    compare.add_argument("--w1-projection-manifest", type=Path, required=True)
    compare.add_argument("--w1-projection-manifest-sha256", required=True)
    compare.add_argument("--w3-run-manifest", type=Path, required=True)
    compare.add_argument("--w3-run-manifest-sha256", required=True)
    compare.add_argument("--output-root", type=Path, required=True)
    return value


def run(args: argparse.Namespace) -> Path:
    if args.command == "project-w1":
        artifact = build_immutable_w1_io_projection_v13(
            physical_config_path=args.physical_config,
            physical_config_sha256=args.physical_config_sha256,
            output_root=args.output_root,
        )
        payload = {
            "phase": "IMMUTABLE_W1_PROJECTION",
            "manifest_path": str(artifact.manifest_path),
            "manifest_sha256": artifact.manifest_sha256,
            "build_id": artifact.build_id,
            "projection_sha256": artifact.projection_sha256,
            "unique_extent_count": artifact.unique_extent_count,
            "logical_compressed_bytes": artifact.logical_compressed_bytes,
            "materialized_output_bytes": artifact.materialized_output_bytes,
        }
    else:
        artifact = compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=args.w1_projection_manifest,
            w1_projection_manifest_sha256=args.w1_projection_manifest_sha256,
            w3_run_manifest_path=args.w3_run_manifest,
            w3_run_manifest_sha256=args.w3_run_manifest_sha256,
            output_root=args.output_root,
        )
        payload = {
            "phase": "FRESH_W3_COMPARISON",
            "manifest_path": str(artifact.manifest_path),
            "manifest_sha256": artifact.manifest_sha256,
            "build_id": artifact.build_id,
            "status": artifact.status,
            "gate_pass": artifact.gate_pass,
            "logical_io_ratio": artifact.logical_io_ratio,
            "materialized_io_ratio": artifact.materialized_io_ratio,
        }
    print(json.dumps(payload, sort_keys=True))
    return artifact.manifest_path


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
