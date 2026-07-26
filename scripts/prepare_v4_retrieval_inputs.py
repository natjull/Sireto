#!/usr/bin/env python3
"""Prepare the exact fresh V4 fit/dev input for the frozen retrieval tools."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4-retrieval-inputs-1"
EXPECTED_COUNTS = {"fit_addition": 819, "dev_new": 305}


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_manifest_output(
    *,
    root: Path,
    manifest: dict[str, Any],
    relative_path: str,
) -> Path:
    path = root / relative_path
    expected = manifest.get("outputs", {}).get(relative_path)
    if not expected or file_sha256(path) != expected:
        raise ValueError(f"Manifest hash mismatch: {path}")
    return path


def _exact_rows(path: Path, role: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["query_id"] = frame["query_id"].astype(str)
    exact = frame[frame["label_kind"].eq("MATCH_EXACT")].copy()
    exact["split"] = role
    exact["ground_truth_siret"] = (
        exact["ground_truth_siret"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(14)
    )
    exact["ground_truth_siren"] = exact["ground_truth_siret"].str[:9]
    if len(exact) != EXPECTED_COUNTS[role]:
        raise ValueError(
            f"{role}: expected {EXPECTED_COUNTS[role]} exact rows, "
            f"found {len(exact)}"
        )
    if exact["query_id"].duplicated().any():
        raise ValueError(f"{role}: duplicate query_id")
    return exact


def prepare_inputs(
    *,
    fresh_dir: Path,
    overlay_store_manifest_path: Path,
    contract_path: Path,
    output_root: Path,
) -> Path:
    """Build one immutable benchmark and two store-specific manifests."""
    fresh_manifest_path = fresh_dir / "manifest.json"
    fresh_manifest = _read_manifest(fresh_manifest_path)
    if fresh_manifest.get("status") != "V4_FRESH_GATE_PASS":
        raise ValueError("Fresh V4 input has not passed its gate")
    if fresh_manifest.get("holdout_model_evaluated") is not False:
        raise ValueError("Fresh V4 holdout is not sealed")

    role_paths = {
        role: _verify_manifest_output(
            root=fresh_dir,
            manifest=fresh_manifest,
            relative_path=f"{role}/benchmark.parquet",
        )
        for role in EXPECTED_COUNTS
    }
    overlay_manifest = _read_manifest(overlay_store_manifest_path)
    overlay_fingerprint = (
        overlay_manifest.get("data_inventory", {}).get("sha256")
    )
    if not overlay_fingerprint:
        raise ValueError("Overlay store manifest has no inventory fingerprint")

    inputs = pd.concat(
        [
            _exact_rows(role_paths[role], role)
            for role in EXPECTED_COUNTS
        ],
        ignore_index=True,
    )
    if inputs["query_id"].duplicated().any():
        raise ValueError("Fresh fit/dev query IDs overlap")
    fit_sirens = set(
        inputs.loc[
            inputs["split"].eq("fit_addition"),
            "ground_truth_siren",
        ]
    )
    dev_sirens = set(
        inputs.loc[
            inputs["split"].eq("dev_new"),
            "ground_truth_siren",
        ]
    )
    if fit_sirens & dev_sirens:
        raise ValueError("Fresh fit/dev exact SIRENs overlap")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "fresh_manifest_sha256": file_sha256(fresh_manifest_path),
        "role_benchmark_sha256": {
            role: file_sha256(path) for role, path in role_paths.items()
        },
        "contract_sha256": file_sha256(contract_path),
        "active_partitions_sha256": fresh_manifest["partitions_sha256"],
        "overlay_partitions_sha256": overlay_fingerprint,
        "expected_counts": EXPECTED_COUNTS,
        "retrieval_policy": "frozen_selective_internal_k5000_budget100",
        "git_commit": _git_commit(),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = output_root / build_id
    output_dir.mkdir(parents=True, exist_ok=False)

    benchmark_path = output_dir / "fresh_exact_benchmark.parquet"
    inputs.to_parquet(benchmark_path, index=False)
    benchmark_hash = file_sha256(benchmark_path)
    common = {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "benchmark_build_id": fresh_manifest["benchmark_build_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "holdout_read": False,
        "old_test_read": False,
        "splits": EXPECTED_COUNTS,
        "input_hashes": identity,
        "output_sha256": {
            benchmark_path.name: benchmark_hash,
        },
    }
    for name, partition_hash in (
        ("manifest_v7.json", fresh_manifest["partitions_sha256"]),
        ("manifest_overlay.json", overlay_fingerprint),
    ):
        manifest = {**common, "partitions_sha256": partition_hash}
        (output_dir / name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = {
        **common,
        "active_partitions_sha256": fresh_manifest["partitions_sha256"],
        "overlay_partitions_sha256": overlay_fingerprint,
        "outputs": {
            benchmark_path.name: benchmark_hash,
            "manifest_v7.json": file_sha256(output_dir / "manifest_v7.json"),
            "manifest_overlay.json": file_sha256(
                output_dir / "manifest_overlay.json"
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument(
        "--overlay-store-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        prepare_inputs(
            fresh_dir=args.fresh_dir,
            overlay_store_manifest_path=args.overlay_store_manifest,
            contract_path=args.contract,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
