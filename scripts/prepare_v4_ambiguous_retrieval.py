#!/usr/bin/env python3
"""Prepare fresh ambiguous V4 rows for label-independent frozen retrieval."""

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


SCHEMA_VERSION = "sireto-v4-ambiguous-retrieval-input-1"
EXPECTED = {"fit_ambiguous": 142, "dev_ambiguous": 53}
SOURCE_ROLES = {
    "fit_ambiguous": "fit_addition",
    "dev_ambiguous": "dev_new",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_role(frame: pd.DataFrame, output_role: str) -> pd.DataFrame:
    ambiguous = frame[frame["label_kind"].eq("AMBIGUOUS")].copy()
    if len(ambiguous) != EXPECTED[output_role]:
        raise ValueError(
            f"{output_role}: expected {EXPECTED[output_role]}, "
            f"found {len(ambiguous)}"
        )
    probes = ambiguous["direct_active_sirets_json"].map(json.loads)
    if not probes.map(len).ge(2).all():
        raise ValueError("AMBIGUOUS rows require at least two direct SIRETs")
    ambiguous["diagnostic_probe_siret"] = probes.map(lambda values: values[0])
    ambiguous["ground_truth_siret"] = ambiguous["diagnostic_probe_siret"]
    ambiguous["ground_truth_siren"] = (
        ambiguous["diagnostic_probe_siret"].str[:9]
    )
    ambiguous["split"] = output_role
    ambiguous["retrieval_uses_diagnostic_probe"] = False
    return ambiguous


def prepare(
    *,
    fresh_dir: Path,
    overlay_manifest_path: Path,
    contract_path: Path,
    output_root: Path,
) -> Path:
    fresh_manifest = _read_json(fresh_dir / "manifest.json")
    if fresh_manifest.get("holdout_model_evaluated") is not False:
        raise ValueError("Fresh holdout is not sealed")
    frames = []
    input_hashes = {}
    for output_role, source_role in SOURCE_ROLES.items():
        relative = f"{source_role}/benchmark.parquet"
        path = fresh_dir / relative
        expected = fresh_manifest["outputs"][relative]
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"Fresh manifest hash mismatch: {path}")
        input_hashes[relative] = observed
        frames.append(prepare_role(pd.read_parquet(path), output_role))
    benchmark = pd.concat(frames, ignore_index=True)
    if benchmark["query_id"].astype(str).duplicated().any():
        raise ValueError("Ambiguous fit/dev query IDs overlap")

    overlay_manifest = _read_json(overlay_manifest_path)
    overlay_hash = overlay_manifest["data_inventory"]["sha256"]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "fresh_manifest_sha256": file_sha256(fresh_dir / "manifest.json"),
        "input_hashes": input_hashes,
        "contract_sha256": file_sha256(contract_path),
        "active_partitions_sha256": fresh_manifest["partitions_sha256"],
        "overlay_partitions_sha256": overlay_hash,
        "git_commit": _git_commit(),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = output_root / build_id
    output_dir.mkdir(parents=True, exist_ok=False)
    benchmark_path = output_dir / "fresh_ambiguous_benchmark.parquet"
    benchmark.to_parquet(benchmark_path, index=False)
    benchmark_hash = file_sha256(benchmark_path)
    common = {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "benchmark_build_id": fresh_manifest["benchmark_build_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "splits": EXPECTED,
        "holdout_read": False,
        "old_test_read": False,
        "diagnostic_probe_affects_retrieval": False,
        "output_sha256": {benchmark_path.name: benchmark_hash},
    }
    for name, partitions_hash in (
        ("manifest_v7.json", fresh_manifest["partitions_sha256"]),
        ("manifest_overlay.json", overlay_hash),
    ):
        (output_dir / name).write_text(
            json.dumps(
                {**common, "partitions_sha256": partitions_hash},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    manifest = {
        **common,
        "identity": identity,
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
    parser.add_argument("--overlay-store-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        prepare(
            fresh_dir=args.fresh_dir,
            overlay_manifest_path=args.overlay_store_manifest,
            contract_path=args.contract,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
