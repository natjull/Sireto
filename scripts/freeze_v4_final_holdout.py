#!/usr/bin/env python3
"""Freeze the complete V4 bundle before the one-shot holdout opening."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_downstream_selective_dataset import (  # noqa: E402
    DATASET_FEATURE_ORDER,
)
from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.v9_dataset import (  # noqa: E402
    V9DatasetManifest,
    file_sha256,
)
from src.xgb_matcher.v9_scene import V9_SCENE_FEATURE_NAMES  # noqa: E402


SCHEMA_VERSION = "sireto-v4-final-freeze-1"
EXPECTED_COUNTS = {
    "MATCH_EXACT": 302,
    "AMBIGUOUS": 52,
    "UNRESOLVED": 991,
}
EXPECTED_THRESHOLD = 1.0
CODE_FILES = (
    "scripts/freeze_v4_final_holdout.py",
    "scripts/prepare_v4_final_holdout.py",
    "scripts/evaluate_v4_final_holdout.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def build_freeze(
    *,
    project_root: Path,
    fresh_dir: Path,
    retrieval_dir: Path,
    ranker_dataset_dir: Path,
    ranker_dir: Path,
    acceptor_dataset_dir: Path,
    acceptor_dir: Path,
    overlay_manifest_path: Path,
    contract_path: Path,
    output_root: Path,
) -> Path:
    project_root = project_root.resolve()
    current_commit = _git_commit()
    code_paths = [project_root / relative for relative in CODE_FILES]
    for path in [contract_path, overlay_manifest_path, *code_paths]:
        _assert_file(path)

    fresh_manifest_path = fresh_dir / "manifest.json"
    retrieval_manifest_path = retrieval_dir / "manifest.json"
    ranker_dataset_manifest_path = ranker_dataset_dir / "manifest.json"
    ranker_metadata_path = ranker_dir / "metadata.json"
    acceptor_dataset_manifest_path = acceptor_dataset_dir / "manifest.json"
    acceptor_metadata_path = acceptor_dir / "metadata.json"
    frozen_files = {
        "fresh_manifest": fresh_manifest_path,
        "retrieval_manifest": retrieval_manifest_path,
        "ranker_dataset_manifest": ranker_dataset_manifest_path,
        "ranker_model": ranker_dir / "ranker.json",
        "ranker_metadata": ranker_metadata_path,
        "acceptor_dataset_manifest": acceptor_dataset_manifest_path,
        "acceptor_model": acceptor_dir / "acceptor_model.joblib",
        "acceptor_calibrator": acceptor_dir / "acceptor_calibrator.joblib",
        "acceptor_metadata": acceptor_metadata_path,
        "overlay_store_manifest": overlay_manifest_path,
        "contract": contract_path,
        **{
            f"code:{relative}": project_root / relative
            for relative in CODE_FILES
        },
    }
    for path in frozen_files.values():
        _assert_file(path)

    fresh = _read_json(fresh_manifest_path)
    retrieval = _read_json(retrieval_manifest_path)
    ranker_dataset = V9DatasetManifest.load(ranker_dataset_manifest_path)
    ranker_metadata = _read_json(ranker_metadata_path)
    acceptor_dataset = V9DatasetManifest.load(
        acceptor_dataset_manifest_path
    )
    acceptor_metadata = _read_json(acceptor_metadata_path)
    overlay = _read_json(overlay_manifest_path)

    observed_counts = fresh["summary"]["roles"]["holdout_sealed"][
        "label_counts"
    ]
    checks = {
        "fresh_gate_pass": fresh.get("status") == "V4_FRESH_GATE_PASS",
        "holdout_not_model_evaluated": (
            fresh.get("holdout_model_evaluated") is False
        ),
        "holdout_counts_match": observed_counts == EXPECTED_COUNTS,
        "zero_exact_siren_overlap": all(
            count == 0
            for count in fresh["summary"]["exact_siren_overlap"][
                "overlap_counts"
            ].values()
        ),
        "retrieval_gate_pass": (
            retrieval.get("verdict") == "GO_RANKER_V4"
            or retrieval.get("status") == "GO_RANKER_V4"
        ),
        "ranker_dataset_compatible": (
            ranker_metadata.get("dataset_manifest_id")
            == ranker_dataset.build_id
        ),
        "acceptor_dataset_compatible": (
            acceptor_metadata.get("dataset_manifest_id")
            == acceptor_dataset.build_id
        ),
        "retrieval_signature_compatible": (
            ranker_dataset.retrieval_signature
            == acceptor_dataset.retrieval_signature
        ),
        "candidate_features_frozen": (
            ranker_dataset.feature_order == DATASET_FEATURE_ORDER
            and acceptor_dataset.feature_order == DATASET_FEATURE_ORDER
            and ranker_metadata.get("feature_order")
            == DATASET_FEATURE_ORDER
        ),
        "scene_features_frozen": (
            acceptor_metadata.get("feature_order")
            == list(V9_SCENE_FEATURE_NAMES)
        ),
        "acceptor_threshold_frozen": (
            float(acceptor_metadata.get("threshold", -1))
            == EXPECTED_THRESHOLD
        ),
        "acceptor_gate_pass": (
            acceptor_metadata.get("training_report", {}).get("verdict")
            == "GO_HOLDOUT_V4"
        ),
        "active_partitions_match": (
            fresh.get("partitions_sha256")
            == fresh.get("partition_fingerprint", {}).get("sha256")
        ),
        "overlay_partitions_declared": bool(
            overlay.get("data_inventory", {}).get("sha256")
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Cannot freeze final bundle: {failed}")

    holdout_declared_hashes = {
        name: fresh["outputs"][f"holdout_sealed/{name}"]
        for name in ("benchmark.parquet", "labels.parquet")
    }
    file_hashes = {
        name: file_sha256(path) for name, path in frozen_files.items()
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": current_commit,
        "frozen_file_hashes": file_hashes,
        "holdout_declared_hashes": holdout_declared_hashes,
        "ranker_dataset_manifest_id": ranker_dataset.build_id,
        "acceptor_dataset_manifest_id": acceptor_dataset.build_id,
        "ranker_model_bundle": ranker_dir.name,
        "acceptor_model_bundle_id": acceptor_metadata["model_bundle_id"],
        "threshold": EXPECTED_THRESHOLD,
    }
    authorization_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = output_root / authorization_id
    output_dir.mkdir(parents=True, exist_ok=False)
    authorization_path = output_dir / "authorization.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "v4_final_holdout_once",
        "status": "FROZEN_AUTHORIZED",
        "authorization_id": authorization_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit,
        "checks": checks,
        "expected_counts": EXPECTED_COUNTS,
        "holdout_declared_hashes": holdout_declared_hashes,
        "frozen_paths": {
            name: str(path.resolve())
            for name, path in frozen_files.items()
        },
        "frozen_file_hashes": file_hashes,
        "paths": {
            "fresh_dir": str(fresh_dir.resolve()),
            "retrieval_dir": str(retrieval_dir.resolve()),
            "ranker_dataset_dir": str(ranker_dataset_dir.resolve()),
            "ranker_dir": str(ranker_dir.resolve()),
            "acceptor_dataset_dir": str(acceptor_dataset_dir.resolve()),
            "acceptor_dir": str(acceptor_dir.resolve()),
            "active_partitions": str(
                (project_root / "data/candidates_v7_all").resolve()
            ),
            "overlay_partitions": str(
                overlay_manifest_path.parent.resolve()
            ),
        },
        "active_partitions_sha256": fresh["partitions_sha256"],
        "overlay_partitions_sha256": overlay["data_inventory"]["sha256"],
        "candidate_feature_order": DATASET_FEATURE_ORDER,
        "scene_feature_order": list(V9_SCENE_FEATURE_NAMES),
        "ranker_dataset_manifest_id": ranker_dataset.build_id,
        "acceptor_dataset_manifest_id": acceptor_dataset.build_id,
        "acceptor_model_bundle_id": acceptor_metadata["model_bundle_id"],
        "acceptor_threshold": EXPECTED_THRESHOLD,
    }
    authorization_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return authorization_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--ranker-dataset-dir", type=Path, required=True)
    parser.add_argument("--ranker-dir", type=Path, required=True)
    parser.add_argument("--acceptor-dataset-dir", type=Path, required=True)
    parser.add_argument("--acceptor-dir", type=Path, required=True)
    parser.add_argument("--overlay-store-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        build_freeze(
            project_root=args.project_root,
            fresh_dir=args.fresh_dir,
            retrieval_dir=args.retrieval_dir,
            ranker_dataset_dir=args.ranker_dataset_dir,
            ranker_dir=args.ranker_dir,
            acceptor_dataset_dir=args.acceptor_dataset_dir,
            acceptor_dir=args.acceptor_dir,
            overlay_manifest_path=args.overlay_store_manifest,
            contract_path=args.contract,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
