#!/usr/bin/env python3
"""Build the exact-only V4 candidate dataset for ranker experiment E1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_downstream_selective_dataset import (  # noqa: E402
    DATASET_FEATURE_ORDER,
    OUTPUT_CANDIDATE_COLUMNS,
    CandidateWriter,
    build_split_candidates,
)
from scripts.evaluate_retrieval_admission import V7_WEIGHTS  # noqa: E402
from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.features import normalize_text  # noqa: E402
from src.xgb_matcher.partitioned_store import (  # noqa: E402
    PartitionedCandidateStore,
)
from src.xgb_matcher.v9_dataset import (  # noqa: E402
    LABEL_COLUMNS,
    QUERY_COLUMNS,
    SCHEMA_VERSION,
    V9DatasetManifest,
    file_sha256,
)
from src.xgb_matcher.v9_features import (  # noqa: E402
    SELECTIVE_RETRIEVAL_CHANNELS,
)


BUILD_SCHEMA_VERSION = "sireto-v4-ranker-dataset-1"
EXPECTED = {"train": 5749, "dev": 305}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_output(
    path: Path,
    manifest_path: Path,
    *,
    key: str | None = None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    expected = manifest.get("outputs", {}).get(key or path.name)
    if not expected or file_sha256(path) != expected:
        raise ValueError(f"Manifest hash mismatch: {path}")
    return manifest


def _canonical_queries(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "query_id": frame["query_id"].astype(str),
            "crm_name": frame["crm_name"].fillna(""),
            "crm_address": frame["crm_address"].fillna(""),
            "crm_postcode": frame["postcode"].fillna(""),
            "crm_city": frame["crm_city"].fillna(""),
            "crm_insee": frame["insee"].fillna(""),
            "reference_date": frame.get(
                "reference_date",
                pd.Series("", index=frame.index),
            ).fillna(""),
            "split": split,
        }
    )
    output["crm_name_norm"] = output["crm_name"].map(normalize_text)
    output["crm_address_norm"] = output["crm_address"].map(normalize_text)
    output["crm_city_norm"] = output["crm_city"].map(normalize_text)
    return output[QUERY_COLUMNS]


def _canonical_labels(
    frame: pd.DataFrame,
    *,
    split: str,
    snapshot_id: str,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "query_id": frame["query_id"].astype(str),
            "label_kind": "MATCH_EXACT",
            "ground_truth_siret": frame["ground_truth_siret"].astype(str),
            "ground_truth_siren": frame["ground_truth_siren"].astype(str),
            "label_source": "qualification_v4_current_snapshot",
            "validator": frame.get(
                "validator",
                pd.Series("", index=frame.index),
            ).fillna(""),
            "validated_at": "",
            "sirene_snapshot_id": snapshot_id,
            "split": split,
        }
    )
    return output[LABEL_COLUMNS]


def relabel_historical_candidates(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    truth_by_query = labels.set_index("query_id")["ground_truth_siret"]
    output = candidates[
        candidates["query_id"].astype(str).isin(truth_by_query.index)
    ].copy()
    output["query_id"] = output["query_id"].astype(str)
    output["candidate_siret"] = (
        output["candidate_siret"].astype(str).str.zfill(14)
    )
    output["split"] = "train"
    output["is_ground_truth"] = (
        output["candidate_siret"]
        == output["query_id"].map(truth_by_query).astype(str).str.zfill(14)
    ).astype(int)
    return output[OUTPUT_CANDIDATE_COLUMNS]


def validate_candidate_rows(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    candidates = candidates.copy()
    candidates["query_id"] = candidates["query_id"].astype(str)
    labels = labels.copy()
    labels["query_id"] = labels["query_id"].astype(str)
    counts = candidates.groupby("query_id").size()
    positives = candidates.groupby("query_id")["is_ground_truth"].sum()
    duplicate_pairs = int(
        candidates.duplicated(["query_id", "candidate_siret"]).sum()
    )
    checks = {
        "all_queries_have_candidates": set(counts.index)
        == set(labels["query_id"]),
        "max_candidates_at_most_100": int(counts.max()) <= 100,
        "zero_duplicate_pairs": duplicate_pairs == 0,
        "exactly_one_positive_per_query": bool(
            positives.reindex(labels["query_id"]).eq(1).all()
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "candidate_count": int(len(candidates)),
        "max_candidates": int(counts.max()),
        "duplicate_pairs": duplicate_pairs,
    }


def _write_frame(writer: CandidateWriter, frame: pd.DataFrame) -> None:
    for start in range(0, len(frame), 100_000):
        chunk = frame.iloc[start : start + 100_000]
        writer.writer.write_table(
            pa.Table.from_pandas(
                chunk,
                schema=writer.schema,
                preserve_index=False,
            )
        )
        writer.count += len(chunk)


def build_dataset(
    *,
    retrieval_gate_dir: Path,
    v4_dir: Path,
    fresh_dir: Path,
    downstream_dir: Path,
    fit_admission_raw: Path,
    dev_admission_raw: Path,
    fit_v7_channels: Path,
    dev_v7_channels: Path,
    fit_overlay_channels: Path,
    dev_overlay_channels: Path,
    v7_partitions: Path,
    overlay_partitions: Path,
    contract_path: Path,
    output_root: Path,
) -> Path:
    gate_manifest = _read_json(retrieval_gate_dir / "manifest.json")
    gate_summary = _read_json(retrieval_gate_dir / "summary.json")
    if gate_summary.get("verdict") != "GO_RANKER_V4":
        raise ValueError("V4 retrieval gate did not authorize ranker training")
    if gate_manifest.get("holdout_read") is not False:
        raise ValueError("Retrieval gate did not preserve holdout sealing")

    gate_fit = _verified_output(
        retrieval_gate_dir / "fit_exact.parquet",
        retrieval_gate_dir / "manifest.json",
    )
    del gate_fit
    fit_gate = pd.read_parquet(retrieval_gate_dir / "fit_exact.parquet")
    dev_gate = pd.read_parquet(retrieval_gate_dir / "dev_exact.parquet")
    fit_gate["query_id"] = fit_gate["query_id"].astype(str)
    dev_gate["query_id"] = dev_gate["query_id"].astype(str)
    historical_ids = set(
        fit_gate.loc[
            fit_gate["subset"].eq("historical_core")
            & fit_gate["hit_at_100"].astype(bool),
            "query_id",
        ]
    )
    fresh_fit_ids = set(
        fit_gate.loc[
            fit_gate["subset"].eq("fit_addition"),
            "query_id",
        ]
    )
    fresh_dev_ids = set(dev_gate["query_id"])

    v4_manifest = _read_json(v4_dir / "manifest.json")
    historical_benchmarks = []
    for old_split in ("train", "dev"):
        relative = f"{old_split}/benchmark.parquet"
        path = v4_dir / relative
        if file_sha256(path) != v4_manifest["outputs"][relative]:
            raise ValueError(f"V4 input hash mismatch: {path}")
        historical_benchmarks.append(pd.read_parquet(path))
    historical_benchmark = pd.concat(
        historical_benchmarks,
        ignore_index=True,
    )
    historical_benchmark["query_id"] = (
        historical_benchmark["query_id"].astype(str)
    )
    historical_benchmark = historical_benchmark[
        historical_benchmark["query_id"].isin(historical_ids)
    ].copy()

    fresh_manifest = _read_json(fresh_dir / "manifest.json")
    fresh_frames: dict[str, pd.DataFrame] = {}
    for role, ids in (
        ("fit_addition", fresh_fit_ids),
        ("dev_new", fresh_dev_ids),
    ):
        relative = f"{role}/benchmark.parquet"
        path = fresh_dir / relative
        if file_sha256(path) != fresh_manifest["outputs"][relative]:
            raise ValueError(f"Fresh V4 input hash mismatch: {path}")
        frame = pd.read_parquet(path)
        frame["query_id"] = frame["query_id"].astype(str)
        fresh_frames[role] = frame[frame["query_id"].isin(ids)].copy()

    snapshot_id = str(
        fresh_frames["fit_addition"]["sirene_snapshot_id"].iloc[0]
    )
    queries = pd.concat(
        [
            _canonical_queries(historical_benchmark, "train"),
            _canonical_queries(fresh_frames["fit_addition"], "train"),
            _canonical_queries(fresh_frames["dev_new"], "dev"),
        ],
        ignore_index=True,
    )
    labels = pd.concat(
        [
            _canonical_labels(
                historical_benchmark,
                split="train",
                snapshot_id=snapshot_id,
            ),
            _canonical_labels(
                fresh_frames["fit_addition"],
                split="train",
                snapshot_id=snapshot_id,
            ),
            _canonical_labels(
                fresh_frames["dev_new"],
                split="dev",
                snapshot_id=snapshot_id,
            ),
        ],
        ignore_index=True,
    )
    if queries["query_id"].duplicated().any():
        raise ValueError("Ranker dataset query IDs overlap")
    if labels.groupby("split").size().to_dict() != EXPECTED:
        raise ValueError("Ranker dataset split counts differ from contract")
    train_sirens = set(
        labels.loc[labels["split"].eq("train"), "ground_truth_siren"]
    )
    dev_sirens = set(
        labels.loc[labels["split"].eq("dev"), "ground_truth_siren"]
    )
    if train_sirens & dev_sirens:
        raise ValueError("Ranker train/dev SIRENs overlap")

    downstream_report = _read_json(downstream_dir / "build_report.json")
    historical_candidates_path = downstream_dir / "candidates.parquet"
    if (
        file_sha256(historical_candidates_path)
        != downstream_report["output_hashes"]["candidates.parquet"]
    ):
        raise ValueError("Historical candidate features hash mismatch")
    historical_candidates = pd.read_parquet(historical_candidates_path)
    historical_labels = labels[
        labels["query_id"].isin(historical_ids)
    ].copy()
    historical_candidates = relabel_historical_candidates(
        historical_candidates,
        historical_labels,
    )

    channel_paths = {
        "fit_v7_channels": fit_v7_channels,
        "dev_v7_channels": dev_v7_channels,
        "fit_overlay_channels": fit_overlay_channels,
        "dev_overlay_channels": dev_overlay_channels,
        "fit_admission_raw": fit_admission_raw,
        "dev_admission_raw": dev_admission_raw,
    }
    for path in channel_paths.values():
        _verified_output(path, path.parent / "manifest.json")
    sources = {
        "train": {
            "admission": pd.read_parquet(fit_admission_raw),
            "v7": pd.read_parquet(fit_v7_channels),
            "overlay": pd.read_parquet(fit_overlay_channels),
            "benchmark": fresh_frames["fit_addition"],
        },
        "dev": {
            "admission": pd.read_parquet(dev_admission_raw),
            "v7": pd.read_parquet(dev_v7_channels),
            "overlay": pd.read_parquet(dev_overlay_channels),
            "benchmark": fresh_frames["dev_new"],
        },
    }
    for values in sources.values():
        for frame in values.values():
            frame["query_id"] = frame["query_id"].astype(str)

    identity = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "retrieval_gate_manifest_sha256": file_sha256(
            retrieval_gate_dir / "manifest.json"
        ),
        "v4_manifest_sha256": file_sha256(v4_dir / "manifest.json"),
        "fresh_manifest_sha256": file_sha256(fresh_dir / "manifest.json"),
        "downstream_manifest_sha256": file_sha256(
            downstream_dir / "manifest.json"
        ),
        "channel_hashes": {
            name: file_sha256(path) for name, path in channel_paths.items()
        },
        "contract_sha256": file_sha256(contract_path),
        "feature_order": DATASET_FEATURE_ORDER,
        "git_commit": _git_commit(),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = output_root / build_id
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates_path = output_dir / "candidates.parquet"
    writer = CandidateWriter(candidates_path)
    diagnostics: dict[str, Any] = {}
    try:
        _write_frame(writer, historical_candidates)
        v7_store = PartitionedCandidateStore(v7_partitions)
        overlay_store = PartitionedCandidateStore(overlay_partitions)
        for split in ("train", "dev"):
            benchmark = sources[split]["benchmark"]
            split_labels = labels[
                labels["query_id"].isin(set(benchmark["query_id"]))
            ]
            diagnostics[split] = build_split_candidates(
                split=split,
                benchmark=benchmark,
                labels=split_labels,
                admission=sources[split]["admission"],
                v7_channels=sources[split]["v7"],
                overlay_channels=sources[split]["overlay"],
                v7_store=v7_store,
                overlay_store=overlay_store,
                writer=writer,
            )
    finally:
        writer.close()
    if any(
        value
        for split_diagnostics in diagnostics.values()
        for value in split_diagnostics.values()
    ):
        raise ValueError(f"Fresh feature build failed: {diagnostics}")

    queries_path = output_dir / "queries.parquet"
    labels_path = output_dir / "labels.parquet"
    queries.to_parquet(queries_path, index=False)
    labels.to_parquet(labels_path, index=False)
    built_candidates = pd.read_parquet(
        candidates_path,
        columns=[
            "query_id",
            "candidate_siret",
            "split",
            "is_ground_truth",
        ],
    )
    validation = validate_candidate_rows(built_candidates, labels)
    if not validation["pass"]:
        raise ValueError(f"Ranker candidate validation failed: {validation}")

    manifest = V9DatasetManifest(
        schema_version=SCHEMA_VERSION,
        build_id=build_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=42,
        sirene_snapshot_id=snapshot_id,
        input_hashes={
            "retrieval_gate_manifest": identity[
                "retrieval_gate_manifest_sha256"
            ],
            "v4_manifest": identity["v4_manifest_sha256"],
            "fresh_manifest": identity["fresh_manifest_sha256"],
            **identity["channel_hashes"],
        },
        retrieval_config={
            "name": "selective_admission_frozen_v4",
            "candidate_budget": 100,
            "channels": SELECTIVE_RETRIEVAL_CHANNELS,
            "weights": V7_WEIGHTS,
        },
        retrieval_signature=file_sha256(
            retrieval_gate_dir / "manifest.json"
        ),
        tokenizer_fingerprint=None,
        feature_order=DATASET_FEATURE_ORDER,
        row_counts={
            "queries": len(queries),
            "labels": len(labels),
            "candidates": writer.count,
        },
        legacy_artifacts_allowed=False,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "dataset_manifest_id": build_id,
        "git_commit": _git_commit(),
        "holdout_read": False,
        "old_test_read": False,
        "positive_injection": False,
        "excluded_fit_missing_positive_query_ids": ["6818", "8109"],
        "split_counts": labels.groupby("split").size().to_dict(),
        "diagnostics": diagnostics,
        "validation": validation,
        "outputs": {
            path.name: file_sha256(path)
            for path in (queries_path, labels_path, candidates_path)
        },
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-gate-dir", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--downstream-dir", type=Path, required=True)
    parser.add_argument("--fit-admission-raw", type=Path, required=True)
    parser.add_argument("--dev-admission-raw", type=Path, required=True)
    parser.add_argument("--fit-v7-channels", type=Path, required=True)
    parser.add_argument("--dev-v7-channels", type=Path, required=True)
    parser.add_argument("--fit-overlay-channels", type=Path, required=True)
    parser.add_argument("--dev-overlay-channels", type=Path, required=True)
    parser.add_argument("--v7-partitions", type=Path, required=True)
    parser.add_argument("--overlay-partitions", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        build_dataset(
            retrieval_gate_dir=args.retrieval_gate_dir,
            v4_dir=args.v4_dir,
            fresh_dir=args.fresh_dir,
            downstream_dir=args.downstream_dir,
            fit_admission_raw=args.fit_admission_raw,
            dev_admission_raw=args.dev_admission_raw,
            fit_v7_channels=args.fit_v7_channels,
            dev_v7_channels=args.dev_v7_channels,
            fit_overlay_channels=args.fit_overlay_channels,
            dev_overlay_channels=args.dev_overlay_channels,
            v7_partitions=args.v7_partitions,
            overlay_partitions=args.overlay_partitions,
            contract_path=args.contract,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
