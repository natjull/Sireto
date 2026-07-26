#!/usr/bin/env python3
"""Build exact plus ambiguous V4 scenes and ranker predictions for E2."""

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
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_downstream_selective_dataset import (  # noqa: E402
    DATASET_FEATURE_ORDER,
    OUTPUT_CANDIDATE_COLUMNS,
    CandidateWriter,
    build_split_candidates,
)
from scripts.build_v4_ranker_dataset import (  # noqa: E402
    _canonical_queries,
)
from scripts.evaluate_retrieval_admission import V7_WEIGHTS  # noqa: E402
from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from scripts.train_v9_ranker import score_rows  # noqa: E402
from src.xgb_matcher.partitioned_store import (  # noqa: E402
    PartitionedCandidateStore,
)
from src.xgb_matcher.v9_dataset import (  # noqa: E402
    LABEL_COLUMNS,
    SCHEMA_VERSION,
    V9DatasetManifest,
    file_sha256,
)
from src.xgb_matcher.v9_features import (  # noqa: E402
    SELECTIVE_RETRIEVAL_CHANNELS,
)


BUILD_SCHEMA_VERSION = "sireto-v4-acceptor-dataset-1"
EXPECTED = {
    "train_exact": 5749,
    "dev_exact": 305,
    "train_ambiguous_historical": 966,
    "train_ambiguous_fresh": 142,
    "dev_ambiguous_fresh": 53,
    "train_all": 6857,
    "dev_all": 358,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ambiguous_labels(
    frame: pd.DataFrame,
    *,
    split: str,
    snapshot_id: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": frame["query_id"].astype(str),
            "label_kind": "AMBIGUOUS",
            "ground_truth_siret": None,
            "ground_truth_siren": None,
            "label_source": "qualification_v4_current_snapshot",
            "validator": frame.get(
                "validator",
                pd.Series("", index=frame.index),
            ).fillna(""),
            "validated_at": "",
            "sirene_snapshot_id": snapshot_id,
            "split": split,
        }
    )[LABEL_COLUMNS]


def relabel_ambiguous_candidates(
    candidates: pd.DataFrame,
    query_ids: set[str],
) -> pd.DataFrame:
    output = candidates[
        candidates["query_id"].astype(str).isin(query_ids)
    ].copy()
    output["query_id"] = output["query_id"].astype(str)
    output["split"] = "train"
    output["is_ground_truth"] = 0
    return output[OUTPUT_CANDIDATE_COLUMNS]


def validate_acceptor_candidates(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    candidates = candidates.copy()
    labels = labels.copy()
    candidates["query_id"] = candidates["query_id"].astype(str)
    labels["query_id"] = labels["query_id"].astype(str)
    counts = candidates.groupby("query_id").size()
    positives = candidates.groupby("query_id")["is_ground_truth"].sum()
    exact_ids = set(
        labels.loc[labels["label_kind"].eq("MATCH_EXACT"), "query_id"]
    )
    ambiguous_ids = set(
        labels.loc[labels["label_kind"].eq("AMBIGUOUS"), "query_id"]
    )
    checks = {
        "all_queries_have_candidates": set(counts.index)
        == set(labels["query_id"]),
        "max_candidates_at_most_100": int(counts.max()) <= 100,
        "zero_duplicate_pairs": not candidates.duplicated(
            ["query_id", "candidate_siret"]
        ).any(),
        "exact_has_one_positive": positives.reindex(exact_ids).eq(1).all(),
        "ambiguous_has_zero_positive": positives.reindex(
            ambiguous_ids
        ).eq(0).all(),
    }
    return {
        "checks": {name: bool(value) for name, value in checks.items()},
        "pass": all(checks.values()),
        "candidate_count": int(len(candidates)),
        "max_candidates": int(counts.max()),
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


def build(
    *,
    exact_dataset_dir: Path,
    ranker_dir: Path,
    v4_dir: Path,
    downstream_dir: Path,
    ambiguous_input_dir: Path,
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
    exact_manifest = V9DatasetManifest.load(
        exact_dataset_dir / "manifest.json"
    )
    exact_report = _read_json(exact_dataset_dir / "build_report.json")
    if exact_report.get("holdout_read") is not False:
        raise ValueError("Exact dataset did not preserve holdout sealing")
    exact_queries = pd.read_parquet(exact_dataset_dir / "queries.parquet")
    exact_labels = pd.read_parquet(exact_dataset_dir / "labels.parquet")
    exact_candidates = pd.read_parquet(
        exact_dataset_dir / "candidates.parquet"
    )
    exact_counts = exact_labels.groupby("split").size().to_dict()
    if exact_counts != {
        "train": EXPECTED["train_exact"],
        "dev": EXPECTED["dev_exact"],
    }:
        raise ValueError("Exact dataset counts differ from E2 contract")

    v4_manifest = _read_json(v4_dir / "manifest.json")
    historical_parts = []
    for old_split in ("train", "dev"):
        relative = f"{old_split}/benchmark.parquet"
        path = v4_dir / relative
        if file_sha256(path) != v4_manifest["outputs"][relative]:
            raise ValueError(f"V4 input hash mismatch: {path}")
        historical_parts.append(pd.read_parquet(path))
    historical = pd.concat(historical_parts, ignore_index=True)
    historical["query_id"] = historical["query_id"].astype(str)
    historical_ambiguous = historical[
        historical["label_kind"].eq("AMBIGUOUS")
    ].copy()
    if len(historical_ambiguous) != EXPECTED[
        "train_ambiguous_historical"
    ]:
        raise ValueError("Historical ambiguous count differs from contract")

    ambiguous_manifest = _read_json(ambiguous_input_dir / "manifest.json")
    ambiguous_benchmark_path = (
        ambiguous_input_dir / "fresh_ambiguous_benchmark.parquet"
    )
    if (
        file_sha256(ambiguous_benchmark_path)
        != ambiguous_manifest["outputs"][ambiguous_benchmark_path.name]
    ):
        raise ValueError("Ambiguous input hash mismatch")
    fresh_ambiguous = pd.read_parquet(ambiguous_benchmark_path)
    fresh_ambiguous["query_id"] = fresh_ambiguous["query_id"].astype(str)
    fresh_fit = fresh_ambiguous[
        fresh_ambiguous["split"].eq("fit_ambiguous")
    ].copy()
    fresh_dev = fresh_ambiguous[
        fresh_ambiguous["split"].eq("dev_ambiguous")
    ].copy()

    snapshot_id = exact_manifest.sirene_snapshot_id
    queries = pd.concat(
        [
            exact_queries,
            _canonical_queries(historical_ambiguous, "train"),
            _canonical_queries(fresh_fit, "train"),
            _canonical_queries(fresh_dev, "dev"),
        ],
        ignore_index=True,
    )
    labels = pd.concat(
        [
            exact_labels,
            _ambiguous_labels(
                historical_ambiguous,
                split="train",
                snapshot_id=snapshot_id,
            ),
            _ambiguous_labels(
                fresh_fit,
                split="train",
                snapshot_id=snapshot_id,
            ),
            _ambiguous_labels(
                fresh_dev,
                split="dev",
                snapshot_id=snapshot_id,
            ),
        ],
        ignore_index=True,
    )
    if queries["query_id"].duplicated().any():
        raise ValueError("Acceptor query IDs overlap")
    if labels.groupby("split").size().to_dict() != {
        "train": EXPECTED["train_all"],
        "dev": EXPECTED["dev_all"],
    }:
        raise ValueError("Acceptor split counts differ from contract")

    downstream_report = _read_json(downstream_dir / "build_report.json")
    historical_candidate_path = downstream_dir / "candidates.parquet"
    if (
        file_sha256(historical_candidate_path)
        != downstream_report["output_hashes"]["candidates.parquet"]
    ):
        raise ValueError("Historical candidate hash mismatch")
    historical_candidates = relabel_ambiguous_candidates(
        pd.read_parquet(historical_candidate_path),
        set(historical_ambiguous["query_id"]),
    )

    paths = {
        "fit_admission": fit_admission_raw,
        "dev_admission": dev_admission_raw,
        "fit_v7": fit_v7_channels,
        "dev_v7": dev_v7_channels,
        "fit_overlay": fit_overlay_channels,
        "dev_overlay": dev_overlay_channels,
    }
    for path in paths.values():
        manifest = _read_json(path.parent / "manifest.json")
        if file_sha256(path) != manifest["outputs"][path.name]:
            raise ValueError(f"Ambiguous retrieval hash mismatch: {path}")
    sources = {
        "train": {
            "benchmark": fresh_fit,
            "labels": labels[
                labels["query_id"].isin(set(fresh_fit["query_id"]))
            ],
            "admission": pd.read_parquet(fit_admission_raw),
            "v7": pd.read_parquet(fit_v7_channels),
            "overlay": pd.read_parquet(fit_overlay_channels),
        },
        "dev": {
            "benchmark": fresh_dev,
            "labels": labels[
                labels["query_id"].isin(set(fresh_dev["query_id"]))
            ],
            "admission": pd.read_parquet(dev_admission_raw),
            "v7": pd.read_parquet(dev_v7_channels),
            "overlay": pd.read_parquet(dev_overlay_channels),
        },
    }
    for values in sources.values():
        for frame in values.values():
            frame["query_id"] = frame["query_id"].astype(str)

    identity = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "exact_manifest_sha256": file_sha256(
            exact_dataset_dir / "manifest.json"
        ),
        "ranker_sha256": file_sha256(ranker_dir / "ranker.json"),
        "v4_manifest_sha256": file_sha256(v4_dir / "manifest.json"),
        "ambiguous_manifest_sha256": file_sha256(
            ambiguous_input_dir / "manifest.json"
        ),
        "retrieval_hashes": {
            name: file_sha256(path) for name, path in paths.items()
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
        _write_frame(writer, exact_candidates[OUTPUT_CANDIDATE_COLUMNS])
        _write_frame(writer, historical_candidates)
        v7_store = PartitionedCandidateStore(v7_partitions)
        overlay_store = PartitionedCandidateStore(overlay_partitions)
        for split in ("train", "dev"):
            values = sources[split]
            diagnostics[split] = build_split_candidates(
                split=split,
                benchmark=values["benchmark"],
                labels=values["labels"],
                admission=values["admission"],
                v7_channels=values["v7"],
                overlay_channels=values["overlay"],
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
        raise ValueError(f"Ambiguous feature build failed: {diagnostics}")

    queries_path = output_dir / "queries.parquet"
    labels_path = output_dir / "labels.parquet"
    queries.to_parquet(queries_path, index=False)
    labels.to_parquet(labels_path, index=False)
    all_candidates = pd.read_parquet(candidates_path)
    validation = validate_acceptor_candidates(all_candidates, labels)
    if not validation["pass"]:
        raise ValueError(f"Acceptor candidate validation failed: {validation}")

    exact_predictions = pd.read_parquet(
        ranker_dir / "ranker_predictions.parquet"
    )
    ambiguous_ids = set(
        labels.loc[labels["label_kind"].eq("AMBIGUOUS"), "query_id"]
    )
    if ambiguous_ids & set(exact_labels["query_id"].astype(str)):
        raise ValueError("Ranker training dataset contains ambiguous queries")
    ambiguous_candidates = all_candidates[
        all_candidates["query_id"].astype(str).isin(ambiguous_ids)
    ].copy()
    ranker = xgb.XGBRanker()
    ranker.load_model(ranker_dir / "ranker.json")
    ambiguous_predictions = score_rows(
        ranker,
        ambiguous_candidates,
        DATASET_FEATURE_ORDER,
        origin="out_of_sample_ambiguous",
        fold=None,
    )
    predictions = pd.concat(
        [exact_predictions, ambiguous_predictions],
        ignore_index=True,
    )
    predicted_ids = set(predictions["query_id"].astype(str))
    if predicted_ids != set(labels["query_id"].astype(str)):
        raise ValueError("Predictions do not cover all acceptor scenes")
    predictions_path = output_dir / "ranker_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    manifest = V9DatasetManifest(
        schema_version=SCHEMA_VERSION,
        build_id=build_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=42,
        sirene_snapshot_id=snapshot_id,
        input_hashes={
            "exact_dataset_manifest": identity["exact_manifest_sha256"],
            "ranker": identity["ranker_sha256"],
            "v4_manifest": identity["v4_manifest_sha256"],
            "ambiguous_manifest": identity["ambiguous_manifest_sha256"],
            **identity["retrieval_hashes"],
        },
        retrieval_config={
            "name": "selective_admission_frozen_v4",
            "candidate_budget": 100,
            "channels": SELECTIVE_RETRIEVAL_CHANNELS,
            "weights": V7_WEIGHTS,
        },
        retrieval_signature=exact_manifest.retrieval_signature,
        tokenizer_fingerprint=None,
        feature_order=DATASET_FEATURE_ORDER,
        row_counts={
            "queries": len(queries),
            "labels": len(labels),
            "candidates": len(all_candidates),
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
        "ranker_fit_excluded_ambiguous": True,
        "label_counts": labels.groupby(
            ["split", "label_kind"]
        ).size().to_dict(),
        "diagnostics": diagnostics,
        "validation": validation,
        "outputs": {
            path.name: file_sha256(path)
            for path in (
                queries_path,
                labels_path,
                candidates_path,
                predictions_path,
            )
        },
    }
    # JSON does not support tuple keys.
    report["label_counts"] = {
        f"{split}:{kind}": int(count)
        for (split, kind), count in labels.groupby(
            ["split", "label_kind"]
        ).size().items()
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-dataset", type=Path, required=True)
    parser.add_argument("--ranker-dir", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--downstream-dir", type=Path, required=True)
    parser.add_argument("--ambiguous-input-dir", type=Path, required=True)
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
        build(
            exact_dataset_dir=args.exact_dataset,
            ranker_dir=args.ranker_dir,
            v4_dir=args.v4_dir,
            downstream_dir=args.downstream_dir,
            ambiguous_input_dir=args.ambiguous_input_dir,
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
